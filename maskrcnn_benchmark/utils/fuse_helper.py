import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import math
from maskrcnn_benchmark.modeling.utils import cat, concat_box_prediction_layers, permute_and_flatten
from timm.models.layers import DropPath

from transformers.activations import ACT2FN


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def _make_conv(input_dim, output_dim, k, stride=1):
    pad = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, (k, k), padding=(pad, pad), stride=(stride, stride)),
        nn.BatchNorm2d(output_dim),
        nn.ReLU(inplace=True)
    )


def _make_mlp(input_dim, output_dim, drop):
    return nn.Sequential(nn.Linear(input_dim, output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(inplace=True),
                         nn.Dropout(drop),
                         nn.Linear(output_dim, output_dim),
                         nn.BatchNorm1d(output_dim),
                         nn.ReLU(inplace=True))


def _make_coord(batch, height, width):
    # relative position encoding
    xv, yv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
    xv_min = (xv.float() * 2 - width) / width
    yv_min = (yv.float() * 2 - height) / height
    xv_max = ((xv + 1).float() * 2 - width) / width
    yv_max = ((yv + 1).float() * 2 - height) / height
    xv_ctr = (xv_min + xv_max) / 2
    yv_ctr = (yv_min + yv_max) / 2
    hmap = torch.ones(height, width) * (1. / height)
    wmap = torch.ones(height, width) * (1. / width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0), \
                                               xv_max.unsqueeze(0), yv_max.unsqueeze(0), \
                                               xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0), \
                                               hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0))
    coord = coord.unsqueeze(0).repeat(batch, 1, 1, 1)
    return coord


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, smooth=1, raw_feature_norm="softmax", eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT

def get_inverse(target,weight,bias):
    w_T=weight.transpose(0,1)
    output = target - bias #view as matmul(target,w_T)
    # import torch.linalg
    output= torch.matmul(output, torch.pinverse(w_T))
    # output = torch.matmul(output, torch.linalg.inv(w_T))
    return output
class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None,first_cross=False,skip_forward=False):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.first_cross=first_cross
        self.skip_forward = skip_forward
        self.CPT_DIM=cfg.MODEL.PROMPT.NUM_TOKENS
        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout
        # ! 假设image feature的维度为 (bsz, token_v, self.v_dim), text feature的维度为(bsz, token_l, self.l_dim)
        if cfg.use_lora_cross:
            import loralib as lora
            LORA_PARAM=16
            self.v_proj = lora.Linear(self.v_dim,
                                    self.embed_dim,r=LORA_PARAM)  # ! 对image进行query映射，将image feature的维度映射为(bsz, token_v, self.embed_dim)，这个embed_dim就是attention空间的维度
            self.l_proj = lora.Linear(self.l_dim,
                                    self.embed_dim,r=LORA_PARAM)  # ! 将image feature的维度映射为(bsz, token_l, self.embed_dim)。上面的这两个映射是为了在同样的维度下做cross attention
            self.values_v_proj = lora.Linear(self.v_dim, self.embed_dim,r=LORA_PARAM)
            self.values_l_proj = lora.Linear(self.l_dim,
                                           self.embed_dim,r=LORA_PARAM)  # ! 这两个则分别将image、text feature映射到value空间，value和attention进行加权

            self.out_v_proj = lora.Linear(self.embed_dim,
                                        self.v_dim,r=LORA_PARAM)  # ! 把做完attention之后的text feature映射为 (bsz, token_v, self.v_dim)
            self.out_l_proj = lora.Linear(self.embed_dim,
                                        self.l_dim,r=LORA_PARAM)  # ! 把做完attention之后的image feature映射回 (bsz, token_l, self.v_dim)
        else:
            self.v_proj = nn.Linear(self.v_dim,
                                    self.embed_dim)  # ! 对image进行query映射，将image feature的维度映射为(bsz, token_v, self.embed_dim)，这个embed_dim就是attention空间的维度
            self.l_proj = nn.Linear(self.l_dim,
                                    self.embed_dim)  # ! 将image feature的维度映射为(bsz, token_l, self.embed_dim)。上面的这两个映射是为了在同样的维度下做cross attention
            self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
            self.values_l_proj = nn.Linear(self.l_dim,
                                           self.embed_dim)  # ! 这两个则分别将image、text feature映射到value空间，value和attention进行加权

            self.out_v_proj = nn.Linear(self.embed_dim,
                                        self.v_dim)  # ! 把做完attention之后的text feature映射为 (bsz, token_v, self.v_dim)
            self.out_l_proj = nn.Linear(self.embed_dim,
                                        self.l_dim)  # ! 把做完attention之后的image feature映射回 (bsz, token_l, self.v_dim)

        # ! 设添加的cross prompt tuning （cpt） 模块的维度为（k(超参), self.embed_dim）, 可以将其广播为（bsz, k , self.embed_dim）
        # ! 需要提取self.v_proj的权重，记为W_vq和B_vq，将CPT反映射到image feature的操作为  （CPT-B_vq）* W_vq^(-1)，因此需要计算W_vq的逆，然后设置一个新的
        # ! nn.Linear(self.embed_dim, self.v_dim), 将其权重设为W_vq的逆，bias设为0，并将其冻结。在CPT经过这个新的linear之前不要忘了减去B_vq
        # ! 反映设到text feature的操作同理。
        # ! 加上反映射的cpt后的image和text feature维度应该分别为 (bsz, token_v + k, self.v_dim), (bsz, token_l + k, self.l_dim)
        self.vpt_only = cfg.vpt_only
        if cfg.vpt_only==-3 and self.first_cross:
            self.cpt_vinput = nn.Parameter(torch.zeros(
                1, self.CPT_DIM,self.embed_dim))
            nn.init.uniform_(self.cpt_vinput.data, -1, 1)
            self.cpt_linput = nn.Parameter(torch.zeros(
                1, self.CPT_DIM, self.embed_dim))
            nn.init.uniform_(self.cpt_linput.data, -1, 1)
            self.cpt_maskpad = nn.Parameter(torch.zeros(
                1, self.CPT_DIM))
        if cfg.vpt_only==-4 and self.first_cross:
            self.cpt_input = nn.Parameter(torch.zeros(
                1, self.CPT_DIM,self.embed_dim))
            if cfg.ECCV.init_way != '0':
                if cfg.ECCV.init_way=='uniform':
                    nn.init.uniform_(self.cpt_input.data, -1, 1)
                if cfg.ECCV.init_way == 'normal':
                    nn.init.normal_(self.cpt_input.data, 0, 1)
                if cfg.ECCV.init_way == 'sparse':
                    nn.init.sparse_(self.cpt_input.data, 0.5, std=0.01)
                if cfg.ECCV.init_way == 'xavier_uniform':
                    nn.init.xavier_uniform_(self.cpt_input.data, gain=nn.init.calculate_gain('relu'))
                if cfg.ECCV.init_way == 'xavier_normal':
                    nn.init.xavier_normal_(self.cpt_input.data)
                if cfg.ECCV.init_way == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(self.cpt_input.data, mode='fan_in', nonlinearity='relu')
                if cfg.ECCV.init_way == 'kaiming_normal':
                    nn.init.kaiming_normal_(self.cpt_input.data, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif cfg.ECCV.init_way=='0':
                pass
            self.cpt_maskpad = nn.Parameter(torch.zeros(
                1, self.CPT_DIM))
        self.stable_softmax_2d = cfg.MODEL.DYHEAD.FUSE_CONFIG.STABLE_SOFTMAX_2D
        self.clamp_min_for_underflow = cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW
        self.clamp_max_for_overflow = cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW

        class Mlp(nn.Module):
            """ Multilayer perceptron."""

            def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
                super().__init__()
                out_features = out_features or in_features
                hidden_features = hidden_features or in_features
                self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
                self.act = act_layer(inplace=False)
                self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
                self.drop = nn.Dropout(drop)

            def forward(self, x):
                x = self.fc1(x)
                x = self.act(x)
                x = self.drop(x)
                x = self.fc2(x)
                x = self.act(x)  # caution: remember uncomment to rollback
                x = self.drop(x)
                return x

        cin = self.embed_dim
        if cfg.pt4_tokens:
            cout=cfg.pt4_tokens
            cout2=cout
            cout3 = cout
            cout4 = cout
        else:
            cout = int(0.65 * cin)
            cout2=cout
            cout3 = cout
            cout4 = cout
        drop = 0
        self.fuse_module_cross_att = cfg.fuse_module_cross_att
        if self.fuse_module_cross_att % 10 >= 1:
            self.adaptera = Mlp(cin, cout, cin, drop=drop)
        if self.fuse_module_cross_att % 100 >= 10:
            self.adapterb = Mlp(cin, cout2, cin, drop=drop)
        if self.fuse_module_cross_att % 1000 >= 100:
            self.adapterc = Mlp(cin, cout3, cin, drop=drop)
        if self.fuse_module_cross_att % 10000 >= 1000:
            self.adapterd = Mlp(cin, cout4, cin, drop=drop)
        # if self.fuse_module_cross_att % 1000 >= 100:
        #     self.adapterc=Mlp(self.v_dim ,int(0.9*self.v_dim), self.v_dim,drop=drop)
        # if self.fuse_module_cross_att % 10000 >= 1000:
        #     self.adapterd =Mlp(self.l_dim, int(0.9*self.l_dim), self.l_dim, drop=drop)
        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_l=None):
        if self.skip_forward:
            return v,l
        bsz, tgt_len, embed_dim = v.size()
        src_len = l.size(1)

        if self.vpt_only == -4 and self.first_cross:
            wv = self.v_proj.weight.data
            bv = self.v_proj.bias.data
            wl = self.l_proj.weight.data
            bl = self.l_proj.bias.data
            v_cpt_ori=get_inverse(self.cpt_input,wv,bv)
            l_cpt_ori = get_inverse(self.cpt_input, wl, bl)
            v=torch.cat((v,v_cpt_ori),dim=1)
            l=torch.cat((l,l_cpt_ori),dim=1)
            tgt_len+=self.CPT_DIM
            src_len+=self.CPT_DIM

        query_states = self.v_proj(v)
        query_states *= self.scale
        key_states = self.l_proj(l)

        if self.vpt_only == -3 and self.first_cross:
            wv = self.v_proj.weight.data
            bv = self.v_proj.bias.data
            wl = self.l_proj.weight.data
            bl = self.l_proj.bias.data
            v_cpt_ori=get_inverse(self.cpt_vinput,wv,bv)
            l_cpt_ori = get_inverse(self.cpt_linput, wl, bl)
            # w_T = wv.transpose(0, 1)
            # output = torch.matmul(v_cpt_ori, w_T)+bv #here ,confirmed get_inverse() x linear() = I!!!!!!!!!!
            query_states=torch.cat((query_states,self.cpt_vinput),dim=1)
            tgt_len+=self.CPT_DIM
            key_states=torch.cat((key_states,self.cpt_linput),dim=1)
            src_len+=self.CPT_DIM
        if self.fuse_module_cross_att % 10 >= 1:
            query_states = self.adaptera(query_states) + query_states
        if self.fuse_module_cross_att % 100 >= 10:
            key_states = self.adapterb(key_states) + key_states
        ################}}}}}

        key_states = self._shape(key_states, -1, bsz)
        # ################
        # if self.fuse_module_cross_att >0:
        #     key_states = self.adapterb(key_states) + key_states
        # ################
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        # src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
        #     )

        # attn_weights_l = nn.functional.softmax(attn_weights.transpose(1, 2), dim=-1)

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights,
                                       min=-50000)  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights,
                                       max=50000)  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = (attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[
            0])
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l,
                                         min=-50000)  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l,
                                         max=50000)  # Do not increase 50000, data type half has quite limited range

        attn_weights_l = attn_weights_l.softmax(dim=-1)  # ! 维度为（bsz*num_heads, token_l, token_v）
        # if self.vpt_only == -4: waste
        #     attn_weights = attn_weights[:, :-self.CPT_DIM, :-self.CPT_DIM]
        #     attn_weights_l = attn_weights_l[:, :-self.CPT_DIM, :-self.CPT_DIM]
        #     tgt_len-=self.CPT_DIM
        #     src_len-=self.CPT_DIM
        if attention_mask_l is not None:
            assert (attention_mask_l.dim() == 2)
            if self.vpt_only <= -3 and self.first_cross:
                attention_mask_l=torch.cat((attention_mask_l,self.cpt_maskpad),dim=1)
                self.cpt_maskpad.requires_grad=False
            attention_mask = attention_mask_l.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_v = nn.functional.softmax(attn_weights, dim=-1)  # ! 维度为（bsz*num_heads, token_v, token_l）

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)
        if self.vpt_only == -3 and self.first_cross:
            attn_probs_v = attn_probs_v[:, :-self.CPT_DIM, :-self.CPT_DIM]
            attn_probs_l = attn_probs_l[:, :-self.CPT_DIM, :-self.CPT_DIM]
            tgt_len-=self.CPT_DIM
            src_len-=self.CPT_DIM
        attn_output_v = torch.bmm(attn_probs_v,
                                  value_l_states)  # ! attention_v 乘以 value_l，得到的维度为 （bsz*num_heads, token_v, self.head_dim)
        attn_output_l = torch.bmm(attn_probs_l,
                                  value_v_states)  # ! attention_l 乘以 value_v，得到的维度为 （bsz*num_heads, token_l, self.head_dim)
        # ! 上述过程为cross attention的过程

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)
        ################
        if self.fuse_module_cross_att % 1000 >= 100:
            attn_output_v = self.out_v_proj(attn_output_v + self.adapterc(attn_output_v))
        else:
            attn_output_v = self.out_v_proj(attn_output_v)
        if self.fuse_module_cross_att % 10000 >= 1000:
            attn_output_l = self.out_l_proj(attn_output_l + self.adapterd(attn_output_l))
        else:
            attn_output_l = self.out_l_proj(attn_output_l)
        # ################
        # if self.fuse_module_cross_att%1000 >=100:
        #     attn_output_v = self.adapterc(v) + attn_output_v
        # if self.fuse_module_cross_att % 10000 >= 1000:
        #     attn_output_l = self.adapterd(l) + attn_output_l
        # ################
        if self.vpt_only == -4 and self.first_cross:
            attn_output_v = attn_output_v[:, :-self.CPT_DIM, :]
            attn_output_l = attn_output_l[:, :-self.CPT_DIM, :]
        return attn_output_v, attn_output_l


# Bi-Direction MHA (text->image, image->text)
class BiAttentionBlock(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, cfg=None):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim,
                                         l_dim=l_dim,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         cfg=cfg)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, v, l, attention_mask_l=None, dummy_tensor=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_l=attention_mask_l)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l


class BiAttentionBlockForCheckpoint(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, cfg=None,first_cross=False,skip_forward=False):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        WUYONGJIAN:this class just pass first_cross/skip_forward to inner KEY block BMHA!
        """
        super(BiAttentionBlockForCheckpoint, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim,
                                         l_dim=l_dim,
                                         embed_dim=embed_dim,
                                         num_heads=num_heads,
                                         dropout=dropout,
                                         cfg=cfg,first_cross=first_cross,skip_forward=skip_forward)

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

        self.cfg = cfg
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL:
            if not self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                self.shrink_lang = FeatureResizer(l_dim * 5, l_dim, 0.1)

    def forward(self, q0, q1, q2, q3, q4, l, attention_mask_l=None, dummy_tensor=None):

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL:
            visu_feat = []
            lang_feat = []
            for ii, feat in enumerate([q0, q1, q2, q3, q4]):
                bs, _, h, w = feat.shape
                q = feat.flatten(2).transpose(1, 2)

                new_v, new_l = self.single_attention_call(q, l, attention_mask_l=attention_mask_l)
                new_v = new_v.transpose(1, 2).contiguous().view(bs, -1, h, w)
                lang_feat.append(new_l)
                visu_feat.append(new_v)
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                pass
            else:
                lang_feat = self.shrink_lang(torch.cat(lang_feat, dim=-1))  # From multiple dimensions
                lang_feat = [lang_feat, None, None, None, None]
        else:
            visu_feat = []
            size_per_level, visual_features_flatten = [], []
            for ii, feat_per_level in enumerate([q0, q1, q2, q3, q4]):
                bs, c, h, w = feat_per_level.shape
                size_per_level.append([h, w])
                feat = permute_and_flatten(feat_per_level, bs, 1, c, h, w)
                visual_features_flatten.append(feat)
            visual_features_flatten = cat(visual_features_flatten, dim=1)
            new_v, new_l = self.single_attention_call(visual_features_flatten, l, attention_mask_l=attention_mask_l)
            # [bs, N, C] -> [bs, C, N]
            new_v = new_v.transpose(1, 2).contiguous()

            start = 0
            for (h, w) in size_per_level:
                new_v_per_level = new_v[:, :, start:start + h * w].view(bs, -1, h, w).contiguous()
                visu_feat.append(new_v_per_level)
                start += h * w

            lang_feat = [new_l, None, None, None, None]

        return visu_feat[0], visu_feat[1], visu_feat[2], visu_feat[3], visu_feat[4], lang_feat[0], lang_feat[1], \
               lang_feat[2], lang_feat[3], lang_feat[4]

    def single_attention_call(self, v, l, attention_mask_l=None, dummy_tensor=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_l=attention_mask_l)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l


# Single Direction MHA
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for both image and text
    """

    def __init__(self, q_dim, k_dim, embed_dim, num_heads, dropout=0.1,
                 clamp_min_for_underflow=False, clamp_max_for_overflow=False):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.q_dim)
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, q, k, v, attention_mask=None, return_attention=False):
        bsz, tgt_len, embed_dim = q.size()

        query_states = self.q_proj(q) * self.scale
        key_states = self._shape(self.k_proj(k), -1, bsz)
        value_states = self._shape(self.v_proj(v), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights,
                                       min=-50000)  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights,
                                       max=50000)  # Do not increase 50000, data type half has quite limited range

        if attention_mask is not None:
            # [bsz, src_len]
            assert (attention_mask.dim() == 2)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if return_attention:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class AttentionMLP(nn.Module):
    def __init__(self, q_dim, hidden_dim, dropout=0.1):
        super(AttentionMLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation_fn = nn.GELU()
        self.fc1 = nn.Linear(q_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, q_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class AttentionT2I(nn.Module):
    def __init__(self, q_dim, k_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-4, mode="i2t", use_layer_scale=False,
                 clamp_min_for_underflow=False, clamp_max_for_overflow=False):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(AttentionT2I, self).__init__()

        # pre_layer norm
        self.layer_norm_q_1 = nn.LayerNorm(q_dim)
        self.layer_norm_k_1 = nn.LayerNorm(k_dim)
        self.attn = MultiHeadAttention(q_dim=q_dim,
                                       k_dim=k_dim,
                                       embed_dim=embed_dim,
                                       num_heads=num_heads,
                                       clamp_min_for_underflow=clamp_min_for_underflow,
                                       clamp_max_for_overflow=clamp_max_for_overflow)
        self.mode = mode

        # add layer scale for training stability
        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.gamma = nn.Parameter(init_values * torch.ones((q_dim)), requires_grad=True)

    def forward(self, q0, q1, q2, q3, q4, k, v, attention_mask, dummy_arg=None):
        qs = []
        for q_index, q in enumerate([q0, q1, q2, q3, q4]):
            bs, _, h, w = q.shape
            # (batch, seq_len, embed_size)
            q = q.flatten(2).transpose(1, 2)
            q = self.layer_norm_q_1(q)
            k, v = self.layer_norm_k_1(k), self.layer_norm_k_1(v)
            delta_q = self.attn(q, k, v, attention_mask=attention_mask)[0]
            if self.use_layer_scale:
                q = q + self.drop_path(self.gamma * delta_q)
            else:
                q = q + delta_q
            q = q.transpose(1, 2).contiguous().view(bs, -1, h, w)
            qs.append(q)

        return qs[0], qs[1], qs[2], qs[3], qs[4]
