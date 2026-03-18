"""
Standalone model modules for VisTex-OVLM inference.
No dependencies on project source code.
Only requires: torch, torchvision, transformers, numpy
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import deform_conv2d, batched_nms

from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertLayer

from transformers.pytorch_utils import apply_chunking_to_forward


# ═══════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════

class BoxList:
    """Minimal bounding box list for postprocessing."""
    def __init__(self, bbox, image_size, mode="xyxy"):
        self.bbox = bbox
        self.size = image_size  # (width, height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, value):
        self.extra_fields[field] = value

    def get_field(self, field):
        return self.extra_fields[field]

    def clip_to_image(self, remove_empty=True):
        w, h = self.size
        self.bbox[:, 0].clamp_(min=0, max=w)
        self.bbox[:, 1].clamp_(min=0, max=h)
        self.bbox[:, 2].clamp_(min=0, max=w)
        self.bbox[:, 3].clamp_(min=0, max=h)
        return self

    def __len__(self):
        return self.bbox.shape[0]

    def __getitem__(self, item):
        new_box = BoxList(self.bbox[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            if isinstance(v, torch.Tensor):
                new_box.add_field(k, v[item])
            else:
                new_box.add_field(k, v)
        return new_box

    def to(self, device):
        self.bbox = self.bbox.to(device)
        for k, v in self.extra_fields.items():
            if isinstance(v, torch.Tensor):
                self.extra_fields[k] = v.to(device)
        return self


def cat_boxlist(boxlists):
    size = boxlists[0].size
    mode = boxlists[0].mode
    bbox = torch.cat([bl.bbox for bl in boxlists], dim=0)
    new_bl = BoxList(bbox, size, mode)
    fields = boxlists[0].extra_fields.keys()
    for field in fields:
        data = torch.cat([bl.get_field(field) for bl in boxlists], dim=0)
        new_bl.add_field(field, data)
    return new_bl


def remove_small_boxes(boxlist, min_size):
    w = boxlist.bbox[:, 2] - boxlist.bbox[:, 0]
    h = boxlist.bbox[:, 3] - boxlist.bbox[:, 1]
    keep = ((w > min_size) & (h > min_size)).nonzero(as_tuple=False).squeeze(1)
    return boxlist[keep]


def boxlist_ml_nms(boxlist, nms_thresh):
    boxes = boxlist.bbox
    scores = boxlist.get_field("scores")
    labels = boxlist.get_field("labels")
    keep = batched_nms(boxes, scores, labels, nms_thresh)
    return boxlist[keep]


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, A, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def to_image_list(tensors, size_divisible=32):
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    if size_divisible > 0:
        max_h = (max_h + size_divisible - 1) // size_divisible * size_divisible
        max_w = (max_w + size_divisible - 1) // size_divisible * size_divisible
    batch = torch.zeros(len(tensors), 3, max_h, max_w,
                        device=tensors[0].device, dtype=tensors[0].dtype)
    image_sizes = []
    for i, t in enumerate(tensors):
        batch[i, :, :t.shape[1], :t.shape[2]] = t
        image_sizes.append((t.shape[1], t.shape[2]))
    return batch, image_sizes


# ═══════════════════════════════════════════════
# Swin Transformer Backbone
# ═══════════════════════════════════════════════

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        r = x.new_empty(shape).bernoulli_(keep)
        if keep > 0.0:
            r.div_(keep)
        return x * r


def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows, ws, H, W):
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class SwinMlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij'))
        coords_flat = torch.flatten(coords, 1)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        rel_index = rel.sum(-1)
        self.register_buffer("relative_position_index", rel_index.long())

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(0.)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)

        relative_position_index_flat = self.relative_position_index.reshape(-1)  # type: ignore[operator]
        rpb = self.relative_position_bias_table[relative_position_index_flat].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        attn = attn + rpb.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=12, shift_size=0, mlp_ratio=4.,
                 drop_path=0.):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwinMlp(dim, int(dim * mlp_ratio))

    def forward(self, x, H, W, attn_mask):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = attn_mask
        else:
            mask = None

        x = window_partition(x, self.window_size)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = self.attn(x, mask=mask)
        x = x.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x.view(B, H * W, C))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        if H % 2 == 1 or W % 2 == 1:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, drop_path=None,
                 downsample=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        dp = drop_path if drop_path is not None else [0.] * depth
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                drop_path=dp[i])
            for i in range(depth)])
        self.downsample = downsample(dim) if downsample is not None else None

    def _make_attn_mask(self, H, W, device):
        Hp = math.ceil(H / self.window_size) * self.window_size
        Wp = math.ceil(W / self.window_size) * self.window_size
        mask = torch.zeros((1, Hp, Wp, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                     slice(-self.window_size, -self.shift_size),
                     slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                     slice(-self.window_size, -self.shift_size),
                     slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                mask[:, h, w, :] = cnt
                cnt += 1
        mw = window_partition(mask, self.window_size).view(-1, self.window_size * self.window_size)
        am = mw.unsqueeze(1) - mw.unsqueeze(2)
        return am.masked_fill(am != 0, -100.0).masked_fill(am == 0, 0.0)

    def forward(self, x, H, W):
        attn_mask = self._make_attn_mask(H, W, x.device)
        for blk in self.blocks:
            x = blk(x, H, W, attn_mask)
        x_out, H_out, W_out = x, H, W
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2
        return x_out, H_out, W_out, x, H, W


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, H, W


class SwinTransformer(nn.Module):
    def __init__(self, embed_dim=192, depths=(2, 2, 18, 2),
                 num_heads=(6, 12, 24, 48), window_size=12,
                 drop_path_rate=0.2, out_indices=(1, 2, 3)):
        super().__init__()
        self.out_indices = out_indices
        self.patch_embed = PatchEmbed(embed_dim)
        self.pos_drop = nn.Dropout(0.)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            dim = embed_dim * (2 ** i)
            layer = BasicLayer(
                dim=dim, depth=depths[i], num_heads=num_heads[i],
                window_size=window_size,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=PatchMerging if i < len(depths) - 1 else None)
            self.layers.append(layer)

        for i in out_indices:
            self.add_module(f'norm{i}', nn.LayerNorm(embed_dim * (2 ** i)))

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        outs = []
        for i, layer in enumerate(self.layers):
            x_out, Ho, Wo, x, H, W = layer(x, H, W)
            if i in self.out_indices:
                norm = getattr(self, f'norm{i}')
                out = norm(x_out)
                B, _, C = out.shape
                outs.append(out.view(B, Ho, Wo, C).permute(0, 3, 1, 2).contiguous())
        return tuple(outs)


# ═══════════════════════════════════════════════
# FPN
# ═══════════════════════════════════════════════

class LastLevelP6P7(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.p6 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.p7 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, p5):
        p6 = self.p6(p5)
        p7 = self.p7(F.relu(p6))
        return p6, p7


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.fpn_inner2 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.fpn_inner3 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.fpn_inner4 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.fpn_layer2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_layer3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.fpn_layer4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.top_blocks = LastLevelP6P7(out_channels)

    def forward(self, features):
        c3, c4, c5 = features
        p5 = self.fpn_inner4(c5)
        p4 = self.fpn_inner3(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.fpn_inner2(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p3, p4, p5 = self.fpn_layer2(p3), self.fpn_layer3(p4), self.fpn_layer4(p5)
        p6, p7 = self.top_blocks(p5)
        return [p3, p4, p5, p6, p7]


class Backbone(nn.Module):
    """Combined Swin + FPN. Keys: backbone.body.* and backbone.fpn.*"""
    def __init__(self):
        super().__init__()
        self.body = SwinTransformer()
        self.fpn = FPN([384, 768, 1536], 256)

    def forward(self, x):
        return self.fpn(self.body(x))


# ═══════════════════════════════════════════════
# Language Backbone (BERT)
# ═══════════════════════════════════════════════

class LanguageBackboneBody(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = BertConfig(
            vocab_size=30522, hidden_size=768, num_hidden_layers=12,
            num_attention_heads=12, intermediate_size=3072, hidden_act='gelu',
            hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
            max_position_embeddings=512, type_vocab_size=2, layer_norm_eps=1e-12,
            attn_implementation="eager")
        self.model = BertModel(cfg, add_pooling_layer=False)

    def forward(self, inp):
        input_ids = inp["input_ids"]
        attention_mask = inp["attention_mask"]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # last encoder layer
        embedded = hidden * attention_mask.unsqueeze(-1).float()
        return {"embedded": embedded, "hidden": hidden,
                "masks": attention_mask, "mlm_labels": None}


class LanguageBackbone(nn.Module):
    """Keys: language_backbone.body.model.*"""
    def __init__(self):
        super().__init__()
        self.body = LanguageBackboneBody()

    def forward(self, inp):
        return self.body(inp)


# ═══════════════════════════════════════════════
# VLDyHead Components
# ═══════════════════════════════════════════════

class h_sigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(x + 3) / 6


class DYReLU(nn.Module):
    def __init__(self, channels=256, reduction=4):
        super().__init__()
        self.oup = channels
        squeeze = channels // reduction
        self.fc = nn.Sequential(
            nn.Linear(channels, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, channels * 4),
            h_sigmoid())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.oup * 4, 1, 1)
        a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
        a1 = (a1 - 0.5) * 2.0 + 1.0
        a2 = (a2 - 0.5) * 2.0
        b1 = b1 - 0.5
        b2 = b2 - 0.5
        return torch.max(x * a1 + b1, x * a2 + b2)


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([init_value]))

    def forward(self, x):
        return x * self.scale


class Conv3x3Norm(nn.Module):
    def __init__(self, in_ch, out_ch, stride, deformable=True, gn_groups=16):
        super().__init__()
        self.deformable = deformable
        self.stride = stride
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn = nn.GroupNorm(gn_groups, out_ch)

    def forward(self, inp, offset=None, mask=None):
        if self.deformable and offset is not None:
            out_h = (inp.shape[2] + 2 * 1 - 3) // self.stride + 1
            out_w = (inp.shape[3] + 2 * 1 - 3) // self.stride + 1
            # Crop offset/mask if larger than output (matches original CUDA kernel behavior)
            if offset.shape[2] > out_h or offset.shape[3] > out_w:
                offset = offset[:, :, :out_h, :out_w].contiguous()
                if mask is not None:
                    mask = mask[:, :, :out_h, :out_w].contiguous()
            x = deform_conv2d(inp, offset, self.conv.weight, self.conv.bias,
                              stride=(self.stride, self.stride), padding=(1, 1),
                              mask=mask)
        else:
            x = self.conv(inp)
        return self.bn(x)


class DyConv(nn.Module):
    def __init__(self, in_ch=256, out_ch=256, deformable=True):
        super().__init__()
        self.DyConv = nn.ModuleList([
            Conv3x3Norm(in_ch, out_ch, 1, deformable=deformable),
            Conv3x3Norm(in_ch, out_ch, 1, deformable=deformable),
            Conv3x3Norm(in_ch, out_ch, 2, deformable=deformable),
        ])
        self.AttnConv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, 1, kernel_size=1),
            nn.ReLU(inplace=True))
        self.h_sigmoid = h_sigmoid()
        self.relu = DYReLU(out_ch)
        self.offset = nn.Conv2d(in_ch, 27, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        visual_feats = inputs["visual"]
        lang = inputs["lang"]
        next_x = []
        for level, feature in enumerate(visual_feats):
            offset_mask = self.offset(feature)
            off = offset_mask[:, :18, :, :]
            msk = offset_mask[:, 18:, :, :].sigmoid()

            temp_fea = [self.DyConv[1](feature, offset=off, mask=msk)]
            if level > 0:
                temp_fea.append(self.DyConv[2](visual_feats[level - 1],
                                               offset=off, mask=msk))
            if level < len(visual_feats) - 1:
                temp_fea.append(F.interpolate(
                    self.DyConv[0](visual_feats[level + 1], offset=off, mask=msk),
                    size=feature.shape[2:], mode='bilinear', align_corners=True))

            attn_fea = []
            res_fea = []
            for fea in temp_fea:
                res_fea.append(fea)
                attn_fea.append(self.AttnConv(fea))
            res_fea = torch.stack(res_fea)
            spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))
            mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)
            next_x.append(mean_fea)

        next_x = [self.relu(item) for item in next_x]
        return {"visual": next_x, "lang": lang}


# ── Bi-Directional Attention ──

class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim=256, l_dim=768, embed_dim=2048, num_heads=8, skip=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = 0.1
        self.skip_forward = skip

        self.v_proj = nn.Linear(v_dim, embed_dim)
        self.l_proj = nn.Linear(l_dim, embed_dim)
        self.values_v_proj = nn.Linear(v_dim, embed_dim)
        self.values_l_proj = nn.Linear(l_dim, embed_dim)
        self.out_v_proj = nn.Linear(embed_dim, v_dim)
        self.out_l_proj = nn.Linear(embed_dim, l_dim)

        self.stable_softmax_2d = False
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, v, l, attention_mask_l=None):
        if self.skip_forward:
            return v, l

        bsz, tgt_len, _ = v.size()
        src_len = l.size(1)

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000)
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000)

        # Language attention (text attending to image)
        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(attn_weights_l, min=-50000)
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(attn_weights_l, max=50000)
        attn_weights_l = attn_weights_l.softmax(dim=-1)

        # Apply attention mask for visual attention
        if attention_mask_l is not None:
            am = attention_mask_l.unsqueeze(1).unsqueeze(1)
            am = am.expand(bsz, 1, tgt_len, src_len)
            am = am.masked_fill(am == 0, -9e15)
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + am
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_v = F.softmax(attn_weights, dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2).reshape(bsz, src_len, self.embed_dim)

        return self.out_v_proj(attn_output_v), self.out_l_proj(attn_output_l)


class BiAttentionBlockForCheckpoint(nn.Module):
    def __init__(self, v_dim=256, l_dim=768, embed_dim=2048, num_heads=8,
                 init_values=0.125, skip=False):
        super().__init__()
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim, l_dim, embed_dim, num_heads, skip=skip)
        self.drop_path = nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones(v_dim))
        self.gamma_l = nn.Parameter(init_values * torch.ones(l_dim))
        self.separate_bidirectional = False  # concatenate all levels

    def forward(self, q0, q1, q2, q3, q4, l, attention_mask_l=None, dummy_tensor=None):
        # Concatenate all visual features
        size_per_level = []
        visual_flat = []
        for feat in [q0, q1, q2, q3, q4]:
            bs, c, h, w = feat.shape
            size_per_level.append((h, w))
            visual_flat.append(permute_and_flatten(feat, bs, 1, c, h, w))
        visual_flat = torch.cat(visual_flat, dim=1)

        v = self.layer_norm_v(visual_flat)
        l_normed = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l_normed, attention_mask_l=attention_mask_l)
        # Residual added to NORMED features (matching original single_attention_call)
        new_v = v + self.drop_path(self.gamma_v * delta_v)
        new_l = l_normed + self.drop_path(self.gamma_l * delta_l)

        new_v = new_v.transpose(1, 2).contiguous()
        visu_feat = []
        start = 0
        for h, w in size_per_level:
            visu_feat.append(new_v[:, :, start:start + h * w].view(bs, -1, h, w).contiguous())
            start += h * w

        return (visu_feat[0], visu_feat[1], visu_feat[2], visu_feat[3], visu_feat[4],
                new_l, None, None, None, None)


class VLFuse(nn.Module):
    def __init__(self, skip=False):
        super().__init__()
        self.b_attn = BiAttentionBlockForCheckpoint(skip=skip)

    def forward(self, x):
        vis = x["visual"]
        lang = x["lang"]
        q0, q1, q2, q3, q4, l0, *_ = self.b_attn(
            vis[0], vis[1], vis[2], vis[3], vis[4],
            lang['hidden'], lang['masks'], None)
        lang = dict(lang)
        lang['hidden'] = l0
        return {"visual": [q0, q1, q2, q3, q4], "lang": lang}


class TowerBertEncoderLayer(nn.Module):
    """BERT self-attention layer for the dyhead tower language path."""
    def __init__(self):
        super().__init__()
        cfg = BertConfig(
            vocab_size=30522, hidden_size=768, num_hidden_layers=12,
            num_attention_heads=12, intermediate_size=3072, hidden_act='gelu',
            hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
            max_position_embeddings=512, type_vocab_size=2, layer_norm_eps=1e-12,
            attn_implementation="eager")
        layer = BertLayer(cfg)
        self.attention = layer.attention
        self.intermediate = layer.intermediate
        self.output = layer.output
        self.chunk_size_feed_forward = cfg.chunk_size_feed_forward
        self.seq_len_dim = 1

    def get_extended_attention_mask(self, mask, input_shape, device=None):
        extended = mask[:, None, None, :]
        extended = (1.0 - extended) * torch.finfo(torch.float32).min
        return extended

    def forward(self, inputs):
        lang = inputs["lang"]
        hidden = lang["hidden"]
        mask = lang["masks"]

        extended_mask = self.get_extended_attention_mask(mask, hidden.size()[:-1])
        attn_out = self.attention(hidden, extended_mask)[0]
        layer_output = apply_chunking_to_forward(
            self._ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attn_out)

        lang = dict(lang)
        lang["hidden"] = layer_output
        return {"visual": inputs["visual"], "lang": lang}

    def _ff_chunk(self, attn_output):
        intermediate = self.intermediate(attn_output)
        return self.output(intermediate, attn_output)


class VLDyHead(nn.Module):
    """The detection head with VL fusion tower."""
    def __init__(self, num_convs=8, skip_stages=None):
        super().__init__()
        if skip_stages is None:
            skip_stages = set()

        # Build tower: 3 modules per stage (VLFuse + BertEncoderLayer + DyConv)
        tower = []
        for i in range(num_convs):
            tower.append(VLFuse(skip=(i in skip_stages)))
            tower.append(TowerBertEncoderLayer())
            tower.append(DyConv(256, 256, deformable=True))
        self.dyhead_tower = nn.Sequential(*tower)

        num_anchors = 1
        num_classes = 80
        self.cls_logits = nn.Conv2d(256, num_anchors * num_classes, 1)
        self.bbox_pred = nn.Conv2d(256, num_anchors * 4, 1)
        self.centerness = nn.Conv2d(256, num_anchors * 1, 1)

        self.dot_product_projection_image = nn.Identity()
        self.dot_product_projection_text = nn.Linear(768, num_anchors * 256, bias=True)
        self.log_scale = nn.Parameter(torch.tensor([0.0]))
        self.bias_lang = nn.Parameter(torch.zeros(768))
        self.bias0 = nn.Parameter(torch.tensor([-math.log((1 - 0.01) / 0.01)]))
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

    def forward(self, features, language_dict_features, embedding):
        feat_inputs = {"visual": features, "lang": language_dict_features}
        dyhead_out = self.dyhead_tower(feat_inputs)

        # Use fused hidden for embedding (USE_FUSED_FEATURES_DOT_PRODUCT=True)
        embedding = dyhead_out["lang"]["hidden"]

        # Dot product scoring
        embedding = F.normalize(embedding, p=2, dim=-1)
        dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)
        dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang) + self.bias0

        logits, bbox_reg, centerness, dot_product_logits = [], [], [], []
        for l, feature in enumerate(features):
            vis = dyhead_out["visual"][l]
            logits.append(self.cls_logits(vis))
            bbox_reg.append(self.scales[l](self.bbox_pred(vis)))
            centerness.append(self.centerness(vis))

            x = vis
            B, C, H, W = x.shape
            proj_queries = permute_and_flatten(x, B, -1, C, H, W)
            A = proj_queries.shape[1]
            bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)
            dot_logit = (torch.matmul(proj_queries, dot_product_proj_tokens.transpose(-1, -2))
                         / self.log_scale.exp()) + bias
            dot_logit = torch.clamp(dot_logit, min=-50000, max=50000)
            dot_product_logits.append(dot_logit)

        return logits, bbox_reg, centerness, dot_product_logits


# ═══════════════════════════════════════════════
# Postprocessing
# ═══════════════════════════════════════════════

class BoxCoder:
    def decode(self, preds, anchors):
        anchors = anchors.to(preds.dtype)
        TO_REMOVE = 1
        widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        wx, wy, ww, wh = 10., 10., 5., 5.
        dx = preds[:, 0::4] / wx
        dy = preds[:, 1::4] / wy
        dw = torch.clamp(preds[:, 2::4] / ww, max=math.log(1000. / 16))
        dh = torch.clamp(preds[:, 3::4] / wh, max=math.log(1000. / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(preds)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)
        return pred_boxes


def convert_grounding_to_od_logits(logits, num_classes, positive_map):
    scores = torch.zeros(logits.shape[0], logits.shape[1], num_classes,
                         device=logits.device)
    for label_j in positive_map:
        indices = torch.LongTensor(positive_map[label_j])
        scores[:, :, label_j - 1] = logits[:, :, indices].mean(-1)
    return scores


class CellAnchors(nn.Module):
    def __init__(self, n):
        super().__init__()
        for i in range(n):
            self.register_buffer(str(i), torch.zeros(1, 4))

    def __len__(self):
        return len(self._buffers)

    def __getitem__(self, idx):
        cell = self._buffers.get(str(idx))
        if cell is None:
            cell = torch.zeros(1, 4)
        return cell


class AnchorGenerator(nn.Module):
    def __init__(self, strides=(8, 16, 32, 64, 128)):
        super().__init__()
        self.strides = strides
        self.cell_anchors = CellAnchors(len(strides))

    def forward(self, image_sizes, feature_maps):
        grid_sizes = [f.shape[-2:] for f in feature_maps]
        all_anchors = []
        for img_h, img_w in image_sizes:
            anchors_in_image = []
            for i, (gh, gw) in enumerate(grid_sizes):
                stride = self.strides[i]
                cell = self.cell_anchors[i]
                device = cell.device
                sx = torch.arange(0, gw * stride, stride, dtype=torch.float32, device=device)
                sy = torch.arange(0, gh * stride, stride, dtype=torch.float32, device=device)
                shift_y, shift_x = torch.meshgrid(sy, sx, indexing='ij')
                shifts = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1),
                                       shift_x.reshape(-1), shift_y.reshape(-1)], dim=1)
                anchors = (shifts.unsqueeze(1) + cell.unsqueeze(0)).reshape(-1, 4)
                bl = BoxList(anchors, (img_w, img_h), mode="xyxy")
                anchors_in_image.append(bl)
            all_anchors.append(anchors_in_image)
        return all_anchors


class ATSSPostProcessor(nn.Module):
    def __init__(self, pre_nms_thresh=0.05, pre_nms_top_n=1000,
                 nms_thresh=0.6, detections_per_img=100, num_classes=81):
        super().__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.num_classes = num_classes
        self.box_coder = BoxCoder()

    def forward_for_single_feature_map(self, box_regression, centerness, anchors,
                                        box_cls, dot_product_logits, positive_map):
        N, _, H, W = box_regression.shape
        A = box_regression.size(1) // 4
        C = box_cls.size(1) // A

        # Dot product path
        dp = dot_product_logits.sigmoid()
        scores = convert_grounding_to_od_logits(dp, C, positive_map)
        box_cls = scores

        box_regression = permute_and_flatten(box_regression, N, A, 4, H, W).reshape(N, -1, 4)
        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.reshape(N, -1).sum(1).clamp(max=self.pre_nms_top_n)

        centerness = permute_and_flatten(centerness, N, A, 1, H, W).reshape(N, -1).sigmoid()
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for per_cls, per_reg, per_top_n, per_cand, per_anch in zip(
                box_cls, box_regression, pre_nms_top_n, candidate_inds, anchors):
            per_cls = per_cls[per_cand]
            per_cls, top_k = per_cls.topk(per_top_n, sorted=False)
            per_nz = per_cand.nonzero()[top_k, :]
            per_loc = per_nz[:, 0]
            per_class = per_nz[:, 1] + 1

            det = self.box_coder.decode(
                per_reg[per_loc].view(-1, 4),
                per_anch.bbox[per_loc].view(-1, 4))
            bl = BoxList(det, per_anch.size, mode="xyxy")
            bl.add_field("labels", per_class)
            bl.add_field("scores", torch.sqrt(per_cls))
            bl = bl.clip_to_image(remove_empty=False)
            bl = remove_small_boxes(bl, 0)
            results.append(bl)
        return results

    def forward(self, box_regression, centerness, anchors, box_cls,
                dot_product_logits, positive_map):
        # Transpose anchors: from [level][image] to [image][level] then back to [level][images_tuple]
        anchors_t = list(zip(*anchors))
        sampled = []
        for idx, (b, c, a) in enumerate(zip(box_regression, centerness, anchors_t)):
            o = box_cls[idx]
            d = dot_product_logits[idx]
            sampled.append(self.forward_for_single_feature_map(b, c, a, o, d, positive_map))
        boxlists = [cat_boxlist(list(bl)) for bl in zip(*sampled)]

        # Select over all levels
        results = []
        for bl in boxlists:
            result = boxlist_ml_nms(bl, self.nms_thresh)
            if len(result) > self.detections_per_img > 0:
                scores = result.get_field("scores")
                thresh, _ = torch.kthvalue(scores.cpu().float(),
                                            len(result) - self.detections_per_img + 1)
                keep = scores >= thresh.item()
                result = result[keep.nonzero(as_tuple=False).squeeze(1)]
            results.append(result)
        return results


# ═══════════════════════════════════════════════
# conv_map_module (fcto768)
# ═══════════════════════════════════════════════

class ConvMapModule(nn.Module):
    """Maps 5-level FPN features to a single 768-dim vector."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.fc = nn.Linear(5 * 256 * 7 * 7, 768)

    def forward(self, features):
        x1 = self.conv4(self.conv3(self.conv2(self.conv1(features[0]))))
        x2 = self.conv4(self.conv3(self.conv2(features[1])))
        x3 = self.conv4(self.conv3(features[2]))
        x4 = self.conv4(features[3])
        x5 = features[4]
        x = torch.cat((x1, x2, x3, x4, x5), dim=0)
        return self.fc(x.flatten())