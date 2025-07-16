from copy import deepcopy
import numpy as np
import torch
from torch import nn

# from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertConfig, RobertaConfig, RobertaModel, BertModel


class BertEncoder(nn.Module):
    def __init__(self, cfg):
        super(BertEncoder, self).__init__()
        self.cfg = cfg
        self.bert_name = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        print("LANGUAGE BACKBONE USE GRADIENT CHECKPOINTING: ", self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT)

        if self.bert_name == "bert-base-uncased":
            config = BertConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT
            self.model = BertModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        elif self.bert_name == "roberta-base":
            config = RobertaConfig.from_pretrained(self.bert_name)
            config.gradient_checkpointing = self.cfg.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT
            self.model = RobertaModel.from_pretrained(self.bert_name, add_pooling_layer=False, config=config)
            self.language_dim = 768
        else:
            raise NotImplementedError

        self.num_layers = cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS
        ################wuyongjian edit block
        class Mlp(nn.Module):
            """ Multilayer perceptron."""

            def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
                super().__init__()
                out_features = out_features or in_features
                hidden_features = hidden_features or in_features
                self.fc1 = nn.Linear(in_features, hidden_features,bias=False)
                self.act = act_layer(inplace=True)
                self.fc2 = nn.Linear(hidden_features, out_features,bias=False)
                self.drop = nn.Dropout(drop)

            def forward(self, x):
                x = self.fc1(x)
                x = self.act(x)
                x = self.drop(x)
                x = self.fc2(x)
                x = self.act(x)#caution: remember uncomment to rollback
                x = self.drop(x)
                return x
        self.adap_mlp = 0
        dim=768*3
        act_layer=nn.ReLU
        drop=0.3
        if self.cfg.use_maple:
            cin = 768
            cout= 192
            self.maple_lang_input = nn.Parameter(torch.zeros(
                1, 1,cin,))
            nn.init.uniform_(self.maple_lang_input.data, -1, 1)
            self.maple_coupling_proj = nn.Sequential(
                Mlp(cin, cout, 192),
            )
        if self.cfg.use_upt:
            # import torch
            # import torch.nn as nn

            class UPT(nn.Module):
                def __init__(self, input_size, hidden_size):
                    super(UPT, self).__init__()

                    self.sa = nn.MultiheadAttention(input_size, num_heads=1)
                    self.ffn = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, input_size)
                    )
                    self.ln = nn.LayerNorm(input_size)

                def forward(self, U):
                    U0 = self.sa(U, U, U)[0] + self.ln(U)
                    U_hat = self.ffn(self.ln(U0)) + self.ln(U0)
                    return U_hat
            self.UPT=UPT(768,100)
        if cfg.lang_adap_mlp:
            self.adap_mlp = cfg.lang_adap_mlp
            adap_dim = int(0.5 * dim)
            if self.adap_mlp > 0:
                self.adapter = Mlp(in_features=dim, hidden_features=adap_dim, act_layer=act_layer, drop=drop)
            if self.adap_mlp > 1:
                self.adapterb = Mlp(in_features=dim, hidden_features=adap_dim, act_layer=act_layer, drop=drop)
            version_of_adapterc = 1
            if self.adap_mlp > 2:
                if version_of_adapterc == 1:
                    self.adapterc = Mlp(in_features=dim, hidden_features=int(0.8 * dim), act_layer=act_layer, drop=drop)
                # else:
                #     self.adapterc = Mlp2(in_features=dim, hidden_features=int(0.8 * dim),
                #                          hidden_features2=int(0.64 * dim), act_layer=act_layer, drop=drop)
    def forward(self, x):
        input = x["input_ids"]
        mask = x["attention_mask"]
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            # with padding, always 256
            outputs = self.model(
                input_ids=input,
                attention_mask=mask,
                output_hidden_states=True,
            )
            # outputs has 13 layers, 1 input layer and 12 hidden layers
            encoded_layers = outputs.hidden_states[1:]
            features = None
            features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)

            # language embedding has shape [len(phrase), seq_len, language_dim]
            features = features / self.num_layers

            embedded = features * mask.unsqueeze(-1).float()
            aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())

        else:
            # without padding, only consider positive_tokens
            max_len = (input != 0).sum(1).max().item()
            outputs = self.model(
                input_ids=input[:, :max_len],
                attention_mask=mask[:, :max_len],
                output_hidden_states=True,
            )
            # outputs has 13 layers, 1 input layer and 12 hidden layers
            encoded_layers = outputs.hidden_states[1:]

            features = None
            features = torch.stack(encoded_layers[-self.num_layers:], 1).mean(1)
            # language embedding has shape [len(phrase), seq_len, language_dim]
            features = features / self.num_layers

            embedded = features * mask[:, :max_len].unsqueeze(-1).float()
            aggregate = embedded.sum(1) / (mask.sum(-1).unsqueeze(-1).float())
        if self.cfg.use_upt:
            seed=torch.randn((1,1,768)).cuda()
            UPT_OUTPUT_of_seed=self.UPT(seed)
            embedded[:, :1, :768] += UPT_OUTPUT_of_seed
        if self.cfg.use_maple:
            embedded[:, :1, :768] += self.maple_lang_input
        if self.cfg.lang_adap_mlp:
            # print('success lang adap!!!!!!!!!!!!!!!!!!!!!')
            flat_embedded=embedded[:,:3,:].flatten()
            flat_embedded+=self.adapter(flat_embedded)
            embedded3=flat_embedded.reshape([1, 3, 768])
            embedded[:, :3, :]=embedded3.clone()
            hidden = encoded_layers[-1]
            flat_hidden = hidden[:, :3, :].flatten()
            flat_hidden += self.adapterb(flat_hidden)
            hidden3 = flat_hidden.reshape([1, 3, 768])
            hidden[:, :3, :] = hidden3.clone()
            ret = {
                "aggregate": aggregate,
                "embedded": embedded,
                "masks": mask,
                "hidden": hidden
            }
        else:
            ret = {
                "aggregate": aggregate,
                "embedded": embedded,
                "masks": mask,
                "hidden": encoded_layers[-1]
            }
        if self.cfg.use_maple:
            return ret,self.maple_coupling_proj(self.maple_lang_input)
        return ret
