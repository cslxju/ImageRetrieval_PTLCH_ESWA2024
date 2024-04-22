# PiT
# Copyright 2021-present NAVER Corp.
# Apache License v2.0

import torch
from einops import rearrange
from torch import nn
import math

from functools import partial
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block as transformer_block
from timm.models.registry import register_model
import numpy as np
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F
####









import numpy as np
import torch
from torch import nn
from torch.nn import init



class LCA(nn.Module):

    def __init__(self, d_model=576,n=1,simple=False):

        super(LCA, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model,d_model)
        if(simple):
            self.position_biases=torch.zeros((n,n))
        else:
            self.position_biases=nn.Parameter(torch.ones((n,n)))
        self.d_model = d_model
        self.n=n
        self.sigmoid=nn.Sigmoid()

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):

        bs, n,dim = input.shape

        q = self.fc_q(input) #bs,n,dim
        k = self.fc_k(input).view(1,bs,n,dim) #1,bs,n,dim
        v = self.fc_v(input).view(1,bs,n,dim) #1,bs,n,dim
        
        numerator=torch.sum(torch.exp(k+self.position_biases.view(n,1,-1,1))*v,dim=2) #n,bs,dim
        z = torch.exp(k+self.position_biases.view(n,1,-1,1))
        # print(q.shape)
        # print(k.shape)
        # print(z)
        # print(v.shape)
        # exit()
        x = torch.exp(k+self.position_biases.view(n,1,-1,1))*v
        # print(self.position_biases.shape)
        
        denominator=torch.sum(torch.exp(k+self.position_biases.view(n,1,-1,1)),dim=2) #n,bs,dim

        out=(numerator/denominator) #n,bs,dim
        v = self.fc_v(input).view(n,bs,dim) #1,bs,n,dim
        # if(out == v):
           # print("111")
        # print(out)
        # print(v)
        # exit()
        out=self.sigmoid(q)*(out.permute(1,0,2)) #bs,n,dim

        return out



    
class Transformer(nn.Module):
    def __init__(self, base_dim, depth, heads, mlp_ratio,
                 drop_rate=.0, attn_drop_rate=.0, drop_path_prob=None):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        embed_dim = base_dim * heads

        if drop_path_prob is None:
            drop_path_prob = [0.0 for _ in range(depth)]

        self.blocks = nn.ModuleList([
            transformer_block(
                dim=embed_dim,
                num_heads=heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_prob[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
            for i in range(depth)])

    def forward(self, x, cls_tokens):
        h, w = x.shape[2:4]
        x = rearrange(x, 'b c h w -> b (h w) c')

        token_length = cls_tokens.shape[1]
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)

        cls_tokens = x[:, :token_length]
        x = x[:, token_length:]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x, cls_tokens


class conv_head_pooling(nn.Module):
    def __init__(self, in_feature, out_feature, stride,
                 padding_mode='zeros'):
        super(conv_head_pooling, self).__init__()

        self.conv = nn.Conv2d(in_feature, out_feature, kernel_size=stride + 1,
                              padding=stride // 2, stride=stride,
                              padding_mode=padding_mode, groups=in_feature)
        self.fc = nn.Linear(in_feature, out_feature)

    def forward(self, x, cls_token):

        x = self.conv(x)
        cls_token = self.fc(cls_token)

        return x, cls_token


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size,
                 stride, padding):
        super(conv_embedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size,
                              stride=stride, padding=padding, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class PoolingTransformer(nn.Module):
    def __init__(self, image_size, patch_size, stride, base_dims, depth, heads,
                 mlp_ratio, bit = 36,  num_classes=1000, in_chans=3,
                 attn_drop_rate=.0, drop_rate=.0, drop_path_rate=.0):
        super(PoolingTransformer, self).__init__()

        total_block = sum(depth)
        padding = 0
        block_idx = 0

        width = math.floor(
            (image_size + 2 * padding - patch_size) / stride + 1)
        
        self.base_dims = base_dims
        self.heads = heads
        self.num_classes = num_classes
        ###
        self.LCA = LCA()
        
        self.hash_layer = nn.Sequential(
            # nn.Dropout(),
           
            # nn.ReLU(inplace=True),
            nn.Linear(1000, bit),
        )
        ###
        self.patch_size = patch_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, base_dims[0] * heads[0], width, width),
            requires_grad=True
        )
        self.patch_embed = conv_embedding(in_chans, base_dims[0] * heads[0],
                                          patch_size, stride, padding)

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, base_dims[0] * heads[0]),
            requires_grad=True
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.transformers = nn.ModuleList([])
        self.pools = nn.ModuleList([])

        for stage in range(len(depth)):
            drop_path_prob = [drop_path_rate * i / total_block
                              for i in range(block_idx, block_idx + depth[stage])]
            block_idx += depth[stage]

            self.transformers.append(
                Transformer(base_dims[stage], depth[stage], heads[stage],
                            mlp_ratio,
                            drop_rate, attn_drop_rate, drop_path_prob)
            )
            if stage < len(heads) - 1:
                self.pools.append(
                    conv_head_pooling(base_dims[stage] * heads[stage],
                                      base_dims[stage + 1] * heads[stage + 1],
                                      stride=2
                                      )
                )

        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]

        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)
        else:
            self.head = nn.Identity()

    def forward_features(self, x):
        
        x = self.patch_embed(x)
        
        pos_embed = self.pos_embed
        
        x = self.pos_drop(x + pos_embed)
        # x1 = x.flatten(2)
        # x1 = self.LCA(x1)
        # x1 = x1.reshape(-1,144,27,27)
        # x = x + x1
       
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        
        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            x, cls_tokens = self.pools[stage](x, cls_tokens)
        x, cls_tokens = self.transformers[-1](x, cls_tokens)
 
        cls_tokens = self.norm(cls_tokens)
        
        return cls_tokens

    def forward(self, x):
        
        cls_token = self.forward_features(x) 
        # print(cls_token.shape)
        # exit()
        x1 = self.LCA(cls_token)
     
        cls_token = cls_token  + x1
        
        
        cls_token = self.head(cls_token[:, 0])
        
        
        cls_token = self.hash_layer(cls_token)
        
        return cls_token




@register_model
def pit_b(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('/home/rh/DeepHash-pytorch-master/pretrainedVIT/pit_b_820.pth', map_location='cpu')
        model.load_state_dict(state_dict,strict=False)
    return model

@register_model
def pit_s(pretrained, **kwargs):
    model = PoolingTransformer(
        image_size=224,
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        **kwargs
    )
    if pretrained:
        state_dict = \
        torch.load('/home/rh/DeepHash-pytorch-master/pretrainedVIT/pit_s_809.pth', map_location='cpu')
        model.load_state_dict(state_dict,strict=False)
    return model




