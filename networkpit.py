# PiT
# Copyright 2021-present NAVER Corp.
# Apache License v2.0
import numpy as np
import torch
from einops import rearrange
from torch import nn
import math
from torch.nn.parameter import Parameter
from functools import partial
from timm.models.layers import trunc_normal_
from torch.nn import init
from timm.models.registry import register_model




class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

       
class transformer_block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn = AFT_FULL(d_model=144, n=730)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
       
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

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
        # print(token_length)1
        x = torch.cat((cls_tokens, x), dim=1)
        # print(x.shape)torch.Size([64, 730, 144])
        for blk in self.blocks:
            # print("11")
            x = blk(x)
        # print(self.blocks)
        # print(x.shape)
        # print(token_length)
        # exit()
        cls_tokens = x[:, :token_length]
        # print('00')
        # print(cls_tokens.shape)#torch.Size([64, 1, 144])
        #torch.Size([64, 1, 144])
        #torch.Size([64, 729, 144])
        #torch.Size([64, 1, 288])
        #torch.Size([64, 196, 288])
        #torch.Size([64, 1, 576])
        #torch.Size([64, 49, 576])
        # exit()
        x = x[:, token_length:]
        # print(x.shape)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        # print("22")
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
                 mlp_ratio, bit=36, num_classes=1000, in_chans=3,
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
            # print(len(depth))
            # print('11')
        
        self.norm = nn.LayerNorm(base_dims[-1] * heads[-1], eps=1e-6)
        self.embed_dim = base_dims[-1] * heads[-1]
        
        self.hash_layer = nn.Sequential(
            # nn.Dropout(),
           
            # nn.ReLU(inplace=True),
            nn.Linear(1000, bit),
        )
        # Classifier head
        if num_classes > 0:
            self.head = nn.Linear(base_dims[-1] * heads[-1], num_classes)
        else:
            self.head = nn.Identity()
        
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # self.apply(self._init_weights)

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
        # print(x.shape)torch.Size([64, 144, 27, 27])
        
        pos_embed = self.pos_embed
        # print(pos_embed.shape)torch.Size([1, 144, 27, 27])
        Z = x + pos_embed
        x = self.pos_drop(x + pos_embed)
        # print(Z.shape)#torch.Size([64, 144, 27, 27])
        # print(x.shape)
        # exit()
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # print(cls_tokens.shape)torch.Size([64, 1, 144])
        # print(total_block)
        # exit()
        for stage in range(len(self.pools)):
            x, cls_tokens = self.transformers[stage](x, cls_tokens)
            # print('11')
            # print(x.shape)
            # print(cls_tokens.shape)
            x, cls_tokens = self.pools[stage](x, cls_tokens)
            # print(x.shape)
            # print(cls_tokens.shape)
            # exit()
            # print(self.heads[stage])
            # print(stage)
            # print(len(self.pools))
            #torch.Size([64, 288, 14, 14])
            # print(cls_tokens.shape)#torch.Size([64, 1, 288])
          
            
            # print(len(self.pools))
            
        # print(x.shape)
        # 
        
        # print('11')
        x, cls_tokens = self.transformers[-1](x, cls_tokens)
        # print(x.shape)
        # print(cls_tokens.shape)
        # print(self.heads[-1])
        # exit()
        # print(x.shape)torch.Size([64, 576, 7, 7])
        # print(cls_tokens.shape)torch.Size([64, 1, 576])
        
        
        cls_tokens = self.norm(cls_tokens)
        # print(cls_tokens.shape)#torch.Size([64, 1, 576])
        # exit()
        return cls_tokens

    def forward(self, x):
        # print(x.shape)torch.Size([64, 3, 224, 224])
        # exit()
        cls_token = self.forward_features(x)
        # print(cls_token.shape)#batch size (64,1,576)
        
        #print(self.embed_dim)#batch size 576
        #print(cls_token[:, 0].shape)torch.Size([64, 576])

        cls_token = self.head(cls_token[:, 0])
      
        cls_token = self.hash_layer(cls_token)
        
        return cls_token




@register_model
def pit_b(pretrained=True, **kwargs):
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
def pit_s(pretrained=True,  **kwargs):
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

