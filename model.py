
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import math

"""" #PVT-v2 backbone """
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

@register_model
def pvt_v2_b0(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b1(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b2(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b3(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b4(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b5(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_v2_b2_li(pretrained=False, **kwargs):
    model = PyramidVisionTransformerV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=True, **kwargs)
    model.default_cfg = _cfg()

    return model


"""## cbam, paspp"""

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class PASPP(nn.Module):
    def __init__(self, inplanes, outplanes, output_stride=4, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        if output_stride == 4:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 4, 6, 10]
        elif output_stride == 2:
            dilations = [1, 12, 24, 36]
        elif output_stride == 16:
            dilations = [1, 2, 3, 4]
        elif output_stride == 1:
            dilations = [1, 16, 32, 48]
        else:
            raise NotImplementedError
        self._norm_layer = BatchNorm
        self.silu = nn.SiLU(inplace=True)
        self.conv1 = self._make_layer(inplanes, inplanes // 4)
        self.conv2 = self._make_layer(inplanes, inplanes // 4)
        self.conv3 = self._make_layer(inplanes, inplanes // 4)
        self.conv4 = self._make_layer(inplanes, inplanes // 4)
        self.atrous_conv1 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[0], padding=dilations[0])
        self.atrous_conv2 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[1], padding=dilations[1])
        self.atrous_conv3 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[2], padding=dilations[2])
        self.atrous_conv4 = nn.Conv2d(inplanes // 4, inplanes // 4, kernel_size=3, dilation=dilations[3], padding=dilations[3])
        self.conv5 = self._make_layer(inplanes // 2, inplanes // 2)
        self.conv6 = self._make_layer(inplanes // 2, inplanes // 2)
        self.convout = self._make_layer(inplanes, inplanes)

    def _make_layer(self, inplanes, outplanes):
        layer = []
        layer.append(nn.Conv2d(inplanes, outplanes, kernel_size = 1))
        layer.append(self._norm_layer(outplanes))
        layer.append(self.silu)
        return nn.Sequential(*layer)

    def forward(self, X):
        x1 = self.conv1(X)
        x2 = self.conv2(X)
        x3 = self.conv3(X)
        x4 = self.conv4(X)

        x12 = torch.add(x1, x2)
        x34 = torch.add(x3, x4)

        x1 = torch.add(self.atrous_conv1(x1),x12)
        x2 = torch.add(self.atrous_conv2(x2),x12)
        x3 = torch.add(self.atrous_conv3(x3),x34)
        x4 = torch.add(self.atrous_conv4(x4),x34)

        x12 = torch.cat([x1, x2], dim = 1)
        x34 = torch.cat([x3, x4], dim = 1)

        x12 = self.conv5(x12)
        x34 = self.conv5(x34)
        x = torch.cat([x12, x34], dim=1)
        x = self.convout(x)
        return x

"""## meta former block"""

class Mlp2(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=nn.GELU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class DepthwiseBlock(nn.Module):
  def __init__(self,dim):
    super().__init__()

    self.norm = nn.BatchNorm2d(dim)
    self.act = nn.GELU()
    self.dwconv1 = nn.Conv2d(dim,dim,kernel_size = 3, stride = 1, padding = 1, groups = dim, bias = False)
    self.dwconv2 = nn.Conv2d(dim,dim,kernel_size = 5, stride = 1, padding = 2, groups = dim, bias = False)
    self.dwconv3 = nn.Conv2d(dim,dim,kernel_size = 7, stride = 1, padding = 3, groups = dim, bias = False)

  def forward(self,x):

    x = self.norm(x)
    x1 = self.dwconv1(x)
    x2 = self.dwconv2(x)
    x3 = self.dwconv3(x)
    x = x1 + x2 + x3
    x = self.act(x)

    return x

class ConvMLPMixer(nn.Module):
  def __init__(self,dim,mlp=Mlp2,norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.1,):
    super().__init__()

    self.dw = DepthwiseBlock(dim)
    self.norm = norm_layer(dim)
    self.mlp = mlp(dim=dim,drop=drop)
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

  def forward(self,x):

    x = x + self.dw(x)
    x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
    x = x + self.drop_path(self.mlp(self.norm(x)))
    x = x.permute(0, 3, 1, 2) #  (B, H, W, C) -> (B, C, H, W)

    return x

"""## Convmixer"""

class activation_block(nn.Module):
  def __init__(self, dim):
    super(activation_block, self).__init__()
    self.gelu = nn.GELU()
    self.batchnorm = nn.BatchNorm2d(dim)

  def forward(self, x):
    x = self.gelu(x)
    x = self.batchnorm(x)
    return x

class DepthwiseConv2d(nn.Module):
  def __init__(self, dim):
    super(DepthwiseConv2d, self).__init__()
    self.depthwise = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

  def forward(self, x):
    out = self.depthwise(x)
    return out

class ConvMixer(nn.Module):
  def __init__(self, dim, kernels_size=1):
    super(ConvMixer, self).__init__()
    self.depthwise = DepthwiseConv2d(dim)
    self.pointwise = nn.Conv2d(dim,dim, kernel_size=1)
    self.activation = activation_block(dim)

  def forward(self, x):
    #Depthwise convolution
    x0 = x
    x = self.depthwise(x)
    x = self.activation(x)
    x = x + x0 #Residual

    #Pointwise convolution
    x = self.pointwise(x)
    x = self.activation(x)
    return x

"""## MLP mixer"""
class SpatialMLP(nn.Module):

    def __init__(self, N, mlp_ratio=4, patch_size=(4, 4), **kwargs):             # N = H * W
        super().__init__()
        self.p1 = patch_size[0]
        self.p2 = patch_size[1]
        N1 = N // (patch_size[0] * patch_size[1])
        self.fc1 = nn.Linear(N1, N1 * mlp_ratio, bias=False)
        self.fc2 = nn.Linear(N1 * mlp_ratio, N1, bias=False)
        self.gelu = nn.GELU()
        #self.rearrange1 = Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) (h w)', p1 = p1, p2 = p2)
        #self.rearrange2 = Rearrange('b (p1 p2 c) (h w) -> b c (h p1) (w p2)', p1 = p1, p2 = p2)


    def forward(self, x):
        B, H, W, C = x.shape
        x = Rearrange('b (h p1) (w p2) c -> b (p1 p2 c) (h w)', p1= self.p1, p2 = self.p2)(x)    # (B, p1*p2*C, H*W/(p1*p2))
        x = self.fc1(x)                     # (B, p1*p2*C, H * W//(p1*p2) * 4)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.gelu(x)                    # (B, p1*p2*C, H * W/(p1*p2))
        x = Rearrange('b (p1 p2 c) (h w) -> b (h p1) (w p2) c', p1= self.p1, p2 = self.p2, h=H//self.p1, w=W//self.p2)(x)
        return x

class MLPMixer(nn.Module):
    """
    Implementation of one MetaFormer block.
    Input: [B, C, H, W]
    """
    def __init__(self, dim, N = 8*8,
                 token_mixer=nn.Identity, mlp=Mlp2,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.1,
                 num_heads = 8, sr_ratio=1
                 ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = SpatialMLP(N=N)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        x = x + self.drop_path1(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2) #  (B, H, W, C) -> (B, C, H, W)

        return x

"""## Attention Map"""

class MapReduce(nn.Module):
    """
    Reduce feature maps into a single edge map
    """
    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)
        # self.bn = nn.BatchNorm2d(1)
        # self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv(x)


class Attention_img(nn.Module):
  """Apply attention mechanism to output images"""

  def __init__(self):
    super().__init__()
    self.bn = nn.BatchNorm2d(1)
    self.relu  = nn.ReLU(inplace=True)
    self.sigmoid = nn.Sigmoid()
    self.conv_out = nn.Conv2d(2,1,kernel_size=1, bias=True)
  def forward(self, x1, x2):
    # x = torch.cat((x1,x2),dim=1)
    # x = self.conv_out(x)
    x = self.bn(x1 + x2)
    x = self.relu(x)
    x = self.sigmoid(x)
    return x1*x+x2
  
"""# Conv Block"""

class DropBlock2D(nn.Module):
    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
    def forward(self, x):
        # shape: (bsize, channels, height, width)
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"
        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self.drop_prob / (self.block_size ** 2)
            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:], device= x.device) < gamma).float()
            # compute block mask
            block_mask = self._compute_block_mask(mask)
            # apply block mask
            out = x * block_mask[:, None, :, :]
            # scale output
            out = out * block_mask.numel() / block_mask.sum()
            return out
    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)
        return block_mask


class conv_block(nn.Sequential):
    def __init__(self, ch_in, ch_out, kernel_size = 3,
                 padding = 1, drop_block=False, block_size = 1, drop_prob = 0):
        super().__init__()
        self.add_module("conv1",nn.Conv2d(ch_in, ch_out, kernel_size, padding = padding,bias=False))
        self.add_module("bn1", nn.BatchNorm2d(ch_out))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv2",nn.Conv2d(ch_out, ch_out, kernel_size, padding = padding,bias=False))
        if drop_block:
            self.add_module("drop_block", DropBlock2D(block_size = block_size, drop_prob = drop_prob))
        self.add_module("bn2", nn.BatchNorm2d(ch_out))
        self.add_module("relu2", nn.ReLU(inplace=True))


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=2,stride=1,padding="same",bias=False,),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

"""# Model"""

class PVTFormerNet(nn.Module):

  def __init__(self,backbone,attention = True,convmix=False,convMLP=True,MLPmix=False):
    super().__init__()
    self.backbone = torch.nn.Sequential(*list(backbone.children()))[:-1]
    for i in [1, 4, 7, 10]:
        self.backbone[i] = torch.nn.Sequential(*list(self.backbone[i].children()))

    self.convmix = convmix
    self.convMLP = convMLP
    self.MLPmix = MLPmix
    self.attention = attention
    self.paspp = PASPP(512,512)

    if self.convmix:
      self.connect1 = ConvMixer(dim = 512)
      self.connect2 = ConvMixer(dim = 320)
      self.connect3 = ConvMixer(dim = 128)
      self.connect4 = ConvMixer(dim = 64)

    if self.MLPmix:
      self.connect1 = MLPMixer(dim = 512,N=64)
      self.connect2 = MLPMixer(dim = 320,N=16*16)
      self.connect3 = MLPMixer(dim = 128,N=32*32)
      self.connect4 = MLPMixer(dim = 64,N=64*64)

    if self.convMLP:
      self.connect1 = ConvMLPMixer(dim=512)
      self.connect2 = ConvMLPMixer(dim=320)
      self.connect3 = ConvMLPMixer(dim=128)
      self.connect4 = ConvMLPMixer(dim=64)

    self.CAM1 = ChannelGate(512)
    self.CAM2 = ChannelGate(320)
    self.CAM3 = ChannelGate(128)
    self.CAM4 = ChannelGate(64)

    self.SA = SpatialGate()

    self.up1 = up_conv(512,320)
    self.up2 = up_conv(320,128)
    self.up3 = up_conv(128,64)

    self.upconv1 = conv_block(1024,512)
    self.upconv2 = conv_block(640,320)
    self.upconv3 = conv_block(256,128)
    self.upconv4 = conv_block(128,64)

    self.mapreduce1 = MapReduce(512)
    self.mapreduce2 = MapReduce(320)
    self.mapreduce3 = MapReduce(128)
    self.mapreduce4 = MapReduce(64)

    self.attn_img = Attention_img()

    self.conv = nn.Conv2d(4,1,kernel_size=1)
    self.sigmoid = nn.Sigmoid()

  def get_pyramid(self,x):
      pyramid = []
      B = x.shape[0]
      for i, module in enumerate(self.backbone):
          if i in [0, 3, 6, 9]:
              x, H, W = module(x)
          elif i in [1, 4, 7,10]:
              for sub_module in module:
                  x = sub_module(x, H, W)
          else:
              x = module(x)
              x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
              pyramid.append(x)
      return pyramid

  def attentionmap(self,x1,x2,x3,x4):
    if self.attention:
      a3 = self.attn_img(x3, x4)
      a2 = self.attn_img(x2, a3)
      a1 = self.attn_img(x1, a2)
      x = self.sigmoid(a1)
      output = [x, self.sigmoid(a2), self.sigmoid(a3), self.sigmoid(x4),  self.sigmoid(x3), self.sigmoid(x2), self.sigmoid(x1)]
    else:
      x = torch.cat([x1,x2,x3,x4],dim=1)
      x = self.conv(x)
      output = self.sigmoid(x)
    return output


  def forward(self,x):

    H, W = x.size()[2:]

    pyramid=self.get_pyramid(x)


    pyramid3=self.paspp(pyramid[3])           #(512,8,8)
    pyramid[3]=self.connect1(pyramid[3])   #(512,8,8)
    pyramid[3]=self.CAM1(pyramid[3])          #(512,8,8)
    pyramid[3]=torch.cat((pyramid[3],pyramid3),dim=1) #(1024,8,8)
    pyramid[3]=self.upconv1(pyramid[3])       #(512,8,8)
    pyramid[3]=self.SA(pyramid[3])            #(512,8,8)

    x1 = self.mapreduce1(pyramid[3])          #(1,8,8)
    x1 = F.interpolate(x1, (H, W), mode="bilinear", align_corners=False) #(1,256,256)

    pyramid[2]=self.connect2(pyramid[2])   #(320,16,16)
    pyramid[2]=self.CAM2(pyramid[2])
    pyramid[3]=self.up1(pyramid[3])           #(320,16,16)
    pyramid[2]=torch.cat((pyramid[3],pyramid[2]),dim=1) #(640,16,16)
    pyramid[2]=self.upconv2(pyramid[2])       #(320,16,16)
    pyramid[2]=self.SA(pyramid[2])

    x2 = self.mapreduce2(pyramid[2])          #(1,16,16)
    x2 = F.interpolate(x2, (H, W), mode="bilinear", align_corners=False) #(1,256,256)

    pyramid[1]=self.connect3(pyramid[1])   #(128,32,32)
    pyramid[1]=self.CAM3(pyramid[1])
    pyramid[2]=self.up2(pyramid[2])           #(128,32,32)
    pyramid[1]=torch.cat((pyramid[2],pyramid[1]),dim=1) #(256,32,32)
    pyramid[1]=self.upconv3(pyramid[1])       #(128,32,32)
    pyramid[1]=self.SA(pyramid[1])

    x3 = self.mapreduce3(pyramid[1])          #(1,32,32)
    x3 = F.interpolate(x3, (H, W), mode="bilinear", align_corners=False) #(1,256,256)

    pyramid[0]=self.connect4(pyramid[0])   #(64,64,64)
    pyramid[0]=self.CAM4(pyramid[0])
    pyramid[1]=self.up3(pyramid[1])           #(64,64,64)
    pyramid[0]=torch.cat((pyramid[1],pyramid[0]),dim=1) #(128,64,64)
    pyramid[0]=self.upconv4(pyramid[0])       #(64,64,64)
    pyramid[0]=self.SA(pyramid[0])

    x4 = self.mapreduce4(pyramid[0])          #(1,64,64)
    x4 = F.interpolate(x4, (H, W), mode="bilinear", align_corners=False) #(1,256,256)

    output = self.attentionmap(x1,x2,x3,x4)

    return output

backbone = pvt_v2_b3()
backbone.load_state_dict(torch.load(PRETRAINED_PATH))

model = PVTFormerNet(backbone=backbone,attention=True,convmix=False,convMLP=True,MLPmix=False)

x = torch.rand(1,3,256,256)
y = model(x)[0]
y.shape
