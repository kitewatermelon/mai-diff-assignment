import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

try:
    from flash_attn import flash_attn_qkvpacked_func
except ImportError:
    pass

from rope import VisionRotaryEmbedding

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., flash=True,
                 rope_size=0, rope_reg_size=0, num_registers=0, reg_theta=10000, qk_norm=False, depth=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.flash = flash
        self.num_registers = num_registers
        self.rope = VisionRotaryEmbedding(head_dim//2, rope_size) if rope_size > 0 else None
        self.rope_reg = VisionRotaryEmbedding(head_dim//2, rope_reg_size, theta=reg_theta) if rope_reg_size > 0 else None

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(self, x):
        B, N, C = x.shape
        reg_idx = N - self.num_registers

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)

        if self.qk_norm:
            qk_dtype = q.dtype
            q = self.q_norm(q).to(qk_dtype)
            k = self.k_norm(k).to(qk_dtype)

        if self.rope is not None:
            q = torch.cat((q[:, :1], self.rope(q[:, 1: reg_idx]), q[:, reg_idx:]), dim=1)
            k = torch.cat((k[:, :1], self.rope(k[:, 1: reg_idx]), k[:, reg_idx:]), dim=1)
        if self.rope_reg is not None:
            q = torch.cat((q[:, :1], q[:, 1: reg_idx], self.rope_reg(q[:, reg_idx:])), dim=1)
            k = torch.cat((k[:, :1], k[:, 1: reg_idx], self.rope_reg(k[:, reg_idx:])), dim=1)
        
        if self.flash:
            qkv = torch.stack([q, k, v], dim=2)
            x = flash_attn_qkvpacked_func(qkv).reshape(B, N, C)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiffAttention5(nn.Module):
    """Differential Attention for vit5 models.

    Splits Q and K into two groups to compute two separate softmax attention maps,
    then takes their difference scaled by a learnable lambda. Supports RoPE,
    register tokens, QK normalization, and flash attention.

    Reference: 'Differential Transformer' - https://arxiv.org/abs/2410.05258
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., flash=True,
                 rope_size=0, rope_reg_size=0, num_registers=0, reg_theta=10000, qk_norm=False, depth=0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads // 2  # half head dim for Q, K split
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.flash = flash
        self.num_registers = num_registers
        self.rope = VisionRotaryEmbedding(self.head_dim // 2, rope_size) if rope_size > 0 else None
        self.rope_reg = VisionRotaryEmbedding(self.head_dim // 2, rope_reg_size, theta=reg_theta) if rope_reg_size > 0 else None

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

        # Differential attention lambda parameters (paper's original formulation)
        self.lambda_q1 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32))
        self.lambda_k1 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32))
        self.lambda_q2 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32))
        self.lambda_k2 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32))

        self.sub_norm = RMSNorm(2 * self.head_dim, eps=1e-5)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)

        nn.init.normal_(self.lambda_q1, mean=0, std=0.1)
        nn.init.normal_(self.lambda_k1, mean=0, std=0.1)
        nn.init.normal_(self.lambda_q2, mean=0, std=0.1)
        nn.init.normal_(self.lambda_k2, mean=0, std=0.1)

    def _compute_lambda(self) -> torch.Tensor:
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        return lambda_1 - lambda_2 + self.lambda_init

    def forward(self, x):
        B, N, C = x.shape
        reg_idx = N - self.num_registers

        q, k, v = self.qkv(x).chunk(3, dim=2)
        # Q, K: 2*num_heads groups with head_dim each
        q = q.reshape(B, N, 2 * self.num_heads, self.head_dim)
        k = k.reshape(B, N, 2 * self.num_heads, self.head_dim)
        # V: num_heads groups with 2*head_dim each
        v = v.reshape(B, N, self.num_heads, 2 * self.head_dim)

        if self.qk_norm:
            qk_dtype = q.dtype
            q = self.q_norm(q).to(qk_dtype)
            k = self.k_norm(k).to(qk_dtype)

        # Apply RoPE to Q, K (shape: B, N, 2*num_heads, head_dim)
        if self.rope is not None:
            q = torch.cat((q[:, :1], self.rope(q[:, 1:reg_idx]), q[:, reg_idx:]), dim=1)
            k = torch.cat((k[:, :1], self.rope(k[:, 1:reg_idx]), k[:, reg_idx:]), dim=1)
        if self.rope_reg is not None:
            q = torch.cat((q[:, :1], q[:, 1:reg_idx], self.rope_reg(q[:, reg_idx:])), dim=1)
            k = torch.cat((k[:, :1], k[:, 1:reg_idx], self.rope_reg(k[:, reg_idx:])), dim=1)

        lambda_full = self._compute_lambda().type_as(q)

        if self.flash:
            # SDPA path — PyTorch selects Flash Attention v2 or memory-efficient
            # backend automatically.  Expects (B, heads, N, dim).
            q, k = q.transpose(1, 2), k.transpose(1, 2)  # (B, 2H, N, d)
            v = v.transpose(1, 2)                          # (B, H, N, 2d)
            # Split Q, K into two groups along the head dimension
            q1, q2 = q[:, :self.num_heads], q[:, self.num_heads:]  # (B, H, N, d)
            k1, k2 = k[:, :self.num_heads], k[:, self.num_heads:]
            # SDPA: softmax(Q @ K^T / sqrt(d)) @ V — drop_p only during training
            dp = self.attn_drop.p if self.training else 0.0
            out1 = F.scaled_dot_product_attention(q1, k1, v, dropout_p=dp, scale=self.scale)
            out2 = F.scaled_dot_product_attention(q2, k2, v, dropout_p=dp, scale=self.scale)
            x = out1 - lambda_full * out2  # (B, H, N, 2d)
            x = self.sub_norm(x)
            x = x * (1 - self.lambda_init)
            x = x.transpose(1, 2).reshape(B, N, C)
            self._last_attn = None  # no attention maps in fused mode
        else:
            # Standard attention path
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            # q, k: (B, 2*num_heads, N, head_dim), v: (B, num_heads, N, 2*head_dim)

            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # (B, 2*num_heads, N, N)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # Split into two attention maps and compute difference
            attn = attn.view(B, self.num_heads, 2, N, N)
            attn1, attn2 = attn[:, :, 0], attn[:, :, 1]
            diff_attn = attn1 - lambda_full * attn2  # (B, num_heads, N, N)

            # Store attention maps for visualization
            self._last_attn = {
                'attn1': attn1.detach(),
                'attn2': attn2.detach(),
                'diff': diff_attn.detach(),
            }

            x = diff_attn @ v  # (B, num_heads, N, 2*head_dim)
            x = self.sub_norm(x)
            x = x * (1 - self.lambda_init)
            x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False,):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp, init_values=1e-4,
                 flash=True, rope_size=0, rope_reg_size=0, reg_theta=10000, num_registers=0, qk_norm=False, layer_scale=True, depth=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, flash=flash,
            rope_size=rope_size, rope_reg_size=rope_reg_size, num_registers=num_registers, qk_norm=qk_norm, reg_theta=reg_theta, depth=depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        if self.layer_scale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x        
    
class vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, ape=True,
                 block_layers=Block, Patch_layer=PatchEmbed, act_layer=nn.GELU, Attention_block=Attention, Mlp_block=Mlp,
                 init_scale=1e-4, flash=True, rope=False, num_registers=0, qk_norm=False, reg_theta=10000, layer_scale=True, **kwargs):
        super().__init__()       
        self.dropout_rate = drop_rate  
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_registers = num_registers

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, num_registers, embed_dim)) if num_registers > 0 else None

        rope_reg_size = int(num_registers ** 0.5)
        assert rope_reg_size ** 2 == num_registers, "num_registers must be a square number"

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) if ape else None

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale,
                flash=flash, rope_size=img_size // patch_size if rope else 0,
                rope_reg_size=rope_reg_size, num_registers=num_registers,
                qk_norm=qk_norm, reg_theta=reg_theta, layer_scale=layer_scale, depth=i)
            for i in range(depth)])
           
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        
        trunc_normal_(self.cls_token, std=.02)
        if ape:
            trunc_normal_(self.pos_embed, std=.02)
        if num_registers > 0:
            trunc_normal_(self.reg_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'reg_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_flash(self, flash):
        """Toggle flash attention on all attention modules."""
        for blk in self.blocks:
            if hasattr(blk.attn, 'flash'):
                blk.attn.flash = flash

    def _prepare_tokens(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        registers = self.reg_token.expand(B, -1, -1) if self.reg_token is not None else None

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = torch.cat((cls_tokens, x), dim=1)
        if registers is not None:
            x = torch.cat((x, registers), dim=1)
        return x

    def forward_features(self, x):
        x = self._prepare_tokens(x)

        for i , blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    @torch.no_grad()
    def get_last_selfattention(self, x):
        """Run a forward pass and return the last block's attention maps.

        Temporarily disables flash attention on the last block so that
        DiffAttention5 stores its attention matrices.  Returns the stored
        ``_last_attn`` dict (keys: 'attn1', 'attn2', 'diff') or, for a
        standard Attention block, the full attention tensor.
        """
        last_blk = self.blocks[-1]
        # temporarily disable flash on the last block
        orig_flash = getattr(last_blk.attn, 'flash', None)
        if orig_flash is not None:
            last_blk.attn.flash = False

        x = self._prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)

        # restore
        if orig_flash is not None:
            last_blk.attn.flash = orig_flash

        return getattr(last_blk.attn, '_last_attn', None)

    def forward(self, x):

        x = self.forward_features(x)

        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)

        return x

@register_model
def deit_small_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, flash=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model
def deit_base_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, flash=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model
def deit_large_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, flash=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model
def vit5_tiny(img_size=224, patch_size=8, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=False, num_registers=4,
        norm_layer=partial(RMSNorm, eps=1e-6), rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model

@register_model
def vit5_small(img_size=224, patch_size=16, flash=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=flash,
        norm_layer=partial(RMSNorm, eps=1e-6), rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model

@register_model
def vit5_base(img_size=224, patch_size=16, flash=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=flash,
        norm_layer=partial(RMSNorm, eps=1e-6), rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model

@register_model
def vit5_large(img_size=224, patch_size=16, flash=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=flash,
        norm_layer=partial(RMSNorm, eps=1e-6), rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model

@register_model
def vit5_xlarge(img_size=224, patch_size=16, flash=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=patch_size, embed_dim=1152, depth=28, num_heads=16, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=flash,
        norm_layer=partial(RMSNorm, eps=1e-6), rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model


# --- Differential Attention vit5 variants ---

@register_model
def vit5_diff_tiny(img_size=224, patch_size=8, flash=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=flash,
        norm_layer=partial(RMSNorm, eps=1e-6), Attention_block=DiffAttention5,
        rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model

@register_model
def vit5_diff_small(img_size=224, patch_size=16, flash=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=flash,
        norm_layer=partial(RMSNorm, eps=1e-6), Attention_block=DiffAttention5,
        rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model

@register_model
def vit5_diff_base(img_size=224, patch_size=16, flash=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=flash,
        norm_layer=partial(RMSNorm, eps=1e-6), Attention_block=DiffAttention5,
        rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model

@register_model
def vit5_diff_large(img_size=224, patch_size=16, flash=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=flash,
        norm_layer=partial(RMSNorm, eps=1e-6), Attention_block=DiffAttention5,
        rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model

@register_model
def vit5_diff_xlarge(img_size=224, patch_size=16, flash=False, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=patch_size, embed_dim=1152, depth=28, num_heads=16, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=flash,
        norm_layer=partial(RMSNorm, eps=1e-6), Attention_block=DiffAttention5,
        rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model


# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

"""Differential Attention

Paper: 'Differential Transformer' - https://arxiv.org/abs/2410.05258

Reference impl: https://github.com/microsoft/unilm/tree/master/Diff-Transformer

Adapted from timm (Ross Wightman) to be self-contained in this file.
"""
from typing import Optional, Type

import torch.nn.functional as F


class RmsNorm(nn.Module):
    """Root mean square layer normalization (no mean subtraction, no bias)."""
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt) * self.weight)


def maybe_add_mask(attn, attn_mask=None):
    return attn if attn_mask is None else attn + attn_mask


def resolve_self_attn_mask(seq_len, attn, attn_mask=None, is_causal=False):
    """Return an additive attention bias, or None if there is no masking."""
    if is_causal:
        assert attn_mask is None, 'attn_mask and is_causal are mutually exclusive'
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=attn.device).tril()
        bias = torch.zeros(seq_len, seq_len, dtype=attn.dtype, device=attn.device)
        return bias.masked_fill(~causal, float('-inf'))
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        return torch.zeros_like(attn_mask, dtype=attn.dtype).masked_fill(~attn_mask, float('-inf'))
    return attn_mask


class DiffAttention(nn.Module):
    """Differential Attention module.

    Computes attention as the difference between two softmax attention maps, which helps
    cancel out noise and promotes sparse attention patterns. The module splits Q and K
    into two groups, computes separate attention maps, and subtracts one from the other
    scaled by a learnable lambda parameter.

    The attention output is computed as:
        Attn = softmax(Q1 @ K1^T) - lambda * softmax(Q2 @ K2^T)
        Output = Attn @ V

    Supports both fused (scaled_dot_product_attention) and manual implementations.
    """
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
            depth: int = 0,
            dual_lambda: bool = False,
            device=None,
            dtype=None,
    ) -> None:
        """Initialize the DiffAttention module.

        Args:
            dim: Input dimension of the token embeddings.
            num_heads: Number of attention heads.
            qkv_bias: Whether to use bias in the query, key, value projections.
            qk_norm: Whether to apply normalization to query and key vectors.
            scale_norm: Whether to apply normalization before the output projection.
            proj_bias: Whether to use bias in the output projection.
            attn_drop: Dropout rate applied to the attention weights.
            proj_drop: Dropout rate applied after the output projection.
            norm_layer: Normalization layer constructor (defaults to RmsNorm).
            depth: Block depth index, used to compute depth-dependent lambda_init.
            dual_lambda: If True, use simplified dual scalar lambda parameterization
                (2 params). If False, use the paper's original formulation with
                lambda_q/k vectors (4 * head_dim params).
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if norm_layer is None:
            norm_layer = RmsNorm
        self.num_heads = num_heads
        self.head_dim = dim // num_heads // 2
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, **dd)
        self.q_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, **dd) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_p = attn_drop
        self.norm = norm_layer(dim, **dd) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dual_lambda = dual_lambda
        if dual_lambda:
            self.lambda_a = nn.Parameter(torch.empty((), dtype=torch.float32, device=device))
            self.lambda_b = nn.Parameter(torch.empty((), dtype=torch.float32, device=device))
            self.lambda_q1 = self.lambda_k1 = self.lambda_q2 = self.lambda_k2 = None
        else:
            self.lambda_a = self.lambda_b = None
            self.lambda_q1 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32, device=device))
            self.lambda_k1 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32, device=device))
            self.lambda_q2 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32, device=device))
            self.lambda_k2 = nn.Parameter(torch.empty(self.head_dim, dtype=torch.float32, device=device))

        self.sub_norm = RmsNorm(2 * self.head_dim, eps=1e-5, **dd)

        self.lambda_init = 0.8
        self.set_lambda_init(depth)
        self.reset_parameters()

    def set_lambda_init(self, depth: int):
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)

    def reset_parameters(self):
        if self.dual_lambda:
            nn.init.zeros_(self.lambda_a)
            nn.init.zeros_(self.lambda_b)
        else:
            nn.init.normal_(self.lambda_q1, mean=0, std=0.1)
            nn.init.normal_(self.lambda_k1, mean=0, std=0.1)
            nn.init.normal_(self.lambda_q2, mean=0, std=0.1)
            nn.init.normal_(self.lambda_k2, mean=0, std=0.1)

    def _compute_lambda(self) -> torch.Tensor:
        if self.lambda_a is not None:
            lambda_1 = torch.exp(self.lambda_a)
            lambda_2 = torch.exp(self.lambda_b)
        else:
            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
            lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        return lambda_1 - lambda_2 + self.lambda_init

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
    ) -> torch.Tensor:
        B, N, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=2)
        q = q.reshape(B, N, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, 2 * self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        lambda_full = self._compute_lambda().type_as(q)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal=is_causal)
        attn = maybe_add_mask(attn, attn_bias)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn = attn.view(B, self.num_heads, 2, N, N)
        attn1, attn2 = attn[:, :, 0], attn[:, :, 1]
        attn = attn1 - lambda_full * attn2
        x = attn @ v

        x = self.sub_norm(x)
        x = x * (1 - self.lambda_init)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        # the two softmax maps are returned separately alongside their difference,
        # each of shape (B, num_heads, N, N)
        return x, {'attn1': attn1, 'attn2': attn2, 'diff': attn}
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mode="sa", depth=0):
        super().__init__()
        self.mode = mode
        self.norm1 = norm_layer(dim)
        if mode == "sa":
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        elif mode == "da":
            self.attn = DiffAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, depth=depth)
        else:
            raise ValueError(f"unknown attention mode: {mode}")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


ATTN_MODES = ('sa', 'da')


def resolve_modes(mode, depth):
    """Expand a per-model `mode` spec into a list of `depth` attention modes.

    Accepted forms:
        'da'                      -> every block uses differential attention
        ['sa', 'da', ...]         -> one entry per block (len must equal depth)
        {0: 'sa', 1: 'da', ...}   -> one entry per block index; every index in
                                     range(depth) must be present unless a
                                     'default' key supplies the rest
        {'default': 'sa', -1: 'da'} -> default for all blocks, overridden by
                                       integer keys (negative indices allowed)
        lambda i: 'da' if i % 2 else 'sa'  -> callable taking the block index
    """
    if isinstance(mode, str):
        modes = [mode] * depth
    elif isinstance(mode, dict):
        default = mode.get('default')
        modes = [default] * depth
        seen = set()
        for k, v in mode.items():
            if k == 'default':
                continue
            if not isinstance(k, int) or isinstance(k, bool):
                raise TypeError(f"mode dict keys must be ints or 'default', got {k!r}")
            if not -depth <= k < depth:
                raise ValueError(f'mode dict index {k} is out of range for depth {depth}')
            modes[k] = v  # negative indices count from the last block
            seen.add(k % depth)
        if default is None:
            missing = sorted(set(range(depth)) - seen)
            if missing:
                raise ValueError(
                    f"mode dict is missing blocks {missing}; list every index or add a 'default' key")
    elif callable(mode):
        modes = [mode(i) for i in range(depth)]
    else:
        modes = list(mode)
        if len(modes) != depth:
            raise ValueError(f'mode sequence has {len(modes)} entries but depth is {depth}')

    for i, m in enumerate(modes):
        if m not in ATTN_MODES:
            raise ValueError(f'block {i}: unknown attention mode {m!r}, expected one of {ATTN_MODES}')
    return modes


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, mode="sa", **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.modes = resolve_modes(mode, depth)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                mode=self.modes[i], depth=i)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
