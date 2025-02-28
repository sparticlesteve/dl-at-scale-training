import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from networks.helpers import DropPath, trunc_normal_
from torch.utils.checkpoint import checkpoint  # <-- Added for checkpointing

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        # We keep a Dropout layer so we can pass its probability into the native function.
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x shape: (B, N, C)
        B, N, C = x.shape

        # Compute query, key, and value projections
        q = self.q(x)  # shape (B, N, C)
        k = self.k(x)
        v = self.v(x)

        # Reshape and permute to (B, num_heads, N, head_dim)
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scale the query
        q = q * self.scale

        # Use PyTorch native flash attention implementation
        attn_out = F.scaled_dot_product_attention(q, k, v,
                                                  dropout_p=self.attn_drop.p,
                                                  is_causal=False)
        # attn_out has shape (B, num_heads, N, head_dim)
        # Merge heads: transpose and reshape back to (B, N, C)
        x = attn_out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=[224, 224], patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size
        num_patches = self.h * self.w
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=[224, 224], patch_size=16, in_chans=3, out_chans=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.out_size = out_chans * patch_size * patch_size
        self.head = nn.Linear(embed_dim, self.out_size, bias=False)

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

        # New: Optionally enable gradient checkpointing to reduce memory usage.
        self.use_checkpoint = kwargs.get('use_checkpoint', False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_head(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.patch_embed.h, self.patch_embed.w, C)
        x = self.head(x)
        x = x.reshape(B, self.patch_embed.h, self.patch_embed.w,
                      self.patch_embed.patch_size, self.patch_embed.patch_size, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, -1, self.img_size[0], self.img_size[1])
        return x

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                # Checkpointing reduces memory usage by recomputing activations during backward pass.
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)
        x = self.forward_head(x)
        return x

def ViT(params, **kwargs):
    model = VisionTransformer(
        img_size=tuple(params.img_size),
        in_chans=params.n_in_channels,
        out_chans=params.n_out_channels,
        patch_size=params.patch_size,
        embed_dim=params.embed_dim,
        depth=params.depth,
        num_heads=params.num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_path_rate=float(params.dropout),
        drop_rate=float(params.dropout),
        attn_drop_rate=float(params.dropout),
        use_checkpoint=getattr(params, 'use_checkpoint', False),  # <-- Pass checkpoint flag
        **kwargs
    )
    return model

