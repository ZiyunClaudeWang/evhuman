import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, causal):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.register_buffer("mask", torch.tril(torch.ones(1000, 1000)))
        self.causal = causal

    def forward(self, x):
        N = x.size(1)

        if self.causal:
            attn_mask = self.mask[:N, :N] == 0  # shape (N, N), bool mask
            out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        else:
            out, _ = self.attn(x, x, x)

        return out

class CausalTransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, mlp_dim, causal):
        super().__init__()
        self.attn = CausalSelfAttention(embed_dim, n_heads, causal=causal)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class CausalTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads, mlp_dim, depth, use_cls=False, causal=True):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(embed_dim, n_heads, mlp_dim, causal=causal) for _ in range(depth)
        ])
        self.output_proj = nn.Linear(embed_dim, input_dim)  # or any output shape you need

        self.use_cls = use_cls
        if self.use_cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x, nothing, return_hidden=False):
        # x: (B, N, input_dim)

        x = self.input_proj(x)

        if self.use_cls:
            cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        for block in self.blocks:
            x = block(x)
        output = self.output_proj(x)  # (B, N, embed_dim)

        if self.use_cls:
            final_vector = output[:, 0, :][None, ...]
            output = output[:, 1:, :]  # (B, N-1, embed_dim)
        else:
            final_vector = output.mean(dim=1)[None, ...]

        # final_vector = output.mean(dim=1)[None, ...] # (1, B, embed_dim)
        if return_hidden:
            return output, final_vector
        return output


