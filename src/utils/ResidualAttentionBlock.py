import torch
import torch.nn as nn
from collections import OrderedDict
from .utils import LayerNorm, QuickGELU

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout: float = 0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.mlp.c_proj.WITH_RESIDUAL = 1
        self.dropout = nn.Dropout(dropout)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x
    

class DecoderResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        
        self.attn_1 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)

        self.attn_2 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_2 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.mlp.c_proj.WITH_RESIDUAL = 1
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tril_mask, pad_mask):
        tril_mask.to(dtype=x.dtype, device=x.device)
        pad_mask.to(dtype=x.dtype, device=x.device)
        x = x + self.attn_1(self.ln_1(x), x, x, need_weights=False, attn_mask=tril_mask)[0]
        x = x + self.attn_2(self.ln_2(x), memory, memory, need_weights=False, attn_mask=pad_mask)[0]
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x