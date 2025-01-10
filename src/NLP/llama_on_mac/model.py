# # Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# github@Matzohe

import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = "mps"


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

# rotary position embedding fucntion
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    compute the angles that will be used in the RoPE

    Args:
        dim (int): the model embedding dimention
        end (int): the max length of the sequence
        theta (float, optional): the theta value. Defaults to 10000.0.

    Returns:
        torch.Tensor: the angles that will be used in the RoPE
    """
    
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    reshape the freqs_cis as the same shape with the x tensor.
    Args:
        freqs_cis (torch.Tensor): the freqs_cis tensor
        x (torch.Tensor): the tensor to be add the embedding

    Returns:
        torch.Tensor: return the reshaped freqs_cis
    """
    ndim = x.ndim
    assert 0 <= 1 < x.ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]

    return freqs_cis.view(*shape)


 
def apply_rotary_emb(xq: torch.tensor, xk: torch.tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    apply the rotary position embedding for the query and key
    Args:
        xq (torch.tensor): the query
        xk (torch.tensor): the key
        freqs_cis (torch.Tensor): position complex numbers

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: the tensors after position embedding
    """

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[: -1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[: -1], -1, 2))

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    for kv cache, each token in the sequence have n_kv_heads heads, we need to repeat the n_kv_heads to n_heads

    Args:
        x (torch.Tensor): input key or value
        n_rep (int): repeat for n times

    Returns:
        torch.Tensor: shape is [batch_size, sequence_length, n_heads, head_dim]
    """

    bsz, sql, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bsz, sql, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, sql, n_kv_heads * n_rep, head_dim)
    )



# apply the kv cache, running on mps device.
class Attention(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.device = args.device
        self.n_heads = args.n_heads
        self.kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        assert args.dim % args.n_heads == 0
        self.head_dim = args.dim // args.n_heads
        self.n_rep = self.n_heads // self.kv_heads

        self.wq = nn.Linear(args.dim, self.head_dim * self.n_heads, bias=False)
        self.wk = nn.Linear(args.dim, self.head_dim * self.kv_heads, bias=False)
        self.wv = nn.Linear(args.dim, self.head_dim * self.kv_heads, bias=False)
        self.wo = nn.Linear(self.head_dim * self.n_heads, args.dim, bias=False)

        self.cache_k = torch.zeros(
            size=(args.max_batch_size, args.max_seq_len, self.kv_heads, self.head_dim),
        ).to(self.device)

        self.cache_v = torch.zeros(
            size=(args.max_batch_size, args.max_seq_len, self.kv_heads, self.head_dim),
        ).to(self.device)

    
    def forward(self, 
                x: torch.Tensor, 
                start_pos: int, 
                freqs_cis: torch.Tensor, 
                mask: Optional[torch.Tensor],
            ) -> torch.Tensor:
        
        bsz, sql, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        
        q = q.view(bsz, sql, self.n_heads, self.head_dim)
        k = k.view(bsz, sql, self.kv_heads, self.head_dim)
        v = v.view(bsz, sql, self.kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        self.cache_k = self.cache_k.to(q)
        self.cache_v = self.cache_v.to(q)

        self.cache_k[:bsz, start_pos: start_pos + sql, :, :] = k
        self.cache_v[:bsz, start_pos: start_pos + sql, :, :] = v

        keys = self.cache_k[:bsz, : start_pos + sql]
        values = self.cache_v[:bsz, : start_pos + sql]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = q.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 3, 1)
        values = values.permute(0, 2, 1, 3)

        attn = torch.matmul(xq, keys) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn.float(), dim=-1).type_as(xq)
        output = torch.matmul(attn, values)
        output = output.permute(0, 2, 1, 3).contiguous().view(bsz, sql, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
    

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)
            # a new line append hear, to deal with the bug exist in mps device
            mask = torch.where(torch.isnan(mask), torch.tensor(0.0).to(device="mps"), mask)

            
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
