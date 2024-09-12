import torch
import torch.nn as nn
import numpy as np


# ===============================================================================
# Mask utils, used to generate mask for attentions
# ===============================================================================

def pad_mask_generator(batch_size, q_size, seq_k):
    # pad mask, q_size is the current sequence length, seq_k is the input sequence before embedding
    pad_attn_mask = (seq_k == 0).unsqueeze(1).expand(batch_size, q_size, seq_k.size(1))
    return pad_attn_mask

def tril_mask_generator(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0