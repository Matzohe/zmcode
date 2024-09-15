import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

from ..utils.ResidualAttentionBlock import ResidualAttentionBlock, DecoderResidualAttentionBlock
from ..utils.MaskUtils import tril_mask_generator, pad_mask_generator
from ..utils.PositionEmbeddingUtils import sinusoidal_position_embedding

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.n_layer = int(config.TRANSFORMER["n_layer"])
        self.embedding_dim = int(config.TRANSFORMER["embedding_dim"])
        self.vocab_size = int(config.TRANSFORMER["vocab_size"])
        self.n_head = int(config.TRANSFORMER["n_head"])
        self.dropout = float(config.TRANSFORMER["dropout"])
        self.max_length = int(config.TRANSFORMER["max_length"])

        self.start_of_translate = ["<SOS>"]

        self.position_embedding = sinusoidal_position_embedding(self.embedding_dim, max_len=self.max_length)
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.encoder = Encoder(self.n_layer, self.embedding_dim, self.n_head, self.dropout)
        self.decoder = Decoder(self.n_layer, self.embedding_dim, self.n_head)
        self.c_proj = nn.Linear(self.embedding_dim, self.vocab_size, bias=False)
        # share the parameters
        self.c_proj.weight = self.token_embedding.weight

    def forward(self, text_idx, pool=True):
        
        embedding_output = self.token_embedding(text_idx) + self.position_embedding(text_idx)
        encoder_output = self.encoder(embedding_output)
        
        decoder_output = self.decoder(encoder_output)
        if pool:
            pooler_output = self.c_proj(decoder_output)
            return pooler_output
        else:
            return decoder_output

    
class Encoder(nn.Module):
    def __init__(self, n_layer, embedding_dim, n_head, dropout: float = 0.1):
        super(Encoder, self).__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.embedding_dim = embedding_dim
        self.model = nn.Sequential(*[ResidualAttentionBlock(self.embedding_dim, self.n_head, dropout=dropout) for _ in range(self.n_layer)])
        self.ln = nn.LayerNorm(self.embedding_dim)

    def forward(self, embedding_output):
        return self.ln(self.model(embedding_output))

    
class Decoder(nn.Modules):
    def __init__(self, n_layer, embedding_dim, n_head, dropout: float = 0.1, max_len=1024):
        self.n_layer = n_layer
        self.n_head = n_head
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.model = nn.Sequential(*[DecoderResidualAttentionBlock(self.embedding_dim, self.n_head, dropout=dropout) for _ in range(self.n_layer)])

    def forward(self, x, encoder_output, text_idx):
        tril_mask = tril_mask_generator(size=x.shape(-1))
        pad_mask = pad_mask_generator(x.shape[0], x.shape[1], text_idx)
        # TODO: How to train a transformer model？How can we get to know the length of the text?
        while len(x) < self.max_len:
            input_info = x
            for model in self.model:
                input_info = model(input_info, encoder_output, tril_mask, pad_mask)
            x = torch.cat([x, input_info[-1, :, :]], dim=1)
        return x