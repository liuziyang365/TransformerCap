import torch
import copy

import torch.nn as nn
import numpy as np
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .attention import MultiHeadedAttention
from .embedding import Embeddings, PositionalEncoding
from .utils import clones, LayerNorm, SublayerConnection, PositionwiseFeedForward


class TransformerCap(nn.Module):

    def __init__(self, config):
        super(TransformerCap, self).__init__()

        self.config = config

        # parameters
        d_model = config.d_model
        d_ff = config.d_ff
        num_layers = config.num_layers
        nhead = config.nhead
        dropout = config.dropout
        feat_dim = config.feat_dim
        vocab_size = config.vocab_size
        lm_dropout = config.lm_dropout

        # built some modules
        c = copy.deepcopy
        attn = MultiHeadedAttention(nhead, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
        decoder_layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)

        # encoder, decoder, generator
        self.encoder = Encoder(encoder_layer, num_layers)
        self.decoder = Decoder(decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.drop = nn.Dropout(lm_dropout)

        # embedding
        self.feat_emb = nn.Sequential(
                                    nn.Linear(feat_dim, d_model),
                                    nn.ReLU(),
                                    nn.Dropout(lm_dropout),
                                  )

        self.cap_emb = Embeddings(d_model, vocab_size)
        self.position = PositionalEncoding(d_model, dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat, cap):
        logit = self.decode(self.encode(feat), cap)
        return logit

    def encode(self, feat):
        feat = self.feat_emb(feat)
        return self.encoder(feat, mask=None)

    def decode(self, memory, cap):
        bs, cap_len = cap.size()
        cap = self.position(self.cap_emb(cap))
        cap_mask = self.subsequent_mask(bs, cap_len).to('cuda')
        out = self.decoder(cap, memory, src_mask=None, tgt_mask=cap_mask)
        logit = self.fc(self.drop(out))
        return logit

    def subsequent_mask(self, bs, size):
        "Mask out subsequent positions."
        attn_shape = (bs, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0
