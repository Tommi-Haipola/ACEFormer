'''
The idea of this "transformer" is to utilise the unique capacities of the transformer architecture to make a single inference call for an arbitrary,
but predefined set of data. The model will be fed a string of data, where the input is the first section, and the rest of the input is just 0s.

the loss will be calculated from the predictions for the sections containing 0s. 

'''

import time
import math
from torch import Tensor
import torch
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as Data
import numpy as np
import pandas as pd
import sys
import multiprocessing
from moduledata import EmdData
from attnset import ProbabilityAttention, FullAttention
from pretreatment import PositionalEmbedding, ExpandEmbedding
from module import Distilling, CrossLayer


class OneFormer(nn.Module):

    def __init__(
        self
        ,layers: int = 2
        ,heads: int = 8
        ,embed_dimension: int = 64
        ,dropout: float = 0.1
        ,activation:str = "relu"
        ):
        super(OneFormer, self).__init__()
        
        self.layer = layers

        ## data embedding -- nothing for now

        ## positional Embedding
        self.PosEmbedding = PositionalEncoding(embed_dimension,dropout)
        ## dropout
        self.dropout = nn.Dropout(p=dropout)

        # attention module
        self.attn = nn.MultiheadAttention(embed_dimension,heads,dropout)

        #projection
        self.full_connect = nn.Linear(embed_dimension,1)
    
    def forward(self, data: Tensor):
        # data should be formatted as:
        # batch,length,depth

        # embed data
        data_emb = data.norm()
        pos = self.PosEmbedding(data)
        embeded = self.dropout(data_emb + pos)

        # attention
        attention = self.attn(embeded)

        # projection
        output = self.full_connect(attention)

        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)