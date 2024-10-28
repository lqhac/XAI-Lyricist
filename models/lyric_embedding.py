import os
import json
import yaml
import pickle
import datetime
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from collections import OrderedDict
from utils.hparams import hparams, set_hparams
from positional_encodings.torch_encodings import PositionalEncoding1D

    
def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)  # param1:词嵌入字典大小； param2：每个词嵌入单词的大小
    nn.init.normal_(m.weight, mean=0,
                    std=embedding_dim ** -0.5)  # 正态分布初始化；e.g.,torch.nn.init.normal_(tensor, mean=0, std=1) 使值服从正态分布N(mean, std)，默认值为0，1
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class LyricEmbedding(nn.Module):
    def __init__(self, tgt_tknzr, tgt_word_emb, lyric2word_dict, d_embed, drop_prob=0.1, use_cond = False):  # d_embedding = dimension of embedding
        super().__init__()
        
        self.use_cond = use_cond
        
        if use_cond:
            self.word_size = len(tgt_tknzr)
            self.rem_size = len(lyric2word_dict['Remainder'])

            self.total_size = self.word_size + self.rem_size

            # Embedding init |  
            self.tgt_word_emb = tgt_word_emb  # [0,1-64],
            self.rem_emb = Embedding(self.rem_size, d_embed, padding_idx=0)  # Note_duration, [0,95]

            self.token_emb_proj = nn.Linear(2 * d_embed, d_embed)
        
        else:
            self.word_size = len(tgt_tknzr)
            self.tgt_word_emb = tgt_word_emb  # [0,1-64],
            self.total_size = self.word_size
        
        self.drop_out = nn.Dropout(p=drop_prob)


    def forward(self, word, remainder):
        if self.use_cond:
            word_embed = self.tgt_word_emb(word)
            rem_embed = self.rem_emb(remainder)

            embeds = [word_embed, rem_embed]
            embeds = torch.cat(embeds, -1)
            embeds = self.token_emb_proj(embeds)
        else:
            word_embed = self.tgt_word_emb(word)
            embeds = word_embed

        return embeds