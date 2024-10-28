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

src_KEYS = ["sentence", "meter", "length", "remainder"]
tgt_KEYS = ["sentence", "word", "syllable", "remainder"]
    
def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)  # param1:词嵌入字典大小； param2：每个词嵌入单词的大小
    nn.init.normal_(m.weight, mean=0,
                    std=embedding_dim ** -0.5)  # 正态分布初始化；e.g.,torch.nn.init.normal_(tensor, mean=0, std=1) 使值服从正态分布N(mean, std)，默认值为0，1
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


class MelodyEmbedding(nn.Module):
    def __init__(self, src_tknzr, src_word_emb, event2word_dict, d_embed, drop_prob=0.1, use_pos = False):  # d_embedding = dimension of embedding
        super().__init__()
        
        self.use_pos = use_pos  ## use bar-position
        self.src_tknzr = src_tknzr
        
        self.meter_size = len(src_tknzr)
        self.leng_size = len(event2word_dict['Length'])
        self.rem_size = len(event2word_dict['Remainder'])

        self.total_size = self.meter_size + self.leng_size + self.rem_size
                          
        # Embedding init |  
        # self.sent_emb = Embedding(self.sent_size, d_embed, padding_idx=0)  # 6. Token Embedding
        self.meter_emb = src_word_emb  # [0,1-64],
        self.leng_emb = Embedding(self.leng_size, d_embed, padding_idx=0)  # 对较短的数据进行padding，padding为0
        self.rem_emb = Embedding(self.rem_size, d_embed, padding_idx=0)  # Note_duration, [0,95]
        
        self.token_emb_proj = nn.Linear(3 * d_embed, d_embed)
        
        self.drop_out = nn.Dropout(p=drop_prob)


    def forward(self, meter, length, remainder):
        # print(f"enc inputs: {self.src_tknzr.decode(list(meter[0, :].detach().cpu().squeeze().numpy()))}")
        # sent_embed = self.sent_emb(sentence)
        meter_embed = self.meter_emb(meter)
        leng_embed = self.leng_emb(length)
        rem_embed = self.rem_emb(remainder)
        
        # print(meter_embed.shape, leng_embed.shape)
        
        embeds = [meter_embed, leng_embed, rem_embed]
        # embeds = [meter_embed, leng_embed]
        embeds = torch.cat(embeds, -1)
        embeds = self.token_emb_proj(embeds)
        # embeds = meter_embed

        return embeds