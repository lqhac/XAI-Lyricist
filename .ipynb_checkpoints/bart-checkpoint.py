import torch
import os
from torch import nn
from tqdm import tqdm
import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
from torch.nn import Parameter
import math
import torch.onnx.operators
import torch.nn.functional as F
from collections import defaultdict
from functools import partial
from utils.infer_utils import temperature_sampling
from bartprompt.melody_embedding import MelodyEmbedding
from bartprompt.lyric_embedding import LyricEmbedding
import numpy as np
from hugtransformers.src.transformers.models.bart.modeling_bart import BartModel, BartForConditionalGeneration
from hugtransformers.src.transformers.models.bart.tokenization_bart import BartTokenizer
from hugtransformers.src.transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from positional_encodings.torch_encodings import PositionalEncoding1D
import prosodic as p

    
class Bart(nn.Module):
    def __init__(self, event2word_dict, word2event_dict, model_pth, tokenizer, hidden_size, enc_layers, num_heads, enc_ffn_kernel_size, dropout, cond=True):
        super(Bart, self).__init__()
        self.event2word_dict = event2word_dict
        self.word2event_dict = word2event_dict
        self.tokenizer = tokenizer
        self.enc_layers = enc_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.cond = cond
        self.pos_enc = PositionalEncoding1D(self.hidden_size)
        
        self.model = BartModel.from_pretrained(model_pth)
        # self.model = BartForConditionalGeneration.from_pretrained(model_pth)

        ## embedding layers
        self.mel_embed = MelodyEmbedding(self.event2word_dict, self.hidden_size, self.dropout)
        self.word_emb = self.model.get_decoder().embed_tokens
        self.lyr_embed = LyricEmbedding(self.tokenizer, self.word_emb, self.event2word_dict, self.hidden_size, self.dropout)
        self.lm_head = nn.Linear(self.hidden_size, self.lyr_embed.total_size)
        # self.lm_head = nn.Linear(self.hidden_size, len(tokenizer))
        

    def forward(self, enc_inputs, dec_inputs):
        cond_embeds = self.mel_embed(**enc_inputs)
        # tgt_embeds = self.word_emb(dec_inputs['word'])
        tgt_embeds = self.lyr_embed(**dec_inputs)
        # seq = list(dec_inputs['word'][0, :].detach().cpu().squeeze().numpy())
        # print(f"{self.tokenizer.decode(seq)}")
        model_outputs = self.model(
            # decoder_input_ids=dec_inputs['word'],
            inputs_embeds=cond_embeds,
            decoder_inputs_embeds=tgt_embeds
        )
        outputs = self.lm_head(model_outputs.last_hidden_state)
        # return outputs
        return self.split_dec_outputs(outputs)
    
    def split_dec_outputs(self, dec_outputs):
        word_out_size = self.lyr_embed.word_size
        rem_out_size = word_out_size + self.lyr_embed.rem_size
        
        word_out = dec_outputs[:, :, : word_out_size]
        rem_out = dec_outputs[:, :, word_out_size : rem_out_size]
        
        return word_out, rem_out
    
    def infer (self, tokenizer, enc_inputs, dec_inputs_gt, sentence_maxlen, temperature, topk, device, num_syllables):
        sampling_func = partial(temperature_sampling, temperature=temperature, topk=topk)

        bsz, _ = dec_inputs_gt['word'].shape
        decode_length = sentence_maxlen  # the max number of Tokens in a midi

        dec_inputs = dec_inputs_gt

        tf_steps = dec_inputs_gt['word'].shape[1]  ## number of teacher-forcing steps
        sentence_len = dec_inputs_gt['word'].shape[1]

        is_end = False
        xe = []
        
        num_syllables_remaining = num_syllables
        for step in tqdm(range(decode_length)):
            cond_embeds = self.mel_embed(**enc_inputs)
            tgt_embeds = self.lyr_embed(**dec_inputs)
            model_outputs = self.model(inputs_embeds=cond_embeds,
                                       decoder_inputs_embeds=tgt_embeds)
            predicts = self.lm_head(model_outputs.last_hidden_state)
            
            word_predict, rem_predict = self.split_dec_outputs(predicts)

            word_logits = word_predict[:, -1, :].cpu().squeeze().detach().numpy()

            word_id = sampling_func(logits=word_logits)
            
            if word_id in tokenizer.encode("</s>"):
                is_end = True
                # xe.append(xe_loss)

            if is_end:
                token_out = list(dec_inputs['word'].cpu().squeeze().detach().numpy())
                lyric_out = tokenizer.decode(token_out)
                break
            
            token_str = tokenizer.decode(word_id)
            word_str = token_str.strip()
            word_txt = p.Text(word_str)
            word_syll_num = len(word_txt.syllables())
            
            if token_str[0] == ' ':
                num_syllables_remaining = num_syllables_remaining - word_syll_num
            num_syllables_token = self.event2word_dict['Remainder'][f"Remain_{num_syllables_remaining}"]
            
            print(f"wordid: {word_id} word: {token_str}, syllable: {word_syll_num}, remain: {num_syllables_remaining}")
            
            dec_inputs = {
                'word': torch.cat((dec_inputs['word'], torch.LongTensor([[word_id]]).to(device)), dim=1),
                'remainder': torch.cat((dec_inputs['remainder'], torch.LongTensor([[num_syllables_token]]).to(device)), dim=1),
            }
            
            # xe_loss = xe_loss(word_predict[:, :-1], tgt_word) * hparams['lambda_word']
            
        if not is_end:
            token_out = list(dec_inputs.cpu().squeeze().detach().numpy())
            lyric_out = f"{tokenizer.decode(token_out)}</s>" 
            # xe.append(xe_loss)

        ppl = 0.0
        # ppl = math.exp(torch.stack(xe).mean())
        return lyric_out, ppl
    
    def saliency (self, tokenizer, enc_inputs, dec_inputs_gt, sentence_maxlen, temperature, topk, device, num_syllables):
        sampling_func = partial(temperature_sampling, temperature=temperature, topk=topk)

        bsz, _ = dec_inputs_gt['word'].shape
        decode_length = sentence_maxlen  # the max number of Tokens in a midi

        dec_inputs = dec_inputs_gt

        tf_steps = dec_inputs_gt['word'].shape[1]  ## number of teacher-forcing steps
        sentence_len = dec_inputs_gt['word'].shape[1]

        is_end = False
        xe = []
        saliency = []
        
        num_syllables_remaining = num_syllables
        for step in tqdm(range(decode_length)):
            cond_embeds = self.mel_embed(**enc_inputs)
            cond_embeds = torch.autograd.Variable(cond_embeds, requires_grad=True)
            cond_embeds.retain_grad()
            tgt_embeds = self.lyr_embed(**dec_inputs)
            model_outputs = self.model(inputs_embeds=cond_embeds,
                                       decoder_inputs_embeds=tgt_embeds)
            predicts = self.lm_head(model_outputs.last_hidden_state)
            
            word_predict, rem_predict = self.split_dec_outputs(predicts)

            word_logits = word_predict[:, -1, :].cpu().squeeze().detach().numpy()

            word_id = sampling_func(logits=word_logits)
            
            relevance = word_predict[0, -1, word_id]
            relevance.backward(retain_graph=True)
            
            ## contribution of template inputs
            sal = cond_embeds.grad.data.abs()
            sal_cur, _ = torch.max(sal, dim=2)
            
            saliency.append(sal_cur.cpu().squeeze().detach().numpy())
            values, indices = torch.topk(sal_cur, k=5)
            values = values.cpu().squeeze().detach().numpy()
            indices = indices.cpu().squeeze().detach().numpy()
            
            contribution = {}
            for k, idx in enumerate(indices):
                skeleton_id, length_id, meter_id = int(enc_inputs['meter'][0, idx]), int(enc_inputs['length'][0, idx]), int(enc_inputs['remainder'][0, idx])
                skeleton_word = self.word2event_dict['Meter'][skeleton_id]
                length_word = self.word2event_dict['Length'][length_id]
                remainder_word = self.word2event_dict['Remainder'][meter_id]
                contribution[f"Top_{k}"] = (idx, skeleton_word, length_word, remainder_word, values[k])

            cur_sent = list(dec_inputs['word'][0, :].cpu().squeeze().detach().numpy())

            print(f"| step: {step}; \n  | cur_sent: {tokenizer.decode(cur_sent)} \n  | cur_word: {tokenizer.decode(word_id)}; \n  | contribution: {contribution} \n")
            
            if word_id in tokenizer.encode("</s>"):
                is_end = True
                # xe.append(xe_loss)

            if is_end:
                token_out = list(dec_inputs['word'].cpu().squeeze().detach().numpy())
                lyric_out = tokenizer.decode(token_out)
                break
            
            token_str = tokenizer.decode(word_id)
            word_str = token_str.strip()
            word_txt = p.Text(word_str)
            word_syll_num = len(word_txt.syllables())
            
            if token_str[0] == ' ':
                num_syllables_remaining = num_syllables_remaining - word_syll_num
            num_syllables_token = self.event2word_dict['Remainder'][f"Remain_{num_syllables_remaining}"]
            
            print(f"wordid: {word_id} word: {token_str}, syllable: {word_syll_num}, remain: {num_syllables_remaining}")
            
            dec_inputs = {
                'word': torch.cat((dec_inputs['word'], torch.LongTensor([[word_id]]).to(device)), dim=1),
                'remainder': torch.cat((dec_inputs['remainder'], torch.LongTensor([[num_syllables_token]]).to(device)), dim=1),
            }
            
            # xe_loss = xe_loss(word_predict[:, :-1], tgt_word) * hparams['lambda_word']
            
        if not is_end:
            token_out = list(dec_inputs.cpu().squeeze().detach().numpy())
            lyric_out = f"{tokenizer.decode(token_out)}</s>" 
            # xe.append(xe_loss)

        ppl = 0.0
        # ppl = math.exp(torch.stack(xe).mean())
        return lyric_out, ppl