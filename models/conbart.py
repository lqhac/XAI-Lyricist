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
from models.melody_embedding import MelodyEmbedding
from models.lyric_embedding import LyricEmbedding
import numpy as np
from transformers import BartForConditionalGeneration
from transformers import BartTokenizer
from transformers import get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from positional_encodings.torch_encodings import PositionalEncoding1D
import prosodic as p

    
class Bart(nn.Module):
    def __init__(self, event2word_dict, word2event_dict, model_pth, src_tknzr, tgt_tknzr, hidden_size, enc_layers, num_heads, enc_ffn_kernel_size, dropout, cond=True):
        super(Bart, self).__init__()
        self.event2word_dict = event2word_dict
        self.word2event_dict = word2event_dict
        self.src_tknzr = src_tknzr
        self.tgt_tknzr = tgt_tknzr
        self.enc_layers = enc_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.cond = cond
        self.pos_enc = PositionalEncoding1D(self.hidden_size)
        
        # self.model = BartModel.from_pretrained(model_pth)
        self.model = BartForConditionalGeneration.from_pretrained(model_pth, ignore_mismatched_sizes=True)

        ## embedding layers
        self.src_word_emb = self.model.get_encoder().embed_tokens
        # print(f"Num embeddings: {self.src_word_emb.num_embeddings}")
        self.mel_embed = MelodyEmbedding(self.src_tknzr, self.src_word_emb, self.event2word_dict, self.hidden_size, self.dropout)
        self.tgt_word_emb = self.model.get_decoder().embed_tokens
        self.lyr_embed = LyricEmbedding(self.tgt_tknzr, self.tgt_word_emb, self.event2word_dict, self.hidden_size, self.dropout)
        # self.lm_head = nn.Linear(self.hidden_size, self.lyr_embed.total_size)
        # self.lm_head = nn.Linear(self.hidden_size, len(tokenizer))
        

    def forward(self, enc_inputs, dec_inputs):
        cond_embeds = self.mel_embed(**enc_inputs)
        tgt_embeds = self.tgt_word_emb(dec_inputs['word'])
        # seq = list(dec_inputs['word'][0, :].detach().cpu().squeeze().numpy())
        # print(f"{self.tokenizer.decode(seq)}")
        
        """
        ## create attention masks
        cond_words = enc_inputs['meter']
        tgt_words = dec_inputs['word']
        batch_size_x, x_len = cond_words.shape
        batch_size_y, y_len = tgt_words.shape
        assert batch_size_x == batch_size_y
        enc_attention_mask = torch.zeros((batch_size_x, x_len)).to(cond_words.device)
        dec_attention_mask = torch.zeros((batch_size_y, y_len)).to(cond_words.device)
        cross_attention_mask = torch.zeros((batch_size_x, 1, y_len, x_len)).to(cond_words.device)
        for i in range(batch_size_x):
            sep_pos_x = (cond_words[i] == self.src_tknzr.encode('</s>')[1]).nonzero(as_tuple=True)[0]
            assert len(sep_pos_x) == 1
            sep_x = sep_pos_x[0]
            enc_attention_mask[i, 0:sep_x+1] = 1
        for i in range(batch_size_y):
            sep_pos_y = (tgt_words[i] == self.tgt_tknzr.encode('</s>')[1]).nonzero(as_tuple=True)[0]
            assert len(sep_pos_y) == 1
            sep_y = sep_pos_y[0]
            dec_attention_mask[i, 0:sep_y+1] = 1
        for i in range(batch_size_x):
            sep_pos_x = (cond_words[i] == self.src_tknzr.encode('<sep>')[1]).nonzero(as_tuple=True)[0]
            sep_pos_y = (tgt_words[i] == self.tgt_tknzr.encode('<sep>')[1]).nonzero(as_tuple=True)[0]
            if len(sep_pos_x) != len(sep_pos_y):
                print(len(sep_pos_x), len(sep_pos_y))
                print(self.src_tknzr.decode(cond_words[i].cpu().squeeze().detach().numpy()))
                print(self.tgt_tknzr.decode(tgt_words[i].cpu().squeeze().detach().numpy()))
                assert len(sep_pos_x) == len(sep_pos_y)
            for j, sep_x in enumerate(sep_pos_x):
                if j == 0:
                    cross_attention_mask[i, 0, 0: sep_pos_y[j]+1, 0:sep_x+1] = 1
                else:
                    cross_attention_mask[i, 0, sep_pos_y[j-1]+1:sep_pos_y[j]+1, sep_pos_x[j-1]+1:sep_x+1] = 1
        """
        
        model_outputs = self.model(# attention_mask=enc_attention_mask,
                                   # decoder_attention_mask=dec_attention_mask,
                                   # encoder_attention_mask=cross_attention_mask,
                                   inputs_embeds=cond_embeds,
                                   decoder_inputs_embeds=tgt_embeds,
                                   labels=dec_inputs['word']) 
        return model_outputs
    
    def split_dec_outputs(self, dec_outputs):
        word_out_size = self.lyr_embed.word_size
        rem_out_size = word_out_size + self.lyr_embed.rem_size
        
        word_out = dec_outputs[:, :, : word_out_size]
        rem_out = dec_outputs[:, :, word_out_size : rem_out_size]
        
        return word_out, rem_out
    
    def infer (self, tgt_tknzr, enc_inputs, dec_inputs_gt, sentence_maxlen, temperature, topk, device, num_syllables):
        sampling_func = partial(temperature_sampling, temperature=temperature, topk=topk)

        bsz, _ = dec_inputs_gt['word'].shape
        decode_length = sentence_maxlen  # the max number of Tokens in a midi

        dec_inputs = dec_inputs_gt

        tf_steps = dec_inputs_gt['word'].shape[1]  ## number of teacher-forcing steps
        sentence_len = dec_inputs_gt['word'].shape[1]

        is_end = False
        xe = []
        
        num_syllables_remaining = num_syllables
        sentence_num = 0
        
        for step in tqdm(range(decode_length)):
            cond_embeds = self.mel_embed(**enc_inputs)
            tgt_embeds = self.tgt_word_emb(dec_inputs['word'])
            # gt_size = dec_inputs['word'].shape[-1]
            # gt_labels = dec_inputs_full['word'][:, :gt_size]
            
            """
            ## create attention masks
            cond_words = enc_inputs['meter']
            tgt_words = dec_inputs['word']
            batch_size_x, x_len = cond_words.shape
            batch_size_y, y_len = tgt_words.shape
            assert batch_size_x == batch_size_y
            enc_attention_mask = torch.ones((batch_size_x, x_len)).to(cond_words.device)
            dec_attention_mask = torch.ones((batch_size_y, y_len)).to(cond_words.device)
            cross_attention_mask = torch.zeros((batch_size_x, 1, y_len, x_len)).to(cond_words.device)
            for i in range(batch_size_x):
                sep_pos_x = (cond_words[i] == self.src_tknzr.encode('<sep>')[1]).nonzero(as_tuple=True)[0]
                sep_pos_y = (tgt_words[i] == self.tgt_tknzr.encode('<sep>')[1]).nonzero(as_tuple=True)[0]
                if len(sep_pos_y) == 0: ## start
                    pass
                else:
                    for j, sep_y in enumerate(sep_pos_y):
                        if j == 0:
                            cross_attention_mask[i, 0, 0: sep_y+1, 0:sep_pos_x[j]+1+1] = 1
                        elif j == len(sep_pos_y)-1:
                            cross_attention_mask[i, 0, sep_pos_y[j]+1:, sep_pos_x[j]+1:sep_pos_x[j+1]+1] = 1
                        else:
                            cross_attention_mask[i, 0, sep_pos_y[j-1]+1:sep_pos_y[j]+1, sep_pos_x[j-1]+1:sep_pos_x[j]+1] = 1
            """
            
            model_outputs = self.model(inputs_embeds=cond_embeds,
                                       decoder_inputs_embeds=tgt_embeds)
                                       # labels=gt_labels) 
            predicts = model_outputs.logits
            
            word_predict = predicts

            word_logits = word_predict[:, -1, :].cpu().squeeze().detach().numpy()

            word_id = sampling_func(logits=word_logits)
            
            # xe_loss = model_outputs.loss
            # print(f"loss: {model_outputs.loss}")
            # xe.append(xe_loss)
            
            """
            if word_id in tgt_tknzr.encode("<sep>"):
                sentence_num += 1
                if sentence_num >= tgt_sent_num:
                    break
            """
            
            if word_id in tgt_tknzr.encode("</s>"):
                is_end = True

            if is_end:
                token_out = list(dec_inputs['word'].cpu().squeeze().detach().numpy())
                lyric_out = tgt_tknzr.decode(token_out)
                break
            
            token_str = tgt_tknzr.decode(word_id)
            word_str = token_str.strip()
            word_txt = p.Text(word_str)
            word_syll_num = len(word_txt.syllables())
            
            if token_str[0] == ' ':
                num_syllables_remaining = num_syllables_remaining - word_syll_num
            # num_syllables_token = self.event2word_dict['Remainder'][f"Remain_{num_syllables_remaining}"]
            num_syllables_token = 0
            
            # print(f"wordid: {word_id} word: {token_str}, syllable: {word_syll_num}, remain: {num_syllables_remaining}")
            
            dec_inputs = {
                'word': torch.cat((dec_inputs['word'], torch.LongTensor([[word_id]]).to(device)), dim=1),
                'remainder': torch.cat((dec_inputs['remainder'], torch.LongTensor([[num_syllables_token]]).to(device)), dim=1),
            }
            
            # xe_loss = xe_loss(word_predict[:, :-1], tgt_word) * hparams['lambda_word']
            
        if not is_end:
            token_out = list(dec_inputs['word'].cpu().squeeze().detach().numpy())
            lyric_out = f"{tgt_tknzr.decode(token_out)}</s>" 
            # xe.append(xe_loss)
        
        ppl = 0.0
        # ppl = math.exp(torch.stack(xe).mean())
        return lyric_out, ppl
    
    def saliency (self, tgt_tknzr, enc_inputs, dec_inputs_gt, sentence_maxlen, temperature, topk, device, num_syllables, out_explain):
        out_lines = []
        # explanation = open(".", "w")
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
            tgt_embeds = self.tgt_word_emb(dec_inputs['word'])
            model_outputs = self.model(inputs_embeds=cond_embeds,
                                       decoder_inputs_embeds=tgt_embeds)
            predicts = model_outputs.logits
            
            word_predict = predicts

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
                skeleton_id, length_id, remainder_id = int(enc_inputs['meter'][0, idx]), int(enc_inputs['length'][0, idx]), int(enc_inputs['remainder'][0, idx])
                skeleton_word = tgt_tknzr.decode(skeleton_id)
                length_word = tgt_tknzr.decode(length_id)
                remainder_word = tgt_tknzr.decode(remainder_id)
                contribution[f"Top_{k}"] = (idx, f"word: {skeleton_word.strip()}; length: {length_word.strip()}; remainder: {remainder_word.strip()}", f"relevance: {values[k]}")

            cur_sent = list(dec_inputs['word'][0, :].cpu().squeeze().detach().numpy())
            
            explanation = f"| step: {step}; \n  | cur_sent: {tgt_tknzr.decode(cur_sent)} \n  | cur_word: {tgt_tknzr.decode(word_id)}; \n  | contribution: {contribution} \n"
            
            out_lines.append(explanation)

            print(explanation)
            
            if word_id in tgt_tknzr.encode("</s>"):
                is_end = True
                # xe.append(xe_loss)

            if is_end:
                token_out = list(dec_inputs['word'].cpu().squeeze().detach().numpy())
                lyric_out = tgt_tknzr.decode(token_out)
                out_lines.append(f"\n\n")
                break
            
            token_str = tgt_tknzr.decode(word_id)
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
            lyric_out = f"{tgt_tknzr.decode(token_out)}</s>" 
            out_lines.append(f"\n\n")
            # xe.append(xe_loss)
        
        out_explain.writelines(out_lines)
        ppl = 0.0
        # ppl = math.exp(torch.stack(xe).mean())
        return lyric_out, ppl