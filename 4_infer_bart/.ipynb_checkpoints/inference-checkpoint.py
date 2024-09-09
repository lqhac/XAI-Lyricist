# infer Transformer
import os, yaml, pickle, glob, random, subprocess, re
import numpy as np
import miditoolkit
from utils.hparams import hparams, set_hparams
from torch.utils.tensorboard import SummaryWriter
from bartprompt.dataloader import *
import datetime, traceback
from utils.tools.get_time import get_time
import statistics
from nltk.translate.bleu_score import sentence_bleu
from bartprompt.conbart import Bart
from hugtransformers.src.transformers.models.bart.tokenization_bart import BartTokenizer
from hugtransformers.src.transformers import get_linear_schedule_with_warmup
import prosodic as p
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed=1234):  # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN在使用deterministic模式时（下面两行），可能会造成性能下降（取决于model）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(checkpoint_path, device):
    model = Bart(event2word_dict=event2word_dict, 
                 word2event_dict=word2event_dict, 
                 model_pth=hparams['custom_model_dir'],
                 src_tknzr=src_tknzr, 
                 tgt_tknzr=tgt_tknzr,
                 hidden_size=hparams['hidden_size'], 
                 enc_layers=hparams['n_layers'], 
                 num_heads=hparams['n_head'], 
                 enc_ffn_kernel_size=hparams['ffn_hidden'], 
                 dropout=hparams['drop_prob'], 
                 cond=hparams['cond']).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
    model.eval()  # switch to evaluation mode
    print(f"| Successfully loaded bart ckpt from {checkpoint_path}.")
    return model


if __name__ == '__main__':
    set_seed()
    set_hparams()
    
    print(f"Using device: {device} for inferences")
    global src_tknzr, tgt_tknzr
    # tokenizer = BartTokenizer.from_pretrained(hparams['tknzr_dir'])
    
    ######## DATE ########
    ckpt_date = hparams['exp_date']
    ######## DATE ########

    # ---------------------------------------------------------------
    # User Interface Parameter
    # ---------------------------------------------------------------
    batch_size = 1
    temperature, topk = 1.2, 10                      # Sampling strategy
    prompt_size = 10
    inference_max_tokens = 1024
    
    # training conditions (for naming the ckpt)
    lr = hparams['lr']
    cond = hparams['cond']
    
    src_tknzr = BartTokenizer.from_pretrained(hparams['enc_tknzr_dir'])
    tgt_tknzr = BartTokenizer.from_pretrained(hparams['dec_tknzr_dir'])

    # BaselineTransformer(M2Lw)_{cur_time}_lr{lr}
    ckpt_dir = hparams['checkpoint_dir']
    infer_ckpt_dir = os.path.join(ckpt_dir, f"BartForConditionalGeneration_{ckpt_date}_lr{lr}")
    ckpt_path = os.path.join(infer_ckpt_dir, 'bestM2LCkpt.pt')

    # load dictionary
    event2word_dict, word2event_dict, lyric2word_dict, word2lyric_dict = pickle.load(open(f"{hparams['binary_data_dir']}/m2l_dict.pkl", 'rb'))
    
    test_dataset = M2LDataset('valid', event2word_dict, lyric2word_dict, hparams, shuffle=False, is_pretrain=True)
    test_dataloader = build_dataloader(dataset=test_dataset, shuffle=False, batch_size=batch_size)
    print(f"Test Datalodaer = {len(test_dataloader)} Songs")

    # load melody generation model based on skeleton framework
    model = load_model(ckpt_path, device)
    model.eval()
    print(">>>> Successfully loaded M2L Generator!")

    # -------------------------------------------------------------------------------------------
    # Inference file path
    # -------------------------------------------------------------------------------------------
    exp_date = get_time()
    data_output_dir_gen = os.path.join(infer_ckpt_dir, f'gen_lyrics_{exp_date}')
    
    bleu_scores = []
    ppl_scores = []
    
    out_file = open("./bartprompt/out_lyric.txt", "w")
    out_lines = []
    
    matched_cnt, total_cnt = 0, 0
    
    for data_idx, data in enumerate(test_dataloader):
        try:
            if data_idx > 200:
                break
            data_name = data_idx

            enc_inputs = {k: data[f'src_{k}'].to(device) for k in src_KEYS}
            dec_inputs = {k: data[f'tgt_{k}'].to(device) for k in tgt_KEYS}
            
            encoded_cond_str = list(enc_inputs['meter'][0].cpu().detach().numpy())
            cond_inputs_str = src_tknzr.decode(encoded_cond_str)
            
            template = re.search('<template> (.*?) <syllable_', cond_inputs_str).group(1)
            cond_syllable_num = re.search('<syllable_(.*?)>', cond_inputs_str).group(1)
            cond_title = re.search('<title> (.*?) <keywords>', cond_inputs_str).group(1)
            cond_keywords = re.search('<keywords> (.*)', cond_inputs_str).group(1)
            
            # cond_inputs_str = tgt_tknzr.decode(list(enc_inputs['meter'][0].cpu().squeeze().detach().numpy()))
            
            dec_inputs_selected = {'word': dec_inputs['word'][:, :1],
                                  'remainder': dec_inputs['remainder'][:, :1]}
            
            decoded_output, ppl = model.infer(tgt_tknzr=tgt_tknzr, 
                                              enc_inputs=enc_inputs, 
                                              dec_inputs_gt=dec_inputs_selected,
                                              # dec_inputs_full=dec_inputs,
                                              sentence_maxlen=inference_max_tokens, 
                                              temperature=temperature, 
                                              topk=topk,
                                              device=device,
                                              num_syllables=30)
            
            gt_lyric = tgt_tknzr.decode(list(data[f'tgt_word'][0, :].cpu().squeeze().detach().numpy()))
            
            bleu = sentence_bleu(gt_lyric, decoded_output)
            bleu_scores.append(bleu)
            ppl_scores.append(ppl)
            
            out_lyric_without_prompt = p.Text(decoded_output.replace('<s>', '').replace('</s>', '').strip())
            syllable_out_num = len(out_lyric_without_prompt.syllables())
            
            ## process the syllable
            out_meter_pattern = ""
            out_length_pattern = ""
            for syllable in out_lyric_without_prompt.syllables():
                if "'" in str(syllable):
                    mtype = "<strong>"
                elif "`" in str(syllable):
                    mtype= "<substrong>"
                else:
                    mtype = "<weak>"
                length = "<long>" if "ː" in str(syllable) else "<short>"
                out_meter_pattern += f"{mtype} "
                out_length_pattern += f"{length} "
            out_meter_pattern = out_meter_pattern.strip()
            out_length_pattern = out_length_pattern.strip()
            
            if (int(syllable_out_num) == int(cond_syllable_num)) and (out_meter_pattern.strip().replace(' ', '') == template.strip().replace(' ', '')):
                matched_cnt += 1
            total_cnt += 1
            # print(f"PPL Score: {ppl}; BLEU Score: {bleu};\nCond Inputs: {cond_inputs_str}\nTarget Lyric: {decoded_output.replace('<s>', '').replace('</s>', '').strip()}\nSyllables: {syllable_out_num}\nInput Length: {enc_inputs['meter'].shape[-1]}")
            
            out_lines.append(f">> skeleton: {template}\n>> input_syllable_num: {cond_syllable_num}\n>> title: {cond_title}\n>> keywords: {cond_keywords}\n>> gen lyrics: {decoded_output};\n>> gt lyrics:{gt_lyric};\n>> out_syllable_num: {syllable_out_num};\n>> out_meter_pattern: {out_meter_pattern};\n>> out_length_pattern: {out_length_pattern}\n\n")
            
            print(f"Lyric Generation Progression: {data_idx+1}/{len(test_dataloader)}")
        except Exception as e:
            traceback.print_exc()
            print(f"-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-\nBad Item: {data_name}")
    
    out_file.writelines(out_lines)
    out_file.close()
    # print(bleu_scores)
    print(f"matched rate: {matched_cnt/total_cnt}")
    print(f"Average Perplexity over test data: {np.mean(ppl_scores)}")