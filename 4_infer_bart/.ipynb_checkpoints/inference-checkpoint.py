# infer Transformer
import os, yaml, pickle, glob, random, subprocess, re
import numpy as np
import miditoolkit
from utils.hparams import hparams, set_hparams
from torch.utils.tensorboard import SummaryWriter
from models.dataloader import *
import datetime, traceback
from utils.tools.get_time import get_time
import statistics
from nltk.translate.bleu_score import sentence_bleu
from models.conbart import Bart
from transformers import BartTokenizer
from transformers import get_linear_schedule_with_warmup
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
    
    print(f"Using device: {device} for inferences custom samples")
    global src_tknzr, tgt_tknzr

    # ---------------------------------------------------------------
    # User Interface Parameter
    # ---------------------------------------------------------------
    batch_size = 1
    temperature, topk = 1.3, 3
    prompt_size = 10
    inference_max_tokens = 1024
    
    lr = hparams['lr']
    cond = hparams['cond']
    
    src_tknzr = BartTokenizer.from_pretrained(hparams['enc_tknzr_dir'])
    tgt_tknzr = BartTokenizer.from_pretrained(hparams['dec_tknzr_dir'])

    ckpt_path = os.path.join('./checkpoints', 'bestM2LCkpt.pt')

    # load dictionary
    event2word_dict, word2event_dict, lyric2word_dict, word2lyric_dict = pickle.load(open(f"{hparams['binary_data_dir']}/m2l_dict.pkl", 'rb'))
    
    test_dataset = M2LDataset('valid', event2word_dict, lyric2word_dict, hparams, shuffle=True, is_pretrain=True)
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
    output_lyrics_dir = hparams['output_lyrics_dir']
    data_output_dir_gen = os.path.join(output_lyrics_dir, f'gen_lyrics_{exp_date}')
    output_dir = data_output_dir_gen
    
    # out_lines = []
    
    for data_idx, data in enumerate(test_dataloader):
        bleu_scores = []
        ppl_scores = []
        out_lines = []
        out_lines_gt = []
        out_lines_gen = []
        matched_cnt, total_cnt = 0, 0
        matched_line_cnt, total_line_cnt = 0, 0
        try:
            if data_idx > 30:
                break

            data_name = data_idx

            enc_inputs = {k: data[f'src_{k}'].to(device) for k in src_KEYS}
            dec_inputs = {k: data[f'tgt_{k}'].to(device) for k in tgt_KEYS}
            
            encoded_cond_str = list(enc_inputs['meter'][0].cpu().detach().numpy())
            cond_inputs_str = src_tknzr.decode(encoded_cond_str)
            
            template = cond_inputs_str
            templates = cond_inputs_str.split('.')

            try:
                cond_title = re.search('<title> (.*?) <syllable_', cond_inputs_str).group(1)
            except Exception as e:
                cond_title = f"{data_idx}"
            
            dec_inputs_selected = {'word': dec_inputs['word'][:, :2],
                                  'remainder': dec_inputs['remainder'][:, :2]}
            
            decoded_output, ppl = model.infer(tgt_tknzr=tgt_tknzr, 
                                              enc_inputs=enc_inputs, 
                                              dec_inputs_gt=dec_inputs_selected,
                                              # dec_inputs_full=dec_inputs,
                                              sentence_maxlen=inference_max_tokens, 
                                              temperature=temperature, 
                                              topk=topk,
                                              device=device,
                                              num_syllables=30)
            ## The human-composed lyric
            gt_lyrics = tgt_tknzr.decode(list(data[f'tgt_word'][0, :].cpu().squeeze().detach().numpy())).replace('<s>', '').replace('</s>', '').strip().split('.')
            
            ppl_scores.append(ppl)
            
            ## Generated Lyrics
            decoded_output_without_fixes = decoded_output.replace('<s>', '').replace('</s>', '').strip()
            decoded_outputs = decoded_output_without_fixes.split('.')
            out_lyric_without_prompt = p.Text(decoded_output.replace('<s>', '').replace('</s>', '').strip())
            out_sents = ""
            out_sents_gt = ""
            out_sents_gen = ""
            assert len(gt_lyrics) == len(templates)
            max_len = max(len(gt_lyrics), len(decoded_outputs))
            
            ## Print the generated lyrics without any prefixes
            print(decoded_output_without_fixes)
            
            total_line_cnt += 1
            is_mismatch = False
            
            if len(gt_lyrics) != len(decoded_outputs):
                is_mismatch = True
                out_sents += f"\n>> Mismatched Line Number \n"
                print(f"Mismatched Lines. GT:{len(gt_lyrics)}; Gen:{len(decoded_outputs)}")
                min_len = min(len(gt_lyrics), len(decoded_outputs))
                gt_lyrics = gt_lyrics[:min_len]
                templates = templates[:min_len]
                decoded_outputs = decoded_outputs[:min_len]
                # continue
            else:
                print(f"matched line number")
                out_sents += f"\n>> Matched Line Number \n"
                matched_line_cnt += 1
                min_len = len(gt_lyrics)
            
            ## Align them
            line_padding = "### Empty Line ###"
            for sent_idx in range(max_len):
                # if not is_mismatch:
                total_cnt += 1
                out_sent = decoded_outputs[sent_idx].strip() if sent_idx < len(decoded_outputs) else line_padding
                gt_sent = gt_lyrics[sent_idx].strip() if sent_idx < len(gt_lyrics) else line_padding
                temp_sent = templates[sent_idx].strip() if sent_idx < len(templates) else line_padding
                print(f"temp_sent: {temp_sent}")
                try:
                    cond_syllable_num = re.search('<syllable_(.*?)>', temp_sent).group(1)
                except Exception as e:
                    # print(e)
                    continue ## skip eos
                try:
                # cond_meter_pattern = re.search('<template>(.*?)<keywords>', temp_sent).group(1)
                    cond_meter_pattern = re.search('<template>(.*?)', temp_sent).group(1)
                except Exception as e:
                    continue ## skip eos
                out_sent_text = p.Text(out_sent.strip())
                syllable_out_num = len(out_sent_text.syllables())
                if abs(syllable_out_num - int(cond_syllable_num)) < 3:
                    matched_cnt += 1
                ## calculate the output skeleton
                out_meter_pattern = ""
                out_length_pattern = ""
                out_ipa_notations = ""
                for syllable in out_sent_text.syllables():
                    out_ipa_notations += f"[{str(syllable)}]\t"
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
                out_ipa_notations = out_ipa_notations.strip()
                
                ## calculate sentence bleu between patterns
                references = [src_tknzr.encode(cond_meter_pattern.strip().replace(" ", ""))]
                generated = src_tknzr.encode(out_meter_pattern.strip().replace(" ", ""))
                meter_bleu_score = sentence_bleu(references, generated)
                bleu_scores.append(meter_bleu_score)
                print(f"bleu: {meter_bleu_score}")
                
                ## calculate GT ipa notations
                gt_ipa_notations = ""
                gt_sent_text = p.Text(gt_sent.strip())
                for syllable in gt_sent_text.syllables():
                    gt_ipa_notations += f"[{str(syllable)}]\t"
                
                out_sents += f"\nLine_{sent_idx}: \n"
                out_sents += f"Skeleton_I:\t{temp_sent}\n"  ## input skeleton
                out_sents += f"Skeleton_O:\t<syllable_{syllable_out_num}> <template> {out_meter_pattern}\n"
                out_sents += f"Lyric_GT:\t{gt_sent.strip()}\n"
                out_sents += f"Lyric_Gen:\t{out_sent.strip()}\n"
                out_sents += f"GT:\t{gt_ipa_notations}\n"
                out_sents += f"GE:\t{out_ipa_notations}\n"
                out_sents += f""
                
                out_sents_gt += f"{gt_sent.strip()}\n"
                out_sents_gen += f"{out_sent.strip()}\n"
            
            avg_bleu = np.mean(bleu_scores)
            ## write files
            if not os.path.exists(f"{output_dir}/{cond_title}"):
                os.makedirs(f"{output_dir}/{cond_title}", exist_ok=True)
            out_file = open(f"{output_dir}/{cond_title}/{cond_title}_exp_t{temperature}_k{topk}_bleu{avg_bleu}.txt", "w")
            out_file_gt = open(f"{output_dir}/{cond_title}/{cond_title}_gt.txt", "w")
            out_file_gen = open(f"{output_dir}/{cond_title}/{cond_title}_gen_t{temperature}_k{topk}_bleu{avg_bleu}.txt", "w")
            
            out_file.write(out_sents)
            out_file_gt.write(out_sents_gt)
            out_file_gen.write(out_sents_gen)
            out_lines.append(f">> title: {cond_title}\n>> generated: {out_sents}\n\n")
            
            out_file.close()
            out_file_gt.close()
            out_file_gen.close()
            
            print(f"Lyric Generation Progression: {data_idx+1}/{len(test_dataloader)}")
        except Exception as e:
            traceback.print_exc()
            print(f"-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-\nBad Item: {data_name}")