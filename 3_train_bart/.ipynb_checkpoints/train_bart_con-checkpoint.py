## train transformer
import os
import pickle
import random
import subprocess
import torch.cuda
import torch
import numpy as np
from utils.hparams import hparams, set_hparams
from torch.utils.tensorboard import SummaryWriter
from utils.earlystopping.protocols import EarlyStopping
# from dataset.dataloader_gpt2 import *
from bartprompt.dataloader import *
import datetime
from utils.get_time import get_time
import gc
from tqdm import tqdm
from utils.warmup import *
import torch.nn.functional as F
# from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
# from hugtransformers.src.transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from bartprompt.conbart import Bart
from hugtransformers.src.transformers.models.bart.tokenization_bart import BartTokenizer
from hugtransformers.src.transformers import get_linear_schedule_with_warmup

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def set_seed(seed=1234):  # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN在使用deterministic模式时（下面两行），可能会造成性能下降（取决于model）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def xe_loss(outputs, targets):
    outputs = outputs.transpose(1, 2)
    return F.cross_entropy(outputs, targets, ignore_index=0, reduction='mean')


def train(train_loader, model, optimizer, scheduler, epoch, total_epoch):
    # define the format of tqdm
    with tqdm(total=len(train_loader), ncols=150, position=0, leave=True) as _tqdm:  # 总长度是data的长度
        _tqdm.set_description('training epoch: {}/{}'.format(epoch + 1, total_epoch))  # 设置前缀更新信息

        # Model Train
        model.train()
        running_loss = 0.0
        train_loss = []
        train_word_loss = []

        for idx, data in enumerate(train_loader):
            # prompt_index = list(data[f'tgt_word'].numpy()).index(50268)
            enc_inputs = {k: data[f'src_{k}'].to(device) for k in src_KEYS}
            dec_inputs = {k: data[f'tgt_{k}'].to(device) for k in tgt_KEYS}
            # dec_inputs = data[f'tgt_word'].to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            word_out = model(enc_inputs, dec_inputs)
            # word_out = word_out.logits
            # tgt_word = (data['tgt_word'].to(device))[:, 1:]
            # word_loss = xe_loss(word_out[:, :-1], tgt_word) * hparams['lambda_word']
            word_loss = word_out.loss
            
            """
            ## sentence loss 
            tgt_sent = (data['tgt_sentence'].to(device))[:, 1:]
            sent_loss = xe_loss(sent_out[:, :-1], tgt_sent) * hparams['lambda_sent']
            """
            
            # word loss
            """
            tgt_word = (data['tgt_word'].to(device))[:, 1:]
            word_loss = xe_loss(word_out[:, :-1], tgt_word) * hparams['lambda_word']
            """
            # word_loss = word_out.loss
            
            """
            ## syllable loss
            tgt_syll = (data['tgt_syllable'].to(device))[:, 1:]
            syll_loss = xe_loss(syll_out[:, :-1], tgt_syll) * hparams['lambda_syll']
            """
            
            ## remainder loss
            """
            tgt_rem = (data['tgt_remainder'].to(device))[:, 1:]
            rem_loss = xe_loss(rem_out[:, :-1], tgt_rem) * hparams['lambda_rem']
            """
            

            # 3) total loss
            total_loss = word_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss.append(total_loss.item())
            running_loss += total_loss.item()
            
            train_word_loss.append(word_loss.item())

            _tqdm.set_postfix(
                loss="{:.6f}, word_loss={:.6f}".format(total_loss, word_loss))
            
            _tqdm.update(2)

    train_loss_avg = np.mean(train_loss)
    train_word_loss_avg = np.mean(train_word_loss)
    
    return train_loss_avg, train_word_loss_avg


def valid(valid_loader, model, epoch, total_epoch):
    # define the format of tqdm
    with tqdm(total=len(valid_loader), ncols=150) as _tqdm:  # 总长度是data的长度
        _tqdm.set_description('validation epoch: {}/{}'.format(epoch + 1, total_epoch))  # 设置前缀更新信息

        model.eval()  # switch to valid mode
        running_loss = 0.0
        val_loss = []
        val_word_loss = []

        with torch.no_grad():
            for idx, data in enumerate((valid_loader)):
                try:
                    # get the inputs;
                    enc_inputs = {k: data[f'src_{k}'].to(device) for k in src_KEYS}
                    dec_inputs = {k: data[f'tgt_{k}'].to(device) for k in tgt_KEYS}
                        
                    word_out = model(enc_inputs, dec_inputs)
                    # word_out = word_out.logits
                    # tgt_word = (data['tgt_word'].to(device))[:, 1:]
                    # word_loss = xe_loss(word_out[:, :-1], tgt_word) * hparams['lambda_word']
                    word_loss = word_out.loss

                    ## sentence loss 
                    """
                    tgt_sent = (data['tgt_sentence'].to(device))[:, 1:]
                    sent_loss = xe_loss(sent_out[:, :-1], tgt_sent) * hparams['lambda_sent']
                    """

                    # word loss
                    """
                    tgt_word = (data['tgt_word'].to(device))[:, 1:]
                    word_loss = xe_loss(word_out[:, :-1], tgt_word) * hparams['lambda_word']
                    """
                    
                    """
                    ## syllable loss
                    tgt_syll = (data['tgt_syllable'].to(device))[:, 1:]
                    syll_loss = xe_loss(syll_out[:, :-1], tgt_syll) * hparams['lambda_syll']
                    """

                    ## remainder loss
                    """
                    tgt_rem = (data['tgt_remainder'].to(device))[:, 1:]
                    rem_loss = xe_loss(rem_out[:, :-1], tgt_rem) * hparams['lambda_rem']  
                    """
                    
                    # 3) total loss
                    total_loss = word_loss
                    val_loss.append(total_loss.item())
                    running_loss += total_loss.item()
                    val_word_loss.append(word_loss.item())

                    _tqdm.set_postfix(
                        loss="{:.6f}, word_loss={:.6f}".format(total_loss, word_loss, ))
                    _tqdm.update(2)
                    
                except Exception as e:
                    print(data)
                    print("Bad Data Item!")
                    print(e)
                    break
            
            val_loss_avg = np.mean(val_loss)
            val_word_loss_avg = np.mean(val_word_loss)

            return val_loss_avg, val_word_loss_avg

            

def train_m2l():
    ## train melody to lyric generation
    gc.collect()
    torch.cuda.empty_cache()
    
    global device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    # args
    set_seed()
    set_hparams()
    event2word_dict, word2event_dict, lyric2word_dict, word2lyric_dict = pickle.load(open(f"{hparams['binary_data_dir']}/m2l_dict.pkl", 'rb'))

    # tensorboard logger
    cur_time = get_time()
    tensorboard_dir = hparams['tensorboard']
    train_log_dir = f'{tensorboard_dir}/{cur_time}/train'
    valid_log_dir = f'{tensorboard_dir}/{cur_time}/valid'
    train_writer = SummaryWriter(log_dir=train_log_dir)
    valid_writer = SummaryWriter(log_dir=valid_log_dir)

    # ------------
    # train
    # ------------
    # load data
    train_dataset = M2LDataset('train', event2word_dict, lyric2word_dict, hparams, shuffle=True, is_pretrain=False)
    valid_dataset = M2LDataset('valid', event2word_dict, lyric2word_dict, hparams, shuffle=False, is_pretrain=False)

    train_loader = build_dataloader(dataset=train_dataset, shuffle=True, batch_size=hparams['batch_size'], endless=False)
    val_loader = build_dataloader(dataset=valid_dataset, shuffle=False, batch_size=hparams['batch_size'], endless=False)
    
    
    src_tknzr = BartTokenizer.from_pretrained(hparams['enc_tknzr_dir'])
    tgt_tknzr = BartTokenizer.from_pretrained(hparams['dec_tknzr_dir'])
    
    print(f"foundation model pth: {hparams['custom_model_dir']}")
    
    def tensor_check_fn(key, param, input_param, error_msgs):
        if param.shape != input_param.shape:
            return False
        return True
    
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
    
    pre_trained_path = hparams['pretrain']
    if pre_trained_path != '':
        current_model_dict = model.state_dict()
        loaded_state_dict = torch.load(pre_trained_path)
        new_state_dict={k:v if v.size()==current_model_dict[k].size() else current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}
        # model.load_state_dict(new_state_dict, strict=False)
        # model.load_state_dict(torch.load(pre_trained_path), strict=False, tensor_check_fn=tensor_check_fn)
        model.load_state_dict(new_state_dict, strict=False)
        print(">>> Load pretrained model successfully")
        
    ## warm up
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams['lr'],
        betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
        weight_decay=hparams['weight_decay'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=hparams['warmup'], num_training_steps=-1
    )

    """
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)
    """

    # training conditions (for naming the ckpt)
    lr = hparams['lr']
    cond = hparams['cond']

    # early stop: initialize the early_stopping object
    # checkpointpath = f"{hparams['checkpoint_dir']}/Cond_{cond}_GPT2_{cur_time}_lr{lr}"
    checkpointpath = f"{hparams['checkpoint_dir']}/BartForConditionalGeneration_{cur_time}_lr{lr}"
    if not os.path.exists(checkpointpath):
        os.mkdir(checkpointpath)
    early_stopping = EarlyStopping(patience=hparams['patience'], verbose=True,
                                   path=f"{checkpointpath}/early_stopping_checkpoint.pt")
    

    # -------- Train & Validation -------- #
    min_valid_running_loss = 1000000  # 假设无限大
    total_epoch = hparams['total_epoch']
    with tqdm(total=total_epoch) as _tqdm:
        for epoch in range(total_epoch):
            # Train
            train_running_loss, train_word_loss = train(train_loader, model, optimizer, scheduler, epoch, total_epoch)
            train_writer.add_scalars("train_epoch_loss", {"running": train_running_loss, 'word': train_word_loss}, epoch)

            # validation  
            valid_running_loss, valid_word_loss = valid(val_loader, model, epoch, total_epoch)
            valid_writer.add_scalars("valid_epoch_loss", {"running": valid_running_loss, 'word': valid_word_loss}, epoch)

            # early stopping Check
            early_stopping(valid_running_loss, model, epoch)
            if early_stopping.early_stop == True:
                print("Validation Loss convergence， Train over")
                break

            # save the best checkpoint
            if valid_running_loss < min_valid_running_loss:
                min_valid_running_loss = valid_running_loss
                torch.save(model.state_dict(), f"{checkpointpath}/bestM2LCkpt.pt")
            print(f"Training Runinng Loss = {train_running_loss}, Validation Running Loss = {min_valid_running_loss}")  
            _tqdm.update(2)



if __name__ == '__main__':
    train_m2l() # train & validation 
