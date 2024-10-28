import os, random, pickle
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.task_utils import batchify
from utils.hparams import hparams, set_hparams
from models.melody_embedding import MelodyEmbedding
from models.lyric_embedding import LyricEmbedding
from utils.indexed_datasets import IndexedDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('GPT2')
embed = GPT2LMHeadModel.from_pretrained('GPT2').transformer.wte

src_KEYS = ["meter", "length", "remainder"]
tgt_KEYS = ["word", "remainder"]

class M2LDataset(Dataset):
    def __init__(self, split, event2word_dict, lyric2word_dict, hparams, shuffle=True, is_pretrain=False):
        super().__init__()
        self.split = split ## train, valid, test
        self.hparams = hparams
        self.batch_size = hparams['batch_size']
        self.event2word_dict = event2word_dict
        self.lyric2word_dict = lyric2word_dict
        self.is_pretrain = is_pretrain
        if self.is_pretrain:
            self.data_dir = f"{hparams['word_data_dir']}"
            self.data_path = f'{hparams["word_data_dir"]}/{self.split}_words.npy'
            print(f"data_dir: {self.data_dir}")
        else:
            self.data_dir = f"{hparams['word_data_dir']}"
            self.data_path = f'{hparams["word_data_dir"]}/{self.split}_words.npy'
        self.ds_name = split ## name of dataset
        self.data = np.load(open(self.data_path, 'rb'), allow_pickle= True)
        # print(f"length of data: {len(self.data)}")
        self.size = np.load(open(f'{hparams["word_data_dir"]}/{self.split}_words_length.npy', 'rb'), allow_pickle= True)
        # print(self.data)
        self.shuffle = shuffle
        self.sent_maxlen = self.hparams['sentence_maxlen'] ## 512
        self.indexed_ds = None ## indexed dataset
        self.indices = [] ## indices to data samples
        
        if shuffle:
            self.indices = list(range(len(self.size)))  ## viz. number of data samples
            random.shuffle(self.indices)
        else:
            self.indices = list(range(len(self.size)))
    
    def ordered_indices(self):
        return self.indices

    def __len__(self):
        return self.size

    def _get_item(self, index):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.ds_name}')
        return self.indexed_ds[index]
    
    def __getitem__(self, idx):
        # obtain one sentence segment according to the index
        item = self._get_item(idx)

        # input and output
        src_tokens = item['src_words']
        tgt_tokens = item['tgt_words']
        
        # print(f"tgt_tokens: {tgt_tokens}")
        
        # to long tensors
        for k in src_KEYS:
            item[f'src_{k}'] = torch.LongTensor([token[k] for token in src_tokens])
        for k in tgt_KEYS:
            item[f'tgt_{k}'] = torch.LongTensor([token[k] for token in tgt_tokens])
            
        return item
    
    @property
    def num_workers(self):
        return int(os.getenv('NUM_WORKERS',10))  
    
    def collater(self, samples):
        # print(samples)
        if len(samples) == 0:
            return {}

        batch = {}
        for k in src_KEYS:
            if k != 'meter':
                batch[f'src_{k}'] = batchify([s[f'src_{k}'] for s in samples], pad_idx=0)
            else:
                batch[f'src_{k}'] = batchify([s[f'src_{k}'] for s in samples], pad_idx=1)
        for k in tgt_KEYS:
            batch[f'tgt_{k}'] = batchify([s[f'tgt_{k}'] for s in samples], pad_idx=1)
        
        batch['n_src_tokens'] = sum([len(s['src_meter']) for s in samples])
        batch['n_tgt_tokens'] = sum([len(s['tgt_word']) for s in samples])
        # batch['n_tokens'] = torch.LongTensor([s['n_tokens'] for s in samples])
        batch['input_path'] = [s['input_path'] for s in samples]
        batch['item_name'] = [s['item_name'] for s in samples]
        return batch
        


def build_dataloader(dataset, shuffle, batch_size=10, endless=False):
    def shuffle_batches(batches):
        np.random.shuffle(batches)  # shuffle： 随机打乱数据
        return batches

    # batch sample and endless
    indices = dataset.ordered_indices()

    batch_sampler = []
    for i in range(0, len(indices), batch_size):
        batch_sampler.append(indices[i:i + batch_size])  # batch size [0:20],

    if shuffle:  # 是否随机打乱
        batches = shuffle_batches(list(batch_sampler))
        if endless:  # 是否增加batches数量，随机重复添加20次
            batches = [b for _ in range(20) for b in shuffle_batches(list(batch_sampler))]
    else:
        batches = batch_sampler
        if endless:
            batches = [b for _ in range(20) for b in batches]
    
    num_workers = dataset.num_workers
    return torch.utils.data.DataLoader(
        dataset, collate_fn=dataset.collater, num_workers=num_workers,
        batch_sampler=batches, pin_memory=False)



if __name__ == '__main__':
    set_hparams()
    # load dictionary
    event2word_dict, word2event_dict, lyric2word_dict, word2lyric_dict = pickle.load(open(f"{hparams['binary_data_dir']}/m2l_dict.pkl", 'rb'))
    batch_size = hparams['batch_size']

    valid_dataset = M2LDataset('valid', event2word_dict, lyric2word_dict, hparams, shuffle=True)
    valid_dataloader = build_dataloader(dataset=valid_dataset, shuffle=True, batch_size=1, endless=True)

    print("length of train_dataloader", len(valid_dataloader))
    
    # Test embedding
    for idx, item in enumerate(tqdm(valid_dataloader)):
        enc_inputs = {k: item[f'src_{k}'] for k in src_KEYS}
        dec_inputs = {k: item[f'tgt_{k}'] for k in tgt_KEYS}
        melody_embed = MelodyEmbedding(tokenizer, embed, event2word_dict, hparams['hidden_size'])
        lyric_embed = LyricEmbedding(tokenizer, embed, lyric2word_dict, hparams['hidden_size'])
        enc_emb_shape = melody_embed(**enc_inputs)
        dec_emb_shape = lyric_embed(**dec_inputs)