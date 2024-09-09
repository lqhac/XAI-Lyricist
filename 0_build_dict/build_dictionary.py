import os, pickle
from utils.hparams import hparams, set_hparams
import collections
import subprocess

# Position
double_positions_bins = set([i * 30 for i in range(0, 64)])
triplet_positions_bins = set([i * 40 for i in range(0, 48)])
positions_bins = sorted((double_positions_bins | triplet_positions_bins))  # 并集

# duration bins, default resol = 480 ticks per beat
double_duration = set([i * 30 for i in range(1, 257)])
triplet_duration = set([40, 80, 160, 320, 640])
duration_bins = list(sorted(double_duration | triplet_duration))

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

def build_dict(save_path):
    # create save dir
    if os.path.exists(save_path):
        subprocess.check_call(f'rm -rf "{save_path}"', shell=True)  # 运行由args参数提供的命令，等待命 执行结束并返回返回码。
        os.makedirs(save_path)
    else:
        os.makedirs(save_path)
    
    # create melody dictionary
    melody_dict = collections.defaultdict(list)
    lyric_dict = collections.defaultdict(list)
        
    lyric_dict['Syllable'].append('0')
    for i in range (1, 10):
        lyric_dict['Syllable'].append(f"Syllable_{i}")
    
    ## phrases
    melody_dict['Phrase'].append('0')
    for i in range(294):
        melody_dict['Phrase'].append(f"Phrase_{i}")
    
    melody_dict['Meter'].append('0') ## padding
    ## strong/weak/substrong
    melody_dict['Meter'].append("Strong")
    melody_dict['Meter'].append("Substrong")
    melody_dict['Meter'].append("Weak")
    
    ## long
    melody_dict['Length'].append('0') ## padding
    melody_dict['Length'].append("Long")
    melody_dict['Length'].append("Short")
    
    melody_dict['Remainder'].append('0')
    for i in range(41):
        melody_dict['Remainder'].append(f"Remain_{i}")
        
    melody_dict['Syllable'].append('0')
    melody_dict['Syllable'].append(f"<Syllable>") ## begin
    for i in range (0, 50):
        melody_dict['Syllable'].append(f"Syllable_{i}")
    
    ## bar
    melody_dict['Bar'].append(0)
    for i in range (0, 200):
        melody_dict['Bar'].append(f"Bar_{i}")
    
    ## position
    melody_dict['Pos'].append(0)
    for pos in positions_bins:
        melody_dict['Pos'].append(f"Pos_{pos}")
    
    ## pitch
    melody_dict['Pitch'].append('<PAD>')
    for pitch in range (0, 128):
        melody_dict['Pitch'].append(f"Pitch_{pitch}")
    
    ## duration
    melody_dict['Dur'].append(0)
    for dur in duration_bins:
        melody_dict['Dur'].append(f"Dur_{dur}") 

    for k, v in melody_dict.items():
         print(f"{k:<15s} : {v}\n")
  
    # melody dictionary
    event2word, word2event = {}, {}
    melody_class = melody_dict.keys()

    for cls in melody_class:
        event2word[cls] = {v:k for k,v in enumerate(melody_dict[cls])}
        word2event[cls] = {k:v for k,v in enumerate(melody_dict[cls])}
    
    # lyric syllable dictionary
    
    ## phrases
    lyric_dict['Phrase'].append("0")
    for i in range(294):
        lyric_dict['Phrase'].append(f"Phrase_{i}")
    
    lyric_dict['Syllable'].append('0')
    for i in range (1, 10):
        lyric_dict['Syllable'].append(f"Syllable_{i}")
    
    lyric_dict['Remainder'].append('0')
    for i in range(41):
        lyric_dict['Remainder'].append(f"Remain_{i}")
    
    lyric2word, word2lyric = {}, {}
    lyric_class = lyric_dict.keys()
    for cls in lyric_class:
        lyric2word[cls] = {v:k for k,v in enumerate(lyric_dict[cls])}
        word2lyric[cls] = {k:v for k,v in enumerate(lyric_dict[cls])}
            
    # pickle.dump((event2word, word2event, lyric2word, word2lyric), open(f'{save_path}/m2l_dict.pkl', 'wb'))
    pickle.dump((event2word, word2event, event2word, word2event), open(f'{save_path}/m2l_dict.pkl', 'wb'))
    
    # print
    print('Melody Dict [class size]')
    for key in melody_class:
        print('> {:20s} : {}'.format(key, len(event2word[key])))
        
    print('Lyric Dict [class size]')
    for key in lyric_class:
        print('> {:20s} : {}'.format(key, len(lyric2word[key])))

    return event2word, word2event, lyric2word, word2lyric



if __name__ == '__main__':
    set_hparams()
    dictionary_save_path = hparams["binary_data_dir"]  # dictionay path
    event2word, word2event, lyric2word, word2lyric = build_dict(save_path=dictionary_save_path)  # build dictionary