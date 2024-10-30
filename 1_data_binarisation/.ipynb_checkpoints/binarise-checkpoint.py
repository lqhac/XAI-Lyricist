import os
import glob
import pickle
import random
import traceback
import subprocess
import numpy as np
import miditoolkit
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from utils.hparams import hparams, set_hparams
from utils.indexed_datasets import IndexedDatasetBuilder
from utils.prosody_utils import getProsody
import cv2 as cv
from sentence_splitter import SentenceSplitter
from transformers import BartTokenizer
import prosodic as p

# tempo interval 
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]
comma = '<SEP>'
ticks_per_beat = 480
def_vel = 80  ## default value of velocity
dur2id_dict = dict([(str(x / 100), 129 + i) for i, x in enumerate(list(range(25, 3325, 25)))])
id2dur_dict = {v:k for k, v in dur2id_dict.items()}

class Item(object):
    def __init__(self, name, start, end,  pitch=0, vel=0, value='0', priority=-1):
        self.name = name  # ['Structure, Phrase, Chord, Notes']
        self.start = start  # start step
        self.end = end  # end step
        self.pitch = pitch
        self.vel = vel
        self.value = value  # Chord type or Structure type
        self.priority = priority  # priority: Structure =1, Phrase = 2, Chord = 3, Notes = 4

    def __repr__(self):
        return f'Item(name={self.name:>8s},  start={self.start:>8d}, end={self.end:>8d}, ' \
               f'vel={self.vel:>3d}, pitch={self.pitch:>3d}, ' \
               f'value={self.value:>4s}, priority={self.priority:>2d})\n'

    def __eq__(self, other):
        return self.name == other.name and self.start == other.start and \
               self.pitch == other.pitch and self.priority == other.priority
    
def str2items(lyric):
    lyr = lyric.split()
    
    note_items = []
    word_items = ""
    
    ## process lyrics
    for word in lyr:
        if word == '[sep]':
            word_items += ", "
        else:
            word_items += f"{word.strip()} "
    word_items += ('<|endoftext|> ')
    
    syllable_items = [1 for _ in range (10)]
    
    assert len(syllable_items) > 0 and len(word_items) > 0
    # assert len(syllable_items)==len(word_items)
    
    return word_items, syllable_items
                

class Event(object):
    def __init__(self, name, value, bar=0, pos=0, pitch=0, dur=0, vel=0): # name_value : Bar_0, Postion_30, Note_64...
        self.name = name
        self.value = value
        self.bar = bar
        self.pos = pos
        self.pitch = pitch
        self.dur = dur
        # self.vel = vel

    def __repr__(self):
        return f'Event(name={self.name:>12s},  value={self.value:>6s}, bar={self.bar:>4d},  pos={self.pos:>6d},  ' \
               f'pitch={self.pitch:>4d}, dur={self.dur:>4d},  vel={self.vel:>4d})\n'

def notenumber2name (pitch, maj=True):
    octave = -2 + pitch // 12
    interval2name_maj = {0:'C', 1:'C#', 2:'D', 3:'D#', 4:'E', 5:'F', 6:'F#', 7:'G', 8:'G#', 9:'A', 10:'A#', 11:'B'}
    interval2name_min = {0:'C', 1:'Db', 2:'D', 3:'Eb', 4:'E', 5:'F', 6:'Gb', 7:'G', 8:'Ab', 9:'A', 10:'Bb', 11:'B'}
    interval = pitch % 12
    
    if maj:
        return f'{interval2name_maj[interval]}_{octave}'
    else:
        return f'{interval2name_min[interval]}_{octave}'
    
def notes2events(notes):
    bar_id = 0
    last_bar = 0
    last_pos = -1
    
    events = []
    
    for note in notes:
        ## Bar
        note_bar = (note.start // (4 * ticks_per_beat)) ## a bar == 4 beat == 4 * 480 ticks
        note_pos = (note.start - (note_bar * 1920)) ## relative position in the current bar
        note_pitch = note.pitch
        note_duration = note.end - note.start   # np.clip(item.end - item.start, 30, 2880).item()
        note_name = note.name
        events.append(Event(
            name = note_name,
            value = notenumber2name(note_pitch),
            bar = note_bar,
            pos = note_pos,
            dur = note_duration
        ))
    
    return events
        
    
def events2words (events, hparams, event2word_dict):
    words = []
    global_bar = 0
    ## padding word
    octupleMIDI_word = {
        'bar': 0,
        'pos': 0,
        'pitch': 0,
        'dur': 0
    }
    
    for e in events:
        if e.name == 'Note':
            words.append({
                'bar': event2word_dict['Bar'][f'Bar_{e.bar}'],
                'pos': event2word_dict['Pos'][f'Pos_{e.pos}'],
                'pitch': event2word_dict['Pitch'][f'Pitch_{e.pitch}'],
                'dur': event2word_dict['Dur'][f'Dur_{e.dur}']
            })
        elif e.name == 'Comma':
            words.append({
                'bar': 0,
                'pos': 0,
                'pitch': 0,
                'dur': 0
            })
            
    return words


def replace_single_quotes(text):
    replacements = {
        "‘": "'",
        "’": "'",
        "ʼ": "'",
        "‛": "'",
        "‚": "'",
        "‘": "'",
        "′": "'",
        "‵": "'",
        "ꞌ": "'",
        "ʹ": "'",
        "ʻ": "'",
    }
    
    for source, target in replacements.items():
        text = text.replace(source, target)
    
    return text


def mid2items(midi_pth, title: str, keywords: list, event2word_dict: dict, lyric2word_dict: dict):
    src_words, tgt_words = [], []
    midi = miditoolkit.MidiFile(midi_pth)
    title_str = title_str.replace('.', '')

    prefix = f"<title>{title_str}"
    encoded_prefix = src_tknzr.encode(prefix)

    ## ----- #src prefix words# ------ ##
    for ep in encoded_prefix:
        if src_tknzr.decode(ep).strip() == '</s>': ## skip the eos
            continue
        src_words.append({
            'sentence': 0,
            'meter': ep,
            'length': 0,
            'remainder': 0,
        })
    ## ----- #tgt prefix words# ------ ##
    tgt_words = [{'sentence': 0,
                  'word': tgt_tknzr.encode('<s>')[1],
                  'syllable': 0,
                  'remainder': 0,
                 }]

    prosody_list = getProsody(midi_pth)
    assert len(prosody_list) == len(midi.instruments[0].notes)
    
    ## group midi by phrase
    group_by_phrase = {}
    start = -1
    for idx in range(len(midi.markers)):
        end = midi.markers[idx].time
        if idx not in group_by_phrase.keys():
            group_by_phrase[idx] = []
        for inst in midi.instruments:
            for nid, note in enumerate(inst.notes):
                if note.start > start and note.start <= end:
                    group_by_phrase[idx].append((nid, note, prosody_list[nid]))
        start = midi.markers[idx].time

    assert len(keywords) == len(group_by_phrase)

    for line_id, line in group_by_phrase.items():
        keyword = keywords[line_id] ## keyword of this line
        prompt = f"<keywords>{keyword_str.strip()}"
        line_syllable_num = len(line)
        # for note in line:
        src_words.append({
            'sentence': 0,
            'meter': src_tknzr.encode(f"<syllable_{line_syllable_num}>")[1],
            'length': 0,
            'remainder': 0,
        })
        src_words.append({
            'sentence': 0,
            'meter': src_tknzr.encode(f"<template>")[1],
            'length': 0,
            'remainder': 0,
        })
        ### src template words
        rem_s = line_syllable_num
        for note in line: ## each note is [nid, midinote, prosody]
            rem_s -= 1  ## decrement
            ## strength symbol
            mtype, length = note[2][0], note[2][1]
            src_words.append({
                'sentence': event2word_dict['Phrase'][f"Phrase_{line_id}"],
                'meter': src_tknzr.encode(mtype)[1],
                'length': event2word_dict['Length'][length],
                'remainder': lyric2word_dict['Remainder'][f"Remain_{rem_s}"],
            })
        assert rem_s == 0

    ## 
    for data_sample in data_samples: ## traverse all sentences
        
        assert len(text.lines()) == 1
        assert len(keywords) == len(group_by_phrase)

        for line_id, line in group_by_phrase.items():
            # print(f"line_{line_id}: {line}")
            # words = line.words()
            # line_syllables = line.syllables()
            # line_syllable_num = len(line_syllables)
            if len(line) > 40:
                # print("Exceed max syllable per line")
                return [], []

            # prompt = f"<syllable_{line_syllable_num}> <title> {title_str.strip()} <keywords> {keyword_str.strip()}"
            prompt = f"<keywords>{keyword_str.strip()}"
            encoded_prompt = src_tknzr.encode(prompt)
            src_words.append({
                'sentence': 0,
                'meter': src_tknzr.encode(f"<syllable_{line_syllable_num}>")[1],
                'length': 0,
                'remainder': 0,
            })
            src_words.append({
                'sentence': 0,
                'meter': src_tknzr.encode(f"<template>")[1],
                'length': 0,
                'remainder': 0,
            })

            ### src template words
            rem_s = line_syllable_num
            for s in line_syllables:
                rem_s -= 1  ## decrement
                ## is accented:
                if "'" in str(s): ## strong
                    mtype = "<strong>"
                elif "`" in str(s):
                    mtype = "<substrong>"
                else:
                    mtype = "<weak>"
                length = "Long" if "ː" in str(s) else "Short"
                src_words.append({
                    'sentence': event2word_dict['Phrase'][f"Phrase_{line_id}"],
                    'meter': src_tknzr.encode(mtype)[1],
                    'length': event2word_dict['Length'][length],
                    'remainder': lyric2word_dict['Remainder'][f"Remain_{rem_s}"],
                })
            assert rem_s == 0

            ### process prompt words:
            for ew in encoded_prompt:
                if ew in src_tknzr.encode('<s>') or ew in src_tknzr.encode('</s>'):
                    continue
                src_words.append({
                    'sentence': 0,
                    'meter': ew,
                    'length': 0,
                    'remainder': 0,
                })
            src_words.append({
                'sentence': 0,
                'meter': src_tknzr.encode('.')[1],
                'length': 0,
                'remainder': 0,
            })
            
            ### --- target words ---
            for idx, word in enumerate(words):                
                if "?" in word.token:
                    # print(f"None detectable token {word.token}")
                    return [], []
                encoded_word = tgt_tknzr.encode(f" {word.token}")
                # assert len(encoded_word) == 1
                syllables = word.syllables()
                num_syllables = len(syllables)
                rem = line_syllable_num - num_syllables
                ### tgt words
                for ew in encoded_word:
                    if ew in tgt_tknzr.encode('<s>') or ew in tgt_tknzr.encode('</s>'):
                        continue
                    if num_syllables == 0:
                        tgt_words.append({'sentence': event2word_dict['Phrase'][f"Phrase_{line_id}"],
                                           'word': ew,
                                           'syllable': 0,
                                           'remainder': lyric2word_dict['Remainder'][f"Remain_{rem}"]})
                    else:
                        tgt_words.append({'sentence': event2word_dict['Phrase'][f"Phrase_{line_id}"],
                                           'word': ew,
                                           'syllable': lyric2word_dict['Syllable'][f"Syllable_{num_syllables}"],
                                           'remainder': lyric2word_dict['Remainder'][f"Remain_{rem}"]})
            ## sep sentence with \n
            tgt_words.append({
                'sentence': 0,
                'word': tgt_tknzr.encode('.')[1],
                'syllable': 0,
                'remainder': 0,
            })
    
    ## ----- end of target sequence -----
    src_words.append({
        'sentence': 0,
        'meter': src_tknzr.encode('</s>')[-1],
        'length': 0,
        'remainder': 0,
    })
    
    tgt_words.append({
        'sentence': 0,
        'word': tgt_tknzr.encode('</s>')[-1],
        'syllable': 0,
        'remainder': 0,
    })
    
    return src_words, tgt_words
    

def lyric2words (lyrics, event2word_dict, lyric2word_dict):
    # import re
    # pattern = r"\([^()]*\)"
    
    data_samples = eval(lyrics)  ## type: list of sentence samples
    src_words, tgt_words = [], []
    # print(data_samples)
    title_str = data_samples[0]['title']  ## title of this song
    title_str = title_str.replace('.', '')
    
    
    prefix = f"<title>{title_str}"
    encoded_prefix = src_tknzr.encode(prefix)
    
    ## ----- #src prefix words# ------ ##
    for ep in encoded_prefix:
        if src_tknzr.decode(ep).strip() == '</s>':
            continue
        src_words.append({
            'sentence': 0,
            'meter': ep,
            'length': 0,
            'remainder': 0,
        })    
    ## ----- #tgt prefix words# ------ ##
    tgt_words = [{'sentence': 0,
                  'word': tgt_tknzr.encode('<s>')[1],
                  'syllable': 0,
                  'remainder': 0,
                 }]

    for line_id, line in enumerate(text.lines()):
        
    for data_sample in data_samples: ## traverse all sentences
        """
        if len(src_words) > 1024 or len(tgt_words) > 1024:
            return [], []
            """
        import string
        translator = str.maketrans('', '', string.punctuation)
        # cleaned_string = re.sub(pattern, "", input_string)
        lyrics_str, title_str, keyword_str = data_sample['sentence'].replace('.', ''), data_sample['title'].translate(translator), data_sample['keyword'].replace('.', '')
        
        ## Process special tokens
        lyrics_str = replace_single_quotes(lyrics_str)
        
        lyrics_str = lyrics_str.replace("cuz", "cause").replace("-", " ")
        
        text = p.Text(lyrics_str.strip())
        
        ## if the seq exceeds max length, return none
        if len(text.words()) > 1024:
            # print(f"{len(text.words())} exceeds max seq length")
            return [], []

        assert len(text.lines()) == 1

        for line_id, line in enumerate(text.lines()):
            # print(f"line_{line_id}: {line}")
            words = line.words()
            line_syllables = line.syllables()
            line_syllable_num = len(line_syllables)
            if line_syllable_num > 40:
                # print("Exceed max syllable per line")
                return [], []

            # prompt = f"<syllable_{line_syllable_num}> <title> {title_str.strip()} <keywords> {keyword_str.strip()}"
            prompt = f"<keywords>{keyword_str.strip()}"
            encoded_prompt = src_tknzr.encode(prompt)
            src_words.append({
                'sentence': 0,
                'meter': src_tknzr.encode(f"<syllable_{line_syllable_num}>")[1],
                'length': 0,
                'remainder': 0,
            })
            src_words.append({
                'sentence': 0,
                'meter': src_tknzr.encode(f"<template>")[1],
                'length': 0,
                'remainder': 0,
            })

            ### src template words
            rem_s = line_syllable_num
            for s in line_syllables:
                rem_s -= 1  ## decrement
                ## is accented:
                if "'" in str(s): ## strong
                    mtype = "<strong>"
                elif "`" in str(s):
                    mtype = "<substrong>"
                else:
                    mtype = "<weak>"
                length = "Long" if "ː" in str(s) else "Short"
                src_words.append({
                    'sentence': event2word_dict['Phrase'][f"Phrase_{line_id}"],
                    'meter': src_tknzr.encode(mtype)[1],
                    'length': event2word_dict['Length'][length],
                    'remainder': lyric2word_dict['Remainder'][f"Remain_{rem_s}"],
                })
            assert rem_s == 0

            ### process prompt words:
            for ew in encoded_prompt:
                if ew in src_tknzr.encode('<s>') or ew in src_tknzr.encode('</s>'):
                    continue
                src_words.append({
                    'sentence': 0,
                    'meter': ew,
                    'length': 0,
                    'remainder': 0,
                })
            src_words.append({
                'sentence': 0,
                'meter': src_tknzr.encode('.')[1],
                'length': 0,
                'remainder': 0,
            })
            
            ### --- target words ---
            for idx, word in enumerate(words):                
                if "?" in word.token:
                    # print(f"None detectable token {word.token}")
                    return [], []
                encoded_word = tgt_tknzr.encode(f" {word.token}")
                # assert len(encoded_word) == 1
                syllables = word.syllables()
                num_syllables = len(syllables)
                rem = line_syllable_num - num_syllables
                ### tgt words
                for ew in encoded_word:
                    if ew in tgt_tknzr.encode('<s>') or ew in tgt_tknzr.encode('</s>'):
                        continue
                    if num_syllables == 0:
                        tgt_words.append({'sentence': event2word_dict['Phrase'][f"Phrase_{line_id}"],
                                           'word': ew,
                                           'syllable': 0,
                                           'remainder': lyric2word_dict['Remainder'][f"Remain_{rem}"]})
                    else:
                        tgt_words.append({'sentence': event2word_dict['Phrase'][f"Phrase_{line_id}"],
                                           'word': ew,
                                           'syllable': lyric2word_dict['Syllable'][f"Syllable_{num_syllables}"],
                                           'remainder': lyric2word_dict['Remainder'][f"Remain_{rem}"]})
            ## sep sentence with \n
            tgt_words.append({
                'sentence': 0,
                'word': tgt_tknzr.encode('.')[1],
                'syllable': 0,
                'remainder': 0,
            })
    
    ## ----- end of target sequence -----
    src_words.append({
        'sentence': 0,
        'meter': src_tknzr.encode('</s>')[-1],
        'length': 0,
        'remainder': 0,
    })
    
    tgt_words.append({
        'sentence': 0,
        'word': tgt_tknzr.encode('</s>')[-1],
        'syllable': 0,
        'remainder': 0,
    })
    
    return src_words, tgt_words

"""
def lyric2words_no_prompt (lyrics, event2word_dict, lyric2word_dict):
    data_samples = eval(lyrics)
    src_words, tgt_words = [], []
    # traverse all sentences
    for data_sample in data_samples:
        lyrics_str, title_str, keyword_str = data_sample['sentence'], data_sample['title'], data_sample['keyword']
        text = p.Text(lyrics_str)
        ## sample src word
        template_word = {
            'meter': 0,
            'length': 0,
            'remainder': 0,
        }
        ## sample tgt word
        syllable_word = {
            'word': 0,
            'remainder': 0,
        }

        if len(text.words()) > 1024:
            return [], []

        assert len(text.lines()) == 1

        for line_id, line in enumerate(text.lines()):
            # print(f"line_{line_id}: {line}")
            words = line.words()
            line_syllables = line.syllables()
            line_syllable_num = len(line_syllables)
            if line_syllable_num > 50:
                return [], []

            prompt = f"<title> {title_str.strip()} <keywords> {keyword_str.strip()} <syllable> {line_syllable_num} <endprompt>"
            encoded_prompt = src_tknzr.encode(prompt)

            ### src words
            rem_s = line_syllable_num
            for s in line_syllables:
                rem_s -= 1  ## decrement
                ## is accented:
                if "'" in str(s): ## strong
                    mtype = "Strong"
                elif "`" in str(s):
                    mtype = "Substrong"
                else:
                    mtype = "Weak"
                length = "Long" if "ː" in str(s) else "Short"
                src_words.append({
                    'sentence': event2word_dict['Phrase'][f"Phrase_{line_id}"],
                    'meter': event2word_dict['Meter'][mtype],
                    'length': event2word_dict['Length'][length],
                    'remainder': lyric2word_dict['Remainder'][f"Remain_{rem_s}"],
                })
            assert rem_s == 0
            
            ## append start of sequence token
            tgt_words = [{'sentence': 0,
                          'word': src_tknzr.encode('<s>')[1],
                          'syllable': 0,
                          'remainder': 0,
            }]

            for idx, word in enumerate(words):                
                if "?" in word.token:
                    # print(f"Err: {word.token}, line: {line}")
                    return [], []
                encoded_word = tokenizer.encode(f" {word.token}")
                # assert len(encoded_word) == 1
                syllables = word.syllables()
                num_syllables = len(syllables)
                rem = line_syllable_num - num_syllables
                ### tgt words
                for ew in encoded_word:
                    if ew in tokenizer.encode('<s>') or ew in tokenizer.encode('</s>'):
                        continue
                    if num_syllables == 0:
                        tgt_words.append({'sentence': event2word_dict['Phrase'][f"Phrase_{line_id}"],
                                           'word': ew,
                                           'syllable': 0,
                                           'remainder': lyric2word_dict['Remainder'][f"Remain_{rem}"]})
                    else:
                        tgt_words.append({'sentence': event2word_dict['Phrase'][f"Phrase_{line_id}"],
                                           'word': ew,
                                           'syllable': lyric2word_dict['Syllable'][f"Syllable_{num_syllables}"],
                                           'remainder': lyric2word_dict['Remainder'][f"Remain_{rem}"]})
            ## sep sentence
            tgt_words.append({
                'sentence': 0,
                'word': tokenizer.encode('</s>')[-1],
                'syllable': 0,
                'remainder': 0,
            })
    
    return src_words, tgt_words
"""  

def data_to_binary (lyric, i, event2word_dict, lyric2word_dict, split, hparams):
    try:
        # word_items, syllable_items = str2items(lyric)
        ## process the source melody
        # note_events = notes2events(note_items)
        # note_words = events2words(note_events, hparams, event2word_dict)
        ## process the target lyrics & syllables
        if len(lyric) < 3:
            return None
        # print(f"use prompt: {hparams['prompt']}")
        # if hparams['prompt']:
        src_words, tgt_words = lyric2words(lyric, event2word_dict, lyric2word_dict)
        cur_max_len = max(len(src_words), len(tgt_words))
        # else:
        #    src_words, tgt_words = lyric2words_no_prompt(lyric, event2word_dict, lyric2word_dict)
        if len(src_words) == 0 or len(tgt_words) == 0 or len(tgt_words) > 1024 or len(src_words) > 1024:
            #print(f"Exceed: {len(tgt_words), len(src_words)}")
            return None
        
        # print(lyric_words, end='\n\n')
        data_sample = {
            'input_path': str(i),
            'item_name': str(i),
            'src_words': src_words,
            'tgt_words': tgt_words,
            'word_length': cur_max_len
        }
        
        return [data_sample]
    
    except Exception as e:
        traceback.print_exc()
        return None
    
"""
def data2binary(dataset_dir, words_dir, split, word2event_dict, event2word_dict, lyric2word_dict, word2lyric_dict):
    # make dir
    save_dir = f'{words_dir}/{split}'
    os.makedirs(save_dir, exist_ok=True)
    
    futures = []
    ds_builder = IndexedDatasetBuilder(save_dir)  # index dataset
    p = mp.Pool(int(os.getenv('N_PROC', 32)))  # 不要开太大，容易内存溢出
    lyric_set_pth = os.path.join(dataset_dir, f'{split}.lyric')
    lyric_set = open(lyric_set_pth, 'r')
    all_lyric = lyric_set.read()
    lyric_set.close()
    lyrics = all_lyric.split("<SEPDATA>")
    start = int(0.0 * len(lyrics))
    end = int(1.0 * len(lyrics))
    lyrics_selected = lyrics[start:end]
    
    for i in range (len(lyrics_selected)):
        futures.append(p.apply_async(data_to_binary, args=[lyrics[i], i, event2word_dict, lyric2word_dict, split, hparams]))
    p.close()

    words_length = []
    all_words = []
    for f in tqdm(futures):
        item = f.get()
        if item is None:
            continue
        for i in range(len(item)):
            sample = item[i]
            words_length.append(sample['word_length'])
            all_words.append(sample)
            ds_builder.add_item(sample) # add item index
            # print(sample['item_name'])

    # save 
    ds_builder.finalize()
    np.save(f'{words_dir}/{split}_words_length.npy', words_length)
    np.save(f'{words_dir}/{split}_words.npy', all_words)
    p.join()
    print(f'| # {split}_tokens: {sum(words_length)}')
    
    return all_words, words_length
"""

def data2binary(dataset_dir, words_dir, split, word2event_dict, event2word_dict, lyric2word_dict, word2lyric_dict):
    # make dir
    save_dir = f'{words_dir}/{split}'
    os.makedirs(save_dir, exist_ok=True)
    
    ds_builder = IndexedDatasetBuilder(save_dir)  # index dataset
    # p = mp.Pool(int(os.getenv('N_PROC', 32)))  # 不要开太大，容易内存溢出
    lyric_set_pth = os.path.join(dataset_dir, f'{split}.lyric')
    lyric_set = open(lyric_set_pth, 'r')
    all_lyric = lyric_set.read()
    lyric_set.close()
    lyrics = all_lyric.split("<SEPDATA>")
    start = int(0.0 * len(lyrics))
    end = int(1.0 * len(lyrics))
    lyrics_selected = lyrics[start:end]
    words_length = []
    all_words = []
    batch_size = 6000
    futures = []

    for batch in range(start, end, batch_size):
        p = mp.Pool(int(os.getenv('N_PROC', 32)))
        futures = []
        if batch + batch_size < end:
            for i in range (batch, batch+batch_size):
                futures.append(p.apply_async(data_to_binary, args=[lyrics[i], i, event2word_dict, lyric2word_dict, split, hparams]))
            p.close()
            for f in tqdm(futures):
                item = f.get()
                if not (item is None):
                    for i in range(len(item)):
                        sample = item[i]
                        words_length.append(sample['word_length'])
                        all_words.append(sample)
                        ds_builder.add_item(sample) # add item index
        else:
            for i in range (batch, end):
                futures.append(p.apply_async(data_to_binary, args=[lyrics[i], i, event2word_dict, lyric2word_dict, split, hparams]))
            p.close()
            for f in tqdm(futures):
                item = f.get()
                if not (item is None):
                    for i in range(len(item)):
                        sample = item[i]
                        words_length.append(sample['word_length'])
                        all_words.append(sample)
                        ds_builder.add_item(sample) # add item index
    p.close()   

    # save 
    ds_builder.finalize()
    np.save(f'{words_dir}/{split}_words_length.npy', words_length)
    np.save(f'{words_dir}/{split}_words.npy', all_words)
    p.join()
    print(f'| # {split}_tokens: {sum(words_length)}')
    
    return all_words, words_length


if __name__ == '__main__':
    print(f"Data Binarisation for Bart Melody2lyric")
    set_hparams()
    
    global src_tknzr, tgt_tknzr
    src_tknzr = BartTokenizer.from_pretrained(hparams['enc_tknzr_dir'])
    tgt_tknzr = BartTokenizer.from_pretrained(hparams['dec_tknzr_dir'])
    assert len(src_tknzr) == 50322
    assert len(tgt_tknzr) <= 50322

    event2word_dict, word2event_dict, lyric2word_dict, word2lyric_dict = pickle.load(open(f"{hparams['binary_data_dir']}/m2l_dict.pkl", 'rb'))

    # create data output dir 
    words_dir = hparams["word_data_dir"]
    if not os.path.exists(words_dir):
        os.makedirs(words_dir)
    else:
        subprocess.check_call(f'rm -rf "{words_dir}"', shell=True)
        os.makedirs(words_dir)
    
    dataset_dir = hparams['dataset_dir']
    train_fns = len(open(os.path.join(dataset_dir, f"train.lyric")).read().split('<SEPDATA>'))
    valid_fns = len(open(os.path.join(dataset_dir, f"valid.lyric")).read().split('<SEPDATA>'))
    
    print(f'train_num = {train_fns}, valid_num = {valid_fns}') 
    
    train_words, train_lengths = data2binary(dataset_dir, words_dir, 'train', word2event_dict, event2word_dict, lyric2word_dict, word2lyric_dict)
    valid_words, valid_lengths = data2binary(dataset_dir, words_dir, 'valid', word2event_dict, event2word_dict, lyric2word_dict, word2lyric_dict)
    
    print(f'usable data: train_num = {len(train_words)}, valid_num = {len(valid_words)}')
    print(f'max data lengths: {max(train_lengths+valid_lengths)}')