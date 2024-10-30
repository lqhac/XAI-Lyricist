#!/usr/bin/env python
# coding: utf-8

# In[1]:


import miditoolkit
import numpy as np
import math
import os, pickle, glob
from tqdm import tqdm
from utils.indexed_datasets import IndexedDatasetBuilder
import multiprocessing as mp
import traceback


# In[2]:


test_sample = "/home/qihao/CS6207/MelodyGLM/MDP/data/processed/wikifonia/7_dedup/wikifonia_1001_seg0_1_Seg1.mid"


# In[3]:

def stress_simple (start, dur, reso=480):
    bar_len = 4 * reso
    pos_in_bar = (start - bar_len * (start // bar_len)) 
    if pos_in_bar== 0:
        return "<strong>"
    else:
        return "<weak>"


# def stress_simple (start, dur, reso=480):
#     bar_len = 4 * reso
#     pos_in_bar = (start - bar_len * (start // bar_len)) 
#     if pos_in_bar== 0:
#         return "<strong>"
#     elif pos_in_bar == reso*2:
#         return "<substrong>"
#     else:
#         return "<weak>"

def stress (start, dur, reso=480):
    if dur in [reso, reso//2, reso//4, reso//8, reso//16, reso*2, reso*4]:
        ## categorise the note duration
        unit_len = 4 * dur
    else:
        unit_len = 4 * reso
    beat_pos = start - (start//unit_len) * unit_len
    beat_num = beat_pos // (unit_len//4)
    # beat_num = start % unit_len
    # print(f"dur:{dur}, pos:{beat_pos}, beat:{beat_num}, unit:{unit_len}")
    if beat_num == 0:
        return "<strong>"
    elif beat_num == 2:
        return "<strong>"
    else:
        return "<weak>"

# In[4]:


def getProsody (midi_pth: str):
    prosody = []
    
    midi = miditoolkit.MidiFile(midi_pth)
    ## group by bar:
    bar = {}
    ## calculate average note length
    note_durs = []
    strength, length = [], []
    reso = midi.ticks_per_beat
    for inst in midi.instruments:
        for i, note in enumerate(inst.notes):
            bar_num = np.floor(note.start/(8*reso))
            if bar_num not in bar.keys():
                bar[bar_num] = []
            bar[bar_num].append(note)
            dur = note.end - note.start
            note_durs.append(dur)
    ## calculate global average
    avg_dur = np.mean(note_durs)
    ## calculate bar level average
    bar_avg = {}
    for bar_num, bar_notes in bar.items():
        bar_avg[bar_num] = np.mean([note.end-note.start for note in bar_notes])
    for bar_num, bar_notes in bar.items():
        for note in bar_notes:
            strength = stress(start=note.start, dur=note.end-note.start, reso=reso)
            length = "Long" if note.end-note.start>bar_avg[bar_num] else "Short"
            # prosody.append(f"<{strength}, {length}>")
            prosody.append((strength, length))
    
    return prosody


def getProsody_simple (midi_pth: str):
    ## use absolute value
    prosody = []
    
    midi = miditoolkit.MidiFile(midi_pth)
    ## group by bar:
    bar = {}
    ## calculate average note length
    note_durs = []
    strength, length = [], []
    reso = midi.ticks_per_beat
    for inst in midi.instruments:
        for i, note in enumerate(inst.notes):
            strength = stress(start=note.start, dur=note.end-note.start, reso=reso)
            length = "Long" if note.end-note.start>reso else "Short"
            prosody.append((strength, length))
    # print(prosody)
    return prosody


# In[6]:


def phrasing (midi_pth: str):
    midi = miditoolkit.MidiFile(midi_pth)
    assert len(midi.instruments) == 1  ## monophonic
    reso = midi.ticks_per_beat
    notes = midi.instruments[0].notes.copy()
    
    long = []
    pause = []
    note_info = []
    
    for idx, note in enumerate(notes):
        note_bar = int(np.floor(note.start / (4 * reso))) ## a bar == 4 beat == 4 * 480 ticks
        note_pos = (note.start - (note_bar * 4 * reso)) ## relative position in the current bar
        note_pitch = note.pitch
        note_dur = note.end - note.start
        note_info.append((note_bar, note_pos, note_pitch, note_dur))
        if note_dur > reso:
            long.append(idx)
        if (idx > 0) and (notes[idx].start-notes[idx-1].end >= reso//2):
            pause.append(idx-1)
    
    union = list(set(long + pause))
    if 0 in union:
        union.remove(0)
    if len(notes)-1 in union:
        union.remove(len(notes)-1)
    union.sort()
    
    def dur(note: miditoolkit.Note):
        return abs(note.end-note.start)
    
    i = 1
    while i<len(union):
        if abs(union[i-1]-union[i]) == 1:
            if abs(dur(notes[union[i-1]])-dur(notes[union[i]])) > 240:
                union.remove(union[i])
            else:
                union.remove(union[i-1])
        i = i + 1
    
    ### annotate
    midi.markers=[]
    for k, b in enumerate(union):
        midi.markers.append(miditoolkit.Marker(time=notes[b].end, text=f"Phrase_{k}"))
    
    # midi.dump(os.path.join('./', os.path.basename(midi_pth)[:-4]+'_phrased.mid'))
    
    is_boundary = []
    for i in range(len(notes)):
        if i in union:
            is_boundary.append("<true>")
        else:
            is_boundary.append("<false>")
    
    assert len(note_info) == len(is_boundary)
    return is_boundary, note_info, union


# In[7]:


def get_notes (midi_pth: str):
    midi = miditoolkit.MidiFile(midi_pth)
    assert len(midi.instruments) == 1  ## monophonic
    reso = midi.ticks_per_beat
    notes = midi.instruments[0].notes.copy()
    
    note_info = []
    
    for note in notes:
        note_info.append()


# In[9]:


def tokenise (midi_pth, event2word_dict):
    prsd = prosody(midi_pth)
    bound, notes, _ = phrasing(midi_pth)
    assert len(prsd) == len(bound)
    src_words, tgt_words = [], []
    
    ## bos
    tgt_words.append({
        'bar':event2word_dict['Bar'][f"<s>"],
        'pos':event2word_dict['Pos'][f"<s>"],
        'token':event2word_dict['Pitch'][f"<s>"],
        'dur':event2word_dict['Dur'][f"<s>"],
        'phrase':event2word_dict['Phrase'][f"<s>"],
    })
    
    for idx in range(len(prsd)):
        src_words.append({
            'strength':event2word_dict['Strength'][prsd[idx][0]],
            'length':event2word_dict['Length'][prsd[idx][1]],
            'phrase':event2word_dict['Phrase'][bound[idx]],
        })
        tgt_words.append({
            'bar':event2word_dict['Bar'][f"Bar_{notes[idx][0]}"],
            'pos':event2word_dict['Pos'][f"Pos_{notes[idx][1]}"],
            'token':event2word_dict['Pitch'][f"Pitch_{notes[idx][2]}"],
            'dur':event2word_dict['Dur'][f"Dur_{notes[idx][3]}"],
            'phrase':event2word_dict['Phrase'][bound[idx]],
        })
    
        ## eos
    tgt_words.append({
        'bar':event2word_dict['Bar'][f"</s>"],
        'pos':event2word_dict['Pos'][f"</s>"],
        'token':event2word_dict['Pitch'][f"</s>"],
        'dur':event2word_dict['Dur'][f"</s>"],
        'phrase':event2word_dict['Phrase'][f"</s>"],
    })
    
    return src_words, tgt_words


# In[10]:


def data_to_binary (midi_pth, i, event2word_dict, split):
    try:
        src_words, tgt_words = tokenise(midi_pth, event2word_dict)
        if len(src_words) == 0 or len(tgt_words) == 0 or len(tgt_words) > 1024:
            return None
        
        data_sample = {
            'input_path': midi_pth,
            'item_name': os.path.basename(midi_pth),
            'src_words': src_words,
            'tgt_words': tgt_words,
            'word_length': len(tgt_words)
        }
        
        return [data_sample]
    
    except Exception as e:
        traceback.print_exc()
        return None


# In[11]:


def data2binary(dataset_dirs, words_dir, split, word2event_dict, event2word_dict):
    # make dir
    save_dir = f'{words_dir}/{split}'
    os.makedirs(save_dir, exist_ok=True)
    
    midi_files = []
    for dataset_dir in dataset_dirs:
        midi_files.extend(glob.glob(os.path.join(os.path.join(dataset_dir, split), "*.mid")))
    
    futures = []
    ds_builder = IndexedDatasetBuilder(save_dir)  # index dataset
    p = mp.Pool(int(os.getenv('N_PROC', 2)))  # 不要开太大，容易内存溢出
    
    for i in range (len(midi_files)):
        futures.append(p.apply_async(data_to_binary, args=[midi_files[i], i, event2word_dict, split]))
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

    # save 
    ds_builder.finalize()
    np.save(f'{words_dir}/{split}_words_length.npy', words_length)
    np.save(f'{words_dir}/{split}_words.npy', all_words)
    p.join()
    print(f'| # {split}_tokens: {sum(words_length)}')
    
    return all_words, words_length


# In[12]:


## shuffle and split dataset
def split_data(output_dir='/home/qihao/CS6207/dataset'):
    dataset_dirs = [
        '/home/qihao/CS6207/MelodyGLM/MDP/data/processed/pop909/7_dedup',
        '/home/qihao/CS6207/MelodyGLM/MDP/data/processed/wikifonia/7_dedup',
        '/home/qihao/CS6207/MelodyGLM/MDP/data/processed/mtc/7_dedup',
        '/home/qihao/CS6207/MelodyGLM/MDP/data/processed/sessions/7_dedup'
    ]
    all_files = []
    for dataset_dir in dataset_dirs:
        all_files.extend(glob.glob(os.path.join(dataset_dir, '*.mid')))
    ## shuffle
    print(f"|>>> Total Files: {len(all_files)}")
    
    indices = [i for i in range(len(all_files))]
    import random, shutil
    random.shuffle(indices)
    train_end = int(np.floor(0.8*len(all_files)))
    valid_end = int(train_end + np.floor(0.1*len(all_files)))
    train_idx = indices[:train_end]
    valid_idx = indices[train_end:valid_end]
    test_idx = indices[valid_end:]
    assert len(all_files) == len(train_idx)+len(valid_idx)+len(test_idx)
    print(f"|>>>>> Train Files: {len(train_idx)}")
    print(f"|>>>>> Valid Files: {len(valid_idx)}")
    print(f"|>>>>> Test Files: {len(test_idx)}")
    
    for split in ['train', 'test', 'valid']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    for t in train_idx:
        shutil.copy(all_files[t], os.path.join(f'{output_dir}/train', os.path.basename(all_files[t])))
    for v in valid_idx:
        shutil.copy(all_files[v], os.path.join(f'{output_dir}/valid', os.path.basename(all_files[v])))
    for t in test_idx:
        shutil.copy(all_files[t], os.path.join(f'{output_dir}/test', os.path.basename(all_files[t])))



from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note
def write(words, output_dir, midi_name, word2event):
    notes_all = []
    markers = []
    bar_cnt = -1
    positions = 0
    midi_obj = miditoolkit.midi.parser.MidiFile()
    event_type_list = []
    notes_all = []

    for event in words:
        bar_id, pos_id, pitch_id, dur_id, phrase_id = event[0], event[1], event[2], event[3], event[4]
        
        bar = word2event['Bar'][bar_id]
        pos = word2event['Pos'][pos_id]
        pitch = word2event['Pitch'][pitch_id]
        dur = word2event['Dur'][dur_id]
        phrase = word2event['Phrase'][phrase_id]
        
        # print(f"{bar}, {pos}, {pitch}, {dur}, {phrase}")
        
        if ("Bar_" not in bar) or ("Pos_" not in pos) or ("Pitch_" not in pitch) or ("Dur_" not in dur) or (("<true>" not in phrase) and ("<false>" not in phrase)):
            continue
        bar_num = int(bar.split('_')[1])
        pos_num = int(pos.split('_')[1])
        pitch_num = int(pitch.split('_')[1])
        dur_num = int(dur.split('_')[1])
        phrase_bool = True if phrase == '<true>' else False
        
        start = bar_num*1920 + pos_num
        end = start + dur_num
        notes_all.append(
            Note(pitch=pitch_num, start=start, end=end, velocity=80)
        )
        if phrase_bool:
            markers.append(Marker(time=start, text='Phrase'))
        
    # tempo
    midi_obj.tempo_changes.append(
                TempoChange(tempo=65, time=0))

    # marker
    midi_obj.markers.extend(markers)

    # track
    piano_track = Instrument(0, is_drum=False, name='melody')

    # notes
    piano_track.notes = notes_all
    midi_obj.instruments = [piano_track]

    # save
    tgt_melody_pth = os.path.join(output_dir, f"{midi_name.strip()}.mid")
    
    midi_obj.dump(tgt_melody_pth)

    return output_dir


# In[152]:


def debinarise (data_item, word2event_dict):
    out_words = []
    for tgt_word in data_item['tgt_words']:
        out_words.append((
            tgt_word['bar'], tgt_word['pos'], tgt_word['token'], tgt_word['dur'], tgt_word['phrase']
        ))
    write(words=out_words, 
          output_dir='/home/qihao/CS6207/debinarise', 
          midi_name=data_item['item_name'], 
          word2event=word2event_dict)