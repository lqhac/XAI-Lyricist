import os
import pretty_midi as pm
import miditoolkit
from glob import glob
from itertools import chain
from tqdm import tqdm
import traceback
import numpy as np
import pickle
import miditoolkit
import pretty_midi
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import track_separate as tc
# ----------------------------------------------
# Step 01: 过滤主旋律44拍
# ----------------------------------------------
def filter_melody_44_job(src_path, dst_dir):
    midi = miditoolkit.MidiFile(src_path)
    ts = midi.time_signature_changes
    # 只选择使用4/4拍的音乐
    if len(ts) > 0:
        flag_44 = True
        for ts_item in ts:
            if ts_item.numerator == 4 and ts_item.denominator == 4:
                continue
            else:
                flag_44 = False
    else:  # 设置为4/4拍
        midi.time_signature_changes.append(miditoolkit.TimeSignature(numerator=4, denominator=4, time=0))
    if flag_44:
        ins = midi.instruments
        for ins_idx, item in enumerate(ins):
            if item.name == 'Lead':
                new_midi = miditoolkit.MidiFile(src_path)
                new_midi.instruments.clear()
                new_midi.instruments.append(item)
                new_midi_path = os.path.join(dst_dir, os.path.basename(src_path))
                new_midi.dump(new_midi_path)
                return new_midi_path

def filter_melody_44_Pop909_job(src_path, dst_dir):
    midi = miditoolkit.MidiFile(src_path)
    ts = midi.time_signature_changes
    # 只选择使用4/4拍的音乐
    if len(ts) > 0:
        flag_44 = True
        for ts_item in ts:
            if ts_item.numerator == 4 and ts_item.denominator == 4:
                continue
            else:
                flag_44 = False
    else:  # 设置为4/4拍
        midi.time_signature_changes.append(miditoolkit.TimeSignature(numerator=4, denominator=4, time=0))
    if flag_44:
        ins = midi.instruments
        for ins_idx, item in enumerate(ins):
            if item.name == 'MELODY':
                new_midi = miditoolkit.MidiFile(src_path)
                new_midi.instruments.clear()
                item.name = 'Lead'  # rename lead melody name to 'Lead'
                new_midi.instruments.append(item)
                new_midi_path = os.path.join(dst_dir, os.path.basename(src_path))
                new_midi.dump(new_midi_path)
                return new_midi_path


# ----------------------------------------------
# Step 02: 量化 ｜ 二等音+三连音
# ----------------------------------------------
grids_triple = 32
grids_normal = 64
default_resolution = 480

# 按小节进行存放，同时删掉时值短于64分音符的note，截断时值长于5拍的音符note
def split_by_bar_SingleTrack(notes, resolution, bar_ticks):
    res_dict = dict()
    for note in notes:
        length = note.end-note.start
        if length >= (resolution/16): # 最小ticks为30，而三连音最小为40ticks，可覆盖
            # clip end
            if length > resolution * 5:
                note.end = note.start + resolution * 5
            key = int(note.start // bar_ticks)  # 按小节进行保存
            if key not in res_dict:
                res_dict[key] = []
            res_dict[key].append(note)
    return res_dict

def divide_notetypes(notes_dict, base_std, len_std_normal, note_name_normal, len_std, acc_interval=0.05, acc_duration=0.17, acc_double=0.2):
    # acc_interval: 默认为0.05 控制间距误差 即相邻音符在acc_interval邻域内满足长度相等
    # acc_duration: 默认为0.1 控制时长误差 即三个音符的和在acc_duration的邻域内满足长度等于标准音符长度
    # acc_double:   默认为0.2 用于测试音符长度的二倍关系 即两个音符的长度比值在acc_double的邻域内满足等于2

    triole_dict = dict()      # 三连音词典
    triole_cnt = 0            # 三连音个数统计
    note_dict = dict()        # 2等分音
    note_cnt = 0              # 2等分音个数统计

    for k, v in notes_dict.items(): # k,小节； v，小节中的音符
        i = 0
        triole_dict[k] = []
        note_dict[k] = []

        while i < len(v): # 每个小节中音符的个数
            cand_note = 0  # 非三连音标记
            # ------三连音------
            # 预判方式一: 通过音符时长找到最可能的三连音类型 ｜ 相邻三个音符时长相同
            for note, length in base_std.items():
                if i + 2 < len(v):  # 防止越界
                    if (abs(v[i + 2].end - v[i].start - length) <= acc_duration * length) or \
                            abs(v[i + 1].end - v[i].start - length) <= acc_duration * length:  # 全音长度小于误差精度
                        cand_note = note  # 可能是三连音
                        break  # 找到一定是最接近的
                elif i + 1 < len(v) and not (i + 2 < len(v)):
                    if abs(v[i + 1].end - v[i].start - length) <= acc_duration * length:  # 全音长度小于误差精度
                        cand_note = note  # 只可能是两个一组三连音
                        break  # 找到一定是最接近的
                else:  # 之后不可能出现三连音
                    break
                    # 如果找不到接近的音符 那么不可能是三连音 直接跳转到下一个音符
            if cand_note == 0:
                dur = v[i].end - v[i].start  # 音符时长
                delta = [abs(dur - len_std_normal[i]) for i in range(len(len_std_normal))]
                note = note_name_normal[delta.index(min(delta))]
                note_dict[k].append({note: v[i]}) # 音符类型：音符
                i += 1
                continue

            dur_a = v[i].end - v[i].start  # vi的音符长度
            dur_b = v[i + 1].end - v[i + 1].start  # vi+1的音符长度

            # 判断音符间隔是否满足精度条件
            if (i + 2 < len(v)) and ((v[i + 1].start - v[i].start) not in len_std) and \
                    abs((v[i + 1].start - v[i].start) - (v[i + 2].start - v[i + 1].start)) <= acc_interval * \
                    base_std[cand_note]:
                triole_dict[k].append({cand_note: [v[i], v[i + 1], v[i + 2]]})
                triole_cnt += 1
                i += 3  # 后移3个音符
            elif (i + 1 < len(v)) and ((v[i + 1].start - v[i].start) not in len_std) and \
                    ((dur_b / dur_a - 2 >= -acc_double and dur_b / dur_a - 2 <= acc_double) or \
                     (dur_a / dur_b - 2 >= -acc_double and dur_a / dur_b - 2 <= acc_double)):  # 长度为近似2倍关系
                triole_dict[k].append({cand_note: [v[i], v[i + 1]]})
                triole_cnt += 1
                i += 2  # 后移2个音符
            else:  # 不是三连音
                dur = v[i].end - v[i].start  # 音符时长
                delta = [abs(dur - len_std_normal[i]) for i in range(len(len_std_normal))]
                note = note_name_normal[delta.index(min(delta))]
                note_dict[k].append({note: v[i]})  # 音符类型：音符
                i += 1

    return triole_dict, triole_cnt, note_dict, note_cnt


def get_std_grid (value):
    # 三连音的网格 grids细分度为16
    step = default_resolution * 4 / grids_triple
    std_left = int(value // step * step)
    std_right = int(std_left + step)
    return std_left if abs(value-std_left) <= abs(value-std_right) else std_right

def get_std_grid_normal (value):
    # 二等分音的网格 grids细分度为64
    step_normal = default_resolution * 4 / grids_normal
    std_left = int(value // step_normal * step_normal)
    std_right = int(std_left + step_normal)
    return std_left if abs(value-std_left) <= abs(value-std_right) else std_right


def get_std_grid_coarse (value):
    step_coarse = 480 * 4 // 16
    std_left = int(value // step_coarse * step_coarse)
    std_right = int(std_left + step_coarse)
    return std_left if abs(value - std_left) <= abs(value - std_right) else std_right

def quantise_refine (note, trioles):
    start = get_std_grid(trioles[0].start)
    acc_delta = 0.1
    interval = 480 * 4 / note  # 每个四等分网格之间的间隔为480ticks
    delta_grids = start % (interval * note) - np.array([interval * i for i in range(note)])
    if min(delta_grids) < acc_delta * (interval / 3):
        trioles[0].start = start // (interval * note) * (interval * note) + interval * np.argmin(delta_grids)
        trioles[0].end = trioles[0].start + base_std[note] // 3
    else:  # 按照二等分音的方式量化
        for subnote in trioles:
            subnote.start = get_std_grid_coarse(subnote.start)
            subnote.end = get_std_grid_coarse(subnote.end)

# ----- 对三连音量化 -----
def _quant_triole(triole_set: dict, base_std, step, acc_interval, acc_duration, acc_double):
    triole_dict = triole_set.copy() # 防止实参被修改
    # 量化三连音
    for bar, tritems in triole_dict.items():
        for tritem in tritems:
            for note, trioles in tritem.items():
                start = get_std_grid(trioles[0].start)
                acc_delta = 0.05
                interval = 480 * 4 / note  # 每个四等分网格之间的间隔
                delta_grids = abs(start % (interval * note) - np.array([interval * i for i in range(note)]))
                # print("delta_grids:", delta_grids)
                # print(min(delta_grids), acc_delta * (interval / 3))
                if min(delta_grids) < acc_delta * (interval / 3): # 按照三连音方式量化
                    trioles[0].start = start // (interval * note) * (interval * note) + interval * np.argmin(
                        delta_grids)
                    if len(trioles) == 3: # 量化剩余两个音符
                        trioles[0].end = trioles[0].start + base_std[note] // 3
                        trioles[1].start = trioles[0].end
                        trioles[1].end = trioles[1].start + base_std[note] // 3
                        trioles[2].start = trioles[1].end
                        trioles[2].end = trioles[2].start + base_std[note] // 3
                    elif len(trioles) == 2: # 量化剩余一个音符
                        duration_1 = trioles[0].end - trioles[0].start
                        duration_2 = trioles[1].end - trioles[1].start
                        time_lag_1 = int(base_std[note] // 3 * 2) if duration_1 > duration_2 else int(
                            base_std[note] // 3)
                        time_lag_2 = base_std[note] - time_lag_1
                        trioles[0].end = trioles[0].start + time_lag_1
                        trioles[1].start = get_std_grid(trioles[1].start)
                        trioles[1].end = trioles[1].start + time_lag_2
                else:  # 按照二等分音的方式量化
                    for subnote in trioles:
                        if subnote.end - subnote.start >= 110:
                            subnote.start = get_std_grid_coarse(subnote.start)
                            subnote.end = get_std_grid_coarse(subnote.end)
                        else:
                            subnote.start = get_std_grid(subnote.start)
                            subnote.end = get_std_grid(subnote.end)
    return triole_dict

# ----- 对二分音量化 -----
def _quant_std_notes(note_set: dict,step_normal,base_std_normal, acc_duration=0.1):
    note_dict = note_set.copy()
    for bar, notelist in note_dict.items():
        for noteitem in notelist:
            for notename, note in noteitem.items():
                # 16分音符以上
                dur = note.end - note.start
                if dur >=120:
                    std1 = int(note.start // 120 * 120)  # 左标准网格
                    std2 = int(std1 + 120)
                    note.start = std1 if abs(note.start - std1) <= abs(note.start - std2) else std2
                    note.end = note.start + base_std_normal[notename]
                else:
                    std1 = int(note.start // step_normal * step_normal)  # 左标准网格
                    std2 = int(std1 + step_normal)
                    note.start = std1 if abs(note.start - std1) <= abs(note.start - std2) else std2
                    note.end = note.start + base_std_normal[notename]
    return note_dict


# 按照两种不同的方式量化
def quant_notetypes(triole_dict, note_dict,base_std,step, step_normal,base_std_normal,acc_interval=0.1, acc_duration=0.17, acc_double=0.2):
    quant_triole_dict = _quant_triole(triole_dict, base_std, step, acc_interval, acc_duration, acc_double)
    quant_std_notedict = _quant_std_notes(note_dict, step_normal,base_std_normal,acc_duration)
    return quant_triole_dict, quant_std_notedict


# 归并两类音符 按照时间排序
def merge_and_sort(quant_triole, quant_note):
    note_list = []
    # 三连音
    for bar, tritems in quant_triole.items():
        for tritem in tritems:
            for note, trioles in tritem.items():
                for subnote in trioles:
                    note_list.append(subnote)
    # 2^n分音符
    for bar, notelist in quant_note.items():
        for noteitem in notelist:
            for notename, note in noteitem.items():
                note_list.append(note)
    # 按照start排序
    sorted_notes = sorted(note_list, key=lambda x:x.start)
    return sorted_notes


# 量化音符 | Quantising the start and ending of the noes
# We use ticks to store a MIDI note's position, it makes the measurement in an absolute value.
def quantise_midi_job(midi_file, save_fn, dest_dir):
    try:
        ###############
        # 操作一：量化时间，包括音符的开始i和结束，节奏，调号，Marker等，其中
        # 音符量化规则为：All notes shorter than a 64th note are discarded and those longer than a half note are clipped.
        ###############
        mf = miditoolkit.MidiFile(midi_file)
        # 1) parameters:
        max_ticks = mf.max_tick
        resolution = mf.ticks_per_beat
        step = resolution * 4 / grids_triple
        step_normal = resolution * 4 / grids_normal
        bar_ticks = resolution * 4
        # 三连音最小单位设置为32分音符
        note_name = [2 ** i for i in range(5)]  # [1, 2, 4, 8, 16, 32], ** 代表乘方
        len_std = [resolution * 4 // s for s in note_name]  # 全音符、二分、四分、八分、十六分、32分音符的标准音长 [1920, 960, 480, 240, 120]
        base_std = dict(zip(note_name, len_std))  # {音符名称:音长}字典
        # 2等音符最小单位设置为64分音符
        note_name_normal = [2 ** i for i in range(7)]  # [1, 2, 4, 8, 16, 32], ** 代表乘方
        len_std_normal = [resolution * 4 // s for s in
                          note_name_normal]  # 全音符、二分、四分、八分、十六分、32分音符的标准音长 [1920, 960, 480, 240, 120, 60]
        base_std_normal = dict(zip(note_name_normal, len_std_normal))  # {音符名称:音长}字典


        # 2) set
        mf.ticks_per_beat =  resolution  # Resolution, 480，每个四分音符有480个Tick. (default)

        # ----- Marker & Tempo & Time Signature 量化
        grids = np.arange(0, max_ticks, resolution / 16, dtype=int)  # 对准拍子 1/64
        for tempo in mf.tempo_changes:
            index_tempo = np.argmin(abs(grids - tempo.time))
            tempo.time = int((resolution / 16) * index_tempo)

        for ks in mf.key_signature_changes:
            index_ks = np.argmin(abs(grids - ks.time))
            ks.time = int((resolution / 16) * index_ks)

        for marker in mf.markers:
            index_marker = np.argmin(abs(grids - marker.time))
            marker.time = int((resolution / 16) * index_marker)

        # ----- 音符量化 -----
        for ins_idx, ins in enumerate(mf.instruments): # 遍历每个轨道
            # 1. 音符按小节进行存放
            notes_dict = split_by_bar_SingleTrack(mf.instruments[ins_idx].notes, resolution,bar_ticks)
            # 2. 音符分类
            triole_dict, triole_cnt, note_dict, note_cnt = divide_notetypes(notes_dict,base_std, len_std_normal, note_name_normal, len_std )
            # 3. 按分类量化音符
            quant_triole_dict, quant_note_dict = quant_notetypes(triole_dict, note_dict, base_std,step, step_normal,base_std_normal)
            # 4. 合并
            sorted_notes = merge_and_sort(quant_triole_dict, quant_note_dict)
            for note in sorted_notes:
                note.start = int(note.start)
                note.end = int(note.end)
            # 5. 重新写入
            mf.instruments[ins_idx].notes.clear()
            # print(ins.notes)
            mf.instruments[ins_idx].notes.extend(sorted_notes)
            # print(ins.notes)

        # save
        midi_fn = f'{dest_dir}/{save_fn}'
        mf.dump(midi_fn)
        return midi_fn

    except Exception as e:
        # traceback.print_exc()
        print(f"| load data error ({type(e)}: {e}): ", midi_file)
        print(traceback.print_exc())
        return None

# ----------------------------------------------
# Step 03: 主旋律净化
# ----------------------------------------------
def segment_clean_job(midi_fn, dst):
    midi = miditoolkit.MidiFile(midi_fn)
    try:
        notes = midi.instruments[0].notes
    except:
        return None
    notes.sort(key=lambda x: (x.start, x.pitch))
    notes_dict = {}
    for note_idx, note in enumerate(notes):
        if note.start not in notes_dict.keys():
            notes_dict[note.start] = [note]
        else:
            notes_dict[note.start].append(note)

    note_list = []
    for key, values in notes_dict.items():
        if len(values) == 1:
            note_list.append(values[0])
        if len(values) > 1:
            values.sort(key=lambda x: (x.pitch))

            note_list.append(values[-1])  # pick max pitch
    note_list.sort(key=lambda x: (x.start))

    # save
    new_midi = miditoolkit.MidiFile(midi_fn)
    new_midi.instruments[0].notes.clear()
    new_midi.instruments[0].notes.extend(note_list)
    new_midi.dump(f'{dst}/{os.path.basename(midi_fn)}')
    return f'{dst}/{os.path.basename(midi_fn)}'

# ----------------------------------------------
# Step 04: 主旋律对齐
# ----------------------------------------------
def melody_mono_job(midi_path, dst):
    midi = miditoolkit.MidiFile(midi_path)
    notes = midi.instruments[0].notes.copy()
    notes.sort(key=lambda x: (x.start, (-x.end)))

    # 修理音符的开始和结束
    for idx, note in enumerate(notes):
        if idx == 0:
            continue
        else:
            if note.start < notes[idx - 1].end:
                notes[idx - 1].end = note.start

    # save
    midi.instruments[0].notes.clear()
    midi.instruments[0].notes.extend(notes)
    midi_fn = f'{dst}/{os.path.basename(midi_path)}'
    midi.dump(midi_fn)
    return f"save midi in {midi_fn}"

# ----------------------------------------------
# Step 04: 主旋律切割
# ----------------------------------------------
def segment_melody_job(midi_path, dst):
    segment_idx = 0
    midi = miditoolkit.MidiFile(midi_path)
    notes_list = []
    save_list = []
    start_bar_id = 0
    ins_notes = midi.instruments[0].notes
    last_note_start = 0
    for note_idx, note in enumerate(ins_notes):
        if note_idx == 0:
            start_bar_id = int((note.start) / (480 * 4))
            if start_bar_id + 1 > 4:
                last_note_start = note.start
                note.start = note.start - (start_bar_id * (480 * 4))
                note.end = note.end - (start_bar_id * (480 * 4))
                notes_list.append(note)
            else:
                notes_list.append(note)
        else:
            bar_id = int((note.start) / (480 * 4))
            last_note_bar_id = int(last_note_start / (480 * 4))
            if (bar_id - last_note_bar_id) <= 4:
                last_note_start = note.start
                note.start = note.start - (start_bar_id * (480 * 4))
                note.end = note.end - (start_bar_id * (480 * 4))
                notes_list.append(note)
            else:
                last_note_start = note.start
                # save
                new_midi = miditoolkit.MidiFile(midi_path)
                new_midi.instruments.clear()
                new_ins = miditoolkit.Instrument(name='Lead', is_drum=False, program=0)
                new_ins.notes.extend(notes_list)
                new_midi.instruments.append(new_ins)
                segment_idx += 1
                new_midi.dump(f'{dst}/{os.path.basename(midi_path)[:-4]}_{segment_idx}.mid')
                save_list.append(f'{dst}/{os.path.basename(midi_path)[:-4]}_{segment_idx}.mid')

                # reload note
                notes_list.clear()
                start_bar_id = int((note.start) / (480 * 4))
                note.start = note.start - (start_bar_id * (480 * 4))
                note.end = note.end - (start_bar_id * (480 * 4))
                notes_list.append(note)

    if len(notes_list) != 0:
        new_midi = miditoolkit.MidiFile(midi_path)
        new_midi.instruments.clear()
        new_ins = miditoolkit.Instrument(name='Lead', is_drum=False, program=0)
        new_ins.notes.extend(notes_list)
        new_midi.instruments.append(new_ins)
        segment_idx += 1
        new_midi.dump(f'{dst}/{os.path.basename(midi_path)[:-4]}_{segment_idx}.mid')
        save_list.append(f'{dst}/{os.path.basename(midi_path)[:-4]}_{segment_idx}.mid')
    return save_list

# Step06
class SkeletonExtractor:
    def __init__(self, miditoolkit_midi, resolution=480, grids=16):
        self.midi = miditoolkit_midi
        self.resolution = resolution  # 默认为 四分音符 480ticks,一拍
        self.grids = grids  # 十六分音符，120ticks
        self.step = resolution * 4 / grids
        self.bar_ticks = resolution * 4
        self.subsections = self._divide_subsections()

    def _divide_subsections(self): # # 按小节进行保存音符
        notes = self.midi.instruments[0].notes
        res_dict = dict()
        for note in notes:
            start = note.start
            key = int(start // self.bar_ticks)
            if key not in res_dict:
                res_dict[key] = []
            res_dict[key].append(note)
        return res_dict

    # ----------
    # 类型一：节拍重音
    # ----------
    def _get_stress(self):
        heavy_dict = dict()
        for bar_id, bar_notes in self.subsections.items():  # k = bar
            start = self.bar_ticks * (bar_id)  # [0,2],[2,4],[4,6]...
            # 设置重音所在拍位： 第一拍 or 第三拍
            first_beat_position = start  #
            third_beat_postion = start + 8 * self.step
            if bar_id not in heavy_dict:
                heavy_dict[bar_id] = []
            for note in bar_notes:
                # 第一拍或第三拍
                if (note.start == first_beat_position) or (note.start == third_beat_postion):
                    heavy_dict[bar_id].append(note)
        return heavy_dict

    # ------------------------------
    # 类型二：长音
    # 若长音有多个相同时值的音符，
    # 1. 若在1、3逻辑重音上，则同时选取
    # ------------------------------
    def _get_long(self):
        long_dict = dict()
        for bar_id, bar_notes in self.subsections.items():
            # 1. 创建长音字典
            if bar_id not in long_dict:
                long_dict[bar_id] = []

            # 2. 获取小节中时值最长的1个或多个音符索引（即存在多个时值最长且相同的音符）
            duration_list = [x.end - x.start for x in bar_notes]
            max_duration = max(duration_list)
            tup = [(i, duration_list[i]) for i in range(len(duration_list))]
            idx_list = [i for i, n in tup if n == max_duration]  # 相同时值长音的索引列表

            # 3. 判断长音是否在逻辑重音的位置上：
            first_beat_position = self.bar_ticks * (bar_id)
            third_beat_position = first_beat_position + 8 * self.step
            if len(idx_list) == 1:  # 当只有一个长音的时候，直接判定为长音
                long_dict[bar_id].append(bar_notes[idx_list[0]])
            elif len(idx_list) > 1:  # 当存在多个长音的时候，只有音符开始的位置在节拍重音上才会被认定为是长音，否则都不是。这些都不是音符会在切分音中被处理到。
                for i in range(len(idx_list)):
                    long_temp = bar_notes[idx_list[i]]
                    if (long_temp.start == first_beat_position) or (long_temp.start == third_beat_position):
                        long_dict[bar_id].append(bar_notes[idx_list[i]])
        return long_dict

    # -------------------------------
    # 类型三：切分音
    # 操作：1）过滤小于16分音符时值的音符，无切分音意义；2）根据4分音符，8分音符和16分音符的所有切分音情形进行筛选
    #      3）时间：开始的时间一定要在点上，然后结束的时间在强拍的弱部分，即需要超过强拍时值的一半
    # ---------------------
    def _get_split(self):
        split_dict = dict()  # 切分音集合
        split_dict_4 = dict()  # 切分音集合 ｜ 测试使用
        split_dict_8 = dict()
        split_dict_16 = dict()

        step16 = self.step
        for bar_id, bar_notes in self.subsections.items():
            if bar_id not in split_dict:
                split_dict[bar_id] = []
                split_dict_4[f'{bar_id}'] = []
                split_dict_8[f'{bar_id}'] = []
                split_dict_16[f'{bar_id}'] = []

            start = self.bar_ticks * bar_id
            note_start_4 = [4 * step16 + start, 12 * step16 + start]
            note_start_8 = [i * step16 + start for i in range(2, 16, 4)]
            note_start_16 = [i * step16 + start for i in range(1, 18, 2)]

            for note in bar_notes:
                # 1）过滤小于16分音符时值的音符，无切分音意义；
                note_duration = note.end - note.start
                if note_duration >= step16:
                    # 2.1）根据4分音符的所有切分音情形进行筛选
                    if (note.start == note_start_4[0]) and (note.end >= (9.5 * step16 + start)):
                        split_dict[bar_id].append(note)
                        split_dict_4[f'{bar_id}'].append(note)
                    elif (note.start == note_start_4[1]) and (note.end >= 17.5 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_4[f'{bar_id}'].append(note)

                    # 2.2）根据8分音符的所有切分音情形进行筛选
                    elif (note.start == note_start_8[0]) and (note.end >= 5 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_8[f'{bar_id}'].append(note)
                    elif (note.start == note_start_8[1]) and (note.end >= 9 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_8[f'{bar_id}'].append(note)
                    elif (note.start == note_start_8[2]) and (note.end >= 13 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_8[f'{bar_id}'].append(note)
                    elif (note.start == note_start_8[3]) and (note.end >= 17 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_8[f'{bar_id}'].append(note)

                    # 2.3）根据16分音符的所有切分音情形进行筛选：音符开头在小节线上，音符结尾至少要超过最近强拍的一半时值
                    elif (note.start == note_start_16[0]) and (note.end >= 2.5 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[1]) and (note.end >= 4.5 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[2]) and (note.end >= 6.5 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[3]) and (note.end >= 8.5 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[4]) and (note.end >= 10.5 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[5]) and (note.end >= 12.5 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[6]) and (note.end >= 14.5 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[7]) and (note.end >= 16.5 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                else:
                    continue
        return split_dict, split_dict_4, split_dict_8, split_dict_16

    # -----------
    # 骨干音提取，筛选规整如下：
    # 1）过滤小于16分音符时值的音符，无切分音意义；
    # 2）三次挑选，查看石墨文档 3.2.1 提取结构线条音
    # -----------
    def get_skeleton_job(self):
        heavy_dict = self._get_stress()
        long_dict = self._get_long()
        split_dict, _, _, _ = self._get_split()
        heavy_list = list(chain(*heavy_dict.values()))
        long_list = list(chain(*long_dict.values()))
        split_list = list(chain(*split_dict.values()))

        step16 = self.step
        skeleton_dict = dict()
        skeleton_dict_temp = dict()

        for k, v in self.subsections.items():  # 遍历每个小节的音符
            if k not in skeleton_dict:
                skeleton_dict[k] = []
                skeleton_dict_temp[k] = []

            for idx, note in enumerate(v):
                # 第1次挑选 ｜ 条件1：当节拍重音只属于节拍重音集合时
                if ((note in heavy_list) and (note not in long_list) and (note not in split_list)):
                    if len(v) == 0:  # 空小节
                        pass
                    elif len(v) == 1:
                        skeleton_dict[k].append(note)
                    elif len(v) >= 2:
                        if idx + 1 == len(v):  # 结尾
                            skeleton_dict[k].append(note)
                        else:
                            if v[idx + 1] not in split_list:
                                note_duration = note.end - note.start
                                if note_duration >= 4 * step16:
                                    skeleton_dict[k].append(note)
                                elif (2 * step16 <= note_duration < 4 * step16):
                                    for note_next in v[idx + 2:]:
                                        if (note_next in split_list) and (note_next.start - note.end) >= 3 * step16:
                                            skeleton_dict[k].append(note)
                                elif (step16 <= note_duration < 2 * step16):
                                    for note_next in v[idx + 2:]:
                                        if (note_next in split_list) and (note_next.start - note.end) >= 4 * step16:
                                            skeleton_dict[k].append(note)
                # 第2次挑选 ｜ 条件2和3：当note同时属于heavy_dict与long_dict时， & 当note同时属于long_dict与split_dict时
                elif ((note in heavy_list) and (note in long_list) and (note not in split_list)) or (
                        (note not in heavy_list) and (note in long_list) and (note in split_list)):
                    skeleton_dict[k].append(note)
                # 第3次挑选 ｜
                elif ((note not in heavy_list) and (note in long_list) and (note not in split_list)) or (
                        (note not in heavy_list) and (note not in long_list) and (note in split_list)):
                    # 1）当前该小节中不存在骨干音
                    if len(skeleton_dict[k]) == 0:
                        # 若上一个小节不存在，直接判定为骨干音
                        if (k - 1 not in skeleton_dict.keys()):
                            skeleton_dict[k].append(note)
                        # 若上一个小节存在，但是不存在骨干音，直接判定为骨干音
                        elif ((k - 1) in skeleton_dict.keys()) and len(skeleton_dict[k - 1]) == 0:
                            skeleton_dict[k].append(note)
                        # 若上一个小节存在，同时存在骨干音，该音符为条件骨干音
                        elif ((k - 1) in skeleton_dict.keys()) and len(skeleton_dict[k - 1]) != 0:
                            val = abs(note.pitch - skeleton_dict[k - 1][-1].pitch) % 12
                            if val == 1 or val == 2 or val == 5 or val == 7:
                                skeleton_dict[k].append(note)
                    # 2）当前该小节中已存在骨干音，该音符为条件骨干音
                    else:
                        if note.start > skeleton_dict[k][-1].start:
                            last_ske_note = skeleton_dict[k][-1]
                            val = abs(note.pitch - last_ske_note.pitch) % 12
                            if val == 1 or val == 2 or val == 5 or val == 7:
                                skeleton_dict[k].append(note)
                        else:
                            for count, ske_note in enumerate(skeleton_dict[k]):
                                # 在该小节中，若该音符前面已存在骨干音，与之最近的骨干音进行条件计算
                                if note.start < ske_note.start:
                                    if count == 0:  # 需要与上一个小节进行对比
                                        if (k - 1 not in skeleton_dict.keys()):
                                            skeleton_dict[k].append(note)
                                            break
                                        # 若上一个小节存在，但是不存在骨干音，直接判定为骨干音
                                        elif ((k - 1) in skeleton_dict.keys()) and len(skeleton_dict[k - 1]) == 0:
                                            skeleton_dict[k].append(note)
                                            break
                                        # 若上一个小节存在，同时存在骨干音，该音符为条件骨干音
                                        elif ((k - 1) in skeleton_dict.keys()) and len(skeleton_dict[k - 1]) != 0:
                                            val = abs(note.pitch - skeleton_dict[k - 1][-1].pitch) % 12
                                            if val == 1 or val == 2 or val == 5 or val == 7:
                                                skeleton_dict[k].append(note)
                                                break
                                    elif count != 0 and count != len(skeleton_dict[k]) - 1:
                                        last_ske_note = skeleton_dict[k][count - 1]
                                        val = abs(note.pitch - last_ske_note.pitch) % 12
                                        if val == 1 or val == 2 or val == 5 or val == 7:
                                            skeleton_dict[k].append(note)
                                        break

        # 音符排序
        for k, v in skeleton_dict.items():
            v.sort(key=lambda x: (x.start, -x.end))

        # ------ 骨干音筛选 ------
        # 1. 相邻的音有相同的pitch 那么选择第一个
        # 2. 两个及以上的骨干音群 如果都属于某一个字典 那么只保留第一个
        full = list(chain(*self.subsections.values()))  # 所有音符集合
        skeleton = list(chain(*skeleton_dict.values()))  # 骨干音符集合
        decoration = list(set(full) - set(skeleton))  # 装饰音集合
        decoration.sort(key=lambda x: (x.start, -x.end))
        dec = full.copy()  # 将骨干音标记为0的装饰音集合 装饰音是骨干音群的分界
        for i in range(len(dec)):
            if dec[i] in skeleton:
                dec[i] = 0

        skeleton_dict_filtered = dict()  # 筛选后的骨干音词典
        for k, v in skeleton_dict.items():
            skeleton_dict_filtered[k] = []
            i = 1
            if len(v) == 0:
                continue
            target = v[0]  # 锁定的目标音
            skeleton_dict_filtered[k].append(target)  # 第一个一定是需要保留的骨干音
            while i < len(v):
                ### 筛选条件
                ## 1. 在同一个骨干音群中
                ## 2. 音群中的pitch相同 或者所属的功能类别相同
                ## 3. 音符之间隔离不能太远 界限设置为10*480ticks
                if ((v[i].pitch == target.pitch) or \
                    (v[i] in heavy_list and target in heavy_list) or \
                    (v[i] in split_list and target in split_list) or \
                    (v[i] in long_list and target in long_list)) and \
                        (0 not in [dec[i] for i in range(full.index(target), full.index(v[i]) + 1)]) and \
                        v[i].start - target.end <= 10 * self.resolution:  # 如果之后的音符与target有相同的作用 不计入
                    i += 1
                    continue
                else:  # 后去遇到不同作用的音符
                    skeleton_dict_filtered[k].append(target)  # 追加现在的音符到新字典中
                    target = v[i]  # 更新新的目标音符
                    i += 1

        return skeleton_dict_filtered

def melody_skeleton_job(midi_path, dst):
    midi_fn = miditoolkit.MidiFile(midi_path) # save markers
    sketeton_notes = []
    skeleton_notes_temp = SkeletonExtractor(midi_fn).get_skeleton_job()
    for k, v in skeleton_notes_temp.items():
        sketeton_notes.extend(v)
    midi_fn.instruments[0].notes.clear()
    midi_fn.instruments[0].notes.extend(sketeton_notes)
    midi_fn.dump(f"{dst}/{os.path.basename(midi_path)}")
    return f"{dst}/{os.path.basename(midi_path)}"

def melody_skeleton_job(midi_path, dst):
    midi_fn = miditoolkit.MidiFile(midi_path) # save markers
    sketeton_notes = []
    skeleton_notes_temp = SkeletonExtractor(midi_fn).get_skeleton_job()
    for k, v in skeleton_notes_temp.items():
        sketeton_notes.extend(v)
    midi_fn.instruments[0].notes.clear()
    midi_fn.instruments[0].notes.extend(sketeton_notes)
    midi_fn.dump(f"{dst}/{os.path.basename(midi_path)}")
    return f"{dst}/{os.path.basename(midi_path)}"


######
def filter_melody_job(midi_path, dst_dir, melody_model, bass_model, chord_model, drum_model):
    # 使用MIDI Miner进行关键轨道角色识别（主旋律轨道、bass轨道），返回相对应的index位置
    pm, melody_tracks_idx = predict_track_with_model(midi_path, melody_model, bass_model, chord_model, drum_model)

    # 识别关键轨道，进行轨道重命名，并重新保存一份
    if pm is None:
        return 'pm is None'
    else:
        pm_new = deepcopy(pm)
        pm_new.instruments = []
        for i, instru_old in enumerate(pm.instruments):
            # track program
            program_old = instru_old.program
            instru = deepcopy(instru_old)
            if i in melody_tracks_idx or 73 <= program_old <= 88:
                instru.name = 'Lead'
                instru.program = 80
                pm_new.instruments.append(instru)
                out_path = f"{dst_dir}/{os.path.basename(midi_path)[:-4]}_{i}.mid"
                pm_new.write(out_path)
                return out_path


def predict_track_with_model(midi_path, melody_model, bass_model, chord_model,drum_model):

    # 删除音符少的轨道
    # retrun pm after removing tracks which are empty or less than 10 notes (删除掉一些空轨道和音符少于10个)
    # empty track基本上已经在一轮已经筛选掉了，这里重点在于筛选掉音符比较少的轨道
    try:
        ret = tc.cal_file_features(midi_path)  # 去除空轨和音符少的轨道， 并计算特征（34 features from midi data）
        features, pm = ret
    except Exception as e:
        features = None
        pm = pretty_midi.PrettyMIDI(midi_path)

    if features is None or features.shape[0] == 0:
        return pm, []

    # predict melody and bass tracks' index
    features = tc.add_labels(features)  # 添加 label
    tc.remove_file_duplicate_tracks(features, pm)  # 去除重复轨道
    features = tc.predict_labels(features, melody_model, bass_model, chord_model, drum_model)  # 预测lead, bass, chord
    predicted_melody_tracks_idx = np.where(features.melody_predict)[0]
    melody_tracks_idx = np.concatenate((predicted_melody_tracks_idx, np.where(features.is_melody)[0]))
    return pm, melody_tracks_idx


###############
# 操作：统计挑选得到的轨道类型中各个轨道的信息和总体的信息
###############
def get_merged_midi_info(midi_fn, instru2program):
    try:
        mf = miditoolkit.MidiFile(midi_fn)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return str(e)

    # merge tracks; 将相同类型的Tracks 合并到同一个列表中
    track_lists_to_merge = get_tracks_to_merge(mf, instru2program)
    # 每一类轨道中需要合并的轨道数量，例如 在经过筛选后的MIDI文件中，钢琴类型的轨道有N条, Bass有...
    n_merge_track = [len(x) for x in track_lists_to_merge]
    # 此处说明: 因为filter_recog_merge_job函数中轨道筛选的两种方式（midi miner和program）会导致有重合，因此，我们采用集合不重复性质进行筛选
    available_instrs = list(set([x2 for x in track_lists_to_merge for x2 in x]))  # Important for 6 tracks

    # notes （velocity、pitch、the number of notes）
    all_vels = [x1.velocity for i, x in enumerate(mf.instruments) if i in available_instrs for x1 in
                x.notes]  # all instruments note connection in a line
    all_pitches = [x1.pitch for i, x in enumerate(mf.instruments) if i in available_instrs for x1 in x.notes]
    n_notes = len(all_vels)  # 音符总数

    if n_notes == 0:
        return 'empty tracks'

    # the number of beats （拍总数）
    n_beats = max([x1.end for i, x in enumerate(mf.instruments)
                   if i in available_instrs for x1 in x.notes]) // mf.ticks_per_beat + 1

    n_instru = len(mf.instruments)
    n_pitches = len(set(all_pitches))  # 曲子中音高种类总数
    vel_mean = np.mean(all_vels)
    vel_std = np.std(all_vels)

    n_cross_bar = 0
    for i, instru in enumerate(mf.instruments):
        if i not in available_instrs:
            continue
        for n in instru.notes:
            start_beat = n.start / mf.ticks_per_beat
            end_beat = n.end / mf.ticks_per_beat
            if (start_beat + 0.25) // 4 < (end_beat - 0.25) // 4 and start_beat % 4 > 0.125:
                n_cross_bar += 1

    return {
        'path_recog_tracks': midi_fn,
        # velocity
        'vel_mean': vel_mean,
        'vel_std': vel_std,
        # stats
        'n_notes': n_notes,
        'n_instru': n_instru,
        'n_beats': n_beats,
        'n_pitches': n_pitches,
        'n_cross_bar': n_cross_bar,
        # tracks
        'n_tracks': n_merge_track,
        'track_lists_to_merge': track_lists_to_merge,
    }


###############
# 操作：创建六种不同乐器的列表，并进行分类到各自的列表中去
###############
def get_tracks_to_merge(mf, instru2program):
    # six kind of Tracks
    track_lists_to_merge = [[] for _ in range(6)]
    # instrument order
    instru_order = {v: k for k, v in enumerate(instru2program.keys())}
    for idx, instr in enumerate(mf.instruments):
        instr_name = instr.name
        if instr_name in instru_order:
            track_lists_to_merge[instru_order[instr_name]].append(idx)
    return track_lists_to_merge

###############
# 操作：创建六种不同乐器的列表，并进行分类到各自的列表中去
###############
def filter_tracks(midi_info, hparams):
    """
    过滤规则：
    1）过滤太长的 n_beats > 10000，太短的 n_beats < 16
    2）过滤音太少的
    3）过滤pitch变化太小的
    4）过滤拍号或者resolution标记错误的 cross_bar_rate > 0.15
    """

    # 1） 过滤太长的 n_beats > 10000，太短的 n_beats < 16
    if midi_info['n_beats'] > hparams['max_n_beats'] or midi_info['n_beats'] < hparams['min_n_beats']:
        return 'invalid beats'

    # 2）过滤短片段
    if midi_info['n_notes'] < hparams['min_n_notes']:
        return 'invalid n_notes'

    # 3）过滤pitch变化太小
    if midi_info['n_pitches'] < hparams['min_n_pitches']:
        return 'Invalid pitches'

    # 4）过滤拍号或者resolution标记错误的 cross_bar_rate > 0.15
    if midi_info['cross_bar_rate'] > hparams['max_cross_bar_rate']:
        return 'Invalid cross_bar'

    return ''