import json
from typing import List
import librosa
import numpy as np
from numba import jit
from numba.experimental import jitclass
import numba.typed
import numba
import requests
import random


context_length = 1
prediction_note_count = context_length * 40
prediction_note_time_length = context_length / prediction_note_count
note_count = 10
note_length = 35


data_dir = "E:/bs-map-generator/data/maps"

@jit(nopython=True)
def get_note_direction_v3(direction, angle, direction_to_angle, angle_to_direction):
    if direction == 8:
        return 8
    
    note_angle = (direction_to_angle[direction] - round(angle/45)*45) % 360
    return angle_to_direction[note_angle]

@jit(nopython=True)
def get_note_direction(angle, angle_to_direction):
    angle = int(angle * 360)
    angle = (angle - angle % 45) % 360
    return angle_to_direction[angle]

@jit(nopython=True)
def get_note_angle(direction, direction_to_angle):
    return direction_to_angle[direction] / 360

@jitclass
class Note:
    time: float
    lineIndex: int
    lineLayer: int
    type: int
    direction: int

    # @jit(nopython=True)
    def __init__(self, time, lineIndex, lineLayer, type, cutDirection):
        self.time = time
        self.lineIndex = lineIndex
        self.lineLayer = lineLayer
        self.type = type
        self.direction = cutDirection

# @jitclass
class Map:
    diff: int
    njs: float
    notes: List[Note]
    note_stats: List[List[float]]

    # @jit(nopython=True)
    def __init__(self, diff, njs, notes):
        self.diff = diff
        self.njs = njs
        self.notes = notes

def preprocess_map(map_json, diff_info, bpm):
    if "_customData" in diff_info and \
        "_requirements" in diff_info["_customData"] and \
        len(diff_info["_customData"]["_requirements"]) > 0:
        return None
    
    map_version = map_json["_version"] if "_version" in map_json else (map_json["version"] if "version" in map_json else "2.0.0")
    
    if map_version < "2.0.0":
        return None
    elif map_version >= "3.0.0":
        if "bpmEvents" in map_json and len(map_json["bpmEvents"]) > 0 or len(map_json["burstSliders"]) > 0 or len(map_json["sliders"]) > 0:
            return None
        
        
        direction_to_angle = numba.typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.types.int64,
        )
        angle_to_direction = numba.typed.Dict.empty(
            key_type=numba.types.int64,
            value_type=numba.types.int64,
        )

        direction_to_angle[0] = 180
        direction_to_angle[1] = 0
        direction_to_angle[2] = 90
        direction_to_angle[3] = 270
        direction_to_angle[4] = 135
        direction_to_angle[5] = 225
        direction_to_angle[6] = 45
        direction_to_angle[7] = 315

        angle_to_direction[180] = 0
        angle_to_direction[0] = 1
        angle_to_direction[90] = 2
        angle_to_direction[270] = 3
        angle_to_direction[135] = 4
        angle_to_direction[225] = 5
        angle_to_direction[45] = 6
        angle_to_direction[315] = 7
        
        
        map_json["colorNotes"].sort(key = lambda note: note["b"])
        processed_notes = [Note(note["b"] * 60 / bpm, note["x"], note["y"], note["c"], get_note_direction_v3(note["d"], note["a"] if "a" in note else 0, direction_to_angle, angle_to_direction)) for note in map_json["colorNotes"]]
    else:
        if "_sliders" in map_json and len(map_json["_sliders"]) > 0:
            return None
        bpm_changes = [e for e in map_json["_events"] if e["_type"] == 100] if "_events" in map_json else []
        if len(bpm_changes) > 0:
            return None
        map_json["_notes"].sort(key = lambda note: note["_time"])
        processed_notes = [Note(note["_time"] * 60 / bpm, note["_lineIndex"], note["_lineLayer"], note["_type"], note["_cutDirection"]) for note in map_json["_notes"]]

    if len(processed_notes) == 0:
        return None
        
    njs = diff_info["_noteJumpMovementSpeed"]
    diff = diff_info["_difficultyRank"]
    
    return Map(diff, njs, processed_notes)
    
    
def load_dat_files(map_dir):
    info_file = f"{map_dir}/info.dat"
    map_hash = map_dir[-40:]
    diffs = []
    
    with open(info_file, "rb") as f:
        info_json = json.load(f)
    
    if info_json["_songTimeOffset"] != 0:
        return None
        # raise Exception("_songTimeOffset not equal to 0")

    for map_set in info_json["_difficultyBeatmapSets"]:
        if map_set["_beatmapCharacteristicName"] == "Standard":
            for diff in map_set["_difficultyBeatmaps"]:
                with open(f"{map_dir}/{diff['_beatmapFilename']}", "rb") as diff_f:
                    diff_json = json.load(diff_f)
                    diff = preprocess_map(diff_json, diff, info_json["_beatsPerMinute"])
                    if diff != None:
                        stats_ai_response = requests.get(f"https://bs-replays-ai.azurewebsites.net/json/{map_hash}/Standard/{diff.diff}/full/time-scale/1")
                        if not stats_ai_response.ok:
                            continue
                        stats_ai_results = stats_ai_response.json()
                        if "rows" not in stats_ai_results["notes"] or len(stats_ai_results["notes"]["rows"]) == 0:
                            continue
                        diff.note_stats = [(row[4], row[0], row[1]) for row in stats_ai_results["notes"]["rows"]]
                        diff.notes = diff.notes[:len(diff.note_stats)]
                        diffs.append(diff)
    song_filename = info_json['_songFilename']
    return song_filename, diffs

def load_audio(song_filename):
    song, sampling_rate = librosa.load(song_filename, mono=False, res_type="linear")
    n_fft = 256
    hop_length = int(n_fft / 1)
    stft_audio = librosa.stft(song, n_fft = n_fft, hop_length = hop_length)
    y_audio = np.abs(stft_audio)
    y_log_audio = librosa.amplitude_to_db(y_audio)
    min_val = np.min(y_log_audio)
    max_val = np.max(y_log_audio)
    song_data = (y_log_audio - min_val)/(max_val - min_val)
    segment_duration = hop_length/sampling_rate
    song_data = np.transpose(song_data, axes=[0, 2, 1])
    return (song_data, segment_duration)

def load_map(map_dir):
    dat_files_data = load_dat_files(map_dir)
    if dat_files_data == None:
        return None

    song_filename, diffs = dat_files_data
    
    if len(diffs) == 0:
        return None
    
    return load_audio(f"{map_dir}/{song_filename}"), diffs

def check_parity_ok(prev_note_segment, curr_note_segment):
    note_segment = (prev_note_segment[0] + curr_note_segment[0], prev_note_segment[1] + curr_note_segment[1])
    for color in range(2):
        last_i = 0
        for i in range(1, len(note_segment[color])):
            if note_segment[color][i][0] == 1:
                j = last_i
                if note_segment[color][j][0] == 1:
                    for note_i_iter, note_i in enumerate(note_segment[color][i][1::2]):
                        if note_i == 1:
                            note_angle_i = note_segment[color][i][1 + note_i_iter * 2 + 1]
                        
                    for note_j_iter, note_j in enumerate(note_segment[color][j][1::2]):
                        if note_j == 1:
                            note_angle_j = note_segment[color][j][1 + note_j_iter * 2 + 1]
                    if note_j_iter == note_i_iter:
                        if min(abs(note_angle_i - note_angle_j), 1 - abs(note_angle_i - note_angle_j)) < 0.2:
                            return False
                    elif min(abs(note_angle_i - note_angle_j), 1 - abs(note_angle_i - note_angle_j)) < 0.1:
                        return False
                last_i = i
    return True


def full_load_map(map_folder):
    load_map_data = load_map(map_folder)
    if load_map_data == None:
        return []
    (song_data, segment_duration), diffs = load_map_data
    direction_to_angle = numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.int64,
    )
    angle_to_direction = numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.int64,
    )
    direction_to_angle[0] = 180
    direction_to_angle[1] = 0
    direction_to_angle[2] = 90
    direction_to_angle[3] = 270
    direction_to_angle[4] = 135
    direction_to_angle[5] = 225
    direction_to_angle[6] = 45
    direction_to_angle[7] = 315

    angle_to_direction[180] = 0
    angle_to_direction[0] = 1
    angle_to_direction[90] = 2
    angle_to_direction[270] = 3
    angle_to_direction[135] = 4
    angle_to_direction[225] = 5
    angle_to_direction[45] = 6
    angle_to_direction[315] = 7
    note_iterator = 0
    note_stats_iterator = 0
    results = []
    for diff in diffs:
        # diffs.sort(key=lambda diff: diff.diff, reverse=True)
        # diff = diffs[0]

        context_steps = int(context_length / segment_duration) + 1
        step_size = context_steps
        
        x_context_prev_audio = []
        x_context_prev_notes = []
        x_context_audio = []
        y_context_notes = []
        z_note_counts = []
        z_timing_counts = []
        z_note_pos_counts = []
        z_acc_prediction = []
        z_speed_prediction = []
        
        prev_note_segment = ([[0]*25 for i in range(prediction_note_count)], [[0]*25 for i in range(prediction_note_count)])
        prev_audio_segment = song_data[:, :context_steps, :]
        for i in range(context_steps + random.randint(0, context_steps), song_data.shape[1] - context_steps, step_size):
            curr_time = i * segment_duration
            note_segment = ([[0]*25 for i in range(prediction_note_count)], [[0]*25 for i in range(prediction_note_count)])
            note_count_segment = 0
            timing_count_segment = 0
            while len(diff.notes) > note_iterator and (diff.notes[note_iterator].type == 3 or diff.notes[note_iterator].time < curr_time):
                note_iterator += 1
            
            acc_prediction = 0
            speed_prediction = 0
            while len(diff.note_stats) > note_stats_iterator and diff.note_stats[note_stats_iterator][0] < curr_time:
                note_stats_iterator += 1
            loc_note_stats_iterator = note_stats_iterator
            while len(diff.note_stats) > loc_note_stats_iterator and diff.note_stats[loc_note_stats_iterator][0] < curr_time + segment_duration * step_size:
                acc_prediction += diff.note_stats[loc_note_stats_iterator][1]
                speed_prediction += diff.note_stats[loc_note_stats_iterator][2]
                loc_note_stats_iterator += 1
            acc_prediction = acc_prediction / (loc_note_stats_iterator - note_stats_iterator + 0.0000001)
            speed_prediction = speed_prediction / (loc_note_stats_iterator - note_stats_iterator + 0.0000001)
            
            loc_note_iterator = note_iterator
            skip_segment = False
            for j in range(prediction_note_count):
                curr_note_time = curr_time + j * prediction_note_time_length
                next_note_time = curr_time + j * prediction_note_time_length + prediction_note_time_length
                while len(diff.notes) > loc_note_iterator and (diff.notes[loc_note_iterator].type == 3 or diff.notes[loc_note_iterator].time < curr_note_time):
                    loc_note_iterator += 1
                while len(diff.notes) > loc_note_iterator and (diff.notes[loc_note_iterator].time < next_note_time):
                    if diff.notes[loc_note_iterator].type == 3 or diff.notes[loc_note_iterator].direction == 8 or skip_segment:
                        skip_segment = True
                        loc_note_iterator += 1
                        continue
                    else:
                        this_note = diff.notes[loc_note_iterator]
                        
                        this_note_segment_step = note_segment[this_note.type][j]
                        if this_note_segment_step[0] == 0:
                            timing_count_segment += 1
                        this_note_segment_step[0] = 1
                        note_count_segment += 1
                        
                        
                        this_note_angle = get_note_angle(this_note.direction, direction_to_angle)
                        this_note_segment_step[1 + this_note.lineIndex * 6 + this_note.lineLayer * 2] = 1
                        this_note_segment_step[1 + this_note.lineIndex * 6 + this_note.lineLayer * 2 + 1] = this_note_angle
                        
                        loc_note_iterator += 1
            
            parity_check = check_parity_ok(prev_note_segment, note_segment) if prev_note_segment is not None else True
                        
            skip_segment = skip_segment or not parity_check

            if skip_segment:
                prev_audio_segment = None
                prev_note_segment = None
                continue

            audio_segment = song_data[:, i:i+context_steps, :]
            
            if prev_audio_segment is None or prev_note_segment is None:
                prev_audio_segment = audio_segment
                prev_note_segment = note_segment
                continue
            
            # sum_note_pos = np.max(np.sum(np.where(np.array(note_segment)[:, 1:] > 0.9, 1, 0), axis=0) + np.sum(np.where(np.array(prev_note_segment)[:, 1:] > 0.9, 1, 0), axis=0))
            
            # filter out very dense maps for now
            if timing_count_segment > 2 and note_count_segment < 20 and timing_count_segment < 15:
                x_context_prev_audio.append(prev_audio_segment)
                x_context_prev_notes.append(prev_note_segment)
                x_context_audio.append(audio_segment)
                y_context_notes.append(note_segment)
                z_timing_counts.append([timing_count_segment])
                z_note_counts.append([note_count_segment])
                z_note_pos_counts.append([0])
                z_acc_prediction.append([acc_prediction])
                z_speed_prediction.append([speed_prediction])
            
            prev_audio_segment = audio_segment
            prev_note_segment = note_segment
        
        if len(y_context_notes) == 0:
            continue
        
        # results.append((np.array(x_precontext_audio), np.array(x_precontext_notes), np.array(x_postcontext_audio), np.array(y_postcontext_notes)))
        results.append((np.array(x_context_prev_audio, dtype=np.float32), np.array(x_context_prev_notes, dtype=np.float32), np.array(x_context_audio, dtype=np.float32), np.array(y_context_notes, dtype=np.float32), np.array(z_timing_counts, dtype=np.int32), np.array(z_note_counts, dtype=np.float32), np.array(z_note_pos_counts, dtype=np.float32), np.array(z_acc_prediction, dtype=np.float32), np.array(z_speed_prediction, dtype=np.float32)))
    
    return results
