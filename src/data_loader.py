import json
from typing import List
import librosa
import numpy as np

data_dir = "E:/bs-map-generator/data/maps"

class Note:
    time: float
    lineIndex: int
    lineLayer: int
    type: int
    direction: int

    def __init__(self, time, lineIndex, lineLayer, type, cutDirection):
        if lineIndex < 0 or lineIndex > 3 or lineLayer < 0 or lineLayer > 2:
            raise Exception("too big note number")
        self.time = time
        self.lineIndex = lineIndex
        self.lineLayer = lineLayer
        self.type = type
        self.direction = cutDirection

class Map:
    diff: int
    njs: float
    notes: List[Note]

    def __init__(self, diff, njs, notes):
        self.diff = diff
        self.njs = njs
        self.notes = notes

def preprocess_map(map_json, bpm, njs, diff):
    if map_json["_version"] >= "2.6.0" or map_json["_version"] < "2.0.0":
        raise Exception("not v2") # I don't like bpm changes :hahaball:
    
    # if len(map_json["_notes"]) > 1000:
    #     return None
    
    map_json["_notes"].sort(key = lambda note: note["_time"])
    processed_notes = [Note(note["_time"] * 60 / bpm, note["_lineIndex"], note["_lineLayer"], note["_type"], note["_cutDirection"]) for note in map_json["_notes"]]
    return Map(diff, njs, processed_notes)
    

def load_map(map_dir):
    info_file = f"{map_dir}/info.dat"
    
    diffs = []
    
    with open(info_file, "rb") as f:
        info_json = json.load(f)
    
    if info_json["_songTimeOffset"] != 0:
        raise Exception("_songTimeOffset not equal to 0")

    for map_set in info_json["_difficultyBeatmapSets"]:
        if map_set["_beatmapCharacteristicName"] == "Standard":
            for diff in map_set["_difficultyBeatmaps"]:
                with open(f"{map_dir}/{diff['_beatmapFilename']}", "rb") as diff_f:
                    diff_json = json.load(diff_f)
                    diff = preprocess_map(diff_json, info_json["_beatsPerMinute"], diff["_noteJumpMovementSpeed"], diff["_difficultyRank"])
                    if diff != None:
                        diffs.append(diff)

    if len(diffs) == 0:
        return ([], 0.1), []
    
    song, sampling_rate = librosa.load(f"{map_dir}/{info_json['_songFilename']}", res_type="linear")
    n_fft = 256
    hop_length = int(n_fft / 1)
    stft_audio = librosa.stft(song, n_fft = n_fft, hop_length = hop_length)
    y_audio = np.abs(stft_audio)
    y_log_audio = librosa.amplitude_to_db(y_audio)
    min_val = np.min(y_log_audio)
    max_val = np.max(y_log_audio)
    song_data = (y_log_audio - min_val)/(max_val - min_val)
    segment_duration = hop_length/sampling_rate
    song_data = np.transpose(song_data)
    return (song_data, segment_duration), diffs

context_length = 1
note_time_delta = 0.05
prediction_note_count = context_length * 50
note_count = 10
note_length = 35

def encode_int(array, min_value, value, limit):
    for i in range(min_value, limit + 1):
        array.append(1 if i == value else 0)

def encode_empty(array):
    note = [0]*note_length
    array.append(note)

def encode_to_array(note, array, time):
    array_note = []
    array_note.append(abs(note.time - time))
    if (note.type == 0):
        encode_int(array_note, 0, note.lineIndex, 3)
        encode_int(array_note, 0, note.lineLayer, 2)
        # encode_int(array_note, 0, 8, 9)
        encode_int(array_note, 0, note.direction, 9)
        encode_int(array_note, 0, -2, 16)
    elif (note.type == 1):
        encode_int(array_note, 0, -2, 16)
        encode_int(array_note, 0, note.lineIndex, 3)
        encode_int(array_note, 0, note.lineLayer, 2)
        # encode_int(array_note, 0, 8, 9)
        encode_int(array_note, 0, note.direction, 9)
    else:
        encode_int(array_note, 0, note.lineIndex, 3)
        encode_int(array_note, 0, note.lineLayer, 2)
        encode_int(array_note, 0, 9, 9)
        encode_int(array_note, 0, note.lineIndex, 3)
        encode_int(array_note, 0, note.lineLayer, 2)
        encode_int(array_note, 0, 9, 9)
    array.append(array_note)

def full_load_map_old(map_folder):
    (song_data, segment_duration), diffs = load_map(map_folder.decode('UTF-8'))
    
    note_iterator = 0
    results = []
    for diff in diffs:
        # diffs.sort(key=lambda diff: diff.diff, reverse=True)
        # diff = diffs[0]

        context_steps = int(context_length / segment_duration) + 1
        step_size = int(note_time_delta / segment_duration) + 1
        
        x_precontext_audio = []
        x_precontext_notes = []
        x_postcontext_audio = []
        y_postcontext_notes = []
        
        for i in range(context_steps, song_data.shape[1] - context_steps, step_size):
            curr_time = i * segment_duration
            
            while len(diff.notes) > note_iterator and diff.notes[note_iterator].time < curr_time:
                note_iterator += 1

            if len(diff.notes) > note_iterator:
                if diff.notes[note_iterator].time < curr_time + note_time_delta:
                    encode_to_array(diff.notes[note_iterator], y_postcontext_notes, curr_time) 
                else:
                    encode_empty(y_postcontext_notes)
            else:
                break

            x_precontext_audio.append(song_data[:, i-context_steps:i])
            x_postcontext_audio.append(song_data[:, i:i+context_steps])

            notes = []
            for _j in range(note_count):
                j = 10 - _j
                curr_iter = note_iterator - j
                if curr_iter < 0 or diff.notes[curr_iter].time < curr_time - context_length:
                    encode_empty(notes)
                else:
                    encode_to_array(diff.notes[curr_iter], notes, curr_time)                
            x_precontext_notes.append(notes)
        
        if len(x_precontext_audio) == 0:
            continue
        
        # results.append((np.array(x_precontext_audio), np.array(x_precontext_notes), np.array(x_postcontext_audio), np.array(y_postcontext_notes)))
        results.append(((x_precontext_audio), (x_precontext_notes), (x_postcontext_audio), (y_postcontext_notes)))
    return results


def full_load_map(map_folder):
    (song_data, segment_duration), diffs = load_map(map_folder.decode('UTF-8'))
    
    note_iterator = 0
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
        
        prev_note_segment = [[0]*217 for i in range(prediction_note_count)]
        prev_audio_segment = song_data[:context_steps, :]
        for i in range(context_steps, song_data.shape[0] - context_steps, step_size):
            curr_time = i * segment_duration
            note_segment = [[0]*217 for i in range(prediction_note_count)]
            while len(diff.notes) > note_iterator and (diff.notes[note_iterator].type == 3 or diff.notes[note_iterator].time < curr_time):
                note_iterator += 1
            loc_note_iterator = note_iterator
            for j in range(prediction_note_count):
                curr_note_time = curr_time + j * 0.02
                next_note_time = curr_time + j * 0.02 + 0.02
                while len(diff.notes) > loc_note_iterator and (diff.notes[loc_note_iterator].type == 3 or diff.notes[loc_note_iterator].time < curr_note_time):
                    loc_note_iterator += 1
                while len(diff.notes) > loc_note_iterator and (diff.notes[loc_note_iterator].time < next_note_time):
                    if diff.notes[loc_note_iterator].type == 3:
                        loc_note_iterator += 1
                        continue
                    else:
                        note_segment[j][0] = 1
                        note_segment[j][1 + diff.notes[loc_note_iterator].type*108 + diff.notes[loc_note_iterator].direction*12 + diff.notes[loc_note_iterator].lineIndex*3 + diff.notes[loc_note_iterator].lineLayer] = 1
                        loc_note_iterator += 1
            
            sum_notes = sum([sum(note_segment_item) for note_segment_item in note_segment])
            
            sum_note_pos = np.max(np.sum(np.array(note_segment)[:, 1:], axis=0))
            
            audio_segment = song_data[i:i+context_steps, :]
            
            # filter out very dense maps for now
            if sum_note_pos < 3 and sum_notes > 3 and sum_notes < 20:
                x_context_prev_audio.append(prev_audio_segment)
                x_context_prev_notes.append(prev_note_segment)
                x_context_audio.append(audio_segment)
                y_context_notes.append(note_segment)
            
            prev_audio_segment = audio_segment
            prev_note_segment = note_segment
        
        if len(y_context_notes) == 0:
            continue
        
        # results.append((np.array(x_precontext_audio), np.array(x_precontext_notes), np.array(x_postcontext_audio), np.array(y_postcontext_notes)))
        results.append((x_context_prev_audio, x_context_prev_notes, x_context_audio, y_context_notes))
    return results
