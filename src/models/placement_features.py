import tensorflow as tf
from models.audio import create_audio_model_stereo

def _create_placement_selection_features_model():
    input_audio = tf.keras.Input(shape=(40, 64), dtype="float32")
    input_timing_selection_style = tf.keras.Input(shape=(256,), dtype="float32")
    input_acc_prediction = tf.keras.Input(shape=(1, ), dtype="float32")
    input_speed_prediction = tf.keras.Input(shape=(1, ), dtype="float32")

    l_timing_selection_style = tf.keras.layers.RepeatVector(40)(input_timing_selection_style)
    l_acc_prediction = tf.keras.layers.RepeatVector(40)(input_acc_prediction)
    l_speed_prediction = tf.keras.layers.RepeatVector(40)(input_speed_prediction)
    l_prediction = tf.keras.layers.Concatenate(axis=2)([input_audio, l_timing_selection_style, l_acc_prediction, l_speed_prediction])
    l_prediction = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(l_prediction)
    l_prediction = tf.keras.layers.LSTM(256, return_sequences=True)(l_prediction)
    l_prediction = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation="tanh"))(l_prediction)
    
    return tf.keras.Model([input_audio, input_timing_selection_style, input_acc_prediction, input_speed_prediction], [l_prediction])

def _create_timing_selection_style_model():
    input_audio = tf.keras.Input(shape=(40, 64), dtype="float32")
    input_notes = tf.keras.Input(shape=(40, 50), dtype="float32")
    
    l = tf.keras.layers.Concatenate(axis=2)([input_audio, input_notes])
    l = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(l)
    l = tf.keras.layers.LSTM(256, return_sequences=True)(l)
    l_timing_selection_style = tf.keras.layers.LSTM(256)(l)
    
    return tf.keras.Model([input_audio, input_notes], [l_timing_selection_style])
    
    
def create_placement_features_model():
    audio_model = create_audio_model_stereo()
    placement_selection_features_model = _create_placement_selection_features_model()
    timing_selection_style_model = _create_timing_selection_style_model()
    
    input_prev_audio = tf.keras.Input(shape=(2, 87, 129, 1), dtype="float32")
    input_audio = tf.keras.Input(shape=(2, 87, 129, 1), dtype="float32")
    input_prev_notes = tf.keras.Input(shape=(2, 40, 25), dtype="float32")
    input_acc_prediction = tf.keras.Input(shape=(1, ), dtype="float32")
    input_speed_prediction = tf.keras.Input(shape=(1, ), dtype="float32")
    input_y_notes = tf.keras.Input(shape=(2, 40, 25), dtype="float32")
    
    l_prev_audio = audio_model(input_prev_audio)
    l_audio = audio_model(input_audio)

    l_prev_notes = tf.keras.layers.Permute((2, 1, 3))(input_prev_notes)
    l_prev_notes = tf.keras.layers.Reshape((40, -1))(l_prev_notes)

    l_y_notes = tf.keras.layers.Permute((2, 1, 3))(input_y_notes)
    l_y_notes = tf.keras.layers.Reshape((40, -1))(l_y_notes)

    l_prev_selection_style = timing_selection_style_model([l_prev_audio, l_prev_notes])
    l_y_selection_style = timing_selection_style_model([l_audio, l_y_notes])
    
    l_prevstyle_prev_prediction = placement_selection_features_model([l_prev_audio, l_prev_selection_style, input_acc_prediction, input_speed_prediction])
    l_prevstyle_prediction = placement_selection_features_model([l_audio, l_prev_selection_style, input_acc_prediction, input_speed_prediction])
    
    l_ystyle_prev_prediction = placement_selection_features_model([l_prev_audio, l_y_selection_style, input_acc_prediction, input_speed_prediction])
    l_ystyle_prediction = placement_selection_features_model([l_audio, l_y_selection_style, input_acc_prediction, input_speed_prediction])

    return tf.keras.Model([input_prev_audio, input_audio, input_prev_notes, input_acc_prediction, input_speed_prediction, input_y_notes], [l_prevstyle_prev_prediction, l_prevstyle_prediction, l_ystyle_prev_prediction, l_ystyle_prediction])
