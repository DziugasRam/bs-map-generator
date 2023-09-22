import tensorflow as tf

class NotePlacementActivationLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(NotePlacementActivationLayer, self).__init__()
    
    def call(self, inputs):
        inputs

def create_placement_selection_model():
    input_features = tf.keras.Input(shape=(40*2, 128), dtype="float32")
    input_prev_notes = tf.keras.Input(shape=(2, 40, 25), dtype="float32")

    l_prev_notes = tf.keras.layers.ZeroPadding2D(((0, 0), (0, 40)))(input_prev_notes)
    l_prev_notes = tf.keras.layers.Permute((2, 1, 3))(l_prev_notes)
    l_prev_notes = tf.keras.layers.Reshape((40*2, -1))(l_prev_notes)
    
    l = tf.keras.layers.Concatenate(axis=2)([input_features, l_prev_notes])
    l = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(l)
    l = tf.keras.layers.LSTM(128, return_sequences=True)(l)
    
    l_timings = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2, activation="sigmoid"))(l)
    l_timings = tf.keras.layers.Reshape((40*2, 2, -1))(l_timings)
    l_timings = tf.keras.layers.Permute((2, 1, 3))(l_timings)
    
    l_position = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(24, activation="sigmoid"))(l)
    l_position = tf.keras.layers.Reshape((40*2, 2, -1))(l_position)
    l_position = tf.keras.layers.Permute((2, 1, 3))(l_position)
    
    l_angle = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(24, activation="linear"))(l)
    l_angle = tf.keras.layers.Reshape((40*2, 2, -1))(l_angle)
    l_angle = tf.keras.layers.Permute((2, 1, 3))(l_angle)
    
    return tf.keras.Model([input_features, input_prev_notes], [l_timings, l_position, l_angle])
