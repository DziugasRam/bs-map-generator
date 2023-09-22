import tensorflow as tf

def _create_audio_model():
    input_audio = tf.keras.Input(shape=(87, 129, 1), dtype="float32")
    l = input_audio
    l = tf.keras.layers.Conv2D(128, 5, activation="relu", padding="same")(l)
    l = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(l)
    l = tf.keras.layers.Conv2D(128, 3, activation="relu")(l)
    l = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(l)
    l = tf.keras.layers.Conv2D(128, 3, activation="relu")(l)
    l = tf.keras.layers.MaxPooling2D(pool_size=(1, 2))(l)
    l = tf.keras.layers.Reshape((40, -1))(l)
    l = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation="relu"))(l)
    return tf.keras.Model(input_audio, l)

def create_audio_model_stereo():
    audio_model = _create_audio_model()
    
    input_audio = tf.keras.Input(shape=(2, 87, 129, 1), dtype="float32")
    l = input_audio
    l = tf.keras.layers.TimeDistributed(audio_model)(l)
    l = tf.keras.layers.Permute((2, 1, 3))(l)
    l = tf.keras.layers.Reshape((40, -1))(l)
    l = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(l)
    l = tf.keras.layers.LSTM(128, return_sequences=True)(l)
    return tf.keras.Model(input_audio, l)
