import tensorflow as tf
import numpy as np

def create_multi_speaker_model():
    """Create a multi-speaker separation model."""
    inputs = tf.keras.Input(shape=(None, 1))
    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    outputs = tf.keras.layers.Conv1D(2, kernel_size=3, activation="sigmoid", padding="same")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def process_with_multi_speaker_model(model, audio_chunk):
    """Process audio chunk using the multi-speaker model."""
    try:
        audio_chunk = np.expand_dims(audio_chunk, axis=0)  # Add batch dimension
        audio_chunk = np.expand_dims(audio_chunk, axis=-1)  # Add channel dimension
        separated_audio = model.predict(audio_chunk)  # Shape: [Batch, Time, Speakers]
        separated_chunks = [
            separated_audio[0, :, i] for i in range(separated_audio.shape[-1])
        ]
        return separated_chunks
    except Exception as e:
        print(f"Error during multi-speaker model inference: {e}")
        return []
