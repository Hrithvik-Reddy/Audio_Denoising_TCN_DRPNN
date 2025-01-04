import tensorflow as tf
import numpy as np

def create_single_speaker_model():
    """Create a simple single-speaker denoising model."""
    inputs = tf.keras.Input(shape=(None, 1))
    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    outputs = tf.keras.layers.Conv1D(1, kernel_size=3, activation="sigmoid", padding="same")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def process_with_single_speaker_model(model, audio_chunk):
    """Process audio using the single-speaker model."""
    audio_chunk = np.expand_dims(audio_chunk, axis=0)  # Add batch dimension
    audio_chunk = np.expand_dims(audio_chunk, axis=-1)  # Add channel dimension
    processed_audio = model.predict(audio_chunk)
    return np.squeeze(processed_audio, axis=(0, -1))  # Remove batch and channel dimensions
