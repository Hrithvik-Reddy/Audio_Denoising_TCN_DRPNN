import os
from scipy.io.wavfile import write
import numpy as np

def set_output_directory(folder_path):
    """Set the directory to save processed audio files."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def save_processed_audio(processed_audio, sample_rate, folder_path, filename="output.wav"):
    """Save processed audio to a specified directory."""
    file_path = os.path.join(folder_path, filename)
    processed_audio = (processed_audio * 32767).astype(np.int16)  # Convert to int16 for WAV
    write(file_path, sample_rate, processed_audio)
    print(f"Processed audio saved to: {file_path}")
