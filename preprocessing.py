import numpy as np
from scipy.signal import stft, istft
import librosa

SAMPLE_RATE = 16000

def compute_stft(audio, n_fft=512, hop_length=256):
    """Compute the Short-Time Fourier Transform (STFT)"""
    _, _, spectrogram = stft(audio, nperseg=n_fft, noverlap=n_fft - hop_length)
    return spectrogram

def reconstruct_audio(magnitude, phase, n_fft=512, hop_length=256):
    """Reconstruct the audio signal from magnitude and phase spectrogram"""
    complex_spectrum = magnitude * np.exp(1j * phase)
    _, audio = istft(complex_spectrum, nperseg=n_fft, noverlap=n_fft - hop_length)
    return audio

def mel_filterbank(audio, sample_rate=SAMPLE_RATE, n_mels=128):
    """Convert the audio to a Mel spectrogram"""
    mel_spectrogram = librosa.feature.melspectrogram(audio, sr=sample_rate, n_mels=n_mels)
    return librosa.power_to_db(mel_spectrogram, ref=np.max)
