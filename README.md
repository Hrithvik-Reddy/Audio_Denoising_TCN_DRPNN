# Audio_Denoising_TCN_DRPNN
Audio Denoising For Single Speaker and Multi Speaker Environments Using TCN and DRPNN.
A Python-based real-time audio processing system that captures live audio streams from a microphone, applies machine learning models for noise reduction and speaker separation, and saves the processed audio to .wav files.
Features
-> Real-Time Audio Streaming
  Capture live audio using the sounddevice library in real-time.
  Process audio in chunks (e.g., 200 ms) to ensure low-latency performance.
-> Single-Speaker Denoising
  A TensorFlow-based neural network model to reduce noise from audio input.
  Ideal for enhancing the clarity of speech in noisy environments.
-> Multi-Speaker Separation
  TensorFlow-based multi-speaker separation model to isolate individual speakers from mixed audio streams.
  Supports real-time scenarios with two or more overlapping speakers.
-> Processed Audio Output
  Saves processed audio to .wav files.
  For single-speaker mode, saves the denoised audio as single_speaker_output.wav.
  For multi-speaker mode, saves each speaker's audio as multi_speaker_output_1.wav, multi_speaker_output_2.wav, etc.

project/
├── main.py                  # Main script for real-time audio processing
├── single_speaker_model.py  # TensorFlow model for single-speaker denoising
├── multi_speaker_model.py   # TensorFlow model for multi-speaker separation
├── audio_callback.py        # Audio streaming and processing logic
├── audio_utils.py           # Helper functions for saving audio files
├── preprocessing.py         # Preprocessing utilities (e.g., STFT, filterbanks)
└── processed_audio_files/   # Folder where processed audio files are saved
