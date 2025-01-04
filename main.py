import numpy as np
import sounddevice as sd
from single_speaker_model import create_single_speaker_model, process_with_single_speaker_model
from multi_speaker_model import create_multi_speaker_model, process_with_multi_speaker_model
from audio_utils import save_processed_audio, set_output_directory

# Configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.2  # 200 ms
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)  # Number of samples per chunk
processed_audio_buffer = []  # Buffer to store processed audio
MODE = "multi"  # Choose between "single" or "multi"

# Load Models
single_speaker_model = create_single_speaker_model()
multi_speaker_model = create_multi_speaker_model()

# Callback Function for Real-Time Audio Processing
def audio_callback(in_data, frames, time, status):
    global processed_audio_buffer
    if status:
        print(f"Audio Status: {status}")

    # Convert input audio to numpy array
    audio_chunk = np.frombuffer(in_data, dtype=np.float32)

    if MODE == "single":
        # Process audio with the single-speaker model
        processed_chunk = process_with_single_speaker_model(single_speaker_model, audio_chunk)
        if processed_chunk.ndim == 1 and len(processed_chunk) > 0:  # Validate output
            processed_audio_buffer.append(processed_chunk)
        else:
            print("Invalid single-speaker model output. Skipping this chunk.")
        return processed_chunk.tobytes()

    elif MODE == "multi":
        # Process audio with the multi-speaker model
        separated_chunks = process_with_multi_speaker_model(multi_speaker_model, audio_chunk)
        if separated_chunks:
            if not processed_audio_buffer:  # Initialize buffers if needed
                processed_audio_buffer.extend([[] for _ in range(len(separated_chunks))])

            # Validate and append each speaker's audio
            for i, chunk in enumerate(separated_chunks):
                if chunk.ndim == 1 and len(chunk) > 0:
                    processed_audio_buffer[i].append(chunk)
                else:
                    print(f"Invalid or empty chunk for speaker {i + 1}. Skipping.")
            return separated_chunks[0].tobytes()  # Play back the first speaker's audio
        else:
            print("Invalid or empty separated_chunks. Skipping.")
            return audio_chunk.tobytes()

# Main Function for Real-Time Audio Streaming
def run_real_time_processing():
    global processed_audio_buffer

    # Set output directory
    output_folder = set_output_directory("processed_audio_files")

    # Start the audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,  # Mono audio
        dtype="float32",
        blocksize=CHUNK_SIZE,
        callback=audio_callback,
    )

    print(f"Starting audio stream in {MODE}-speaker mode. Speak into your microphone.")
    print("Press Ctrl+C to stop.")
    stream.start()

    try:
        while stream.active:
            pass  # Keep the stream running
    except KeyboardInterrupt:
        print("\nStopping audio stream...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'stream' in locals() and stream.active:
            stream.stop()
            stream.close()

        # Save processed audio
        if MODE == "single":
            if len(processed_audio_buffer) > 0:
                # Concatenate and save single-speaker processed audio
                processed_audio = np.concatenate(processed_audio_buffer)
                save_processed_audio(processed_audio, SAMPLE_RATE, output_folder, filename="single_speaker_output.wav")
            else:
                print("No audio data recorded in single-speaker mode. Skipping save.")
        elif MODE == "multi":
            if len(processed_audio_buffer) > 0:
                for i, speaker_audio in enumerate(processed_audio_buffer):
                    if len(speaker_audio) > 0:  # Check if speaker buffer has data
                        try:
                            print(f"Concatenating audio for speaker {i + 1}...")
                            processed_audio = np.concatenate(speaker_audio)
                            save_processed_audio(
                                processed_audio, SAMPLE_RATE, output_folder, filename=f"multi_speaker_output_{i + 1}.wav"
                            )
                        except ValueError as e:
                            print(f"Error concatenating audio for speaker {i + 1}: {e}")
                    else:
                        print(f"No audio data recorded for speaker {i + 1}. Skipping save.")
            else:
                print("No audio data recorded in multi-speaker mode. Skipping save.")


if __name__ == "__main__":
    run_real_time_processing()
