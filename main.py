import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import threading
import time
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np


def play_wavfile(path):
    import sounddevice as sd
    from scipy.io.wavfile import read
    fs, data = read(path)
    sd.play(data, fs)
    sd.wait()


class Transcriber:
    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

    def transcribe_audio(self, file_name) -> str:
        result = self._pipe(
            file_name,
            return_timestamps=True,
            # generate_kwargs={"language": "russian", "task": "translate"}
        )
        return result["text"]


class Recorder:
    '''
    Class contains methods start_record() and stop_record(). On start_record starts thread that accumulate data from input.
    On stop_record() should stop thread and return accumulated data
    '''
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recording = []
        self.recording_active = False
        self.record_thread = None

    def start_record(self):
        """Start recording audio in a separate thread"""
        if self.recording_active:
            print("Already recording!")
            return

        self.recording = []
        self.recording_active = True

        def record_audio():
            def audio_callback(indata, frames, time, status):
                if self.recording_active:
                    # Append the recorded data to our buffer
                    self.recording.append(indata.copy())

            # Start the audio stream
            with sd.InputStream(samplerate=self.sample_rate,
                              channels=1,
                              dtype='int16',
                              callback=audio_callback):
                while self.recording_active:
                    time.sleep(0.1)  # Small sleep to prevent busy waiting

        # Start recording in a separate thread
        self.record_thread = threading.Thread(target=record_audio)
        self.record_thread.daemon = True
        self.record_thread.start()

    def stop_record_to_np_buffer(self):
        '''
        Stop recording and return accumulated data as (`np.ndarray` of shape (n, ) of type `np.float32` or `np.float64`).
        '''
        if not self.recording_active:
            print("Not currently recording!")
            return np.array([], dtype=np.float32)

        self.recording_active = False

        # Wait for the recording thread to finish
        if self.record_thread:
            self.record_thread.join()

        # Concatenate all recorded chunks
        if self.recording:
            full_recording = np.concatenate(self.recording, axis=0)

            # Convert to float32 and normalize if needed
            if full_recording.dtype == np.int16:
                full_recording = full_recording.astype(np.float32) / 32768.0
            elif full_recording.dtype != np.float32 and full_recording.dtype != np.float64:
                full_recording = full_recording.astype(np.float32)

            return full_recording.flatten()
        else:
            print("No audio data recorded")
            return np.array([], dtype=np.float32)

    def stop_record_to_file(self, filename="record.wav"):
        """Stop recording and save the accumulated data to a file"""
        if not self.recording_active:
            print("Not currently recording!")
            return None

        self.recording_active = False

        # Wait for the recording thread to finish
        if self.record_thread:
            self.record_thread.join()

        # Concatenate all recorded chunks
        if self.recording:
            full_recording = np.concatenate(self.recording, axis=0)

            # Write to WAV file
            from scipy.io.wavfile import write
            write(filename, self.sample_rate, full_recording)
            print(f"Recording saved to {filename}")
            return filename
        else:
            print("No audio data recorded")
            return None


def main():
    from pynput import keyboard

    # Initialize the recorder
    recorder = Recorder()
    stt = Transcriber()

    print("Press and hold 'PageDown' to record audio. Release to transcribe.")
    print("Press 'Esc' to exit the program.")

    def on_press(key):
        try:
            if key == keyboard.Key.page_down and not recorder.recording_active:
                print("\nStarted recording... Release PageDown to transcribe.")

                # Start recording
                recorder.start_record()
        except AttributeError:
            # Special keys (like PageDown) are handled differently
            pass

    def on_release(key):
        if key == keyboard.Key.page_down and recorder.recording_active:
            print("\nStopped recording. Processing transcription...")

            # Stop recording and get the filename
            file_name = recorder.stop_record_to_np_buffer()

            if file_name.any():
                print("\nTranscribing...")
                transcription = stt.transcribe_audio(file_name)
                print(f"\nTranscription: {transcription}")
                print("\nPress and hold 'PageDown' to record again.")

        if key == keyboard.Key.esc:
            print("Exiting...")
            if recorder.recording_active:
                recorder.stop_record_to_file()  # Stop any ongoing recording
            return False  # Stop listener

    # Start the keyboard listener
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
