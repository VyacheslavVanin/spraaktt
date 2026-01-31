import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import threading
import time
import sounddevice as sd
import numpy as np
import sys


class Transcriber:
    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
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

    def transcribe_audio(
        self, data: str | np.ndarray, language_hint=None, translate=False
    ) -> str:
        options = {}
        if language_hint:
            options.update({"language": language_hint})
        if translate:
            options.update({"task": "translate"})

        result = self._pipe(
            data,
            return_timestamps=True,
            generate_kwargs=options,
        )
        return result["text"]


class RecorderAlreadyRunning(Exception):
    def __init__(self, message="Recorder is already running"):
        super().__init__(message)


class RecorderIsNotRunning(Exception):
    def __init__(self, message="Recorder is not running"):
        super().__init__(message)


class Recorder:
    """
    Class contains methods start_record() and stop_record(). On start_record starts thread that accumulate data from input.
    On stop_record() should stop thread and return accumulated data
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recording = []
        self.recording_active = False
        self.record_thread = None

    def start_record(self):
        """Start recording audio in a separate thread"""
        if self.recording_active:
            raise RecorderAlreadyRunning()

        self.recording = []
        self.recording_active = True

        def record_audio():
            def audio_callback(indata, frames, time, status):
                if self.recording_active:
                    # Append the recorded data to our buffer
                    self.recording.append(indata.copy())

            # Start the audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                callback=audio_callback,
            ):
                while self.recording_active:
                    time.sleep(0.1)  # Small sleep to prevent busy waiting

        # Start recording in a separate thread
        self.record_thread = threading.Thread(target=record_audio)
        self.record_thread.daemon = True
        self.record_thread.start()

    def stop_record_to_np_buffer(self):
        """
        Stop recording and return accumulated data as (`np.ndarray` of shape (n, ) of type `np.float32` or `np.float64`).
        """
        if not self.recording_active:
            raise RecorderIsNotRunning()

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
            elif (
                full_recording.dtype != np.float32
                and full_recording.dtype != np.float64
            ):
                full_recording = full_recording.astype(np.float32)

            return full_recording.flatten()
        else:
            return np.array([], dtype=np.float32)


class SpeachRecognizer:
    """
    Class contains methods listen() and transcribe().
    On listen starts listen and on transcribe() returns transcribed text.
    """

    def __init__(self, sample_rate=16000, language_hint=None, translate=False):
        self.recorder = Recorder(sample_rate=sample_rate)
        self.transcriber = Transcriber()
        self.language_hint = language_hint
        self.translate = translate
        self.last_recording = None

    def listen(self):
        self.recorder.start_record()

    def stop_listening(self):
        return self.recorder.stop_record_to_np_buffer()

    def stop_listening_and_transcribe(self):
        self.last_recording = self.recorder.stop_record_to_np_buffer()
        if self.last_recording is None or not self.last_recording.any():
            return ""

        transcription = self.transcriber.transcribe_audio(
            self.last_recording,
            language_hint=self.language_hint,
            translate=self.translate,
        )
        return transcription


def main():
    speech_recognizer = SpeachRecognizer()

    def start_listen():
        speech_recognizer.listen()

    def stop_listen():
        text = speech_recognizer.stop_listening_and_transcribe()
        print(text)

    def stop_and_quit():
        try:
            speech_recognizer.stop_listening()
        finally:
            exit()

    commands = {
        "start": start_listen,
        "stop": stop_listen,
        "quit": stop_and_quit,
        "exit": stop_and_quit,
    }

    print("Server started...\nEnter command (start/stop/quit):", file=sys.stderr)
    while True:
        try:
            user_input = input()
            command = commands.get(user_input)
            if command is not None:
                command()
        except KeyboardInterrupt:
            quit()
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
