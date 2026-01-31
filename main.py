import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import threading
import time
import sounddevice as sd
import numpy as np
import sys
import argparse
import os


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

    def __init__(
        self,
        sample_rate=16000,
        language_hint=None,
        translate=False,
        enhance_text=False,
        enhancer_model="gpt-3.5-turbo",
        enhancer_base_url=None,
    ):
        self.recorder = Recorder(sample_rate=sample_rate)
        self.transcriber = Transcriber()
        self.language_hint = language_hint
        self.translate = translate
        self.enhance_text = enhance_text
        self.enhancer_base_url = enhancer_base_url
        self.last_recording = None

        if enhance_text:
            self.text_enhancer = TextEnhancer(
                model_name=enhancer_model, base_url=enhancer_base_url
            )

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

        if self.enhance_text:
            print(f"raw result: {transcription}")
            transcription = self.text_enhancer.enhance(transcription)

        return transcription


class TextEnhancer:
    """
    Uses provided url, model_name and api_key (get from common environment variables or token files) to query llm to enchance recognized text.
    Use official openai library.
    In prompt:
        - explain that this is a text retrieved from speach recognition
        - text need to be fixed if some words are do not match context
        - remove parasite/filler words
        - remove unnecesary repetiotions in speach for example when person tries to refrase just said sentence
    """

    def __init__(self, model_name="gpt-3.5-turbo", api_key=None, base_url=None):
        import openai
        import os

        # Get API key from parameter, environment variable, or default location
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        if api_key is None:
            # Try to read from common token files
            possible_paths = [
                os.path.expanduser("~/.openai/token"),
                os.path.expanduser("~/.config/openai/token"),
                "./openai_token.txt",
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        api_key = f.read().strip()
                        break

        if api_key is None:
            raise ValueError(
                "API key not provided and not found in environment variables or token files"
            )

        # Configure OpenAI
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def enhance(self, text: str) -> str:
        """
        Enhance the provided text using an LLM.

        Args:
            text: The text to enhance (typically from speech recognition)

        Returns:
            Enhanced text with corrections, removed filler words, and improved clarity
        """
        if not text.strip():
            return text

        prompt = f"""Please improve the following text that was obtained from speech recognition.
The text may contain errors due to misrecognition, filler words, or repetitions.
Please:

1. Correct any words that don't fit the context based on similar-sounding words
2. Remove filler words like 'um', 'uh', 'like', 'you know', etc.
3. Remove unnecessary repetitions where someone tries to rephrase a sentence
4. Maintain the original meaning while making the text more readable and coherent
5. Reply only with improved text
6. Preserve original language

Text to improve:
{text}"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for more consistent corrections
            reasoning_effort="low",
        )
        enhanced_text = response.choices[0].message.content.strip()
        return enhanced_text


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Speech recognition and transcription tool"
    )
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        help="Input language hint for transcription (e.g., 'en', 'no', 'de')",
    )
    parser.add_argument(
        "--translate", "-t", action="store_true", help="Enable translation to English"
    )
    parser.add_argument(
        "--enhance", "-e", action="store_true", help="Enable text enhancement using LLM"
    )
    parser.add_argument(
        "--enhancer-model",
        type=str,
        default="gpt-3.5-turbo",
        help="Model name for text enhancement (default: gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--enhancer-url",
        type=str,
        help="Base URL for the text enhancement service (optional)",
    )
    parser.add_argument(
        "--stdout-file", type=str, help="File path to redirect stdout messages"
    )
    parser.add_argument(
        "--stderr-file", type=str, help="File path to redirect stderr messages"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Redirect stdout and stderr if file paths are provided
    if args.stdout_file:
        stdout_fd = os.open(
            args.stdout_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644
        )
        os.dup2(stdout_fd, sys.stdout.fileno())
        os.close(stdout_fd)

    if args.stderr_file:
        stderr_fd = os.open(
            args.stderr_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644
        )
        os.dup2(stderr_fd, sys.stderr.fileno())
        os.close(stderr_fd)

    speech_recognizer = SpeachRecognizer(
        language_hint=args.language,
        translate=args.translate,
        enhance_text=args.enhance,
        enhancer_model=args.enhancer_model,
        enhancer_base_url=args.enhancer_url,
    )

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
