import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import sys
import threading
import time


class Transcriber:
    def __init__(self, idle_unload_time=None):
        self.model_id = "openai/whisper-large-v3"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._pipe = None
        self._model_loaded = False
        self.idle_unload_time = idle_unload_time
        self._last_activity_time = time.time()
        self._idle_timer = None

        # Start idle monitoring if enabled
        if self.idle_unload_time is not None and self.idle_unload_time > 0:
            self._start_idle_monitor()

    def _start_idle_monitor(self):
        """Start the idle monitoring thread"""
        if self._idle_timer:
            self._idle_timer.cancel()

        self._idle_timer = threading.Timer(self.idle_unload_time, self._idle_callback)
        self._idle_timer.daemon = True
        self._idle_timer.start()

    def _idle_callback(self):
        """Callback function when idle time is reached"""
        # Check if we're still idle (no activity since the timer started)
        time_since_last_activity = time.time() - self._last_activity_time
        if time_since_last_activity >= self.idle_unload_time:
            print(f"Unloading model due to {self.idle_unload_time}s of inactivity...", file=sys.stderr)
            self.unload_model()

    def _reset_idle_timer(self):
        """Reset the idle timer when activity occurs"""
        self._last_activity_time = time.time()

        if self.idle_unload_time is not None and self.idle_unload_time > 0:
            if self._idle_timer:
                self._idle_timer.cancel()
            self._start_idle_monitor()

    def load_model(self):
        """Load the model into memory"""
        if not self._model_loaded:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.to(self.device)

            processor = AutoProcessor.from_pretrained(self.model_id)

            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            self._model_loaded = True

        # Reset idle timer when model is loaded
        self._reset_idle_timer()

    def unload_model(self):
        """Unload the model from memory"""
        if self._model_loaded:
            # Clear the pipeline and free memory
            del self._pipe
            self._pipe = None
            # Explicitly clear CUDA cache if using GPU
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            self._model_loaded = False

    def transcribe_audio(
        self, data: str | np.ndarray, language_hint=None, translate=False
    ) -> dict():
        # Ensure model is loaded before transcribing
        if not self._model_loaded:
            self.load_model()

        # Reset idle timer when transcription is performed
        self._reset_idle_timer()

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
        return result

    def reset_idle_timer(self):
        """Manually reset the idle timer"""
        self._reset_idle_timer()
