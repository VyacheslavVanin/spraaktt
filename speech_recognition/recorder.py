import threading
import time
import sounddevice as sd
import numpy as np


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

    def __init__(self, sample_rate=16000, blocksize=256, blocks_to_process=1000):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.blocks_to_process = blocks_to_process
        self.recording = []
        self.recording_active = False
        self.record_thread = None
        self.processing_thread = None
        self.processing_data_lock = threading.Lock()
        self.processing_data = []
        self.processing_data_len = 0

    def _send_to_processing(self):
        with self.processing_data_lock:
            self.processing_data.append(self.recording)
        self.recording = []

    def start_continuous_record(self, callback):
        if self.recording_active:
            raise RecorderAlreadyRunning()

        self.recording = []
        self.recording_active = True

        def record_audio_thread():
            def audio_callback(indata, frames, time, status):
                if self.recording_active:
                    # Append the recorded data to our buffer
                    self.recording.append(indata.copy())

                    if len(self.recording) >= self.blocks_to_process:
                        self._send_to_processing()

            # Start the audio stream
            with sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                channels=1,
                dtype="int16",
                callback=audio_callback,
            ):
                while self.recording_active:
                    time.sleep(0.1)  # Small sleep to prevent busy waiting

        def processing_thread():
            while self.recording_active:
                time.sleep(0.1)
                data_to_process = None
                with self.processing_data_lock:
                    if self.processing_data != []:
                        data_to_process = self.processing_data.pop(0)
                if data_to_process:
                    callback(self._to_full_recording(data_to_process))

        self.processing_thread = threading.Thread(target=processing_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        # Start recording in a separate thread
        self.record_thread = threading.Thread(target=record_audio_thread)
        self.record_thread.daemon = True
        self.record_thread.start()

    def stop_continuous(self):
        if not self.recording_active:
            raise RecorderIsNotRunning()

        self.recording_active = False
        if self.record_thread:
            self.record_thread.join()

        self._send_to_processing()
        if self.processing_thread:
            self.processing_thread.join()

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

    def _to_full_recording(self, data):
        if data:
            full_recording = np.concatenate(data, axis=0)

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

        if self.recording:
            return self._to_full_recording(self.recording)
