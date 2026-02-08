from .recorder import Recorder
from .transcriber import Transcriber


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
        blocksize=256,
        blocks_to_process=1000,
        idle_unload_time=None,
    ):
        self.recorder = Recorder(
            sample_rate=sample_rate,
            blocksize=blocksize,
            blocks_to_process=blocks_to_process,
        )
        self.transcriber = Transcriber(idle_unload_time=idle_unload_time)
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
        )["text"]

        return transcription

    def listen_and_continuously_transcribe(self, user_callback):
        """
        def user_callback(text: str)
        """

        def callback(data):
            if data is None or not data.any():
                return ""

            transcription = self.transcriber.transcribe_audio(
                data,
                language_hint=self.language_hint,
                translate=self.translate,
            )["text"]

            user_callback(transcription)

        self.recorder.start_continuous_record(callback)

    def stop_continuous(self):
        self.recorder.stop_continuous()
