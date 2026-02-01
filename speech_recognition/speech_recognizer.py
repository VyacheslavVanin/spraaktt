from .recorder import Recorder
from .transcriber import Transcriber
from .text_enhancer import TextEnhancer


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
