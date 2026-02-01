import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np


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
    ) -> dict():
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
        # Example of result:
        # ```
        # {'chunks': [{'text': 'First chunk text. ',
        #              'timestamp': (0.0, 7.52)},
        #             {'text': ' Second chunk text. ',
        #              'timestamp': (7.52, 17.7)},
        #             {'text': ' Other text ',
        #              'timestamp': (17.7, 26.88)}],
        #  'text': ' First chunk text. Second chunk text. Other text '
        # }
        # ```
        return result
