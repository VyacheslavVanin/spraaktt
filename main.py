import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def record_microfone():
    '''
    Read audio from default system microfone and write it to record.wav file.
    returns recorded file name
    '''
    import sounddevice as sd
    from scipy.io.wavfile import write
    fs = 16000
    seconds = 5
    filename = "record.wav"
    print("Recording...")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    print("Finish Recording...")
    
    return filename


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


stt = Transcriber()
while True:
    print('Start record...')
    file_name = record_microfone()
    print('Finish record...')
    print(stt.transcribe_audio(file_name))
