# SpraakTT - Speech Recognition and Transcription Tool

A real-time speech recognition and transcription tool built with Whisper (OpenAI) and PyTorch. This tool allows you to record audio, transcribe it to text, and optionally translate it to English.

## Features

- Real-time audio recording and transcription
- Support for multiple languages with language hints
- Translation to English option
- Text enhancement using LLM to improve transcription quality
- Command-line interface for easy interaction
- File redirection for stdout/stderr

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd spraaktt
   ```

2. Install dependencies using uv (recommended):
   ```bash
   uv sync
   ```
   
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

The project relies on the following key packages:
- `torch` - Deep learning framework
- `transformers` - Hugging Face transformers library
- `sounddevice` - Audio input/output
- `numpy` - Numerical computing
- `scipy` - Scientific computing

## Usage

Run the application with the following command:

```bash
uv python main.py [options]
```

## Text Enhancement Feature

The text enhancement feature uses a Large Language Model (LLM) to improve the quality of transcribed text by:

- Correcting words that don't fit the context based on similar-sounding words
- Removing filler words like 'um', 'uh', 'like', 'you know', etc.
- Eliminating unnecessary repetitions where someone tries to rephrase a sentence
- Maintaining the original meaning while making the text more readable and coherent
- Preserving the original language

To use this feature, enable the `--enhance` flag and configure the LLM service with the appropriate model and URL.

### Options

- `-l, --language <lang>`: Specify the input language hint (e.g., 'en', 'no', 'de')
- `-t, --translate`: Enable translation to English
- `-e, --enhance`: Enable text enhancement using LLM to improve transcription quality
- `--enhancer-model <model>`: Model name for text enhancement (default: gpt-3.5-turbo)
- `--enhancer-url <url>`: Base URL for the text enhancement service (optional)
- `--stdout-file <path>`: Redirect stdout to a file
- `--stderr-file <path>`: Redirect stderr to a file

### Interactive Commands

Once the application is running, you can use the following commands:

- `start`: Begin recording audio
- `stop`: Stop recording and transcribe the audio
- `quit` or `exit`: Exit the application

### Example Usage

```bash
# Basic usage with Norwegian language
python main.py --language no

# Translate speech to English
python main.py --language de --translate

# Enable text enhancement using external LLM service
python main.py --language en --enhance --enhancer-model gpt-oss-20b-UD-Q4_K_XL --enhancer-url "http://localhost:11434/v1"

# Redirect output to files
python main.py --language en --stdout-file output.txt --stderr-file errors.txt
```
