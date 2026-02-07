# SpraakTT - Speech Recognition and Transcription Tool

A real-time speech recognition and transcription tool built with Whisper (OpenAI) and PyTorch. This tool allows you to record audio, transcribe it to text, and optionally translate it to English.

## Features

- Real-time audio recording and transcription
- Support for multiple languages with language hints
- Translation to English option
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
uv run main.py [options]
```

### Options

- `-l, --language <lang>`: Specify the input language hint (e.g., 'en', 'no', 'de')
- `-t, --translate`: Enable translation to English
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
uv run main.py --language no

# Translate speech to English
uv run main.py --language de --translate

# Redirect output to files
uv run main.py --language en --stdout-file output.txt --stderr-file errors.txt
```
