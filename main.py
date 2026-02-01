import sys
import argparse
import os
from speech_recognition.speech_recognizer import SpeachRecognizer


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
