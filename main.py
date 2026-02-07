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
        "--stdout-file", type=str, help="File path to redirect stdout messages"
    )
    parser.add_argument(
        "--stderr-file", type=str, help="File path to redirect stderr messages"
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=256,
        help="Block size for audio recording (default: 256)",
    )
    parser.add_argument(
        "--blocks-to-process",
        type=int,
        default=1000,
        help="Number of blocks to process at once (default: 1000)",
    )

    return parser.parse_args()


def redirect_std_outputs(stdout_file, stderr_file):
    """Redirect stdout and stderr if file paths are provided."""
    if stdout_file:
        stdout_fd = os.open(stdout_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(stdout_fd, sys.stdout.fileno())
        os.close(stdout_fd)

    if stderr_file:
        stderr_fd = os.open(stderr_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.dup2(stderr_fd, sys.stderr.fileno())
        os.close(stderr_fd)


def main():
    args = parse_arguments()

    redirect_std_outputs(args.stdout_file, args.stderr_file)

    speech_recognizer = SpeachRecognizer(
        language_hint=args.language,
        translate=args.translate,
        blocksize=args.blocksize,
        blocks_to_process=args.blocks_to_process,
    )

    def start_listen():
        speech_recognizer.listen()

    def stop_listen():
        text = speech_recognizer.stop_listening_and_transcribe()
        print(text)

    def stop_and_quit():
        try:
            speech_recognizer.stop_listening()
            speech_recognizer.stop_continuous()
        finally:
            exit()

    def start_continuous_listen():
        def cb(text):
            print(text)

        speech_recognizer.listen_and_continuously_transcribe(cb)

    def stop_continuous():
        try:
            speech_recognizer.stop_continuous()
        finally:
            pass

    commands = {
        "start": start_listen,
        "stop": stop_listen,
        "quit": stop_and_quit,
        "exit": stop_and_quit,
        "startc": start_continuous_listen,
        "stopc": stop_continuous,
    }

    print(
        "Server started...\nEnter command (start/stop/startc/stopc/quit):",
        file=sys.stderr,
    )
    while True:
        try:
            user_input = input()
            command = commands.get(user_input)
            if command is not None:
                command()
        except KeyboardInterrupt:
            quit()
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
