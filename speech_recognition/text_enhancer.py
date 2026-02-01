import openai
import os


class TextEnhancer:
    """
    Uses provided url, model_name and api_key (get from common environment variables or token files) to query llm to enchance recognized text.
    Use official openai library.
    In prompt:
        - explain that this is a text retrieved from speach recognition
        - text need to be fixed if some words are do not match context
        - remove parasite/filler words
        - remove unnecesary repetiotions in speach for example when person tries to refrase just said sentence
    """

    def __init__(self, model_name="gpt-3.5-turbo", api_key=None, base_url=None):
        # Get API key from parameter, environment variable, or default location
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        if api_key is None:
            # Try to read from common token files
            possible_paths = [
                os.path.expanduser("~/.openai/token"),
                os.path.expanduser("~/.config/openai/token"),
                "./openai_token.txt",
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, "r") as f:
                        api_key = f.read().strip()
                        break

        if api_key is None:
            raise ValueError(
                "API key not provided and not found in environment variables or token files"
            )

        # Configure OpenAI
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def enhance(self, text: str) -> str:
        """
        Enhance the provided text using an LLM.

        Args:
            text: The text to enhance (typically from speech recognition)

        Returns:
            Enhanced text with corrections, removed filler words, and improved clarity
        """
        if not text.strip():
            return text

        prompt = f"""Please improve the following text that was obtained from speech recognition.
The text may contain errors due to misrecognition, filler words, or repetitions.
Please:

1. Correct any words that don't fit the context based on similar-sounding words
2. Remove filler words like 'um', 'uh', 'like', 'you know', etc.
3. Remove unnecessary repetitions where someone tries to rephrase a sentence
4. Maintain the original meaning while making the text more readable and coherent
5. Reply only with improved text
6. Preserve original language

Text to improve:
{text}"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for more consistent corrections
            reasoning_effort="low",
        )
        enhanced_text = response.choices[0].message.content.strip()
        return enhanced_text
