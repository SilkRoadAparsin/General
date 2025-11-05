import os
import sys
from openai import OpenAI


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


def translate_text(text: str, target_language: str, model: str = "gpt-4o-mini") -> str:
    """
    Translate `text` into `target_language`.
    e.g., target_language = "French", "German", "Dutch", etc.
    """
    prompt = f"Translate the following text into {target_language}:\n\n\"\"\"\n{text}\n\"\"\"\n"
    response = client.responses.create(
        model=model,
        input=[
            {
            "role": "system",
            "content": [
                {
                "type": "input_text",
                "text": "You are a helpful translator. The output should only contain the translated text."
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "input_text",
                "text": prompt
                }
            ]
            }
        ],
        temperature=1,
        max_output_tokens=256,

    )
    translation = response.output[0].content[0].text.strip()
    return translation

if __name__ == "__main__":
    original = "Hello, how are you today?"
    target = "Dutch"  # change to desired language
    result = translate_text(original, target)
    print(f"Original: {original}")
    print(f"Translated into {target}: {result}")
