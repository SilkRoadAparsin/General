import json
from openai import OpenAI


def extract_keywords(client: OpenAI, user_prompt: str, language: str, dialect: str, accent: str, target_language: str) -> list[str]:
    system_prompt = (
        "You are an expert google search keyword extractor."
        "Given a text, you will extract the most relevant keywords that can be used to search for similar content on Google."
        f"The keywords should be relevant to {language} language, {dialect} dialect, {accent} accent."
        f"The keywords should also consider the target language {target_language} for potential translations."
        "Provide the keywords in a json array format."
        "Template:"
        "{'keywords': ['keyword1', 'keyword2', ...]}"
    )
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(response.choices[0].message.content)
        if type(data) is list:
            return data
        if type(data) is dict:
            if 'keywords' in data:
                return data['keywords']
            if len(data) == 1:
                return list(data.values())[0]
    except json.JSONDecodeError:
        data = extract_keywords(client, user_prompt, language, dialect, accent, target_language)
    return data


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import config

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    language = "فارسی"
    dialect = "فارسی"
    accent = "اصفهانی"
    target_language = "فارسی معیار"

    user_prompt = "چند عبارت مناسب برای جستجو در گوگل برای یافتن نمونه‌های لهجه اصفهانی ارائه دهید."

    keywords = extract_keywords(client, user_prompt, language, dialect, accent, target_language)
    print("Extracted Keywords:", keywords)