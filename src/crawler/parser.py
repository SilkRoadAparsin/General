import os
import sys
import json
import pandas as pd

from langchain_community.document_loaders import WebBaseLoader
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def extract_samples(client: OpenAI, content: str, language: str, dialect: str, accent: str, target_language: str) -> list[str]:
    prompt = (
        f"Extract any samples of sentences or phrases for {language} language, {dialect} dialect, {accent} accent from the following text:\n\n{content}"
        f"Also if there is translations available in the {target_language}, include them as well in a json array format."
        "format the output as a json array of objects with 'original' and 'translation' fields."
        "data: "
        "["
        "{'original': '<original sentence>', 'translation': '<translation in target language>'}"
        "..."
        "]"
    )
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(response.choices[0].message.content)
        if type(data) is list:
            return data
        if type(data) is dict:
            if 'data' in data:
                return data['data']
            if len(data) == 1:
                return list(data.values())[0]
    except json.JSONDecodeError:
        data = extract_samples(client, content, language, dialect, accent, target_language)
    return data

def automatic_validation(samples: list[dict[str, str]], source_context: str) -> list[dict[str, str]]:
    for sample in samples:
        original_check = sample['original'] in source_context
        translation_check = bool(sample.get('translation'))
        sample['original_check'] = original_check
        sample['translation_check'] = translation_check
    return samples

if __name__ == "__main__":
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    data_path = f'{config.DATA_DIR}/google_search.jsonl'
    translation_data_path = f'{config.DATA_DIR}/extracted_samples.csv'

    language = "فارسی"
    dialect = "فارسی"
    accent = "اصفهانی"
    target_language = "فارسی معیار"

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            links = []
            for item in result.get("organic_results", []):
                link = item.get("link")
                if link:
                    links.append(link)

    if os.path.exists(translation_data_path):
        translation_data = pd.read_csv(translation_data_path)
    else:
        translation_data = pd.DataFrame(columns=["original", "translation", "original_check", "translation_check", "source"])

    for link in links:
        if link in translation_data['source'].values:
            print(f"Skipping already processed link: {link}")
            continue
        if link:
            markdown_doc = WebBaseLoader(link).load()
            for doc in markdown_doc:
                samples = extract_samples(
                    client,
                    doc.page_content,
                    language=language,
                    dialect=dialect,
                    accent=accent,
                    target_language=target_language
                )
                
                samples = automatic_validation(samples, doc.page_content)
                samples_df = pd.DataFrame(samples)
                samples_df['source'] = link
                translation_data = pd.concat([translation_data, samples_df], ignore_index=True)
                translation_data.to_csv(translation_data_path, index=False)
                        