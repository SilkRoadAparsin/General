import os
import sys
import json

import pandas as pd
from tqdm import tqdm
from langchain_community.document_loaders import WebBaseLoader
from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from crawler.google_crawler import search_google
from crawler.keyword_extractor import extract_keywords
from crawler.parser import extract_samples, automatic_validation


client = OpenAI(api_key=config.OPENAI_API_KEY)

language = "فارسی"
dialect = "فارسی"
accent = "اصفهانی"
target_language = "فارسی معیار"
keyword_extraction_prompt = "چند عبارت مناسب برای جستجو در گوگل برای یافتن نمونه‌های لهجه اصفهانی ارائه دهید."
search_result_path = f'{config.DATA_DIR}/google_search.jsonl'
data_path = f'{config.DATA_DIR}/extracted_samples.csv'

# print("Extracting keywords ...")
# keywords = extract_keywords(client, keyword_extraction_prompt, language, dialect, accent, target_language)
# print("Extracted Keywords:", keywords)

# print("Searching Google for each keyword ...")
# for keyword in keywords:
#     print(f"Searching Google for keyword: {keyword} ...")
#     search_google(keyword, search_result_path, config.SERPAPI_API_KEY)

print("Parsing search results and extracting samples ...")
links = []
with open(search_result_path, 'r', encoding='utf-8') as f:
    for line in f:
        result = json.loads(line)
        for item in result.get("organic_results", []):
            link = item.get("link")
            if link and link not in links:
                links.append(link)

if os.path.exists(data_path):
    translation_data = pd.read_csv(data_path)
else:
    translation_data = pd.DataFrame(columns=["original", "translation", "original_check", "translation_check", "source", "human_original_check", "human_translation_check"])

for link in tqdm(links):
    if link in translation_data['source'].values:
        print(f"Skipping already processed link: {link}")
        continue
    print(f"Processing link: {link} ...")
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
        
        if samples == []:
            samples = [{"original": "NO SAMPLE", "translation": "NO SAMPLE"}]
            samples = automatic_validation(samples, doc.page_content)
            samples_df = pd.DataFrame(samples)
            samples_df['source'] = link
            translation_data = pd.concat([translation_data, samples_df], ignore_index=True)
    translation_data.to_csv(data_path, index=False)