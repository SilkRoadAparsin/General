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
from langchain_text_splitters import RecursiveCharacterTextSplitter


client = OpenAI(api_key=config.OPENAI_API_KEY)

languages_info = {
    "Yazdi": {
        "language": "فارسی",
        "dialect": "یزدی",
        "accent": "یزدی",
        "target_language": "فارسی معیار",
        "keyword_extraction_prompt": "چند عبارت مناسب برای جستجو در گوگل برای یافتن نمونه‌های لهجه یا گویش یزدی ارائه دهید.  ",
    },
    "Semnani": {
        "language": "فارسی",
        "dialect": "سمنانی",
        "accent": "سمنانی",
        "target_language": "فارسی معیار",
        "keyword_extraction_prompt": "چند عبارت مناسب برای جستجو در گوگل برای یافتن نمونه‌های لهجه یا گویش سمنانی ارائه دهید.  ",
    },
    "Zorastrian": {
        "language": "فارسی",
        "dialect": "زرتشتی",
        "accent": "زرتشتی",
        "target_language": "فارسی معیار",
        "keyword_extraction_prompt": "چند عبارت مناسب برای جستجو در گوگل برای یافتن نمونه‌های لهجه یا گویش زرتشتی ارائه دهید.  ",
    },
    "Dezfuli": {
        "language": "فارسی",
        "dialect": "دزفولی",
        "accent": "دزفولی",
        "target_language": "فارسی معیار",
        "keyword_extraction_prompt": "چند عبارت مناسب برای جستجو در گوگل برای یافتن نمونه‌های لهجه یا گویش دزفولی ارائه دهید.  ",
    },
    "Shirazi": {
        "language": "فارسی",
        "dialect": "شیرازی",
        "accent": "شیرازی",
        "target_language": "فارسی معیار",
        "keyword_extraction_prompt": "چند عبارت مناسب برای جستجو در گوگل برای یافتن نمونه‌های لهجه یا گویش شیرازی ارائه دهید.  ",
    },
    "Kaboli": {
        "language": "دری",
        "dialect": "دری",
        "accent": "کابلی",
        "target_language": "دری معیار",
        "keyword_extraction_prompt": "چند عبارت مناسب برای جستجو در گوگل برای یافتن نمونه‌های لهجه یا گویش دری کابلی ارائه دهید.  ",
    },
}

for selected_language in languages_info.keys():

    language = languages_info[selected_language]["language"]
    dialect = languages_info[selected_language]["dialect"]
    accent = languages_info[selected_language]["accent"]
    target_language = languages_info[selected_language]["target_language"]
    keyword_extraction_prompt = languages_info[selected_language]["keyword_extraction_prompt"]
    search_result_path = f'{config.DATA_DIR}/{selected_language}_google_search.jsonl'
    data_path = f'{config.DATA_DIR}/{selected_language}_extracted_samples.csv'

    if not os.path.exists(data_path):
        print("Extracting keywords ...")
        keywords = extract_keywords(client, keyword_extraction_prompt, language, dialect, accent, target_language)
        print("Extracted Keywords:", keywords)

        print("Searching Google for each keyword ...")
        for keyword in keywords:
            print(f"Searching Google for keyword: {keyword} ...")
            search_google(keyword, search_result_path, config.SERPAPI_API_KEY)

    print("Parsing search results and extracting samples ...")
    links = []
    with open(search_result_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            for item in result.get("organic_results", []):
                link = item.get("link")
                if link and link not in links:
                    if "youtube" in link.lower():
                        continue
                    if "www.aparat.com" in link.lower():
                        continue
                    if ".pdf" in link.lower():
                        continue
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
            # Check if document is too large and chunk if necessary
            if doc.metadata.get('token_count') is None:
                estimated_tokens = len(doc.page_content)
            else:
                estimated_tokens = doc.metadata.get('token_count')

            if estimated_tokens > 100000:
                
                # Create text splitter with overlap
                text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=100000,
                chunk_overlap=4000,
                )
                
                # Split the document into chunks
                chunks = text_splitter.split_text(doc.page_content)
                
                # Process each chunk separately
                for chunk in chunks:
                    chunk_samples = extract_samples(
                        client,
                        chunk,
                        language=language,
                        dialect=dialect,
                        accent=accent,
                        target_language=target_language
                    )
                    
                    if chunk_samples == []:
                        continue
                    
                    chunk_samples = automatic_validation(chunk_samples, chunk)
                    chunk_samples_df = pd.DataFrame(chunk_samples)
                    chunk_samples_df['source'] = link
                    translation_data = pd.concat([translation_data, chunk_samples_df], ignore_index=True)
                
                continue  # Skip the normal processing below
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