import os
import sys

from serpapi import GoogleSearch
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

data_path = f'{config.DATA_DIR}/google_search.jsonl'
query = "مثال هایی از لهجه اصفهانی"

params = {
  "engine": "google",
  "q": query,
  "safeSearch": "strict",
  "first": "20",
  "count": "30",
  "api_key": "f7163a898f27062389dd911f850c881ef6604dcfc277db4144c30aa536c20fe9"
}

search = GoogleSearch(params)
results = search.get_dict()
organic_results = results["organic_results"]
with open(data_path, 'a', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False)
    f.write('\n')
