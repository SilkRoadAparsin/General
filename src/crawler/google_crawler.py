from serpapi import GoogleSearch
import json


def search_google(query: str, data_path: str, api_key: str):
  params = {
    "engine": "google",
    "q": query,
    "safeSearch": "strict",
    "first": "20",
    "count": "30",
    "api_key": api_key
  }

  search = GoogleSearch(params)
  results = search.get_dict()
  with open(data_path, 'a', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False)
    f.write('\n')
  return results


if __name__ == "__main__":
  import os
  import sys
  sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
  import config

  query = "مثال هایی از لهجه اصفهانی"
  data_path = f'{config.DATA_DIR}/google_search.jsonl'
  search_google(query, data_path, config.SERPAPI_API_KEY)