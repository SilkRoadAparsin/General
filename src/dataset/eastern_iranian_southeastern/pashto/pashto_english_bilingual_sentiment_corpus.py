import pandas as pd
from kaggle import api


def get_mode(index: int, total: int) -> str:
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)
    
    if index < train_end:
        return "train"
    elif index < val_end:
        return "validation"
    else:
        return "test"
    

def get_label(row) -> str:
    if row["AnnotatorOne"] == row["AnnotatorTwo"]:
        return row["AnnotatorOne"]
    else:
        return "Neutral"


def download_pashto_english_bilingual_sentiment_corpus_dataset(cache_dir: str = "data/topic_modeling/") -> pd.DataFrame:
    file_path = f'{cache_dir}/PashtoCorpusUpdated.csv'
    api.dataset_download_files('farhadkhan66/pashto-translated-corpus', path=cache_dir, unzip=True)

    df = pd.read_csv(file_path)

    data = {
        "text": [],
        "label": [],
        "mode": [],
        "metadata": []
    }

    for index, row in df.iterrows():
        data["text"].append(row["PashtoText"])
        data["label"].append(get_label(row))
        data["mode"].append(get_mode(index, len(df)))
        data["metadata"].append({k: v for k, v in row.items() if k not in ["PashtoText", "Id"]})

    return pd.DataFrame(data)

if __name__ == "__main__":
    file_dir = "data/topic_modeling/"
    df = download_pashto_english_bilingual_sentiment_corpus_dataset(file_dir)
    print(df.head())