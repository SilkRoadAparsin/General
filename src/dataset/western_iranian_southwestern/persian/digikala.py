import pandas as pd
from datasets import load_dataset


def download_digikala_dataset():
    ds = load_dataset("ParsiAI/digikala-sentiment-analysis")

    data = {
        "text": [],
        "label": [],
        "mode": [],
        "metadata": []
    }

    label_map = {1: "Negative", 2: "Neutral", 3: "Positive"}

    for split in ds.keys():
        for item in ds[split]:
            data["text"].append(item["Text"])
            data["label"].append(label_map[item["Suggestion"]])
            data["mode"].append(split)
            data["metadata"].append({k: v for k, v in item.items() if k not in ["Text", "Suggestion"]})

    return pd.DataFrame(data)

if __name__ == "__main__":
    df = download_digikala_dataset()
    print(df.head())