import pandas as pd
from datasets import load_dataset


def download_snappfood_dataset():
    ds = load_dataset("ParsiAI/snappfood-sentiment-analysis")

    data = {
        "text": [],
        "label": [],
        "mode": [],
        "metadata": []
    }

    for split in ds.keys():
        for item in ds[split]:
            data["text"].append(item["comment"])
            data["label"].append(item["label"])
            data["mode"].append(split)
            data["metadata"].append({k: v for k, v in item.items() if k not in ["comment", "label"]})

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = download_snappfood_dataset()
    print(df.head())
