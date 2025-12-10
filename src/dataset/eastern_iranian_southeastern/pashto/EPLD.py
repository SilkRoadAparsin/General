# English-Pashto Language Dataset (EPLD) Download Script

import requests
import os
import pandas as pd

url = "https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/4ffa173e-22db-4ce7-937c-2fe9b4785a99"

def download_epld_dataset():
    df = pd.read_excel(url)

    data = {
        "text": [],
        "label": [],
        "mode": [],
        "metadata": []
    }

    for index, row in df.iterrows():
        data["text"].append(row["Unnamed: 3"])
        data["label"].append(pd.NA)
        data["mode"].append(pd.NA)
        data["metadata"].append({k: v for k, v in row.items() if k in ["English", "Pashto"]})

    return pd.DataFrame(data)


if __name__ == "__main__":
    df = download_epld_dataset()
    print(df.head())