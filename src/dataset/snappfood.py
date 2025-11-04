import pandas as pd
from datasets import load_dataset

def download_snappfood_dataset():
    ds = load_dataset("ParsiAI/snappfood-sentiment-analysis")
    return ds
