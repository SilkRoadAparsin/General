import pandas as pd
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize


test_df = pd.read_csv("data/language_detection_data.csv")
train_df = pd.read_csv("datasets/topic_modeling.csv")
hazaragi_df = pd.read_csv("data/Hazaragi_final_dataset_filtered (3).csv")
yazdi_df = pd.read_excel("data/Yazdi_final_dataset_filtered (2).xlsx")
dari_folder = "data/1763620850534-Dari Literature Corpus/"

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

dari_sentences = []

for txt_file in sorted(Path(dari_folder).glob("*.txt")):
    with txt_file.open("r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()
    if not content:
        continue

    sentences = [s.strip() for s in sent_tokenize(content) if s.strip()]
    dari_sentences.extend(sentences)
    
# Choose 500 samples from the Dari sentences to add to the training set randomly
import random
random.seed(42)
dari_samples = random.sample(dari_sentences, min(500, len(dari_sentences)))
dari_df = pd.DataFrame({
    "language": "Persian",
    "dialect": "Dari",
    "accent": "General",
    "country": "Afghanistan",
    "text": dari_samples
})

hazaragi_df = hazaragi_df[(hazaragi_df['translation'].notna()) & (hazaragi_df['translation'] != '') & 
            (hazaragi_df['human_original_check'] == 1) & (hazaragi_df['human_translation_check'] == 1)]

yazdi_df = yazdi_df[(yazdi_df['translation'].notna()) & (yazdi_df['translation'] != '') & 
            (yazdi_df['human_original_check'] == 1) & (yazdi_df['human_translation_check'] == 1)]

cols = ["language", "dialect"]
test_df[cols] = test_df[cols].replace({
    "Unclassified": "Dezfuli",
})

# 1. Extract country from parentheses
test_df["country"] = test_df["dialect"].str.extract(r"\((.*?)\)")

# 2. Remove ", Iranian" (or "Iranian") from dialect
test_df["dialect"] = test_df["dialect"].str.replace(r"\s*,?\s*Iranian", "", regex=True)

# 3. Remove "(country)" from dialect
test_df["dialect"] = test_df["dialect"].str.replace(r"\s*\(.*?\)", "", regex=True)

language_map = {
    "Persian": "Persian",
    "Pashto": "Pashto",
    "Kurdish": "Kurdish",
    "Caspian": "Caspian",
    "Semnani": "Semnani",
    "Luri": "Luri",
    "Unclassified": "Dezfuli",
}

dialect_map = {
    "Persian, Iranian": "Persian",
    "Pashto": "Pashto, Central",
    "Southern Kurdish": "Kurdish, Southern",
    "Mazandarani": "Mazandarani",
    "Semnani": "Semnani",
    "Dari, Zoroastrian": "Dari, Zoroastrian",
    "Dezfuli": "Dezfuli",
    "Bakhtiari": "Luri, Southern",
}

accent_map = {
    "General": "General",
    "Isfahani": "Isfahani",
    "Southern Kurdish": "Kurdish, Southern",
    "Tonekaboni": "Tonekaboni",
    "Semnani": "Semnani",
    "Yazdi": "Yazdi",
    "Dezfuli": "Dezfuli",
    "Shirazi": "Shirazi",
    "Chaharmahali": "Luri, Southern",
}

def fix_language(row):
    if row["dialect"] == "Dari, Zoroastrian":
        return "Dari, Zoroastrian"
    return row["language"]


def infer_country(row):
    d = str(row["dialect"])
    a = str(row["accent"])
    l = str(row["language"])

    # Afghanistan
    if "Hazaragi" in d or "Hazaragi" in a:
        return "Afghanistan"
    if d.strip() == "Dari":
        return "Afghanistan"

    # Pakistan
    if "Pashto" in l:
        return "Pakistan"

    # Iran (default for your dataset)
    return "Iran"


train_df["language"] = train_df.apply(fix_language, axis=1)
test_df["language"] = test_df.apply(fix_language, axis=1)

train_df["language"] = train_df["language"].map(language_map).fillna(train_df["language"])
train_df["dialect"] = train_df["dialect"].map(dialect_map).fillna(train_df["dialect"])
train_df["accent"] = train_df["accent"].map(accent_map).fillna(train_df["accent"])

train_df["country"] = train_df.apply(infer_country, axis=1)

test_df = test_df[["language", "dialect", "accent", "country", "text"]]
train_df = train_df[["language", "dialect", "accent", "country", "text"]]

# Add Hazaragi data to the training set
hazaragi_df["language"] = "Persian"
hazaragi_df["dialect"] = "Hazaragi"
hazaragi_df["accent"] = "Hazaragi"
hazaragi_df["country"] = "Afghanistan"
hazaragi_df["text"] = hazaragi_df["original"]
hazaragi_df = hazaragi_df[["language", "dialect", "accent", "country", "text"]]

yazdi_df["language"] = "Persian"
yazdi_df["dialect"] = "Persian"
yazdi_df["accent"] = "Yazdi"
yazdi_df["country"] = "Iran"
yazdi_df["text"] = yazdi_df["original"]
yazdi_df = yazdi_df[["language", "dialect", "accent", "country", "text"]]

train_df = pd.concat([train_df, hazaragi_df], ignore_index=True)
train_df = pd.concat([train_df, yazdi_df], ignore_index=True)
train_df = pd.concat([train_df, dari_df], ignore_index=True)

# Save the cleaned datasets
train_df.to_csv("datasets/language_detection_train.csv", index=False)
test_df.to_csv("datasets/language_detection_test.csv", index=False)