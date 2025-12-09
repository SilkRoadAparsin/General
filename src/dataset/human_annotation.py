import pandas as pd
import os

languages_info = {
    "Yazdi": {
        "language": "Persian",
        "dialect": "Persian, Iranian",
        "accent": "Yazdi",
        "target_language": "Standard Persian",
    },
    "Semnani": {
        "language": "Semnani",
        "dialect": "Semnani",
        "accent": "Semnani",
        "target_language": "Standard Persian",
    },
    "Zorastrian": {
        "language": "Persian",
        "dialect": "Dari, Zoroastrian",
        "accent": "Yazdi",
        "target_language": "Standard Persian",
    },
    "Dezfuli": {
        "language": "Unclassified",
        "dialect": "Dezfuli",
        "accent": "Dezfuli",
        "target_language": "Standard Persian",
    },
    "Shirazi": {
        "language": "Persian",
        "dialect": "Persian, Iranian",
        "accent": "Shirazi",
        "target_language": "Standard Persian",
    },
    "Kaboli": {
        "language": "Dari",
        "dialect": "Dari",
        "accent": "Kaboli",
        "target_language": "Standard Dari",
    },
    "Lori": {
        "language": "Luri",
        "dialect": "Bakhtiari",
        "accent": "Chaharmahali",
        "target_language": "Standard Persian",
    },
    "Khorasani": {
        "language": "Persian",
        "dialect": "Persian, Iranian",
        "accent": "Khorasani",
        "target_language": "Standard Persian",
    },
    "Pashto": {
        "language": "Pashto",
        "dialect": "Pashto", 
        "accent": "General",  
        "target_language": "Standard Persian",
    },

    "Hazaragi": {
        "language": "Dari",    
        "dialect": "Hazaragi",
        "accent": "Hazaragi",
        "target_language": "Standard Dari",
    },
}

folder = "/home/sadegh/SilkRoadLang/Sentiment/data/human_annotation"

all_dfs = []
for language, info in languages_info.items():
    file_path = f"{folder}/{language}.csv"
    if not os.path.exists(file_path):
        continue
    df = pd.read_csv(file_path)
    df.insert(0, 'language', info['language'])
    df.insert(1, 'dialect', info['dialect'])
    df.insert(2, 'accent', info['accent'])
    df.insert(3, 'target_language', info['target_language'])
    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)
output_file = f"{folder}/Human_annotated_crawled_dataset.csv"
combined_df.to_csv(output_file, index=False)