import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

language_list = [
    "Southern_Kurdish",
    "Tonekaboni",
    "Isfahani",
    "Yazdi",
    "Semnani",
    "Zorastrian",
    "Dezfuli",
    "Shirazi",
    "Kaboli",
    "Lori",
    "Khorasani",
]

def filter_final_dataset(input_file: str, output_file: str, min_word_count: int = 50):
    """
    Filters the final dataset to remove samples with fewer than min_word_count words.
    Remove redundant samples if any.
    """
    input_df = pd.read_csv(input_file)
    filtered_df = input_df[input_df['original'] != 'NO SAMPLE']
    filtered_df = filtered_df[filtered_df['original'].str.split().str.len() >= min_word_count]
    # Remove duplicates, prioritizing rows where both checks are true, then original_check only, then both false
    filtered_df = filtered_df.sort_values(
        by=['original_check', 'translation_check'],
        ascending=False
    )
    filtered_df = filtered_df.drop_duplicates(subset=['original'], keep='first')
    filtered_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    for language in language_list:
        input_file = f"{config.DATA_DIR}/{language}_extracted_samples.csv"
        output_file = f"{config.DATA_DIR}/final/{language}_final_dataset_filtered.csv"
        filter_final_dataset(input_file, output_file, min_word_count=3)