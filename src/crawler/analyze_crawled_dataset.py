import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

data_files = [
    "Dezfuli.csv",
    "Hazaragi.csv",
    "Isfahani.csv",
    "Khorasani.csv",
    "Lori.csv",
    "Semnani.csv",
    "Shirazi.csv",
    "Southern_Kurdish.csv",
    "Tonekaboni.csv",
    "Yazdi.xlsx"
]

data_dir = "data/human_annotation"

result = {
    'language': [],
    'num_samples': [],
    'human_original_check': [],
    'human_translation_check': [],
}

for file_name in data_files:
    file_path = os.path.join(data_dir, file_name)
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_name.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        print(f"Unsupported file format: {file_name}")
        continue
    
    result['language'].append(file_name.replace('.csv', '').replace('.xlsx', ''))
    result['num_samples'].append(len(df))
    human_original_check = df['human_original_check'].value_counts().to_dict()
    result['human_original_check'].append(human_original_check.get(1, 0) / len(df))
    human_translation_check = df['human_translation_check'].value_counts().to_dict()
    result['human_translation_check'].append(human_translation_check.get(1, 0) / len(df))
    
result_df = pd.DataFrame(result)
output_path = f"{config.RESULT_DIR}/search_engine/human_annotation_summary.csv"
result_df.to_csv(output_path, index=False)