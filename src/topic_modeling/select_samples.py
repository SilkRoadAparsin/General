import pandas as pd

data_path = "/home/sadegh/SilkRoadLang/Sentiment/datasets/topic_modeling.csv"
data = pd.read_csv(data_path)
print(data['Topic'].value_counts())
print("-----")
# For each unique combination of (language, dialect, accent), select 10 samples from 10 different topics
result_dfs = []

for (lang, dialect, accent), group in data.groupby(['language', 'dialect', 'accent']):
    # Sort by source, prioritizing 'manual' labeled data
    group_sorted = group.sort_values(
        by='source',
        key=lambda x: x.map(lambda val: 0 if 'manual' in str(val).lower() else 1)
    )
    
    selected_samples = []
    topics = group_sorted['Topic'].unique()
    print(lang, dialect, accent, len(topics))
    print("Available topics:", topics)
    print("-----")
    
    # Try to get 1 sample from each of 10 different topics
    for topic in topics[:10]:
        topic_samples = group_sorted[group_sorted['Topic'] == topic]
        if len(topic_samples) > 0:
            # Take the first one (which will be manual label if available due to sorting)
            selected_samples.append(topic_samples.head(1))
    
    # If we don't have 10 samples yet, randomly select from the rest
    current_count = len(selected_samples)
    if current_count < 10:
        already_selected_ids = pd.concat(selected_samples).index if selected_samples else pd.Index([])
        remaining = group_sorted[~group_sorted.index.isin(already_selected_ids)]
        if len(remaining) > 0:
            n_needed = min(10 - current_count, len(remaining))
            selected_samples.append(remaining.head(n_needed))
    
    if selected_samples:
        result_dfs.append(pd.concat(selected_samples))

# Combine all selected samples
final_selection = pd.concat(result_dfs, ignore_index=True)
print(f"Total samples selected: {len(final_selection)}")
print(final_selection.groupby(['language', 'dialect', 'accent']).size())

output_path = "/home/sadegh/SilkRoadLang/Sentiment/datasets/100_samples.csv"
final_selection = final_selection.drop(columns=['Topic'])
final_selection.to_csv(output_path, index=False)