import pandas as pd

data_path = "/home/sadegh/SilkRoadLang/Sentiment/datasets/topic_modeling.csv"
data = pd.read_csv(data_path)
print(data['Topic'].value_counts())