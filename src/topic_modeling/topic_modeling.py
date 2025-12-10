import sys
import os
import re
import pandas as pd

import openai
from bertopic.representation import OpenAI
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

import datamapplot
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from dataset.western_iranian_southwestern.persian.snappfood import download_snappfood_dataset
from dataset.western_iranian_southwestern.persian.digikala import download_digikala_dataset
from dataset.eastern_iranian_southeastern.pashto.pashto_english_bilingual_sentiment_corpus import download_pashto_english_bilingual_sentiment_corpus_dataset
from dataset.eastern_iranian_southeastern.pashto.EPLD import download_epld_dataset
from utils import translate_text

all_data = pd.DataFrame(columns=["language", "dialect", "accent", "target_language", "text", "translation", "source", "Topic"])

data = download_snappfood_dataset().sample(1000)
data["source"] = "ParsiAI/snappfood-sentiment-analysis"
data = pd.concat([data, download_digikala_dataset().sample(1000)], ignore_index=True)
data.loc[data.index[len(data) - 1000:], "source"] = "ParsiAI/digikala-sentiment-analysis"
data["language"] = "Persian"
data["dialect"] = "Persian, Iranian"
data["accent"] = "General"
data["target_language"] = "Standard Persian"
data["translation"] = data["text"]
data = data[["language", "dialect", "accent", "target_language", "text", "translation", "source"]]
all_data = pd.concat([all_data, data], ignore_index=True)

data = download_pashto_english_bilingual_sentiment_corpus_dataset()
data["source"] = "Pashto-English Bilingual Sentiment Corpus"
data = pd.concat([data, download_epld_dataset()], ignore_index=True)
data.loc[data.index[len(data) - len(download_epld_dataset()):], "source"] = "EPLD"
data["language"] = "Pashto"
data["dialect"] = "Pashto"
data["accent"] = "General"
data["target_language"] = "Standard Pashto"
data["translation"] = data["text"]
data = data[["language", "dialect", "accent", "target_language", "text", "translation", "source"]]
all_data = pd.concat([all_data, data], ignore_index=True)

data = pd.read_csv("/home/sadegh/SilkRoadLang/Sentiment/datasets/Human_annotated_crawled_dataset.csv")
data = data[(data['translation'].notna()) & (data['translation'] != '') & 
            (data['human_original_check'] == 1) & (data['human_translation_check'] == 1)]
data["text"] = data["original"]
data = data[["language", "dialect", "accent", "target_language", "text", "translation", "source"]]
all_data = pd.concat([all_data, data], ignore_index=True)

all_data = all_data
docs = list(all_data['text'])

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
representation_model = OpenAI(client, model="gpt-4o-mini", chat=True)
topic_model = BERTopic(representation_model=representation_model)
embedding_model = SentenceTransformer("BAAI/bge-m3")
umap_model = UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

embeddings = embedding_model.encode(docs, show_progress_bar=True)
reduced_embeddings = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)


topic_model = BERTopic(
    # Sub-models
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    representation_model=representation_model,

    # Hyperparameters
    top_n_words=10,
    verbose=True,
    nr_topics=10,
)

# Train model
topics, probs = topic_model.fit_transform(docs, embeddings)



# Create a label for each document
llm_labels = [re.sub(r'\W+', ' ', label[0][0].split("\n")[0].replace('"', '')) for label in topic_model.get_topics(full=True)["Main"].values()]
print(llm_labels)
translated_labels = [translate_text(label, target_language="English") for label in llm_labels if label]
print(translated_labels)
all_labels = [translated_labels[topic+topic_model._outliers] if topic != -1 else "Unlabelled" for topic in topics]

# Run the visualization
image = datamapplot.create_plot(
    reduced_embeddings,
    all_labels,
    label_font_size=11,
    title="SnappFood Topic Modeling",
    sub_title="Topics labeled with GPT-4o-Mini, BGE-M3 embeddings, UMAP & HDBSCAN",
    label_wrap_width=20,
    use_medoids=True,
    logo_width=0.16,
)

output_path = f"{config.RESULT_DIR}/topic_modeling/topic_modeling.png"
df_output_path = "datasets/topic_modeling.csv"
all_data["Topic"] = all_labels
all_data.to_csv(df_output_path, index=False)
print(f"Saved data to {df_output_path}")

# image may be (Figure, Axes) or just a Figure
fig = image[0] if isinstance(image, (tuple, list)) else image
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved plot to {output_path}")