import sys
import os
import re

import openai
from bertopic.representation import OpenAI
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

import datamapplot
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import config
from dataset.western_iranian_southwestern.persian.snappfood import download_snappfood_dataset
from topic_modeling.utils import translate_text


client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
representation_model = OpenAI(client, model="gpt-4o-mini", chat=True)
topic_model = BERTopic(representation_model=representation_model)
embedding_model = SentenceTransformer("BAAI/bge-m3")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=20, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

data = download_snappfood_dataset()
docs = list(data['text'])

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
    nr_topics=20,
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


result_dir = f"{config.RESULT_DIR}/topic_modeling/western_iranian_southwestern/persian/persian_Iranian_iran"
output_path = f"{result_dir}/SnappFood.png"

# image may be (Figure, Axes) or just a Figure
fig = image[0] if isinstance(image, (tuple, list)) else image
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved plot to {output_path}")