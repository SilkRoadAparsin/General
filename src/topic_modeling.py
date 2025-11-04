import sys
import os

import openai
from bertopic.representation import OpenAI
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
# from cuml.manifold import UMAP
# from cuml.cluster import HDBSCAN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config


client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
representation_model = OpenAI(client, model="gpt-4o-mini", chat=True)
topic_model = BERTopic(representation_model=representation_model)
embedding_model = SentenceTransformer("BAAI/bge-small-en")
# umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
# hdbscan_model = HDBSCAN(min_cluster_size=400, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

docs = [
    "I love programming in Python. It's such a versatile language!",
    "The weather today is sunny and warm, perfect for a walk in the park.",
    "Artificial Intelligence is transforming the world in incredible ways.",
    "I enjoy cooking new recipes and experimenting with flavors.",
    "Traveling to new countries broadens my perspective and enriches my life.",
    "The future of technology is both exciting and uncertain.",
    "Reading books allows me to explore different worlds and ideas.",
    "Music has the power to evoke deep emotions and memories.",
    "Fitness and health are essential for a balanced lifestyle.",
    "Learning new languages opens up opportunities for communication and understanding.",
    "Traveling enhances creativity and inspires new ideas.",
]

embeddings = embedding_model.encode(docs, show_progress_bar=True)


topic_model = BERTopic(

  # Sub-models
  embedding_model=embedding_model,
#   umap_model=umap_model,
#   hdbscan_model=hdbscan_model,
  representation_model=representation_model,

  # Hyperparameters
  top_n_words=10,
  verbose=True
)

# Train model
topics, probs = topic_model.fit_transform(docs, embeddings)