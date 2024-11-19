from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import os
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from IPython.core.display import display, HTML
from collections import Counter
nltk.download('stopwords')

# Additional filler words to remove
CUSTOM_STOPWORDS = set([
    "would", "could", "should", "also", "many", "may", "much", "one", "two", "three", "four", "five", "good",
    "like", "however", "therefore", "thus", "make", "made", "need", "use", "new", "time", "include", "provided"
])
STOPWORDS = set(stopwords.words('english')) | CUSTOM_STOPWORDS

def preprocess_text(text: str) -> str:
    """Cleans text by removing special characters, numbers, and extra spaces, and custom stopwords."""
    text = re.sub(r"http\S+", "", text)  # Remove links
    text = re.sub("[^A-Za-z]+", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = text.strip().lower()
    return " ".join(word for word in text.split() if word not in STOPWORDS and len(word) > 2)  # Keep words > 2 chars

def load_and_preprocess_data(folder_path: str) -> list:
    """Loads and preprocesses documents."""
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                documents.append(preprocess_text(text))
        except (UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Error reading file {file_name}: {e}")
    return documents

def compute_topic_coherence(topics, documents):
    """Computes topic coherence using NPMI."""
    tokenized_docs = [doc.split() for doc in documents]
    dictionary = Dictionary(tokenized_docs)
    coherence_model = CoherenceModel(
        topics=topics,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_npmi'
    )
    return coherence_model.get_coherence()

def extract_keywords_bertopic(documents, n_topics: int = 10, n_words: int = 10):
    """Performs topic modeling and extracts top keywords for each topic."""
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.1, metric="cosine")

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model
    )

    # Fit Bertopic
    topics, probs = topic_model.fit_transform(documents)

    # Extract topics and their keywords
    topic_info = topic_model.get_topic_info()
    keywords = {}
    topic_words_cleaned = []

    for topic_id in range(min(len(topic_info), n_topics)):
        if topic_id == -1:  # Skip outliers
            continue
        topic_keywords = topic_model.get_topic(topic_id)
        if not topic_keywords or isinstance(topic_keywords, bool):
            print(f"Topic {topic_id} has no valid keywords.")
            continue
        filtered_keywords = [word for word, _ in topic_keywords if word not in CUSTOM_STOPWORDS][:n_words]
        keywords[topic_id] = filtered_keywords
        topic_words_cleaned.append(filtered_keywords)

    return keywords, topic_model, topic_words_cleaned

def run_topic_modeling(folder_path: str, period_label: str):
    """Runs BERTopic for a specific time period, validates coherence, and saves the visualization."""
    documents = load_and_preprocess_data(folder_path)

    if len(documents) < 10:
        raise ValueError(f"Not enough documents for topic modeling in period {period_label}. Increase the dataset size.")

    keywords, model, topic_words_cleaned = extract_keywords_bertopic(documents, n_topics=10, n_words=10)

    print(f"Results for {period_label}:")
    for topic, words in keywords.items():
        print(f"Topic {topic}: {', '.join(words)}")

    # Validate topic coherence
    coherence_score = compute_topic_coherence(topic_words_cleaned, documents)
    print(f"Coherence Score for {period_label}: {coherence_score:.4f}")

    # Visualizations
    try:
        fig = model.visualize_barchart(top_n_topics=5)
        fig_file = f"barchart_{period_label}.html"
        fig.write_html(fig_file)
        print(f"Bar chart for {period_label} saved as {fig_file}")
    except Exception as e:
        print(f"Bar chart visualization failed for {period_label}: {e}")

    try:
        umap_fig = model.visualize_documents(documents)
        umap_file = f"umap_{period_label}.html"
        umap_fig.write_html(umap_file)
        print(f"UMAP plot for {period_label} saved as {umap_file}")
    except Exception as e:
        print(f"UMAP plot generation failed for {period_label}: {e}")

if __name__ == "__main__":

    run_topic_modeling("documents/pre_SVB", "pre_SVB")

    run_topic_modeling("documents/post_SVB", "post_SVB")
