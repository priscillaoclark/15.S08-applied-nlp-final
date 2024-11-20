from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
import os
import re
from nltk.corpus import stopwords
import nltk
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from hdbscan import HDBSCAN

nltk.download('stopwords')


# Additional filler words to remove
CUSTOM_STOPWORDS = set([
    "would", "could", "should", "also", "many", "may", "much", "one", "two", "three", "four", "five", "good",
    "like", "however", "therefore", "thus", "make", "made", "need", "use", "new", "time", "include", "provided",
    "information", "section", "data", "proposed", "rule", "final", "notification", "order"
])
STOPWORDS = set(stopwords.words('english')) | CUSTOM_STOPWORDS

def preprocess_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)  # Remove links
    text = re.sub("[^A-Za-z]+", " ", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = text.strip().lower()
    return " ".join(word for word in text.split() if word not in STOPWORDS and len(word) > 2)  # Keep words > 2 chars

def load_and_preprocess_data(folder_path: str) -> list:
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
    tokenized_docs = [doc.split() for doc in documents]
    dictionary = Dictionary(tokenized_docs)
    coherence_model = CoherenceModel(
        topics=topics,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_npmi'
    )
    return coherence_model.get_coherence()

def extract_keywords_bertopic(documents, n_topics: int = 10, n_words: int = 10, regularized: bool = False):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(
        n_neighbors=10,
        n_components=5,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )

    # Explicit HDBSCAN model for control
    hdbscan_model = HDBSCAN(
        min_cluster_size=20 if regularized else 5,
        min_samples=5,
        cluster_selection_epsilon=0.01
    )

    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=20 if regularized else None,
        nr_topics="auto" if regularized else None
    )

    topics, probs = topic_model.fit_transform(documents)

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
    documents = load_and_preprocess_data(folder_path)

    if len(documents) < 10:
        raise ValueError(f"Not enough documents for topic modeling in period {period_label}. Increase the dataset size.")

    # Standard BERTopic Model
    keywords, model, topic_words_cleaned = extract_keywords_bertopic(documents, n_topics=10, n_words=10, regularized=False)

    # Regularized BERTopic Model (using heuristic)
    reg_keywords, reg_model, reg_topic_words_cleaned = extract_keywords_bertopic(documents, n_topics=10, n_words=10, regularized=True)

    print(f"Results for {period_label} - Standard:")
    for topic, words in keywords.items():
        print(f"Topic {topic}: {', '.join(words)}")

    print(f"Results for {period_label} - Regularized:")
    for topic, words in reg_keywords.items():
        print(f"Topic {topic}: {', '.join(words)}")

    # Coherence Scores
    coherence_score = compute_topic_coherence(topic_words_cleaned, documents)
    reg_coherence_score = compute_topic_coherence(reg_topic_words_cleaned, documents)
    print(f"Coherence Score for {period_label} - Standard: {coherence_score:.4f}")
    print(f"Coherence Score for {period_label} - Regularized: {reg_coherence_score:.4f}")

    # Visualizations with BERTopic built-in methods
    try:
        # Standard Model Visualizations
        fig = model.visualize_barchart(top_n_topics=10)
        fig.write_html(f"barchart_{period_label}_standard.html")
        print(f"Bar chart for {period_label} (Standard) saved as barchart_{period_label}_standard.html")

        topic_fig = model.visualize_topics()
        topic_fig.write_html(f"topics_{period_label}_standard.html")
        print(f"Topic map for {period_label} (Standard) saved as topics_{period_label}_standard.html")

        # Regularized Model Visualizations
        reg_fig = reg_model.visualize_barchart(top_n_topics=10)
        reg_fig.write_html(f"barchart_{period_label}_regularized.html")
        print(f"Bar chart for {period_label} (Regularized) saved as barchart_{period_label}_regularized.html")

        # Skip topic map visualization for the regularized model
    except Exception as e:
        print(f"Visualization failed for {period_label}: {e}")

if __name__ == "__main__":
    run_topic_modeling("documents/pre_SVB", "pre_SVB")
    run_topic_modeling("documents/post_SVB", "post_SVB")
