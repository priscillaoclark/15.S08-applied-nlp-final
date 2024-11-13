import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
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
    return " ".join(word for word in text.split() if word not in STOPWORDS and len(word) > 2)

def load_and_preprocess_data(metadata_path: str, event_date: str, num_docs: int = 100):
    """Loads metadata and preprocesses document texts for a specific time period."""
    df = pd.read_csv(metadata_path)
    df = df[(df['type'] == 'Rule') | (df['type'] == 'Proposed Rule')]
    print(f"Total documents available: {len(df)}")

    # Filter by event date and date offset
    event_date = pd.to_datetime(event_date)
    start_date = event_date - pd.DateOffset(months=18)
    end_date = event_date + pd.DateOffset(months=18)
    df = df[(df["posted_date"] >= str(start_date)) & (df["posted_date"] <= str(end_date))]

    df = df.head(num_docs)
    doc_dict = {}
    for _, row in df.iterrows():
        try:
            with open(f"documents/{row['filename']}", "r", encoding="utf-8") as file:
                text = file.read()
                doc_dict[row["filename"]] = preprocess_text(text)
        except UnicodeDecodeError:
            # Retry with a different encoding
            try:
                with open(f"documents/{row['filename']}", "r", encoding="latin-1") as file:
                    text = file.read()
                    doc_dict[row["filename"]] = preprocess_text(text)
            except Exception as e:
                print(f"Error reading file {row['filename']}: {e}")
        except FileNotFoundError:
            print(f"File not found: {row['filename']}")

    return list(doc_dict.values()), doc_dict

def compute_coherence_and_perplexity(lda_model, corpus, dictionary, tokenized_docs):
    """Computes topic coherence and perplexity for an LDA model."""
    # Compute coherence
    coherence_model = CoherenceModel(model=lda_model, texts=tokenized_docs, dictionary=dictionary, coherence='c_npmi')
    coherence_score = coherence_model.get_coherence()

    # Compute perplexity
    log_perplexity = lda_model.log_perplexity(corpus)
    perplexity_score = np.exp(-log_perplexity)

    return coherence_score, perplexity_score

def plot_topics(lda_model, num_words=10, output_file="topics.png"):
    """Generates a word cloud visualization for each topic."""
    topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
    plt.figure(figsize=(15, 10))

    for idx, topic in topics:
        plt.subplot(3, 3, idx + 1)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(topic))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"Topic {idx}")

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()
    print(f"Topic visualization saved as {output_file}")

def run_lda_topic_modeling(metadata_path: str, event_date: str, period_label: str):
    """Runs LDA for a specific time period, validates coherence and perplexity, and generates visualizations."""
    # Load and preprocess data
    documents, _ = load_and_preprocess_data(metadata_path, event_date, num_docs=200)
    tokenized_docs = [doc.split() for doc in documents]

    # Create dictionary and corpus
    dictionary = Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    # Train LDA model
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, random_state=42, passes=10, alpha='auto', eta='auto')

    # Compute validation metrics
    coherence_score, perplexity_score = compute_coherence_and_perplexity(lda_model, corpus, dictionary, tokenized_docs)
    print(f"Coherence Score for {period_label}: {coherence_score:.4f}")
    print(f"Perplexity Score for {period_label}: {perplexity_score:.4f}")

    # Visualize topics
    plot_file = f"lda_topics_{period_label}.png"
    plot_topics(lda_model, output_file=plot_file)

    # Print top keywords for each topic
    topics = lda_model.show_topics(num_topics=5, num_words=10, formatted=False)
    for idx, topic in topics:
        print(f"Topic {idx}: {', '.join([word for word, _ in topic])}")

if __name__ == "__main__":
    metadata_path = "data_preparation/metadata.csv"

    # Run LDA for 2008
    run_lda_topic_modeling(metadata_path, "2008-09-15", "2008")

    # Run LDA for 2023
    run_lda_topic_modeling(metadata_path, "2023-03-15", "2023")