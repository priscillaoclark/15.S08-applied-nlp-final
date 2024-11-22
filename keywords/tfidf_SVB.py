import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def process_documents(period_1, folder_name):
        # Create dictionary to store text and filename
        doc_dict = {}

        # Get stopwords once
        stop_words = set(stopwords.words('english'))
        # Add custom stopwords
        custom_stops = {
            'shall', 'may', 'must', 'would', 'could', 'should', 'also', 'pursuant', 'therefore',
            'whereas', 'hereby', 'herein', 'therein', 'thereof', 'thereto', 'therewith',
            'federal', 'register', 'regulation', 'regulatory', 'rule', 'rules', 'rulemaking',
            'agency', 'agencies', 'department', 'gov', 'www', 'http', 'https', 'com',
            'section', 'subsection', 'paragraph', 'subparagraph', 'part', 'title',
            'effective', 'date', 'dates', 'amended', 'amendment', 'amend'
        }
        stop_words.update(custom_stops)

        # Compile regex patterns once
        html_pattern = re.compile('<.*?>')
        nonletters_pattern = re.compile('[^a-zA-Z\s]')

        # Loop through docs
        for index, row in period_1.iterrows():
            try:
                with open(f'documents/{folder_name}/' + row['filename'], 'r') as file:
                    try: 
                        text = file.read()
                    except UnicodeDecodeError:
                        continue
                    # Process text using compiled patterns
                    text = html_pattern.sub('', text)
                    text = nonletters_pattern.sub(' ', text)
                    text = text.lower()
                    text = ' '.join(text.split())
                    # Remove stopwords more efficiently
                    text = ' '.join(w for w in text.split() if w not in stop_words)
                    doc_dict[row['filename']] = text
            except FileNotFoundError:
                continue

        return list(doc_dict.values()), doc_dict

def get_svb(number_of_docs=100000):
    # Import metadata.csv
    df = pd.read_csv('data_preparation/metadata.csv')
    df = df[(df['type'] == 'Rule') | (df['type'] == 'Proposed Rule')]

    # Create df for SVB, March 10, 2023
    event_1 = pd.to_datetime('2023-03-10')
    period_1 = df[df['posted_date'] > '2021-01-01']

    # Calculate periods
    prior_SVB = event_1 - pd.DateOffset(months=18)
    after_SVB = event_1 + pd.DateOffset(months=18)
    prior_SVB = prior_SVB.strftime('%Y-%m-%d')
    after_SVB = after_SVB.strftime('%Y-%m-%d')
    event_1 = event_1.strftime('%Y-%m-%d')

    # Filter periods efficiently using boolean indexing
    pre_period = period_1[(period_1['posted_date'] >= prior_SVB) & (period_1['posted_date'] < event_1)].head(number_of_docs)
    post_period = period_1[(period_1['posted_date'] >= event_1) & (period_1['posted_date'] <= after_SVB)].head(number_of_docs)
    
    # Process documents
    pre_texts, pre_dict = process_documents(pre_period, 'pre_SVB')
    post_texts, post_dict = process_documents(post_period, 'post_SVB')

    return (pd.DataFrame(pre_texts, index=list(pre_dict.keys()), columns=['text']), 
            pd.DataFrame(post_texts, index=list(post_dict.keys()), columns=['text']))

def extract_keywords_tfidf(df, text_column, top_n=50):
    """
    Enhanced TF-IDF keyword extraction with improved parameters and preprocessing
    """
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.3,
        lowercase=True,  # Already lowercased in preprocessing
        strip_accents='unicode'
    )
    
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    feature_names = vectorizer.get_feature_names_out()
    
    # Vectorized operations for better performance
    avg_tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    top_indices = np.argsort(avg_tfidf_scores)[-top_n:][::-1]
    
    return ([(feature_names[i], avg_tfidf_scores[i]) for i in top_indices], 
            feature_names, 
            avg_tfidf_scores)

# Run analysis with smaller sample for faster processing
pre_df, post_df = get_svb(100)

# Get keywords and scores
pre_keywords, pre_features, pre_scores = extract_keywords_tfidf(pre_df, 'text')
post_keywords, post_features, post_scores = extract_keywords_tfidf(post_df, 'text')

# Create score dictionaries using dict comprehension
pre_dict = dict(zip(pre_features, pre_scores))
post_dict = dict(zip(post_features, post_scores))

# Calculate mean score once
mean_post_score = np.mean(list(post_dict.values()))

# Find significant changes more efficiently
significant_changes = []
for term, post_score in post_dict.items():
    pre_score = pre_dict.get(term, 0)
    if pre_score == 0:
        if post_score > mean_post_score:
            significant_changes.append((term, "New term", 0, post_score))
    elif (post_score - pre_score) / pre_score > 0.5:
        significant_changes.append((term, "Increased", pre_score, post_score))

# Sort and display results
significant_changes.sort(key=lambda x: x[3], reverse=True)

print("\nSignificant keyword changes after SVB:")
print("Term | Change Type | Pre-SVB Score | Post-SVB Score")
print("-" * 60)
for term, change_type, pre_score, post_score in significant_changes[:20]:
    print(f"{term:30} | {change_type:10} | {pre_score:.4f} | {post_score:.4f}")

print("\nTop 50 keywords in Pre-SVB period:")
for term, score in pre_keywords:
    print(f"{term}: {score:.4f}")

print("\nTop 50 keywords in Post-SVB period:")
for term, score in post_keywords:
    print(f"{term}: {score:.4f}")
