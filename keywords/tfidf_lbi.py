import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def get_lbi(number_of_docs=100000):
    # Import metadata.csv
    df = pd.read_csv('metadata.csv')
    df = df[(df['type'] == 'Rule') | (df['type'] == 'Proposed Rule')]
    #print(df.head())

    print("---------------LBI-----------------")
    # Create df for Lehman period: Sept. 15, 2008
    event_1 = '2008-09-15'
    event_1 = pd.to_datetime(event_1)
    period_1 = df[df['posted_date'] < '2021-01-01']
    print("Min date of downloaded: ",period_1['posted_date'].min())
    print("Max date of downloaded: ",period_1['posted_date'].max())
    # Count docs in period_1
    print("Total LBI docs downloaded: ",period_1['id'].count())

    # Calculate prior to event_1
    prior_LBI = event_1 - pd.DateOffset(months=18)
    # Convert to string
    prior_LBI = prior_LBI.strftime('%Y-%m-%d')
    print("18 months prior: ",prior_LBI)
    # Calculate after event_1
    after_LBI = event_1 + pd.DateOffset(months=18)
    after_LBI = after_LBI.strftime('%Y-%m-%d')
    print("18 months after: ",after_LBI)

    # Filter period_1 to dates between one_year_prior_LBI and one_year_after_LBI
    period_1 = period_1[(period_1['posted_date'] >= prior_LBI) & (period_1['posted_date'] <= after_LBI)]
    # Count docs in period_1
    print("Total LBI docs for period: ",period_1['id'].count())

    # Limit period_1 to first few docs
    period_1 = period_1.head(number_of_docs)

    # Create dictionary to store text and filename
    doc_dict = {}

    # Initialize the stemmer
    #stemmer = nltk.stem.PorterStemmer()

    # Loop through the docs in period_1 and store text and filename in the dictionary
    for index, row in period_1.iterrows():
        try:
            with open('documents/' + row['filename'], 'r') as file:
                try: 
                    text = file.read()
                except UnicodeDecodeError:
                    print("UnicodeDecodeError: ", row['filename'])
                    continue
                text = re.sub('<.*?>', '', text)
                # Remove numbers with regex
                text = re.sub(r'\d+', '', text)
                text = ' '.join(text.split())
                # Stemming
                #text = ' '.join([stemmer.stem(word) for word in text.split()])
                #text = text[:100000]
                doc_dict[row['filename']] = text
        except FileNotFoundError:
            print("File not found: ", row['filename'])
    #print("Number of documents: ", len(doc_dict))
    #print(list(doc_dict.keys()))
    # Print the first document
    #print(doc_dict[list(doc_dict.keys())[0]])
    # Create a list of texts
    texts = list(doc_dict.values())
    #print("Number of texts: ", len(texts))

    # Create a dataframe with the texts with the filename as the index
    df = pd.DataFrame(texts, index=list(doc_dict.keys()), columns=['text'])
    #print(df.head())
    return df

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords_tfidf(df, text_column, top_n=10):
    """
    This function takes a dataframe and a specified text column, performs TF-IDF
    vectorization, and extracts the top keywords for each row of text.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with a text column.
    text_column (str): The name of the column containing text data.
    top_n (int): The number of top keywords to extract for each row of text.
    
    Returns:
    pd.DataFrame: A dataframe with original text and the extracted keywords.
    """
    
    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df = 10, max_df=0.4)
    
    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    
    # Get the feature names (i.e., the words)
    feature_names = vectorizer.get_feature_names_out()
    
    # Function to extract top N keywords for each text entry
    def extract_top_keywords(row):
        tfidf_scores = row.toarray().flatten()
        top_indices = tfidf_scores.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]
        return ", ".join(top_keywords)
    
    # Apply the function to each row of the tfidf_matrix
    df['keywords'] = [extract_top_keywords(row) for row in tfidf_matrix]
    
    # Return a new dataframe with original text and extracted keywords
    return df[[text_column, 'keywords']]

# Example usage:
# df = pd.DataFrame({'text': ['This is a sample document.', 'Another text data example.']})
# result_df = extract_keywords_tfidf(df, 'text')
# print(result_df)

df = get_lbi(200)
result_df = extract_keywords_tfidf(df, 'text', 20)
print(result_df)
