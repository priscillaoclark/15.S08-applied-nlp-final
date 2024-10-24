import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download('stopwords')
stopwords.words("english")
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#https://medium.com/@theDrewDag/text-clustering-with-tf-idf-in-python-c94cd26a31e7

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

    # Loop through the docs in period_1 and store text and filename in the dictionary
    for index, row in period_1.iterrows():
        try:
            with open('documents/' + row['filename'], 'r') as file:
                text = file.read()
                text = re.sub('<.*?>', '', text)
                text = ' '.join(text.split())
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

def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """This utility function sanitizes a string by:
    - removing links
    - removing special characters
    - removing numbers
    - removing stopwords
    - transforming in lowercase
    - removing excessive whitespaces
    Args:
        text (str): the input text you want to clean
        remove_stopwords (bool): whether or not to remove stopwords
    Returns:
        str: the cleaned text
    """

    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    if remove_stopwords:
        # 1. tokenize
        tokens = nltk.word_tokenize(text)
        # 2. check if stopword
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        # 3. join back together
        text = " ".join(tokens)
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
    return text


df = get_lbi(100)
df['cleaned'] = df['text'].apply(lambda x: preprocess_text(x, remove_stopwords=True))
#print(df.head())

# initialize the vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.75)
# fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
X = vectorizer.fit_transform(df['cleaned'])

# initialize kmeans with 3 centroids
kmeans = KMeans(n_clusters=3, random_state=42)
# fit the model
kmeans.fit(X)
# store cluster labels in a variable
clusters = kmeans.labels_

# initialize PCA with 2 components
pca = PCA(n_components=2, random_state=42)
# pass our X to the pca and store the reduced vectors into pca_vecs
pca_vecs = pca.fit_transform(X.toarray())
# save our two dimensions into x0 and x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]

# assign clusters and pca vectors to our dataframe 
df['cluster'] = clusters
df['x0'] = x0
df['x1'] = x1

def get_top_keywords(n_terms):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(X.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names_out() # access tf-idf terms
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) # for each row of the dataframe, find the n terms that have the highest tf idf score
            
get_top_keywords(10)
