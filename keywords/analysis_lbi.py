import pandas as pd
import logging
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
import re
import openai
import time
from keybert import KeyBERT
from rake_nltk import Rake
import yake
import pke
from tqdm import tqdm
from spacy.matcher import Matcher
import spacy
import numpy as np
# Load the spacy model
nlp = spacy.load("en_core_web_sm")

#https://towardsdatascience.com/keyword-extraction-a-benchmark-of-7-algorithms-in-python-8a905326d93f
# initiate BERT outside of functions
bert = KeyBERT()

# 1. RAKE
def rake_extractor(text):
    """
    Uses Rake to extract the top 5 keywords from a text
    Arguments: text (str)
    Returns: list of keywords (list)
    """
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()[:5]

# 2. YAKE
def yake_extractor(text):
    """
    Uses YAKE to extract the top 5 keywords from a text
    Arguments: text (str)
    Returns: list of keywords (list)
    """
    keywords = yake.KeywordExtractor(lan="en", n=3, windowsSize=3, top=5).extract_keywords(text)
    results = []
    for scored_keywords in keywords:
        for keyword in scored_keywords:
            if isinstance(keyword, str):
                results.append(keyword) 
    return results 


# 3. PositionRank
def position_rank_extractor(text):
    """
    Uses PositionRank to extract the top 5 keywords from a text
    Arguments: text (str)
    Returns: list of keywords (list)
    """
    # define the valid Part-of-Speeches to occur in the graph
    pos = {'NOUN', 'PROPN', 'ADJ', 'ADV'}
    extractor = pke.unsupervised.PositionRank()
    extractor.load_document(text, language='en')
    extractor.candidate_selection(maximum_word_number=5)
    # 4. weight the candidates using the sum of their word's scores that are
    #    computed using random walk biaised with the position of the words
    #    in the document. In the graph, nodes are words (nouns and
    #    adjectives only) that are connected if they occur in a window of
    #    3 words.
    extractor.candidate_weighting(window=3, pos=pos)
    # 5. get the 5-highest scored candidates as keyphrases
    keyphrases = extractor.get_n_best(n=5)
    results = []
    for scored_keywords in keyphrases:
        for keyword in scored_keywords:
            if isinstance(keyword, str):
                results.append(keyword) 
    return results 

# 4. SingleRank
def single_rank_extractor(text):
    """
    Uses SingleRank to extract the top 5 keywords from a text
    Arguments: text (str)
    Returns: list of keywords (list)
    """
    pos = {'NOUN', 'PROPN', 'ADJ', 'ADV'}
    extractor = pke.unsupervised.SingleRank()
    extractor.load_document(text, language='en')
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting(window=3, pos=pos)
    keyphrases = extractor.get_n_best(n=5)
    results = []
    for scored_keywords in keyphrases:
        for keyword in scored_keywords:
            if isinstance(keyword, str):
                results.append(keyword) 
    return results 

# 5. MultipartiteRank
def multipartite_rank_extractor(text):
    """
    Uses MultipartiteRank to extract the top 5 keywords from a text
    Arguments: text (str)
    Returns: list of keywords (list)
    """
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(text, language='en')
    pos = {'NOUN', 'PROPN', 'ADJ', 'ADV'}
    extractor.candidate_selection(pos=pos)
    # 4. build the Multipartite graph and rank candidates using random walk,
    #    alpha controls the weight adjustment mechanism, see TopicRank for
    #    threshold/method parameters.
    extractor.candidate_weighting(alpha=1.1, threshold=0.74, method='average')
    keyphrases = extractor.get_n_best(n=5)
    results = []
    for scored_keywords in keyphrases:
        for keyword in scored_keywords:
            if isinstance(keyword, str):
                results.append(keyword) 
    return results

# 6. TopicRank
def topic_rank_extractor(text):
    """
    Uses TopicRank to extract the top 5 keywords from a text
    Arguments: text (str)
    Returns: list of keywords (list)
    """
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(text, language='en')
    pos = {'NOUN', 'PROPN', 'ADJ', 'ADV'}
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=5)
    results = []
    for scored_keywords in keyphrases:
        for keyword in scored_keywords:
            if isinstance(keyword, str):
                results.append(keyword) 
    return results

# 7. KeyBERT
def keybert_extractor(text):
    """
    Uses KeyBERT to extract the top 5 keywords from a text
    Arguments: text (str)
    Returns: list of keywords (list)
    """
    keywords = bert.extract_keywords(text, keyphrase_ngram_range=(3, 5), stop_words="english", top_n=5)
    results = []
    for scored_keywords in keywords:
        for keyword in scored_keywords:
            if isinstance(keyword, str):
                results.append(keyword)
    return results

def extract_keywords_from_corpus(extractor, corpus):
    """This function uses an extractor to retrieve keywords from a list of documents"""
    extractor_name = extractor.__name__.replace("_extractor", "")
    logging.info(f"Starting keyword extraction with {extractor_name}")
    corpus_kws = {}
    start = time.time()
    # logging.info(f"Timer initiated.") <-- uncomment this if you want to output start of timer
    for idx, text in tqdm(enumerate(corpus), desc="Extracting keywords from corpus..."):
        corpus_kws[idx] = extractor(text)
    end = time.time()
    # logging.info(f"Timer stopped.") <-- uncomment this if you want to output end of timer
    elapsed = time.strftime("%H:%M:%S", time.gmtime(end - start))
    logging.info(f"Time elapsed: {elapsed}")
    
    return {"algorithm": extractor.__name__, 
            "corpus_kws": corpus_kws, 
            "elapsed_time": elapsed}
    
def match(keyword):
    """This function checks if a list of keywords match a certain POS pattern"""
    patterns = [
        [{'POS': 'PROPN'}, {'POS': 'VERB'}, {'POS': 'VERB'}],
        [{'POS': 'NOUN'}, {'POS': 'VERB'}, {'POS': 'NOUN'}],
        [{'POS': 'VERB'}, {'POS': 'NOUN'}],
        [{'POS': 'ADJ'}, {'POS': 'ADJ'}, {'POS': 'NOUN'}],  
        [{'POS': 'NOUN'}, {'POS': 'VERB'}],
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}],
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'NOUN'}],
        [{'POS': 'ADJ'}, {'POS': 'NOUN'}],
        [{'POS': 'ADJ'}, {'POS': 'NOUN'}, {'POS': 'NOUN'}, {'POS': 'NOUN'}],
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'ADV'}, {'POS': 'PROPN'}],
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'VERB'}],
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}],
        [{'POS': 'NOUN'}, {'POS': 'NOUN'}],
        [{'POS': 'ADJ'}, {'POS': 'PROPN'}],
        [{'POS': 'PROPN'}, {'POS': 'ADP'}, {'POS': 'PROPN'}],
        [{'POS': 'PROPN'}, {'POS': 'ADJ'}, {'POS': 'NOUN'}],
        [{'POS': 'PROPN'}, {'POS': 'VERB'}, {'POS': 'NOUN'}],
        [{'POS': 'NOUN'}, {'POS': 'ADP'}, {'POS': 'NOUN'}],
        [{'POS': 'PROPN'}, {'POS': 'NOUN'}, {'POS': 'PROPN'}],
        [{'POS': 'VERB'}, {'POS': 'ADV'}],
        [{'POS': 'PROPN'}, {'POS': 'NOUN'}],
        ]
    matcher = Matcher(nlp.vocab)
    matcher.add("pos-matcher", patterns)
    # create spacy object
    doc = nlp(keyword)
    # iterate through the matches
    matches = matcher(doc)
    # if matches is not empty, it means that it has found at least a match
    if len(matches) > 0:
        return True
    return False

def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def benchmark(corpus, shuffle=True):
    """This function runs the benchmark for the keyword extraction algorithms"""
    logging.info("Starting benchmark...\n")
    
    # Shuffle the corpus
    if shuffle:
        random.shuffle(corpus)

    # extract keywords from corpus
    results = []
    extractors = [
        rake_extractor, 
        yake_extractor, 
        topic_rank_extractor, 
        position_rank_extractor,
        single_rank_extractor,
        multipartite_rank_extractor,
        keybert_extractor,
    ]
    for extractor in extractors:
        result = extract_keywords_from_corpus(extractor, corpus)
        results.append(result)

    # compute average number of extracted keywords
    for result in results:
        len_of_kw_list = []
        for kws in result["corpus_kws"].values():
            len_of_kw_list.append(len(kws))
        result["avg_keywords_per_document"] = np.mean(len_of_kw_list)

    # match keywords
    for result in results:
        for idx, kws in result["corpus_kws"].items():
            match_results = []
            for kw in kws:
                match_results.append(match(kw))
                result["corpus_kws"][idx] = match_results

    # compute average number of matched keywords
    for result in results:
        len_of_matching_kws_list = []
        for idx, kws in result["corpus_kws"].items():
            len_of_matching_kws_list.append(len([kw for kw in kws if kw]))
        result["avg_matched_keywords_per_document"] = np.mean(len_of_matching_kws_list)
        # compute average percentange of matching keywords, round 2 decimals
        result["avg_percentage_matched_keywords"] = round(result["avg_matched_keywords_per_document"] / result["avg_keywords_per_document"], 2)
        
    # create score based on the avg percentage of matched keywords divided by time elapsed (in seconds)
    for result in results:
        elapsed_seconds = get_sec(result["elapsed_time"]) + 0.1
        # weigh the score based on the time elapsed
        result["performance_score"] = round(result["avg_matched_keywords_per_document"] / elapsed_seconds, 2)
    
    # delete corpus_kw
    for result in results:
        del result["corpus_kws"]

    # create results dataframe
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    logging.info("Benchmark finished. Results saved to results.csv")
    return df


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
period_1 = period_1.head(5)

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
print("Number of texts: ", len(texts))

results = benchmark(texts, shuffle=True)


