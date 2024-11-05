import pandas as pd
import re
from keybert import KeyBERT

custom_stop_words = ['fdic','federal','rule', 'rules', 'proposed', 'comment', 'comments', 'commenting', 'section', 'sections', 'act', 'acts', 'file', 'files', 'document', 'documents', 'page', 'pages', 'agency', 'agencies', 'federal', 'register', 'regulations', 'regulation', 'regulatory', 'register', 'published', 'publication', 'filing', 'required', 'auditor', 'audit','audits','auditing','requirements']

# Initialize KeyBERT model
bert = KeyBERT()

def keybert_extractor(text):
    """
    Uses KeyBERT to extract the top 5 keywords from a text
    Arguments: text (str)
    Returns: list of keywords (list)
    """
    keywords = bert.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=30)
    results = []
    for scored_keywords in keywords:
        for keyword in scored_keywords:
            if isinstance(keyword, str):
                results.append(keyword)
    return results

# Import metadata.csv
df = pd.read_csv('data_preparation/metadata.csv')
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
            #text = ' '.join(text.split())
            # Remove numbers and stopwords
            text = ' '.join(word for word in text.split() if word.lower() not in custom_stop_words and not word.isdigit())
            
            #text = text[:100000]
            keywords = keybert_extractor(text)
            doc_dict[row['filename']] = text
            doc_dict[row['filename'] + '_keywords'] = keywords
    except FileNotFoundError:
        print("File not found: ", row['filename'])
        
# Print the keywords for each document
for key, value in doc_dict.items():
    if 'keywords' in key:
        print(key, value)
        print("\n")