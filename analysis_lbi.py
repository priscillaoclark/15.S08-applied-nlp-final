import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

# Import metadata.csv
df = pd.read_csv('metadata.csv')
print(df.head())

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

# Loop through docs in period_1 and count words for each and add to a list with the filename
word_count = []
for index, row in period_1.iterrows():
    # Open the file
    with open('documents/' + row['filename'], 'r') as file:
        # Read the file
        text = file.read()
        # Split the text into words
        words = text.split()
        # Count the number of words
        count = len(words)
        # Append to word_count list
        word_count.append({'filename': row['filename'], 'word_count': count})
# Convert to DataFrame
df_word_count = pd.DataFrame(word_count)
# Sort by longest document
df_word_count = df_word_count.sort_values(by='word_count', ascending=False)
print(df_word_count.head())

