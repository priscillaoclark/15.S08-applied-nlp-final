import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
import re
import openai
import time

# Import metadata.csv
df = pd.read_csv('metadata.csv')
df = df[(df['type'] == 'Rule') | (df['type'] == 'Proposed Rule')]
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

"""
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
#print(df_word_count.head())
"""

# Limit period_1 to first few docs
period_1 = period_1.head(50)

# Loop through the docs in period_1 and summarize the text
for index, row in period_1.iterrows():
    
    # Check if file exists
    try:
        # Open the file
        with open('documents/' + row['filename'], 'r') as file:
            # Read the file
            text = file.read()
            # Remove html tags
            text = re.sub('<.*?>', '', text)
            # Clean up extra white space
            text = ' '.join(text.split())
            # Limit to first 100000 characters
            text = text[:100000]
            # Print the first 1000 characters
            #print(text[:1000])

        prompt = f"""
        --- Text for review:
        ```
        {text}
        ```
        """

        assistant_id = 'asst_js2ojaiY97QvXd9sPzjmpKjH'
 
        client = openai.Client(api_key=OPENAI_API_KEY)
        thread = client.beta.threads.create()
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        start_time = time.time()
        timeout = 300  # 5 minutes timeout
        while run.status not in ["completed", "failed"]:
            if time.time() - start_time > timeout:
                print(f"Run timed out after {timeout} seconds")
            print("Waiting 10 seconds to prevent rate limiting...")  # tokens per min (TPM): Limit 30000
            time.sleep(10)  # Increase sleep time to reduce API calls
            print(f"Analyzing document {row['filename']}...")
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            #print(f"Current run status: {run.status}")  # Debug print

        if run.status == "failed":
            print(f"Run failed. Error: {run.last_error.message}")

        messages = client.beta.threads.messages.list(thread_id=thread.id)
        output = messages.data[0].content[0].text.value
        #print(output)

        # Strip out filename from extension
        file = row['filename'].split('.')[0]
        doc_type = row['type']
        # Save as JSON file in ai_outputs directory
        with open(f"ai_outputs/lbi/{file}-{doc_type}.json", 'w') as f:
            f.write(output)
            
    except FileNotFoundError:
        print(f"File {row['filename']} not found")
