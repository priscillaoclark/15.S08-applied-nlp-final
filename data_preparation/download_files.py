import pandas as pd
import requests
import os

# Import metadata.csv
df = pd.read_csv('data_preparation/metadata.csv')
# Limit to rule or proposed rule
df = df[(df['type'] == 'Rule') | (df['type'] == 'Proposed Rule')]   
print(df.head())

# Create the documents directory if it doesn't exist
os.makedirs('documents', exist_ok=True)


# Download each file and save it with the document id
for index, row in df.iterrows():
    print("Downloading", row['url'])
    url = row['url']
    document_id = row['id']
    extension = url.split('.')[-1]
    if extension == 'htm':
        response = requests.get(url)
        if response.status_code == 200:
            with open(f'documents/{document_id}.{extension}', 'wb') as file:
                file.write(response.content)
        else:
            print(f"Failed to download {url}")

"""

# Get list of downloaded files
files = os.listdir('documents')
# Split out the document id
files = [file.split('.')[0] for file in files]
print(files[0])

# Download other file types if the document id is not in the list
for index, row in df.iterrows():
    url = row['url']
    document_id = row['id']
    extension = url.split('.')[-1]
    if document_id not in files:
        response = requests.get(url)
        if response.status_code == 200:
            with open(f'documents/{document_id}.{extension}', 'wb') as file:
                file.write(response.content)
        else:
            print(f"Failed to download {url}")


"""