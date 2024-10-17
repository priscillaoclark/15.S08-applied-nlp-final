import pandas as pd
import os
import requests

# Open documents.xlsx
df = pd.read_excel('qa.xlsx')
# Remove . from extension
df['extension'] = df['extension'].str.replace('.', '')
# Sort by id
df = df.sort_values(by='id')
#print(df.head())
#print(df.shape)

# Pull list of files in the documents directory
files = os.listdir('documents')
df_files = pd.DataFrame(files, columns=['filename'])
print(df_files.shape)

# Merge df to df_files and check which files are missing in the documents directory
df_files['id'] = df_files['filename'].str.split('.').str[0]
df_files['extension'] = df_files['filename'].str.split('.').str[1]
df_missing = pd.merge(df, df_files, how='left', on='id')
df_missing_2 = df_missing[df_missing['filename'].isnull()]
print(df_missing_2.head())

# Download the missing files using the url from the documents.xlsx
for index, row in df_missing_2.iterrows():
    url = row['url']
    filename = row['id'] + '.' + row['extension']
    response = requests.get(url)
    with open('documents/' + filename, 'wb') as f:
        f.write(response.content)
        print('Downloaded', filename)

# See where there are multiple documents with the same id
df_duplicates = df_files[df_files['id'].duplicated(keep=False)]
# Sort by id
df_duplicates = df_duplicates.sort_values(by='id')
# Filter to non-htm records
df_duplicates = df_duplicates[df_duplicates['extension'] != 'htm']
print(df_duplicates.head())

# Remove these files
for index, row in df_duplicates.iterrows():
    os.remove('documents/' + row['filename'])
    print('Removed', row['filename'])

df_missing = pd.merge(df_files, df, how='left', on='id')
df_missing_2 = df_missing[df_missing['filename'].isnull()]
print(df_missing_2.head())
