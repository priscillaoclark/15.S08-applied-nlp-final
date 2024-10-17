import pandas as pd
import os
import requests

# Open documents.xlsx
df = pd.read_excel('qa.xlsx')
# Create filename column
df['filename'] = df['id'] + df['extension']
# Sort by filename
df = df.sort_values(by='filename')
print(df.head())
print(df.shape)

# Pull list of files in the documents directory
files = os.listdir('documents')
df_files = pd.DataFrame(files, columns=['filename'])
# Sort by filename
df_files = df_files.sort_values(by='filename')
print(df_files.head())
print(df_files.shape)

# Join df to df_files on filename
df_combined = pd.merge(df, df_files, how='inner', on='filename')
# Sort by id
df_combined = df_combined.sort_values(by='id')
print(df_combined.head())
print(df_combined.shape)

"""
# Which filenames exist in the documents directory but are not in the metadata?
df_missing = pd.merge(df_files, df, how='left', on='filename')
df_missing_2 = df_missing[df_missing['id'].isnull()]
print(df_missing_2.head())
# Remove these files
for index, row in df_missing_2.iterrows():
    os.remove('documents/' + row['filename'])
    print('Removed', row['filename'])
"""

# Export to csv
df_combined.to_csv('metadata.csv', index=False)