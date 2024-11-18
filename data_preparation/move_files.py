import pandas as pd
import os
import shutil
import pytz
from datetime import datetime

# Import metadata.csv
df = pd.read_csv('data_preparation/metadata.csv')

# Limit to rule or proposed rule
df = df[(df['type'] == 'Rule') | (df['type'] == 'Proposed Rule')]

# Convert 'posted_date' to datetime (timezone-naive)
df['posted_date'] = pd.to_datetime(df['posted_date']).dt.tz_localize(None)

print(df.head())

print("---------------SVB-----------------")
# Create df for SVB period: Mar 10, 2023
event = pd.to_datetime('2023-03-10')  # timezone-naive
period = df[df['posted_date'] >= '2021-01-01']
print("Min date of downloaded: ", period['posted_date'].min())
print("Max date of downloaded: ", period['posted_date'].max())
print("Total SVB docs downloaded: ", period['id'].count())

# Calculate periods
prior_SVB = event - pd.DateOffset(months=18)
prior_SVB = prior_SVB.strftime('%Y-%m-%d')
print("18 months prior: ", prior_SVB)

after_SVB = event + pd.DateOffset(months=18)
after_SVB = after_SVB.strftime('%Y-%m-%d')
print("18 months after: ", after_SVB)

# Filter period_2 to dates between prior_SVB and after_SVB
period_2 = period[
    (period['posted_date'] >= prior_SVB) & 
    (period['posted_date'] <= after_SVB)
]

# Count docs in period_2
print("Total SVB docs for period: ", period_2['id'].count())
print(period_2['agency'].value_counts())

# Create directories safely
os.makedirs('documents/pre_SVB', exist_ok=True)
os.makedirs('documents/post_SVB', exist_ok=True)

def move_file_safely(src, dst):
    """Safely move a file with error handling."""
    try:
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved: {src} -> {dst}")
        else:
            print(f"File not found: {src}")
    except Exception as e:
        print(f"Error moving {src}: {str(e)}")

# Move files to pre_SVB
for i in period[period['posted_date'] < event]['id']:
    for i in period[period['posted_date'] >= prior_SVB]['id']:
        move_file_safely(f'documents/{i}.htm', f'documents/pre_SVB/{i}.htm')

# Move files to post_SVB
for i in period[period['posted_date'] >= event]['id']:
    for i in period[period['posted_date'] <= after_SVB]['id']:
        move_file_safely(f'documents/{i}.htm', f'documents/post_SVB/{i}.htm')
    
# 368
# 482