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

event = pd.to_datetime('2023-03-10') 
after_SVB = event + pd.DateOffset(months=18)
after_SVB = after_SVB.strftime('%Y-%m-%d')

# Filter df to dates between event and after_SVB
df = df[
    (df['posted_date'] > after_SVB)
]

# Print min and max dates
print(df['posted_date'].min())
print(df['posted_date'].max())

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

"""
# Move files to post_SVB
for i in df[df['posted_date'] >= event]['id']:
    for i in df[df['posted_date'] <= after_SVB]['id']:
        move_file_safely(f'documents/pre_SVB/{i}.htm', f'documents/post_SVB/{i}.htm')"""

# Move documents that are after_SVB
for i in df[df['posted_date'] > after_SVB]['id']:
    move_file_safely(f'documents/pre_SVB/{i}.htm', f'documents/{i}.htm')