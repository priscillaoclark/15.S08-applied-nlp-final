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

#print(df.head())

###############################
# PRE EVENT

# Pull in files in pre_SVB
pre_SVB_files = os.listdir('documents/pre_SVB')
print("Files in pre_SVB: ", len(pre_SVB_files))

# Join file in pre_SVB to metadata
pre_SVB = pd.DataFrame(pre_SVB_files, columns=['filename'])
pre_SVB['id'] = pre_SVB['filename'].str.split('.').str[0]
pre_SVB = pre_SVB.merge(df, on='id', how='left')
#print(pre_SVB.head())
print(len(pre_SVB))

###############################
# POST EVENT

# Pull in files in post_SVB
post_SVB_files = os.listdir('documents/post_SVB')
print("Files in post_SVB: ", len(post_SVB_files))

# Join file in post_SVB to metadata
post_SVB = pd.DataFrame(post_SVB_files, columns=['filename'])
post_SVB['id'] = post_SVB['filename'].str.split('.').str[0]
post_SVB = post_SVB.merge(df, on='id', how='left')
#print(post_SVB.head())
print(len(post_SVB))

###############################
# TOTAL DOCUMENTS

# Combine pre_SVB and post_SVB
combined = pd.concat([pre_SVB, post_SVB])
#print(combined.head())
print(len(combined))

import matplotlib.pyplot as plt

# Count the number of documents per agency
agency_counts = combined['agency'].value_counts()

# Plot a pie chart
plt.figure(figsize=(10, 8))
agency_counts.plot.pie(autopct=lambda p: f'{p:.1f}% ({int(p * agency_counts.sum() / 100)})', startangle=90, cmap='viridis')
plt.title('Number of Documents per Agency')
plt.ylabel('')  # Hide the y-label
plt.savefig('data_preparation/agency_pie_chart.png')

# Extract year and month from 'posted_date'
combined['year_month'] = combined['posted_date'].dt.to_period('M')

# Count the number of documents per month and type
monthly_counts = combined.groupby(['year_month', 'type']).size().unstack(fill_value=0)

# Plot a line graph
plt.figure(figsize=(12, 6))
monthly_counts.plot(kind='line', marker='o')
plt.axvline(pd.Timestamp('2023-03-10').to_period('M'), color='red', linestyle='--', label='SVB Collapse')
plt.title('Number of Documents per Month by Type')
plt.xlabel('Month')
plt.ylabel('Number of Documents')
plt.legend(title='Document Type')
plt.grid(True)
plt.savefig('data_preparation/documents_per_month_by_type.png')

# See the minimum and maximum 'posted_date'
print(combined['posted_date'].min())
print(combined['posted_date'].max())


