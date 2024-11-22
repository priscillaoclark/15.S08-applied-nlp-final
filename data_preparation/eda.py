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
print(combined.head())

import matplotlib.pyplot as plt

# Count the number of documents per agency
agency_counts = combined['agency'].value_counts()

# Plot a pie chart
colors = ['#0158BF', '#76C6FC', '#00143F', '#a7a7a7', '#535353', '#15b8ae', '#85d0ccff']
fig, ax = plt.subplots(figsize=(10, 8))

# Create pie chart
wedges, texts, autotexts = ax.pie(agency_counts, 
                                 labels=agency_counts.index,
                                 autopct=lambda p: f'{p:.1f}% ({int(p * agency_counts.sum() / 100)})',
                                 startangle=90,
                                 colors=colors,
                                 wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                                 pctdistance=0.85)

# Set colors for texts
plt.setp(autotexts, color='white')  # percentage labels in white
plt.setp(texts, color='black')      # agency labels in black

plt.title('Number of Documents per Agency')
plt.ylabel('')

# Draw a white circle at the center to make it look like a donut chart
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

plt.savefig('data_preparation/agency_pie_chart.png')

# Extract year and month from 'posted_date'
combined['year_month'] = combined['posted_date'].dt.to_period('M')

# Count the number of documents per month and type
monthly_counts = combined.groupby(['year_month', 'type']).size().unstack(fill_value=0)

# Plot a line graph
plt.figure(figsize=(12, 6))
monthly_counts.plot(kind='line', marker='o', color=colors)
plt.axvline(pd.Timestamp('2023-03-10').to_period('M'), color='#15b8ae', linestyle='--', label='SVB Collapse')
plt.title('Number of Documents per Month by Type')
plt.xlabel('Month')
plt.ylabel('Number of Documents')
plt.legend(title='Document Type')
plt.grid(True)
plt.savefig('data_preparation/documents_per_month_by_type.png')

# See the minimum and maximum 'posted_date'
print(combined['posted_date'].min())
print(combined['posted_date'].max())

# Count the number of words in each document
def count_words(file_path):
    with open(file_path, 'r') as file:
        return len(file.read().split())

combined['word_count'] = combined['filename_y'].apply(lambda x: count_words(f'documents/pre_SVB/{x}') if x in pre_SVB_files else count_words(f'documents/post_SVB/{x}'))

# Plot a histogram of the distribution of the lengths
plt.figure(figsize=(12, 6))
n, bins, patches = plt.hist(combined['word_count'], bins=30, color='#0158BF', edgecolor='black')
plt.title('Distribution of Document Lengths')
plt.xlabel('Document Length')
plt.ylabel('Frequency')

# Add values at the top of the bars
for i in range(len(patches)):
    plt.text(patches[i].get_x() + patches[i].get_width() / 2, patches[i].get_height(), 
             str(int(patches[i].get_height())), ha='center', va='bottom', fontsize=10, color='black')

# Remove the outline
for patch in patches:
    patch.set_edgecolor('none')

plt.savefig('data_preparation/document_length_distribution.png')

# Filter documents with less than 100,000 words
filtered_combined = combined[combined['word_count'] < 25000]

# Plot a histogram of the distribution of the lengths for filtered documents
plt.figure(figsize=(12, 6))
n, bins, patches = plt.hist(filtered_combined['word_count'], bins=30, color='#0158BF', edgecolor='black')
plt.title('Distribution of Document Lengths (Less than 25,000 words)')
plt.xlabel('Document Length')
plt.ylabel('Frequency')

# Add values at the top of the bars
for i in range(len(patches)):
    plt.text(patches[i].get_x() + patches[i].get_width() / 2, patches[i].get_height(), 
             str(int(patches[i].get_height())), ha='center', va='bottom', fontsize=10, color='black')

# Remove the outline
for patch in patches:
    patch.set_edgecolor('none')

plt.savefig('data_preparation/document_length_distribution_filtered.png')