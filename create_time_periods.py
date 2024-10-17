import pandas as pd

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
# Count the docs per agency
print(period_1['agency'].value_counts())

print("---------------SVB-----------------")
# Create df for SVB period: Mar 10, 2023
event_2 = '2023-03-10'
event_2 = pd.to_datetime(event_2)
period_2 = df[df['posted_date'] >= '2021-01-01']
print("Min date of downloaded: ",period_2['posted_date'].min())
print("Max date of downloaded: ",period_2['posted_date'].max())
# Count docs in period_2
print("Total SVB docs downloaded: ",period_2['id'].count())

# Calculate one year prior to event_2
prior_SVB = event_2 - pd.DateOffset(months=18)
prior_SVB = prior_SVB.strftime('%Y-%m-%d')
print("18 months prior: ",prior_SVB)
# Calculate one year after event_1
after_SVB = event_2 + pd.DateOffset(months=18)
after_SVB = after_SVB.strftime('%Y-%m-%d')
print("18 months after: ",after_SVB)

# Filter period_2 to dates between one_year_prior_SVB and one_year_after_SVB
period_2 = period_2[(period_2['posted_date'] >= prior_SVB) & (period_2['posted_date'] <= after_SVB)]
# Count docs in period_2
print("Total SVB docs for period: ",period_2['id'].count())
# Count the docs per agency
print(period_2['agency'].value_counts())