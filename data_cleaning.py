# python3 data_cleaning.py

import pandas as pd
import matplotlib.pyplot as plt
#import re

# How many rows to parse
nrows = None
input_file = 'LWC-messages/parsed_messages.csv'
output_file = 'LWC-messages/cleaned_messages.csv'

# Columns: ['message_id', 'user_province', 'user_city', 'user_gender', 'user_screen_name', 
#           'text', 'wordcount', 'wordboundaries', 'postags']
print(f"Reading {nrows if nrows else 'all'} rows from parsed_messages.csv")
df = pd.read_csv(input_file, encoding='utf-8', nrows=nrows)
df['text'] = df['text'].astype(str)
df['wordcount'] = pd.to_numeric(df['wordcount'], errors='coerce', downcast='integer')  # Convert wordcount to numeric, coerce errors
df= df[['message_id', 'user_province', 'user_city', 'user_gender', 'user_screen_name', 'text', 'wordcount']]
print(f"\nParsed Posts {df.shape}, Columns: {df.columns.tolist()}")

# Remove all posts containing 'http' or 'https'
df = df[~df['text'].str.contains('http|https', na=False)]
print(f"\nAfter removing posts with URLs: {df.shape}")

# Remove posts with wordcount less than 3
df.dropna(subset=['wordcount'], inplace=True)  # Drop rows where wordcount is NaN
df = df[df['wordcount'] >= 3]
print(f"\nAfter removing posts with wordcount < 3: {df.shape}")

# Split rows into non-short sentences
print("\nSplitting text into sentences...")
df['text'] = df['text'].str.split(r'[。，！？\r\n]').apply(lambda x: [s.strip() for s in x if s.strip()])
df = df.explode('text').reset_index(drop=True) # Flatten the list of sentences into a single column
df['text'] = df['text'].astype(str).str.strip() # Strip whitespace from sentences
df['wordcount_sentence'] = df['text'].apply(lambda x: len(x.split()))
print(f"After splitting into sentences: {df.shape}")
df['text'] = df['text'].astype(str)

# Reset index after cleaning
df.reset_index(drop=True, inplace=True)

# Plot word count distribution
plt.figure(figsize=(10, 6))
plt.hist(df['wordcount_sentence'], bins=20, color='blue', alpha=0.7)
plt.title('Word Count Distribution')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig('LWC-messages/word_count_distribution.png')


print(f"\nFinal DataFrame shape: {df.shape}, Columns: {df.columns.tolist()}")
df.to_csv(output_file, index=False, encoding='utf-8')

print(df)