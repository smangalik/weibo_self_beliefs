# python3 data_extraction.py

import csv
import pandas as pd
from tqdm import tqdm

def read_csv(file_path):
    """Read a CSV file and return its contents as a list of dictionaries."""
    malformed_rows = 0
    empty_rows = 0
    rows = []
    # Open the CSV file and read its contents
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Reading Leiden Weibo Corpus", unit="row"):
            # Skip invalid rows
            if len(row) != 9:
                malformed_rows += 1
                continue
            # Skip empty rows
            if row['wordcount'] == '\\N':
                empty_rows += 1
                continue
            rows.append(row)
        
    # Print the number of skipped rows
    print(f"\nMalformed rows skipped: {malformed_rows}")
    print(f"Empty rows skipped: {empty_rows}")
    
    row_df = pd.DataFrame(rows)
    return row_df
    
df = read_csv('LWC-messages/parsed_messages.txt')

print(f"\nParsed Posts {df.shape}, Columns: {df.columns.tolist()}")
print(df)

df.to_csv('LWC-messages/parsed_messages.csv', index=False, encoding='utf-8')