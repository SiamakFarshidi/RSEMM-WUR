import os
import json
from collections import Counter

# Directory containing JSON files
directory = r'C:\Users\User\Downloads\RS-Zenonfo\RS-Repo-Zenodo - v2'

# Collect all doi_urls
doi_urls = []

# Loop through all JSON files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                doi_url = data.get('doi_url')
                if doi_url:
                    doi_urls.append(doi_url)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {filename}. Skipping.")

# Count duplicates
doi_counts = Counter(doi_urls)
duplicate_dois = {doi: count for doi, count in doi_counts.items() if count > 1}

# Print the results
print(f"Total duplicate DOI entries: {len(duplicate_dois)}")
for doi, count in duplicate_dois.items():
    print(f"{doi} - {count} occurrences")

