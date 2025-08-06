
import os
import json
import csv

# Directory containing the generated JSON files
json_directory = r'C:\Users\User\Downloads\RSE-Analysis\RS4.0'
# Path to the output CSV file (adjust as needed)
csv_filepath = 'aggregated_output.csv'

# List to store all JSON data entries
data_entries = []

# Loop over every file in the directory
for filename in os.listdir(json_directory):
    if filename.endswith('.json'):
        filepath = os.path.join(json_directory, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                data_entries.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding {filename}: {e}")
                continue

# Collect all keys present in any JSON entry
all_keys = set()
for entry in data_entries:
    all_keys.update(entry.keys())
# Sort keys to produce a consistent column order
all_keys = sorted(all_keys)

# Write the aggregated data to a CSV file
with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=all_keys)
    writer.writeheader()
    for entry in data_entries:
        # Convert list values to strings (semicolon-separated) for CSV output
        for key in entry:
            if isinstance(entry[key], list):
                entry[key] = '; '.join(map(str, entry[key]))
        writer.writerow(entry)

print("CSV file generated successfully at:", csv_filepath)
