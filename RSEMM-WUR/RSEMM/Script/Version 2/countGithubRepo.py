import os
import json

# Directory containing JSON files
directory = r"C:\Users\User\Downloads\RS-Zenodo\RS-Repo-Zenodo - v4"
# Counter for JSONs with GitHub repos
github_count = 0

# Loop through all JSON files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.json'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                # Check for 'github' in related_identifiers or custom fields
                found = False
                if 'metadata' in data:
                    metadata = data.get('metadata', {})
                    # Check in related_identifiers
                    for related in metadata.get('related_identifiers', []):
                        if 'github.com' in related.get('identifier', ''):
                            found = True
                            break
                    # Check in custom field (if any)
                    custom = metadata.get('custom', {})
                    for key, value in custom.items():
                        if isinstance(value, str) and 'github.com' in value:
                            found = True
                            break
                if found:
                    github_count += 1
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {filename}. Skipping.")

print(f"Total JSON files with GitHub repositories: {github_count}")

