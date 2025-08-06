import os
import json
import shutil

# Configuration: adjust these paths as needed
input_dir = r'C:\Users\User\Downloads\RS-Repo-Zenodo - Copy'
output_dir = r'C:\Users\User\Downloads\RS-Repo-Zenodo - Copy\RS-Repo-Zenodo'
keywords_file = './IEEE_Taxonomy/ieee_taxonomy_flat_L1_L2_filtered.txt'

# 1. Load the flat keyword list into a set
with open(keywords_file, 'r', encoding='utf-8') as kf:
    keywords = { line.strip() for line in kf if line.strip() }

# 2. Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# 3. Iterate over each JSON file in the input directory
found_keywords = set()

for fname in os.listdir(input_dir):
    if not fname.lower().endswith('.json'):
        continue

    src_path = os.path.join(input_dir, fname)
    try:
        with open(src_path, 'r', encoding='utf-8') as jf:
            data = json.load(jf)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Skip files that aren’t valid JSON
        continue

    # 4. Extract the 'query_term' from metadata
    query_term = None
    metadata = data.get('metadata', {})
    if isinstance(metadata, dict):
        query_term = metadata.get('query_term')

    # 5. If query_term exists and is in our keyword set, move the file
    if query_term and query_term in keywords:
        dst_path = os.path.join(output_dir, fname)
        shutil.move(src_path, dst_path)
        found_keywords.add(query_term)

# 6. Print out the unique list of keywords that were found
print("Unique keywords found and moved:")
for term in sorted(found_keywords):
    print(" -", term)

