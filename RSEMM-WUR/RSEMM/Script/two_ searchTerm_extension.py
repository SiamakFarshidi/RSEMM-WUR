import os
import json

# Path to the taxonomy file
TAXONOMY_FILE = "./IEEE_Taxonomy/ieee_taxonomy_flat_L1_L2_filtered.txt"
OUTPUT_DIR = r"C:\Users\User\Downloads\RS-Zenodo\RS-Repo-Zenodo - v4"

# Load taxonomy terms
with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
    taxonomy_terms = [line.strip() for line in f if line.strip()]

# Function to check if term appears in text
def term_in_text(term, text):
    return term.lower() in (text or "").lower()

# Process each JSON file in OUTPUT_DIR
for fname in os.listdir(OUTPUT_DIR):
    if not fname.endswith(".json"):
        continue

    path = os.path.join(OUTPUT_DIR, fname)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Could not read {path}: {e}")
        continue

    metadata = data.get("metadata", {})
    if not metadata:
        continue

    title = metadata.get("title", "")
    description = metadata.get("description", "") or metadata.get("abstract", "")

    # Convert existing query_term into a list
    query_terms = metadata.get("query_term")
    if isinstance(query_terms, str):
        query_terms = [query_terms]
    elif not isinstance(query_terms, list):
        query_terms = []

    # Check and add taxonomy terms if they appear in title/description
    for term in taxonomy_terms:
        if term not in query_terms and (term_in_text(term, title) or term_in_text(term, description)):
            query_terms.append(term)

    # Update the metadata
    metadata["query_term"] = sorted(set(query_terms))  # Sort & deduplicate

    # Save back to file
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Updated {path}")
    except Exception as e:
        print(f"Could not write {path}: {e}")

print("Done processing all files.")

