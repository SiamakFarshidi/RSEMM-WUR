import json

# 1. Load the existing IEEE‐taxonomy JSON
with open('ieee_taxonomy_clean.json', 'r', encoding='utf-8') as f:
    taxonomy = json.load(f)

# 2. Collect Level 1 and Level 2 names into one flat list
flat_terms = []
for node in taxonomy:
    flat_terms.append(node['name'])  # Level 1
    for child in node.get('children', []):  # Level 2
        flat_terms.append(child['name'])

# 3. Remove duplicates while preserving order
flat_terms = list(dict.fromkeys(flat_terms))

# 4. Filter out single‐word/general terms (keep only phrases with at least one space)
filtered_terms = [term for term in flat_terms if ' ' in term]

# 5. Write filtered terms to a plain text file, one term per line
with open('ieee_taxonomy_flat_L1_L2_filtered.txt', 'w', encoding='utf-8') as out_f:
    for term in filtered_terms:
        out_f.write(term + '\n')

print(f"Wrote {len(filtered_terms)} keywords to ieee_taxonomy_flat_L1_L2_filtered.txt")
