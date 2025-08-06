import os
import requests
import json
import re
from time import sleep

ZENODO_API_URL = "https://zenodo.org/api/records"

KEYWORDS = [
    "Animal Sciences",
    "Aquaculture and Marine Resource Management",
    "Biobased Sciences",
    "Bioinformatics",
    "Biology",
    "Biosystems Engineering",
    "Biotechnology",
    "Climate Studies",
    "Communication and Innovation (specialisation)",
    "Consumer Studies",
    "Data Science for Food and Health",
    "Development and Rural Innovation",
    "Earth and Environment",
    "Economics of Sustainability",
    "Environmental Sciences",
    "Food Quality Management",
    "Food Safety",
    "Food Technology",
    "Forest and Nature Conservation",
    "Geo-information Science",
    "Geographical Information Management and Applications",
    "Governance of Sustainability Transformations",
    "Health and Society (specialisation)",
    "International Development Studies",
    "International Land and Water Management",
    "Landscape Architecture and Planning",
    "Metropolitan Analysis, Design and Engineering",
    "Molecular Life Sciences",
    "Nutrition and Health",
    "Plant Biotechnology",
    "Plant Sciences",
    "Resilient Farming and Food Systems",
    "Sustainable Business and Innovation",
    "Sustainable Supply Chain Analytics",
    "Tourism, Society and Environment",
    "Urban Environmental Management",
    "Water Technology"
]

# Keywords to detect WUR affiliation
WUR_AFFILIATIONS = [
    "wageningen university",
    "wageningen research",
    "wageningen university & research",
    "wur"
]

def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

def has_wur_affiliation(record):
    """Check if any creator is affiliated with WUR."""
    creators = record.get("metadata", {}).get("creators", [])
    for creator in creators:
        affiliation = creator.get("affiliation") or ""
        if any(wur_keyword in affiliation.lower() for wur_keyword in WUR_AFFILIATIONS):
            return True
    return False

def fetch_software_records(keyword, max_pages=5, page_size=200):
    all_records = []
    seen_ids = set()
    query = f'{keyword} AND type:software'

    for page in range(1, max_pages + 1):
        params = {
            'q': query,
            'size': page_size,
            'page': page
        }
        response = requests.get(ZENODO_API_URL, params=params)
        if response.status_code != 200:
            print(f"[ERROR] Failed to fetch page {page} for '{keyword}': {response.status_code}")
            break

        data = response.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break

        for hit in hits:
            record_id = hit.get("id")
            if record_id not in seen_ids:
                seen_ids.add(record_id)
                all_records.append(hit)

        sleep(0.3)

    return all_records

def save_record_json(record, folder):
    os.makedirs(folder, exist_ok=True)
    record_id = record.get("id")
    filename = os.path.join(folder, f"{record_id}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"  📁 Saved record {record_id} to {folder}")

def main():
    base_output_wur = "wur_software"
    base_output_non_wur = "non_wur_software"
    os.makedirs(base_output_wur, exist_ok=True)
    os.makedirs(base_output_non_wur, exist_ok=True)

    for keyword in KEYWORDS:
        print(f"\n🔍 Fetching software for: {keyword}")
        records = fetch_software_records(keyword)
        print(f"   🔹 Found {len(records)} records")

        for record in records:
            is_wur = has_wur_affiliation(record)
            subfolder = sanitize_filename(keyword)
            output_dir = os.path.join(base_output_wur if is_wur else base_output_non_wur, subfolder)
            save_record_json(record, output_dir)

    print("\n✅ Done organizing software into WUR and non-WUR folders.")

if __name__ == "__main__":
    main()
