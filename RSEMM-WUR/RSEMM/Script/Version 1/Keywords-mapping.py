import os
import json
import re

def normalize_text(text):
    """
    Normalize text for comparison:
      - Convert to lowercase.
      - Replace hyphens and underscores with spaces.
      - Collapse multiple spaces.
    """
    text = text.lower()
    text = re.sub(r'[\-\_]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_keyword_pattern(keyword):
    """
    Build a regex pattern for the normalized keyword such that:
      - It is matched as a whole word/phrase.
      - Optionally allow a trailing "s" or "ing" on the last token.
    For example, for "test" it matches "test", "tests", or "testing"
    but not when "test" is embedded in another word like "latest".
    """
    # Split keyword into tokens
    tokens = keyword.split()
    # Escape each token for regex purposes
    escaped_tokens = [re.escape(token) for token in tokens]
    
    # For the last token, allow an optional trailing "s" or "ing"
    if escaped_tokens:
        escaped_tokens[-1] = escaped_tokens[-1] + r'(?:s|ing)?'
    
    # Join tokens with \s+ to allow flexible whitespace between words
    pattern = r'(?<!\w)' + r'\s+'.join(escaped_tokens) + r'(?!\w)'
    return pattern

# Load keywords from Keywords.json
with open("Keywords.json", "r") as f:
    keywords_data = json.load(f)

# Fields where keywords should be searched
fields_to_search = [
    "GitHub_branches_names",
    "GitHub_commits_messages",
    "GitHub_description",
    "GitHub_full_name",
    "GitHub_topics",
    "Zenodo_description",
    "Zenodo_keywords",
    "Zenodo_title"
]

# Set the directory containing your JSON files
directory = r"C:\Users\User\Downloads\RSE-Analysis\RS_2.0"

# Process each JSON file in the directory (excluding Keywords.json)
for filename in os.listdir(directory):
    if filename.endswith(".json") and filename != "Keywords.json":
        filepath = os.path.join(directory, filename)
        with open(filepath, "r") as f:
            data = json.load(f)

        # Initialize each keyword category as an empty list
        for category in keywords_data:
            data[category] = []

        # Combine text from all target fields for searching
        combined_text = ""
        for field in fields_to_search:
            if field in data and data[field]:
                if isinstance(data[field], list):
                    combined_text += " " + " ".join(data[field])
                elif isinstance(data[field], str):
                    combined_text += " " + data[field]
        normalized_text = normalize_text(combined_text)

        # Search for each keyword in each category
        for category, keywords in keywords_data.items():
            found_keywords = set()  # To ensure uniqueness
            for keyword in keywords:
                # Normalize the keyword for matching
                normalized_keyword = normalize_text(keyword)
                pattern = build_keyword_pattern(normalized_keyword)
                # Use regex search with the constructed pattern
                if re.search(pattern, normalized_text):
                    found_keywords.add(keyword)  # Add the original keyword
            data[category] = list(found_keywords)

        # Save the updated JSON back to file
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Processed file: {filename}")

