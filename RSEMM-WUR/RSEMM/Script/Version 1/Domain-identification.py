import os
import json
import re
import logging
import requests
import difflib
from bs4 import BeautifulSoup
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------
# API Token and Configurations
# -------------------------------
TOGETHER_TOKEN = os.environ.get("TOGETHER_TOKEN", "2a599ef68edac73a79b8efa9f3b1fd2e94766ca57c4836fa57565d13cd069a4e")

# -------------------------------
# Allowed Domains Configuration
# -------------------------------
ALLOWED_DOMAINS = sorted([
    "Agrotechnology and Food Sciences",
    "Human Nutrition and Health",
    "Biomolecular Sciences",
    "Biobased Sciences",
    "Animal Sciences",
    "Environmental Sciences",
    "Plant Sciences",
    "Social Sciences",
    "Business Science",
    "Communication and Philosophy",
    "Economics",
    "Space, Place and Society",
    "Sustainable Governance",
    "Computer Science",
    "Gender Studies"
])

# -------------------------------
# Load Spacy Large Model
# -------------------------------
# Make sure you have installed the 'en_core_web_lg' model:
# python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")

# -------------------------------
# Text Cleaning and Augmentation Functions
# -------------------------------
def clean_text(text: str) -> str:
    """
    Remove HTML tags, markdown code fences, and extraneous symbols.
    """
    text = BeautifulSoup(text, 'html.parser').get_text(separator=' ')
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'[^\w\s.,;:!?()-]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def prepare_extended_description(data: dict, title: str, description: str) -> str:
    """
    Combine Zenodo_title and Zenodo_description.
    If the text is less than 40 words, append Zenodo_keywords.
    """
    parts = []
    if title:
        parts.append(title)
    if description:
        parts.append(description)
    base_text = clean_text(" ".join(parts))
    if len(base_text.split()) < 40 and "Zenodo_keywords" in data and isinstance(data["Zenodo_keywords"], list):
        keywords_text = clean_text(" ".join(data["Zenodo_keywords"]))
        base_text += " " + keywords_text
    return base_text

# -------------------------------
# TogetherAI Classification
# -------------------------------
def generate_code_together(prompt: str) -> str:
    url = "https://api.together.xyz/v1/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        "prompt": prompt,
        "max_tokens": 200
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        if result.get("choices") and len(result["choices"]) > 0:
            code_text = result["choices"][0].get("text", "")
            if code_text.startswith("```python"):
                code_text = code_text[len("```python"):].strip()
                if code_text.endswith("```"):
                    code_text = code_text[:-3].strip()
            return code_text.strip()
    except Exception as e:
        logging.error("Error with Together API: %s", e)
    return ""

def extract_final_answer(response: str) -> str:
    """
    If the response contains a phrase like 'final answer is:', extract the text that follows.
    """
    match = re.search(r'final answer is[:\s]*(.*)', response, flags=re.IGNORECASE)
    if match:
        answer = match.group(1).splitlines()[0].strip()
        return answer
    return response.strip()

def get_RSE_domain_together(title: str, description: str, data: dict) -> str:
    """
    Use TogetherAI to classify the domain based on the extended description.
    Returns a valid domain if the response is close to one of the allowed domains; otherwise, returns None.
    """
    extended_desc = prepare_extended_description(data, title, description)
    allowed_str = ", ".join(ALLOWED_DOMAINS)
    prompt = (
        f"Below is research software metadata (cleaned and extended):\n\n"
        f"{extended_desc}\n\n"
        f"From the following list, return exactly ONE of the allowed category names with no extra text, formatting, or commentary.\n"
        f"Allowed categories: {allowed_str}.\n"
        f"For example, if the correct category is Animal Sciences, just output: Animal Sciences"
    )
    response = generate_code_together(prompt)
    logging.info("TogetherAI raw response: %s", response)
    if response:
        extracted = extract_final_answer(response)
        logging.info("Extracted answer: %s", extracted)
        matches = difflib.get_close_matches(extracted, ALLOWED_DOMAINS, n=1, cutoff=0.8)
        if matches:
            return matches[0]
        else:
            logging.info("TogetherAI response did not match any allowed domains.")
    return None

# -------------------------------
# Spacy-based Fallback Classification
# -------------------------------
def get_RSE_domain_spacy(title: str, description: str, data: dict) -> str:
    """
    Use spacy to extract noun phrases from the extended description and
    compare them to each allowed category, returning the one with highest average similarity.
    """
    extended_desc = prepare_extended_description(data, title, description)
    doc = nlp(extended_desc)
    noun_phrases = list(doc.noun_chunks)
    if not noun_phrases:
        noun_phrases = [doc]  # fallback to the entire doc if no noun phrases are found
    
    best_score = -1
    best_category = None
    for category in ALLOWED_DOMAINS:
        cat_doc = nlp(category)
        scores = [cat_doc.similarity(np) for np in noun_phrases]
        avg_score = sum(scores) / len(scores)
        logging.debug("Category: %s, Avg Similarity: %f", category, avg_score)
        if avg_score > best_score:
            best_score = avg_score
            best_category = category
    return best_category if best_category else "Others"

# -------------------------------
# Combined Domain Classification
# -------------------------------
def get_RSE_domain(title: str, description: str, data: dict) -> str:
    """
    First attempt to classify using TogetherAI.
    If that fails to return a valid domain, fall back to spacy-based classification.
    """
    domain = get_RSE_domain_together(title, description, data)
    if domain is None:
        logging.info("Falling back to spacy-based classification.")
        domain = get_RSE_domain_spacy(title, description, data)
    return domain

# -------------------------------
# Process JSON Files and Update with RSEdomain
# -------------------------------
def process_files_add_domain(root_dir: str) -> None:
    """
    Walk through JSON files in the given root directory, determine their domain
    using only Zenodo fields, and update each JSON file with a new "RSEdomain" field.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(dirpath, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logging.error("Error reading %s: %s", filepath, e)
                continue

            title = data.get("Zenodo_title", "")
            description = data.get("Zenodo_description", "")
            
            if not title and not description:
                logging.warning("File %s does not have Zenodo_title or Zenodo_description; skipping classification.", filepath)
                continue

            domain = get_RSE_domain(title, description, data)
            data["RSEdomain"] = domain

            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                logging.info("Updated %s with RSEdomain: %s", filepath, domain)
            except Exception as e:
                logging.error("Error writing %s: %s", filepath, e)

def main() -> None:
    # Update to your actual directory path
    root_dir = r"C:\Users\User\Downloads\RSE-Analysis\RS_2.0"
    process_files_add_domain(root_dir)

if __name__ == '__main__':
    main()
