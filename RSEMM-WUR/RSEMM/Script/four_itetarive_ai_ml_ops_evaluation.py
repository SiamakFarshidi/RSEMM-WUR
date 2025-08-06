#!/usr/bin/env python3
"""
batch_classify_repos.py

Iterate over all JSON files in a given directory, check for GitHub repository URLs
(even if they are embedded in HTML), classify each repo for AIOps/MLOps functionality
(including GPT fallback for “Ambiguous”), and insert/update the keys:
  - "repo-profile" (if not present)
    - "ai-ml-ops": <classification result>

If no GitHub URL is found, still add the "ai-ml-ops" key with null/empty fields.
Handles GitHub API rate limits and transient network errors by sleeping and retrying.
"""

import os
import re
import sys
import json
import time
import base64
import argparse
import datetime
import requests
import openai

# ----------------------------------------------
# 1. Keyword lists for broad classification
# ----------------------------------------------
AIOPS_KEYWORDS = [
    "aiops",
    "root cause analysis",
    "self-healing",
    "auto-remediation",
    "log anomaly detection",
    "intelligent automation",
    "predictive maintenance",
    "machine learning for it operations",
    "it operations analytics",
    "rca automation",
    "ml-driven monitoring",
    "unsupervised log analysis",
]

MLOPS_KEYWORDS = [
    "mlops",
    "ci/cd",
    "continuous integration",
    "continuous deployment",
    "kubeflow",
    "dvc",
    "ml pipeline",
    "model serving",
    "model deployment",
    "ml lifecycle",
    "pipeline orchestration",
    "mlflow",
    "feature store",
    "model monitoring",
    "experiment tracking",
    "data versioning",
    "automated retraining",
    "deployment automation",
    "model registry",
]

# ----------------------------------------------
# 2. “Type” keyword groups for AIOps
# ----------------------------------------------
AIOPS_TYPES = {
    "Anomaly Detection": [
        "anomaly detection",
        "outlier detection",
        "unsupervised anomaly",
        "supervised anomaly",
        "change point detection",
        "anomaly scoring",
        "statistical deviation"
    ],
    "Log Analysis / Parsing": [
        "log analysis",
        "log parsing",
        "drain",
        "grok",
        "logstash",
        "fluentd",
        "log classification",
        "log pattern mining",
        "log vectorization"
    ],
    "Metrics / Monitoring": [
        "metrics",
        "monitoring dashboard",
        "metric collection",
        "prometheus",
        "grafana",
        "telemetry",
        "alert thresholds",
        "real-time monitoring",
        "time series metrics"
    ],
    "Root-Cause Analysis": [
        "root cause analysis",
        "rca",
        "cause identification",
        "dependency mapping",
        "impact analysis",
        "incident correlation",
        "fault localization"
    ],
    "Anomaly Prediction / Forecasting": [
        "anomaly prediction",
        "forecasting",
        "time series prediction",
        "trend analysis",
        "capacity planning",
        "predictive modeling",
        "future anomaly detection"
    ],
    "Self-Healing / Automated Remediation": [
        "self healing",
        "self-healing",
        "automated remediation",
        "auto-remediation",
        "self-recovery",
        "proactive resolution",
        "intelligent remediation"
    ],
    "AIOps Infrastructure": [
        "aiops pipeline",
        "aiops framework",
        "aiops infrastructure",
        "data pipeline",
        "stream processing",
        "event stream",
        "data ingestion",
        "real-time analytics"
    ],
    "Public Dataset / Benchmark": [
        "dataset",
        "benchmark",
        "public data",
        "open dataset",
        "log dataset",
        "kpi dataset",
        "anomaly benchmark",
        "time series dataset"
    ],
}

# ----------------------------------------------
# 3. “Type” keyword groups for MLOps
# ----------------------------------------------
MLOPS_TYPES = {
    "CI/CD / Orchestration": [
        "ci/cd",
        "continuous integration",
        "continuous deployment",
        "pipeline orchestration",
        "jenkins",
        "github actions",
        "airflow",
        "kubeflow"
    ],
    "Data Versioning / DVC": [
        "dvc",
        "data version control",
        "git-lfs",
        "lakeFS"
    ],
    "Experiment Tracking / MLflow": [
        "mlflow",
        "experiment tracking",
        "ml experiments",
        "neptune",
        "weights & biases",
        "wandb"
    ],
    "Model Serving / Deployment": [
        "model serving",
        "model deployment",
        "seldon",
        "bentoml",
        "tensorflow serving",
        "torchserve",
        "kfserving"
    ],
    "Monitoring / Observability": [
        "model monitoring",
        "monitoring models",
        "prometheus",
        "grafana"
    ],
    "Feature Store / Data Pipelines": [
        "feature store",
        "feature engineering",
        "data pipeline"
    ],
    "Dataset Maintenance / ETL": [
        "data preprocessing",
        "data cleaning",
        "etl",
        "etl pipeline"
    ],
}

# ----------------------------------------------
# 4. GitHub API helper functions (with rate-limit handling)
# ----------------------------------------------
GITHUB_TOKEN = "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"  # ← replace with a valid token

def get_github_headers():
    """
    Return a headers dict, adding Authorization if GITHUB_TOKEN is set.
    Also set Accept-header to grab 'topics'.
    """
    headers = {
        "Accept": "application/vnd.github.mercy-preview+json"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    else:
        print("Warning: No GITHUB_TOKEN found; you will be limited to 60 requests/hour.")
    return headers

def safe_get(url, headers, max_retries=5):
    """
    Perform requests.get with rate-limit and network retry handling.
    If a 403 due to rate limit is encountered, sleep until reset.
    On network errors or other transient errors, retry with exponential backoff.
    """
    backoff = 5
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers)
        except requests.RequestException:
            # Network error: sleep and retry
            time.sleep(backoff)
            backoff *= 2
            continue

        # If rate-limited (403 + X-RateLimit-Remaining=0), sleep until reset
        if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            reset_ts = resp.headers.get("X-RateLimit-Reset")
            if reset_ts is not None:
                reset_time = int(reset_ts)
                sleep_duration = max(reset_time - int(time.time()) + 5, 5)
                print(f"Rate limit exceeded. Sleeping for {sleep_duration} seconds...")
                time.sleep(sleep_duration)
                continue
        return resp

    # After retries, return last response anyway
    return resp

def fetch_repo_metadata(owner, repo):
    """
    Fetch repository metadata (name, description, topics) from GitHub API.
    Returns a dict with keys: 'name', 'description', 'topics'.
    Raises RuntimeError on HTTP error (404/other).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    resp = safe_get(url, get_github_headers())
    if resp.status_code == 404:
        raise RuntimeError("Repository not found or is private.")
    if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
        # Should have retried already; if still 403, bubble up
        raise RuntimeError("Access denied or rate limit exceeded.")
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch repo metadata: {resp.status_code} {resp.text}")
    data = resp.json()
    return {
        "name": data.get("name") or "",
        "description": data.get("description") or "",
        "topics": data.get("topics") or [],
    }

def fetch_readme(owner, repo):
    """
    Fetch the README content (decoded from Base64 if found).
    If no README or an error, return an empty string.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    resp = safe_get(url, get_github_headers())
    if resp.status_code == 200:
        data = resp.json()
        content_b64 = data.get("content", "")
        try:
            decoded = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
        except Exception:
            decoded = ""
        return decoded
    # 404 means no README; other errors we ignore and return empty
    return ""

def fetch_commit_messages(owner, repo, max_commits=300):
    """
    Fetch recent commit messages (up to max_commits) from the default branch.
    Handles pagination. Returns a list of commit message strings.
    """
    messages = []
    page = 1
    per_page = 100
    while len(messages) < max_commits:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page={per_page}&page={page}"
        resp = safe_get(url, get_github_headers())
        if resp.status_code != 200:
            break
        data = resp.json()
        if not data:
            break
        for c in data:
            msg = c.get("commit", {}).get("message", "")
            if msg:
                messages.append(msg)
        if len(data) < per_page:
            break
        page += 1
    return messages[:max_commits]

# ----------------------------------------------
# 5. URL-parsing helper (UNCHANGED)
# ----------------------------------------------
def parse_github_url(url: str):
    """
    Given a GitHub URL (HTTPS or SSH), return (owner, repo).
    Examples:
      - https://github.com/owner/repo
      - https://www.github.com/owner/repo/
      - git@github.com:owner/repo.git
    Raises ValueError if it doesn't look like a GitHub repo URL.
    """
    url = url.strip().rstrip("/")
    url = re.sub(r"\.git$", "", url)
    # Strip optional www.
    url = re.sub(r"^https?://www\.", "https://", url)

    # HTTPS style: https://github.com/owner/repo
    m = re.match(r"^https?://github\.com/([^/]+)/([^/]+)$", url)
    if m:
        return m.group(1), m.group(2)

    # SSH style: git@github.com:owner/repo
    m = re.match(r"^git@github\.com:([^/]+)/([^/]+)$", url)
    if m:
        return m.group(1), m.group(2)

    # If URL includes extra path segments (e.g., /tree/main), strip after repo
    m = re.match(r"^https?://github\.com/([^/]+)/([^/]+)/", url)
    if m:
        return m.group(1), m.group(2)

    raise ValueError(f"Could not parse GitHub URL '{url}' as 'owner/repo'.")

# ----------------------------------------------
# 6. Classification logic (UNCHANGED)
# ----------------------------------------------
def normalize_text(text: str):
    """
    Lowercase and replace punctuation with spaces to catch variants
    like 'ml-pipeline' or 'CI_CD'.
    """
    text_lower = text.lower()
    return re.sub(r"[^\w\s]", " ", text_lower)

def classify_text(text: str):
    """
    Given a blob of text, check for any AIOps vs. MLOps keywords.
    Returns two booleans: (found_aiops, found_mlops).
    """
    text_norm = normalize_text(text)
    found_aiops = any(kw in text_norm for kw in (k.lower() for k in AIOPS_KEYWORDS))
    found_mlops = any(kw in text_norm for kw in (k.lower() for k in MLOPS_KEYWORDS))
    return found_aiops, found_mlops

def extract_types(text: str, type_dict: dict):
    """
    Given a blob of text and a dict mapping type_name -> [keywords],
    return a list of type_names whose keywords appear in text.
    """
    text_norm = normalize_text(text)
    detected = []
    for type_name, kw_list in type_dict.items():
        for kw in kw_list:
            if kw.lower() in text_norm:
                detected.append(type_name)
                break
    return detected

def classify_with_gpt(blob: str):
    """
    Use GPT-3.5-turbo to classify an ambiguous repository.
    Returns a dict with keys: 'category', 'types', 'reasoning'.
    """
    # Ensure OPENAI_API_KEY is set in your environment
    openai.api_key = "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"
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot call GPT.")

    prompt = (
        "Below is combined text from a GitHub repository (name, description, topics, README, commits).\n\n"
        f"{blob}\n\n"
        "Please respond as JSON with keys:\n"
        "  • category: one of \"AIOps\", \"MLOps\", \"Ambiguous\", or \"None\".\n"
        "  • types: a JSON array of specific sub-types (e.g., \"Anomaly Detection\", \"Model Serving\").\n"
        "  • reasoning: a brief explanation of why you chose that category.\n"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a software categorization assistant. "
                    "Classify repositories into AIOps, MLOps, Ambiguous, or None, and return JSON."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=400,
    )
    text = response.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"Could not parse JSON from GPT response: {text}")

def classify_repository(owner: str, repo: str):
    """
    Fetch metadata, README, and commit messages; combine into one large string,
    then decide (AIOps, MLOps, Ambiguous, None) and extract types.
    Uses GPT fallback if category == "Ambiguous".
    Returns a tuple: (meta_dict, category_str, types_list).
    """
    meta = fetch_repo_metadata(owner, repo)
    readme_content = fetch_readme(owner, repo)
    commit_messages = fetch_commit_messages(owner, repo)

    combined_parts = [
        meta["name"],
        meta["description"],
        *meta["topics"],
        readme_content,
        "\n".join(commit_messages) if commit_messages else ""
    ]
    big_blob = "\n".join(filter(None, combined_parts))

    # 1. Keyword-based classification
    found_aiops, found_mlops = classify_text(big_blob)
    if found_aiops and not found_mlops:
        category = "AIOps"
    elif found_mlops and not found_aiops:
        category = "MLOps"
    elif not found_aiops and not found_mlops:
        category = "None"
    else:
        category = "Ambiguous"

    types = []
    if category == "AIOps":
        types = extract_types(big_blob, AIOPS_TYPES)
    elif category == "MLOps":
        types = extract_types(big_blob, MLOPS_TYPES)
    elif category == "Ambiguous":
        # GPT fallback for ambiguous
        llm_result = classify_with_gpt(big_blob)
        category = llm_result.get("category", category)
        types = llm_result.get("types", [])
    # If category == "None", leave types empty

    types = sorted(set(types))
    return meta, category, types

# ----------------------------------------------
# 7. Helper: find first GitHub URL in JSON record (UPDATED)
# ----------------------------------------------
def find_github_url_in_record(record: dict):
    """
    Search through the JSON record for any GitHub URL. If a string contains HTML
    with an <a href="…github.com/…">, extract that URL. Otherwise, if a string is
    plain text containing "github.com", use a regex to pull out the URL.
    Returns the first matching URL, or None if not found.
    """
    # 1) Check metadata.related_identifiers[*].identifier first
    md = record.get("metadata", {})
    related = md.get("related_identifiers", [])
    for entry in related:
        identifier = entry.get("identifier", "")
        if isinstance(identifier, str) and "github.com" in identifier:
            # Try to extract a clean URL out of that string
            urls = re.findall(r"https?://github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+", identifier)
            if urls:
                return urls[0]
            else:
                # If no sub-match, maybe it's already a clean URL
                return identifier.strip()

    # 2) Otherwise, recursively scan the entire JSON object for any string containing 'github.com'
    def scan(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                res = scan(v)
                if res:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = scan(item)
                if res:
                    return res
        elif isinstance(obj, str):
            if "github.com" in obj:
                # First try to pull out any "https://github.com/owner/repo" inside the string
                urls = re.findall(r"https?://github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+", obj)
                if urls:
                    return urls[0]
                else:
                    # Fallback: treat the entire string as a URL candidate (hoping parse_github_url will catch it)
                    return obj.strip()
        return None

    return scan(record)

# ----------------------------------------------
# 8. Main: iterate over JSON files, classify, and update
# ----------------------------------------------

def getAIMLOPsEvaluation(filepath):
    # 1) Load JSON
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error loading JSON: {e}. Skipping.")
        return {}

    # 2) Find GitHub URL
    gh_url = find_github_url_in_record(data)
    classification_result = {}
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if gh_url:
        try:
            owner, repo = parse_github_url(gh_url)
        except ValueError:
            print(f"  Found GitHub-like string but could not parse URL: '{gh_url}'.")
            owner = repo = None

        if owner and repo:
            try:
                meta, category, types = classify_repository(owner, repo)
                classification_result = {
                    "repository": f"{owner}/{repo}",
                    "category": category,
                    "types": types,
                    "name": meta.get("name", ""),
                    "description": meta.get("description", ""),
                    "topics": meta.get("topics", []),
                    "timestamp": timestamp
                }
                print(f"  Classified '{owner}/{repo}' → {category}, types: {types}")
            except Exception as e:
                print(f"  Error classifying '{owner}/{repo}': {e}")
                classification_result = {
                    "repository": f"{owner}/{repo}",
                    "category": None,
                    "types": [],
                    "name": "",
                    "description": "",
                    "topics": [],
                    "timestamp": timestamp,
                    "error": str(e)
                }
        else:
            classification_result = {
                "repository": None,
                "category": None,
                "types": [],
                "name": "",
                "description": "",
                "topics": [],
                "timestamp": timestamp
            }
    else:
        print("  No GitHub URL found in record.")
        classification_result = {
            "repository": None,
            "category": None,
            "types": [],
            "name": "",
            "description": "",
            "topics": [],
            "timestamp": timestamp
        }

    # 3) Ensure "repo-profile" exists
    if not isinstance(data.get("repo-profile"), dict):
        data["repo-profile"] = {}

    # 4) Insert/update "ai-ml-ops"
    data["repo-profile"]["ai-ml-ops"] = classification_result

    # 5) Write back to the same file
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"  Updated with 'repo-profile.ai-ml-ops'.")
    except Exception as e:
        print(f"  Error writing JSON: {e}")


def process_directory(input_dir: str):
    """
    Walk through all .json files in input_dir (non-recursive), process each one:
      - Load JSON
      - Find GitHub URL (even if embedded in HTML)
      - If found: classify repo and build classification_result
      - If not found: build null classification_result
      - Insert/update record["repo-profile"]["ai-ml-ops"] = classification_result
      - Write JSON back to file (overwrite)
    """
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".json"):
            continue
        filepath = os.path.join(input_dir, filename)
        print(f"\nProcessing '{filepath}'...")

        getAIMLOPsEvaluation(filepath)

if __name__ == "__main__":
    input_dir = r"C:\Users\User\Downloads\RS-Zenodo\RS-Repo-Zenodo - v4"


    if not os.path.isdir(input_dir):
        print(f"Error: '{input_dir}' is not a directory or does not exist.")
        sys.exit(1)

    process_directory(input_dir)
