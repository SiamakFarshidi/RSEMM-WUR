#!/usr/bin/env python3
"""
classify_repo_with_types.py

Given a GitHub repository URL (hardcoded or passed as an argument),
fetch its metadata, README, and recent commit messages, then heuristically:
  1. Classify as: AIOps, MLOps, Ambiguous, or None
  2. Extract “types” of AIOps / MLOps functionality found
  3. Print results and write them to a JSON file

Usage:
  (If you leave repo_url hardcoded, simply run:)
    python classify_repo_with_types.py

  (Or, to accept a URL via command‐line, run:)
    python classify_repo_with_types.py https://github.com/owner/repo
"""

import sys
import os
import re
import json
import base64
import requests
import openai
import argparse
import datetime

# ----------------------------------------------
# 1. Keyword lists for broad classification
# ----------------------------------------------
AIOPS_KEYWORDS = [
    "aiops",
    "anomaly detection",
    "log analysis",
    "machine learning",
    "metrics",
    "monitoring",
    "root cause analysis",
    "self healing",
    "itops",
]

MLOPS_KEYWORDS = [
    "mlops",
    "ci/cd",
    "continuous integration",
    "continuous deployment",
    "kubeflow",
    "airflow",
    "dvc",
    "ml pipeline",
    "model serving",
    "model deployment",
    "ml lifecycle",
    "pipeline orchestration",
    "mlflow",
]

# ----------------------------------------------
# 2. “Type” keyword groups for AIOps
# ----------------------------------------------
AIOPS_TYPES = {
    "Anomaly Detection": [
        "anomaly detection",
        "outlier detection",
        "unsupervised anomaly",
        "supervised anomaly"
    ],
    "Log Analysis / Parsing": [
        "log analysis",
        "log parsing",
        "drain",
        "grok",
        "logstash",
        "fluentd"
    ],
    "Metrics / Monitoring": [
        "metrics",
        "monitoring dashboard",
        "metric collection",
        "prometheus",
        "grafana"
    ],
    "Root-Cause Analysis": [
        "root cause analysis",
        "rca",
        "cause identification"
    ],
    "Anomaly Prediction / Forecasting": [
        "anomaly prediction",
        "forecasting",
        "time series prediction",
        "trend analysis"
    ],
    "Self-Healing / Automated Remediation": [
        "self healing",
        "self-healing",
        "automated remediation",
        "auto-remediation"
    ],
    "AIOps Infrastructure": [
        "aiops pipeline",
        "aiops framework",
        "aiops infrastructure",
        "data pipeline",
        "stream processing"
    ],
    "Public Dataset / Benchmark": [
        "dataset",
        "benchmark",
        "public data",
        "open dataset"
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
# 4. GitHub API helper functions
# ----------------------------------------------
def get_github_headers():
    """
    Return a headers dict, adding Authorization if GITHUB_TOKEN is set.
    Also set Accept‐header to grab 'topics'.
    """
    headers = {
        "Accept": "application/vnd.github.mercy-preview+json"
    }
    token = "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"
    if token:
        headers["Authorization"] = f"token {token}"
    else:
        print("Warning: No GITHUB_TOKEN found; you will be limited to 60 requests/hour.")
    return headers

def fetch_repo_metadata(owner, repo):
    """
    Fetch repository metadata (name, description, topics) from GitHub API.
    Returns a dict with keys: 'name', 'description', 'topics'.
    Raises RuntimeError on HTTP error.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    resp = requests.get(url, headers=get_github_headers())
    if resp.status_code == 404:
        raise RuntimeError("Repository not found or is private.")
    if resp.status_code == 403:
        # Possibly rate-limited or unauthorized
        raise RuntimeError(f"Access denied or rate limit exceeded: {resp.status_code}")
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
    resp = requests.get(url, headers=get_github_headers())
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
        resp = requests.get(url, headers=get_github_headers())
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
# 5. URL‐parsing helper
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
# 6. Classification logic
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
    Falls back to GPT when ambiguous.
    Returns a tuple: (category, [types]).
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
        # Use GPT to refine both category and types
        llm_result = classify_with_gpt(big_blob)
        category = llm_result.get("category", category)
        types = llm_result.get("types", [])
    # If category == "None", leave types empty

    types = sorted(set(types))
    return meta, category, types

# ----------------------------------------------
# 7. Main: parse arguments + call classify + write JSON
# ----------------------------------------------
def main():

    repo_url = "https://github.com/Netflix/servo"

    try:
        owner, repo = parse_github_url(repo_url)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        meta, category, types = classify_repository(owner, repo)
    except Exception as e:
        print(f"Error while classifying '{owner}/{repo}': {e}")
        sys.exit(1)

    # Print to stdout
    print(f"Repository: {owner}/{repo}")
    print(f"  → Category: {category}")
    if types:
        print("  → Detected types:")
        for t in types:
            print(f"      • {t}")
    else:
        print("  → No specific AIOps/MLOps types detected.")

    # Build a result dictionary
    result = {
        "repository": f"{owner}/{repo}",
        "category": category,
        "types": types,
        "name": meta.get("name", ""),
        "description": meta.get("description", ""),
        "topics": meta.get("topics", []),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    # Write to a JSON file
    output_filename = "classification_result.json"
    try:
        with open(output_filename, "w", encoding="utf-8") as jf:
            json.dump(result, jf, indent=4)
        print(f"\n→ Results also written to '{output_filename}'")
    except Exception as write_err:
        print(f"Error writing JSON file: {write_err}")

if __name__ == "__main__":
    main()
