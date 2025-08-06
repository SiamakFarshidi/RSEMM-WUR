#!/usr/bin/env python3
"""
all_repo_analysis.py

A monolithic script that combines:
  1. FAIRness assessment
  2. SE practices analysis
  3. Code-generation detection
  4. AI/ML/Ops detection

for every Zenodo JSON file in a given directory. It centralizes all helper
functions (rate-limit handling, GitHub/OpenAI wrappers, parsing, etc.)
without dropping any functionality from the original sub-scripts.
"""

import os
import sys
import re
import json
import time
import base64
import shutil
import stat
import tempfile
import requests
import openai
import socket
import urllib.request
import urllib.error
import datetime
from pathlib import Path
from dateutil.relativedelta import relativedelta
from requests.exceptions import ConnectionError, Timeout, RequestException
from openai.error import RateLimitError

# ─────────────────────────────────────────────────────────────────────────────
# 0) GLOBAL CONFIGURATION: TOKENS, PATHS, ETC.
# ─────────────────────────────────────────────────────────────────────────────

# Put your tokens here (only in this file).
GITHUB_TOKEN = "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"
OPENAI_API_KEY = "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"

# Export OpenAI key as env var so OpenAI client libraries pick it up.
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Directory containing all Zenodo JSON files
INPUT_DIR = r"C:\Users\User\Downloads\RS-Zenodo\RS-Repo-Zenodo - v3"

# Progress file to track processed Zenodo JSON filenames
PROGRESS_FILE = "processed_files.json"

# OpenAI / GitHub rate-limit configuration
OPENAI_MAX_RETRIES = 5
OPENAI_BACKOFF_BASE = 5   # seconds, doubles each retry
GITHUB_DEFAULT_SLEEP = 5  # seconds if a generic GitHub error (not full rate-limit)
NETWORK_SLEEP = 30        # seconds to wait on network errors

# ─────────────────────────────────────────────────────────────────────────────
# 1) HELPER: GITHUB REQUESTS WITH RATE-LIMIT HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def get_github_headers():
    """
    Return headers for GitHub API calls, adding Authorization if GITHUB_TOKEN is set.
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "all-repo-analysis-script"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers

def handle_github_rate_limit(resp: requests.Response):
    """
    If response indicates GitHub rate-limit (status 403 or 429 with X-RateLimit-Remaining=0),
    parse X-RateLimit-Reset and sleep until that time + buffer. Otherwise, sleep default.
    """
    if resp.status_code in (403, 429):
        rem = resp.headers.get("X-RateLimit-Remaining")
        if rem == "0":
            reset_ts = resp.headers.get("X-RateLimit-Reset")
            if reset_ts:
                try:
                    reset_ts = int(reset_ts)
                    now = int(time.time())
                    wait = reset_ts - now + 5
                    if wait > 0:
                        print(f"GitHub rate limit hit. Sleeping for {wait} seconds…")
                        time.sleep(wait)
                        return
                except ValueError:
                    pass
        # If not a full exhaustion or no reset header, sleep default
        print(f"GitHub returned {resp.status_code}. Sleeping for {GITHUB_DEFAULT_SLEEP}s…")
        time.sleep(GITHUB_DEFAULT_SLEEP)

def github_get(url: str, params: dict = None) -> requests.Response:
    """
    Wrapper around requests.get for GitHub API. Retries on rate-limit or network errors.
    """
    while True:
        try:
            resp = requests.get(url, headers=get_github_headers(), params=params or {}, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"Network error hitting GitHub ({e}). Sleeping for {NETWORK_SLEEP}s…")
            time.sleep(NETWORK_SLEEP)
            continue
        if resp.status_code == 200:
            return resp
        handle_github_rate_limit(resp)
        # Then loop & retry

def github_head(url: str) -> requests.Response:
    """
    Wrapper around requests.head for GitHub/raw URLs. Retries on rate-limit or network errors.
    """
    while True:
        try:
            resp = requests.head(url, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"Network error on HEAD {url} ({e}). Sleeping for {NETWORK_SLEEP}s…")
            time.sleep(NETWORK_SLEEP)
            continue
        if resp.status_code == 200:
            return resp
        if resp.status_code in (403, 429):
            handle_github_rate_limit(resp)
            continue
        # Other non-200: return as-is
        return resp

def get_rate_limit_reset_time() -> int:
    """
    Query GitHub /rate_limit endpoint to find the 'core' reset time (epoch seconds).
    Fallback: now + 60s if any error.
    """
    url = "https://api.github.com/rate_limit"
    try:
        resp = requests.get(url, headers=get_github_headers(), timeout=10)
        if resp.status_code != 200:
            return int(time.time()) + 60
        data = resp.json()
        core = data.get("resources", {}).get("core", {})
        reset_ts = core.get("reset", int(time.time()) + 60)
        return int(reset_ts)
    except Exception:
        return int(time.time()) + 60

def sleep_until_rate_limit_resets():
    """
    Sleep until GitHub rate-limit reset plus a small buffer.
    """
    now = int(time.time())
    reset_ts = get_rate_limit_reset_time()
    wait = max(reset_ts - now + 5, 5)
    print(f"→ GitHub rate limit exhausted. Sleeping for {wait} seconds (until reset)…")
    time.sleep(wait)

def is_rate_limit_exception(exc: Exception) -> bool:
    """
    Heuristic: True if exception indicates GitHub rate-limit (e.g., 403 + 'rate limit' in message).
    """
    msg = str(exc).lower()
    return ("rate limit" in msg) or ("403" in msg and "github" in msg)

def is_network_exception(exc: Exception) -> bool:
    """
    Returns True if exception is a network/dns resolution error.
    """
    if isinstance(exc, RequestException):
        inner = str(exc).lower()
        if "failed to resolve" in inner or "nameresolutionerror" in inner or "getaddrinfo" in inner:
            return True
        if isinstance(exc, (ConnectionError, Timeout)):
            return True
    if isinstance(exc, socket.gaierror):
        return True
    return False

def wait_for_network():
    """
    Loop until GitHub root is reachable. Sleeps NETWORK_SLEEP between attempts.
    """
    while True:
        try:
            requests.head("https://api.github.com", timeout=5)
            return
        except Exception:
            print("→ Network still unavailable. Retrying in 30 seconds…")
            time.sleep(NETWORK_SLEEP)

# ─────────────────────────────────────────────────────────────────────────────
# 2) HELPER: OPENAI REQUESTS WITH RATE-LIMIT HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def call_openai_chat_completion(messages: list, model="gpt-3.5-turbo", max_tokens=400, temperature=0.0):
    """
    Wrapper around openai.ChatCompletion.create with retry-on-RateLimitError.
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot call GPT.")

    backoff = OPENAI_BACKOFF_BASE
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return resp
        except RateLimitError:
            if attempt < OPENAI_MAX_RETRIES - 1:
                print(f"OpenAI rate limit hit (attempt {attempt+1}/{OPENAI_MAX_RETRIES}). Sleeping for {backoff} seconds…")
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                raise RuntimeError("OpenAI rate limit exceeded on final attempt.")
        except Exception as e:
            raise RuntimeError(f"OpenAI error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 3) HELPER: SOURCE MANAGEMENT (CAN REMOVE TEMP DIR) FOR SE PRACTICES
# ─────────────────────────────────────────────────────────────────────────────

def on_rm_error(func, path, exc_info):
    """
    Handler for shutil.rmtree to make files writable if needed.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# 4) HELPER: PARSE GITHUB OWNER/REPO FROM URL
# ─────────────────────────────────────────────────────────────────────────────

def parse_owner_repo(url: str):
    """
    Given a GitHub URL (HTTPS or SSH), return (owner, repo).
    Accepts:
      - https://github.com/owner/repo
      - https://github.com/owner/repo/
      - https://github.com/owner/repo/tree/...
      - git@github.com:owner/repo.git
    Raises ValueError if not parseable.
    """
    u = url.strip().rstrip("/")
    u = re.sub(r"\.git$", "", u)
    # HTTPS style: https://github.com/owner/repo[/...]
    m = re.match(r"^https?://github\.com/([^/]+)/([^/]+)(?:/.*)?$", u)
    if m:
        return m.group(1), m.group(2)
    # SSH style: git@github.com:owner/repo
    m = re.match(r"^git@github\.com:([^/]+)/([^/]+)$", u)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(f"Cannot parse owner/repo from '{url}'")

# ─────────────────────────────────────────────────────────────────────────────
# 5) SECTION: AI/ML/Ops DETECTION (from ai-ml-ops-detection.py)
# ─────────────────────────────────────────────────────────────────────────────

# 5.1. Keyword lists for broad classification
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

# 5.2. “Type” keyword groups for AIOps
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

# 5.3. “Type” keyword groups for MLOps
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

def normalize_text(text: str) -> str:
    """
    Lowercase and replace punctuation with spaces for keyword matching.
    """
    text_lower = text.lower()
    return re.sub(r"[^\w\s]", " ", text_lower)

def classify_text(text: str):
    """
    Given a blob of text, return (found_aiops, found_mlops).
    """
    text_norm = normalize_text(text)
    found_aiops = any(kw in text_norm for kw in (k.lower() for k in AIOPS_KEYWORDS))
    found_mlops = any(kw in text_norm for kw in (k.lower() for k in MLOPS_KEYWORDS))
    return found_aiops, found_mlops

def extract_types(text: str, type_dict: dict):
    """
    Given a blob of text and type_dict mapping type_name -> [keywords],
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

def fetch_repo_metadata(owner: str, repo: str):
    """
    Fetch repository metadata from GitHub API: name, description, topics.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    resp = github_get(url)
    data = resp.json()
    return {
        "name": data.get("name") or "",
        "description": data.get("description") or "",
        "topics": data.get("topics") or [],
    }

def fetch_readme(owner: str, repo: str):
    """
    Fetch the README content via GitHub API, decode from base64.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    resp = github_get(url)
    if resp.status_code == 200:
        data = resp.json()
        content_b64 = data.get("content", "")
        try:
            decoded = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
        except Exception:
            decoded = ""
        return decoded
    return ""

def fetch_commit_messages(owner: str, repo: str, max_commits: int = 300):
    """
    Fetch up to max_commits recent commit messages, handling pagination.
    """
    messages = []
    page = 1
    per_page = 100
    while len(messages) < max_commits:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params = {"per_page": per_page, "page": page}
        resp = github_get(url, params=params)
        if resp.status_code != 200:
            break
        data = resp.json()
        if not data:
            break
        for c in data:
            msg = c.get("commit", {}).get("message", "")
            if msg:
                messages.append(msg)
            if len(messages) >= max_commits:
                break
        if len(data) < per_page:
            break
        page += 1
    return messages[:max_commits]

def classify_with_gpt_repo(blob: str):
    """
    Use GPT to classify an ambiguous repository blob.
    Returns dict with keys: 'category', 'types', 'reasoning'.
    """
    prompt = (
        "Below is combined text from a GitHub repository (name, description, topics, README, commits).\n\n"
        f"{blob}\n\n"
        "Please respond as JSON with keys:\n"
        "  • category: one of \"AIOps\", \"MLOps\", \"Ambiguous\", or \"None\".\n"
        "  • types: a JSON array of specific sub-types (e.g., \"Anomaly Detection\", \"Model Serving\").\n"
        "  • reasoning: a brief explanation of why you chose that category.\n"
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a software categorization assistant. "
                "Classify repositories into AIOps, MLOps, Ambiguous, or None, and return JSON."
            )
        },
        {"role": "user", "content": prompt}
    ]
    resp = call_openai_chat_completion(messages, model="gpt-3.5-turbo", max_tokens=400, temperature=0.0)
    text = resp.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"Could not parse JSON from GPT response: {text}")

def classify_repository(owner: str, repo: str):
    """
    Fetch metadata, README, commit messages, combine into blob, then:
      - Keyword-based classification into AIOps/MLOps/None/Ambiguous.
      - If Ambiguous: call GPT to refine category/types.
    Returns (meta_dict, category, types_list).
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
        llm_result = classify_with_gpt_repo(big_blob)
        category = llm_result.get("category", category)
        types = llm_result.get("types", [])
    # If "None", leave types empty

    types = sorted(set(types))
    return meta, category, types

# ─────────────────────────────────────────────────────────────────────────────
# 6) SECTION: FAIRNESS ASSESSMENT (from assessFAIRness.py)
# ─────────────────────────────────────────────────────────────────────────────

def assess_fairness_metadata(record: dict) -> dict:
    """
    Return a dict mapping each FAIR4RS sub-principle to "yes"/"no"/"partial"/"unknown"/"not applicable".
    Sub-principles: F1, F1.1, F1.2, F2, F3, F4, A1, A1.2, A2, I1, I2, R1.1, R1.2, R2, R3.
    """
    results = {}
    metadata = record.get("metadata", {})

    # F1: DOI present?
    results["doi_present"] = "yes" if metadata.get("doi") else "no"

    # F1.1: Subcomponent identifiers (not in JSON => unknown)
    results["subcomponent_identifiers"] = "unknown"

    # F1.2: Version identifiers? Look for metadata["relations"]["version"]
    relations = metadata.get("relations", {}).get("version", [])
    results["version_identifiers"] = "yes" if relations else "no"

    # F2: Rich metadata? (title, creators, description, license, language)
    required_f2 = ["title", "creators", "description", "license", "language"]
    missing_f2 = [f for f in required_f2 if not metadata.get(f)]
    results["rich_metadata_present"] = "yes" if not missing_f2 else "no"

    # F3: Metadata explicitly includes its own DOI?
    results["metadata_includes_doi"] = "yes" if metadata.get("doi") and record.get("doi") else "no"

    # F4: Metadata accessible? (we have it in memory => yes)
    results["metadata_accessible"] = "yes" if metadata else "no"

    # A1: Software retrievable by identifier? (links.doi + files list)
    links = record.get("links", {})
    files = record.get("files", [])
    results["software_retrievable"] = "yes" if (links.get("doi") and files) else "no"

    # A1.2: Authentication required? (if access_right == "open" => not applicable; else unknown)
    results["authentication_required"] = (
        "not applicable" if metadata.get("access_right") == "open" else "unknown"
    )

    # A2: Metadata remain accessible if software removed? (we still have metadata => yes)
    results["metadata_persistent"] = "yes" if metadata else "no"

    # I1: Uses standard data formats? (cannot infer from JSON => unknown)
    results["uses_standard_data_formats"] = "unknown"

    # I2: Qualified refs to external objects? (look for DOI URLs in metadata["references"])
    refs = metadata.get("references", []) or []
    qualified = [r for r in refs if isinstance(r, str) and r.startswith("https://doi.org")]
    results["qualified_references_to_objects"] = "yes" if qualified else "no"

    # R1.1: Clear license present? (metadata["license"]["id"])
    results["clear_license"] = "yes" if metadata.get("license", {}).get("id") else "no"

    # R1.2: Provenance via ORCID? (all creators have orcid => yes; some => partial; none => no)
    creators = metadata.get("creators", []) or []
    if creators:
        orcids = [c.get("orcid") for c in creators]
        if all(orcids):
            results["provenance_with_orcid"] = "yes"
        elif any(orcids):
            results["provenance_with_orcid"] = "partial"
        else:
            results["provenance_with_orcid"] = "no"
    else:
        results["provenance_with_orcid"] = "no"

    # R2: Qualified refs to other software? (cannot infer => unknown)
    results["qualified_references_to_software"] = "unknown"

    # R3: Meets community standards (CI, linters)? (cannot infer from JSON => unknown)
    results["ci_or_community_standards"] = "unknown"

    return results

def categorize_principle(scores: list, thresholds=(0.75, 0.5)) -> str:
    """
    Given numeric scores [0, 0.5, 1], return "High"/"Medium"/"Low".
    """
    if not scores:
        return "Low"
    avg = sum(scores) / len(scores)
    if avg >= thresholds[0]:
        return "High"
    elif avg >= thresholds[1]:
        return "Medium"
    else:
        return "Low"

def score_value(val: str) -> float:
    """
    Map "yes"→1.0, "partial"→0.5, others→0.0.
    """
    return {"yes": 1.0, "partial": 0.5}.get(val, 0.0)

def estimate_fairness_from_subprinciples(sub: dict) -> dict:
    """
    From sub-principle results (strings), compute:
      {
        "principle_categories": {Findable, Accessible, Interoperable, Reusable},
        "overall_fairness": "High"/"Medium"/"Low"
      }
    """
    groups = {
        "Findable": [
            "doi_present",
            "subcomponent_identifiers",
            "version_identifiers",
            "rich_metadata_present",
            "metadata_includes_doi",
            "metadata_accessible"
        ],
        "Accessible": [
            "software_retrievable",
            "authentication_required",
            "metadata_persistent"
        ],
        "Interoperable": [
            "uses_standard_data_formats",
            "qualified_references_to_objects"
        ],
        "Reusable": [
            "clear_license",
            "provenance_with_orcid",
            "qualified_references_to_software",
            "ci_or_community_standards"
        ]
    }

    principle_categories = {}
    principle_scores = []

    for principle, keys in groups.items():
        numeric_scores = [score_value(sub.get(k, "unknown")) for k in keys]
        cat = categorize_principle(numeric_scores)
        principle_categories[principle] = cat
        avg_val = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
        principle_scores.append(avg_val)

    overall_avg = sum(principle_scores) / len(principle_scores) if principle_scores else 0.0
    if overall_avg >= 0.75:
        overall = "High"
    elif overall_avg >= 0.5:
        overall = "Medium"
    else:
        overall = "Low"

    return {
        "principle_categories": principle_categories,
        "overall_fairness": overall
    }

def extract_git_url_from_zenodo(record: dict) -> str:
    """
    1) Check metadata["related_identifiers"] for a "scheme":"url" entry starting with "https://github.com/".
    2) If none, search metadata["description"] for the first "https://github.com/owner/repo".
    Return the base "https://github.com/owner/repo" (strip any /tree/... suffix) or None.
    """
    metadata = record.get("metadata", {})

    # (1) related_identifiers
    for rid in metadata.get("related_identifiers", []) or []:
        ident = rid.get("identifier", "")
        scheme = rid.get("scheme", "").lower()
        if scheme == "url" and ident.startswith("https://github.com/"):
            m = re.match(r"(https?://github\.com/[^/]+/[^/]+)", ident)
            if m:
                return m.group(1)

    # (2) fallback: search description
    desc = metadata.get("description", "") or ""
    m2 = re.search(r"https?://github\.com/[^/\s]+/[^/\s]+", desc)
    if m2:
        return m2.group(0)

    return None

def check_repository_exists(owner: str, repo: str) -> bool:
    """
    Check if GET /repos/{owner}/{repo} returns 200.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    resp = github_get(url)
    return resp.status_code == 200

def check_license_exists(owner: str, repo: str) -> bool:
    """
    Check if GET /repos/{owner}/{repo}/license returns 200.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/license"
    resp = github_get(url)
    return resp.status_code == 200

def fetch_readme_text(owner: str, repo: str) -> str:
    """
    Fetch repository README via GET /repos/{owner}/{repo}/readme,
    decode base64, return text or "".
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    resp = github_get(url)
    if resp.status_code != 200:
        return ""
    try:
        j = resp.json()
        b64 = j.get("content", "")
        data = base64.b64decode(b64)
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def check_registry_badge_in_readme(readme_text: str) -> bool:
    """
    Look for Zenodo or shields.io badge: "zenodo.org/badge" or "shields.io/badge/zenodo".
    """
    return bool(re.search(r"(zenodo\.org/badge|shields\.io/badge/zenodo)", readme_text, re.IGNORECASE))

def check_citation_file_exists(owner: str, repo: str) -> bool:
    """
    Check if GET /repos/{owner}/{repo}/contents/CITATION.cff or CITATION.md returns 200.
    """
    url1 = f"https://api.github.com/repos/{owner}/{repo}/contents/CITATION.cff"
    resp1 = github_get(url1)
    if resp1.status_code == 200:
        return True
    url2 = f"https://api.github.com/repos/{owner}/{repo}/contents/CITATION.md"
    resp2 = github_get(url2)
    return resp2.status_code == 200

def check_quality_badge_in_readme(readme_text: str) -> bool:
    """
    Look for an OpenSSF or bestpractices.dev badge in README.
    """
    return bool(re.search(r"(bestpractices\.dev|shields\.io/badge/best_practices)", readme_text, re.IGNORECASE))

def run_howfairis_checks(repo_url: str) -> dict:
    """
    Perform 5 checks via GitHub API / README parsing:
      1) Public repo exists?
      2) License exists?
      3) Registry badge in README?
      4) CITATION.cff or CITATION.md exists?
      5) Quality badge in README?
    Return:
      {
        "howfairis_score": 0–5,
        "howfairis_details": {
            "repository": "yes"/"no",
            "license": "yes"/"no",
            "registry": "yes"/"no",
            "citation": "yes"/"no",
            "quality": "yes"/"no"
        }
      }
    """
    details = {"repository": "no", "license": "no", "registry": "no", "citation": "no", "quality": "no"}
    score = 0

    try:
        owner, repo = parse_owner_repo(repo_url)
    except ValueError:
        return {"howfairis_score": 0, "howfairis_details": details}

    # 1) Public repo?
    if check_repository_exists(owner, repo):
        details["repository"] = "yes"
        score += 1

        # 2) License?
        if check_license_exists(owner, repo):
            details["license"] = "yes"
            score += 1

        # 3) & 5) Fetch README once
        readme_text = fetch_readme_text(owner, repo)

        # 3) Registry badge?
        if check_registry_badge_in_readme(readme_text):
            details["registry"] = "yes"
            score += 1

        # 5) Quality badge?
        if check_quality_badge_in_readme(readme_text):
            details["quality"] = "yes"
            score += 1

        # 4) Citation file?
        if check_citation_file_exists(owner, repo):
            details["citation"] = "yes"
            score += 1

    return {"howfairis_score": score, "howfairis_details": details}

# ─────────────────────────────────────────────────────────────────────────────
# 7) SECTION: SE PRACTICES ANALYSIS (from SE-practises-analysis.py)
# ─────────────────────────────────────────────────────────────────────────────

# 7.1. Definitions for comment syntax
LINE_COMMENT_EXTENSIONS = {
    '.py':  '#',
    '.sh':  '#',
    '.rb':  '#',
}

BLOCK_COMMENT_EXTENSIONS = {
    '.c':    ('/*', '*/', '//'),
    '.cpp':  ('/*', '*/', '//'),
    '.h':    ('/*', '*/', '//'),
    '.hpp':  ('/*', '*/', '//'),
    '.java': ('/*', '*/', '//'),
    '.js':   ('/*', '*/', '//'),
    '.jsx':  ('/*', '*/', '//'),
    '.cc':   ('/*', '*/', '//'),
    '.cxx':  ('/*', '*/', '//'),
    '.cs':   ('/*', '*/', '//'),
    '.go':   ('/*', '*/', '//'),
    '.kt':   ('/*', '*/', '//'),
    '.kts':  ('/*', '*/', '//'),
}

def analyze_source_file(path: str):
    """
    Count code vs. comment lines for a single source file at 'path'.
    Returns (code_count, comment_count).
    """
    _, ext = os.path.splitext(path.lower())
    code_ct = 0
    comment_ct = 0

    if ext in LINE_COMMENT_EXTENSIONS:
        line_marker = LINE_COMMENT_EXTENSIONS[ext]
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for raw in f:
                    line = raw.rstrip('\n')
                    stripped = line.lstrip()
                    if not stripped:
                        continue
                    if stripped.startswith(line_marker):
                        comment_ct += 1
                    else:
                        code_ct += 1
        except Exception:
            pass
        return code_ct, comment_ct

    elif ext in BLOCK_COMMENT_EXTENSIONS:
        block_start, block_end, line_marker = BLOCK_COMMENT_EXTENSIONS[ext]
        in_block = False
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for raw in f:
                    line = raw.rstrip('\n')
                    stripped = line.lstrip()
                    if not stripped:
                        continue
                    if in_block:
                        comment_ct += 1
                        if block_end in stripped:
                            in_block = False
                        continue
                    if stripped.startswith(line_marker):
                        comment_ct += 1
                    elif block_start in stripped:
                        idx = stripped.find(block_start)
                        before = stripped[:idx].strip()
                        comment_ct += 1
                        if block_end not in stripped:
                            in_block = True
                    else:
                        code_ct += 1
        except Exception:
            pass
        return code_ct, comment_ct

    else:
        # Unrecognized extension → skip
        return 0, 0

def count_sloc_cloc_python(repo_path: str):
    """
    Walk repo_path recursively. For each file with recognized extension,
    call analyze_source_file. Return (total_code, total_comment).
    """
    total_code = 0
    total_comment = 0
    valid_exts = set(LINE_COMMENT_EXTENSIONS.keys()) | set(BLOCK_COMMENT_EXTENSIONS.keys())

    for root, _, files in os.walk(repo_path):
        if '.git' in root.split(os.sep):
            continue
        for fname in files:
            _, ext = os.path.splitext(fname.lower())
            if ext not in valid_exts:
                continue
            full = os.path.join(root, fname)
            code_ct, comment_ct = analyze_source_file(full)
            total_code += code_ct
            total_comment += comment_ct

    return total_code, total_comment

TEST_FRAMEWORK_PATTERNS = [
    r"(?i)\bimport\s+unittest\b",
    r"(?i)\bfrom\s+django\.test\b",
    r"(?i)\bimport\s+pytest\b",
    r"(?i)\bimport\s+nose\b",
    r"(?i)\bimport\s+RSpec\b",
    r"(?i)\bimport\s+Minitest\b",
    r"(?i)\bimport\s+JUnit\b",
    r"(?i)\bimport\s+TestNG\b",
    r"(?i)\bimport\s+googletest\b",
    r"(?i)\b#import\s+<gtest/gtest\.h>",
    r"(?i)\bimport\s+mocha\b",
    r"(?i)\bimport\s+jest\b",
    r"(?i)\bimport\s+kotlin\.test\b",
    r"(?i)\bimport\s+xunit\b",
    r"(?i)\b@test\b",
]

def is_test_file_by_content(path: str) -> bool:
    """
    Returns True if file at path contains any TEST_FRAMEWORK_PATTERNS.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return False
    for pat in TEST_FRAMEWORK_PATTERNS:
        if re.search(pat, content):
            return True
    return False

def is_test_file_by_name_or_dir(repo_path: str, rel_path: str, fname: str) -> bool:
    """
    Returns True if rel_path or fname implies a test file by naming convention.
    """
    lower_rel = rel_path.replace(os.sep, "/").lower()
    lower_fname = fname.lower()
    if re.search(r'(^|/)tests?(/|$)', lower_rel):
        return True
    if lower_fname.startswith("test_"):
        return True
    if re.search(r'test\.(py|java|js|c|cpp|h|cs|go|sh|rb)$', lower_fname):
        return True
    return False

def traverse_and_count_tests_exact_python(repo_path: str):
    """
    Identify test files exactly as in Sect 4.7, count their code lines (SLOC).
    Return sloc_tests (int).
    """
    exts = set(LINE_COMMENT_EXTENSIONS.keys()) | set(BLOCK_COMMENT_EXTENSIONS.keys())
    test_files = []
    for root, _, files in os.walk(repo_path):
        if ".git" in root.split(os.sep):
            continue
        for fname in files:
            _, ext = os.path.splitext(fname.lower())
            if ext not in exts:
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, repo_path)
            if is_test_file_by_content(full):
                test_files.append(full)
                continue
            if is_test_file_by_name_or_dir(repo_path, rel, fname):
                test_files.append(full)
    sloc_tests = 0
    for tf in test_files:
        code_ct, _ = analyze_source_file(tf)
        sloc_tests += code_ct
    return sloc_tests

def fetch_all_commits(owner: str, repo: str):
    """
    Return a list of {"date": datetime, "author": "Name <email>"} for each commit,
    using GET /repos/{owner}/{repo}/commits with pagination.
    """
    commits_data = []
    page = 1
    per_page = 100
    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params = {"per_page": per_page, "page": page}
        resp = github_get(url, params=params)
        if resp.status_code != 200:
            break
        arr = resp.json()
        if not isinstance(arr, list) or len(arr) == 0:
            break
        for item in arr:
            date_str = item["commit"]["author"]["date"]
            dt = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            name = item["commit"]["author"].get("name", "").strip()
            email = item["commit"]["author"].get("email", "").strip()
            author = f"{name} <{email}>"
            commits_data.append({"date": dt, "author": author})
        page += 1
    return commits_data

def compute_git_history_metrics_via_api(owner: str, repo: str):
    """
    Using GitHub commits API, compute:
      - total_commits (int)
      - first_commit_date (datetime)
      - last_commit_date  (datetime)
      - author_counts     (dict: "Name <email>" -> commit_count)
    """
    commits = fetch_all_commits(owner, repo)
    total_commits = len(commits)
    if total_commits == 0:
        first_date = last_date = datetime.datetime.utcnow()
        author_counts = {}
    else:
        dates = [c["date"] for c in commits]
        first_date = min(dates)
        last_date = max(dates)
        author_counts = {}
        for c in commits:
            author_counts[c["author"]] = author_counts.get(c["author"], 0) + 1
    return total_commits, first_date, last_date, author_counts

def compute_core_contributors(author_counts: dict):
    """
    Smallest set of authors accounting for ≥ 80% of all commits.
    """
    total = sum(author_counts.values())
    if total == 0:
        return 0
    threshold = 0.8 * total
    cum = 0
    core = 0
    for cnt in sorted(author_counts.values(), reverse=True):
        cum += cnt
        core += 1
        if cum >= threshold:
            break
    return core

def compute_duration_months(start_dt: datetime.datetime, end_dt: datetime.datetime):
    """
    Compute months (float) between start_dt and end_dt. Min value is 1/30.
    """
    delta = relativedelta(end_dt, start_dt)
    months = delta.years * 12 + delta.months + delta.days / 30.0
    return max(months, 1/30)

def get_default_branch(owner: str, repo: str):
    """
    Get default branch name via GET /repos/{owner}/{repo}.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    resp = github_get(url)
    data = resp.json()
    return data.get("default_branch", "main")

def download_and_unpack_tarball(owner: str, repo: str, branch: str = "main"):
    """
    Download GitHub tarball for given branch, unpack into temp dir.
    Return (tmp_root, top_folder_path).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/tarball/{branch}"
    while True:
        try:
            resp = requests.get(url, headers=get_github_headers(), stream=True, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"Network error downloading tarball ({e}). Sleeping for {NETWORK_SLEEP}s…")
            time.sleep(NETWORK_SLEEP)
            continue
        if resp.status_code == 200:
            break
        if resp.status_code in (403, 429):
            handle_github_rate_limit(resp)
            continue
        raise RuntimeError(f"Failed to download tarball: {resp.status_code} {resp.text}")

    tmp_tar = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    for chunk in resp.iter_content(chunk_size=1 << 20):
        tmp_tar.write(chunk)
    tmp_tar.flush()
    tmp_tar.close()

    tmp_dir = tempfile.mkdtemp(prefix="gh_metrics_")
    with tarfile.open(tmp_tar.name, "r:gz") as tf:
        tf.extractall(path=tmp_dir)

    entries = os.listdir(tmp_dir)
    if len(entries) != 1:
        raise RuntimeError(f"Expected exactly one top-level directory in {tmp_dir}, got {entries!r}")
    top_folder = os.path.join(tmp_dir, entries[0])

    os.unlink(tmp_tar.name)
    return tmp_dir, top_folder

def fetch_issue_metrics(owner: str, repo: str):
    """
    Count:
      - total_issues   (open + closed, excluding PRs)
      - closed_issues  (subset)
      - total_comments (on issues)
    Return (total_issues, closed_issues, total_comments).
    """
    total_issues = closed_issues = total_comments = 0
    page = 1
    per_page = 100
    base = f"https://api.github.com/repos/{owner}/{repo}"
    # Issues list
    while True:
        url = f"{base}/issues"
        params = {"state": "all", "per_page": per_page, "page": page}
        try:
            resp = requests.get(url, headers=get_github_headers(), params=params, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching issues ({e}). Sleeping {NETWORK_SLEEP}s…")
            time.sleep(NETWORK_SLEEP)
            continue
        if resp.status_code == 200:
            arr = resp.json()
            if not arr or (isinstance(arr, dict) and arr.get("message")):
                break
            for issue in arr:
                if "pull_request" in issue:
                    continue
                total_issues += 1
                if issue.get("state") == "closed":
                    closed_issues += 1
            page += 1
            continue
        if resp.status_code in (403, 429):
            handle_github_rate_limit(resp)
            continue
        break

    # Issue comments
    page = 1
    while True:
        url = f"{base}/issues/comments"
        params = {"per_page": per_page, "page": page}
        try:
            resp = requests.get(url, headers=get_github_headers(), params=params, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching issue comments ({e}). Sleeping {NETWORK_SLEEP}s…")
            time.sleep(NETWORK_SLEEP)
            continue
        if resp.status_code == 200:
            arr = resp.json()
            if not arr or (isinstance(arr, dict) and arr.get("message")):
                break
            total_comments += len(arr)
            page += 1
            continue
        if resp.status_code in (403, 429):
            handle_github_rate_limit(resp)
            continue
        break

    return total_issues, closed_issues, total_comments

def detect_ci_flag(owner: str, repo: str) -> int:
    """
    Check if any known CI config paths exist:
      - .travis.yml
      - appveyor.yml
      - .github/workflows
      - circleci/config.yml
      - azure-pipelines.yml
      - .gitlab-ci.yml
    Return 1 if found, else 0.
    """
    patterns = [
        ".travis.yml",
        "appveyor.yml",
        ".github/workflows",
        "circleci/config.yml",
        "azure-pipelines.yml",
        ".gitlab-ci.yml"
    ]
    for p in patterns:
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{p}"
        resp = github_get(url)
        if resp.status_code == 200:
            return 1
    return 0

def detect_license_flag(owner: str, repo: str, local_clone_path: str = None) -> int:
    """
    1) Try GET /repos/{owner}/{repo}/license.
    2) If 404, scan local clone for common license file or keywords in first 30 lines.
    Return 1 if found, else 0.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/license"
    resp = github_get(url)
    if resp.status_code == 200:
        return 1

    if local_clone_path:
        LICENSE_KEYWORDS = [
            "mit license",
            "apache license",
            "gnu general public license",
            "gpl",
            "bsd license",
            "mozilla public license",
            "eclipse public license",
            "creative commons license",
        ]
        for root, _, files in os.walk(local_clone_path):
            if ".git" in root.split(os.sep):
                continue
            for fname in files:
                lower = fname.lower()
                if lower in ("license", "license.txt", "license.md", "licence", "notice"):
                    return 1
                path = os.path.join(root, fname)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        for _ in range(30):
                            line = f.readline()
                            if not line:
                                break
                            ll = line.lower()
                            for kw in LICENSE_KEYWORDS:
                                if kw in ll:
                                    return 1
                except Exception:
                    continue
    return 0

ORG_THRESHOLDS = {
    "community":     2.0000,      # core contributors
    "ci":            1.0000,      # CI flag
    "documentation": 0.0018660,   # comment_ratio
    "history":       2.0895,      # commits_per_month
    "issues":        0.022989,    # issue_events_per_month
    "license":       1.0000,      # license flag
    "unittest":      0.0010160    # test_ratio
}

def label_continuous(value: float, threshold: float) -> str:
    """
    Given a continuous metric and its threshold:
      - "low" if value < 0.5 * threshold
      - "medium" if 0.5*threshold ≤ value < threshold
      - "high" if value ≥ threshold
    """
    half = 0.5 * threshold
    if value < half:
        return "low"
    elif value < threshold:
        return "medium"
    else:
        return "high"

def label_binary(flag: int) -> str:
    """
    For CI and license flags (0 or 1):
      0 → "low",  1 → "high"
    """
    return "high" if flag >= 1 else "low"

# ─────────────────────────────────────────────────────────────────────────────
# 8) SECTION: CODE-GENERATION DETECTION (from code-gen-detection.py)
# ─────────────────────────────────────────────────────────────────────────────

# 8.1. Define script file extensions
SCRIPT_EXTENSIONS = {
    ".py", ".js", ".java", ".cpp", ".c", ".h", ".cs",
    ".ts", ".rb", ".go", ".php", ".rs", ".sh",
    ".m",      # MATLAB / Octave
    ".r",      # R scripts
    ".rmd",    # R Markdown
    ".jl",     # Julia
    ".scala",  # Scala
    ".swift",  # Swift
    ".kt",     # Kotlin
    ".ps1"     # PowerShell
}

OPENAI_MODEL = "gpt-3.5-turbo"
NUM_RECENT_FILES = 10

def is_script_file(path: str) -> bool:
    """
    Return True if path ends with one of SCRIPT_EXTENSIONS.
    """
    lower = path.lower()
    for ext in SCRIPT_EXTENSIONS:
        if lower.endswith(ext):
            return True
    return False

def handle_rate_limit_codegen(response: requests.Response):
    """
    If GitHub returns a rate-limit for code-gen (status 403 or 429 with no remaining),
    parse reset header and sleep until reset + buffer. Otherwise, sleep default.
    """
    if response.status_code in (403, 429):
        rem = response.headers.get("X-RateLimit-Remaining")
        if rem == "0":
            reset_ts = response.headers.get("X-RateLimit-Reset")
            if reset_ts:
                try:
                    reset_ts = int(reset_ts)
                    now = int(time.time())
                    wait = reset_ts - now + 5
                    if wait > 0:
                        print(f"[CodeGen] GitHub rate limit hit. Sleeping {wait}s…")
                        time.sleep(wait)
                        return
                except ValueError:
                    pass
        print(f"[CodeGen] GitHub returned {response.status_code}. Sleeping {GITHUB_DEFAULT_SLEEP}s…")
        time.sleep(GITHUB_DEFAULT_SLEEP)

def github_get_codegen(url: str, params: dict = None) -> requests.Response:
    """
    Wrapper around requests.get for code-gen. Retries on rate-limit or network errors.
    """
    while True:
        try:
            resp = requests.get(url, headers=get_github_headers(), params=params or {}, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"[CodeGen] Network error hitting GitHub ({e}). Sleeping {NETWORK_SLEEP}s…")
            time.sleep(NETWORK_SLEEP)
            continue
        if resp.status_code == 200:
            return resp
        handle_rate_limit_codegen(resp)

def get_recent_script_files(owner: str, repo: str, branch: str, n: int) -> list:
    """
    Walk commits on default branch (paginated, descending date). For each commit,
    fetch commit details (files changed), collect script file paths until n unique.
    """
    selected = []
    seen = set()
    per_page = 30
    page = 1

    while len(selected) < n:
        commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params = {"sha": branch, "per_page": per_page, "page": page}
        resp = github_get_codegen(commits_url, params=params)
        commits = resp.json()
        if not commits:
            break
        for commit_summary in commits:
            sha = commit_summary["sha"]
            commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
            commit_resp = github_get_codegen(commit_url)
            commit_data = commit_resp.json()
            for f in commit_data.get("files", []):
                path = f.get("filename")
                if path and is_script_file(path) and path not in seen:
                    seen.add(path)
                    selected.append(path)
                    if len(selected) >= n:
                        break
            if len(selected) >= n:
                break
        if len(selected) >= n:
            break
        page += 1

    return selected[:n]

def raw_file_exists(owner: str, repo: str, branch: str, path: str) -> bool:
    """
    Check if raw file URL returns 200 via HEAD.
    """
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    resp = github_head(raw_url)
    return resp.status_code == 200

def fetch_raw_file(owner: str, repo: str, branch: str, path: str) -> str:
    """
    Fetch raw file contents from raw.githubusercontent.com. Raises if not 200.
    """
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    while True:
        try:
            resp = requests.get(raw_url, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"[CodeGen] Network error fetching raw file ({e}). Sleeping {NETWORK_SLEEP}s…")
            time.sleep(NETWORK_SLEEP)
            continue
        if resp.status_code == 200:
            return resp.text
        if resp.status_code in (403, 429):
            handle_rate_limit_codegen(resp)
            continue
        raise RuntimeError(f"Could not fetch raw file {path}: {resp.status_code}")

def classify_code_with_gpt(code_snippet: str) -> str:
    """
    Send code_snippet to OpenAI GPT to label "AI" or "Human". Retries on rate-limit.
    """
    if len(code_snippet) > 4000:
        truncated = (
            code_snippet[:2000]
            + "\n\n# … [TRUNCATED] …\n\n"
            + code_snippet[-2000:]
        )
        prompt_header = (
            "You are a classification assistant. Given the following code (TRUNCATED in the middle), "
            "decide if it was generated by an AI (e.g. ChatGPT/Codex) or written by a human. "
            "Respond with exactly one word: AI or Human.\n\n"
        )
        code_to_classify = truncated
    else:
        prompt_header = (
            "You are a classification assistant. Given the following code, "
            "decide if it was generated by an AI (e.g. ChatGPT/Codex) or written by a human. "
            "Respond with exactly one word: AI or Human.\n\n"
        )
        code_to_classify = code_snippet

    full_prompt = prompt_header + "```" + code_to_classify + "```"
    messages = [
        {"role": "system", "content": "You are an expert at detecting machine-generated code."},
        {"role": "user", "content": full_prompt}
    ]

    # Short sleep to avoid bursts
    time.sleep(1)

    backoff = OPENAI_BACKOFF_BASE
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            resp = call_openai_chat_completion(messages, model=OPENAI_MODEL, max_tokens=8, temperature=0.0)
            text = resp.choices[0].message.content.strip()
            lower = text.lower()
            if lower.startswith("ai"):
                return "AI"
            elif lower.startswith("human"):
                return "Human"
            else:
                if "human" in lower:
                    return "Human"
                elif "ai" in lower or "machine" in lower:
                    return "AI"
                else:
                    return "Human"
        except RuntimeError as e:
            if "rate limit" in str(e).lower() and attempt < OPENAI_MAX_RETRIES - 1:
                print(f"[CodeGen] OpenAI rate limit (attempt {attempt+1}). Sleeping {backoff}s…")
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                print(f"[CodeGen] OpenAI error or final attempt: {e}. Defaulting to 'Human'.")
                return "Human"
    return "Human"

# ─────────────────────────────────────────────────────────────────────────────
# 9) SECTION: MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def load_progress() -> set:
    """
    Load PROGRESS_FILE (list of filenames). Return set of processed filenames.
    """
    if os.path.isfile(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as pf:
                lst = json.load(pf)
                if isinstance(lst, list):
                    return set(lst)
        except Exception:
            pass
    return set()

def save_progress(done_set: set):
    """
    Save sorted list(done_set) to PROGRESS_FILE.
    """
    tmp = sorted(done_set)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as pf:
        json.dump(tmp, pf, indent=2)

def main():
    processed = load_progress()

    for root, dirs, files in os.walk(INPUT_DIR):
        for fname in files:
            if not fname.lower().endswith(".json"):
                continue
            if fname in processed:
                print(f"Skipping '{fname}' (already processed).")
                continue

            json_path = os.path.join(root, fname)
            while True:
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        record = json.load(f)
                except Exception as e:
                    print(f"Skipping '{json_path}' (could not load JSON): {e}", file=sys.stderr)
                    break  # skip permanently

                # ───────────
                # 1) Extract GitHub URL from Zenodo JSON
                # ───────────
                git_url = extract_git_url_from_zenodo(record)
                has_git_url = isinstance(git_url, str) and git_url.startswith("https://github.com/")

                # ───────────
                # 2) Run FAIRness Assessment
                # ───────────
                fair_subp = assess_fairness_metadata(record)
                fair_est = estimate_fairness_from_subprinciples(fair_subp)

                if has_git_url:
                    try:
                        hf = run_howfairis_checks(git_url)
                    except Exception as e:
                        if is_rate_limit_exception(e):
                            sleep_until_rate_limit_resets()
                            continue
                        elif is_network_exception(e):
                            wait_for_network()
                            continue
                        else:
                            hf = {
                                "howfairis_score": 0,
                                "howfairis_details": {
                                    "repository": "no",
                                    "license": "no",
                                    "registry": "no",
                                    "citation": "no",
                                    "quality": "no"
                                }
                            }
                else:
                    hf = {
                        "howfairis_score": 0,
                        "howfairis_details": {
                            "repository": "no",
                            "license": "no",
                            "registry": "no",
                            "citation": "no",
                            "quality": "no"
                        }
                    }

                fairness_result = {
                    "subprinciple_assessment": fair_subp,
                    "principle_categories":    fair_est["principle_categories"],
                    "overall_fairness":        fair_est["overall_fairness"],
                    "git_url":                 git_url or None,
                    "howfairis_score":         hf["howfairis_score"],
                    "howfairis_details":       hf["howfairis_details"]
                }

                # ───────────
                # 3) Run SE Practices Analysis
                # ───────────
                if has_git_url:
                    try:
                        owner, repo_name = parse_owner_repo(git_url)
                    except ValueError:
                        has_git_url = False

                if has_git_url:
                    try:
                        default_branch = get_default_branch(owner, repo_name)
                    except Exception as e:
                        if is_rate_limit_exception(e):
                            sleep_until_rate_limit_resets()
                            continue
                        elif is_network_exception(e):
                            wait_for_network()
                            continue
                        else:
                            default_branch = None

                    if default_branch:
                        try:
                            tmp_root, clone_dir = download_and_unpack_tarball(owner, repo_name, default_branch)
                        except Exception as e:
                            if is_rate_limit_exception(e):
                                sleep_until_rate_limit_resets()
                                continue
                            elif is_network_exception(e):
                                wait_for_network()
                                continue
                            else:
                                print(f"  [SE] Could not download tarball for {owner}/{repo_name}: {e}", file=sys.stderr)
                                se_result = {
                                    "dimensions": {
                                        "community":   {"actual": 0,   "estimation": "low"},
                                        "ci":          {"actual": 0,   "estimation": "low"},
                                        "documentation":{"actual": 0.0, "estimation": "low"},
                                        "history":     {"actual": 0.0, "estimation": "low"},
                                        "issues":      {"actual": 0.0, "estimation": "low"},
                                        "license":     {"actual": 0,   "estimation": "low"},
                                        "unittest":    {"actual": 0.0, "estimation": "low"}
                                    },
                                    "raw": {
                                        "total_sloc":        0,
                                        "total_cloc":        0,
                                        "test_sloc":         0,
                                        "total_commits":     0,
                                        "first_commit_date": None,
                                        "last_commit_date":  None,
                                        "duration_months":   0.0,
                                        "issue_events":      0
                                    }
                                }
                                clone_dir = None
                        else:
                            try:
                                total_sloc, total_cloc = count_sloc_cloc_python(clone_dir)
                                comment_ratio = total_cloc / (total_sloc + total_cloc) if (total_sloc + total_cloc) > 0 else 0.0

                                sloc_tests = traverse_and_count_tests_exact_python(clone_dir)

                                total_commits, first_date, last_date, author_counts = \
                                    compute_git_history_metrics_via_api(owner, repo_name)
                                core_contribs = compute_core_contributors(author_counts)
                                duration_months = compute_duration_months(first_date, last_date)
                                commit_frequency = total_commits / duration_months

                                total_issues, closed_issues, total_issue_comments = \
                                    fetch_issue_metrics(owner, repo_name)
                                total_issue_events = total_issues + closed_issues + total_issue_comments
                                issue_frequency = total_issue_events / duration_months

                                ci_flag = detect_ci_flag(owner, repo_name)
                                license_flag = detect_license_flag(owner, repo_name, local_clone_path=clone_dir)

                                test_ratio = sloc_tests / total_sloc if total_sloc > 0 else 0.0

                                labels = {}
                                c_val = float(core_contribs)
                                c_th = ORG_THRESHOLDS["community"]
                                labels["community"] = {
                                    "value": c_val,
                                    "estimation": label_continuous(c_val, c_th)
                                }

                                ci_val = float(ci_flag)
                                labels["ci"] = {
                                    "value": ci_val,
                                    "estimation": label_binary(ci_flag)
                                }

                                d_val = comment_ratio
                                d_th = ORG_THRESHOLDS["documentation"]
                                labels["documentation"] = {
                                    "value": d_val,
                                    "estimation": label_continuous(d_val, d_th)
                                }

                                h_val = commit_frequency
                                h_th = ORG_THRESHOLDS["history"]
                                labels["history"] = {
                                    "value": h_val,
                                    "estimation": label_continuous(h_val, h_th)
                                }

                                i_val = issue_frequency
                                i_th = ORG_THRESHOLDS["issues"]
                                labels["issues"] = {
                                    "value": i_val,
                                    "estimation": label_continuous(i_val, i_th)
                                }

                                l_val = float(license_flag)
                                labels["license"] = {
                                    "value": l_val,
                                    "estimation": label_binary(license_flag)
                                }

                                u_val = test_ratio
                                u_th = ORG_THRESHOLDS["unittest"]
                                labels["unittest"] = {
                                    "value": u_val,
                                    "estimation": label_continuous(u_val, u_th)
                                }

                                se_result = {
                                    "dimensions": {
                                        "community": {
                                            "actual": labels["community"]["value"],
                                            "estimation": labels["community"]["estimation"]
                                        },
                                        "ci": {
                                            "actual": labels["ci"]["value"],
                                            "estimation": labels["ci"]["estimation"]
                                        },
                                        "documentation": {
                                            "actual": labels["documentation"]["value"],
                                            "estimation": labels["documentation"]["estimation"]
                                        },
                                        "history": {
                                            "actual": labels["history"]["value"],
                                            "estimation": labels["history"]["estimation"]
                                        },
                                        "issues": {
                                            "actual": labels["issues"]["value"],
                                            "estimation": labels["issues"]["estimation"]
                                        },
                                        "license": {
                                            "actual": labels["license"]["value"],
                                            "estimation": labels["license"]["estimation"]
                                        },
                                        "unittest": {
                                            "actual": labels["unittest"]["value"],
                                            "estimation": labels["unittest"]["estimation"]
                                        }
                                    },
                                    "raw": {
                                        "total_sloc":        total_sloc,
                                        "total_cloc":        total_cloc,
                                        "test_sloc":         sloc_tests,
                                        "total_commits":     total_commits,
                                        "first_commit_date": first_date.isoformat(),
                                        "last_commit_date":  last_date.isoformat(),
                                        "duration_months":   duration_months,
                                        "issue_events":      total_issue_events
                                    }
                                }

                            except Exception as e:
                                if is_rate_limit_exception(e):
                                    try:
                                        shutil.rmtree(tmp_root, onerror=on_rm_error)
                                    except Exception:
                                        pass
                                    sleep_until_rate_limit_resets()
                                    continue
                                elif is_network_exception(e):
                                    try:
                                        shutil.rmtree(tmp_root, onerror=on_rm_error)
                                    except Exception:
                                        pass
                                    wait_for_network()
                                    continue
                                else:
                                    print(f"  [SE] Processing error for {owner}/{repo_name}: {e}", file=sys.stderr)
                                    se_result = {
                                        "dimensions": {
                                            "community":   {"actual": 0,   "estimation": "low"},
                                            "ci":          {"actual": 0,   "estimation": "low"},
                                            "documentation":{"actual": 0.0, "estimation": "low"},
                                            "history":     {"actual": 0.0, "estimation": "low"},
                                            "issues":      {"actual": 0.0, "estimation": "low"},
                                            "license":     {"actual": 0,   "estimation": "low"},
                                            "unittest":    {"actual": 0.0, "estimation": "low"}
                                        },
                                        "raw": {
                                            "total_sloc":        0,
                                            "total_cloc":        0,
                                            "test_sloc":         0,
                                            "total_commits":     0,
                                            "first_commit_date": None,
                                            "last_commit_date":  None,
                                            "duration_months":   0.0,
                                            "issue_events":      0
                                        }
                                    }
                            finally:
                                try:
                                    shutil.rmtree(tmp_root, onerror=on_rm_error)
                                except Exception:
                                    pass
                    else:
                        se_result = {
                            "dimensions": {
                                "community":   {"actual": 0,   "estimation": "low"},
                                "ci":          {"actual": 0,   "estimation": "low"},
                                "documentation":{"actual": 0.0, "estimation": "low"},
                                "history":     {"actual": 0.0, "estimation": "low"},
                                "issues":      {"actual": 0.0, "estimation": "low"},
                                "license":     {"actual": 0,   "estimation": "low"},
                                "unittest":    {"actual": 0.0, "estimation": "low"}
                            },
                            "raw": {
                                "total_sloc":        0,
                                "total_cloc":        0,
                                "test_sloc":         0,
                                "total_commits":     0,
                                "first_commit_date": None,
                                "last_commit_date":  None,
                                "duration_months":   0.0,
                                "issue_events":      0
                            }
                        }
                else:
                    se_result = {
                        "dimensions": {
                            "community":   {"actual": 0,   "estimation": "low"},
                            "ci":          {"actual": 0,   "estimation": "low"},
                            "documentation":{"actual": 0.0, "estimation": "low"},
                            "history":     {"actual": 0.0, "estimation": "low"},
                            "issues":      {"actual": 0.0, "estimation": "low"},
                            "license":     {"actual": 0,   "estimation": "low"},
                            "unittest":    {"actual": 0.0, "estimation": "low"}
                        },
                        "raw": {
                            "total_sloc":        0,
                            "total_cloc":        0,
                            "test_sloc":         0,
                            "total_commits":     0,
                            "first_commit_date": None,
                            "last_commit_date":  None,
                            "duration_months":   0.0,
                            "issue_events":      0
                        }
                    }

                # ───────────
                # 4) Run Code-Generation Detection
                # ───────────
                if has_git_url:
                    try:
                        cg_owner, cg_repo = parse_owner_repo(git_url)
                    except ValueError:
                        has_git_url = False

                if has_git_url:
                    try:
                        default_branch_cg = get_default_branch(cg_owner, cg_repo)
                    except Exception as e:
                        if is_rate_limit_exception(e):
                            sleep_until_rate_limit_resets()
                            continue
                        elif is_network_exception(e):
                            wait_for_network()
                            continue
                        else:
                            default_branch_cg = None

                    if default_branch_cg:
                        try:
                            tree_resp = github_get_codegen(
                                f"https://api.github.com/repos/{cg_owner}/{cg_repo}/git/trees/{default_branch_cg}",
                                params={"recursive": "1"}
                            )
                            tree = tree_resp.json().get("tree", [])
                        except Exception as e:
                            if is_rate_limit_exception(e):
                                sleep_until_rate_limit_resets()
                                continue
                            elif is_network_exception(e):
                                wait_for_network()
                                continue
                            else:
                                print(f"  [CodeGen] Error listing files for {cg_owner}/{cg_repo}: {e}", file=sys.stderr)
                                codegen_result = {
                                    "#AllFiles":              0,
                                    "#Selected Recent Files": 0,
                                    "#Human":                 0,
                                    "#AI":                    0,
                                    "ai_ratio":               0.0,
                                    "estimation":             "Low",
                                    "analyzed-scripts":       {}
                                }
                                tree = []
                        else:
                            all_files = [entry["path"] for entry in tree if entry.get("type") == "blob"]
                            script_files = [p for p in all_files if is_script_file(p)]
                            total_script_count = len(script_files)

                            target_candidates = NUM_RECENT_FILES * 3
                            try:
                                recent_candidates = get_recent_script_files(
                                    cg_owner, cg_repo, default_branch_cg, target_candidates
                                )
                            except Exception as e:
                                if is_rate_limit_exception(e):
                                    sleep_until_rate_limit_resets()
                                    continue
                                elif is_network_exception(e):
                                    wait_for_network()
                                    continue
                                else:
                                    print(f"  [CodeGen] Error fetching recent scripts for {cg_owner}/{cg_repo}: {e}", file=sys.stderr)
                                    recent_candidates = []

                            valid_recent = []
                            for pth in recent_candidates:
                                if len(valid_recent) >= NUM_RECENT_FILES:
                                    break
                                try:
                                    exists = raw_file_exists(cg_owner, cg_repo, default_branch_cg, pth)
                                except Exception:
                                    exists = False
                                if exists:
                                    valid_recent.append(pth)

                            human_count = 0
                            ai_count = 0
                            analyzed_scripts = {}

                            for script_path in valid_recent:
                                print(f"[CodeGen] Fetching file: {script_path} …")
                                try:
                                    code_text = fetch_raw_file(
                                        cg_owner, cg_repo, default_branch_cg, script_path
                                    )
                                except Exception as fe:
                                    if is_rate_limit_exception(fe):
                                        sleep_until_rate_limit_resets()
                                        human_count = 0
                                        ai_count = 0
                                        analyzed_scripts = {}
                                        continue
                                    elif is_network_exception(fe):
                                        wait_for_network()
                                        human_count = 0
                                        ai_count = 0
                                        analyzed_scripts = {}
                                        continue
                                    else:
                                        print(f"    [CodeGen] Could not fetch raw file {script_path}: {fe}", file=sys.stderr)
                                        continue

                                print(f"[CodeGen] Classifying {script_path} …")
                                label = classify_code_with_gpt(code_text)
                                analyzed_scripts[script_path] = label
                                if label == "Human":
                                    human_count += 1
                                else:
                                    ai_count += 1

                            total_classified = human_count + ai_count
                            ai_ratio = (ai_count / total_classified) if total_classified > 0 else 0.0
                            if ai_ratio > 0.66:
                                estimation_cg = "High"
                            elif ai_ratio > 0.33:
                                estimation_cg = "Medium"
                            else:
                                estimation_cg = "Low"

                            codegen_result = {
                                "#AllFiles":              total_script_count,
                                "#Selected Recent Files": total_classified,
                                "#Human":                 human_count,
                                "#AI":                    ai_count,
                                "ai_ratio":               round(ai_ratio, 3),
                                "estimation":             estimation_cg,
                                "analyzed-scripts":       analyzed_scripts
                            }
                    else:
                        codegen_result = {
                            "#AllFiles":              0,
                            "#Selected Recent Files": 0,
                            "#Human":                 0,
                            "#AI":                    0,
                            "ai_ratio":               0.0,
                            "estimation":             "Low",
                            "analyzed-scripts":       {}
                        }
                else:
                    codegen_result = {
                        "#AllFiles":              0,
                        "#Selected Recent Files": 0,
                        "#Human":                 0,
                        "#AI":                    0,
                        "ai_ratio":               0.0,
                        "estimation":             "Low",
                        "analyzed-scripts":       {}
                    }

                # ───────────
                # 5) Run AI/ML/Ops Detection
                # ───────────
                if has_git_url:
                    try:
                        am_owner, am_repo = parse_owner_repo(git_url)
                    except ValueError:
                        has_git_url = False

                if has_git_url:
                    try:
                        meta_info, category_am, types_am = classify_repository(am_owner, am_repo)
                        am_result = {
                            "repository": f"{am_owner}/{am_repo}",
                            "category":   category_am,
                            "types":      types_am,
                            "name":       meta_info.get("name", ""),
                            "description":meta_info.get("description", ""),
                            "topics":     meta_info.get("topics", []),
                            "timestamp":  datetime.datetime.now(datetime.timezone.utc).isoformat()
                        }
                    except Exception as e:
                        if is_rate_limit_exception(e):
                            sleep_until_rate_limit_resets()
                            continue
                        elif is_network_exception(e):
                            wait_for_network()
                            continue
                        else:
                            print(f"  [AI/ML/Ops] Error classifying {am_owner}/{am_repo}: {e}", file=sys.stderr)
                            am_result = {
                                "repository": None,
                                "category":   "None",
                                "types":      [],
                                "name":       "",
                                "description":"",
                                "topics":     [],
                                "timestamp":  None
                            }
                else:
                    am_result = {
                        "repository": None,
                        "category":   "None",
                        "types":      [],
                        "name":       "",
                        "description":"",
                        "topics":     [],
                        "timestamp":  None
                    }

                # ───────────
                # 6) Combine all four results under "repo-profile" and write back
                # ───────────
                record["repo-profile"] = {
                    "fairness_estimation":    fairness_result,
                    "se_estimation":          se_result,
                    "genai_detection":        codegen_result,
                    "ai_ml_ops_detection":    am_result
                }

                try:
                    with open(json_path, "w", encoding="utf-8") as out_f:
                        json.dump(record, out_f, indent=2)
                    print(f"Updated '{json_path}' with combined repo-profile.")
                except Exception as e:
                    print(f"Failed to write back to '{json_path}': {e}", file=sys.stderr)
                    break  # Do not mark as processed if write fails

                processed.add(fname)
                save_progress(processed)
                break  # Exit the retry loop for this file

            # End of while True retry loop

    print("All files scanned. Exiting.")

if __name__ == "__main__":
    main()
