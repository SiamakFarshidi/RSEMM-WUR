#!/usr/bin/env python3
# detect_repo_gen_github.py

import os
import sys
import json
import time
import requests
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

# ──────────────────────────────────────────────────────────────────────────────
# 1) Configuration & thresholds
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "distilgpt2"
DEVICE     = "cpu"   # change to "cuda" if you have GPU + torch-cuda installed

# Estimation thresholds (AI_ratio > HIGH → “High”; etc.)
EST_HIGH_THRESHOLD   = 0.66
EST_MEDIUM_THRESHOLD = 0.33

# File extensions to consider
SOURCE_EXTS = (".py", ".js", ".java", ".cpp", ".c", ".rs", ".go")

# Up to how many files to analyze
MAX_FILES = 20

# GitHub API base
GITHUB_API = "https://api.github.com"

# Grab a token from environment (optional, but avoids low rate limit if you set one)
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    **({"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {})
}

# ──────────────────────────────────────────────────────────────────────────────
# 2) Load tokenizer & model
# ──────────────────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval().to(DEVICE)

max_length = (
    model.config.n_positions
    if hasattr(model.config, "n_positions")
    else model.config.n_ctx
)

def compute_logprob(code: str) -> float:
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    avg_nll     = outputs.loss.item()
    token_count = inputs["input_ids"].size(1)
    return -avg_nll * token_count

def generate_perturbation(code: str, mask_frac: float = 0.1) -> str:
    lines      = code.splitlines()
    drop_count = max(1, int(len(lines) * mask_frac))
    to_drop    = set(random.sample(range(len(lines)), drop_count))
    return "\n".join(line for i, line in enumerate(lines) if i not in to_drop)

def check_genAI_script_content(code_str: str, n_perturb: int = 5, mask_frac: float = 0.1) -> bool:
    """
    Given file content as a string, return True if it’s likely AI-generated.
    """
    baseline_lp = compute_logprob(code_str)
    perturbed_lps = []
    for _ in range(n_perturb):
        pert_code = generate_perturbation(code_str, mask_frac=mask_frac)
        perturbed_lps.append(compute_logprob(pert_code))

    avg_perturbed  = sum(perturbed_lps) / len(perturbed_lps)
    flatness_score = baseline_lp - avg_perturbed
    return flatness_score > 0

# ──────────────────────────────────────────────────────────────────────────────
# 3) GitHub-specific helper routines
# ──────────────────────────────────────────────────────────────────────────────

def parse_github_url(repo_url: str) -> tuple[str, str]:
    """
    Given a GitHub repo URL (with or without “.git” at end), return (owner, repo).
    Examples it accepts:
      - https://github.com/owner/reponame.git
      - git@github.com:owner/reponame.git
      - https://github.com/owner/reponame
    """
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]

    # Several URL patterns to handle
    if repo_url.startswith("git@github.com:"):
        _, path = repo_url.split(":", 1)
    elif repo_url.startswith("https://github.com/"):
        path = repo_url[len("https://github.com/"):]
    elif repo_url.startswith("http://github.com/"):
        path = repo_url[len("http://github.com/"):]
    else:
        raise ValueError("Unsupported GitHub URL format")

    # Now path is like “owner/reponame” (possibly with trailing slash)
    path = path.rstrip("/")
    parts = path.split("/")
    if len(parts) != 2:
        raise ValueError("URL must be of the form github.com/owner/repo")
    owner, repo = parts
    return owner, repo

def get_default_branch(owner: str, repo: str) -> str:
    """
    Returns the default branch name (e.g. “main” or “master”).
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch repo info: {resp.status_code} {resp.text}")
    data = resp.json()
    return data.get("default_branch", "main")

def list_all_files(owner: str, repo: str, branch: str) -> list[str]:
    """
    Calls the Git Trees API to get a recursive list of all file paths in <branch>.
    Returns a list of paths like ['src/main.py', 'README.md', ...].
    """
    # The “recursive=1” tree call returns every blob and tree entry.
    url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch tree: {resp.status_code} {resp.text}")
    data = resp.json()

    file_paths = []
    for entry in data.get("tree", []):
        if entry["type"] == "blob":  # only blobs (i.e. file contents, not folders)
            file_paths.append(entry["path"])
    return file_paths

def get_file_last_commit_timestamp(owner: str, repo: str, path: str, branch: str) -> int:
    """
    Use the “List commits” endpoint with `?path=<path>&per_page=1` to get the latest commit
    that touched that file. Return its UNIX timestamp.
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
    params = {
        "path": path,
        "sha":  branch,
        "per_page": 1
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        # If something goes wrong (e.g. path was deleted), return 0
        return 0

    commits = resp.json()
    if not commits:
        return 0
    # The commit object has “commit.committer.date” in ISO 8601
    date_str = commits[0]["commit"]["committer"]["date"]  # e.g. "2025-05-28T14:23:00Z"
    # Convert to UNIX epoch
    t_struct = time.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    return int(time.mktime(t_struct))

def fetch_raw_file_content(owner: str, repo: str, path: str, branch: str) -> str:
    """
    Fetch the raw text for a given file path (on the specified branch).
    Uses the “raw.githubusercontent.com” pattern to avoid an extra API call.
    """
    # e.g. https://raw.githubusercontent.com/owner/repo/<branch>/src/main.py
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    resp = requests.get(raw_url)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch raw file: {raw_url} → {resp.status_code}")
    return resp.text

# ──────────────────────────────────────────────────────────────────────────────
# 4) Main logic (GitHub API approach)
# ──────────────────────────────────────────────────────────────────────────────

def main():


    repo_url = r"https://github.com/SiamakFarshidi/curriculum.git"

    try:
        owner, repo = parse_github_url(repo_url)
    except ValueError as e:
        print(f"Error parsing URL: {e}")
        sys.exit(1)

    print(f"Owner: {owner}    Repo: {repo}")

    # 4.1) Get default branch
    default_branch = get_default_branch(owner, repo)
    print(f"Default branch is '{default_branch}'")

    # 4.2) List *all* files in that branch
    print("Listing every file in the tree…")
    all_paths = list_all_files(owner, repo, default_branch)

    # 4.3) Filter for our extensions, then get commit timestamps for each
    candidates = [p for p in all_paths if p.lower().endswith(SOURCE_EXTS)]
    if not candidates:
        print("No source files found. Exiting.")
        sys.exit(1)

    print(f"Found {len(candidates)} total source files. Checking commit dates…")
    file_dates = []
    for path in candidates:
        ts = get_file_last_commit_timestamp(owner, repo, path, default_branch)
        # If timestamp = 0, we’ll skip it later
        if ts > 0:
            file_dates.append((path, ts))

    if not file_dates:
        print("No valid file commit dates found. Exiting.")
        sys.exit(1)

    # 4.4) Sort descending by timestamp, pick top MAX_FILES
    file_dates.sort(key=lambda x: x[1], reverse=True)
    top_files = file_dates[:MAX_FILES]

    print(f"Analyzing the {len(top_files)} most-recent files…")
    ai_count    = 0
    human_count = 0

    for (rel_path, ts) in top_files:
        try:
            content = fetch_raw_file_content(owner, repo, rel_path, default_branch)
        except Exception as e:
            print(f"  • Skipping {rel_path} (cannot fetch raw: {e})")
            continue

        is_ai = check_genAI_script_content(content)
        ts_human = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))
        if is_ai:
            ai_count += 1
            print(f"  [AI]    {rel_path}    (last commit: {ts_human} UTC)")
        else:
            human_count += 1
            print(f"  [HUMAN] {rel_path}    (last commit: {ts_human} UTC)")

    total = ai_count + human_count
    ai_ratio = (ai_count / total) if total > 0 else 0.0

    if   ai_ratio > EST_HIGH_THRESHOLD:
        estimation = "High"
    elif ai_ratio > EST_MEDIUM_THRESHOLD:
        estimation = "Medium"
    else:
        estimation = "Low"

    result = {
        "#Files":     total,
        "#Human":     human_count,
        "#AI":        ai_count,
        "estimation": estimation
    }

    out_path = Path.cwd() / "repo_ai_detection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults written to {out_path.resolve()}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
