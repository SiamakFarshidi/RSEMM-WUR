#!/usr/bin/env python3
"""
detectgpt_repo_analysis.py

Given a GitHub repo URL (hard‐coded via a variable in main), this script:
  1) Lists all files in the default branch (via GitHub’s “get tree” endpoint),
  2) Counts how many of them are “script files” (based on extensions),
  3) Finds up to NUM_RECENT_FILES*3 most recently modified script files (by walking recent commits),
     then filters out any that 404 on raw.githubusercontent, so we end up with exactly NUM_RECENT_FILES valid files (if possible),
  4) For each valid file, fetches its raw contents and applies a DetectGPT‐style “log‐prob curvature” test:
       • Compute total log‐prob of the original snippet under code-davinci-002,
       • Generate k small perturbations of the snippet, compute each of their log‐probs,
       • Calculate curvature = (avg perturbed log‐prob) − (original log‐prob),
       • If curvature > THRESHOLD, label “AI”; else “Human”.
  5) Respects GitHub’s rate limits (sleeping/retrying when necessary),
  6) Implements a bounded retry‐with‐backoff loop for OpenAI rate limits,
  7) Emits a summary JSON (“repo_ai_detection.json”) with:
       • "#AllFiles":            total number of script files in the repo,
       • "#Selected Recent Files": number of files actually classified (human + AI),
       • "#Human":               count of files labeled "Human",
       • "#AI":                  count of files labeled "AI",
       • "ai_ratio":             the ratio of AI‐labeled files (rounded to 3 decimals),
       • "estimation":           "High" if ai_ratio > 0.66, "Medium" if 0.33 < ai_ratio ≤ 0.66, else "Low".
"""

import os
import re
import time
import json
import random
import requests
import openai
from pathlib import Path
from typing import List, Dict, Tuple

# ------------------------------------------------------------------
#  CONFIGURATION / CONSTANTS
# ------------------------------------------------------------------

# 1) Put your GitHub repo URL here. It must be of the form https://github.com/owner/repo
GITHUB_URL =  "https://github.com/SiamakFarshidi/curriculum"  # ← <– REPLACE with the actual URL

# 2) (Optional) If you set GITHUB_TOKEN as an environment variable, we’ll use it to raise
#    the rate limit from 60 → 5,000 requests/hour. If not set, unauthenticated calls will be used.
GITHUB_TOKEN ="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"

# 3) OpenAI API key must be set as an env var named “OPENAI_API_KEY”
#    (e.g. export OPENAI_API_KEY="sk‐…")
openai.api_key = "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"

if not openai.api_key:
    raise RuntimeError("Please set OPENAI_API_KEY as an environment variable.")

# 4) How many recent script files to classify
NUM_RECENT_FILES = 5

# 5) What file extensions we consider “script/code files.”
SCRIPT_EXTENSIONS = {
    ".py", ".js", ".java", ".cpp", ".c", ".h", ".cs",
    ".ts", ".rb", ".go", ".php", ".rs", ".sh",
    ".m",    # MATLAB / Octave
    ".r",    # R scripts
    ".rmd",  # R Markdown
    ".jl",   # Julia
    ".scala",# Scala
    ".swift",# Swift
    ".kt",   # Kotlin
    ".ps1"   # PowerShell
}

# 6) Which OpenAI model to use for log‐prob‐based classification.
#    code-davinci-002 supports returning token logprobs and is currently available.
OPENAI_MODEL = "code-davinci-002"

# 7) DetectGPT parameters: number of perturbations (k) and decision threshold (τ)
DETECTGPT_K = 3
DETECTGPT_THRESHOLD = 0.0

# 8) How many seconds to wait before retrying if GitHub rate limit is hit.
RATE_LIMIT_SLEEP = 5  # will dynamically adjust if we read reset‐timestamp from headers

# 9) Maximum retries for OpenAI RateLimit + base backoff seconds
OPENAI_MAX_RETRIES = 5
OPENAI_BACKOFF_BASE = 10  # first retry waits 10s, next waits 20s, etc.

# ------------------------------------------------------------------
#  UTILITIES FOR GITHUB API
# ------------------------------------------------------------------

def github_api_headers() -> Dict[str, str]:
    """
    Return appropriate headers for GitHub API. If GITHUB_TOKEN is set, use it for Authorization.
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "detectgpt-repo-analysis"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers

def handle_rate_limit(response: requests.Response) -> None:
    """
    If response indicates a GitHub rate limit (403 or 429 with X-RateLimit-Remaining=0),
    parse X-RateLimit-Reset and sleep until then (plus a buffer). Otherwise, sleep briefly.
    """
    if response.status_code in (403, 429):
        remaining = response.headers.get("X-RateLimit-Remaining")
        if remaining == "0":
            reset_ts = response.headers.get("X-RateLimit-Reset")
            if reset_ts:
                reset_ts = int(reset_ts)
                now_ts = int(time.time())
                wait_sec = reset_ts - now_ts + 5
                if wait_sec > 0:
                    print(f"GitHub rate limit hit. Sleeping for {wait_sec} seconds…")
                    time.sleep(wait_sec)
                    return
        # Fallback brief sleep
        print(f"GitHub returned {response.status_code}. Sleeping for {RATE_LIMIT_SLEEP}s…")
        time.sleep(RATE_LIMIT_SLEEP)

def github_get(url: str, params: dict = None) -> requests.Response:
    """
    Wrapper around requests.get that handles rate limits automatically:
      • If we hit rate limit, sleep & retry until successful.
    """
    while True:
        resp = requests.get(url, headers=github_api_headers(), params=params or {})
        if resp.status_code == 200:
            return resp
        handle_rate_limit(resp)

def extract_owner_repo(github_url: str) -> Tuple[str, str]:
    """
    Given a GitHub URL like “https://github.com/owner/repo” or “https://github.com/owner/repo/”,
    return (owner, repo). Raises if URL is not in expected form.
    """
    pattern = r"https?://github\.com/([^/]+)/([^/]+)(?:/.*)?"
    m = re.match(pattern, github_url)
    if not m:
        raise ValueError(f"Invalid GitHub URL: {github_url}")
    owner, repo = m.group(1), m.group(2).rstrip("/")
    return owner, repo

def get_default_branch(owner: str, repo: str) -> str:
    """
    GET /repos/{owner}/{repo} → look at “default_branch”
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    resp = github_get(url)
    data = resp.json()
    return data.get("default_branch", "main")

def list_all_files(owner: str, repo: str, branch: str) -> List[Dict]:
    """
    Uses “Get a repository tree” endpoint with recursive=1 to fetch every file (and directory)
    in the default branch, returning a list of { "path": ..., "mode": ..., "type": "blob"/"tree", "sha": ..., "url": ... }.
    Endpoint: GET /repos/{owner}/{repo}/git/trees/{branch}?recursive=1
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}"
    params = {"recursive": "1"}
    resp = github_get(url, params=params)
    data = resp.json()
    if "tree" not in data:
        raise RuntimeError("Could not retrieve repo tree.")
    return data["tree"]

def is_script_file(path: str) -> bool:
    """
    Return True if the path ends with one of our SCRIPT_EXTENSIONS.
    """
    lower = path.lower()
    for ext in SCRIPT_EXTENSIONS:
        if lower.endswith(ext):
            return True
    return False

def get_recent_script_files(owner: str, repo: str, branch: str, n: int) -> List[str]:
    """
    Walk the commits on the default branch (paginated) in descending date order.
    For each commit, fetch its details, look at commit["files"], and collect those .path
    that are “script files” (via is_script_file). Stop once we have n unique paths.
    Return that list of n file‐paths.
    """
    selected = []
    seen = set()
    per_page = 30
    page = 1

    while len(selected) < n:
        commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params = {"sha": branch, "per_page": per_page, "page": page}
        resp = github_get(commits_url, params=params)
        commits = resp.json()
        if not commits:
            break
        for commit_summary in commits:
            sha = commit_summary["sha"]
            commit_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
            commit_resp = github_get(commit_url)
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
    Check if “https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}”
    returns HTTP 200 (exists). We do a HEAD request for speed.
    """
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    resp = requests.head(raw_url)
    return resp.status_code == 200

def fetch_raw_file(owner: str, repo: str, branch: str, path: str) -> str:
    """
    Given a file path in the repo, fetch its raw contents from:
      https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}
    Returns the plain-text content. Raises if not found (non‐200).
    """
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    resp = requests.get(raw_url)
    if resp.status_code != 200:
        raise RuntimeError(f"Could not fetch raw file {path}: {resp.status_code}")
    return resp.text

# ------------------------------------------------------------------
#  UTILITIES FOR DETECTGPT‐STYLE CLASSIFICATION
# ------------------------------------------------------------------

def perturb_code(code: str, k: int = DETECTGPT_K) -> List[str]:
    """
    Return up to k slightly‐perturbed variants of the original code.
    Implements:
      1) Random whitespace insertion/deletion (lines),
      2) Swap two top-level statements (if possible).
    """
    variants = []
    lines = code.splitlines()

    for _ in range(k):
        cand = lines.copy()

        # 1) Randomly delete a blank line or insert a blank line
        if random.random() < 0.5:
            blank_indices = [idx for idx, ln in enumerate(cand) if ln.strip() == ""]
            if blank_indices:
                idx = random.choice(blank_indices)
                del cand[idx]
        else:
            idx = random.randint(0, len(cand))
            cand.insert(idx, "")

        # 2) Swap two top-level lines (no leading whitespace, not comments)
        top_level = [
            idx for idx, ln in enumerate(cand)
            if ln.strip() and not ln.strip().startswith("#") and not ln.startswith(" ")
        ]
        if len(top_level) >= 2:
            i1, i2 = random.sample(top_level, 2)
            cand[i1], cand[i2] = cand[i2], cand[i1]

        variants.append("\n".join(cand))

    return variants

def compute_logprob(code_snippet: str) -> float:
    """
    Returns the sum of token log-probabilities for 'code_snippet' under OPENAI_MODEL.
    Uses max_tokens=0 & logprobs=0 to fetch token logprobs only.
    """
    # Truncate if too long
    if len(code_snippet) > 2000:
        code_snippet = (
            code_snippet[:1000]
            + "\n# … [TRUNCATED] …\n"
            + code_snippet[-1000:]
        )

    backoff = OPENAI_BACKOFF_BASE
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            resp = openai.Completion.create(
                model=OPENAI_MODEL,
                prompt=code_snippet,
                max_tokens=0,
                temperature=0.0,
                logprobs=0,  # request logprobs for all tokens
            )
            token_lps = resp.choices[0].logprobs.token_logprobs or []
            total_lp = sum(lp for lp in token_lps if lp is not None)
            return total_lp

        except openai.RateLimitError:
            if attempt < OPENAI_MAX_RETRIES - 1:
                print(f"OpenAI rate limit hit (attempt {attempt+1}/{OPENAI_MAX_RETRIES}). Sleeping for {backoff} seconds…")
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                print(f"OpenAI rate limit hit on final attempt ({attempt+1}). Defaulting to very low log-prob.")
                return float("-1e9")

        except openai.OpenAIError as e:
            print(f"Unexpected OpenAI error: {e}. Defaulting to very low log-prob.")
            return float("-1e9")

    # Fallback (should not reach here)
    return float("-1e9")

def classify_with_detectgpt(code_snippet: str, k: int = DETECTGPT_K, threshold: float = DETECTGPT_THRESHOLD) -> str:
    """
    Implements a mini-DetectGPT:
      1) Compute log-prob of original snippet
      2) Generate k perturbations, compute each log-prob
      3) Curvature = (avg_logprob_perturbations - logprob_original)
      4) If curvature > threshold → "AI", else "Human"
    """
    L_orig = compute_logprob(code_snippet)
    variants = perturb_code(code_snippet, k=k)
    L_vals = [compute_logprob(v) for v in variants]
    avg_L_pert = sum(L_vals) / len(L_vals)
    curvature = avg_L_pert - L_orig

    if curvature > threshold:
        return "AI"
    else:
        return "Human"

# ------------------------------------------------------------------
#  RESULT SUMMARY
# ------------------------------------------------------------------

def summarize_results(overall_file_count: int, human_count: int, ai_count: int) -> None:
    """
    Given the total number of script files in the repo (overall_file_count),
    plus counts of human‐ and AI‐labeled files, write a JSON report containing:
      • "#AllFiles":            overall number of script files in the repo
      • "#Selected Recent Files": number of files actually classified (human + AI)
      • "#Human":               count of files labeled "Human"
      • "#AI":                  count of files labeled "AI"
      • "ai_ratio":             ratio of AI-labeled files (rounded to 3 decimals)
      • "estimation":           "High" if ai_ratio > 0.66, "Medium" if 0.33 < ai_ratio ≤ 0.66, else "Low"
    """
    total_classified = human_count + ai_count
    ai_ratio = (ai_count / total_classified) if total_classified > 0 else 0.0

    if ai_ratio > 0.66:
        estimation = "High"
    elif ai_ratio > 0.33:
        estimation = "Medium"
    else:
        estimation = "Low"

    result = {
        "#AllFiles": overall_file_count,
        "#Selected Recent Files": total_classified,
        "#Human": human_count,
        "#AI": ai_count,
        "ai_ratio": round(ai_ratio, 3),
        "estimation": estimation
    }

    out_path = Path.cwd() / "repo_ai_detection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults written to {out_path.resolve()}")
    print(json.dumps(result, indent=2))

# ------------------------------------------------------------------
#  MAIN SCRIPT
# ------------------------------------------------------------------

def main():
    # We’ll need to reference this in summarize_results
    global total_file_count

    owner, repo = extract_owner_repo(GITHUB_URL)
    print(f"Owner: {owner}, Repo: {repo}")

    # 1) Find default branch
    print("Fetching default branch…")
    default_branch = get_default_branch(owner, repo)
    print(f"Default branch: {default_branch}")

    # 2) List ALL files in the repo’s default branch (via git/trees?recursive=1)
    print("Listing all files in the repo’s tree…")
    tree = list_all_files(owner, repo, default_branch)
    all_files = [entry["path"] for entry in tree if entry["type"] == "blob"]

    # 3) Filter by “script file” extension
    script_files = [p for p in all_files if is_script_file(p)]
    total_file_count = len(script_files)
    print(f"Total script/code files found: {total_file_count}")

    # 4) Find up to NUM_RECENT_FILES*3 most recent script files,
    #    then filter to keep only those whose raw URL returns 200.
    target_candidates = NUM_RECENT_FILES * 3
    print(f"Finding up to {target_candidates} recent script files…")
    recent_candidates = get_recent_script_files(owner, repo, default_branch, target_candidates)
    valid_recent: List[str] = []

    for path in recent_candidates:
        if len(valid_recent) >= NUM_RECENT_FILES:
            break
        if raw_file_exists(owner, repo, default_branch, path):
            valid_recent.append(path)
        else:
            print(f"Warning: raw URL 404 for {path}, skipping.")

    if len(valid_recent) < NUM_RECENT_FILES:
        print(f"Warning: only found {len(valid_recent)} valid files out of {NUM_RECENT_FILES} desired.")
    else:
        print(f"Selected the following {len(valid_recent)} valid recent files:")
    for p in valid_recent:
        print("  -", p)

    # 5) For each valid recent file, fetch raw contents & classify with DetectGPT
    human_count = 0
    ai_count = 0

    for path in valid_recent:
        print(f"\nFetching file: {path} …")
        try:
            code_text = fetch_raw_file(owner, repo, default_branch, path)
        except RuntimeError as e:
            print(f"  Error fetching {path}: {e}. Skipping classification.")
            continue

        print(f"  Classifying {path} with DetectGPT…")
        label = classify_with_detectgpt(code_text, k=DETECTGPT_K, threshold=DETECTGPT_THRESHOLD)
        print(f"    → {label}")

        if label == "Human":
            human_count += 1
        else:
            ai_count += 1

    # 6) Summarize results into JSON
    summarize_results(total_file_count, human_count, ai_count)


if __name__ == "__main__":
    main()
