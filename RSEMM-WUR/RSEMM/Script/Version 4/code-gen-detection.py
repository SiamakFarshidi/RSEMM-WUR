#!/usr/bin/env python3
"""
zero_shot_detect.py

Given a GitHub repo URL (hard‐coded via a variable in main), this script:
  1) Lists all files in the default branch (via GitHub’s “get tree” endpoint),
  2) Counts how many of them are “script files” (based on extensions),
  3) Finds up to NUM_RECENT_FILES*3 most recently modified script files (by walking recent commits),
     then filters out any that 404 on raw.githubusercontent, so we end up with exactly NUM_RECENT_FILES valid files (if possible),
  4) For each valid file, fetches its raw contents and asks OpenAI’s GPT‐3.5‐Turbo 
     (zero‐shot) to decide whether that snippet is AI‐generated or human‐written,
  5) Respects GitHub’s rate limits (sleeping/retrying when necessary),
  6) Implements a bounded retry‐with‐backoff loop for OpenAI rate limits,
  7) Emits a summary JSON (“repo_ai_detection.json”) with:
       • "#AllFiles":  total number of files in the repo
       • "#Selected Recent Files": count of files actually classified (human+AI),
       • "#Human": count of files labeled "Human",
       • "#AI":    count of files labeled "AI",
       • "ai_ratio": the ratio of AI‐labeled files (rounded to 3 decimals),
       • "estimation": “High” if ai_ratio>0.66, “Medium” if 0.33<ai_ratio<=0.66, else “Low”,
       • "analyzed-scripts": a mapping from filename → label ("AI" or "Human").
"""

import os
import re
import time
import json
import requests
import openai   # pip install openai>=1.0.0
from pathlib import Path
from typing import List, Dict, Tuple
from openai.error import RateLimitError

# ------------------------------------------------------------------------------

# 1) Put your GitHub repo URL here. It must be of the form https://github.com/owner/repo
GITHUB_URL = "https://github.com/SiamakFarshidi/curriculum"  # ← <– REPLACE with the actual URL

# 2) (Optional) If you set GITHUB_TOKEN as an environment variable, we’ll use it to raise
#    the rate limit from 60 → 5,000 requests/hour. If not set, unauthenticated calls will be used.
GITHUB_TOKEN = "token"

# 3) OpenAI API key must be set as an env var named “OPENAI_API_KEY”
#    (e.g. export OPENAI_API_KEY="sk‐…")
openai.api_key = "token"

if not openai.api_key:
    raise RuntimeError("Please set OPENAI_API_KEY as an environment variable.")

# 4) How many recent script files to classify
NUM_RECENT_FILES = 10

# 5) What file extensions we consider “script/code files.” 
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

# 6) Which OpenAI model to use for zero‐shot “AI vs. Human” classification.
OPENAI_MODEL = "gpt-3.5-turbo"

# 7) How many seconds to wait before retrying if GitHub rate limit is hit.
RATE_LIMIT_SLEEP = 5

# 8) Maximum retries for OpenAI RateLimit + base backoff seconds
OPENAI_MAX_RETRIES = 5
OPENAI_BACKOFF_BASE = 10  # first retry waits 10s, next waits 20s, then 40s, etc.

# ------------------------------------------------------------------------------

def github_api_headers() -> Dict[str, str]:
    """
    Return appropriate headers for GitHub API. If GITHUB_TOKEN is set, use it for Authorization.
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "script-zero-shot-detect"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def handle_rate_limit(response: requests.Response) -> None:
    """
    If the response indicates we’ve hit GitHub’s rate limit (status_code 403 or 429,
    with “X-RateLimit-Remaining” = 0), parse the “X-RateLimit-Reset” header and sleep
    until that time (plus a small buffer). Otherwise, sleep a short default interval
    before retrying.
    """
    if response.status_code in (403, 429):
        ratelimit_remaining = response.headers.get("X-RateLimit-Remaining")
        if ratelimit_remaining == "0":
            reset_ts = response.headers.get("X-RateLimit-Reset")
            if reset_ts:
                reset_ts = int(reset_ts)
                now_ts = int(time.time())
                wait_sec = reset_ts - now_ts + 5
                if wait_sec > 0:
                    print(f"GitHub rate limit hit. Sleeping for {wait_sec} seconds…")
                    time.sleep(wait_sec)
                    return
        # If we get here, either RateLimit‐Remaining ≠ 0 or no reset header; sleep briefly
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
        # then loop & retry


def extract_owner_repo(github_url: str) -> Tuple[str, str]:
    """
    Given a GitHub URL like “https://github.com/owner/repo” or “https://github.com/owner/repo/”,
    return (owner, repo). Raises if the URL is not in the expected form.
    """
    pattern = r"https?://github\.com/([^/]+)/([^/]+)(?:/.*)?"
    m = re.match(pattern, github_url)
    if not m:
        raise ValueError(f"Invalid GitHub URL: {github_url}")
    owner, repo = m.group(1), m.group(2).rstrip("/")
    return owner, repo


def get_default_branch(owner: str, repo: str) -> str:
    """
    GET /repos/{owner}/{repo}  → look at “default_branch”
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    resp = github_get(url)
    data = resp.json()
    return data.get("default_branch", "main")


def list_all_files(owner: str, repo: str, branch: str) -> List[Dict]:
    """
    Uses the “Get a repository tree” endpoint with recursive=1 to fetch
    every file (and directory) in the default branch, returning a list of
    { "path": ..., "mode": ..., "type": "blob"/"tree", "sha": ..., "url": ... }.

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
    E.g. “src/main.py” → True
    """
    lower = path.lower()
    for ext in SCRIPT_EXTENSIONS:
        if lower.endswith(ext):
            return True
    return False


def get_recent_script_files(owner: str, repo: str, branch: str, n: int) -> List[str]:
    """
    Walk the commits on the default branch (paginated) in descending date order.
    For each commit, fetch its details (GET /repos/{owner}/{repo}/commits/{sha}),
    look at commit["files"], and collect those .path that are “script files”
    (via is_script_file). Stop once we have n unique paths. Return that list of n file‐paths.
    """
    selected = []
    seen = set()
    per_page = 30
    page = 1

    while len(selected) < n:
        commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params = {
            "sha": branch,
            "per_page": per_page,
            "page": page
        }
        resp = github_get(commits_url, params=params)
        commits = resp.json()
        if not commits:
            break  # no more commits
        for commit_summary in commits:
            sha = commit_summary["sha"]
            # Fetch commit details
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
    Returns the plain‐text content. Raises if not found (non‐200).
    """
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    resp = requests.get(raw_url)
    if resp.status_code != 200:
        raise RuntimeError(f"Could not fetch raw file {path}: {resp.status_code}")
    return resp.text


def classify_with_gpt(code_snippet: str) -> str:
    """
    Sends ‘code_snippet’ to OpenAI’s GPT (via chat.completions.create) and asks it (zero‐shot)
    to label whether it was generated by AI or written by a human. Returns exactly "AI" or "Human".

    Implements:
      • A short sleep before making the API call (to space out requests).
      • A retry‐with‐exponential backoff loop up to OPENAI_MAX_RETRIES if a RateLimitError is encountered.
      • After max retries, logs a warning and defaults to labeling “Human” so the script can continue.
    """
    # 1) If code is too large, truncate: keep first 2000 chars + last 2000 chars
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

    # 2) Short sleep to avoid bursts
    time.sleep(1)

    # 3) Retry with exponential backoff
    backoff = OPENAI_BACKOFF_BASE
    for attempt in range(OPENAI_MAX_RETRIES):
        try:
            resp = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert at detecting machine-generated code."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.0,
                max_tokens=8,
            )
            text = resp.choices[0].message.content.strip()
            lower = text.lower()
            if lower.startswith("ai"):
                return "AI"
            elif lower.startswith("human"):
                return "Human"
            else:
                # If GPT says something like “Probably human”, force‐map:
                if "human" in lower:
                    return "Human"
                elif "ai" in lower or "machine" in lower:
                    return "AI"
                else:
                    # Unclear output → default to “Human”
                    return "Human"

        except RateLimitError:
            if attempt < OPENAI_MAX_RETRIES - 1:
                print(f"OpenAI rate limit hit (attempt {attempt+1}/{OPENAI_MAX_RETRIES}). "
                      f"Sleeping for {backoff} seconds…")
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                print(f"OpenAI rate limit hit on final attempt ({attempt+1}). Defaulting to 'Human'.")
                return "Human"
        except Exception as e:
            # For other errors (network, invalid request, etc.), print and default:
            print(f"Unexpected OpenAI error: {e}. Defaulting to 'Human'.")
            return "Human"

    # Fallback (should not reach here)
    return "Human"


def summarize_results(overall_file_count: int,
                      human_count: int,
                      ai_count: int,
                      analyzed_scripts: Dict[str, str]) -> None:
    """
    Given the total number of files in the repo, counts of human‐ and AI‐labeled files,
    and a mapping analyzed_scripts: filename → "AI"/"Human", write a JSON report containing:
      • "#AllFiles":  overall number of files in the repo
      • "#Selected Recent Files": number of files actually classified
      • "#Human":     count of files labeled "Human"
      • "#AI":        count of files labeled "AI"
      • "ai_ratio":   ratio of AI‐labeled files (rounded to 3 decimals)
      • "estimation": "High" if ai_ratio > 0.66, "Medium" if 0.33 < ai_ratio ≤ 0.66, else "Low"
      • "analyzed-scripts": mapping from filename → label
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
        "estimation": estimation,
        "analyzed-scripts": analyzed_scripts
    }

    out_path = Path.cwd() / "repo_ai_detection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults written to {out_path.resolve()}")
    print(json.dumps(result, indent=2))


def main():
    owner, repo = extract_owner_repo(GITHUB_URL)
    print(f"Owner: {owner}, Repo: {repo}")

    # 1) Find default branch
    print("Fetching default branch…")
    default_branch = get_default_branch(owner, repo)
    print(f"Default branch: {default_branch}")

    # 2) List ALL files in the repo’s default branch
    print("Listing all files in the repo’s tree…")
    tree = list_all_files(owner, repo, default_branch)
    all_files = [entry["path"] for entry in tree if entry["type"] == "blob"]

    # 3) Filter by “script file” extension
    script_files = [p for p in all_files if is_script_file(p)]
    total_script_count = len(script_files)
    print(f"Total script/code files found: {total_script_count}")

    # 4) Find up to NUM_RECENT_FILES*3 most recent script files via commits
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

    # 5) For each valid recent file, fetch raw contents & classify with GPT
    human_count = 0
    ai_count = 0

    # NEW: keep a mapping filename → label
    analyzed_scripts: Dict[str, str] = {}

    for path in valid_recent:
        print(f"\nFetching file: {path} …")
        try:
            code_text = fetch_raw_file(owner, repo, default_branch, path)
        except RuntimeError as e:
            print(f"  Error fetching {path}: {e}. Skipping classification.")
            continue

        print(f"  Classifying {path} with GPT…")
        label = classify_with_gpt(code_text)
        print(f"    → {label}")

        # Record the label
        analyzed_scripts[path] = label

        if label == "Human":
            human_count += 1
        else:
            ai_count += 1

    # 6) Summarize results into JSON (including per-file labels)
    summarize_results(total_script_count, human_count, ai_count, analyzed_scripts)


if __name__ == "__main__":
    main()
