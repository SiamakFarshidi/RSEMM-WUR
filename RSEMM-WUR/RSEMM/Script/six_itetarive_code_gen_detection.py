#!/usr/bin/env python3
"""
zero_shot_detect_bulk.py

Iterate over all JSON files in a given directory. For each file:
  1) Parse the JSON and attempt to find a GitHub repository URL.
  2) If a GitHub URL is found, measure current rate-limit remaining, then perform
     the “zero‐shot” AI vs. Human code analysis on that repository, producing a summary dict.
  3) After processing, measure rate-limit remaining again to compute how many tokens were used.
     • Print “remaining before,” “remaining after,” and “used” for each repo.
     • If a repo used more than 4000 tokens, skip it (do not write any summary).
  4) If no GitHub URL is found, produce a default summary with zeroed counts.
  5) In either case (unless skipped by the 4000‐token rule), ensure the JSON has a
     “repo-profile” key (create if missing), then add or overwrite a subkey
     “code-gen-evaluation” whose value is the summary dict.
  6) Write the updated JSON back to disk (unless skipped).

All original functionality – including GitHub rate-limit handling, network-retry loops,
OpenAI classification, and script-file detection – is preserved. If any network error
occurs when talking to GitHub or fetching raw files, the script will sleep and retry until
the network is stable. If the GitHub rate limit is hit, it will wait until reset.
"""

import os
import re
import time
import json
import requests
import openai   # pip install openai>=1.0.0
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from openai.error import RateLimitError
from requests.exceptions import RequestException
from pathlib import Path

# ------------------------------------------------------------------------------

# 1) Input directory containing JSON files
INPUT_DIR = Path(r"C:\Users\User\Downloads\RS-Zenodo\RS-Repo-Zenodo - v4")

# 2) (Optional) GitHub token to increase rate limit from 60 → 5,000 req/hour
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "Key")

# 3) OpenAI API key must be set as an env var named “OPENAI_API_KEY” (or hard-coded here)
openai.api_key = os.getenv("OPENAI_API_KEY", "Key")
if not openai.api_key:
    raise RuntimeError("Please set OPENAI_API_KEY as an environment variable.")

# 4) How many recent script files to classify per repository
NUM_RECENT_FILES = 5

# 5) Which file extensions we consider “script/code files.”
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

# 6) Which OpenAI model to use for zero-shot “AI vs. Human” classification.
OPENAI_MODEL = "gpt-3.5-turbo"

# 7) How many seconds to wait if GitHub rate limit is hit and no reset header is provided
RATE_LIMIT_SLEEP = 5

# 8) Maximum retries for OpenAI RateLimit + base backoff seconds
OPENAI_MAX_RETRIES = 5
OPENAI_BACKOFF_BASE = 10  # first retry waits 10s, next waits 20s, then 40s, etc.

# 9) How long to sleep (seconds) on network errors before retrying
NETWORK_SLEEP = 5

# Regex to detect a GitHub URL like “https://github.com/owner/repo”
GITHUB_URL_PATTERN = re.compile(r"https?://github\.com/([^/]+)/([^/]+)(?:/.*)?", re.IGNORECASE)


def github_api_headers() -> Dict[str, str]:
    """
    Return appropriate headers for GitHub API. If GITHUB_TOKEN is set, use it for Authorization.
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "script-zero-shot-detect-bulk"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def handle_rate_limit(response: requests.Response) -> None:
    """
    If the response indicates we've hit GitHub's rate limit (status_code 403 or 429),
    check for 'Retry-After' first. If present, sleep that many seconds + a small buffer.
    Otherwise, parse the 'X-RateLimit-Remaining' and 'X-RateLimit-Reset' headers.
    Sleep until reset_ts (no extra cushion), then an additional 2 seconds to be safe.
    If none of those headers exist, sleep a short default interval before retrying.
    """
    if response.status_code in (403, 429):
        rem = response.headers.get("X-RateLimit-Remaining")
        reset_hdr = response.headers.get("X-RateLimit-Reset")
        retry_after = response.headers.get("Retry-After")

        now_ts = int(time.time())
        print(f"[GitHub] HTTP {response.status_code}; X-RateLimit-Remaining={rem}; "
              f"X-RateLimit-Reset={reset_hdr}; Retry-After={retry_after}")
        print(f"[GitHub] local now_ts = {now_ts} ({time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(now_ts))} UTC)")

        if retry_after:
            # GitHub explicitly told us how many seconds to wait
            wait_sec = int(retry_after) + 2
            print(f"[GitHub] Sleeping for Retry-After {retry_after} + 2s buffer = {wait_sec} seconds…")
            time.sleep(wait_sec)
            return

        if rem == "0" and reset_hdr:
            reset_ts = int(reset_hdr)
            # Sleep until reset_ts, then 2-second buffer
            wait_sec = max(0, reset_ts - now_ts) + 2
            print(f"[GitHub] Sleeping until reset_ts={reset_ts} "
                  f"({time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(reset_ts))} UTC) + 2s buffer = {wait_sec} seconds…")
            time.sleep(wait_sec)
            return

        # Fallback: no relevant headers or remaining != 0
        print(f"[GitHub] Fallback sleep for {RATE_LIMIT_SLEEP}s…")
        time.sleep(RATE_LIMIT_SLEEP)


def github_get(url: str, params: Dict[str, Any] = None) -> requests.Response:
    """
    Wrapper around requests.get that handles GitHub rate limits and network errors automatically:
      • If we hit rate limit, sleep & retry until successful.
      • If network error, sleep & retry until network is stable.
      • If genuine 404/401 (repo not found or unauthorized), raise immediately.
    """
    while True:
        try:
            resp = requests.get(url, headers=github_api_headers(), params=params or {})
        except RequestException as e:
            print(f"[Network] Error when GET {url}: {e}. Sleeping for {NETWORK_SLEEP}s…")
            time.sleep(NETWORK_SLEEP)
            continue

        # If the repo truly doesn’t exist (404) or is private without access (401), bail out immediately:
        if resp.status_code == 404:
            raise RuntimeError(f"GitHub resource not found (404): {url}")
        if resp.status_code == 401:
            raise RuntimeError(f"Unauthorized (401) accessing: {url}")

        # Normal success
        if resp.status_code == 200:
            return resp

        # Otherwise, possibly a rate-limit (403/429) or a transient server error
        handle_rate_limit(resp)
        # then loop & retry


def extract_owner_repo(github_url: str) -> Tuple[str, str]:
    """
    Given a GitHub URL like “https://github.com/owner/repo” or “https://github.com/owner/repo.git”,
    return (owner, repo). Strips a trailing “.git” if present. Raises if the URL is not in the expected form.
    """
    m = GITHUB_URL_PATTERN.match(github_url)
    if not m:
        raise ValueError(f"Invalid GitHub URL: {github_url}")
    owner = m.group(1)
    repo = m.group(2).rstrip("/")
    # If it ends with “.git”, strip that off so GitHub API won't 404/403
    if repo.lower().endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def get_default_branch(owner: str, repo: str) -> str:
    """
    GET /repos/{owner}/{repo}  → look at “default_branch”.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    resp = github_get(url)
    data = resp.json()
    return data.get("default_branch", "main")


def list_all_files(owner: str, repo: str, branch: str) -> List[Dict[str, Any]]:
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
        raise RuntimeError(f"Could not retrieve repo tree for {owner}/{repo}.")
    return data["tree"]


def is_script_file(path: str) -> bool:
    """
    Return True if the path ends with one of our SCRIPT_EXTENSIONS.
    E.g. “src/main.py” → True
    """
    lower = path.lower()
    return any(lower.endswith(ext) for ext in SCRIPT_EXTENSIONS)


def get_recent_script_files(owner: str, repo: str, branch: str, n: int) -> List[str]:
    """
    Walk the commits on the default branch (paginated) in descending date order.
    For each commit, fetch its details (GET /repos/{owner}/{repo}/commits/{sha}),
    look at commit["files"], and collect those .path that are “script files”
    (via is_script_file). Stop once we have n unique paths. Return that list of n file‐paths.
    """
    selected: List[str] = []
    seen: set = set()
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
            sha = commit_summary.get("sha")
            if not sha:
                continue
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
    Retries on network errors.
    """
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    while True:
        try:
            resp = requests.head(raw_url)
        except RequestException as e:
            print(f"[Network] HEAD {raw_url} error: {e}. Sleeping {NETWORK_SLEEP}s…")
            time.sleep(NETWORK_SLEEP)
            continue
        return resp.status_code == 200


def fetch_raw_file(owner: str, repo: str, branch: str, path: str) -> str:
    """
    Given a file path in the repo, fetch its raw contents from:
      https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}
    Returns the plain‐text content. Retries on network errors; raises if non-200.
    """
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    while True:
        try:
            resp = requests.get(raw_url)
        except RequestException as e:
            print(f"[Network] GET {raw_url} error: {e}. Sleeping {NETWORK_SLEEP}s…")
            time.sleep(NETWORK_SLEEP)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"Could not fetch raw file {path}: HTTP {resp.status_code}")
        return resp.text


def classify_with_gpt(code_snippet: str) -> str:
    """
    Sends ‘code_snippet’ to OpenAI’s GPT (via chat.completions.create) and asks it (zero‐shot)
    to label whether it was generated by AI or written by a human. Returns exactly "AI" or "Human".

    Implements:
      • Truncation if code > 4000 chars
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
                print(f"[OpenAI] Rate limit hit (attempt {attempt+1}/{OPENAI_MAX_RETRIES}). "
                      f"Sleeping for {backoff} seconds…")
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                print(f"[OpenAI] Rate limit hit on final attempt ({attempt+1}). Defaulting to 'Human'.")
                return "Human"
        except Exception as e:
            # For other errors (network, invalid request, etc.), print and default:
            print(f"[OpenAI] Unexpected error: {e}. Defaulting to 'Human'.")
            return "Human"

    # Fallback (should not reach here)
    return "Human"


def summarize_results(overall_file_count: int,
                      human_count: int,
                      ai_count: int,
                      analyzed_scripts: Dict[str, str]) -> Dict[str, Union[int, float, str, Dict[str, str]]]:
    """
    Given the total number of script files in the repo, counts of human‐ and AI‐labeled files,
    and a mapping analyzed_scripts: filename → "AI"/"Human", construct a summary dict containing:
      • "#AllFiles":  overall number of script/code files in the repo
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
    return result


def default_summary() -> Dict[str, Union[int, float, str, Dict[str, str]]]:
    """
    Return a default summary dict when no GitHub URL is found:
      • All counts zero, ai_ratio = 0.0, estimation = "Low", analyzed-scripts empty.
    """
    return {
        "#AllFiles": 0,
        "#Selected Recent Files": 0,
        "#Human": 0,
        "#AI": 0,
        "ai_ratio": 0.0,
        "estimation": "Low",
        "analyzed-scripts": {}
    }


def find_github_url(data: Any) -> Union[str, None]:
    """
    Recursively search through a JSON-like structure (dicts, lists) for a string that matches
    a GitHub URL pattern (https://github.com/owner/repo). Return the first match, or None.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                m = GITHUB_URL_PATTERN.match(value)
                if m:
                    owner, repo = m.group(1), m.group(2)
                    return f"https://github.com/{owner}/{repo}"
            else:
                maybe = find_github_url(value)
                if maybe:
                    return maybe
    elif isinstance(data, list):
        for item in data:
            maybe = find_github_url(item)
            if maybe:
                return maybe
    return None


def process_repository(github_url: str) -> Dict[str, Union[int, float, str, Dict[str, str]]]:
    """
    Given a GitHub URL (https://github.com/owner/repo), run the entire zero‐shot analysis:
      1) Determine default branch
      2) List all files, filter script/code files, count them
      3) Identify up to NUM_RECENT_FILES*3 recent script files; keep up to NUM_RECENT_FILES that exist in raw
      4) For each valid recent file, fetch raw content & classify with GPT
      5) Return a summary dict via summarize_results()

    Any network or rate‐limit issues are handled internally (sleep & retry).
    """
    try:
        owner, repo = extract_owner_repo(github_url)
    except ValueError as e:
        print(f"[WARN] Invalid GitHub URL '{github_url}': {e}")
        return default_summary()

    print(f"[Repo] Processing {owner}/{repo}…")

    # 1) Find default branch
    try:
        print("  Fetching default branch…")
        default_branch = get_default_branch(owner, repo)
        print(f"  Default branch: {default_branch}")
    except Exception as e:
        print(f"  [ERROR] Could not fetch default branch for {owner}/{repo}: {e}")
        return default_summary()

    # 2) List all files in the repo’s default branch
    try:
        print("  Listing all files in the repo’s tree…")
        tree = list_all_files(owner, repo, default_branch)
        all_blobs = [entry["path"] for entry in tree if entry.get("type") == "blob"]
        script_files = [p for p in all_blobs if is_script_file(p)]
        total_script_count = len(script_files)
        print(f"  Total script/code files found: {total_script_count}")
    except Exception as e:
        print(f"  [ERROR] Could not list files for {owner}/{repo}: {e}")
        return default_summary()

    # 3) Find up to NUM_RECENT_FILES*3 most recent script files via commits
    target_candidates = NUM_RECENT_FILES * 3
    print(f"  Finding up to {target_candidates} recent script files…")
    try:
        recent_candidates = get_recent_script_files(owner, repo, default_branch, target_candidates)
    except Exception as e:
        print(f"  [ERROR] Could not fetch recent commits for {owner}/{repo}: {e}")
        recent_candidates = []

    valid_recent: List[str] = []
    for path in recent_candidates:
        if len(valid_recent) >= NUM_RECENT_FILES:
            break
        try:
            if raw_file_exists(owner, repo, default_branch, path):
                valid_recent.append(path)
            else:
                print(f"    [WARN] raw URL 404 for {path}, skipping.")
        except Exception as e:
            print(f"    [WARN] Error checking raw existence for {path}: {e}. Retrying later.")
            continue

    if len(valid_recent) < NUM_RECENT_FILES:
        print(f"  [WARN] Only found {len(valid_recent)} valid files out of {NUM_RECENT_FILES} desired.")
    else:
        print(f"  Selected {len(valid_recent)} valid recent files for classification.")

    # 4) For each valid recent file, fetch raw contents & classify with GPT
    human_count = 0
    ai_count = 0
    analyzed_scripts: Dict[str, str] = {}

    for path in valid_recent:
        print(f"    Fetching file: {path} …")
        try:
            code_text = fetch_raw_file(owner, repo, default_branch, path)
        except RuntimeError as e:
            print(f"      [ERROR] Could not fetch {path}: {e}. Skipping classification.")
            continue

        print(f"      Classifying {path} with GPT…")
        label = classify_with_gpt(code_text)
        print(f"        → {label}")

        analyzed_scripts[path] = label
        if label == "Human":
            human_count += 1
        else:
            ai_count += 1

    # 5) Summarize results into a dict
    summary = summarize_results(total_script_count, human_count, ai_count, analyzed_scripts)
    return summary


def get_core_rate_limit() -> Tuple[int, int]:
    """
    Return (limit, remaining) for the 'core' rate-limit category via GitHub's /rate_limit endpoint.
    If the request fails, return (-1, -1).
    """
    try:
        resp = requests.get("https://api.github.com/rate_limit", headers=github_api_headers())
        data = resp.json().get("resources", {}).get("core", {})
        return data.get("limit", -1), data.get("remaining", -1)
    except Exception:
        return -1, -1


def ensure_rate_limit(threshold=50):
    """
    Check current rate-limit remaining. If remaining < threshold, sleep until reset + 2 seconds.
    """
    try:
        resp = requests.get("https://api.github.com/rate_limit", headers=github_api_headers())
        core_data = resp.json().get("resources", {}).get("core", {})
        limit = core_data.get("limit", -1)
        remaining = core_data.get("remaining", -1)
        reset_ts = core_data.get("reset", 0)
    except Exception:
        # Could not fetch rate-limit; just return
        return

    if remaining < threshold and reset_ts > 0:
        now_ts = int(time.time())
        to_sleep = reset_ts - now_ts + 2
        if to_sleep > 0:
            print(f"[RateLimit] only {remaining} remaining ({limit} total); "
                  f"sleeping {to_sleep} seconds until reset at {reset_ts}…")
            time.sleep(to_sleep)



def getGenCodeUsageEvaluation(json_path: Union[str, Path]):
    json_path = Path(json_path)

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not read '{json_path}': {e}. Skipping.")
        return

    # Skip if already has code-gen-evaluation
    if isinstance(data.get("repo-profile"), dict) and "code-gen-evaluation" in data["repo-profile"]:
        print("  → Skipping (already contains code-gen-evaluation).")
        return

    # Before processing, ensure we have enough tokens left
    ensure_rate_limit(threshold=200)

    # 1) Attempt to find a GitHub URL anywhere in the JSON
    github_url = find_github_url(data)
    if not github_url:
        print("  No GitHub URL found in JSON. Generating default summary.")
        summary = default_summary()
        # Write default summary back immediately
        if "repo-profile" not in data or not isinstance(data["repo-profile"], dict):
            data["repo-profile"] = {}
        data["repo-profile"]["code-gen-evaluation"] = summary
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"  Updated '{json_path.name}' with default code-gen-evaluation.")
        except Exception as e:
            print(f"  [ERROR] Could not write updated JSON to '{json_path}': {e}.")
        return

    # If we reached here, there's a GitHub URL
    print(f"  Found GitHub URL: {github_url}")

    # Snapshot: tokens remaining before processing
    before_limit, before_remain = get_core_rate_limit()
    if before_limit >= 0:
        print(f"    → [Pre-Check] core.limit={before_limit}, remaining={before_remain}")
    else:
        print("    → [Pre-Check] Could not fetch /rate_limit")

    # Process the repository (this will burn tokens)
    try:
        summary = process_repository(github_url)
    except RuntimeError as e:
        print(f"    [ERROR] Processing repository failed: {e}. Writing default summary.")
        summary = default_summary()

    # Snapshot: tokens remaining after processing
    after_limit, after_remain = get_core_rate_limit()
    if after_limit >= 0:
        print(f"    → [Post-Check] core.limit={after_limit}, remaining={after_remain}")
        if before_remain >= 0 and after_remain >= 0:
            used = before_remain - after_remain
            print(f"    → [Usage] This repo used {used} API calls.")
            # If a repo used more than 4000 API calls, skip writing summary
            if used > 4000:
                print(f"    *** SKIPPING {github_url} (used {used} API calls > 4000).")
                return
    else:
        print("    → [Post-Check] Could not fetch /rate_limit")

    # 2) Ensure “repo-profile” exists and is a dict
    if "repo-profile" not in data or not isinstance(data["repo-profile"], dict):
        data["repo-profile"] = {}

    # 3) Insert or overwrite “code-gen-evaluation” under “repo-profile”
    data["repo-profile"]["code-gen-evaluation"] = summary

    # 4) Write the updated JSON back to the same file
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  Updated '{json_path.name}' with code-gen-evaluation.")
    except Exception as e:
        print(f"  [ERROR] Could not write updated JSON to '{json_path}': {e}.")





def main():
    # Ensure the input directory exists
    if not INPUT_DIR.is_dir():
        raise RuntimeError(f"Input directory '{INPUT_DIR}' does not exist or is not a directory.")

    # Iterate over all *.json files in the directory
    json_files = list(INPUT_DIR.glob("*.json"))
    if not json_files:
        print(f"[Info] No JSON files found in '{INPUT_DIR}'. Nothing to do.")
        return

    for json_path in json_files:
        print(f"\n[File] Processing '{json_path.name}'…")
        getGenCodeUsageEvaluation(json_path)



    print("\n[Done] All JSON files processed.")


if __name__ == "__main__":
    main()
