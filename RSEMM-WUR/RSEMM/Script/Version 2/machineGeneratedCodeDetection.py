#!/usr/bin/env python3
"""
detect_repo_ai.py

Given a GitHub repository URL, fetches the 20 most recently committed source files
(.py, .js, .java, .cpp, .c, .rs, .go), and uses a zero‐shot DetectGPT4Code
approach (Incoder‐6B for FIM perturbations + PolyCoder‐160M for tail scoring)
to estimate whether each file is AI‐generated. Outputs counts and an overall
“High/Medium/Low” AI‐use estimation.

Usage:
    python detect_repo_ai.py <github_repo_url>

Example:
    python detect_repo_ai.py https://github.com/username/reponame.git
"""

import os
import sys
import time
import requests
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ──────────────────────────────────────────────────────────────────────────────
# 1) Configuration & thresholds
# ──────────────────────────────────────────────────────────────────────────────

# File extensions to consider (source code)
SOURCE_EXTS = (".py", ".js", ".java", ".cpp", ".c", ".rs", ".go")

# Up to how many files to analyze
MAX_FILES = 20

# Estimation thresholds (AI_ratio > HIGH → “High”; etc.)
EST_HIGH_THRESHOLD   = 0.66
EST_MEDIUM_THRESHOLD = 0.33

# GitHub API base
GITHUB_API = "https://api.github.com"

# Grab a token from environment (optional)
GITHUB_TOKEN = "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"
HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    **({"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {})
}

# Device (GPU if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# 2) Load surrogate models & tokenizers
# ──────────────────────────────────────────────────────────────────────────────

# (A) For FIM perturbations, use Incoder-6B
SURROGATE_FIM_MODEL    = "facebook/incoder-6B"
fim_tokenizer = AutoTokenizer.from_pretrained(SURROGATE_FIM_MODEL)
fim_model     = AutoModelForCausalLM.from_pretrained(
    SURROGATE_FIM_MODEL,
    trust_remote_code=True
).eval().to(DEVICE)

# (B) For tail‐scoring, use PolyCoder-160M
SURROGATE_SCORER_MODEL = "PolyCoder-160M"
score_tokenizer = AutoTokenizer.from_pretrained(SURROGATE_SCORER_MODEL)
score_model     = AutoModelForCausalLM.from_pretrained(
    SURROGATE_SCORER_MODEL
).eval().to(DEVICE)

# ──────────────────────────────────────────────────────────────────────────────
# 3) GitHub helper routines
# ──────────────────────────────────────────────────────────────────────────────

def parse_github_url(repo_url: str) -> tuple[str, str]:
    """
    Given a GitHub repo URL (with or without “.git”), return (owner, repo).
    Supports:
      - https://github.com/owner/reponame.git
      - git@github.com:owner/reponame.git
      - https://github.com/owner/reponame
    """
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]

    if repo_url.startswith("git@github.com:"):
        _, path = repo_url.split(":", 1)
    elif repo_url.startswith("https://github.com/"):
        path = repo_url[len("https://github.com/"):]
    elif repo_url.startswith("http://github.com/"):
        path = repo_url[len("http://github.com/"):]
    else:
        raise ValueError("Unsupported GitHub URL format")

    path = path.rstrip("/")
    parts = path.split("/")
    if len(parts) != 2:
        raise ValueError("URL must be of the form github.com/owner/repo")
    return parts[0], parts[1]

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
    Uses the Git Trees API to get a recursive list of all file paths in <branch>.
    Returns a list of paths like ['src/main.py', 'README.md', ...].
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch tree: {resp.status_code} {resp.text}")
    data = resp.json()

    file_paths = []
    for entry in data.get("tree", []):
        if entry["type"] == "blob":
            file_paths.append(entry["path"])
    return file_paths

def get_file_last_commit_timestamp(owner: str, repo: str, path: str, branch: str) -> int:
    """
    Use the “List commits” endpoint with `?path=<path>&per_page=1` to get the latest commit
    that touched that file. Returns its UNIX timestamp.
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
    params = {
        "path": path,
        "sha":  branch,
        "per_page": 1
    }
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        return 0  # fallback
    commits = resp.json()
    if not commits:
        return 0
    date_str = commits[0]["commit"]["committer"]["date"]  # e.g. "2025-05-28T14:23:00Z"
    t_struct = time.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    return int(time.mktime(t_struct))

def fetch_raw_file_content(owner: str, repo: str, path: str, branch: str) -> str:
    """
    Fetch the raw text for a given file path (on the specified branch).
    """
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    resp = requests.get(raw_url)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch raw file: {raw_url} → {resp.status_code}")
    return resp.text

# ──────────────────────────────────────────────────────────────────────────────
# 4) DetectGPT4Code: FIM perturbations + γ‐tail scoring
# ──────────────────────────────────────────────────────────────────────────────

def compute_continuation_logprob(code: str, gamma: float = 0.9) -> float:
    """
    Compute log p(code[γ·|code| : ] | code[:γ·|code|]) under PolyCoder.
    We split the tokenized sequence at γ, use past_key_values to cache prefix,
    then accumulate log‐probs for each token in the continuation.
    """
    # Tokenize the full code
    inputs = score_tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=score_model.config.n_positions
    ).to(DEVICE)

    input_ids = inputs["input_ids"][0]  # (seq_len,)
    total_tokens = input_ids.size(0)
    split_idx    = max(1, int(total_tokens * gamma))

    prefix_ids       = input_ids[:split_idx].unsqueeze(0)       # (1, split_idx)
    continuation_ids = input_ids[split_idx:].unsqueeze(0)       # (1, rest)

    logprob = 0.0
    with torch.no_grad():
        # Run prefix once, get past_key_values
        prefix_out = score_model(prefix_ids, use_cache=True)
        past_kv     = prefix_out.past_key_values

        # Score each token in continuation
        for i in range(continuation_ids.size(1)):
            token_id = int(continuation_ids[0, i].item())
            # Get logits for next token
            out = score_model(
                continuation_ids[:, i : i + 1],
                past_key_values=past_kv,
                use_cache=True
            )
            logits   = out.logits  # (1,1,vocab_size)
            past_kv  = out.past_key_values
            # Compute log‐prob of the actual token
            token_logprob = torch.log_softmax(logits[0, 0], dim=-1)[token_id].item()
            logprob += token_logprob

    return logprob

def generate_fim_perturbations(code: str, n_perturb: int = 20, mask_lines: int = 4) -> list[str]:
    """
    Use Incoder-6B to fill‐in‐the‐middle: mask out `mask_lines` consecutive lines,
    then ask Incoder to refill. Repeat n_perturb times.
    Returns a list of perturbed code strings.
    """
    lines = code.splitlines()
    total_lines = len(lines)
    perturbed_versions = []

    for _ in range(n_perturb):
        if total_lines <= mask_lines:
            start = 0
        else:
            start = random.randint(0, total_lines - mask_lines)
        end = start + mask_lines

        prefix = "\n".join(lines[:start])
        suffix = "\n".join(lines[end:])
        fim_input = prefix + "\n[MASK]\n" + suffix

        inputs = fim_tokenizer(
            fim_input,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(DEVICE)

        outputs = fim_model.generate(
            **inputs,
            max_new_tokens=mask_lines * 50,  # ~50 tokens per masked line
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=fim_tokenizer.eos_token_id
        )

        decoded = fim_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        perturbed_versions.append(decoded)

    return perturbed_versions

def detectgpt4code(code_str: str,
                   n_perturb: int = 20,
                   mask_lines: int = 4,
                   gamma: float = 0.9,
                   threshold: float = 0.0) -> bool:
    """
    Zero‐shot AI‐generated code detection (DetectGPT4Code):

      1) Compute orig_lp = log p(tail(code) | head(code)) with PolyCoder.
      2) Generate n_perturb FIM perturbations via Incoder-6B.
      3) For each perturbed code, compute the same tail log‐prob.
      4) flatness = orig_lp ‐ avg(perturbed_lps).
      5) Return True if flatness > threshold.
    """
    # (1) Original tail log‐prob
    orig_lp = compute_continuation_logprob(code_str, gamma=gamma)

    # (2) FIM perturbations
    perturbed_list = generate_fim_perturbations(
        code_str,
        n_perturb=n_perturb,
        mask_lines=mask_lines
    )

    # (3) Score each perturbed version
    pert_lp_list = []
    for pert_code in perturbed_list:
        try:
            lp = compute_continuation_logprob(pert_code, gamma=gamma)
        except Exception:
            lp = orig_lp  # fallback if scoring fails
        pert_lp_list.append(lp)

    avg_pert_lp = sum(pert_lp_list) / len(pert_lp_list)
    flatness     = orig_lp - avg_pert_lp

    return flatness > threshold

# ──────────────────────────────────────────────────────────────────────────────
# 5) Main logic: Fetch top‐N files, detect, and compute estimation
# ──────────────────────────────────────────────────────────────────────────────

def main():

    repo_url = "https://github.com/keras-team/keras"
    try:
        owner, repo = parse_github_url(repo_url)
    except ValueError as e:
        print(f"Error parsing URL: {e}")
        sys.exit(1)

    print(f"Owner: {owner}    Repo: {repo}")

    # 5.1) Get default branch
    default_branch = get_default_branch(owner, repo)
    print(f"Default branch is '{default_branch}'")

    # 5.2) List all files in that branch
    print("Listing every file in the tree…")
    all_paths = list_all_files(owner, repo, default_branch)

    # 5.3) Filter for our source extensions and gather commit timestamps
    candidates = [p for p in all_paths if p.lower().endswith(SOURCE_EXTS)]
    if not candidates:
        print("No source files found. Exiting.")
        sys.exit(1)

    print(f"Found {len(candidates)} total source files. Checking commit dates…")
    file_dates = []
    for path in candidates:
        ts = get_file_last_commit_timestamp(owner, repo, path, default_branch)
        if ts > 0:
            file_dates.append((path, ts))

    if not file_dates:
        print("No valid file commit dates found. Exiting.")
        sys.exit(1)

    # 5.4) Sort descending by timestamp, pick top MAX_FILES
    file_dates.sort(key=lambda x: x[1], reverse=True)
    top_files = file_dates[:MAX_FILES]

    print(f"Analyzing the {len(top_files)} most‐recent files…")
    ai_count    = 0
    human_count = 0

    for (rel_path, ts) in top_files:
        try:
            content = fetch_raw_file_content(owner, repo, rel_path, default_branch)
        except Exception as e:
            print(f"  • Skipping {rel_path} (cannot fetch raw: {e})")
            continue

        # Run DetectGPT4Code on this file
        is_ai = detectgpt4code(
            code_str=content,
            n_perturb=20,
            mask_lines=4,
            gamma=0.9,
            threshold=0.0
        )

        ts_human = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))
        if is_ai:
            ai_count += 1
            print(f"  [AI]    {rel_path}    (last commit: {ts_human} UTC)")
        else:
            human_count += 1
            print(f"  [HUMAN] {rel_path}    (last commit: {ts_human} UTC)")

    total   = ai_count + human_count
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
        "AI_Ratio":   round(ai_ratio, 4),
        "Estimation": estimation
    }

    out_path = Path.cwd() / "repo_ai_detection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        import json
        json.dump(result, f, indent=2)

    print(f"\nResults written to {out_path.resolve()}")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

