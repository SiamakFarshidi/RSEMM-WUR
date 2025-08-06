#!/usr/bin/env python3
# combined_repo_profile.py

import os
import sys
import json
import time
import re
import shutil
import stat
import tempfile
import subprocess
import tarfile
import requests
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ──────────────────────────────────────────────────────────────────────────────
# 0) Hard-coded configuration (replace with your token)
# ──────────────────────────────────────────────────────────────────────────────

input_dir = r"C:\Users\User\Downloads\RS-Zenodo\RS-Repo-Zenodo - v3"

GITHUB_TOKEN = "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"  # ←–– Replace with your token
GITHUB_API   = "https://api.github.com"
HEADERS      = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {GITHUB_TOKEN}"
}

# For GenAI detection
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL_NAME = "distilgpt2"
DEVICE     = "cpu"      # change to "cuda" if you have a GPU + torch installed
EST_HIGH_THRESHOLD   = 0.66
EST_MEDIUM_THRESHOLD = 0.33
SOURCE_EXTS = (".py", ".js", ".java", ".cpp", ".c", ".rs", ".go")
MAX_FILES   = 20

# For SE estimation
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
]
ORG_THRESHOLDS = {
    "community":     2.0000,
    "ci":            1.0000,
    "documentation": 0.0018660,
    "history":       2.0895,
    "issues":        0.022989,
    "license":       1.0000,
    "unittest":      0.0010160,
}

# For FAIRness estimation
FAIR_REQUIRED_F2_FIELDS = ["title", "creators", "description", "license", "language"]
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


# ──────────────────────────────────────────────────────────────────────────────
# 1) GenAI-in-development detection helpers
# ──────────────────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval().to(DEVICE)
max_length = model.config.n_positions if hasattr(model.config, "n_positions") else model.config.n_ctx


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
    if len(lines) <= 1:
        return code
    drop_count = max(1, int(len(lines) * mask_frac))
    # use torch.randperm to pick random lines to drop
    idxs = torch.randperm(len(lines))[:drop_count].tolist()
    to_drop = set(idxs)
    return "\n".join(line for i, line in enumerate(lines) if i not in to_drop)

def check_genai_file_content(code_str: str, n_perturb: int = 5, mask_frac: float = 0.1) -> bool:
    baseline_lp = compute_logprob(code_str)
    perturbed_lps = []
    for _ in range(n_perturb):
        pert_code = generate_perturbation(code_str, mask_frac=mask_frac)
        perturbed_lps.append(compute_logprob(pert_code))
    avg_perturbed  = sum(perturbed_lps) / len(perturbed_lps)
    flatness_score = baseline_lp - avg_perturbed
    return flatness_score > 0

# ──────────────────────────────────────────────────────────────────────────────
# 2) GitHub API–based helpers (shared across GenAI & SE)
# ──────────────────────────────────────────────────────────────────────────────

def parse_github_url(repo_url: str) -> tuple[str, str, str | None]:
    """
    Given a GitHub URL like:
      - https://github.com/owner/repo.git
      - git@github.com:owner/repo.git
      - https://github.com/owner/repo
      - https://github.com/owner/repo/tree/branch_or_tag

    Return (owner, repo, branch_or_tag_or_None). If no “/tree/...” suffix, branch is None.
    """
    # Strip trailing “.git”
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]

    # Extract the path after “github.com”
    if repo_url.startswith("git@github.com:"):
        _, path = repo_url.split(":", 1)
    elif repo_url.startswith("https://github.com/"):
        path = repo_url[len("https://github.com/"):]
    elif repo_url.startswith("http://github.com/"):
        path = repo_url[len("http://github.com/"):]
    else:
        raise ValueError(f"Unsupported GitHub URL format: {repo_url}")

    path = path.rstrip("/")
    parts = path.split("/")

    # Case: "owner/repo"
    if len(parts) == 2:
        owner, repo = parts
        return owner, repo, None

    # Case: "owner/repo/tree/branch"
    if len(parts) == 4 and parts[2] == "tree":
        owner, repo, _, branch = parts
        return owner, repo, branch

    # Otherwise, unsupported
    raise ValueError(f"URL must be github.com/owner/repo or github.com/owner/repo/tree/: {repo_url}")


def get_default_branch(owner: str, repo: str) -> str:
    url = f"{GITHUB_API}/repos/{owner}/{repo}"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return "main"  # fallback
    data = resp.json()
    return data.get("default_branch", "main")

def list_all_files(owner: str, repo: str, branch: str) -> list[str]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    return [entry["path"] for entry in data.get("tree", []) if entry["type"] == "blob"]

def get_file_last_commit_timestamp(owner: str, repo: str, path: str, branch: str) -> int:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
    params = {"path": path, "sha": branch, "per_page": 1}
    resp = requests.get(url, headers=HEADERS, params=params)
    if resp.status_code != 200:
        return 0
    commits = resp.json()
    if not commits:
        return 0
    date_str = commits[0]["commit"]["committer"]["date"]
    t_struct = time.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    return int(time.mktime(t_struct))

def fetch_raw_file_content(owner: str, repo: str, path: str, branch: str) -> str:
    raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    resp = requests.get(raw_url)
    resp.raise_for_status()
    return resp.text

# ──────────────────────────────────────────────────────────────────────────────
# 3) GenAI-detection on a single repo
# ──────────────────────────────────────────────────────────────────────────────

def run_genai_detection(owner: str, repo: str, branch: str | None) -> dict:
    """
    Perform GenAI‐in‐development detection on up to MAX_FILES most‐recent source files
    in (owner, repo, branch). If branch is None, fall back to default.
    On any GitHub‐API failure or no source files, return { "#Files":0, "#Human":0, "#AI":0, "estimation":"Low" }.
    """
    stub = { "#Files": 0, "#Human": 0, "#AI": 0, "estimation": "Low" }
    try:
        default_branch = branch if branch else get_default_branch(owner, repo)
        file_paths     = list_all_files(owner, repo, default_branch)  # may raise HTTPError

        # Filter for source‐file extensions
        candidates = [p for p in file_paths if p.lower().endswith(SOURCE_EXTS)]
        if not candidates:
            return stub

        # Get last‐commit timestamps, skip any zero‐timestamps
        file_dates = []
        for path in candidates:
            ts = get_file_last_commit_timestamp(owner, repo, path, default_branch)
            if ts > 0:
                file_dates.append((path, ts))
        if not file_dates:
            return stub

        # Sort descending by timestamp, keep top MAX_FILES
        file_dates.sort(key=lambda x: x[1], reverse=True)
        top_files = file_dates[:MAX_FILES]

        ai_count = human_count = 0
        for rel_path, _ in top_files:
            try:
                content = fetch_raw_file_content(owner, repo, rel_path, default_branch)
                is_ai   = check_genai_file_content(content)
                if is_ai:
                    ai_count += 1
                else:
                    human_count += 1
            except requests.HTTPError:
                # skip if raw fetch fails
                continue
            except Exception:
                # any other error analyzing ⇒ count as human
                human_count += 1

        total = ai_count + human_count
        if total == 0:
            return stub

        ratio = ai_count / total
        if ratio > EST_HIGH_THRESHOLD:
            estimation = "High"
        elif ratio > EST_MEDIUM_THRESHOLD:
            estimation = "Medium"
        else:
            estimation = "Low"

        return {
            "#Files": total,
            "#Human": human_count,
            "#AI": ai_count,
            "estimation": estimation
        }

    except (requests.HTTPError, ValueError):
        return stub
    except Exception:
        return stub
# ──────────────────────────────────────────────────────────────────────────────
# 4) SE estimation helpers
# ──────────────────────────────────────────────────────────────────────────────

def on_rm_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def count_sloc_cloc_with_cloc(path: str) -> tuple[int, int]:
    """
    Runs `cloc <path> --json --quiet` locally (requires cloc installed).
    Returns (total_sloc, total_cloc).
    """
    try:
        out = subprocess.check_output(
            ["cloc", path, "--json", "--quiet"],
            stderr=subprocess.DEVNULL
        )
        data = json.loads(out.decode("utf-8"))
    except Exception:
        return 0, 0

    if "SUM" in data:
        return data["SUM"].get("code", 0), data["SUM"].get("comment", 0)
    total_sloc = total_cloc = 0
    for lang, stats in data.items():
        if lang == "header":
            continue
        total_sloc += stats.get("code", 0)
        total_cloc += stats.get("comment", 0)
    return total_sloc, total_cloc

def is_test_file_by_content(path: str) -> bool:
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
    lower_rel  = rel_path.replace(os.sep, "/").lower()
    lower_fname = fname.lower()
    if re.search(r"(^|/)(tests?)(/|$)", lower_rel):
        return True
    if lower_fname.startswith("test_"):
        return True
    if re.search(r"test\.(py|java|js|c|cpp|h|cs|go|sh|rb)$", lower_fname):
        return True
    return False

def traverse_and_count_tests_exact(repo_path: str) -> int:
    test_file_paths = []
    exts = {'.py','.java','.js','.c','.cpp','.h','.cs','.go','.sh','.rb'}
    for root, _, files in os.walk(repo_path):
        if ".git" in root.split(os.sep):
            continue
        for fname in files:
            _, ext = os.path.splitext(fname.lower())
            if ext not in exts:
                continue
            full_path = os.path.join(root, fname)
            rel_path  = os.path.relpath(full_path, repo_path)
            if is_test_file_by_content(full_path) or is_test_file_by_name_or_dir(repo_path, rel_path, fname):
                test_file_paths.append(rel_path)

    if not test_file_paths:
        return 0

    list_txt = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    try:
        for rel in test_file_paths:
            list_txt.write(f"{os.path.join(repo_path, rel)}\n")
        list_txt.flush()
    finally:
        list_txt.close()

    try:
        out = subprocess.check_output(
            ["cloc", f"--list-file={list_txt.name}", "--json", "--quiet"],
            stderr=subprocess.DEVNULL
        )
        data = json.loads(out.decode("utf-8"))
    except Exception:
        os.unlink(list_txt.name)
        return 0

    os.unlink(list_txt.name)
    if "SUM" in data:
        return data["SUM"].get("code", 0)
    total = 0
    for lang, stats in data.items():
        if lang == "header":
            continue
        total += stats.get("code", 0)
    return total

def fetch_all_commits(owner: str, repo: str) -> list[dict]:
    commits_data = []
    page = 1
    per_page = 100
    while True:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
        params = {"per_page": per_page, "page": page}
        resp = requests.get(url, headers=HEADERS, params=params)
        resp.raise_for_status()
        arr = resp.json()
        if not isinstance(arr, list) or len(arr) == 0:
            break
        for item in arr:
            date_str = item["commit"]["author"]["date"]
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            name = item["commit"]["author"].get("name", "").strip()
            email = item["commit"]["author"].get("email", "").strip()
            author = f"{name} <{email}>"
            commits_data.append({"date": dt, "author": author})
        page += 1
    return commits_data

def compute_git_history_metrics_via_api(owner: str, repo: str) -> tuple[int, datetime, datetime, dict]:
    commits = fetch_all_commits(owner, repo)
    total_commits = len(commits)
    if total_commits == 0:
        now = datetime.utcnow()
        return 0, now, now, {}
    dates = [c["date"] for c in commits]
    first_date = min(dates)
    last_date  = max(dates)
    author_counts = {}
    for c in commits:
        author_counts[c["author"]] = author_counts.get(c["author"], 0) + 1
    return total_commits, first_date, last_date, author_counts

def compute_core_contributors(author_counts: dict) -> int:
    total = sum(author_counts.values())
    threshold = 0.8 * total
    cum = 0
    core = 0
    for cnt in sorted(author_counts.values(), reverse=True):
        cum += cnt
        core += 1
        if cum >= threshold:
            break
    return core

def compute_duration_months(start_dt: datetime, end_dt: datetime) -> float:
    delta = relativedelta(end_dt, start_dt)
    months = delta.years * 12 + delta.months + delta.days / 30.0
    return max(months, 1/30)

def label_continuous(value: float, threshold: float) -> str:
    half = 0.5 * threshold
    if value < half:
        return "low"
    elif value < threshold:
        return "medium"
    else:
        return "high"

def label_binary(flag: int) -> str:
    return "high" if flag >= 1 else "low"

def github_headers() -> dict:
    return HEADERS.copy()

def fetch_issue_metrics(owner: str, repo: str) -> tuple[int, int, int]:
    headers = github_headers()
    session = requests.Session()
    session.headers.update(headers)
    base = f"{GITHUB_API}/repos/{owner}/{repo}"
    total_issues = 0
    closed_issues = 0
    total_comments = 0
    page = 1
    per_page = 100

    # Issues (state=all), skip PRs
    while True:
        resp = session.get(
            f"{base}/issues",
            params={"state": "all", "per_page": per_page, "page": page}
        )
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

    # Issue comments
    page = 1
    while True:
        resp = session.get(f"{base}/issues/comments", params={"per_page": per_page, "page": page})
        arr = resp.json()
        if not arr or (isinstance(arr, dict) and arr.get("message")):
            break
        total_comments += len(arr)
        page += 1

    return total_issues, closed_issues, total_comments

def detect_ci_flag(owner: str, repo: str) -> int:
    patterns = [
        ".travis.yml",
        "appveyor.yml",
        ".github/workflows",
        "circleci/config.yml",
        "azure-pipelines.yml",
        ".gitlab-ci.yml"
    ]
    for p in patterns:
        resp = requests.get(f"{GITHUB_API}/repos/{owner}/{repo}/contents/{p}", headers=HEADERS)
        if resp.status_code == 200:
            return 1
    return 0

def detect_license_flag(owner: str, repo: str, local_clone_path: str = None) -> int:
    resp = requests.get(f"{GITHUB_API}/repos/{owner}/{repo}/license", headers=HEADERS)
    if resp.status_code == 200:
        return 1
    if local_clone_path:
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

def strip_dot_git_suffix(s: str) -> str:
    return s[:-4] if s.lower().endswith(".git") else s

def download_and_unpack_tarball(owner: str, repo: str, branch: str) -> tuple[str, str]:
    headers = github_headers()
    url = f"{GITHUB_API}/repos/{owner}/{repo}/tarball/{branch}"
    resp = requests.get(url, headers=headers, stream=True)
    resp.raise_for_status()
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
        shutil.rmtree(tmp_dir, onerror=on_rm_error)
        os.unlink(tmp_tar.name)
        raise RuntimeError(f"Expected exactly one top‐level directory in {tmp_dir}, got {entries!r}")
    top_folder = os.path.join(tmp_dir, entries[0])
    os.unlink(tmp_tar.name)
    return tmp_dir, top_folder

def run_se_estimation(owner: str, repo: str, branch: str | None) -> dict:
    """
    Compute SE‐estimation for (owner, repo, branch). If branch is None, use default.
    Returns:
      {
        "dimensions": { community/ci/documentation/history/issues/license/unittest: { actual, estimation } },
        "raw": { total_sloc, total_cloc, test_sloc, total_commits, first_commit_date,
                 last_commit_date, duration_months, issue_events }
      }
    On any tarball‐download or cloc error, returns zeros/“low” across all fields.
    """
    stub_dimensions = {
        "community":     {"actual": 0,   "estimation": "low"},
        "ci":            {"actual": 0,   "estimation": "low"},
        "documentation": {"actual": 0.0, "estimation": "low"},
        "history":       {"actual": 0.0, "estimation": "low"},
        "issues":        {"actual": 0.0, "estimation": "low"},
        "license":       {"actual": 0,   "estimation": "low"},
        "unittest":      {"actual": 0.0, "estimation": "low"},
    }
    stub_raw = {
        "total_sloc":        0,
        "total_cloc":        0,
        "test_sloc":         0,
        "total_commits":     0,
        "first_commit_date": None,
        "last_commit_date":  None,
        "duration_months":   0.0,
        "issue_events":      0
    }
    try:
        default_branch = branch if branch else get_default_branch(owner, repo)
        tmp_root, clone_dir = download_and_unpack_tarball(owner, repo, default_branch)

        try:
            # (a) SLOC/CLOC via cloc
            total_sloc, total_cloc = count_sloc_cloc_with_cloc(clone_dir)
            comment_ratio = total_cloc / (total_sloc + total_cloc) if (total_sloc + total_cloc) > 0 else 0.0

            # (b) Test SLOC
            sloc_tests  = traverse_and_count_tests_exact(clone_dir)
            test_ratio  = sloc_tests / total_sloc if total_sloc > 0 else 0.0

            # (c) Git history metrics
            total_commits, first_date, last_date, author_counts = compute_git_history_metrics_via_api(owner, repo)
            core_contribs   = compute_core_contributors(author_counts)
            duration_months = compute_duration_months(first_date, last_date)
            commit_freq     = total_commits / duration_months if duration_months > 0 else 0.0

            # (d) Issue metrics
            total_issues, closed_issues, total_issue_comments = fetch_issue_metrics(owner, repo)
            total_issue_events = total_issues + closed_issues + total_issue_comments
            issue_freq        = total_issue_events / duration_months if duration_months > 0 else 0.0

            # (e) CI & License flags
            ci_flag      = detect_ci_flag(owner, repo)
            license_flag = detect_license_flag(owner, repo, local_clone_path=clone_dir)

            # Label dimensions
            labels = {}
            # 1) Community
            c_val = float(core_contribs)
            c_th  = ORG_THRESHOLDS["community"]
            labels["community"] = {"value": c_val, "estimation": label_continuous(c_val, c_th)}
            # 2) CI
            ci_val = float(ci_flag)
            labels["ci"] = {"value": ci_val, "estimation": label_binary(ci_flag)}
            # 3) Documentation
            d_val = comment_ratio
            d_th  = ORG_THRESHOLDS["documentation"]
            labels["documentation"] = {"value": d_val, "estimation": label_continuous(d_val, d_th)}
            # 4) History
            h_val = commit_freq
            h_th  = ORG_THRESHOLDS["history"]
            labels["history"] = {"value": h_val, "estimation": label_continuous(h_val, h_th)}
            # 5) Issues
            i_val = issue_freq
            i_th  = ORG_THRESHOLDS["issues"]
            labels["issues"] = {"value": i_val, "estimation": label_continuous(i_val, i_th)}
            # 6) License
            l_val = float(license_flag)
            labels["license"] = {"value": l_val, "estimation": label_binary(license_flag)}
            # 7) Unit Testing
            u_val = test_ratio
            u_th  = ORG_THRESHOLDS["unittest"]
            labels["unittest"] = {"value": u_val, "estimation": label_continuous(u_val, u_th)}

            output = {
                "dimensions": {
                    "community":     {"actual": labels["community"]["value"],     "estimation": labels["community"]["estimation"]},
                    "ci":            {"actual": labels["ci"]["value"],            "estimation": labels["ci"]["estimation"]},
                    "documentation": {"actual": labels["documentation"]["value"], "estimation": labels["documentation"]["estimation"]},
                    "history":       {"actual": labels["history"]["value"],       "estimation": labels["history"]["estimation"]},
                    "issues":        {"actual": labels["issues"]["value"],        "estimation": labels["issues"]["estimation"]},
                    "license":       {"actual": labels["license"]["value"],       "estimation": labels["license"]["estimation"]},
                    "unittest":      {"actual": labels["unittest"]["value"],      "estimation": labels["unittest"]["estimation"]},
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
        finally:
            shutil.rmtree(tmp_root, onerror=on_rm_error)

        return output

    except Exception:
        return {
            "dimensions": stub_dimensions,
            "raw": stub_raw
        }

# ──────────────────────────────────────────────────────────────────────────────
# 5) FAIRness estimation helpers
# ──────────────────────────────────────────────────────────────────────────────

def assess_fairness(record: dict) -> dict:
    results = {}
    metadata = record.get("metadata", {})

    # F1: DOI exists?
    results["doi_present"] = "yes" if metadata.get("doi") else "no"

    # F1.1: Subcomponent identifiers → unknown (no direct field)
    results["subcomponent_identifiers"] = "unknown"

    # F1.2: Version identifiers?
    relations = metadata.get("relations", {}).get("version", [])
    results["version_identifiers"] = "yes" if relations else "no"

    # F2: Rich metadata?
    missing_f2 = [f for f in FAIR_REQUIRED_F2_FIELDS if not metadata.get(f)]
    results["rich_metadata_present"] = "yes" if not missing_f2 else "no"

    # F3: Metadata includes its own DOI?
    results["metadata_includes_doi"] = "yes" if (metadata.get("doi") and record.get("doi")) else "no"

    # F4: Metadata accessible (we have JSON, so “yes”)
    results["metadata_accessible"] = "yes" if metadata else "no"

    # A1: Software retrievable by identifier? (DOI link + files exist)
    links = record.get("links", {})
    files = record.get("files", [])
    results["software_retrievable"] = "yes" if links.get("doi") and files else "no"

    # A1.2: Authentication required? → "not applicable" if access_right is “open”, else “unknown”
    results["authentication_required"] = "not applicable" if metadata.get("access_right") == "open" else "unknown"

    # A2: Metadata persistency
    results["metadata_persistent"] = "yes" if metadata else "no"

    # I1: Standard data formats? → unknown
    results["uses_standard_data_formats"] = "unknown"

    # I2: Qualified references to external objects
    references = metadata.get("references", []) or []
    qualified_refs = [r for r in references if isinstance(r, str) and r.startswith("https://doi.org")]
    results["qualified_references_to_objects"] = "yes" if qualified_refs else "no"

    # R1.1: Clear license present?
    license_field = metadata.get("license", {})
    results["clear_license"] = "yes" if license_field.get("id") else "no"

    # R1.2: Provenance via ORCID
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

    # R2: Qualified refs to other software → unknown
    results["qualified_references_to_software"] = "unknown"

    # R3: CI/community standards → unknown
    results["ci_or_community_standards"] = "unknown"

    return results

def extract_git_url(record: dict) -> str | None:
    metadata    = record.get("metadata", {})
    notes       = metadata.get("notes", "") or ""
    description = metadata.get("description", "") or ""
    text = notes + " " + description
    match = re.search(r"https?://(?:github\.com|gitlab\.com)/[\w\-/]+", text)
    return match.group(0) if match else None

def _on_rm_error_clone(func, path, exc_info):
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWRITE)
        func(path)
    else:
        raise

def clone_and_inspect_repo(repo_url: str, temp_dir: str = "temp_repo") -> dict:
    results = {"license_file": False, "ci_config": False, "dependency_files": []}
    if not repo_url:
        return results

    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, onerror=_on_rm_error_clone)

    try:
        subprocess.run(
            ["git", "clone", repo_url, temp_dir],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return results

    # LICENSE file?
    if os.path.isfile(os.path.join(temp_dir, "LICENSE")) or os.path.isfile(os.path.join(temp_dir, "LICENSE.txt")):
        results["license_file"] = True

    # CI config?
    if (os.path.isdir(os.path.join(temp_dir, ".github", "workflows")) 
        or os.path.isfile(os.path.join(temp_dir, ".gitlab-ci.yml"))):
        results["ci_config"] = True

    # Dependency files
    for fname in ["requirements.txt", "setup.py", "environment.yml", "pyproject.toml"]:
        if os.path.isfile(os.path.join(temp_dir, fname)):
            results["dependency_files"].append(fname)

    shutil.rmtree(temp_dir, onerror=_on_rm_error_clone)
    return results

def assess_fairness_extended(record: dict) -> dict:
    sub_assess = assess_fairness(record)
    git_url    = extract_git_url(record)
    if git_url:
        repo_inspection = clone_and_inspect_repo(git_url)
    else:
        repo_inspection = {"license_file": False, "ci_config": False, "dependency_files": []}

    # Adjust sub‐principles based on repo inspection
    if sub_assess.get("clear_license") == "no" and repo_inspection.get("license_file"):
        sub_assess["clear_license"] = "partial"
    if repo_inspection.get("ci_config"):
        sub_assess["ci_or_community_standards"] = "yes"
    elif sub_assess.get("ci_or_community_standards") == "unknown":
        sub_assess["ci_or_community_standards"] = "no"
    if repo_inspection.get("dependency_files"):
        sub_assess["qualified_references_to_software"] = "yes"
    elif sub_assess.get("qualified_references_to_software") == "unknown":
        sub_assess["qualified_references_to_software"] = "no"

    return {
        "subprinciple_assessment": sub_assess,
        "git_url": git_url or "none found",
        "repo_inspection": repo_inspection
    }

def score_value(val: str) -> float:
    return {"yes": 1.0, "partial": 0.5}.get(val, 0.0)

def categorize_principle(scores: list[float], thresholds: tuple[float, float] = (0.75, 0.5)) -> str:
    if not scores:
        return "Low"
    avg = sum(scores) / len(scores)
    if avg >= thresholds[0]:
        return "High"
    elif avg >= thresholds[1]:
        return "Medium"
    else:
        return "Low"

def estimate_fairness(subprinciples: dict) -> dict:
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
    principle_scores     = []
    for principle, keys in groups.items():
        numeric_scores = [score_value(subprinciples.get(k, "unknown")) for k in keys]
        category = categorize_principle(numeric_scores)
        principle_categories[principle] = category
        avg_num = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
        principle_scores.append(avg_num)

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

def run_fairness_estimation(record: dict) -> dict:
    extended = assess_fairness_extended(record)
    subpr   = extended["subprinciple_assessment"]
    overall = estimate_fairness(subpr)
    return {
        "subprinciple_assessment": subpr,
        "principle_categories": overall["principle_categories"],
        "overall_fairness": overall["overall_fairness"],
        "git_url": extended["git_url"],
        "repo_inspection": extended["repo_inspection"]
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6) Main loop: open each Zenodo JSON, run all three analyses, merge results
# ──────────────────────────────────────────────────────────────────────────────

def process_zenodo_file(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        record = json.load(f)

    # 6.1) FAIRness estimation always runs (metadata only + minimal repo‐clone)
    fairness_json = run_fairness_estimation(record)

    # 6.2) If a Git URL is present, run GenAI + SE; else, stub them as empty/zero
    git_url = fairness_json.get("git_url")
    if git_url and git_url != "none found":
        try:
            owner, repo = parse_github_url(git_url)
        except ValueError:
            owner = repo = None
        if owner and repo:
            genai_json = run_genai_detection(owner, repo)
            se_json    = run_se_estimation(owner, repo)
        else:
            genai_json = {"#Files": 0, "#Human": 0, "#AI": 0, "estimation": "Low"}
            se_json    = {
                "dimensions": {
                    "community":     {"actual": 0, "estimation": "low"},
                    "ci":            {"actual": 0, "estimation": "low"},
                    "documentation": {"actual": 0.0, "estimation": "low"},
                    "history":       {"actual": 0.0, "estimation": "low"},
                    "issues":        {"actual": 0.0, "estimation": "low"},
                    "license":       {"actual": 0, "estimation": "low"},
                    "unittest":      {"actual": 0.0, "estimation": "low"},
                },
                "raw": {
                    "total_sloc":       0,
                    "total_cloc":       0,
                    "test_sloc":        0,
                    "total_commits":    0,
                    "first_commit_date": None,
                    "last_commit_date":  None,
                    "duration_months":   0.0,
                    "issue_events":      0
                }
            }
    else:
        genai_json = {"#Files": 0, "#Human": 0, "#AI": 0, "estimation": "Low"}
        se_json    = {
            "dimensions": {
                "community":     {"actual": 0, "estimation": "low"},
                "ci":            {"actual": 0, "estimation": "low"},
                "documentation": {"actual": 0.0, "estimation": "low"},
                "history":       {"actual": 0.0, "estimation": "low"},
                "issues":        {"actual": 0.0, "estimation": "low"},
                "license":       {"actual": 0, "estimation": "low"},
                "unittest":      {"actual": 0.0, "estimation": "low"},
            },
            "raw": {
                "total_sloc":       0,
                "total_cloc":       0,
                "test_sloc":        0,
                "total_commits":    0,
                "first_commit_date": None,
                "last_commit_date":  None,
                "duration_months":   0.0,
                "issue_events":      0
            }
        }

    # 6.3) Merge into record under "repo-profile"
    record["repo-profile"] = {
        "genai_detection":    genai_json,
        "se_estimation":      se_json,
        "fairness_estimation": fairness_json
    }

    # 6.4) Write back (overwrites the original JSON)
    with open(json_path, "w", encoding="utf-8") as out_f:
        json.dump(record, out_f, indent=2)

    print(f"Processed and updated: {json_path}")

def main():

    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory.")
        sys.exit(1)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".json"):
            continue
        json_path = os.path.join(input_dir, fname)

        # 1) Load the JSON and check if genAI already ran correctly
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except Exception:
            # Couldn’t even load it—skip
            print(f"Skipping (cannot load JSON): {json_path}")
            continue

        repo_profile = record.get("repo-profile")
        if repo_profile:
            genai = repo_profile.get("genai_detection", {})
            # If "#Files" > 0, GenAI detection already ran successfully—skip this file
            if isinstance(genai.get("#Files"), int) and genai["#Files"] > 0:
                print(f"Skipping (genAI already succeeded): {json_path}")
                continue

        # 2) Otherwise, re‐run the full pipeline on this JSON
        try:
            process_zenodo_file(json_path)
        except Exception as e:
            print(f"Error processing {json_path}: {e}")

if __name__ == "__main__":
    main()
