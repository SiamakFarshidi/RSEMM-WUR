#!/usr/bin/env python3

import os
import re
import shutil
import stat
import sys
import tempfile
import requests
import tarfile
import json
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ─────────────────────────────────────────────────────────────────────────────
# 0) Hardcoded GitHub token (must have at least "public_repo" scope)
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_TOKEN = "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"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Recursively remove a directory, making files writable if needed
# ─────────────────────────────────────────────────────────────────────────────

def on_rm_error(func, path, exc_info):
    """
    Handler for shutil.rmtree to change file permissions and retry.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# 1) “cloc in Python”: count source vs. comment lines by extension
# ─────────────────────────────────────────────────────────────────────────────

LINE_COMMENT_EXTENSIONS = {
    '.py': '#',
    '.sh': '#',
    '.rb': '#',
    '.m' : '%',
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

def analyze_source_file(path):
    """
    Read `path` line by line. Return (code_count, comment_count).
    """
    _, ext = os.path.splitext(path.lower())
    code_ct = 0
    comment_ct = 0

    if ext in LINE_COMMENT_EXTENSIONS:
        marker = LINE_COMMENT_EXTENSIONS[ext]
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                line = raw.rstrip('\n')
                stripped = line.lstrip()
                if not stripped:
                    continue
                if stripped.startswith(marker):
                    comment_ct += 1
                else:
                    code_ct += 1
        return code_ct, comment_ct

    elif ext in BLOCK_COMMENT_EXTENSIONS:
        block_start, block_end, line_marker = BLOCK_COMMENT_EXTENSIONS[ext]
        in_block = False
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
                    comment_ct += 1
                    if block_end not in stripped:
                        in_block = True
                else:
                    code_ct += 1

        return code_ct, comment_ct

    else:
        return 0, 0

def count_sloc_cloc_python(repo_path):
    """
    Walk `repo_path` recursively. Return (total_sloc, total_cloc).
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

# ─────────────────────────────────────────────────────────────────────────────
# 2) Detect test files and count their SLOC
# ─────────────────────────────────────────────────────────────────────────────

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

def is_test_file_by_content(path):
    """
    Returns True if `path` contains any of the TEST_FRAMEWORK_PATTERNS.
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

def is_test_file_by_name_or_dir(repo_path, rel_path, fname):
    """
    Returns True if `rel_path` or `fname` implies a test file.
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

def traverse_and_count_tests_exact_python(repo_path):
    """
    Identify “test files” exactly as in Sect 4.7, then sum their SLOC.
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

# ─────────────────────────────────────────────────────────────────────────────
# 3) GitHub API calls: commits, issues, CI, license
# ─────────────────────────────────────────────────────────────────────────────

def github_headers(token=None):
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers

def github_get(url, params=None, stream=False):
    """
    Wrap requests.get to handle rate limiting and network errors.
    """
    headers = github_headers(GITHUB_TOKEN)
    while True:
        try:
            resp = requests.get(url, headers=headers, params=params, stream=stream)
        except requests.exceptions.RequestException:
            print("Network error, retrying in 60 seconds…")
            time.sleep(60)
            continue

        # Handle 409 explicitly here — BEFORE raise_for_status()
        if resp.status_code == 409:
            print(f"[WARN] GitHub API returned 409 Conflict for URL {url}")
            return resp  # allow caller to handle this

        if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            sleep_secs = max(reset_ts - int(time.time()), 60)
            print(f"Rate limit reached, sleeping for {sleep_secs} seconds…")
            time.sleep(sleep_secs)
            continue

        resp.raise_for_status()
        return resp

def fetch_all_commits(owner, repo, token=None):
    """
    Return a list of { "date": datetime, "author": "Name <email>" } for each commit.
    """
    commits_data = []
    page = 1
    per_page = 100

    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params = {"per_page": per_page, "page": page}

        try:
            resp = github_get(url, params=params)
            print(f"[DEBUG] fetch_all_commits(): response status_code = {resp.status_code}")
            if resp.status_code == 409:
                print(f"[WARN] GitHub API returned 409 Conflict — no commits available.")
                return []
        except Exception as e:
            print(f"[ERROR] fetch_all_commits(): failed on page {page}: {e}")
            raise        
        
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

def compute_git_history_metrics_via_api(owner, repo, token=None):

    """
    Using GitHub’s commits API, compute:
      - total_commits (int)
      - first_commit_date (datetime)
      - last_commit_date  (datetime)
      - author_counts     (dict: "Name <email>" -> commit_count)
    """
    commits = fetch_all_commits(owner, repo, token)
    print(f"[DEBUG] compute_git_history_metrics_via_api: fetched {len(commits)} commits.")

    total_commits = len(commits)
    if total_commits == 0:
        first_date = last_date = datetime.utcnow()
        author_counts = {}
    else:
        dates = [c["date"] for c in commits]
        first_date = min(dates)
        last_date = max(dates)
        author_counts = {}
        for c in commits:
            author_counts[c["author"]] = author_counts.get(c["author"], 0) + 1
    return total_commits, first_date, last_date, author_counts

def compute_core_contributors(author_counts):
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

def compute_duration_months(start_dt, end_dt):
    """
    Compute months (float) between start_dt and end_dt.
    If zero, return a small positive number.
    """
    delta = relativedelta(end_dt, start_dt)
    months = delta.years * 12 + delta.months + delta.days / 30.0
    return max(months, 1/30)

def fetch_issue_metrics(owner, repo, token=None):
    """
    Count:
      - total_issues   (open + closed, excluding PRs)
      - closed_issues  (subset)
      - total_comments (on issues)
    Return (total_issues, closed_issues, total_comments).
    """
    base = f"https://api.github.com/repos/{owner}/{repo}"
    total_issues = closed_issues = total_comments = 0
    page = 1
    per_page = 100

    while True:
        params = {"state": "all", "per_page": per_page, "page": page}
        resp = github_get(f"{base}/issues", params=params)
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

    page = 1
    while True:
        params = {"per_page": per_page, "page": page}
        resp = github_get(f"{base}/issues/comments", params=params)
        arr = resp.json()
        if not arr or (isinstance(arr, dict) and arr.get("message")):
            break
        total_comments += len(arr)
        page += 1

    return total_issues, closed_issues, total_comments

def detect_ci_flag(owner, repo, token=None):
    """
    Check if any known CI config paths exist in the repo.
    Return 1 if found, otherwise 0.
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
        try:
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{p}"
            resp = github_get(url)  # always assigns resp or raises
            if resp.status_code == 200:
                return 1
        except Exception as e:
            # Any HTTPError or network error → treat as “no CI file found” for this pattern
            # and continue to next pattern. Do not let resp be undefined.
            # Logging limited to keep output concise.
            pass

    return 0

def detect_license_flag(owner, repo, token=None, local_clone_path=None):
    """
    1) Try GitHub’s /license endpoint.
    2) If 404, scan for known license‐keyword lines in any file (up to 30 lines).
    Return 1 if found, else 0.
    """
    try:
        url = f"https://api.github.com/repos/{owner}/{repo}/license"
        resp = github_get(url)
        if resp.status_code == 200:
            return 1
    except Exception:
        pass

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

# ─────────────────────────────────────────────────────────────────────────────
# 4) Organization‐trained thresholds
# ─────────────────────────────────────────────────────────────────────────────

ORG_THRESHOLDS = {
    "community":     2.0000,    # core contributors
    "ci":            1.0000,    # CI flag
    "documentation": 0.0018660, # comment_ratio
    "history":       2.0895,    # commits_per_month
    "issues":        0.022989,  # issue_events_per_month
    "license":       1.0000,    # license flag
    "unittest":      0.0010160  # test_ratio
}

def label_continuous(value, threshold):
    """
    Given a continuous metric value and its threshold:
      - “low”    if value < 0.5 * threshold
      - “medium” if 0.5*threshold ≤ value < threshold
      - “high”   if value ≥ threshold
    """
    half = 0.5 * threshold
    if value < half:
        return "low"
    elif value < threshold:
        return "medium"
    else:
        return "high"

def label_binary(flag):
    """
    For CI and license (0 or 1):
      0 → "low",  1 → "high"
    """
    return "high" if flag >= 1 else "low"

# ─────────────────────────────────────────────────────────────────────────────
# 5) Download & unpack the GitHub repo tarball (no `git clone`)
# ─────────────────────────────────────────────────────────────────────────────

def get_default_branch(owner: str, repo: str, token: str = None):
    """
    GET https://api.github.com/repos/{owner}/{repo} to find the default branch.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        resp = github_get(url)
        repo_info = resp.json()
        archived = repo_info.get("archived", False)
        if archived:
            print("   → Repository is archived — some metrics may be unavailable.")
        return repo_info.get("default_branch", "main")
    except Exception as e:
        print(f"   ERROR while fetching default branch: {e}. Defaulting to 'main'.")
        return "main"

def strip_dot_git_suffix(s: str) -> str:
    """
    If the string ends with ".git", strip it; else return unchanged.
    """
    return s[:-4] if s.lower().endswith(".git") else s

def download_and_unpack_tarball(owner: str, repo: str, branch: str = "main", token: str = None):
    """
    Download https://api.github.com/repos/{owner}/{repo}/tarball/{branch},
    unpack into a temp dir, and return (tmp_root, top_folder_path).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/tarball/{branch}"
    #print(f"   → Downloading tarball from: {url}")
    try:
        resp = github_get(url, stream=True)
    except Exception as e:
        raise RuntimeError(f"Could not download tarball: {e}")

    tmp_tar = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    for chunk in resp.iter_content(chunk_size=1 << 20):
        tmp_tar.write(chunk)
    tmp_tar.flush()
    tmp_tar.close()

    tmp_dir = tempfile.mkdtemp(prefix="gh_metrics_")
    try:
        with tarfile.open(tmp_tar.name, "r:gz") as tf:
            tf.extractall(path=tmp_dir)
    except Exception as e:
        os.unlink(tmp_tar.name)
        shutil.rmtree(tmp_dir, onerror=on_rm_error)
        raise RuntimeError(f"Failed to extract tarball: {e}")

    os.unlink(tmp_tar.name)
    entries = os.listdir(tmp_dir)
    if len(entries) != 1:
        # Clean up and error out if unexpected structure
        shutil.rmtree(tmp_dir, onerror=on_rm_error)
        raise RuntimeError(f"Expected exactly one top-level directory in {tmp_dir}, got {entries!r}")
    top_folder = os.path.join(tmp_dir, entries[0])
    return tmp_dir, top_folder

# ─────────────────────────────────────────────────────────────────────────────
# 6) Utility: extract GitHub repo URL from a JSON record (handles “/tree/…”)
# ─────────────────────────────────────────────────────────────────────────────

def find_github_repo_url(json_data):
    """
    Search entire JSON for the first GitHub repo URL of the form:
    https://github.com/owner/repo (possibly followed by /tree/…)
    """
    text = json.dumps(json_data)
    # Matches “https://github.com/owner/repo” (ignores any trailing “/tree/…”)
    m = re.search(r"https?://github\.com/([\w\-\._]+)/([\w\-\._]+)", text)
    if m:
        owner = m.group(1)
        repo  = m.group(2)
        return f"https://github.com/{owner}/{repo}"
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 7) Build default (empty) output when no repo is found or an error occurs
# ─────────────────────────────────────────────────────────────────────────────

def build_default_output(repo_url):
    """
    Build a default metrics structure with zeros and 'low' estimations.
    """
    dimensions = {}
    raw = {}
    dimensions["community"]     = {"actual": 0.0, "estimation": label_continuous(0.0, ORG_THRESHOLDS["community"])}
    dimensions["ci"]            = {"actual": 0.0, "estimation": label_binary(0)}
    dimensions["documentation"] = {"actual": 0.0, "estimation": label_continuous(0.0, ORG_THRESHOLDS["documentation"])}
    dimensions["history"]       = {"actual": 0.0, "estimation": label_continuous(0.0, ORG_THRESHOLDS["history"])}
    dimensions["issues"]        = {"actual": 0.0, "estimation": label_continuous(0.0, ORG_THRESHOLDS["issues"])}
    dimensions["license"]       = {"actual": 0.0, "estimation": label_binary(0)}
    dimensions["unittest"]      = {"actual": 0.0, "estimation": label_continuous(0.0, ORG_THRESHOLDS["unittest"])}

    raw["total_sloc"]         = 0
    raw["total_cloc"]         = 0
    raw["test_sloc"]          = 0
    raw["total_commits"]      = 0
    raw["first_commit_date"]  = ""
    raw["last_commit_date"]   = ""
    raw["duration_months"]    = 0.0
    raw["issue_events"]       = 0

    return {
        "repo": repo_url,
        "dimensions": dimensions,
        "raw": raw
    }

# ─────────────────────────────────────────────────────────────────────────────
# 8) Helper: “normalize” a GitHub URL by following any redirect from GitHub’s web interface
# ─────────────────────────────────────────────────────────────────────────────

def resolve_github_redirect(owner, repo_name):
    """
    Attempt a HEAD request to https://github.com/{owner}/{repo_name}
    without following redirects. If GitHub responds with 301/302 and a 'Location'
    header, parse that to extract the actual {owner}/{real_repo_name}.
    Otherwise, return (owner, repo_name) unchanged.
    """
    url = f"https://github.com/{owner}/{repo_name}"
    try:
        resp = requests.head(url, allow_redirects=False, timeout=10)
    except Exception:
        # Network problem; just return the original owner/repo_name
        return owner, repo_name

    if resp.status_code in (301, 302) and "Location" in resp.headers:
        loc = resp.headers["Location"]
        m = re.search(r"github\.com/([^/]+)/([^/]+)", loc)
        if m:
            return m.group(1), m.group(2)
    return owner, repo_name

# ─────────────────────────────────────────────────────────────────────────────
# 9) Main: iterate over JSON files, compute or default metrics, and update each JSON
# ─────────────────────────────────────────────────────────────────────────────

def getSEestimation(full_path):
    output={}

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"   ERROR: Could not load JSON {full_path}: {e}")
        return output

    # 1) Extract “https://github.com/owner/repo” (if it exists anywhere in JSON)
    repo_url = find_github_repo_url(json_data)
    #print(f"   Parsed repo_url from JSON: {repo_url!r}")

    if not repo_url:
        print("   → No GitHub URL found; using default (all-zero) output.")
        output = build_default_output(repo_url)
    else:
        # 2) Strip any trailing “.git”
        repo_url = repo_url.rstrip(".git")

        # 3) Parse owner/repo_name
        m = re.search(r"github\.com/([^/]+)/([^/]+)", repo_url)
        if not m:
            #print("   → Regex failed to parse owner/repo; using default (zeros).")
            output = build_default_output(repo_url)
        else:
            owner = m.group(1)
            repo_name = m.group(2)
            #print(f"   Initial owner/repo: {owner}/{repo_name}")

            # 4) Follow any GitHub redirect
            owner2, repo_name2 = resolve_github_redirect(owner, repo_name)
            if (owner2, repo_name2) != (owner, repo_name):
                #print(f"   → GitHub redirected: {owner}/{repo_name} → {owner2}/{repo_name2}")
                owner, repo_name = owner2, repo_name2
                repo_url = f"https://github.com/{owner}/{repo_name}"

            try:
                # 5) Discover default branch
                default_branch = get_default_branch(owner, repo_name, GITHUB_TOKEN)
                #print(f"   Default branch for {owner}/{repo_name} = '{default_branch}'")

                # 6) Download & unpack tarball
                tmp_root, clone_dir = download_and_unpack_tarball(
                    owner, repo_name, branch=default_branch, token=GITHUB_TOKEN
                )

                try:
                    # 7) Count SLOC/CLOC
                    #print("   → Counting SLOC/CLOC …")
                    total_sloc, total_cloc = count_sloc_cloc_python(clone_dir)
                    comment_ratio = (total_cloc / (total_sloc + total_cloc)) if (total_sloc + total_cloc) > 0 else 0.0

                    # 8) Count test SLOC
                    #print("   → Counting test SLOC …")
                    sloc_tests = traverse_and_count_tests_exact_python(clone_dir)
                    test_ratio = (sloc_tests / total_sloc) if total_sloc > 0 else 0.0

                    # 9) Git history metrics
                    #print("   → Gathering Git history metrics …")
                    total_commits, first_date, last_date, author_counts = compute_git_history_metrics_via_api(owner, repo_name, GITHUB_TOKEN)
                    core_contribs = compute_core_contributors(author_counts)
                    duration_months = compute_duration_months(first_date, last_date)
                    commit_frequency = (total_commits / duration_months) if duration_months > 0 else 0.0

                    # 10) Issue metrics
                    #print("   → Fetching issue metrics …")
                    total_issues, closed_issues, total_issue_comments = fetch_issue_metrics(owner, repo_name, GITHUB_TOKEN)
                    total_issue_events = total_issues + closed_issues + total_issue_comments
                    issue_frequency = (total_issue_events / duration_months) if duration_months > 0 else 0.0

                    # 11) CI & License flags
                    #print("   → Detecting CI …")
                    ci_flag = detect_ci_flag(owner, repo_name, GITHUB_TOKEN)

                    #print("   → Detecting license …")
                    license_flag = detect_license_flag(owner, repo_name, GITHUB_TOKEN, local_clone_path=clone_dir)

                    # 12) Label each dimension
                    labels = {}

                    c_val = float(core_contribs)
                    labels["community"] = {
                        "value": c_val,
                        "estimation": label_continuous(c_val, ORG_THRESHOLDS["community"])
                    }

                    ci_val = float(ci_flag)
                    labels["ci"] = {
                        "value": ci_val,
                        "estimation": label_binary(ci_flag)
                    }

                    d_val = comment_ratio
                    labels["documentation"] = {
                        "value": d_val,
                        "estimation": label_continuous(d_val, ORG_THRESHOLDS["documentation"])
                    }

                    h_val = commit_frequency
                    labels["history"] = {
                        "value": h_val,
                        "estimation": label_continuous(h_val, ORG_THRESHOLDS["history"])
                    }

                    i_val = issue_frequency
                    labels["issues"] = {
                        "value": i_val,
                        "estimation": label_continuous(i_val, ORG_THRESHOLDS["issues"])
                    }

                    l_val = float(license_flag)
                    labels["license"] = {
                        "value": l_val,
                        "estimation": label_binary(license_flag)
                    }

                    u_val = test_ratio
                    labels["unittest"] = {
                        "value": u_val,
                        "estimation": label_continuous(u_val, ORG_THRESHOLDS["unittest"])
                    }

                    # 13) Build JSON output
                    dimensions = {
                        "community":     {"actual": labels["community"]["value"],
                                            "estimation": labels["community"]["estimation"]},
                        "ci":            {"actual": labels["ci"]["value"],
                                            "estimation": labels["ci"]["estimation"]},
                        "documentation": {"actual": labels["documentation"]["value"],
                                            "estimation": labels["documentation"]["estimation"]},
                        "history":       {"actual": labels["history"]["value"],
                                            "estimation": labels["history"]["estimation"]},
                        "issues":        {"actual": labels["issues"]["value"],
                                            "estimation": labels["issues"]["estimation"]},
                        "license":       {"actual": labels["license"]["value"],
                                            "estimation": labels["license"]["estimation"]},
                        "unittest":      {"actual": labels["unittest"]["value"],
                                            "estimation": labels["unittest"]["estimation"]},
                    }

                    raw = {
                        "total_sloc":        total_sloc,
                        "total_cloc":        total_cloc,
                        "test_sloc":         sloc_tests,
                        "total_commits":     total_commits,
                        "first_commit_date": first_date.isoformat(),
                        "last_commit_date":  last_date.isoformat(),
                        "duration_months":   duration_months,
                        "issue_events":      total_issue_events,
                    }

                    output = {
                        "repo":       repo_url,
                        "dimensions": dimensions,
                        "raw":        raw
                    }

                finally:
                    # 14) Always clean up the tmp directory
                    shutil.rmtree(tmp_root, onerror=on_rm_error)

            except Exception:
                print("   → An error occurred during metrics collection; using default (zeros).")
                output = build_default_output(repo_url)
    # 15) Insert (or overwrite) “repo-profile” in JSON
    if "repo-profile" not in json_data:
        json_data["repo-profile"] = {}
    json_data["repo-profile"]["software-engineering-efforts"] = output

    # 16) Write updated JSON back to disk
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        size = os.path.getsize(full_path)
        #print(f"   WROTE updated JSON: {full_path} (size = {size} bytes)")
    except Exception as e:
        print(f"   ERROR: Failed to write updated JSON: {e}")
    return output


def main():
    # ─── Change this path to point at your Zenodo JSON directory
    input_dir = r"C:\Users\User\Downloads\RS-Zenodo\RS-Repo-Zenodo - v4"
    print(f"Working directory: {os.getcwd()}")
    print(f"Using input_dir: {input_dir}")

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".json"):
            continue

        full_path = os.path.join(input_dir, fname)
        print(f"\n→ Processing file: {fname}")
        print(f"   Full path: {full_path}")


        getSEestimation(full_path)

if __name__ == "__main__":
    main()
