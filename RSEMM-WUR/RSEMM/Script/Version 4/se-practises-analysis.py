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
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ─────────────────────────────────────────────────────────────────────────────
# 0) Hardcoded GitHub token
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_TOKEN =  "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"

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

# We’ll support the same extensions that cloc would for Munaiah et al.’s seven
# languages (plus shell, Ruby, etc.) and use simple per‐line logic:
#  - For Python, Shell, Ruby: lines starting with “#” (after whitespace) are comments.
#  - For C/C++/Java/JavaScript/C#/Go: 
#       • “//” lines count as comment‐only (unless there’s code before it).
#       • “/* … */” block comments are counted line by line as comment.
#       • Any line with code before a “//” or “/*” is counted as code.
#
# Blank lines are ignored (neither code nor comment).
#
# This matches cloc’s approach of “comment‐only” vs “code” lines.

# Define which extensions use which comment syntax:
LINE_COMMENT_EXTENSIONS = {
    # Python, Shell, Ruby
    '.py':  '#',
    '.sh':  '#',
    '.rb':  '#',
}
BLOCK_COMMENT_EXTENSIONS = {
    # C, C++, Java, JavaScript, C#, Go
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
    '.kt':   ('/*', '*/', '//'),  # Kotlin
    '.kts':  ('/*', '*/', '//'),
}

# Count for each source file: (code_lines, comment_lines)
def analyze_source_file(path):
    """
    Read `path` line by line. Return (code_count, comment_count).
    A “comment‐only” line is counted as comment if:
      - For extensions in LINE_COMMENT_EXTENSIONS: line.strip().startswith(comment_char)
      - For extensions in BLOCK_COMMENT_EXTENSIONS: 
          • if inside /* ... */ block → comment
          • elif stripped line starts with '//' → comment
          • elif line contains '/*' → start block comment, count as comment
      - Otherwise, if stripped line is nonblank → code. 
      - Blank lines (after strip) are ignored.
    """
    _, ext = os.path.splitext(path.lower())
    code_ct = 0
    comment_ct = 0

    # Which comment style to use?
    if ext in LINE_COMMENT_EXTENSIONS:
        line_marker = LINE_COMMENT_EXTENSIONS[ext]
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                line = raw.rstrip('\n')
                stripped = line.lstrip()
                if not stripped:
                    continue  # blank line
                if stripped.startswith(line_marker):
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
                    continue  # blank

                if in_block:
                    comment_ct += 1
                    if block_end in stripped:
                        # If the same line has code after */, count that as code?
                        # But for simplicity, we treat the entire line as comment if it started inside a block.
                        in_block = False
                    continue

                # Not already in a block comment:
                if stripped.startswith(line_marker):  # e.g. "//"
                    comment_ct += 1
                elif block_start in stripped:
                    # If line has code before /*, count as code. Else count as comment.
                    idx = stripped.find(block_start)
                    before = stripped[:idx].strip()
                    comment_ct += 1
                    if block_end not in stripped:
                        in_block = True
                else:
                    code_ct += 1

        return code_ct, comment_ct

    else:
        # Unrecognized extension → skip entirely
        return 0, 0

def count_sloc_cloc_python(repo_path):
    """
    Walk `repo_path` recursively. For each file whose extension is in 
    LINE_COMMENT_EXTENSIONS or BLOCK_COMMENT_EXTENSIONS, call analyze_source_file().
    Return (total_sloc, total_cloc).
    """
    total_code = 0
    total_comment = 0
    valid_exts = set(LINE_COMMENT_EXTENSIONS.keys()) | set(BLOCK_COMMENT_EXTENSIONS.keys())

    for root, _, files in os.walk(repo_path):
        # Skip .git folder if it somehow got in
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
# 2) Detect test files exactly as in the paper (Sect 4.7), but count with Python
# ─────────────────────────────────────────────────────────────────────────────

TEST_FRAMEWORK_PATTERNS = [
    r"(?i)\bimport\s+unittest\b",           # Python's built‐in unittest
    r"(?i)\bfrom\s+django\.test\b",          # Django tests
    r"(?i)\bimport\s+pytest\b",              # pytest
    r"(?i)\bimport\s+nose\b",                # nose
    r"(?i)\bimport\s+RSpec\b",               # RSpec (Ruby)
    r"(?i)\bimport\s+Minitest\b",            # Minitest (Ruby)
    r"(?i)\bimport\s+JUnit\b",               # JUnit (Java)
    r"(?i)\bimport\s+TestNG\b",              # TestNG (Java)
    r"(?i)\bimport\s+googletest\b",          # Google Test (C++)
    r"(?i)\b#import\s+<gtest/gtest\.h>",     # Google Test (Obj‐C/C++)
    r"(?i)\bimport\s+mocha\b",               # Mocha (JavaScript)
    r"(?i)\bimport\s+jest\b",                # Jest (JavaScript)
    r"(?i)\bimport\s+kotlin\.test\b",        # Kotlin
    r"(?i)\bimport\s+xunit\b",               # xUnit (C#)
    r"(?i)\b@test\b",                        # Generic @Test
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
    Returns True if `rel_path` or `fname` implies a test:
      1. any directory component “test” or “tests”
      2. filename starts with “test_”
      3. filename ends with “test.<ext>”
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
    Identify “test files” exactly as in Sect 4.7, then sum their code lines.
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
            # (1) Check content:
            if is_test_file_by_content(full):
                test_files.append(full)
                continue
            # (2) Fallback name/dir:
            if is_test_file_by_name_or_dir(repo_path, rel, fname):
                test_files.append(full)

    sloc_tests = 0
    for tf in test_files:
        code_ct, _ = analyze_source_file(tf)
        sloc_tests += code_ct

    return sloc_tests

# ─────────────────────────────────────────────────────────────────────────────
# 3) Git history metrics via GitHub API
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_commits(owner, repo, token=None):
    """
    Return a list of { "date": datetime, "author": "Name <email>" } for each commit.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    commits_data = []
    page = 1
    per_page = 100

    while True:
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params = {"per_page": per_page, "page": page}
        resp = requests.get(url, headers=headers, params=params)
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

def compute_git_history_metrics_via_api(owner, repo, token=None):
    """
    Using GitHub’s commits API, compute:
      - total_commits (int)
      - first_commit_date (datetime)
      - last_commit_date  (datetime)
      - author_counts     (dict: "Name <email>" -> commit_count)
    """
    commits = fetch_all_commits(owner, repo, token)
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

# ─────────────────────────────────────────────────────────────────────────────
# 4) GitHub API–based metrics (issues, CI, license)
# ─────────────────────────────────────────────────────────────────────────────

def github_headers(token=None):
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers

def fetch_issue_metrics(owner, repo, token=None):
    """
    Count:
      - total_issues   (open + closed, excluding PRs)
      - closed_issues  (subset)
      - total_comments (on issues)
    Return (total_issues, closed_issues, total_comments).
    """
    headers = github_headers(token)
    session = requests.Session()
    session.headers.update(headers)

    base = f"https://api.github.com/repos/{owner}/{repo}"
    total_issues = closed_issues = total_comments = 0
    page = 1
    per_page = 100

    while True:
        resp = session.get(f"{base}/issues", params={"state": "all", "per_page": per_page, "page": page})
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
        resp = session.get(f"{base}/issues/comments", params={"per_page": per_page, "page": page})
        arr = resp.json()
        if not arr or (isinstance(arr, dict) and arr.get("message")):
            break
        total_comments += len(arr)
        page += 1

    return total_issues, closed_issues, total_comments

def detect_ci_flag(owner, repo, token=None):
    """
    Check if any known CI config paths exist in the repo:
      - .travis.yml
      - appveyor.yml
      - .github/workflows
      - circleci/config.yml
      - azure-pipelines.yml
      - .gitlab-ci.yml
    Return 1 if found, otherwise 0.
    """
    headers = github_headers(token)
    patterns = [
        ".travis.yml",
        "appveyor.yml",
        ".github/workflows",
        "circleci/config.yml",
        "azure-pipelines.yml",
        ".gitlab-ci.yml"
    ]
    for p in patterns:
        resp = requests.get(f"https://api.github.com/repos/{owner}/{repo}/contents/{p}", headers=headers)
        if resp.status_code == 200:
            return 1
    return 0

def detect_license_flag(owner, repo, token=None, local_clone_path=None):
    """
    1) Try GitHub’s /license endpoint.
    2) If 404, scan for known license‐keyword lines in any file (up to 30 lines).
    Return 1 if found, else 0.
    """
    headers = github_headers(token)
    resp = requests.get(f"https://api.github.com/repos/{owner}/{repo}/license", headers=headers)
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

# ─────────────────────────────────────────────────────────────────────────────
# 5) Organization‐trained thresholds (from Munaiah et al. Table 2)
# ─────────────────────────────────────────────────────────────────────────────

ORG_THRESHOLDS = {
    "community":     2.0000,      # core contributors
    "ci":            1.0000,      # CI flag
    "documentation": 0.0018660,   # comment_ratio
    "history":       2.0895,      # commits_per_month
    "issues":        0.022989,    # issue_events_per_month
    "license":       1.0000,      # license flag
    "unittest":      0.0010160    # test_ratio
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
# 6) Download & unpack the GitHub repo tarball (no `git clone`)
# ─────────────────────────────────────────────────────────────────────────────

def get_default_branch(owner: str, repo: str, token: str = None):
    """
    GET https://api.github.com/repos/{owner}/{repo} to find the default branch.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        repo_info = resp.json()
        return repo_info.get("default_branch", "main")
    except Exception:
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
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    url = f"https://api.github.com/repos/{owner}/{repo}/tarball/{branch}"
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
        raise RuntimeError(f"Expected exactly one top-level directory in {tmp_dir}, got {entries!r}")
    top_folder = os.path.join(tmp_dir, entries[0])

    os.unlink(tmp_tar.name)
    return tmp_dir, top_folder

# ─────────────────────────────────────────────────────────────────────────────
# 7) Main: prompt for URL, compute all dimensions, and dump JSON
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ─── 7.1) Prompt the user for the GitHub repo URL (no .git suffix needed)
    repo_url = "https://github.com/luigifeola/Multi_Augmented_Realty_for_Kilobots_M-ARK"

    # ─── 7.2) Use the hardcoded GitHub token
    token = GITHUB_TOKEN

    # ─── 7.3) Parse owner and repo_name from URL
    m = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$", repo_url)
    if not m:
        print("Error: could not parse owner/repo from URL:", repo_url)
        sys.exit(1)

    owner = m.group("owner")
    repo_name = strip_dot_git_suffix(m.group("repo"))

    # ─── 7.4) Discover default branch
    default_branch = get_default_branch(owner, repo_name, token)
    print(f"Default branch for {owner}/{repo_name} = '{default_branch}'")

    # ─── 7.5) Download & unpack the tarball
    print(f"Downloading {owner}/{repo_name} tarball (branch '{default_branch}') …")
    try:
        tmp_root, clone_dir = download_and_unpack_tarball(owner, repo_name, branch=default_branch, token=token)
    except requests.exceptions.HTTPError as e:
        print("ERROR: could not download tarball. HTTPError:", e)
        sys.exit(1)
    except Exception as e:
        print("ERROR: failed to unpack tarball:", e)
        sys.exit(1)

    try:
        # ─── 7.6) Compute total SLOC and CLOC via Python
        print("Counting SLOC/CLOC in Python (no cloc) …")
        total_sloc, total_cloc = count_sloc_cloc_python(clone_dir)
        comment_ratio = total_cloc / (total_sloc + total_cloc) if (total_sloc + total_cloc) > 0 else 0.0

        # ─── 7.7) Compute TestRatio using the same Python logic
        print("Identifying test files and counting their SLOC …")
        sloc_tests = traverse_and_count_tests_exact_python(clone_dir)
        test_ratio = sloc_tests / total_sloc if total_sloc > 0 else 0.0

        # ─── 7.8) Git history metrics
        print("Gathering Git history metrics via API …")
        total_commits, first_date, last_date, author_counts = \
            compute_git_history_metrics_via_api(owner, repo_name, token)
        core_contribs = compute_core_contributors(author_counts)
        duration_months = compute_duration_months(first_date, last_date)
        commit_frequency = total_commits / duration_months

        # ─── 7.9) Issue metrics
        print("Fetching issue metrics from GitHub API …")
        total_issues, closed_issues, total_issue_comments = fetch_issue_metrics(owner, repo_name, token)
        total_issue_events = total_issues + closed_issues + total_issue_comments
        issue_frequency = total_issue_events / duration_months

        # ─── 7.10) CI & License flags
        print("Detecting CI configuration …")
        ci_flag = detect_ci_flag(owner, repo_name, token)
        print("Detecting license …")
        license_flag = detect_license_flag(owner, repo_name, token, local_clone_path=clone_dir)

        # ─── 7.11) Label each dimension
        labels = {}

        # (1) Community
        c_val = float(core_contribs)
        c_th  = ORG_THRESHOLDS["community"]
        labels["community"] = {
            "value": c_val,
            "estimation": label_continuous(c_val, c_th)
        }

        # (2) CI
        ci_val = float(ci_flag)
        labels["ci"] = {
            "value": ci_val,
            "estimation": label_binary(ci_flag)
        }

        # (3) Documentation
        d_val = comment_ratio
        d_th  = ORG_THRESHOLDS["documentation"]
        labels["documentation"] = {
            "value": d_val,
            "estimation": label_continuous(d_val, d_th)
        }

        # (4) History
        h_val = commit_frequency
        h_th  = ORG_THRESHOLDS["history"]
        labels["history"] = {
            "value": h_val,
            "estimation": label_continuous(h_val, h_th)
        }

        # (5) Issues
        i_val = issue_frequency
        i_th  = ORG_THRESHOLDS["issues"]
        labels["issues"] = {
            "value": i_val,
            "estimation": label_continuous(i_val, i_th)
        }

        # (6) License
        l_val = float(license_flag)
        labels["license"] = {
            "value": l_val,
            "estimation": label_binary(license_flag)
        }

        # (7) Unit Testing
        u_val = test_ratio
        u_th  = ORG_THRESHOLDS["unittest"]
        labels["unittest"] = {
            "value": u_val,
            "estimation": label_continuous(u_val, u_th)
        }

        # ─── 7.12) Build JSON output
        output = {
            "repo": repo_url,
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
                "total_sloc": total_sloc,
                "total_cloc": total_cloc,
                "test_sloc": sloc_tests,
                "total_commits": total_commits,
                "first_commit_date": first_date.isoformat(),
                "last_commit_date": last_date.isoformat(),
                "duration_months": duration_months,
                "issue_events": total_issue_events
            }
        }

        # ─── 7.13) Write JSON to disk
        out_path = "software-engineering-efforts.json"
        with open(out_path, "w", encoding="utf-8") as jf:
            json.dump(output, jf, indent=2)
        print(f"\nJSON results written to: {out_path}")

    finally:
        # Clean up: delete the temporary directory (handles read-only files)
        shutil.rmtree(tmp_root, onerror=on_rm_error)


if __name__ == "__main__":
    main()
