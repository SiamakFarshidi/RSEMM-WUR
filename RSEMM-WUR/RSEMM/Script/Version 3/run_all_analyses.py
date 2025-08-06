#!/usr/bin/env python3
"""
run_all_analyses.py

For every Zenodo JSON file in the given directory, perform:
  1. FAIRness assessment
  2. SE practices analysis
  3. Code‐generation detection (now including per‐file labels)
  4. AI/ML/Ops detection

Then combine all results under "repo-profile" in the same JSON file,
overwriting any existing "repo-profile".

This version adds:
  - A progress file (“processed_files.json”) to remember which Zenodo files are done.
  - Helpers to detect GitHub rate limits and network‐down errors, wait, and then retry.
"""

import os
import sys
import json
import shutil
import tempfile
import datetime
import time
import re
import socket
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

from importlib.machinery import SourceFileLoader

# ────────────────────────────────────────────────────────────────────────────────
# 1) PATHS TO THE FOUR ANALYSIS SCRIPTS
# ────────────────────────────────────────────────────────────────────────────────
MODULE_PATH_ASSESS   = r".\assessFAIRness.py"
MODULE_PATH_SE       = r".\SE-practises-analysis.py"
MODULE_PATH_CODEGEN  = r".\code-gen-detection.py"
MODULE_PATH_AIMLOPS  = r".\ai-ml-ops-detection.py"

def load_module_from_path(module_name: str, path: str):
    loader = SourceFileLoader(module_name, path)
    return loader.load_module()

assess_module  = load_module_from_path("assess_module", MODULE_PATH_ASSESS)
se_module      = load_module_from_path("se_module", MODULE_PATH_SE)
codegen_module = load_module_from_path("codegen_module", MODULE_PATH_CODEGEN)
aimlops_module = load_module_from_path("aimlops_module", MODULE_PATH_AIMLOPS)

# centrally define your token (or read it once from env):
HARDCODED_GITHUB = "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"

se_module.GITHUB_TOKEN      = HARDCODED_GITHUB
assess_module.GITHUB_TOKEN  = HARDCODED_GITHUB
codegen_module.GITHUB_TOKEN = HARDCODED_GITHUB

def get_github_headers_override():
    headers = {"Accept": "application/vnd.github.mercy-preview+json"}
    if HARDCODED_GITHUB:
        headers["Authorization"] = f"token {HARDCODED_GITHUB}"
    else:
        print("Warning: No GITHUB_TOKEN available.")
    return headers

# Replace its original with ours:
aimlops_module.get_github_headers = get_github_headers_override


# ────────────────────────────────────────────────────────────────────────────────
# 2) DIRECTORY CONTAINING ALL ZENODO JSON FILES (INPUT) & PROGRESS FILE
# ────────────────────────────────────────────────────────────────────────────────
INPUT_DIR      = r"C:\Users\User\Downloads\RS-Zenodo\RS-Repo-Zenodo - v3"
PROGRESS_FILE  = "processed_files.json"   # <— Will keep track of completed Zenodo JSON filenames

# ────────────────────────────────────────────────────────────────────────────────
# 3) HELPER: PARSE owner & repo FROM A GITHUB URL (REUSED ACROSS MODULES)
# ────────────────────────────────────────────────────────────────────────────────
def parse_owner_repo_from_url(github_url: str):
    """
    Return (owner, repo) or raise ValueError. Accepts only
    well‐formed owner/repo names (alphanumeric + dash + underscore + dot).
    """
    url = github_url.strip().rstrip("/")
    url = re.sub(r"\.git$", "", url)

    m1 = re.match(r"^https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)(?:/.*)?$", url)
    if m1:
        return m1.group(1), m1.group(2)

    m2 = re.match(r"^git@github\.com:([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$", url)
    if m2:
        return m2.group(1), m2.group(2)

    raise ValueError(f"Cannot parse owner/repo from '{github_url}'")

# ────────────────────────────────────────────────────────────────────────────────
# 4) GITHUB RATE-LIMIT & NETWORK-DOWN HANDLING
# ────────────────────────────────────────────────────────────────────────────────
GITHUB_API_BASE = "https://api.github.com"

def get_rate_limit_reset_time(token: str) -> int:
    """
    Query GitHub API /rate_limit. Return the 'core' reset epoch (as integer seconds).
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {token}"
    }
    try:
        resp = requests.get(f"{GITHUB_API_BASE}/rate_limit", headers=headers, timeout=10)
    except Exception:
        # If we can’t query rate_limit for some reason (network, DNS), default to sleeping 60 seconds
        return int(time.time()) + 60
    if resp.status_code != 200:
        return int(time.time()) + 60
    data = resp.json()
    core = data.get("resources", {}).get("core", {})
    reset_ts = core.get("reset", int(time.time()) + 60)
    return int(reset_ts)

def sleep_until_rate_limit_resets(token: str):
    """
    Inspect GitHub’s /rate_limit, find reset time, sleep until that time + 5 sec buffer.
    """
    now = int(time.time())
    reset_ts = get_rate_limit_reset_time(token)
    wait_secs = max(reset_ts - now + 5, 5)
    print(f"→ GitHub rate limit exhausted. Sleeping for {wait_secs} seconds (until reset)...")
    time.sleep(wait_secs)

def is_rate_limit_exception(exc: Exception) -> bool:
    """
    Inspect an exception (coming from a GitHub API failure) to decide if it's a rate-limit.
    Heuristic: if the exception message contains 'rate limit' or '403'.
    """
    msg = str(exc).lower()
    if "rate limit" in msg or ("403" in msg and "github" in msg):
        return True
    return False

def is_network_exception(exc: Exception) -> bool:
    """
    Return True if 'exc' represents a DNS‐resolution or other network‐down error
    (e.g. Failed to resolve 'api.github.com', connection refused, timeouts, etc.).
    """
    # If requests wrapped a lower‐level socket.gaierror or NameResolutionError
    if isinstance(exc, RequestException):
        msg = str(exc).lower()
        if "failed to resolve" in msg or "nameresolutionerror" in msg or "getaddrinfo" in msg:
            return True
        if isinstance(exc, ConnectionError) or isinstance(exc, Timeout):
            return True

    # Direct socket.gaierror
    if isinstance(exc, socket.gaierror):
        return True

    return False

def wait_for_network():
    """
    Loop until we can successfully hit GitHub's root. Sleeps 30s between tries.
    """
    while True:
        try:
            requests.head("https://api.github.com", timeout=5)
            return
        except Exception:
            print("→ Network still unavailable. Retrying in 30 seconds…")
            time.sleep(30)

# ────────────────────────────────────────────────────────────────────────────────
# 5) LOAD (OR INITIALIZE) PROGRESS FILE
# ────────────────────────────────────────────────────────────────────────────────
def load_progress() -> set:
    """
    Return a set of filenames (just the base names, e.g. '1234567.json')
    that have already been processed. If PROGRESS_FILE doesn’t exist, return empty set.
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
    Overwrite PROGRESS_FILE with the sorted list of completed filenames.
    """
    tmp = list(sorted(done_set))
    with open(PROGRESS_FILE, "w", encoding="utf-8") as pf:
        json.dump(tmp, pf, indent=2)

# ────────────────────────────────────────────────────────────────────────────────
# 6) MAIN LOOP: WALK THROUGH EVERY .JSON IN INPUT_DIR,
#    SKIPPING ALREADY-PROCESSED FILES, AND RETRYING ON RATE LIMIT OR NETWORK ERRORS
# ────────────────────────────────────────────────────────────────────────────────
def main():
    # (1) Load or initialize progress
    processed = load_progress()

    # (2) Walk all JSON files in INPUT_DIR
    for root, dirs, files in os.walk(INPUT_DIR):
        for fname in files:
            if not fname.lower().endswith(".json"):
                continue

            # If we've already fully processed this Zenodo JSON, skip it:
            if fname in processed:
                print(f"Skipping '{fname}' (already processed).")
                continue

            json_path = os.path.join(root, fname)

            # Wrap each entire-file processing inside a retry loop
            while True:
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        record = json.load(f)
                except Exception as e:
                    print(f"Skipping '{json_path}' (could not load JSON): {e}", file=sys.stderr)
                    break  # Skip this file permanently

                # ───────────
                # 6.1) EXTRACT GITHUB URL FROM ZENODO JSON
                # ───────────
                git_url = assess_module.extract_git_url_from_zenodo(record)
                has_git_url = isinstance(git_url, str) and git_url.startswith("https://github.com/")

                # ───────────
                # 6.2) RUN FAIRness ASSESSMENT
                # ───────────
                fair_subp = assess_module.assess_fairness_metadata(record)
                fair_est  = assess_module.estimate_fairness_from_subprinciples(fair_subp)

                if has_git_url:
                    try:
                        howfair = assess_module.run_howfairis_checks(git_url)
                    except Exception as e:
                        if is_rate_limit_exception(e):
                            sleep_until_rate_limit_resets(assess_module.GITHUB_TOKEN)
                            continue  # retry entire file
                        elif is_network_exception(e):
                            wait_for_network()
                            continue  # retry entire file
                        else:
                            howfair = {
                                "howfairis_score": 0,
                                "howfairis_details": {
                                    "repository": "no", "license": "no",
                                    "registry": "no", "citation": "no", "quality": "no"
                                }
                            }
                else:
                    howfair = {
                        "howfairis_score": 0,
                        "howfairis_details": {
                            "repository": "no", "license": "no",
                            "registry": "no", "citation": "no", "quality": "no"
                        }
                    }

                fairness_result = {
                    "subprinciple_assessment": fair_subp,
                    "principle_categories":    fair_est["principle_categories"],
                    "overall_fairness":        fair_est["overall_fairness"],
                    "git_url":                 git_url or None,
                    "howfairis_score":         howfair["howfairis_score"],
                    "howfairis_details":       howfair["howfairis_details"]
                }

                # ───────────
                # 6.3) RUN SE PRACTICES ANALYSIS
                # ───────────
                if has_git_url:
                    try:
                        owner, repo_name = parse_owner_repo_from_url(git_url)
                    except ValueError:
                        has_git_url = False

                if has_git_url:
                    # 6.3.a) Find default branch
                    try:
                        default_branch = se_module.get_default_branch(owner, repo_name, se_module.GITHUB_TOKEN)
                    except Exception as e:
                        if is_rate_limit_exception(e):
                            sleep_until_rate_limit_resets(se_module.GITHUB_TOKEN)
                            continue
                        elif is_network_exception(e):
                            wait_for_network()
                            continue
                        else:
                            default_branch = None

                    if default_branch:
                        # 6.3.b) Download & unpack tarball
                        try:
                            tmp_root, clone_dir = se_module.download_and_unpack_tarball(
                                owner, repo_name, branch=default_branch, token=se_module.GITHUB_TOKEN
                            )
                        except Exception as e:
                            if is_rate_limit_exception(e):
                                sleep_until_rate_limit_resets(se_module.GITHUB_TOKEN)
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
                            # 6.3.c) Compute metrics & labels
                            try:
                                total_sloc, total_cloc = se_module.count_sloc_cloc_python(clone_dir)
                                comment_ratio = total_cloc / (total_sloc + total_cloc) if (total_sloc + total_cloc) > 0 else 0.0

                                sloc_tests = se_module.traverse_and_count_tests_exact_python(clone_dir)

                                total_commits, first_date, last_date, author_counts = \
                                    se_module.compute_git_history_metrics_via_api(owner, repo_name, se_module.GITHUB_TOKEN)
                                core_contribs = se_module.compute_core_contributors(author_counts)
                                duration_months = se_module.compute_duration_months(first_date, last_date)

                                total_issues, closed_issues, total_issue_comments = \
                                    se_module.fetch_issue_metrics(owner, repo_name, se_module.GITHUB_TOKEN)
                                total_issue_events = total_issues + closed_issues + total_issue_comments
                                issue_frequency = total_issue_events / duration_months

                                ci_flag = se_module.detect_ci_flag(owner, repo_name, se_module.GITHUB_TOKEN)
                                license_flag = se_module.detect_license_flag(
                                    owner, repo_name, se_module.GITHUB_TOKEN, local_clone_path=clone_dir
                                )

                                test_ratio = sloc_tests / total_sloc if total_sloc > 0 else 0.0

                                labels = {}
                                c_val = float(core_contribs)
                                c_th  = se_module.ORG_THRESHOLDS["community"]
                                labels["community"] = {
                                    "value": c_val,
                                    "estimation": se_module.label_continuous(c_val, c_th)
                                }
                                ci_val = float(ci_flag)
                                labels["ci"] = {
                                    "value": ci_val,
                                    "estimation": se_module.label_binary(ci_flag)
                                }
                                d_val = comment_ratio
                                d_th  = se_module.ORG_THRESHOLDS["documentation"]
                                labels["documentation"] = {
                                    "value": d_val,
                                    "estimation": se_module.label_continuous(d_val, d_th)
                                }
                                h_val = total_commits / duration_months
                                h_th  = se_module.ORG_THRESHOLDS["history"]
                                labels["history"] = {
                                    "value": h_val,
                                    "estimation": se_module.label_continuous(h_val, h_th)
                                }
                                i_val = issue_frequency
                                i_th  = se_module.ORG_THRESHOLDS["issues"]
                                labels["issues"] = {
                                    "value": i_val,
                                    "estimation": se_module.label_continuous(i_val, i_th)
                                }
                                l_val = float(license_flag)
                                labels["license"] = {
                                    "value": l_val,
                                    "estimation": se_module.label_binary(license_flag)
                                }
                                u_val = test_ratio
                                u_th  = se_module.ORG_THRESHOLDS["unittest"]
                                labels["unittest"] = {
                                    "value": u_val,
                                    "estimation": se_module.label_continuous(u_val, u_th)
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
                                        shutil.rmtree(tmp_root, onerror=se_module.on_rm_error)
                                    except Exception:
                                        pass
                                    sleep_until_rate_limit_resets(se_module.GITHUB_TOKEN)
                                    continue
                                elif is_network_exception(e):
                                    try:
                                        shutil.rmtree(tmp_root, onerror=se_module.on_rm_error)
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

                            try:
                                shutil.rmtree(tmp_root, onerror=se_module.on_rm_error)
                            except Exception:
                                pass
                        # End of SE processing block
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
                # 6.4) RUN CODE‐GENERATION DETECTION
                # ───────────
                if has_git_url:
                    try:
                        cg_owner, cg_repo = parse_owner_repo_from_url(git_url)
                    except ValueError:
                        has_git_url = False

                if has_git_url:
                    # 6.4.a) Find default branch for CodeGen
                    try:
                        default_branch_cg = codegen_module.get_default_branch(cg_owner, cg_repo)
                    except Exception as e:
                        if is_rate_limit_exception(e):
                            sleep_until_rate_limit_resets(se_module.GITHUB_TOKEN)
                            continue
                        elif is_network_exception(e):
                            wait_for_network()
                            continue
                        else:
                            default_branch_cg = None

                    if default_branch_cg:
                        # 6.4.b) List all files
                        try:
                            full_tree = codegen_module.list_all_files(cg_owner, cg_repo, default_branch_cg)
                        except Exception as e:
                            if is_rate_limit_exception(e):
                                sleep_until_rate_limit_resets(se_module.GITHUB_TOKEN)
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
                                full_tree = []
                        else:
                            all_files = [entry["path"] for entry in full_tree if entry["type"] == "blob"]
                            script_files = [p for p in all_files if codegen_module.is_script_file(p)]
                            total_script_count = len(script_files)

                            # 6.4.c) Get recent candidates
                            target_candidates = codegen_module.NUM_RECENT_FILES * 3
                            try:
                                recent_candidates = codegen_module.get_recent_script_files(
                                    cg_owner, cg_repo, default_branch_cg, target_candidates
                                )
                            except Exception as e:
                                if is_rate_limit_exception(e):
                                    sleep_until_rate_limit_resets(se_module.GITHUB_TOKEN)
                                    continue
                                elif is_network_exception(e):
                                    wait_for_network()
                                    continue
                                else:
                                    print(f"  [CodeGen] Error fetching recent scripts for {cg_owner}/{cg_repo}: {e}", file=sys.stderr)
                                    recent_candidates = []

                            # 6.4.d) Filter by raw-file existence
                            valid_recent = []
                            for pth in recent_candidates:
                                if len(valid_recent) >= codegen_module.NUM_RECENT_FILES:
                                    break
                                try:
                                    exists = codegen_module.raw_file_exists(cg_owner, cg_repo, default_branch_cg, pth)
                                except Exception:
                                    exists = False
                                if exists:
                                    valid_recent.append(pth)

                            # 6.4.e) Fetch each valid file & classify
                            human_count = 0
                            ai_count    = 0
                            analyzed_scripts = {}

                            for script_path in valid_recent:
                                try:
                                    code_text = codegen_module.fetch_raw_file(
                                        cg_owner, cg_repo, default_branch_cg, script_path
                                    )
                                except Exception as fe:
                                    if is_rate_limit_exception(fe):
                                        sleep_until_rate_limit_resets(se_module.GITHUB_TOKEN)
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

                                label = codegen_module.classify_with_gpt(code_text)
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
                # 6.5) RUN AI/ML/OPS DETECTION
                # ───────────
                if has_git_url:
                    try:
                        am_owner, am_repo = parse_owner_repo_from_url(git_url)
                    except ValueError:
                        has_git_url = False

                if has_git_url:
                    try:
                        meta_info, category_am, types_am = \
                            aimlops_module.classify_repository(am_owner, am_repo)
                        am_result = {
                            "repository": f"{am_owner}/{am_repo}",
                            "category":   category_am,
                            "types":      types_am,
                            "name":       meta_info.get("name", ""),
                            "description":meta_info.get("description", ""),
                            "topics":     meta_info.get("topics", []),
                            "timestamp":  datetime.datetime.now(
                                datetime.timezone.utc
                            ).isoformat()
                        }
                    except Exception as e:
                        if is_rate_limit_exception(e):
                            sleep_until_rate_limit_resets(se_module.GITHUB_TOKEN)
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
                # 6.6) COMBINE ALL FOUR RESULTS UNDER "repo-profile"
                # ───────────
                record["repo-profile"] = {
                    "fairness_estimation":    fairness_result,
                    "se_estimation":          se_result,
                    "genai_detection":        codegen_result,
                    "ai_ml_ops_detection":    am_result
                }

                # ───────────
                # 6.7) WRITE THE UPDATED JSON BACK TO DISK (OVERWRITE)
                # ───────────
                try:
                    with open(json_path, "w", encoding="utf-8") as out_f:
                        json.dump(record, out_f, indent=2)
                    print(f"Updated '{json_path}' with combined repo-profile.")
                except Exception as e:
                    print(f"Failed to write back to '{json_path}': {e}", file=sys.stderr)
                    break  # Don’t mark as processed if write fails

                # ───────────
                # 6.8) MARK THIS FILE AS “DONE” IN PROGRESS FILE
                # ───────────
                processed.add(fname)
                save_progress(processed)
                break  # Exit the retry loop for this file

            # End of while-True (per-file retry loop)

    print("All files scanned. Exiting.")

if __name__ == "__main__":
    main()
