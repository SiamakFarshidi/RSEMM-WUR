#!/usr/bin/env python3
import os
import json
import time
import logging
import re
import base64
from urllib.parse import urlparse

import requests

# Set up basic logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Hardcoded input configuration.
input_dir = r"C:\Users\User\Downloads\RSE-Analysis\RS2.0"
output_dir = r"C:\Users\User\Downloads\RSE-Analysis\RS3.0"
token = "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"
keywords_file = r"Keywords.json"

# List of file extensions considered "code".
CODE_EXTENSIONS = {
    '.py', '.pyw', '.pyi', '.pyx', '.pxd', '.ipynb',
    '.r', '.R', '.Rmd', '.rds',
    '.cs', '.cpp', '.c',
    '.m', '.mm', '.mlx',
    '.java', '.jar',
    '.bat', '.cmd',
    '.pl', '.pm', '.t',
    '.f', '.for', '.f90', '.f95', '.f03', '.f08',
    '.jl',
    '.rb', '.erb', '.rake', '.gemspec',
    '.pyx', '.pxd', '.pxi',
    '.lua',
    '.asm', '.s', '.S',
    '.nb', '.wl', '.wls', '.m',  # note: .m is used by both MATLAB & Wolfram.
    '.cu', '.cuh',
}

GITHUB_API_BASE = "https://api.github.com"

# Create a global requests session.
session = requests.Session()

# Simple cache for GitHub API responses.
github_cache = {}

def github_get_json(url, token, params=None):
    """Get JSON response from GitHub API with caching."""
    key = f"{url}?{json.dumps(params, sort_keys=True)}" if params else url
    if key in github_cache:
        logging.info(f"Cache hit for {url}")
        return github_cache[key]
    response = github_api_get(url, token, params=params)
    json_data = response.json()
    github_cache[key] = json_data
    return json_data

def github_api_get(url, token, params=None, attempt=0, max_attempts=5, timeout=10):
    headers = {"Authorization": f"token {token}"} if token else {}
    try:
        response = session.get(url, headers=headers, params=params, timeout=timeout)
    except Exception as e:
        if attempt < max_attempts:
            logging.warning(f"Request exception for {url}: {e}. Retrying in 5 seconds (attempt {attempt+1}/{max_attempts}).")
            time.sleep(5)
            return github_api_get(url, token, params, attempt=attempt+1, max_attempts=max_attempts, timeout=timeout)
        else:
            raise

    if response.status_code == 504:
        if attempt < max_attempts:
            logging.warning(f"504 Gateway Timeout for {url}. Retrying in 3 seconds (attempt {attempt+1}/{max_attempts}).")
            time.sleep(3)
            return github_api_get(url, token, params, attempt=attempt+1, max_attempts=max_attempts, timeout=timeout)
        else:
            response.raise_for_status()
    if response.status_code == 403:
        # Handle rate limiting.
        if response.headers.get("X-RateLimit-Remaining") == "0":
            reset_ts = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            sleep_time = max(reset_ts - int(time.time()), 10)
            logging.warning(f"Rate limit exceeded for {url}. Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
            return github_api_get(url, token, params, attempt=attempt+1, max_attempts=max_attempts, timeout=timeout)
    if response.status_code == 404:
        logging.warning(f"404 Not Found for {url}")
        return response
    response.raise_for_status()
    return response

def parse_github_url(html_url):
    """Parse a GitHub URL (e.g. https://github.com/owner/repo) into (owner, repo)."""
    parsed = urlparse(html_url)
    path_parts = parsed.path.strip("/").split("/")
    if len(path_parts) >= 2:
        return path_parts[0], path_parts[1]
    else:
        raise ValueError("Invalid GitHub URL format")

def get_recent_files(owner, repo, token, limit=50):
    """
    Iterate over recent commits to accumulate up to 'limit' unique code file paths.
    If a commit detail request fails (e.g. due to timeouts), skip that commit.
    """
    recent_files = {}
    per_page = 30
    page = 1
    while len(recent_files) < limit and page < 10:
        commits_url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits"
        params = {"per_page": per_page, "page": page}
        commits = github_get_json(commits_url, token, params=params)
        if not commits:
            break

        for commit in commits:
            sha = commit.get("sha")
            commit_detail_url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{sha}"
            try:
                commit_detail = github_get_json(commit_detail_url, token)
            except Exception as e:
                logging.warning(f"Skipping commit {sha} due to error: {e}")
                continue  # Skip this commit if we cannot get its details.
            commit_date = commit_detail.get("commit", {}).get("author", {}).get("date", "")
            files = commit_detail.get("files", [])
            for f in files:
                file_path = f.get("filename")
                _, ext = os.path.splitext(file_path)
                if ext.lower() in CODE_EXTENSIONS:
                    # Only update if commit_date is available and is more recent.
                    if commit_date and (file_path not in recent_files or commit_date > recent_files[file_path]):
                        recent_files[file_path] = commit_date
            if len(recent_files) >= limit:
                break
        page += 1

    file_list = [{"path": path, "last_update": date} for path, date in recent_files.items()]
    file_list.sort(key=lambda x: x["last_update"], reverse=True)
    return file_list[:limit]

def download_file_content(owner, repo, file_path, token):
    """Download and decode the content of a file from the GitHub repository.
    If the file download fails, return an empty string."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{file_path}"
    try:
        response = github_api_get(url, token)
        if response.status_code == 404:
            return ""
        content_json = response.json()
        if content_json.get("encoding") == "base64":
            return base64.b64decode(content_json.get("content", "")).decode('utf-8', errors='replace')
    except Exception as e:
        logging.warning(f"Failed to download {file_path}: {e}")
    return ""

def list_repository_root(owner, repo, token):
    """List files in the repository root directory."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents"
    return github_get_json(url, token)

# --- The remaining helper functions remain largely unchanged ---
def is_wolfram_language(file_content):
    return "(*" in file_content or "*)" in file_content

def extract_comments(file_content, file_extension):
    ext = file_extension.lower()
    if not ext.startswith('.'):
        ext = '.' + ext
    lines = file_content.splitlines()
    comments = []
    if ext in ['.py', '.pyw', '.pyi', '.pyx', '.pxd', 
               '.r', '.rmd', 
               '.pl', '.pm', '.t', 
               '.jl', 
               '.rb', '.erb', '.rake', '.gemspec']:
        for line in lines:
            if line.strip().startswith("#"):
                comments.append(line.strip())
    elif ext == '.ipynb':
        try:
            nb = json.loads(file_content)
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code":
                    for line in cell.get("source", []):
                        if line.strip().startswith("#"):
                            comments.append(line.strip())
        except Exception:
            pass
    elif ext in ['.js', '.java', '.c', '.cpp', '.cc', '.h', '.hpp',
                 '.cs', '.ts', '.go', '.php', '.rs', '.cu', '.cuh']:
        in_block = False
        for line in lines:
            stripped = line.strip()
            if in_block:
                comments.append(stripped)
                if "*/" in stripped:
                    in_block = False
            elif stripped.startswith("//"):
                comments.append(stripped)
            elif "/*" in stripped:
                in_block = True
                comments.append(stripped)
    elif ext == '.m':
        if is_wolfram_language(file_content):
            in_block = False
            for line in lines:
                stripped = line.strip()
                if in_block:
                    comments.append(stripped)
                    if "*)" in stripped:
                        in_block = False
                elif stripped.startswith("(*"):
                    in_block = True
                    comments.append(stripped)
        else:
            for line in lines:
                if line.strip().startswith("%"):
                    comments.append(line.strip())
    elif ext in ['.mm', '.mlx']:
        for line in lines:
            if line.strip().startswith("%"):
                comments.append(line.strip())
    elif ext in ['.bat', '.cmd']:
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith("REM") or stripped.startswith("::"):
                comments.append(stripped)
    elif ext in ['.f', '.for', '.f90', '.f95', '.f03', '.f08']:
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("!") or (stripped and stripped[0] in ['c', 'C', '*']):
                comments.append(stripped)
    elif ext in ['.lua']:
        in_block = False
        for line in lines:
            stripped = line.strip()
            if in_block:
                comments.append(stripped)
                if "]]" in stripped:
                    in_block = False
            elif stripped.startswith("--"):
                comments.append(stripped)
                if stripped.startswith("--[[") and "]]" not in stripped:
                    in_block = True
    elif ext in ['.asm', '.s', '.S']:
        for line in lines:
            if line.strip().startswith(";"):
                comments.append(line.strip())
    elif ext in ['.nb', '.wl', '.wls']:
        in_block = False
        for line in lines:
            stripped = line.strip()
            if in_block:
                comments.append(stripped)
                if "*)" in stripped:
                    in_block = False
            elif stripped.startswith("(*"):
                in_block = True
                comments.append(stripped)
    else:
        for line in lines:
            if line.strip().startswith("#"):
                comments.append(line.strip())
    return "\n".join(comments)

def compute_documentation_ratio(file_content, file_path):
    _, ext = os.path.splitext(file_path)
    comment_text = extract_comments(file_content, ext)
    total_len = len(file_content)
    return len(comment_text) / total_len if total_len else 0.0

def check_code_gen_sign(file_content, code_gen_keywords):
    count = 0
    file_lower = file_content.lower()
    for kw in code_gen_keywords:
        count += file_lower.count(kw.lower())
    return count

def detect_gen_ai(file_content, code_gen_keywords, threshold=1):
    lower_content = file_content.lower()
    if "generated by" in lower_content or "auto-generated" in lower_content:
        return True
    if check_code_gen_sign(file_content, code_gen_keywords) >= threshold:
        return True
    return False

def get_package_list(owner, repo, token):
    packages = set()
    root_files = list_repository_root(owner, repo, token)
    filenames = {fi.get("name"): fi for fi in root_files if "name" in fi}
    if "requirements.txt" in filenames:
        req_url = filenames["requirements.txt"].get("url")
        try:
            resp = github_get_json(req_url, token)
            content = base64.b64decode(resp.get("content", "")).decode('utf-8', errors='replace')
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    pkg = re.split(r'[<=>]', line)[0].strip()
                    if pkg:
                        packages.add(pkg)
        except Exception:
            logging.info("requirements.txt could not be parsed.")
    else:
        logging.info("requirements.txt not found in repository root.")
    if "package.json" in filenames:
        pkg_json_url = filenames["package.json"].get("url")
        try:
            resp = github_get_json(pkg_json_url, token)
            content = base64.b64decode(resp.get("content", "")).decode('utf-8', errors='replace')
            pkg_data = json.loads(content)
            for dep_field in ["dependencies", "devDependencies"]:
                if dep_field in pkg_data:
                    for pkg in pkg_data[dep_field]:
                        packages.add(pkg)
        except Exception:
            logging.info("package.json could not be parsed.")
    else:
        logging.info("package.json not found in repository root.")
    return list(packages)

def get_readme_length(owner, repo, token):
    root_files = list_repository_root(owner, repo, token)
    readme_file = None
    for file_info in root_files:
        name = file_info.get("name", "").lower()
        if name.startswith("readme"):
            readme_file = file_info
            break
    if readme_file:
        try:
            resp = github_get_json(readme_file.get("url"), token)
            content = base64.b64decode(resp.get("content", "")).decode('utf-8', errors='replace')
            return len(content)
        except Exception:
            logging.info("README could not be parsed.")
            return 0
    else:
        logging.info("README not found in repository root.")
        return 0

def process_repository(github_html_url, token, code_gen_keywords, test_keywords):
    owner, repo = parse_github_url(github_html_url)
    logging.info(f"Processing repository {owner}/{repo}")
    analysis = {}

    recent_files = get_recent_files(owner, repo, token, limit=20)
    file_details = []
    doc_ratios = []
    for file_info in recent_files:
        file_path = file_info["path"]
        try:
            content = download_file_content(owner, repo, file_path, token)
            # Skip files with no content.
            if not content:
                continue
            ratio = compute_documentation_ratio(content, file_path)
            doc_ratios.append(ratio)
            code_gen_count = check_code_gen_sign(content, code_gen_keywords)
            gen_ai_detected = detect_gen_ai(content, code_gen_keywords, threshold=1)
            file_details.append({
                "path": file_path,
                "last_update": file_info["last_update"],
                "documentation_ratio": ratio,
                "code_gen_keyword_occurrences": code_gen_count,
                "gen-ai": gen_ai_detected
            })
        except Exception as e:
            logging.warning(f"Failed to process file {file_path}: {e}")
            continue

    avg_doc_ratio = sum(doc_ratios) / len(doc_ratios) if doc_ratios else 0.0
    analysis["recent_files_analysis"] = file_details
    analysis["average_documentation_ratio"] = avg_doc_ratio
    analysis["packages_used"] = get_package_list(owner, repo, token)
    analysis["readme_length"] = get_readme_length(owner, repo, token)
    
    # Attempt to read a test log.
    root_files = list_repository_root(owner, repo, token)
    filenames = {fi.get("name"): fi for fi in root_files if "name" in fi}
    if "test.log" in filenames:
        try:
            test_log_url = filenames["test.log"].get("url")
            resp = github_get_json(test_log_url, token)
            content = base64.b64decode(resp.get("content", "")).decode('utf-8', errors='replace')
            analysis["test_log_length"] = len(content)
        except Exception:
            logging.info("Failed to parse test.log.")
    else:
        logging.info("No test.log found in repository root.")
    
    return analysis

def process_json_file(json_path, output_dir, token, code_gen_keywords, test_keywords):
    base_name = os.path.basename(json_path)
    out_path = os.path.join(output_dir, base_name)
    data = None

    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if "repo_analysis_error" not in existing_data:
                logging.info(f"Output file {out_path} exists and appears valid. Skipping processing.")
                return
            else:
                logging.info(f"Output file {out_path} contains errors; reprocessing for error fixing.")
                data = existing_data
        except Exception as e:
            logging.info(f"Output file {out_path} exists but could not be read ({e}). Reprocessing.")
    if data is None:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if (data.get("GitHub_html_url") or "").strip():
        try:
            repo_analysis = process_repository(data["GitHub_html_url"], token, code_gen_keywords, test_keywords)
            data["repo_analysis"] = repo_analysis
            if "repo_analysis_error" in data:
                del data["repo_analysis_error"]
            logging.info(f"Successfully processed repository: {data['GitHub_html_url']}")
        except Exception as e:
            logging.error(f"Error processing repository {data['GitHub_html_url']}: {e}")
            data["repo_analysis_error"] = str(e)
    else:
        logging.info(f"No GitHub_html_url in {json_path} or it is empty; skipping repository processing.")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Processed file saved to {out_path}")

def main():
    if os.path.exists(keywords_file):
        with open(keywords_file, "r", encoding="utf-8") as f:
            keywords_data = json.load(f)
        test_keywords = keywords_data.get("test_keywords", [])
        code_gen_keywords = keywords_data.get("code_gen_keywords", [])
        logging.info(f"Loaded test_keywords: {test_keywords} and code_gen_keywords: {code_gen_keywords} from {keywords_file}")
    else:
        logging.warning(f"Keywords file {keywords_file} not found. Using default keywords.")
        test_keywords = ["test case"]
        code_gen_keywords = ["openai"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        for file_name in os.listdir(input_dir):
            if file_name.lower().endswith(".json"):
                json_path = os.path.join(input_dir, file_name)
                logging.info(f"Processing JSON file: {json_path}")
                process_json_file(json_path, output_dir, token, code_gen_keywords, test_keywords)
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected. Exiting gracefully.")

if __name__ == "__main__":
    main()
