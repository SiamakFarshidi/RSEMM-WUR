import os
import json
import re
import requests
import time  # ✅ required for sleep and rate limit handling

# 👉 Set your GitHub token (optional but highly recommended)
# Option 1: Load from environment variable
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Option 2 (fallback): Hardcode token here (not recommended for shared code)
GITHUB_TOKEN = "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"

HEADERS = {
    "Accept": "application/vnd.github+json"
}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"

ZENODO_DIRS = ["wur_software", "non_wur_software"]

def extract_github_links(metadata):
    links = set()

    # related_identifiers
    for rel in metadata.get("related_identifiers", []):
        url = rel.get("identifier", "")
        if "github.com" in url:
            links.add(url.strip())

    # metadata URL
    url = metadata.get("url", "")
    if "github.com" in url:
        links.add(url.strip())

    # description body
    desc = metadata.get("description", "")
    if "github.com" in desc:
        found = re.findall(r'https?://github\.com/[^\s"\'<>]+', desc)
        links.update([link.strip().rstrip(').,') for link in found])

    return list(links)

def parse_github_repo(url):
    match = re.match(r'https?://github\.com/([^/]+)/([^/]+)', url)
    if match:
        owner = match.group(1)
        repo = match.group(2).split('/')[0].replace(".git", "").strip('/')
        return owner, repo
    return None, None

def github_api_request(url):
    while True:
        response = requests.get(url, headers=HEADERS)
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")

        if response.status_code == 403 and remaining == "0":
            reset_time = int(reset)
            sleep_time = max(reset_time - int(time.time()), 1)
            print(f"⏳ Rate limit hit. Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
            continue  # Retry
        return response

def fetch_github_repo_data(owner, repo):
    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    endpoints = {
        "repo_info": base_url,
        "languages": f"{base_url}/languages",
        "contributors": f"{base_url}/contributors",
        "commits": f"{base_url}/commits",
        "branches": f"{base_url}/branches",
        "topics": f"{base_url}/topics",
        "workflow": f"{base_url}/actions/workflows"
    }

    data = {}
    for key, url in endpoints.items():
        r = github_api_request(url)
        print(f"     🔗 {key}: {r.status_code}")
        if r.status_code == 200:
            data[key] = r.json()
        else:
            try:
                error_msg = r.json().get("message", "Unknown error")
            except:
                error_msg = r.text
            data[key] = {"error": f"Failed to fetch ({r.status_code})", "message": error_msg}
    return data

def should_skip_file(record):
    """Skip if 'Github Repo' exists and contains successful GitHub data."""
    if "Github Repo" not in record:
        return False
    for value in record["Github Repo"].values():
        if isinstance(value, dict):
            if all("error" not in v for v in value.values()):
                return True  # Contains valid data, skip
    return False  # Contains error or empty, reprocess

def enrich_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        record = json.load(f)

    if should_skip_file(record):
        print("  ✅ Skipping (already enriched).")
        return

    metadata = record.get("metadata", {})
    github_links = extract_github_links(metadata)

    github_data = {}
    if github_links:
        for url in github_links:
            owner, repo = parse_github_repo(url)
            if owner and repo:
                print(f"  🔍 Fetching GitHub data for {owner}/{repo}")
                github_data[url] = fetch_github_repo_data(owner, repo)
            else:
                github_data[url] = {"error": "Could not parse repo"}
    else:
        print("  🚫 No GitHub link found.")

    record["Github Repo"] = github_data if github_data else {}

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

def main():
    for base_dir in ZENODO_DIRS:
        print(f"\n📂 Processing directory: {base_dir}")
        for subdir, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".json"):
                    path = os.path.join(subdir, file)
                    print(f"\n📄 Enriching: {path}")
                    try:
                        enrich_json_file(path)
                    except Exception as e:
                        print(f"  ❌ Error processing {file}: {e}")

if __name__ == "__main__":
    main()
