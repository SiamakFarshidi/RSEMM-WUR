import os
import json
import csv
import re
import requests

# Default flat structure for output JSON
DEFAULT_FLAT_DATA = {
    # Zenodo fields
    "Zenodo_doi_url": None,
    "Zenodo_title": None,
    "Zenodo_revision": None,
    "Zenodo_resource_type": None,
    "Zenodo_recid": None,
    "Zenodo_publication_date": None,
    "Zenodo_modified_updated_date": None,
    "Zenodo_license": None,
    "Zenodo_language": None,
    "Zenodo_id": None,
    "Zenodo_stats_views": None,
    "Zenodo_stats_downloads": None,
    "zenodo_files_count": None,
    "Zenodo_keywords": [],
    "zenodo_grants_name": [],
    "zenodo_grant_url": [],
    "zenodo_grant_program": [],
    "zenodo_grant_acronym": [],
    "zenodo_files_filenames": [],
    "Zenodo_description": None,
    "Zenodo_creators_orcid": [],
    "Zenodo_creators_names": [],
    "Zenodo_creators_affiliations": [],
    "Zenodo_creation_date": None,
    "Zenodo_communities": [],
    "Zenodo_access_right": None,
    # GitHub fields
    "GitHub_has_projects": None,
    "GitHub_has_issues": None,
    "GitHub_has_wiki": None,
    "GitHub_has_pages": None,
    "GitHub_has_discussions": None,
    "GitHub_is_mirrored": None,
    "GitHub_is_archived": None,
    "GitHub_visibility": None,
    "GitHub_private": None,
    "GitHub_fork": None,
    "GitHub_open_issues_count": None,
    "GitHub_workflow_total_count": None,
    "GitHub_watchers_count": None,
    "GitHub_subscribers_count": None,
    "GitHub_stargazers_count": None,
    "GitHub_size": None,
    "GitHub_network_count": None,
    "GitHub_forks_count": None,
    "GitHub_commits_comment_count": None,
    "GitHub_git_number_of_followers": None,
    "GitHub_git_number_of_updates": None,
    "GitHub_git_number_of_authors": None,
    "GitHub_git_number_of_committers": None,
    "GitHub_git_number_of_branches": None,
    "GitHub_contributors_count": None,
    "GitHub_commits_verification_verified_count": None,
    "GitHub_git_license": None,
    "GitHub_html_url": None,
    "GitHub_git_url": None,
    "GitHub_full_name": None,
    "GitHub_description": None,
    "GitHub_default_branch": None,
    "GitHub_created_at": None,
    "GitHub_pushed_at": None,
    "GitHub_updated_at": None,
    "GitHub_topics": [],
    "GitHub_commits_verification_reasons": [],
    "GitHub_languages": [],
    "GitHub_workflows_names": [],
    "GitHub_commits_messages": [],
    "GitHub_branches_names": []
}

def dedup_list(items):
    """Return a deduplicated list excluding falsy values."""
    seen = set()
    unique = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique

def clean_text(text):
    """Clean text by removing typical code symbols and extra whitespace."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[`{}<>;#@()\[\]]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fetch_owner_followers(url):
    """Fetch owner followers count from the given URL using a GET request."""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Assumes the returned JSON contains a field 'followers'
            return data.get('followers')
        else:
            return None
    except Exception as e:
        print(f"Error fetching owner followers from {url}: {e}")
        return None

def flatten_grants(grants):
    """Flatten grants into separate lists."""
    grants_name = []
    grant_url = []
    grant_program = []
    grant_acronym = []
    for grant in grants:
        grants_name.append(grant.get("title", None))
        grant_url.append(grant.get("url", None))
        grant_program.append(grant.get("program", None))
        grant_acronym.append(grant.get("acronym", None))
    return {
        "zenodo_grants_name": dedup_list(grants_name),
        "zenodo_grant_url": dedup_list(grant_url),
        "zenodo_grant_program": dedup_list(grant_program),
        "zenodo_grant_acronym": dedup_list(grant_acronym)
    }

def extract_zenodo_metadata(data):
    # Clean description by removing HTML tags
    desc = data.get("metadata", {}).get("description", None)
    if desc:
        desc_clean = re.sub(r'<[^>]+>', ' ', desc)
        desc_clean = re.sub(r'\s+', ' ', desc_clean).strip()
    else:
        desc_clean = None

    extracted = {
        "Zenodo_title": data.get("title", None),
        "Zenodo_stats_views": data.get("stats", {}).get("views", None),
        "Zenodo_stats_downloads": data.get("stats", {}).get("downloads", None),
        "Zenodo_revision": str(data.get("revision", None)) if data.get("revision") is not None else None,
        "Zenodo_resource_type": data.get("metadata", {}).get("resource_type", {}).get("title", None),
        "Zenodo_recid": data.get("recid", None),
        "Zenodo_publication_date": data.get("metadata", {}).get("publication_date", None),
        "Zenodo_modified_updated_date": data.get("updated", None),
        "Zenodo_license": data.get("metadata", {}).get("license", {}).get("id", None),
        "Zenodo_language": data.get("metadata", {}).get("language", None),
        "Zenodo_keywords": dedup_list(data.get("metadata", {}).get("keywords", [])),
        "Zenodo_id": str(data.get("id", None)) if data.get("id") is not None else None,
        "zenodo_files_filenames": dedup_list([f.get("key", None) for f in data.get("files", [])]),
        "zenodo_files_count": len(data.get("files", [])) if data.get("files") is not None else None,
        "Zenodo_description": desc_clean,
        "Zenodo_creation_date": data.get("created", None),
        "Zenodo_access_right": data.get("metadata", {}).get("access_right", None),
        "Zenodo_doi_url": data.get("doi_url", None)
    }
    # Process creators
    creators = data.get("metadata", {}).get("creators", [])
    extracted["Zenodo_creators_names"] = dedup_list([creator.get("name", None) for creator in creators])
    extracted["Zenodo_creators_affiliations"] = dedup_list([creator.get("affiliation", None) for creator in creators])
    extracted["Zenodo_creators_orcid"] = dedup_list([creator.get("orcid", None) for creator in creators])
    
    # Process grants
    grants = data.get("metadata", {}).get("grants", [])
    flattened_grants = flatten_grants(grants)
    extracted.update(flattened_grants)
    
    # Process communities (rename field to Zenodo_communities)
    communities = data.get("metadata", {}).get("communities", [])
    extracted["Zenodo_communities"] = dedup_list([community.get("id", None) for community in communities])
    
    return extracted

def extract_github_metadata(data):
    github_data = data.get("Github Repo", {})
    # Initialize merged dictionary for GitHub fields
    merged = {
        "GitHub_has_projects": None,
        "GitHub_has_issues": None,
        "GitHub_has_wiki": None,
        "GitHub_has_pages": None,
        "GitHub_has_discussions": None,
        "GitHub_is_mirrored": None,
        "GitHub_is_archived": None,
        "GitHub_visibility": None,
        "GitHub_private": None,
        "GitHub_fork": None,
        "GitHub_open_issues_count": None,
        "GitHub_workflow_total_count": 0,
        "GitHub_watchers_count": 0,
        "GitHub_subscribers_count": 0,
        "GitHub_stargazers_count": 0,
        "GitHub_size": 0,
        "GitHub_network_count": 0,
        "GitHub_forks_count": 0,
        "GitHub_commits_comment_count": 0,
        "GitHub_git_number_of_followers": 0,
        "GitHub_git_number_of_updates": 0,
        "GitHub_git_number_of_authors": 0,
        "GitHub_git_number_of_committers": 0,
        "GitHub_git_number_of_branches": 0,
        "GitHub_contributors_count": 0,
        "GitHub_commits_verification_verified_count": 0,
        "GitHub_git_license": None,
        "GitHub_html_url": None,
        "GitHub_git_url": None,
        "GitHub_full_name": None,
        "GitHub_description": None,
        "GitHub_default_branch": None,
        "GitHub_created_at": None,
        "GitHub_pushed_at": None,
        "GitHub_updated_at": None,
        "GitHub_topics": [],
        "GitHub_commits_verification_reasons": [],
        "GitHub_languages": [],
        "GitHub_workflows_names": [],
        "GitHub_commits_messages": [],
        "GitHub_branches_names": []
    }
    
    # Sets for aggregating unique contributor and committer logins and branch names
    authors_set = set()
    committers_set = set()
    branches_set = set()
    total_commits = 0
    followers_list = []  # To collect followers count per repo

    for repo_url, repo_data in github_data.items():
        repo_info = repo_data.get("repo_info", {})
        if not repo_info:
            continue

        # --- License extraction ---
        license_data = repo_info.get("license")
        if license_data:
            merged["GitHub_git_license"] = license_data.get("spdx_id") or license_data.get("name") or None
        else:
            merged["GitHub_git_license"] = None

        # --- Boolean and direct fields ---
        merged["GitHub_has_issues"] = repo_info.get("has_issues", None)
        merged["GitHub_has_wiki"] = repo_info.get("has_wiki", None)
        merged["GitHub_has_projects"] = repo_info.get("has_projects", None)
        merged["GitHub_has_discussions"] = repo_info.get("has_discussions", None)
        merged["GitHub_is_archived"] = repo_info.get("archived", None)
        merged["GitHub_is_mirrored"] = True if repo_info.get("mirror_url") else False

        merged["GitHub_visibility"] = repo_info.get("visibility") or merged["GitHub_visibility"]
        merged["GitHub_private"] = repo_info.get("private") if "private" in repo_info else merged["GitHub_private"]
        merged["GitHub_fork"] = repo_info.get("fork") if "fork" in repo_info else merged["GitHub_fork"]
        merged["GitHub_open_issues_count"] = repo_info.get("open_issues_count") or merged["GitHub_open_issues_count"]
        merged["GitHub_html_url"] = repo_info.get("html_url") or merged["GitHub_html_url"]
        merged["GitHub_git_url"] = repo_info.get("git_url") or merged["GitHub_git_url"]
        merged["GitHub_full_name"] = repo_info.get("full_name") or merged["GitHub_full_name"]
        merged["GitHub_description"] = clean_text(repo_info.get("description")) if repo_info.get("description") is not None else merged["GitHub_description"]
        merged["GitHub_default_branch"] = repo_info.get("default_branch") or merged["GitHub_default_branch"]
        merged["GitHub_created_at"] = repo_info.get("created_at") or merged["GitHub_created_at"]
        merged["GitHub_updated_at"] = repo_info.get("updated_at") or merged["GitHub_updated_at"]
        merged["GitHub_pushed_at"] = repo_info.get("pushed_at") or merged["GitHub_pushed_at"]

        # --- Numeric aggregation (using max) ---
        merged["GitHub_watchers_count"] = max(merged["GitHub_watchers_count"], repo_info.get("watchers_count", 0))
        merged["GitHub_subscribers_count"] = max(merged["GitHub_subscribers_count"], repo_info.get("subscribers_count", 0))
        merged["GitHub_stargazers_count"] = max(merged["GitHub_stargazers_count"], repo_info.get("stargazers_count", 0))
        merged["GitHub_size"] = max(merged["GitHub_size"], repo_info.get("size", 0))
        merged["GitHub_network_count"] = max(merged["GitHub_network_count"], repo_info.get("network_count", 0))
        merged["GitHub_forks_count"] = max(merged["GitHub_forks_count"], repo_info.get("forks_count", 0))

        # --- GitHub_git_number_of_followers ---
        # Try fetching owner followers via GET if possible; else use subscribers_count.
        owner = repo_info.get("owner")
        if owner and owner.get("followers_url"):
            followers_count = fetch_owner_followers(owner.get("followers_url"))
            if followers_count is not None:
                followers_list.append(followers_count)
        else:
            followers_list.append(repo_info.get("subscribers_count", 0))

        # --- Workflows and topics ---
        workflows = repo_data.get("workflow", {}).get("workflows", [])
        for wf in workflows:
            wf_name = wf.get("name")
            if wf_name:
                merged["GitHub_workflows_names"].append(wf_name)
        merged["GitHub_workflow_total_count"] += repo_data.get("workflow", {}).get("total_count", 0)

        topics = repo_info.get("topics", [])
        merged["GitHub_topics"].extend([t for t in topics if t])

        # --- Languages ---
        languages = repo_info.get("languages", {})
        for lang in languages.keys():
            if lang:
                merged["GitHub_languages"].append(lang)
        repo_language = repo_info.get("language")
        if repo_language:
            merged["GitHub_languages"].append(repo_language)

        # --- Process commits ---
        commits = repo_data.get("commits", [])
        total_commits += len(commits)
        for commit in commits:
            if isinstance(commit, dict):
                commit_info = commit.get("commit", {})
                verification = commit_info.get("verification", {})
                if verification.get("verified", False):
                    merged["GitHub_commits_verification_verified_count"] += 1
                reason = verification.get("reason")
                if reason:
                    merged["GitHub_commits_verification_reasons"].append(reason)
                message = commit_info.get("message")
                if message:
                    merged["GitHub_commits_messages"].append(message)
                merged["GitHub_commits_comment_count"] += commit.get("comment_count", 0)
                # Determine committer: try commit_info.committer.email, else commit.committer.login
                committer_email = None
                if commit_info.get("committer") and isinstance(commit_info.get("committer"), dict):
                    committer_email = commit_info.get("committer").get("email")
                if not committer_email and commit.get("committer") and isinstance(commit.get("committer"), dict):
                    committer_email = commit.get("committer").get("login")
                if committer_email:
                    committers_set.add(committer_email)
            else:
                print(f"Warning: Commit entry is not a dictionary: {commit}")

        # --- Process branches ---
        branches = repo_data.get("branches", [])
        for branch in branches:
            branch_name = None
            if isinstance(branch, dict):
                branch_name = branch.get("name")
            elif isinstance(branch, str):
                branch_name = branch
            if branch_name:
                branches_set.add(branch_name)
                merged["GitHub_branches_names"].append(branch_name)

        # --- Count contributors (unique logins) ---
        contributors = repo_data.get("contributors", [])
        if isinstance(contributors, list):
            for contributor in contributors:
                if isinstance(contributor, dict):
                    login = contributor.get("login")
                    if login:
                        authors_set.add(login)
                elif isinstance(contributor, str):
                    # If contributor is a string, assume it's the login.
                    authors_set.add(contributor)
            merged["GitHub_contributors_count"] += len(contributors)
    
    # Final aggregations:
    merged["GitHub_git_number_of_authors"] = len(authors_set) if authors_set else None
    merged["GitHub_git_number_of_committers"] = len(committers_set) if committers_set else None
    merged["GitHub_git_number_of_updates"] = total_commits if total_commits > 0 else None
    merged["GitHub_git_number_of_branches"] = len(dedup_list(list(branches_set))) if branches_set else None
    merged["GitHub_git_number_of_followers"] = max(followers_list) if followers_list else None

    # Deduplicate list fields
    merged["GitHub_topics"] = dedup_list(merged["GitHub_topics"])
    merged["GitHub_workflows_names"] = dedup_list(merged["GitHub_workflows_names"])
    merged["GitHub_commits_messages"] = dedup_list(merged["GitHub_commits_messages"])
    merged["GitHub_languages"] = dedup_list(merged["GitHub_languages"])
    merged["GitHub_branches_names"] = dedup_list(merged["GitHub_branches_names"])
    merged["GitHub_commits_verification_reasons"] = dedup_list(merged["GitHub_commits_verification_reasons"])

    return merged

def process_json_file(input_filepath, output_filepath):
    with open(input_filepath, "r", encoding="utf_8") as infile:
        try:
            data = json.load(infile)
        except Exception as e:
            print(f"Error reading {input_filepath}: {e}")
            return

    flat_data = DEFAULT_FLAT_DATA.copy()
    
    zenodo_meta = extract_zenodo_metadata(data)
    flat_data.update(zenodo_meta)
    
    github_meta = extract_github_metadata(data)
    flat_data.update(github_meta)
    
    # Write the flat JSON data directly to the output directory (no subdirectories)
    with open(output_filepath, "w", encoding="utf_8") as outfile:
        json.dump(flat_data, outfile, indent=2)

def traverse_and_process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".json"):
                input_filepath = os.path.join(root, file)
                # Write all output files directly into output_dir (ignoring subdirectory structure)
                output_filepath = os.path.join(output_dir, file)
                print(f"Processing {input_filepath} -> {output_filepath}")
                process_json_file(input_filepath, output_filepath)

def generate_csv(json_dir, csv_filepath):
    """Combine all JSON files in json_dir into a single CSV file."""
    all_data = []
    for root, dirs, files in os.walk(json_dir):
        for file in files:
            if file.lower().endswith(".json"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf_8") as f:
                        data = json.load(f)
                    all_data.append(data)
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
    if not all_data:
        print("No JSON files found in", json_dir)
        return

    header = set()
    for row in all_data:
        header.update(row.keys())
    header = list(header)
    header.sort()
    
    for row in all_data:
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                row[key] = json.dumps(value)

    try:
        with open(csv_filepath, "w", encoding="utf_8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            for row in all_data:
                writer.writerow(row)
        print(f"CSV file generated at: {csv_filepath}")
    except Exception as e:
        print(f"Error writing CSV file {csv_filepath}: {e}")

def main():
    # Update these paths as needed
    input_dir = r"C:\Users\User\Downloads\RSE-Analysis\RS"
    output_dir = r"C:\Users\User\Downloads\RSE-Analysis\RS_2.0"
    traverse_and_process(input_dir, output_dir)
    
    csv_filepath = os.path.join(output_dir, "all_data.csv")
    generate_csv(output_dir, csv_filepath)

if __name__ == "__main__":
    main()
