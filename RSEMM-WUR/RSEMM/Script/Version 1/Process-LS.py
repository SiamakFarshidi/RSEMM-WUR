import json
import difflib
from collections import OrderedDict

# Define the standard keys (in fixed order) for each group.
git_metadata_keys = [
    "branches_names", "commits_messages", "contributors_count", "description", "fork",
    "forks_count", "number_of_updates", "pushed_at", "updated_at", "number_of_authors",
    "number_of_branches", "number_of_committers", "has_discussions", "has_issues",
    "has_projects", "has_wiki", "is_archived", "Script language", "languages",
    "open_issues_count", "size", "stargazers_count", "subscribers/watchers_count",
    "workflow_total_count", "license"
]

repo_mining_keys = [
    "agile_keywords", "ai_keywords", "code_gen_keywords", "code_quality_keywords",
    "code_review_keywords", "dep_keywords", "devops_keywords", "documentation_keywords",
    "readme.md", "readme_length", "gen-ai-coding-files", "gen_ai_keywords",
    "security_keywords", "test_keywords", "requirements.txt", "used-ai-packages"
]

rs_metadata_keys = [
    "license", "communities", "modified_updated_date", "creators_affiliations",
    "creators_names", "creators_orcid", "description", "keywords", "resource_type",
    "revision", "stats_downloads", "stats_views", "files_count", "grant acquired",
    "git Repo Exist", "html_url", "doi_url"
]

# Optional custom mappings for keys that are not directly close in name.
custom_mapping = {
    "accessibility_keywords": "code_quality_keywords",
    # You can add additional custom mappings as needed.
    # For example:
    # "usability_keywords": "code_review_keywords"
}

def fuzzy_match_key(input_key, standard_keys, cutoff=0.75):
    """
    Return the best matching key from the list of standard_keys using fuzzy matching.
    The function normalizes keys by removing underscores and dashes then compares.
    If a custom mapping exists for the input_key, that mapping is used directly.
    """
    if input_key in custom_mapping:
        return custom_mapping[input_key]
    
    processed_input = input_key.lower().replace("_", "").replace("-", "")
    best_match = None
    best_score = 0
    for std in standard_keys:
        processed_std = std.lower().replace("_", "").replace("-", "")
        score = difflib.SequenceMatcher(None, processed_input, processed_std).ratio()
        if score > best_score:
            best_score = score
            best_match = std
    return best_match if best_score >= cutoff else None

def transform_record(record):
    """
    Transform a single record so that:
      - Fixed keys (valid, title, year, repo_number, venue, focus) are preserved.
      - All three groups (git, repo, rs) are present in the output using a fixed order.
      - For groups relevant to the record's 'focus', fuzzy matching is used to map values 
        from the input 'factors'. For other groups, the keys are set to "N/A".
    The output is an OrderedDict that preserves the order.
    """
    out_rec = OrderedDict()
    
    # Fixed keys, always first.
    fixed_keys = ["valid", "title", "year", "repo_number", "venue", "focus"]
    for key in fixed_keys:
        out_rec[key] = record.get(key, "N/A")
    
    # Define all groups.
    groups = {
        "git": git_metadata_keys,
        "repo": repo_mining_keys,
        "rs": rs_metadata_keys
    }
    
    # Determine allowed groups based on the focus.
    focus = record.get("focus", "").lower()
    if focus == "github":
        allowed_groups = {"git", "repo"}
    elif focus == "zenodo":
        allowed_groups = {"rs"}
    elif focus == "both":
        allowed_groups = {"git", "repo", "rs"}
    else:
        allowed_groups = set()  # No groups are relevant if focus is unrecognized.
    
    # Prepare a dictionary to hold the mapped factor values for each group.
    group_data = {"git": {}, "repo": {}, "rs": {}}
    factors = record.get("factors", {})
    
    # For each factor, check the allowed groups and try to assign the factor's value.
    # (Note: We simply assign the value so that each cell contains only one value.)
    for factor_key, factor_value in factors.items():
        for group_name in allowed_groups:
            std_keys = groups[group_name]
            matched_std = fuzzy_match_key(factor_key, std_keys)
            if matched_std:
                group_data[group_name][matched_std] = factor_value
    
    # Build the output record with a fixed order.
    # The order is: Git Metadata (prefix "git_"), then Repo Mining ("repo_"), then RS Metadata ("rs_").
    for prefix in ["git", "repo", "rs"]:
        for std in groups[prefix]:
            full_key = f"{prefix}_{std}"
            out_rec[full_key] = group_data[prefix].get(std, "N/A")
                
    return out_rec

def transform_json(input_json):
    """
    Given a list of input records, transform each record with the full ordered structure.
    Returns a list of OrderedDict instances.
    """
    return [transform_record(rec) for rec in input_json]

def main():
    # Read the input JSON from 'LS.json'
    with open('LS.json', 'r') as infile:
        input_data = json.load(infile)
    
    # Transform the data.
    transformed_data = transform_json(input_data)
    
    # Write the transformed data to 'output.json'.
    # Python 3.7+ maintains insertion order when using json.dump.
    with open('output.json', 'w') as outfile:
        json.dump(transformed_data, outfile, indent=2)

if __name__ == "__main__":
    main()
