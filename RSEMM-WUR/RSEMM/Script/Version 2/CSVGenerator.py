import os
import json
import csv
from pathlib import Path
import re

# 1. Define the directory containing your JSON files:
JSON_DIR = Path(r"C:\Users\User\Downloads\RS-Zenodo\RS-Repo-Zenodo - v4")

# 2. Define the exact CSV columns you want, in order:
COLUMNS = [
    "created", "modified", "id", "conceptrecid", "doi", "conceptdoi", "doi_url",
    "metadata.title", "metadata.doi", "metadata.publication_date", "metadata.description",
    "metadata.access_right",
    "creators.names", "creators.affiliations",
    "metadata.keywords", "metadata.version", "language",
    "metadata.resource_type.title", "metadata.resource_type.type",
    "metadata.license.id",
    "metadata.communities", "metadata.notes", "metadata.query_terms",
    "grants", "grant.funder.names", "grant.funder.acronym",
    "grant.titles", "grant.acronym", "grant.program",
    "communities.ids",
    "title", "updated", "recid", "revision", "swh", "status",
    "stats.downloads", "stats.unique_downloads", "stats.views", "stats.unique_views",
    "stats.version_downloads", "stats.version_unique_downloads",
    "stats.version_unique_views", "stats.version_views",
    "state", "submitted",
    # repo-profile.software-engineering-efforts.*
    "repo-profile.software-engineering-efforts.repo",
    "repo-profile.software-engineering-efforts.dimensions.community.actual",
    "repo-profile.software-engineering-efforts.dimensions.community.estimation",
    "repo-profile.software-engineering-efforts.dimensions.ci.actual",
    "repo-profile.software-engineering-efforts.dimensions.ci.estimation",
    "repo-profile.software-engineering-efforts.dimensions.documentation.actual",
    "repo-profile.software-engineering-efforts.dimensions.documentation.estimation",
    "repo-profile.software-engineering-efforts.dimensions.history.actual",
    "repo-profile.software-engineering-efforts.dimensions.history.estimation",
    "repo-profile.software-engineering-efforts.dimensions.issues.actual",
    "repo-profile.software-engineering-efforts.dimensions.issues.estimation",
    "repo-profile.software-engineering-efforts.dimensions.license.actual",
    "repo-profile.software-engineering-efforts.dimensions.license.estimation",
    "repo-profile.software-engineering-efforts.dimensions.unittest.actual",
    "repo-profile.software-engineering-efforts.dimensions.unittest.estimation",
    "repo-profile.software-engineering-efforts.raw.total_sloc",
    "repo-profile.software-engineering-efforts.raw.total_cloc",
    "repo-profile.software-engineering-efforts.raw.test_sloc",
    "repo-profile.software-engineering-efforts.raw.total_commits",
    "repo-profile.software-engineering-efforts.raw.first_commit_date",
    "repo-profile.software-engineering-efforts.raw.last_commit_date",
    "repo-profile.software-engineering-efforts.raw.duration_months",
    "repo-profile.software-engineering-efforts.raw.issue_events",
    # ai-ml-ops
    "repo-profile.ai-ml-ops.repository",
    "repo-profile.ai-ml-ops.category",
    "repo-profile.ai-ml-ops.types",
    "repo-profile.ai-ml-ops.name",
    "repo-profile.ai-ml-ops.description",
    "repo-profile.ai-ml-ops.topics",
    "repo-profile.ai-ml-ops.timestamp",
    # code-gen-evaluation
    "repo-profile.code-gen-evaluation.AllFiles",
    "repo-profile.code-gen-evaluation.SelectedRecentFiles",
    "repo-profile.code-gen-evaluation.ai_ratio",
    "repo-profile.code-gen-evaluation.estimation",
    "repo-profile.code-gen-evaluation.analyzed-scripts",
    # FAIR assessment
    "fair-assessment.subprinciple_assessment.doi_present",
    "fair-assessment.subprinciple_assessment.subcomponent_identifiers",
    "fair-assessment.subprinciple_assessment.version_identifiers",
    "fair-assessment.subprinciple_assessment.rich_metadata_present",
    "fair-assessment.subprinciple_assessment.metadata_includes_doi",
    "fair-assessment.subprinciple_assessment.metadata_accessible",
    "fair-assessment.subprinciple_assessment.software_retrievable",
    "fair-assessment.subprinciple_assessment.authentication_required",
    "fair-assessment.subprinciple_assessment.metadata_persistent",
    "fair-assessment.subprinciple_assessment.uses_standard_data_formats",
    "fair-assessment.subprinciple_assessment.qualified_references_to_objects",
    "fair-assessment.subprinciple_assessment.clear_license",
    "fair-assessment.subprinciple_assessment.provenance_with_orcid",
    "fair-assessment.subprinciple_assessment.qualified_references_to_software",
    "fair-assessment.subprinciple_assessment.ci_or_community_standards",
    "fair-assessment.principle_categories.Findable",
    "fair-assessment.principle_categories.Accessible",
    "fair-assessment.principle_categories.Interoperable",
    "fair-assessment.principle_categories.Reusable",
    "fair-assessment.overall_fairness",
    "fair-assessment.git_url",
    "fair-assessment.howfairis_score",
    "fair-assessment.howfairis_details.repository",
    "fair-assessment.howfairis_details.license",
    "fair-assessment.howfairis_details.registry",
    "fair-assessment.howfairis_details.citation",
    "fair-assessment.howfairis_details.quality",
        # ---- GitHub metadata ----
    "Github-metadata.GitHub_full_name",
    "Github-metadata.GitHub_description",
    "Github-metadata.GitHub_fork",
    "Github-metadata.GitHub_forks_count",
    "Github-metadata.GitHub_git_url",
    "Github-metadata.GitHub_created_at",
    "Github-metadata.GitHub_updated_at",
    "Github-metadata.GitHub_pushed_at",
    "Github-metadata.GitHub_size",
    "Github-metadata.GitHub_stargazers_count",
    "Github-metadata.GitHub_watchers_count",
    "Github-metadata.GitHub_network_count",
    "Github-metadata.GitHub_open_issues_count",
    "Github-metadata.GitHub_default_branch",
    "Github-metadata.GitHub_private",
    "Github-metadata.GitHub_is_archived",
    "Github-metadata.GitHub_is_mirrored",
    "Github-metadata.GitHub_has_issues",
    "Github-metadata.GitHub_has_projects",
    "Github-metadata.GitHub_has_wiki",
    "Github-metadata.GitHub_has_discussions",
    "Github-metadata.GitHub_visibility",
    "Github-metadata.GitHub_git_license",
    "Github-metadata.GitHub_languages",
    "Github-metadata.GitHub_topics",
    "Github-metadata.GitHub_workflow_total_count",
    "Github-metadata.GitHub_workflows_names",
    "Github-metadata.GitHub_has_README",
    "Github-metadata.GitHub_branches_names",
    "Github-metadata.GitHub_git_number_of_branches",
    "Github-metadata.GitHub_contributors_count",
    "Github-metadata.GitHub_git_number_of_updates",
    "Github-metadata.GitHub_commits_messages",
    "Github-metadata.GitHub_commits_verification_reasons",
    "Github-metadata.GitHub_commits_verification_verified_count",
    "Github-metadata.GitHub_git_number_of_authors",
    "Github-metadata.GitHub_git_number_of_committers",
]

def get_nested(data, path, default=""):
    """Safely get a nested value, where path is like 'metadata.title'."""
    for key in path.split("."):
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data

def list_to_csv(l):
    """Convert list to comma-joined string, or return empty string."""
    if isinstance(l, list):
        return ", ".join(str(item) for item in l)
    return ""


def strip_html(text: str) -> str:
    """Remove HTML tags, commas, semicolons, line breaks, and collapse whitespace."""
    # 1) strip out any HTML tags
    no_tags = re.sub(r'<[^>]+>', '', text or "")

    # 2) remove commas and semicolons
    no_punct = re.sub(r'[;,]', '', no_tags)

    # 3) remove all kinds of line breaks (CR/LF) so we don't get stray newlines in CSV fields
    no_breaks = re.sub(r'[\r\n]+', ' ', no_punct)

    # 4) collapse any runs of whitespace into a single space, then trim
    return re.sub(r'\s+', ' ', no_breaks).strip()

def sanitize(val: str) -> str:
    # flatten to string, strip HTML (if you like), then remove commas, semicolons, quotes, newlines
    s = strip_html(str(val))
    return re.sub(r'[,\;"\r\n]+', ' ', s).strip()



def extract_row(json_data):
    row = {}
    # top-level and simple nested fields
    for col in COLUMNS:
        if col in ("creators.names", "creators.affiliations"):
            # special handling below
            continue
        if col.startswith("grant."):
            # special handling below
            continue
        if col.startswith("metadata.") or col.startswith("stats.") \
           or col.startswith("repo-profile.") or col.startswith("fair-assessment."):
            row[col] = get_nested(json_data, col, "")
        else:
            row[col] = json_data.get(col, "")

    # --- NOW OVERRIDE description & notes with HTML-stripped versions ---
    raw_desc  = get_nested(json_data, "metadata.description", "")
    row["metadata.description"] = strip_html(raw_desc)

    raw_notes = get_nested(json_data, "metadata.notes", "")
    row["metadata.notes"] = strip_html(raw_notes)

    # creators: names & affiliations
    creators = json_data.get("metadata", {}).get("creators", [])
    names = [c.get("name", "") for c in creators]
    affs  = [c.get("affiliation", "") for c in creators]
    row["creators.names"] = list_to_csv(names)
    row["creators.affiliations"] = list_to_csv(affs)
    # keywords, communities, query_terms
    row["metadata.keywords"]     = list_to_csv(get_nested(json_data, "metadata.keywords", []))
    row["metadata.communities"]  = list_to_csv([c.get("id","") for c in get_nested(json_data, "metadata.communities", [])])
    row["metadata.query_terms"]  = list_to_csv(get_nested(json_data, "metadata.query_term", []))
    # grants
    grants = get_nested(json_data, "metadata.grants", [])
    row["grants"] = len(grants)
    row["grant.funder.names"]    = list_to_csv([g.get("funder",{}).get("name","") for g in grants])
    row["grant.funder.acronym"]  = list_to_csv([g.get("funder",{}).get("acronym","") for g in grants])
    row["grant.titles"]          = list_to_csv([g.get("title","") for g in grants])
    row["grant.acronym"]         = list_to_csv([g.get("acronym","") for g in grants])
    row["grant.program"]         = list_to_csv([g.get("program","") for g in grants])
    # communities.ids (from top-level communities)
    comms = json_data.get("metadata", {}).get("communities", [])
    row["communities.ids"] = list_to_csv([c.get("id","") for c in comms])


        # --- pull in GitHub-metadata ---
    gh = json_data.get("Github-metadata", {})

    # simple scalars
    row["Github-metadata.GitHub_full_name"]                = gh.get("GitHub_full_name", "")
    row["Github-metadata.GitHub_description"]              = strip_html(gh.get("GitHub_description", ""))
    row["Github-metadata.GitHub_fork"]                     = gh.get("GitHub_fork", "")
    row["Github-metadata.GitHub_forks_count"]              = gh.get("GitHub_forks_count", "")
    row["Github-metadata.GitHub_git_url"]                  = gh.get("GitHub_git_url", "")
    row["Github-metadata.GitHub_created_at"]               = gh.get("GitHub_created_at", "")
    row["Github-metadata.GitHub_updated_at"]               = gh.get("GitHub_updated_at", "")
    row["Github-metadata.GitHub_pushed_at"]                = gh.get("GitHub_pushed_at", "")
    row["Github-metadata.GitHub_size"]                     = gh.get("GitHub_size", "")
    row["Github-metadata.GitHub_stargazers_count"]         = gh.get("GitHub_stargazers_count", "")
    row["Github-metadata.GitHub_watchers_count"]           = gh.get("GitHub_watchers_count", "")
    row["Github-metadata.GitHub_network_count"]            = gh.get("GitHub_network_count", "")
    row["Github-metadata.GitHub_open_issues_count"]        = gh.get("GitHub_open_issues_count", "")
    row["Github-metadata.GitHub_default_branch"]           = gh.get("GitHub_default_branch", "")
    row["Github-metadata.GitHub_private"]                  = gh.get("GitHub_private", "")
    row["Github-metadata.GitHub_is_archived"]              = gh.get("GitHub_is_archived", "")
    row["Github-metadata.GitHub_is_mirrored"]              = gh.get("GitHub_is_mirrored", "")
    row["Github-metadata.GitHub_has_issues"]               = gh.get("GitHub_has_issues", "")
    row["Github-metadata.GitHub_has_projects"]             = gh.get("GitHub_has_projects", "")
    row["Github-metadata.GitHub_has_wiki"]                 = gh.get("GitHub_has_wiki", "")
    row["Github-metadata.GitHub_has_discussions"]          = gh.get("GitHub_has_discussions", "")
    row["Github-metadata.GitHub_visibility"]               = gh.get("GitHub_visibility", "")
    row["Github-metadata.GitHub_git_license"]              = gh.get("GitHub_git_license", "")
    row["Github-metadata.GitHub_workflow_total_count"]     = gh.get("GitHub_workflow_total_count", "")
    row["Github-metadata.GitHub_has_README"]               = gh.get("GitHub_has_README", "")
    row["Github-metadata.GitHub_git_number_of_branches"]   = gh.get("GitHub_git_number_of_branches", "")
    row["Github-metadata.GitHub_contributors_count"]       = gh.get("GitHub_contributors_count", "")
    row["Github-metadata.GitHub_git_number_of_updates"]    = gh.get("GitHub_git_number_of_updates", "")
    row["Github-metadata.GitHub_git_number_of_authors"]    = gh.get("GitHub_git_number_of_authors", "")
    row["Github-metadata.GitHub_git_number_of_committers"] = gh.get("GitHub_git_number_of_committers", "")

    # list‐valued fields → comma‐separated
    row["Github-metadata.GitHub_languages"]                 = list_to_csv(gh.get("GitHub_languages", []))
    row["Github-metadata.GitHub_topics"]                    = list_to_csv(gh.get("GitHub_topics", []))
    row["Github-metadata.GitHub_workflows_names"]           = list_to_csv(gh.get("GitHub_workflows_names", []))
    row["Github-metadata.GitHub_branches_names"]            = list_to_csv(gh.get("GitHub_branches_names", []))
    row["Github-metadata.GitHub_commits_messages"]          = len( strip_html (list_to_csv(gh.get("GitHub_commits_messages", []))))
    row["Github-metadata.GitHub_commits_verification_reasons"] = list_to_csv(gh.get("GitHub_commits_verification_reasons", []))


    return row

def main():
    rows = []
    errors = []
    for p in JSON_DIR.rglob("*.json"):
        try:
            data = json.loads(p.read_text())
            print("one row added!")
            rows.append(extract_row(data))
        except Exception as e:
            errors.append(f"{p}: {e}")

    # write CSV
    with open("output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # report any errors
    if errors:
        print("Some files could not be processed:")
        for err in errors:
            print(" ", err)
    else:
        print("All files processed successfully. CSV written to output.csv")

if __name__ == "__main__":
    main()
