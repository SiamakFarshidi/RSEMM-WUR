import os
import json
import re
from bs4 import BeautifulSoup  # pip install beautifulsoup4
from rapidfuzz import fuzz


# --- RSMM Model Definition (unchanged) ---
#RSMM_MODEL = from the json

# --- RSMM Heuristics Mapping (unchanged) ---
RSMM_HEURISTICS_MAPPING_WEIGHTED = {
    "File issues in an issue tracker": lambda d: score_flag(d.get("has_issues", False)),
    "Act on feedback": lambda d: score_flag(fuzzy_contains(normalize_text(d.get("description", "")), "feedback"), weight=1),
    "Manage requirements explicitly": lambda d: score_flag(fuzzy_contains(normalize_text(d.get("description", "")), "requirements"), weight=1),
    "Perform release management": lambda d: score_flag("release" in normalize_text(d.get("description", "")), weight=1),
    "Communicate roadmap": lambda d: score_flag("roadmap" in normalize_text(d.get("description", "")), weight=1),
    "Provide executable tests": lambda d: score_flag(d.get("testing", False)),
    "Use crash reporting": lambda d: score_flag(fuzzy_contains(normalize_text(d.get("git_repo_desc", "")), "crash"), weight=1),
    "Conduct security reviews": lambda d: score_flag(d.get("security_practices", False)),
    "Define code coverage targets": lambda d: score_flag(d.get("code_quality", False)),
    "Execute tests in a public workflow": lambda d: score_flag(d.get("ci_cd_present", False)),
    "Follow an industry standard for security": lambda d: score_flag(d.get("security_practices", False)),
    "Store project in public repository": lambda d: score_flag(bool(d.get("git_repo_html_url", ""))),
    "Use public communication platform": lambda d: score_flag(d.get("has_discussions", False)),
    "Provide newsletter": lambda d: 0,  # Not implemented yet
    "Provide community website": lambda d: score_flag(d.get("documentation_practices", False)),
    "Define a clear audience": lambda d: score_flag("audience" in normalize_text(d.get("description", ""))),
    "Perform infrequent impact measurement": lambda d: 0,
    "Evaluate if audience's goals are met": lambda d: 0,
    "Perform continuous impact measurement": lambda d: 0,
    "Write software management plan": lambda d: 0,
    "Acquire viable sustainability pathways": lambda d: 0,
    "Secure continuous funding": lambda d: 1 if d.get("zenodo_num_funding_entries", 0) > 0 else 0,
    "Define end-of-life policy": lambda d: 0,
    "Make code citable": lambda d: 1 if any("doi" in kw.lower() for kw in d.get("zenodo_keywords", [])) else 0,
    "Enable indexing of project meta-data": lambda d: 1 if d.get("zenodo_related_publications", []) else 0,
    "Publish in a research software directory": lambda d: score_flag("directory" in normalize_text(d.get("description", ""))),
    "Document the cost of running the application": lambda d: score_flag("cost" in normalize_text(d.get("description", ""))),
    "Consider total energy consumption": lambda d: score_flag("energy" in normalize_text(d.get("description", ""))),
    "Acknowledge partners and funding agencies": lambda d: 1 if len(d.get("grants", [])) > 0 else 0,
    "Develop advanced partnership model": lambda d: 0,
    "Onboard researchers as part of the community": lambda d: 1 if len(d.get("Creators", [])) > 1 else 0,
    "Impose community norms": lambda d: score_flag("code of conduct" in normalize_text(d.get("git_repo_desc", ""))),
    "Develop code of conduct": lambda d: score_flag("code of conduct" in normalize_text(d.get("git_repo_desc", ""))),
    "Organize community events": lambda d: 0,
    "Document how to join the team": lambda d: score_flag(d.get("documentation_practices", False)),
    "Make developer names publicly available": lambda d: score_flag(bool(d.get("git_authors", []))),
    "Provide developer training": lambda d: 0,
    "Select a license": lambda d: score_flag(bool(d.get("license", {}))),
    "Evaluate license policy regularly": lambda d: 0,
    "Provide a statement of purpose": lambda d: score_flag("purpose" in normalize_text(d.get("description", ""))),
    "Provide a simple how-to use guide": lambda d: score_flag(any(word in normalize_text(d.get("description", "")) for word in SYNONYMS.get("how to", []))),
    "Provide online tutorials": lambda d: score_flag("tutorial" in normalize_text(d.get("description", ""))),
    "Provide a readme file with project explanation": lambda d: score_flag(any("readme" in s.lower() for s in d.get("zenodo_keywords", [])) or "readme" in normalize_text(d.get("description", ""))),
    "Provide a how-to guide": lambda d: score_flag(any(word in normalize_text(d.get("description", "")) for word in SYNONYMS.get("how to", []))),
    "Provide common example usage": lambda d: score_flag("example" in normalize_text(d.get("description", ""))),
    "Provide API documentation": lambda d: score_flag("api" in normalize_text(d.get("description", ""))),
    "Use common non-exotic technology": lambda d: score_flag(bool(d.get("language", []))),
    "Facilitate integration into scientific workflow": lambda d: score_flag("workflow" in normalize_text(d.get("description", ""))),
    "Provide instructions on how to put into research workflow": lambda d: score_flag("workflow" in normalize_text(d.get("description", ""))),
    "Provide instructions on how to make part of a replication package": lambda d: score_flag("replication" in normalize_text(d.get("description", ""))),
    "Develop generic educational materials": lambda d: score_flag("educational" in normalize_text(d.get("description", ""))),
    "Organize training events": lambda d: score_flag("training" in normalize_text(d.get("description", ""))),
    "Provide standard deployment tools": lambda d: score_flag(d.get("devops_practices", False)),
    "Enable deployment on a wide range of technology": lambda d: score_flag(d.get("devops_practices", False)),
    "Generate SBOM": lambda d: score_flag("sbom" in normalize_text(d.get("description", "")))
}


# A simple synonym dictionary for demonstration.
SYNONYMS = {
    "how to": ["how to", "tutorial", "guide", "instructions"],
    "readme": ["readme", "project explanation"],
    "crash": ["crash", "crash reporting", "error report"],
    "code of conduct": ["code of conduct", "community norms"],
    "workflow": ["workflow", "ci", "cd", "continuous integration", "continuous delivery"]
}

def contains_keyword_with_synonyms(text, canonical):
    """
    Check if the normalized text contains the canonical keyword or any of its synonyms.
    """
    synonyms = SYNONYMS.get(canonical, [canonical])
    for word in synonyms:
        if word in text:
            return True
    return False


def evaluate_rsmm_maturity_weighted(dashboard, rsmm_model):
    """
    Evaluate RSMM maturity by aggregating weighted scores from the heuristics.
    For each focus area, we multiply the required level by the heuristic score
    and then average the values.
    """
    rsmm_profile = {}
    for focus_area, capabilities in rsmm_model.items():
        cap_levels = []
        for cap_name, cap_data in capabilities.items():
            practice_levels = []
            for practice in cap_data.get("practices", []):
                pname = practice.get("name")
                required_level = practice.get("level", 0)
                # Get the heuristic weighted score; default to 0 if not implemented.
                score = RSMM_HEURISTICS_MAPPING_WEIGHTED.get(pname, lambda d: 0)(dashboard)
                if score:
                    practice_levels.append(required_level * score)
            cap_level = round(sum(practice_levels) / len(practice_levels)) if practice_levels else 0
            cap_levels.append(cap_level)
        focus_area_level = round(sum(cap_levels) / len(cap_levels)) if cap_levels else 0
        rsmm_profile[focus_area] = focus_area_level
    return rsmm_profile

# --- Example Integration --


def normalize_text(text):
    """
    Strip HTML, convert to lowercase, and remove punctuation.
    """
    # Strip HTML tags
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def fuzzy_contains(text, keyword, threshold=80):
    """
    Return True if the fuzzy partial ratio between the keyword and text is at least the threshold.
    """
    score = fuzz.partial_ratio(keyword, text)
    return score >= threshold

def score_flag(flag, weight=1):
    """
    Return a weighted score if flag is True; otherwise 0.
    """
    return weight if flag else 0

# --- Utility Functions ---
def strip_html(html_text):
    """Return plain text by stripping HTML tags."""
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def extract_zenodo_data(data):
    meta = data.get("metadata", data)
    version = meta.get("version", "")
    versions = [version] if version else []
    raw_desc = meta.get("description", "")
    plain_desc = strip_html(raw_desc)
    total_size = sum(f.get("size", 0) for f in data.get("files", []))
    stats = data.get("stats", {})
    package_name = meta.get("title", "").split()[0] if meta.get("title") else ""
    
    # Extract programming language from custom field if available
    programming_language = []
    custom = meta.get("custom", {})
    if custom and "code:programmingLanguage" in custom:
        langs = custom["code:programmingLanguage"]
        if isinstance(langs, list):
            for lang in langs:
                title = lang.get("title", {}).get("en")
                if title:
                    programming_language.append(title)
        elif isinstance(langs, dict):
            title = langs.get("title", {}).get("en")
            if title:
                programming_language.append(title)
        else:
            programming_language = [langs]
    
    # Extract venue from meeting field if available
    venue = None
    meeting = meta.get("meeting", {})
    if meeting:
        venue = meeting.get("title")
    
    zenodo = {
        "title": meta.get("title", ""),
        "doi": meta.get("doi", data.get("doi", "")),
        "doi_url": meta.get("doi_url", data.get("doi_url", "")),
        "publication_date": meta.get("publication_date", ""),
        "description": plain_desc,
        "access_right": meta.get("access_right", ""),
        "Creators": meta.get("creators", []),
        "versions": versions,
        "license": meta.get("license", {}),
        "grants": meta.get("grants", []),
        "communities": meta.get("communities", []),
        "revision": data.get("revision", ""),
        "size": total_size,
        "downloads": stats.get("downloads", 0),
        "views": stats.get("views", 0),
        "keywords": meta.get("keywords", []),
        "package_name": package_name,
        "version": version,
        "related_identifiers": meta.get("related_identifiers", []),
        "programming_language": programming_language,
        "venue": venue
    }
    zenodo["zenodo_num_funding_entries"] = len(zenodo.get("grants", []))
    return zenodo

def extract_github_data(data):
    gh_data = {}
    github_repos = data.get("Github Repo", {})
    if github_repos:
        repo_key, repo_data = next(iter(github_repos.items()))
        repo_info = repo_data.get("repo_info", {})
        languages = list((repo_data.get("languages") or {}).keys())
        contributors = repo_data.get("contributors", [])
        commits = repo_data.get("commits", [])
        branches = repo_data.get("branches", [])
        
        committers = { (c.get("author") or {}).get("login")
                       for c in commits
                       if isinstance(c, dict) and c.get("author") and isinstance(c.get("author"), dict) and c.get("author").get("login") }
        authors = { (c.get("commit") or {}).get("author", {}).get("name")
                    for c in commits
                    if isinstance(c, dict) and c.get("commit") and isinstance(c.get("commit"), dict) and 
                       c.get("commit").get("author") and isinstance(c.get("commit").get("author"), dict) and 
                       c.get("commit").get("author").get("name") }
        
        all_topics = repo_info.get("topics", [])
        gh_desc = repo_info.get("description", "") or ""
        gh_data = {
            "git_repo_name": repo_info.get("name", ""),
            "git_repo_full_name": repo_info.get("full_name", ""),
            "git_repo_html_url": repo_info.get("html_url", ""),
            "git_repo_desc": gh_desc,
            "git_repo_size": repo_info.get("size", 0),
            "stargazers_count": repo_info.get("stargazers_count", 0),
            "watchers_count": repo_info.get("watchers_count", 0),
            "language": languages,
            "has_issues": repo_info.get("has_issues", False),
            "has_projects": repo_info.get("has_projects", False),
            "has_downloads": repo_info.get("has_downloads", False),
            "has_wiki": repo_info.get("has_wiki", False),
            "has_pages": repo_info.get("has_pages", False),
            "has_discussions": repo_info.get("has_discussions", False),
            "forks_count": repo_info.get("forks_count", 0),
            "mirror_url": repo_info.get("mirror_url", ""),
            "archived": repo_info.get("archived", False),
            "open_issues_count": repo_info.get("open_issues_count", 0),
            "git_license": (repo_info.get("license") or {}).get("spdx_id", ""),
            "topics": all_topics,
            "visibility": repo_info.get("visibility", ""),
            "git_contributor_count": len(contributors),
            "git_contributors_ids": [c.get("id") for c in contributors if isinstance(c, dict) and c.get("id")],
            "git_number_of_commits": len(commits),
            "git_number_of_committers": len(committers),
            "CVE_codes": [],
            "git_number_of_followers": repo_info.get("watchers_count", 0),
            "git_number_of_updates": len(commits),
            "git_number_of_authors": len(authors),
            "git_authors": list(authors),
            "git_number_of_branches": len(branches),
            "ci_cd_present": repo_data.get("workflow", {}).get("total_count", 0) > 0,
            "last_update": repo_info.get("updated_at", "")
        }
    return gh_data

def evaluate_rsmm_maturity(dashboard, rsmm_model):
    """
    Evaluate RSMM maturity based on the RSMM model and dashboard indicators.
    """
    rsmm_profile = {}
    for focus_area, capabilities in rsmm_model.items():
        cap_levels = []
        for cap_name, cap_data in capabilities.items():
            practice_levels = []
            for practice in cap_data.get("practices", []):
                pname = practice.get("name")
                required_level = practice.get("level", 0)
                is_fulfilled = RSMM_HEURISTICS_MAPPING.get(pname, lambda d: False)(dashboard)
                if is_fulfilled:
                    practice_levels.append(required_level)
            cap_level = max(practice_levels) if practice_levels else 0
            cap_levels.append(cap_level)
        focus_area_level = round(sum(cap_levels) / len(cap_levels)) if cap_levels else 0
        rsmm_profile[focus_area] = focus_area_level
    return rsmm_profile

def evaluate_rs_category(dashboard):
    """
    Determine the RS category based on dashboard indicators.
    """
    robust_indicators = [
         "ci_cd_present",
         "testing",
         "dependency_management",
         "documentation_practices",
         "devops_practices",
         "security_practices",
         "code_quality"
    ]
    robust_count = sum(1 for key in robust_indicators if dashboard.get(key))
    ai_in_se = dashboard.get("ai_in_se", False)
    
    if robust_count >= 4:
        return "AI in RSE" if ai_in_se else "Research Software Engineering (RSE)"
    else:
        return "AI-Assisted Exploratory Coding" if ai_in_se else "Exploratory Coding"

def compute_heuristics(zenodo, gh_data):
    """
    Compute additional heuristics from Zenodo and GitHub data.
    """
    zenodo_keywords = [kw.lower() for kw in zenodo.get("keywords", [])]
    if not zenodo_keywords:
        zenodo_keywords = re.findall(r'\w+', zenodo.get("description", "").lower())
    indicators = {}
    indicators["testing"] = any(any(tk in kw for tk in test_keywords) for kw in zenodo_keywords)
    indicators["dependency_management"] = any(kw in zenodo.get("description", "").lower() for kw in dep_keywords) or \
                                           any(any(kw in k for kw in dep_keywords) for k in zenodo_keywords)
    indicators["ai_in_se"] = any(any(akw in kw for akw in ai_keywords) for kw in zenodo_keywords) or \
                              any(akw in zenodo.get("description", "").lower() for akw in ai_keywords)
    indicators["documentation_practices"] = any(kw in gh_data.get("git_repo_desc", "").lower() for kw in documentation_keywords)
    gh_topics = [t.lower() for t in gh_data.get("topics", [])]
    gh_languages = [l.lower() for l in gh_data.get("language", [])]
    gh_repo_desc = gh_data.get("git_repo_desc", "").lower() if gh_data.get("git_repo_desc") else ""
    indicators["devops_practices"] = any(kw in gh_topics for kw in devops_keywords) or \
                                      any(kw in gh_languages for kw in devops_keywords) or \
                                      any(kw in gh_repo_desc for kw in devops_keywords)
    indicators["security_practices"] = any(kw in gh_topics for kw in security_keywords) or \
                                         any(kw in gh_repo_desc for kw in security_keywords)
    indicators["code_quality"] = any(kw in gh_topics for kw in code_quality_keywords) or \
                                 any(kw in gh_repo_desc for kw in code_quality_keywords)
    # You can extend similar checks for agile and code review if needed.
    return indicators

def build_dashboard_analysis(zenodo, gh_data):
    """
    Build the dashboard_analysis field based on Zenodo and GitHub data.
    """
    dashboard = {}
    # Zenodo fields
    dashboard["title"] = zenodo.get("title", "")
    dashboard["doi"] = zenodo.get("doi", "")
    dashboard["doi_url"] = zenodo.get("doi_url", "")
    dashboard["publication_date"] = zenodo.get("publication_date", "")
    dashboard["description"] = zenodo.get("description", "")
    dashboard["access_right"] = zenodo.get("access_right", "")
    dashboard["Creators"] = zenodo.get("Creators", [])
    dashboard["versions"] = zenodo.get("versions", [])
    dashboard["license"] = zenodo.get("license", {})
    dashboard["grants"] = zenodo.get("grants", [])
    dashboard["communities"] = zenodo.get("communities", [])
    dashboard["revision"] = zenodo.get("revision", "")
    dashboard["size"] = zenodo.get("size", 0)
    dashboard["downloads"] = zenodo.get("downloads", 0)
    dashboard["views"] = zenodo.get("views", 0)
    
    # GitHub fields
    dashboard["git_repo_name"] = gh_data.get("git_repo_name", "")
    dashboard["git_repo_html_url"] = gh_data.get("git_repo_html_url", "")
    dashboard["git_repo_size"] = gh_data.get("git_repo_size", 0)
    dashboard["stargazers_count"] = gh_data.get("stargazers_count", 0)
    dashboard["watchers_count"] = gh_data.get("watchers_count", 0)
    # Use GitHub language if available; otherwise, fall back to Zenodo programming language
    gh_language = gh_data.get("language", [])
    if not gh_language:
        gh_language = zenodo.get("programming_language", [])
    dashboard["language"] = gh_language
    dashboard["has_issues"] = gh_data.get("has_issues", False)
    dashboard["has_projects"] = gh_data.get("has_projects", False)
    dashboard["has_downloads"] = gh_data.get("has_downloads", False)
    dashboard["has_wiki"] = gh_data.get("has_wiki", False)
    dashboard["has_pages"] = gh_data.get("has_pages", False)
    dashboard["has_discussions"] = gh_data.get("has_discussions", False)
    dashboard["forks_count"] = gh_data.get("forks_count", 0)
    dashboard["mirror_url"] = gh_data.get("mirror_url", "")
    dashboard["archived"] = gh_data.get("archived", False)
    dashboard["open_issues_count"] = gh_data.get("open_issues_count", 0)
    dashboard["git_license"] = gh_data.get("git_license", "")
    dashboard["topics"] = gh_data.get("topics", [])
    dashboard["visibility"] = gh_data.get("visibility", "")
    dashboard["git_contributor_count"] = gh_data.get("git_contributor_count", 0)
    dashboard["git_contributors_ids"] = gh_data.get("git_contributors_ids", [])
    dashboard["git_number_of_commits"] = gh_data.get("git_number_of_commits", 0)
    dashboard["git_number_of_committers"] = gh_data.get("git_number_of_committers", 0)
    dashboard["git_number_of_followers"] = gh_data.get("git_number_of_followers", 0)
    dashboard["git_number_of_updates"] = gh_data.get("git_number_of_updates", 0)
    dashboard["git_number_of_authors"] = gh_data.get("git_number_of_authors", 0)
    dashboard["git_authors"] = gh_data.get("git_authors", [])
    dashboard["git_number_of_branches"] = gh_data.get("git_number_of_branches", 0)
    dashboard["ci_cd_present"] = gh_data.get("ci_cd_present", False)
    dashboard["last_update"] = gh_data.get("last_update", "")
    
    # Additional Zenodo info
    dashboard["zenodo_keywords"] = zenodo.get("keywords", [])
    dashboard["zenodo_version"] = zenodo.get("versions", [])
    dashboard["zenodo_publication_date"] = zenodo.get("publication_date", "")
    dashboard["zenodo_license"] = zenodo.get("license", {})
    dashboard["zenodo_description_length"] = len(zenodo.get("description", ""))
    dashboard["zenodo_num_funding_entries"] = zenodo.get("zenodo_num_funding_entries", 0)
    # Add venue from Zenodo (from the meeting field)
    dashboard["venue"] = zenodo.get("venue", "")
    
    related_links = []
    for ident in zenodo.get("related_identifiers", []):
        if any(x in ident.get("identifier", "").lower() for x in ['doi', 'arxiv', 'sciencedirect']):
            related_links.append(ident.get("identifier", ""))
    dashboard["zenodo_related_publications"] = related_links

    # Compute extra heuristics from Zenodo and GitHub data
    extra_indicators = compute_heuristics(zenodo, gh_data)
    dashboard.update(extra_indicators)
    
    # RSMM maturity profile and RS Category Assignment
    dashboard["rsmm_profile"] = evaluate_rsmm_maturity_weighted(dashboard, RSMM_MODEL)
    dashboard["legacy_maturity_level"] = estimate_maturity_level(dashboard)
    dashboard["rs_category"] = evaluate_rs_category(dashboard)

    return dashboard

def estimate_maturity_level(dashboard):
    score = 0
    if dashboard.get("git_contributor_count", 0) > 3:
        score += 1
    if dashboard.get("git_number_of_commits", 0) > 30:
        score += 1
    if dashboard.get("testing", False):
        score += 1
    if dashboard.get("ci_cd_present", False):
        score += 1
    if dashboard.get("documentation_practices", False):
        score += 1
    if dashboard.get("dependency_management", False):
        score += 1
    if dashboard.get("code_quality", False):
        score += 1
    if dashboard.get("devops_practices", False):
        score += 1
    if dashboard.get("security_practices", False):
        score += 1
    if dashboard.get("zenodo_description_length", 0) > 200:
        score += 1
    if dashboard.get("git_number_of_updates", 0) > 10:
        score += 1
    if score >= 9:
        return "High"
    elif score >= 5:
        return "Medium"
    else:
        return "Low"

# --- Original file processing function (if needed) ---
def process_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return

    zenodo = extract_zenodo_data(data)
    gh_data = extract_github_data(data)
    if gh_data and "git_repo_desc" not in gh_data:
        repo_info = data.get("Github Repo", {}).get(next(iter(data.get("Github Repo", {})), {}), {}).get("repo_info") or {}
        gh_data["git_repo_desc"] = repo_info.get("description", "")
    
    dashboard_analysis = build_dashboard_analysis(zenodo, gh_data)
    data["dashboard_analysis"] = dashboard_analysis

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Processed and updated {filepath}")
    except Exception as e:
        print(f"Error writing to {filepath}: {e}")

def flatten_grants(grants):
    """Return a list of grants with a fixed structure."""
    fixed = []
    for grant in grants:
        fixed.append({
            "code": grant.get("code", ""),
            "internal_id": grant.get("internal_id", ""),
            "funder": {
                "name": grant.get("funder", {}).get("name", ""),
                "doi": grant.get("funder", {}).get("doi", ""),
                "acronym": grant.get("funder", {}).get("acronym", "")
            },
            "title": grant.get("title", ""),
            "acronym": grant.get("acronym", ""),
            "program": grant.get("program", ""),
            "url": grant.get("url", "")
        })
    return fixed

def flatten_communities(communities):
    """Return a list of communities with a fixed structure."""
    fixed = []
    for community in communities:
        fixed.append({
            "id": community.get("id", "")
        })
    return fixed

# --- Modified flatten_json function ---
def flatten_json(data, rs_domain):
    da = data.get("dashboard_analysis", {})
    # Extract creator information
    creators_data = da.get("Creators", [])
    creators = [c.get("name") for c in creators_data]
    affiliations = list({c.get("affiliation") for c in creators_data if c.get("affiliation")})
    orcids = [c.get("orcid") for c in creators_data if c.get("orcid")]

    # Flatten rsmm_profile fields
    rsmm = da.get("rsmm_profile", {})
    rsmm_flat = {
        "rsmm_profile_software_project_management": rsmm.get("Software Project Management", 0),
        "rsmm_profile_research_software_management": rsmm.get("Research Software Management", 0),
        "rsmm_profile_community_engagement": rsmm.get("Community Engagement", 0),
        "rsmm_profile_software_adoptability": rsmm.get("Software Adoptability", 0),
    }

    # Rename some nested license keys
    license_id = da.get("license", {}).get("id")
    zenodo_license_id = da.get("zenodo_license", {}).get("id")

    # Keys to exclude (handled separately)
    exclude_keys = {"Creators", "rsmm_profile", "license", "zenodo_license", "grants", "communities"}

    flat = {
        "creators": creators,
        "affiliations": affiliations,
        "orcids": orcids,
        "legacy_maturity_level": da.get("legacy_maturity_level"),
        "rs_category": da.get("rs_category"),
        "rs_domain": rs_domain  # Already computed (with underscores replaced by spaces)
    }

    if license_id:
        flat["license_id"] = license_id
    if zenodo_license_id:
        flat["zenodo_license_id"] = zenodo_license_id

    # Include remaining top-level dashboard fields (except those excluded)
    for key, value in da.items():
        if key not in exclude_keys:
            flat[key] = value

    # Merge flattened RSMM profile data
    flat.update(rsmm_flat)

    # Transform grants into separate lists for name, acronym, program, and url.
    grants = da.get("grants", [])
    flat["grants_name"] = [grant.get("title", "") for grant in grants]
    flat["grant_acronym"] = [grant.get("acronym", "") for grant in grants]
    flat["grant_program"] = [grant.get("program", "") for grant in grants]
    flat["grant_url"] = [grant.get("url", "") for grant in grants]

    # Transform communities into a list of community ids.
    communities = da.get("communities", [])
    flat["communities"] = [community.get("id", "") for community in communities]

    return flat

def apply_origin_prefix(flat_dict):
    """
    For each key in the flat dictionary, if it does not already start with "git_" or "zenodo_", 
    add "zenodo_" as a prefix.
    """
    new_dict = {}
    for key, value in flat_dict.items():
        if key.startswith("git_") or key.startswith("zenodo_"):
            new_key = key
        else:
            new_key = "zenodo_" + key
        new_dict[new_key] = value
    return new_dict

def add_prefix_recursively(item, prefix):
    """
    Recursively add the given prefix to all keys in dictionaries.
    If the item is a list, process each dictionary element.
    """
    if isinstance(item, dict):
        new_dict = {}
        for key, value in item.items():
            new_key = prefix + key  # Add the prefix to each key
            new_dict[new_key] = add_prefix_recursively(value, prefix)
        return new_dict
    elif isinstance(item, list):
        return [add_prefix_recursively(elem, prefix) if isinstance(elem, dict) else elem for elem in item]
    else:
        return item

# --- New function to process and flatten each file ---
def process_and_flatten_file(input_filepath, input_root, output_root):
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {input_filepath}: {e}")
        return

    # Process file to generate dashboard_analysis
    zenodo = extract_zenodo_data(data)
    gh_data = extract_github_data(data)
    if gh_data and "git_repo_desc" not in gh_data:
        repo_info = data.get("Github Repo", {}).get(next(iter(data.get("Github Repo", {})), {}), {}).get("repo_info") or {}
        gh_data["git_repo_desc"] = repo_info.get("description", "")
    dashboard_analysis = build_dashboard_analysis(zenodo, gh_data)
    data["dashboard_analysis"] = dashboard_analysis

    # Determine the rs_domain from the file's subdirectory (relative to input_root)
    relative_dir = os.path.relpath(os.path.dirname(input_filepath), input_root)
    rs_domain = os.path.basename(relative_dir).replace("_", " ")

    # Get flattened JSON with the rs_domain key added
    flat_data = flatten_json(data, rs_domain)

    # Determine filename prefix based on available GitHub and Zenodo data
    git_flag = bool(dashboard_analysis.get("git_repo_name"))
    zenodo_flag = bool(dashboard_analysis.get("doi"))
    if git_flag and zenodo_flag:
        prefix = "git_zenodo_"
    elif git_flag:
        prefix = "git_"
    elif zenodo_flag:
        prefix = "zenodo_"
    else:
        prefix = ""

    # Apply the origin prefix: keys already starting with "git_" remain unchanged;
    # all others get a "zenodo_" prefix (unless they already have it)
    flat_data = apply_origin_prefix(flat_data)

    # Create the corresponding output directory (preserving subdirectory structure)
    output_dir = os.path.join(output_root, relative_dir)
    os.makedirs(output_dir, exist_ok=True)
    original_filename = os.path.basename(input_filepath)
    output_filename = prefix + original_filename
    output_filepath = os.path.join(output_dir, output_filename)

    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(flat_data, f, indent=2)
        print(f"Flattened file saved to {output_filepath}")
    except Exception as e:
        print(f"Error writing to {output_filepath}: {e}")

# --- New main function for flattening ---
def main_flatten():
    INPUT_DIR = r"C:\Users\User\Downloads\RSE-Analysis\RS"    # update this path as needed
    OUTPUT_DIR = r"C:\Users\User\Downloads\RSE-Analysis\RS_2.0"   # update this path as needed
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(".json"):
                input_filepath = os.path.join(root, file)
                process_and_flatten_file(input_filepath, INPUT_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main_flatten()
