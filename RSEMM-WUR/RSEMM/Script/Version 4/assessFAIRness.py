import json
import os
import re
import base64
import urllib.request
import urllib.error

# ────────────────────────────────────────────────────────────────────────────────
# 1) HARD‐CODED SETTINGS ─────────────────────────────────────────────────────────
#
# (a) GitHub personal access token (for higher rate‐limit or private repos).
GITHUB_TOKEN =  "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"

# (b) Path to the Zenodo‐style JSON record on your local disk
RECORD_JSON_PATH = r"C:\Users\farsh001\Downloads\script-RSE-20250603T181429Z-1-001\script-RSE\Repo\RS-Repo-Zenodo - v3\14006944.json"

# (c) Where to write the merged output JSON:
OUTPUT_JSON_PATH = "fairness_assessment.json"
# ────────────────────────────────────────────────────────────────────────────────


def github_api_request(path: str):
    """
    Perform a GET to https://api.github.com{path}, using GITHUB_TOKEN for Authorization.
    Return (status_code, decoded_body_bytes) or (status_code, None) on HTTP errors.
    """
    url = f"https://api.github.com{path}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "python-urllib"
    })
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.getcode(), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, None
    except urllib.error.URLError:
        return None, None


def parse_github_owner_repo(url: str):
    """
    Extract (owner, repo) from a GitHub URL like https://github.com/owner/repo
    Strips any /tree/... suffix. Raises ValueError if it doesn’t match.
    """
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)", url)
    if not m:
        raise ValueError(f"Cannot parse GitHub owner/repo from: {url}")
    owner, repo = m.group(1), m.group(2)
    return owner, repo


# ────────────────────────────────────────────────────────────────────────────────
# 2) FAIR4RS SUB‐PRINCIPLE CHECKS (Paper 1) ────────────────────────────────────────
def assess_fairness_metadata(record: dict) -> dict:
    """
    Return a dict mapping each FAIR4RS sub‐principle to "yes"/"no"/"partial"/"unknown"/"not applicable".
    Sub‐principles: F1, F1.1, F1.2, F2, F3, F4, A1, A1.2, A2, I1, I2, R1.1, R1.2, R2, R3.
    """
    results = {}
    metadata = record.get("metadata", {})

    # F1: DOI present?
    results["doi_present"] = "yes" if metadata.get("doi") else "no"

    # F1.1: Subcomponent identifiers (not in JSON => unknown)
    results["subcomponent_identifiers"] = "unknown"

    # F1.2: Version identifiers? Look for metadata["relations"]["version"]
    relations = metadata.get("relations", {}).get("version", [])
    results["version_identifiers"] = "yes" if relations else "no"

    # F2: Rich metadata? (title, creators, description, license, language)
    required_f2 = ["title", "creators", "description", "license", "language"]
    missing_f2 = [f for f in required_f2 if not metadata.get(f)]
    results["rich_metadata_present"] = "yes" if not missing_f2 else "no"

    # F3: Metadata explicitly includes its own DOI?
    results["metadata_includes_doi"] = "yes" if metadata.get("doi") and record.get("doi") else "no"

    # F4: Metadata accessible? (we have it in memory => yes)
    results["metadata_accessible"] = "yes" if metadata else "no"

    # A1: Software retrievable by identifier? (links.doi + files list)
    links = record.get("links", {})
    files = record.get("files", [])
    results["software_retrievable"] = "yes" if (links.get("doi") and files) else "no"

    # A1.2: Authentication required? (if access_right == "open" => not applicable; else unknown)
    results["authentication_required"] = (
        "not applicable" if metadata.get("access_right") == "open" else "unknown"
    )

    # A2: Metadata remain accessible if software removed? (we still have metadata => yes)
    results["metadata_persistent"] = "yes" if metadata else "no"

    # I1: Uses standard data formats? (cannot infer from JSON => unknown)
    results["uses_standard_data_formats"] = "unknown"

    # I2: Qualified refs to external objects? (look for DOI URLs in metadata["references"])
    refs = metadata.get("references", []) or []
    qualified = [r for r in refs if isinstance(r, str) and r.startswith("https://doi.org")]
    results["qualified_references_to_objects"] = "yes" if qualified else "no"

    # R1.1: Clear license present? (metadata["license"]["id"])
    results["clear_license"] = "yes" if metadata.get("license", {}).get("id") else "no"

    # R1.2: Provenance via ORCID? (all creators have orcid => yes; some => partial; none => no)
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

    # R2: Qualified refs to other software? (cannot infer => unknown)
    results["qualified_references_to_software"] = "unknown"

    # R3: Meets community standards (CI, linters)? (cannot infer from JSON => unknown)
    results["ci_or_community_standards"] = "unknown"

    return results


def categorize_principle(scores: list, thresholds=(0.75, 0.5)) -> str:
    """
    Given numeric scores [0, 0.5, 1], return "High"/"Medium"/"Low".
    """
    if not scores:
        return "Low"
    avg = sum(scores) / len(scores)
    if avg >= thresholds[0]:
        return "High"
    elif avg >= thresholds[1]:
        return "Medium"
    else:
        return "Low"


def score_value(val: str) -> float:
    """
    Map "yes"→1.0, "partial"→0.5, others→0.0.
    """
    return {"yes": 1.0, "partial": 0.5}.get(val, 0.0)


def estimate_fairness_from_subprinciples(sub: dict) -> dict:
    """
    From sub-principle results (strings), compute:
      {
        "principle_categories": {Findable, Accessible, Interoperable, Reusable},
        "overall_fairness": "High"/"Medium"/"Low"
      }
    """
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
    principle_scores = []

    for principle, keys in groups.items():
        numeric_scores = [score_value(sub.get(k, "unknown")) for k in keys]
        cat = categorize_principle(numeric_scores)
        principle_categories[principle] = cat
        avg_val = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
        principle_scores.append(avg_val)

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


# ────────────────────────────────────────────────────────────────────────────────
# 3) EXTRACT GITHUB URL FROM ZENODO JSON ─────────────────────────────────────────
def extract_git_url_from_zenodo(record: dict) -> str:
    """
    1) Check metadata["related_identifiers"] for a "scheme":"url" entry starting with "https://github.com/".
    2) If none, search metadata["description"] for the first "https://github.com/owner/repo".
    Return the base "https://github.com/owner/repo" (strip any /tree/... suffix).
    """
    metadata = record.get("metadata", {})

    # (1) related_identifiers
    for rid in metadata.get("related_identifiers", []) or []:
        ident = rid.get("identifier", "")
        scheme = rid.get("scheme", "").lower()
        if scheme == "url" and ident.startswith("https://github.com/"):
            m = re.match(r"(https?://github\.com/[^/]+/[^/]+)", ident)
            if m:
                return m.group(1)

    # (2) fallback: search description
    desc = metadata.get("description", "") or ""
    m2 = re.search(r"https?://github\.com/[^/\s]+/[^/\s]+", desc)
    if m2:
        return m2.group(0)

    return None


# ────────────────────────────────────────────────────────────────────────────────
# 4) IMPLEMENT PAPER 2 (“howfairis” CHECKS) VIA GITHUB API ONLY ──────────────────

def check_repository_exists(owner: str, repo: str) -> bool:
    """
    Check if GET /repos/{owner}/{repo} → status 200. If 404 or error, return False.
    """
    status, _ = github_api_request(f"/repos/{owner}/{repo}")
    return status == 200


def check_license_exists(owner: str, repo: str) -> bool:
    """
    Check if GET /repos/{owner}/{repo}/license → status 200.
    """
    status, _ = github_api_request(f"/repos/{owner}/{repo}/license")
    return status == 200


def fetch_readme_text(owner: str, repo: str) -> str:
    """
    Fetch repository README via GET /repos/{owner}/{repo}/readme
    (which returns JSON with "content" in base64). Decode and return UTF-8 text.
    If not found or error, return empty string.
    """
    status, body = github_api_request(f"/repos/{owner}/{repo}/readme")
    if status != 200 or body is None:
        return ""
    try:
        j = json.loads(body.decode("utf-8", errors="ignore"))
        b64 = j.get("content", "")
        data = base64.b64decode(b64)
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def check_registry_badge_in_readme(readme_text: str) -> bool:
    """
    Look for a Zenodo (or other registry) badge. E.g., contains "zenodo.org/badge"
    or "shields.io/badge/zenodo".
    """
    return bool(re.search(r"(zenodo\.org|shields\.io\/badge\/zenodo)", readme_text, re.IGNORECASE))


def check_citation_file_exists(owner: str, repo: str) -> bool:
    """
    Check if GET /repos/{owner}/{repo}/contents/CITATION.cff or CITATION.md → status 200.
    """
    status1, _ = github_api_request(f"/repos/{owner}/{repo}/contents/CITATION.cff")
    if status1 == 200:
        return True
    status2, _ = github_api_request(f"/repos/{owner}/{repo}/contents/CITATION.md")
    return status2 == 200


def check_quality_badge_in_readme(readme_text: str) -> bool:
    """
    Look for an OpenSSF (bestpractices.dev) badge in README. E.g.:
      https://bestpractices.dev
      or shields for best_practices (pattern "best_practices" or "best-practices")
    """
    return bool(re.search(r"(bestpractices\.dev|shields\.io\/badge\/best_practices)", readme_text, re.IGNORECASE))


def run_howfairis_checks(repo_url: str) -> dict:
    """
    Perform exactly five checks (Paper 2 via howfairis) using GitHub API / README parsing:
      1) Public, version-controlled repo → 1 point if it exists (status 200).
      2) License → 1 point if /license endpoint returns 200.
      3) Registry registration → 1 point if README contains a Zenodo (or similar) badge.
      4) Software citation enabled → 1 point if CITATION.cff or CITATION.md exists.
      5) Quality badge → 1 point if README contains a bestpractices.dev badge.
    Return a dict:
      {
        "howfairis_score": 0–5,
        "howfairis_details": {
            "repository": "yes"/"no",
            "license": "yes"/"no",
            "registry": "yes"/"no",
            "citation": "yes"/"no",
            "quality": "yes"/"no"
        }
      }
    """
    details = {"repository": "no", "license": "no", "registry": "no", "citation": "no", "quality": "no"}
    score = 0

    # Parse owner/repo
    try:
        owner, repo = parse_github_owner_repo(repo_url)
    except ValueError:
        return {"howfairis_score": 0, "howfairis_details": details}

    # 1) Public repo?
    if check_repository_exists(owner, repo):
        details["repository"] = "yes"
        score += 1

        # 2) License?
        if check_license_exists(owner, repo):
            details["license"] = "yes"
            score += 1

        # 3 & 5) Fetch README text once, then search badges
        readme_text = fetch_readme_text(owner, repo)

        # 3) Registry badge?
        if check_registry_badge_in_readme(readme_text):
            details["registry"] = "yes"
            score += 1

        # 5) Quality badge?
        if check_quality_badge_in_readme(readme_text):
            details["quality"] = "yes"
            score += 1

        # 4) Citation file?
        if check_citation_file_exists(owner, repo):
            details["citation"] = "yes"
            score += 1

    return {"howfairis_score": score, "howfairis_details": details}


# ────────────────────────────────────────────────────────────────────────────────
# 4) MAIN ENTRY POINT ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    final_output = {}

    # 1) Load Zenodo JSON & FAIR4RS sub-principles
    if os.path.isfile(RECORD_JSON_PATH):
        with open(RECORD_JSON_PATH, "r", encoding="utf-8") as f:
            record = json.load(f)
        subp = assess_fairness_metadata(record)
        est = estimate_fairness_from_subprinciples(subp)
        final_output["subprinciple_assessment"] = subp
        final_output["principle_categories"] = est["principle_categories"]
        final_output["overall_fairness"] = est["overall_fairness"]
    else:
        print(f"Warning: RECORD_JSON_PATH not found: '{RECORD_JSON_PATH}'. Skipping metadata checks.")
        final_output["subprinciple_assessment"] = {}
        final_output["principle_categories"] = {}
        final_output["overall_fairness"] = None

    # 2) Extract GitHub URL from JSON
    git_url = None
    if os.path.isfile(RECORD_JSON_PATH):
        git_url = extract_git_url_from_zenodo(record)
        final_output["git_url"] = git_url or "none found"
    else:
        final_output["git_url"] = "none found"

    # 3) Run howfairis‐style checks (Paper 2) if git_url exists
    if git_url:
        hf = run_howfairis_checks(git_url)
    else:
        hf = {"howfairis_score": 0, "howfairis_details": {
            "repository": "no", "license": "no", "registry": "no", "citation": "no", "quality": "no"
        }}
    final_output["howfairis_score"] = hf["howfairis_score"]
    final_output["howfairis_details"] = hf["howfairis_details"]

    # 4) Write combined JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as out_f:
        json.dump(final_output, out_f, indent=2)

    print(f"Written combined output to '{OUTPUT_JSON_PATH}'")
