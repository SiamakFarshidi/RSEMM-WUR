import os
import sys
import re
import json
import time
import base64
import logging
from urllib.parse import urljoin
import requests

# ──────────────────────────────── CONFIGURATION ────────────────────────────────

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
DATACITE_API_URL = "https://api.datacite.org/dois/"
SPDX_LICENSE_LIST_URL = "https://raw.githubusercontent.com/spdx/license-list-data/master/json/licenses.json"

INPUT_DIR = r"C:\Users\User\Downloads\RS-Zenodo\RS-Repo-Zenodo - v4"

TIMEOUT = 10
MAX_RETRIES = 3
RETRY_SLEEP = 5

STANDARD_FORMATS = {
    ".csv", ".json", ".xml", ".h5", ".nc", ".yaml", ".yml",
}

_SPdx_LICENSES = None

# ──────────────────────────────── HELPERS ────────────────────────────────────

def _http_head(url, headers=None):
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.head(url, headers=headers or {}, timeout=TIMEOUT, allow_redirects=True)
            return r.status_code, r.headers
        except requests.RequestException:
            time.sleep(RETRY_SLEEP)
    return None, {}

def _http_get_json(url, headers=None):
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=headers or {}, timeout=TIMEOUT)
            if r.ok:
                return r.json()
            return None
        except requests.RequestException:
            time.sleep(RETRY_SLEEP)
    return None

def load_spdx_licenses():
    global _SPdx_LICENSES
    if _SPdx_LICENSES is None:
        data = _http_get_json(SPDX_LICENSE_LIST_URL)
        if data and "licenses" in data:
            _SPdx_LICENSES = {lic["licenseId"].upper() for lic in data["licenses"]}
        else:
            _SPdx_LICENSES = set()
    return _SPdx_LICENSES

def github_api(path, method="GET", **kwargs):
    url = urljoin("https://api.github.com/", path.lstrip("/"))
    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    for attempt in range(MAX_RETRIES):
        r = requests.request(method, url, headers=headers, timeout=TIMEOUT, **kwargs)
        if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            time.sleep(max(reset - int(time.time()), 1))
            continue
        return r
    return None

def parse_github_url(url):
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)(?:\.git)?", url)
    if not m:
        raise ValueError(f"Not a valid GitHub URL: {url}")
    return m.group(1), m.group(2)

def extract_github_url(metadata):
    # 1) custom field
    url = metadata.get("custom", {}).get("code:codeRepository")
    if url and "github.com" in url:
        return url
    # 2) related_identifiers
    for rel in metadata.get("related_identifiers", []):
        idf = rel.get("identifier", "")
        if "github.com" in idf:
            return idf
    # 3) HTML text fields
    content = (metadata.get("method", "") or "") + " " + (metadata.get("description", "") or "")
    m = re.search(r"https?://github\.com/[^\s\"'<>]+", content)
    if m:
        return m.group(0)
    return None

# ─────────────────────── FAIR4RS SUB-PRINCIPLE CHECKERS ───────────────────────

def check_identifier_global(doi):
    if not doi or not re.match(r"10\.\d{4,9}/\S+", doi):
        return False
    code, _ = _http_head(f"https://doi.org/{doi}")
    if not (200 <= code < 400):
        return False
    resp = requests.get(DATACITE_API_URL + doi, timeout=TIMEOUT)
    return resp.status_code == 200

def check_granular_ids(files):
    ids = [f.get("id") for f in files if f.get("id")]
    return len(ids) == len(files) and len(ids) == len(set(ids))

def check_version_identifiers(versions, record):
    if not versions:
        return False
    for rel in versions:
        if "identifier" in rel and "scheme" in rel:
            ident = rel["identifier"]
            scheme = rel["scheme"].lower()
            if scheme == "doi":
                if not check_identifier_global(ident):
                    return False
            else:
                code, _ = _http_head(ident)
                if not (code and 200 <= code < 400):
                    return False
        elif "parent" in rel:
            p = rel["parent"]
            ptype = p.get("pid_type", "").lower()
            pvalue = p.get("pid_value")
            if ptype in ("recid", "conceptrecid"):
                parent_doi = record.get("conceptdoi") or record.get("doi")
            elif ptype == "doi":
                parent_doi = pvalue
            else:
                return False
            if not parent_doi or not check_identifier_global(parent_doi):
                return False
        else:
            return False
    return True

def check_rich_metadata(metadata, owner=None, repo=None):
    core = {"title", "creators", "description", "license"}
    if not core.issubset(metadata.keys()):
        return False
    if not metadata.get("language") and not metadata.get("custom", {}).get("code:programmingLanguage"):
        return False
    ld = metadata.get("custom", {}).get("schema_org_json_ld", {})
    if ld.get("@context") and ld.get("@type") == "SoftwareSourceCode":
        return True
    if owner and repo:
        r = github_api(f"/repos/{owner}/{repo}/contents/codemeta.json")
        if r and r.status_code == 200:
            try:
                raw = base64.b64decode(r.json().get("content", ""))
                cm = json.loads(raw)
                if cm.get("@context") and cm.get("@type") == "SoftwareSourceCode":
                    return True
            except Exception:
                pass
    return False

def check_metadata_includes_own_id(metadata, record_doi):
    ld = metadata.get("custom", {}).get("schema_org_json_ld", {})
    if ld.get("@id") and record_doi and ld["@id"].endswith(record_doi):
        return True
    if metadata.get("doi") == record_doi:
        return True
    return False

def check_metadata_indexed(doi):
    resp = requests.get(DATACITE_API_URL + doi, timeout=TIMEOUT)
    return resp.status_code == 200

def check_accessible_by_protocol(url):
    code, _ = _http_head(url)
    if not code:
        return False, "unknown"
    if code in (401, 403):
        return False, "auth_required"
    proto = url.split(":", 1)[0].lower()
    open_proto = proto in {"http", "https", "ftp"}
    return (200 <= code < 400 and open_proto), ("open" if open_proto else "restricted")

def check_persistent_metadata(metadata_url):
    ok, _ = check_accessible_by_protocol(metadata_url)
    return ok

def check_standard_data_formats(owner, repo):
    r = github_api(f"/repos/{owner}/{repo}/git/trees/HEAD?recursive=1")
    if not r or r.status_code != 200:
        return False, []
    tree = r.json().get("tree", [])
    exts = {os.path.splitext(e["path"])[1].lower() for e in tree}
    found = list(exts & STANDARD_FORMATS)
    return bool(found), found

def check_qualified_references(metadata):
    refs = metadata.get("references", []) + metadata.get("related_identifiers", [])
    if not refs:
        return False
    for r in refs:
        idf = r.get("identifier")
        scheme = r.get("scheme", "").lower()
        if not idf or not scheme:
            return False
        if scheme == "doi":
            if not check_identifier_global(idf):
                return False
        else:
            code, _ = _http_head(idf)
            if not (code and 200 <= code < 400):
                return False
    return True

def check_license_spdx(metadata):
    lic = metadata.get("license", {}).get("id", "")
    if not lic:
        return False
    norm = re.sub(r'[-_ ]license$', '', lic, flags=re.I).upper()
    return norm in load_spdx_licenses()

def check_provenance(owner, repo):
    r = github_api(f"/repos/{owner}/{repo}/stats/contributors")
    if not r or r.status_code != 200:
        return {}
    contribs = r.json()
    total = sum(c.get("total", 0) for c in contribs)
    unique = len(contribs)
    latest = None
    r2 = github_api(f"/repos/{owner}/{repo}/commits?per_page=1")
    if r2 and r2.status_code == 200 and r2.json():
        latest = r2.json()[0]["commit"]["author"]["date"]
    orcid = any(
        "orcid.org" in (github_api(f"/users/{c.get('author', {}).get('login')}").json().get("bio", ""))
        for c in contribs if c.get("author", {}).get("login")
    )
    return {"commits": total, "contributors": unique, "latest_commit": latest, "orcid_present": orcid}

def check_dependencies(owner, repo):
    manifests = {
        "requirements.txt": lambda b: [l.split("==")[0] for l in b.decode().splitlines() if l and not l.startswith("#")],
        "environment.yml": lambda b: [],  # omitted
        "pyproject.toml": lambda b: [],   # omitted
        "package.json": lambda b: [],     # omitted
    }
    found, unresolved = [], []
    for fname, parser in manifests.items():
        r = github_api(f"/repos/{owner}/{repo}/contents/{fname}")
        if r and r.status_code == 200:
            content = base64.b64decode(r.json().get("content", ""))
            for pkg in parser(content):
                code, _ = _http_head(f"https://pypi.org/project/{pkg}")
                (found if code and 200 <= code < 400 else unresolved).append(pkg)
    return {"dependencies_found": found, "dependencies_unresolved": unresolved}

def check_technical_standards(owner, repo):
    r = github_api(f"/repos/{owner}/{repo}/git/trees/HEAD?recursive=1")
    if not r or r.status_code != 200:
        return {}
    paths = {e["path"] for e in r.json().get("tree", [])}
    return {
        "has_tests": any(p.startswith(("tests/", "test/")) for p in paths),
        "ci_workflows": any(p.startswith(".github/workflows/") for p in paths),
        "packaging": any(p in {"setup.py", "pyproject.toml", "Dockerfile"} for p in paths),
        "docs": any(p.startswith("docs/") for p in paths),
    }

# ────────────────────────── AGGREGATION & SCORING ───────────────────────────

def score_yes_no(val):
    return 1.0 if val else 0.0

def categorize(score, thresholds=(0.75, 0.5)):
    if score >= thresholds[0]:
        return "High"
    if score >= thresholds[1]:
        return "Medium"
    return "Low"

def estimate_principles(sub):
    groups = {
        "Findable": [
            sub["doi_valid"], sub["granular_ids"], sub["version_ids"],
            sub["rich_metadata"], sub["metadata_includes_id"], sub["metadata_indexed"]
        ],
        "Accessible": [
            sub["software_accessible"], sub["metadata_persistent"]
        ],
        "Interoperable": [
            sub["standard_data_formats"], sub["qualified_references"]
        ],
        "Reusable": [
            sub["license_spdx"], sub["provenance"].get("commits", 0) > 0,
            bool(sub["dependencies_found"]), sub["technical_standards"].get("has_tests", False)
        ]
    }
    cats, overall = {}, []
    for k, vals in groups.items():
        sc = sum(score_yes_no(v) for v in vals) / len(vals) if vals else 0
        cats[k] = categorize(sc)
        overall.append(sc)
    return {"principle_categories": cats, "overall_fairness": categorize(sum(overall) / len(overall))}

# ────────────────────────── MAIN PROCESSING ─────────────────────────────────

def assess_record(record):
    md = record.get("metadata", {}) or {}
    rec_doi = md.get("doi") or record.get("doi")

    git_url = extract_github_url(md)
    owner = repo = None
    if git_url:
        owner, repo = parse_github_url(git_url)

    doi_valid = check_identifier_global(rec_doi)
    granular_ids = check_granular_ids(record.get("files", []))
    version_ids = check_version_identifiers(md.get("relations", {}).get("version", []), record)
    rich_metadata = check_rich_metadata(md, owner, repo)
    metadata_includes_id = check_metadata_includes_own_id(md, rec_doi)
    metadata_indexed = check_metadata_indexed(rec_doi)

    software_url = md.get("url") or record.get("links", {}).get("self")
    software_accessible, _ = check_accessible_by_protocol(software_url) if software_url else (False, "")
    metadata_url = record.get("links", {}).get("self")
    metadata_persistent = check_persistent_metadata(metadata_url)

    standard_data_formats, formats_found = (False, [])
    if owner and repo:
        standard_data_formats, formats_found = check_standard_data_formats(owner, repo)
    qualified_references = check_qualified_references(md)

    license_spdx = check_license_spdx(md)
    provenance = check_provenance(owner, repo) if owner else {}
    deps = check_dependencies(owner, repo) if owner else {"dependencies_found": [], "dependencies_unresolved": []}
    tech = check_technical_standards(owner, repo) if owner else {}

    sub = {
        "doi_valid": doi_valid,
        "granular_ids": granular_ids,
        "version_ids": version_ids,
        "rich_metadata": rich_metadata,
        "metadata_includes_id": metadata_includes_id,
        "metadata_indexed": metadata_indexed,
        "software_accessible": software_accessible,
        "metadata_persistent": metadata_persistent,
        "standard_data_formats": standard_data_formats,
        "formats_found": formats_found,
        "qualified_references": qualified_references,
        "license_spdx": license_spdx,
        "provenance": provenance,
        "dependencies_found": deps["dependencies_found"],
        "dependencies_unresolved": deps["dependencies_unresolved"],
        "technical_standards": tech
    }
    scores = estimate_principles(sub)

    return {
        "subprinciple_assessment": sub,
        **scores,
        "git_url": git_url or None
    }

def process_file(path):
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)
    assessment = assess_record(rec)
    rec["fair_assessment"] = assessment
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    print(f"Updated {path}")

def main():
    if not os.path.isdir(INPUT_DIR):
        logging.error(f"Invalid input directory: {INPUT_DIR}")
        sys.exit(1)
    for root, _, files in os.walk(INPUT_DIR):
        for fn in files:
            if fn.lower().endswith(".json"):
                process_file(os.path.join(root, fn))

if __name__ == "__main__":
    main()
