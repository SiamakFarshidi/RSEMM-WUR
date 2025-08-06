import os
import time
import json
import logging
import re
import requests
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ←── CONFIGURE THESE ───────────────────────────────────────────────────────
GITHUB_TOKEN = "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"
JSON_DIR = r"C:\Users\User\Downloads\RS-Zenodo\RS-Repo-Zenodo - v4"
# ────────────────────────────────────────────────────────────────────────────→

# set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# create a single Session with retries & backoff
session = requests.Session()
session.headers.update({
    "Authorization": f"token {GITHUB_TOKEN}",
    # for core v3 + topics preview
    "Accept": "application/vnd.github.v3+json, application/vnd.github.mercy-preview+json"
})
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

def github_request(url, params=None):
    """Wrapper around session.get that honors rate limits & retries."""
    while True:
        resp = session.get(url, params=params)
        if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            sleep_for = max(reset_ts - time.time(), 0) + 5
            logging.warning(f"Rate limit hit. Sleeping for {sleep_for:.0f}s …")
            time.sleep(sleep_for)
            continue
        resp.raise_for_status()
        return resp

def paginate(url, params=None):
    """Paginate through GitHub list endpoints."""
    items = []
    while url:
        resp = github_request(url, params=params)
        items.extend(resp.json())
        link = resp.headers.get("Link", "")
        next_url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                next_url = part.split(";")[0].strip()[1:-1]
                break
        url = next_url
        params = None
    return items

def extract_owner_repo(raw_url):
    """
    Parse owner/repo from various GitHub URL formats.
    """
    raw = raw_url.strip()
    if raw.startswith("git@"):
        raw = raw.replace("git@", "https://").replace(":", "/", 1)
    parsed = urlparse(raw)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse owner/repo from {raw_url}")
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo

def gather_github_metadata(owner, repo):
    base = f"https://api.github.com/repos/{owner}/{repo}"
    meta = {}

    # 1) repo‐level info
    try:
        rd = github_request(base).json()
        meta.update({
            "GitHub_full_name": rd.get("full_name"),
            "GitHub_description": rd.get("description"),
            "GitHub_fork": rd.get("fork"),
            "GitHub_forks_count": rd.get("forks_count"),
            "GitHub_git_url": rd.get("git_url"),
            "GitHub_created_at": rd.get("created_at"),
            "GitHub_updated_at": rd.get("updated_at"),
            "GitHub_pushed_at": rd.get("pushed_at"),
            "GitHub_size": rd.get("size"),
            "GitHub_stargazers_count": rd.get("stargazers_count"),
            "GitHub_watchers_count": rd.get("watchers_count"),
            "GitHub_network_count": rd.get("network_count"),
            "GitHub_open_issues_count": rd.get("open_issues_count"),
            "GitHub_default_branch": rd.get("default_branch"),
            "GitHub_private": rd.get("private"),
            "GitHub_is_archived": rd.get("archived"),
            "GitHub_is_mirrored": rd.get("mirror_url") is not None,
            "GitHub_has_issues": rd.get("has_issues"),
            "GitHub_has_projects": rd.get("has_projects"),
            "GitHub_has_wiki": rd.get("has_wiki"),
            "GitHub_has_discussions": rd.get("has_discussions"),
            "GitHub_visibility": rd.get("visibility"),
            "GitHub_git_license": (rd.get("license") or {}).get("spdx_id")
                                  or (rd.get("license") or {}).get("name"),
        })
    except Exception as e:
        logging.error(f"Repo info fetch failed for {owner}/{repo}: {e}")

    # 2) languages
    try:
        langs = github_request(base + "/languages").json()
        meta["GitHub_languages"] = list(langs.keys())
    except Exception as e:
        logging.warning(f"Languages fetch failed: {e}")
        meta["GitHub_languages"] = []

    # 3) topics
    try:
        tops = github_request(base + "/topics").json()
        meta["GitHub_topics"] = tops.get("names", [])
    except Exception as e:
        logging.warning(f"Topics fetch failed: {e}")
        meta["GitHub_topics"] = []

    # 4) workflows
    try:
        wf = github_request(base + "/actions/workflows").json()
        meta["GitHub_workflow_total_count"] = wf.get("total_count", 0)
        meta["GitHub_workflows_names"] = [w.get("name") for w in wf.get("workflows", [])]
    except Exception as e:
        logging.warning(f"Workflows fetch failed: {e}")
        meta.update({
            "GitHub_workflow_total_count": 0,
            "GitHub_workflows_names": []
        })

    # 5) README
    try:
        r = session.get(base + "/readme")
        meta["GitHub_has_README"] = (r.status_code == 200)
    except Exception as e:
        logging.warning(f"Readme check failed: {e}")
        meta["GitHub_has_README"] = False

    # 6) branches
    try:
        branches = paginate(base + "/branches", params={"per_page": 100})
        names = [b.get("name") for b in branches]
        meta["GitHub_branches_names"] = names
        meta["GitHub_git_number_of_branches"] = len(names)
    except Exception as e:
        logging.warning(f"Branches fetch failed: {e}")
        meta.update({
            "GitHub_branches_names": [],
            "GitHub_git_number_of_branches": 0
        })

    # 7) contributors
    try:
        contribs = paginate(base + "/contributors", params={"per_page": 100, "anon": "true"})
        meta["GitHub_contributors_count"] = len(contribs)
    except Exception as e:
        logging.warning(f"Contributors fetch failed: {e}")
        meta["GitHub_contributors_count"] = 0

    # 8) commits & verification
    try:
        commits = paginate(base + "/commits", params={"per_page": 100})
        msgs, reasons = [], []
        verified = 0
        authors, committers = set(), set()
        for c in commits:
            cm = c.get("commit", {})
            msgs.append(cm.get("message"))
            v = cm.get("verification", {})
            if v.get("verified"):
                verified += 1
                reasons.append(v.get("reason"))
            if c.get("author") and c["author"].get("login"):
                authors.add(c["author"]["login"])
            if c.get("committer") and c["committer"].get("login"):
                committers.add(c["committer"]["login"])
        meta.update({
            "GitHub_commits_messages": msgs,
            "GitHub_commits_verification_reasons": reasons,
            "GitHub_commits_verification_verified_count": verified,
            "GitHub_git_number_of_updates": len(commits),
            "GitHub_git_number_of_authors": len(authors),
            "GitHub_git_number_of_committers": len(committers),
        })
    except Exception as e:
        logging.warning(f"Commits fetch failed: {e}")
        meta.update({
            "GitHub_commits_messages": [],
            "GitHub_commits_verification_reasons": [],
            "GitHub_commits_verification_verified_count": 0,
            "GitHub_git_number_of_updates": 0,
            "GitHub_git_number_of_authors": 0,
            "GitHub_git_number_of_committers": 0,
        })

    return meta

def find_github_url(rec):
    """Try multiple fields and a regex fallback to find a GitHub URL."""
    # 1) related_identifiers
    for rid in rec.get("metadata", {}).get("related_identifiers", []):
        for key in ("related_identifier_url", "identifier_url", "relatedIdentifier", "identifier"):
            url = rid.get(key, "")
            if url and "github.com" in url:
                return url
    # 2) metadata.urls
    for u in rec.get("metadata", {}).get("urls", []):
        url = u.get("url", "")
        if url and "github.com" in url:
            return url
    # 3) ai-ml-ops
    repo = rec.get("repo-profile", {}).get("ai-ml-ops", {}).get("repository", "")
    if repo and "github.com" in repo:
        return repo
    # 4) fair-assessment.git_url
    giturl = rec.get("fair-assessment", {}).get("git_url", "")
    if giturl and "github.com" in giturl:
        return giturl
    # 5) regex fallback
    dump = json.dumps(rec)
    m = re.search(r'(?:https?://|git@)github\.com[:/][^\s"\\\']+', dump)
    if m:
        return m.group(0)
    return None

if __name__ == "__main__":
    for fname in os.listdir(JSON_DIR):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(JSON_DIR, fname)
        logging.info(f"Processing {fname} …")
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)

        gh_url = find_github_url(rec)
        if not gh_url:
            logging.warning("  → no GitHub URL found, skipping")
            continue

        try:
            owner, repo = extract_owner_repo(gh_url)
            meta = gather_github_metadata(owner, repo)
            # retry once if empty
            if not meta.get("GitHub_full_name"):
                logging.warning("  → empty metadata, retrying…")
                time.sleep(2)
                meta = gather_github_metadata(owner, repo)
        except Exception as e:
            logging.error(f"  → final extraction failed for {gh_url}: {e}")
            continue

        rec["Github-metadata"] = meta
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        logging.info(f"  → written metadata for {owner}/{repo}")
