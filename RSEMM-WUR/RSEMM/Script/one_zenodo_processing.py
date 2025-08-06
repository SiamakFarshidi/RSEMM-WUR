import os
import time
import json
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Configuration ─────────────────────────────────────────────────────────────

# Path to your list of IEEE taxonomy terms (one per line)
TAXONOMY_FILE = "./IEEE_Taxonomy/ieee_taxonomy_flat_L1_L2_filtered.txt"

# Directory where you want individual record JSONs to live
OUTPUT_DIR     = r"C:\Users\User\Downloads\RS-Zenodo\RS-Repo-Zenodo - v4"

# Master JSON file to keep track of URLs already seen
MASTER_INDEX   = "processed_urls.json"

# Zenodo API base URL
ZENODO_API_BASE = "https://zenodo.org/api/records"

# How many results per page (Zenodo allows up to 100)
PAGE_SIZE      = 100

# Time (in seconds) to sleep between consecutive API requests
BASE_SLEEP     = 1.0

# If we get HTTP 429 (Too Many Requests), back off by this amount (seconds).
BACKOFF_SLEEP  = 5.0

# Maximum number of worker threads
MAX_WORKERS    = 5

# ─── Shared State & Locks ──────────────────────────────────────────────────────

# Thread‐safe set of already‐processed URLs
processed_urls = set()
processed_lock = threading.Lock()

# ─── Terms to Skip ──────────────────────────────────────────────────────────────

skip_terms = {
}

# ─── Helper Functions ─────────────────────────────────────────────────────────

def load_master_index(path):
    """
    Load (or create) the master index of processed URLs.
    Returns a Python set of URLs.
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                urls = json.load(f)
            except json.JSONDecodeError:
                urls = []
    else:
        urls = []
    return set(urls)

def save_master_index(path, url_set):
    """
    Write the set of processed URLs back to the JSON file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(url_set)), f, indent=2)

def ensure_output_dir(path):
    """
    Make sure the output directory exists.
    """
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def term_in_text(term, text):
    """
    Return True if the entire term (lowercased) appears
    as a substring in 'text' (lowercased).
    """
    return term.lower() in (text or "").lower()

def fetch_page(term, page, size=PAGE_SIZE):
    """
    Query Zenodo for a single page of results, but only for
    records created on/after 2022-11-01.
    Returns the parsed JSON or {"status_code": 429} if rate‐limited.
    """
    lucene_query = f"{term} AND created:[2022-11-01 TO *]"
    params = {
        "q": lucene_query,
        "page": page,
        "size": size,
    }
    try:
        r = requests.get(ZENODO_API_BASE, params=params, timeout=30)
    except requests.RequestException as e:
        print(f"    [Error] network error for term='{term}', page={page}: {e}")
        return None

    if r.status_code == 429:
        return {"status_code": 429}

    if not r.ok:
        print(f"    [Error] HTTP {r.status_code} for term='{term}', page={page}")
        return None

    return r.json()

def process_term(term):
    """
    Fetch all pages for a single term, apply filters, write JSONs, and update processed_urls.
    """
    saved_count = 0
    page = 1
    first_page = True

    print(f"[Start] Processing term '{term}'")

    while True:
        print(f"  [Term '{term}'] Fetching page {page} …")
        resp = fetch_page(term, page, size=PAGE_SIZE)

        if resp is None:
            # Non‐429 error—stop trying further pages for this term
            print(f"  [Warning] ('{term}') stopping pagination due to error.")
            break

        if isinstance(resp, dict) and resp.get("status_code") == 429:
            # Rate limit: back off, then retry same page
            print(f"    [Warning] ('{term}') Rate limit on page {page}. Sleeping {BACKOFF_SLEEP}s…")
            time.sleep(BACKOFF_SLEEP)
            continue

        hits_obj   = resp.get("hits", {})
        hits       = hits_obj.get("hits", [])
        total_hits = hits_obj.get("total", 0)

        if first_page:
            print(f"  [Term '{term}'] (post-2022-11-01) ≈ {total_hits} total records.")
            first_page = False

        if not hits:
            print(f"  [Term '{term}'] No more results at page {page}.")
            break

        print(f"  [Term '{term}'] Page {page} returned {len(hits)} hits.")

        for item in hits:
            metadata = item.get("metadata", {})

            # ─── type check: must be "software" ─────────────────────────────────
            resource_type = metadata.get("resource_type", {}).get("type", "")
            if resource_type.lower() != "software":
                # skip non-software records
                continue

            title       = metadata.get("title", "") or ""
            description = metadata.get("description", "") or metadata.get("abstract", "") or ""

            # Quick relevance check: ensure 'term' appears in title or description
            if not (term_in_text(term, title) or term_in_text(term, description)):
                continue

            # Get canonical URL for deduplication
            record_url = (
                item.get("links", {}).get("html")
                or item.get("doi_url")
                or item.get("doi")
            )
            if not record_url:
                continue

            # Thread-safe check & update against processed_urls
            with processed_lock:
                if record_url in processed_urls:
                    continue

            # Add the query term into the metadata before saving
            metadata["query_term"] = term

            rec_id   = item.get("id")
            fname    = f"{rec_id}.json"
            out_path = os.path.join(OUTPUT_DIR, fname)

            # Write JSON file for this record
            try:
                with open(out_path, "w", encoding="utf-8") as out_f:
                    json.dump(item, out_f, ensure_ascii=False, indent=2)
                with processed_lock:
                    processed_urls.add(record_url)
                saved_count += 1
                print(f"    [Saved] ({term}) ID {rec_id} → {out_path}")
            except Exception as e:
                print(f"    [Error] Could not write '{out_path}': {e}")
                # Do NOT mark as processed in this case

        # If fewer than PAGE_SIZE hits, we’re at the last page
        if len(hits) < PAGE_SIZE:
            print(f"  [Term '{term}'] Finished after page {page} ({len(hits)} hits).")
            break

        page += 1
        time.sleep(BASE_SLEEP)

    print(f"[End] Term '{term}' completed. Saved {saved_count} new record(s).")

# ─── Main Workflow ────────────────────────────────────────────────────────────

def main():
    global processed_urls

    # 1) Load or initialize master index of URLs
    processed_urls = load_master_index(MASTER_INDEX)
    print(f"[Info] Loaded {len(processed_urls)} already-processed URLs from '{MASTER_INDEX}'.")

    # 2) Ensure output directory exists
    ensure_output_dir(OUTPUT_DIR)

    # 3) Scan OUTPUT_DIR for existing JSON files and pull their record URLs
    for existing_fname in os.listdir(OUTPUT_DIR):
        if not existing_fname.lower().endswith('.json'):
            continue
        existing_path = os.path.join(OUTPUT_DIR, existing_fname)
        try:
            with open(existing_path, 'r', encoding='utf-8') as ef:
                existing_data = json.load(ef)
            record_url = (
                existing_data.get("links", {}).get("html")
                or existing_data.get("doi_url")
                or existing_data.get("doi")
            )
            if record_url:
                processed_urls.add(record_url)
        except Exception:
            # Skip files that can't be read/parsed
            continue

    print(f"[Info] After scanning '{OUTPUT_DIR}', total processed URLs = {len(processed_urls)}.")

    # 4) Read taxonomy terms, skipping the specified ones
    if not os.path.exists(TAXONOMY_FILE):
        print(f"[Error] Cannot find '{TAXONOMY_FILE}'. Make sure it's in the working directory.")
        return

    with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
        terms = [line.strip() for line in f if line.strip()]

    # Filter out any terms that are in skip_terms
    terms_to_process = [t for t in terms if t not in skip_terms]
    print(f"[Info] {len(terms)} total terms in '{TAXONOMY_FILE}'.")
    print(f"[Info] {len(terms_to_process)} terms after skipping general/single terms.")

    # 5) Launch a ThreadPool: one worker per term (up to MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_term = {executor.submit(process_term, term): term for term in terms_to_process}

        # As each thread finishes, report its completion
        for fut in as_completed(future_to_term):
            t = future_to_term[fut]
            try:
                fut.result()
                print(f"[Done] Thread for term '{t}' has finished.")
            except Exception as exc:
                print(f"[Error] Thread for term '{t}' raised an exception: {exc}")

    # 6) At the very end, persist the master index of URLs
    save_master_index(MASTER_INDEX, processed_urls)
    print(f"\n[Done] Updated '{MASTER_INDEX}' with {len(processed_urls)} total URLs.")

if __name__ == "__main__":
    main()
