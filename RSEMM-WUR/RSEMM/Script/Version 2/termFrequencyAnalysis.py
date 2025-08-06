import os
import time
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Configuration ─────────────────────────────────────────────────────────────

TAXONOMY_FILE = "./IEEE_Taxonomy/ieee_taxonomy_flat_L1_L2_filtered.txt"
OUTPUT_COUNT_JSON = "term_counts_sorted_with_total.json"

ZENODO_API_BASE = "https://zenodo.org/api/records"
PAGE_SIZE = 1  # We only need total count → no need to fetch full pages
BASE_SLEEP = 1.0
BACKOFF_SLEEP = 5.0
MAX_WORKERS = 5

# ─── Terms to Skip ──────────────────────────────────────────────────────────────

skip_terms = {
    # Example: "General", "Other"
}

# ─── Helper Functions ─────────────────────────────────────────────────────────

def fetch_total_hits(term):
    """
    Query Zenodo for the total number of *software* results for a term.
    """
    lucene_query = f'{term} AND resource_type.type:"software" AND created:[2022-11-01 TO *]'
    params = {
        "q": lucene_query,
        "page": 1,
        "size": PAGE_SIZE,
    }

    while True:
        try:
            r = requests.get(ZENODO_API_BASE, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"    [Error] network error for term='{term}': {e}")
            return None

        if r.status_code == 429:
            print(f"    [Warning] Rate limit for term='{term}'. Sleeping {BACKOFF_SLEEP}s…")
            time.sleep(BACKOFF_SLEEP)
            continue

        if not r.ok:
            print(f"    [Error] HTTP {r.status_code} for term='{term}'.")
            return None

        resp = r.json()
        hits_obj = resp.get("hits", {})
        total_hits = hits_obj.get("total", 0)
        return total_hits

# ─── Main Workflow ────────────────────────────────────────────────────────────

def main():
    # 1) Read taxonomy terms, skipping the specified ones
    if not os.path.exists(TAXONOMY_FILE):
        print(f"[Error] Cannot find '{TAXONOMY_FILE}'. Make sure it's in the working directory.")
        return

    with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
        terms = [line.strip() for line in f if line.strip()]

    terms_to_process = [t for t in terms if t not in skip_terms]
    print(f"[Info] {len(terms)} total terms in '{TAXONOMY_FILE}'.")
    print(f"[Info] {len(terms_to_process)} terms after skipping.")

    term_counts = {}

    # 2) Launch a ThreadPool: one worker per term (up to MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_term = {executor.submit(fetch_total_hits, term): term for term in terms_to_process}

        # As each thread finishes, store its result
        for fut in as_completed(future_to_term):
            term = future_to_term[fut]
            try:
                count = fut.result()
                if count is not None:
                    term_counts[term] = count
                    print(f"[Done] Term '{term}' → {count} results.")
                else:
                    print(f"[Warning] Skipped term '{term}' due to error.")
            except Exception as exc:
                print(f"[Error] Thread for term '{term}' raised an exception: {exc}")

    # 3) Sort counts descending
    sorted_term_counts = dict(sorted(term_counts.items(), key=lambda item: item[1], reverse=True))

    # 4) Compute total count
    total_repos = sum(sorted_term_counts.values())

    # 5) Prepare final output dict
    output_data = {
        "total_repos": total_repos,
        "terms": sorted_term_counts
    }

    # 6) Write to JSON
    with open(OUTPUT_COUNT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[Done] Wrote term counts and total to '{OUTPUT_COUNT_JSON}'. Total repos: {total_repos}")

if __name__ == "__main__":
    main()
