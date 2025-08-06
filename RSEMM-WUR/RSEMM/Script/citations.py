import requests

def get_openalex_id(doi):
    """Resolve DOI to OpenAlex Work ID"""
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    headers = {
        "User-Agent": "DecisionModelGalaxy/1.0 (mailto:siamak.farshidi@gmail.com)"
    }
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()["id"]  # e.g., "https://openalex.org/W1234567890"

def check_openalex(doi):
    """Get all works citing the given DOI"""
    try:
        openalex_id = get_openalex_id(doi)
        base_url = "https://api.openalex.org/works"
        headers = {
            "User-Agent": "DecisionModelGalaxy/1.0 (mailto:siamak.farshidi@gmail.com)"
        }

        all_results = []
        cursor = "*"
        while True:
            params = {
                "filter": f"cites:{openalex_id}",
                "per_page": 25,
                "cursor": cursor,
                "mailto": "siamak.farshidi@gmail.com"
            }
            r = requests.get(base_url, params=params, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()

            for item in data.get("results", []):
                all_results.append({
                    "title": item.get("title"),
                    "doi": item.get("doi"),
                    "id": item.get("id"),
                    "year": item.get("publication_year")                
                    })

            # Pagination check
            next_cursor = data.get("meta", {}).get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        return {
            "status": "success",
            "source": "OpenAlex",
            "citations_found": len(all_results),
            "citations": all_results
        }
    except Exception as e:
        return {
            "status": "error",
            "source": "OpenAlex",
            "error": str(e)
        }


def check_crossref(doi):
    """(Optional) Check Crossref for references to this DOI — not reliable for citations"""
    base_url = "https://api.crossref.org/works"
    query_url = f"{base_url}?filter=reference:{doi}"
    try:
        r = requests.get(query_url, timeout=10)
        r.raise_for_status()
        data = r.json()
        items = data.get("message", {}).get("items", [])
        return {"status": "success", "source": "Crossref", "citations_found": len(items)}
    except Exception as e:
        return {"status": "error", "source": "Crossref", "error": str(e)}

def check_europepmc(doi):
    """Check EuropePMC for mentions of the DOI in publications"""
    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    query = f'doi:"{doi}"'
    query_url = f"{base_url}?query={query}&format=json&pageSize=25"
    try:
        r = requests.get(query_url, timeout=10)
        r.raise_for_status()
        data = r.json()
        hit_count = int(data.get("hitCount", 0))
        return {"status": "success", "source": "EuropePMC", "citations_found": hit_count}
    except Exception as e:
        return {"status": "error", "source": "EuropePMC", "error": str(e)}

def run_checks_for_doi(doi):
    """Unified citation retrieval: prioritizes OpenAlex, falls back to others"""
    
    # Try OpenAlex first
    openalex_result = check_openalex(doi)
    if openalex_result["status"] == "success" and openalex_result.get("citations_found", 0) > 0:
        return {
            "citations_found": openalex_result["citations_found"],
            "citations": openalex_result.get("citations", [])
        }

    # Fallback: EuropePMC
    europepmc_result = check_europepmc(doi)
    if europepmc_result["status"] == "success" and europepmc_result.get("citations_found", 0) > 0:
        return {
            "citations_found": europepmc_result["citations_found"],
            "citations": []  # EuropePMC API does not return full metadata by default
        }

    # Final fallback: Crossref
    crossref_result = check_crossref(doi)
    if crossref_result["status"] == "success" and crossref_result.get("citations_found", 0) > 0:
        return {
            "citations_found": crossref_result["citations_found"],
            "citations": []  # You can enrich this if needed
        }

    # If everything failed or returned 0
    return {
        "citations_found": 0,
        "citations": []
    }


def main():
    doi = "10.5281/zenodo.3588071"
    results = run_checks_for_doi(doi)


    print(results)

if __name__ == "__main__":
    main()
