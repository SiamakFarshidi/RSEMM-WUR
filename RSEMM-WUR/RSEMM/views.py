import os
import re
import time
import uuid
import json
import logging
import requests
import openai  # Make sure to pip install openai

import glob

from urllib.parse import urlparse

from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseNotAllowed
from django.views.decorators.csrf import csrf_exempt

from typing import Optional             # ← add this

from RSEMM.Script.three_itetarive_se_practises_evaluation import getSEestimation
from RSEMM.Script.four_itetarive_ai_ml_ops_evaluation import getAIMLOPsEvaluation
from RSEMM.Script.five_itetarive_fairness_assessment import getFAIRnessAssessment
from RSEMM.Script.six_itetarive_code_gen_detection import getGenCodeUsageEvaluation
from RSEMM.Script.citations import run_checks_for_doi



from django.views.decorators.http import require_GET


logger = logging.getLogger(__name__)

# ─── Zenodo Fetch Configuration ─────────────────────────────────────────────────

# relative to BASE_DIR
ZENODO_STORAGE_DIR = getattr(settings, "ZENODO_STORAGE_DIR", "zenodo_records")
ZENODO_API_BASE    = "https://zenodo.org/api/records"

MAX_RETRIES    = 3
BACKOFF_FACTOR = 5.0  # seconds * attempt number on 429

# ─── Helpers ────────────────────────────────────────────────────────────────────

def _extract_record_id(url: str) -> Optional[str]:
    """
    Pull the numeric record ID out of a Zenodo URL/DOI.
    Returns the ID string or None.
    """
    # Support both /record/12345 and /records/12345
    m = re.search(r"/records?/(\d+)", url)
    if m:
        return m.group(1)
    # Also support DOIs
    m = re.search(r"zenodo\.(\d+)", url)
    if m:
        return m.group(1)
    return None


def fetch_and_store_zenodo_json(repo_url: str) -> str:
    rec_id = _extract_record_id(repo_url)
    if not rec_id:
        raise ValueError(f"Cannot parse Zenodo ID from '{repo_url}'")

    out_dir = os.path.join(settings.BASE_DIR, 'RSEMM', ZENODO_STORAGE_DIR)
    os.makedirs(out_dir, exist_ok=True)

    # Use stable filename → one file per record ID
    fname = f"{rec_id}.json"
    save_fp = os.path.join(out_dir, fname)

    # If file already exists, don't download again
    if os.path.exists(save_fp):
        logger.info("Zenodo record %s already exists → %s", rec_id, save_fp)
        return save_fp

    # Fetch from API
    api_url = f"{ZENODO_API_BASE}/{rec_id}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(api_url, timeout=30)
        except requests.RequestException as e:
            wait = BACKOFF_FACTOR * attempt
            logger.warning("Network error on attempt %s: %s — retrying in %ss", attempt, e, wait)
            time.sleep(wait)
            continue

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.isdigit() else BACKOFF_FACTOR * attempt
            logger.warning("Zenodo 429, Retry-After=%s — sleeping %ss", retry_after, wait)
            time.sleep(wait)
            continue

        resp.raise_for_status()
        break
    else:
        raise requests.HTTPError(f"Failed to fetch after {MAX_RETRIES} retries")

    record = resp.json()
    with open(save_fp, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    logger.info("Saved Zenodo record %s → %s", rec_id, save_fp)
    return save_fp



# ─── Your Existing Views ────────────────────────────────────────────────────────

def landing_page(request):
    """
    Render the landing page for the Project Maturity Estimator.
    """
    return render(request, 'index.html')

@csrf_exempt
def estimate_maturity(request):
    # DEBUG: make sure we’re hitting this view
    logger.debug("estimate_maturity called; headers=%r", request.headers)
    logger.debug("Raw body: %r", request.body)

    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])

    # parse JSON (or bail)
    try:
        payload = json.loads(request.body.decode('utf-8') or "{}")
    except ValueError:
        return JsonResponse({'error': 'Invalid JSON.'}, status=400)

    # accept either key
    raw_url = payload.get('url') or payload.get('repo_url', '')
    url     = raw_url.strip()

    if not url:
        return JsonResponse({ 'error': 'Missing "url" parameter.' }, status=400)

    # only allow GitHub or Zenodo
    hostname = (urlparse(url).hostname or "").lower()
    if hostname not in ('github.com','www.github.com','zenodo.org','www.zenodo.org','doi.org','www.doi.org'):
        return JsonResponse({'error': 'URL must be a GitHub or Zenodo link.'}, status=400)



    # ─── If it's Zenodo, fetch & store the JSON ────────────────────────────────
    if 'zenodo.org' in hostname or 'doi.org' in hostname:
        try:
            save_fp = fetch_and_store_zenodo_json(url)
            getSEestimation(save_fp)
            getAIMLOPsEvaluation(save_fp)
            getFAIRnessAssessment(save_fp)
            getGenCodeUsageEvaluation(save_fp)

            json_path = save_fp

        except ValueError as ve:
            return JsonResponse({ 'error': str(ve) }, status=400)
        except requests.HTTPError as he:
            logger.exception("HTTP error fetching Zenodo record")
            return JsonResponse({ 'error': f'Upstream error: {he}' }, status=400)
        except Exception as e:
            logger.exception("Unexpected error fetching Zenodo")
            return JsonResponse({ 'error': f'Error fetching Zenodo record: {e}' }, status=500)
    
    # ===== Your black-boxed maturity logic lives here =====

    try:
        response_data,json_data = estimate_project_maturity(url, json_path)
    except Exception as e:
        logger.exception("estimate_project_maturity failed")
        return JsonResponse({'error': f'Unexpected error: {e}'}, status=500)

    return JsonResponse({"HighLevelEstimation": response_data, "DetailedEstimation":json_data})


def estimate_project_maturity(url: str, json_path: str) -> dict:
    """
    Load the Zenodo record JSON and calculate a maturity score + populate fields.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Initialize score parts (you can adjust weights)
    score = 0
    max_score = 100

    # Extract fields:
    se = json_data.get('repo-profile', {}).get('software-engineering-efforts', {})
    dimensions = se.get('dimensions', {})
    codegen = json_data.get('repo-profile', {}).get('code-gen-evaluation', {})
    ai_ml_ops = json_data.get('repo-profile', {}).get('ai-ml-ops', {})
    fair = json_data.get('repo-profile', {}).get('fair-assessment', {})
    fair_cats = fair.get('principle_categories', {})

    # Build response
    result = {
        'score': 0,   # will set below
        'message': "Calculated maturity score.",
        'zenodo_url': json_data.get('doi_url'),
        'github_url': se.get('repo'),

        'community': dimensions.get('community', {}).get('estimation'),
        'ci': dimensions.get('ci', {}).get('estimation'),
        'documentation': dimensions.get('documentation', {}).get('estimation'),
        'history': dimensions.get('history', {}).get('estimation'),
        'issues': dimensions.get('issues', {}).get('estimation'),
        'license': dimensions.get('license', {}).get('estimation'),
        'unittest': dimensions.get('unittest', {}).get('estimation'),

        'codegen_estimation': codegen.get('estimation'),
        'ai_detected': 'yes' if (codegen.get('ai_ratio', 0) > 0) else 'no',

        'category': ai_ml_ops.get('category'),
        'types': ', '.join(ai_ml_ops.get('types', [])) if isinstance(ai_ml_ops.get('types'), list) else ai_ml_ops.get('types'),

        'findable': fair_cats.get('Findable'),
        'accessible': fair_cats.get('Accessible'),
        'interoperable': fair_cats.get('Interoperable'),
        'reusable': fair_cats.get('Reusable'),
        'overall_fairness': fair.get('overall_fairness'),
        'howfairis_score': fair.get('howfairis_score'),
    }

    recommendation = maybe_generate_recommendation(json_data, json_path)
    result['recommendation'] = recommendation

    citations = maybe_check_doi_citations(json_data, json_path)
    result['citations'] = citations


    # Example simple scoring logic (you can design your own):
    # Each "high" estimation in SE gives +10 points
    for key in ['community', 'ci', 'documentation', 'history', 'issues', 'license', 'unittest']:
        if result[key] == 'high':
            score += 10

    # Codegen estimation (Low = better?)
    if result['codegen_estimation'] == 'Low':
        score += 10

    # AI detected (penalize AI-generated code)
    if result['ai_detected'] == 'no':
        score += 5

    # FAIR principles
    fair_mapping = {'High': 10, 'Medium': 5, 'Low': 0}
    for key in ['findable', 'accessible', 'interoperable', 'reusable']:
        score += fair_mapping.get(result[key], 0)

    # Howfairis score: up to 10 points
    try:
        hf_score = int(result['howfairis_score'])
        score += min(hf_score, 10)
    except (ValueError, TypeError):
        pass

    # Clamp score to 0-100
    score = max(0, min(score, max_score))
    result['score'] = score

    return result,json_data


def list_stored_zenodo_records(request):
    out_dir = os.path.join(settings.BASE_DIR, 'RSEMM', ZENODO_STORAGE_DIR)
    json_files = glob.glob(os.path.join(out_dir, '*.json'))

    records = []
    for json_fp in json_files:
        try:
            with open(json_fp, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            estimation, _ = estimate_project_maturity("", json_fp)

            metadata = json_data.get('metadata', {})
            authors = metadata.get('creators', [])
            author_names = ", ".join([a.get('name', '') for a in authors])

            # Safe affiliations extraction
            affiliations = []
            for a in authors:
                aff_data = a.get('affiliation')
                if isinstance(aff_data, list):
                    for aff in aff_data:
                        if isinstance(aff, dict):
                            affiliations.append(aff.get('name', ''))
                        elif isinstance(aff, str):
                            affiliations.append(aff)
                elif isinstance(aff_data, dict):
                    affiliations.append(aff_data.get('name', ''))
                elif isinstance(aff_data, str):
                    affiliations.append(aff_data)
            affiliations_text = ", ".join(affiliations)

            # Topics (Keywords) — prefer "keywords", fallback to "subjects"
            if 'keywords' in metadata and metadata['keywords']:
                topics = ", ".join(metadata['keywords'])
            elif 'subjects' in metadata and metadata['subjects']:
                topics = ", ".join([s.get('term', '') for s in metadata['subjects']])
            else:
                topics = ''
            # Description (snippet)
            description_full = metadata.get('description', '')
            description_short = description_full[:200] + ('...' if len(description_full) > 200 else '')

            record = {
                'filename': os.path.basename(json_fp),  # Needed for load button
                'zenodo_url': estimation.get('zenodo_url', ''),
                'github_url': estimation.get('github_url', ''),
                'title': metadata.get('title', 'Unknown'),
                'authors': author_names,
                'affiliations': affiliations_text,
                'topics': topics,
                'score': estimation.get('score', 0)
            }
            records.append(record)
        except Exception as e:
            logger.exception("Error processing %s", json_fp)
            continue

    records.sort(key=lambda r: r['score'], reverse=True)

    return JsonResponse({'records': records})

@require_GET
def load_stored_record(request, filename):
    try:
        out_dir = os.path.join(settings.BASE_DIR, 'RSEMM', ZENODO_STORAGE_DIR)
        json_path = os.path.join(out_dir, filename)

        if not os.path.isfile(json_path):
            return JsonResponse({'error': 'File not found.'}, status=404)

        # Use the same estimation function
        response_data, json_data = estimate_project_maturity("", json_path)

        return JsonResponse({"HighLevelEstimation": response_data, "DetailedEstimation": json_data})

    except Exception as e:
        logger.exception("Error loading stored record %s", filename)
        return JsonResponse({'error': f'Unexpected error: {e}'}, status=500)

#---------------------------------------------------------------------------------------

def maybe_generate_recommendation(json_data, json_path):
    try:
        if 'recommendation' in json_data.get('repo-profile', {}):
            return json_data['repo-profile']['recommendation']
        
        # Construct prompt based on maturity results
        prompt = f"""You are assessing a software repository for best practices. Based on the following metadata, provide 5–7 concise, actionable recommendations to improve software engineering maturity and FAIRness:
Community: {json_data['repo-profile']['software-engineering-efforts']['dimensions']['community']['estimation']}
CI: {json_data['repo-profile']['software-engineering-efforts']['dimensions']['ci']['estimation']}
Documentation: {json_data['repo-profile']['software-engineering-efforts']['dimensions']['documentation']['estimation']}
Unit Testing: {json_data['repo-profile']['software-engineering-efforts']['dimensions']['unittest']['estimation']}
License: {json_data['repo-profile']['software-engineering-efforts']['dimensions']['license']['estimation']}
Findable: {json_data['repo-profile']['fair-assessment']['principle_categories']['Findable']}
Reusable: {json_data['repo-profile']['fair-assessment']['principle_categories']['Reusable']}
AI Detected: {'Yes' if json_data['repo-profile']['code-gen-evaluation'].get('ai_ratio', 0) > 0 else 'No'}
Codegen Estimation: {json_data['repo-profile']['code-gen-evaluation']['estimation']}
"""
        openai.api_key = os.getenv("OPENAI_API_KEY", "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")

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{ "role": "user", "content": prompt }],
            temperature=0.6,
            max_tokens=500,
        )
        recommendations = response['choices'][0]['message']['content'].strip()

        # Save back into JSON
        json_data['repo-profile']['recommendation'] = recommendations
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        return recommendations

    except Exception as e:
        logger.exception("Failed to generate recommendation")
        return "Recommendation could not be generated at this time."



#---------------------------------------------------------------------------------------

def maybe_check_doi_citations(json_data, json_path):
    try:
        # Avoid re-fetching if citations already exist
        if 'citations' in json_data.get('repo-profile', {}):
            return json_data['repo-profile']['citations']

        # Make sure DOI exists in record
        doi = json_data.get('doi') or json_data.get('metadata', {}).get('doi')
        if not doi:
            raise ValueError("DOI not found in Zenodo record.")

        # Fetch citations using unified structure
        citations = run_checks_for_doi(doi)

        # Store back into JSON
        json_data.setdefault('repo-profile', {})['citations'] = citations
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        return citations

    except Exception as e:
        logger.exception("Failed to retrieve or store citations")
        return {
            "citations_found": 0,
            "citations": [],
            "error": "citations could not be found at this time"
        }
