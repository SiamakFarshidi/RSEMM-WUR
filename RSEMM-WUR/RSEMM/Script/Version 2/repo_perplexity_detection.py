#!/usr/bin/env python3
"""
repo_perplexity_detection.py

Given a GitHub repo URL (hard‐coded via a variable in main), this script:
  1) Downloads the repo tarball (no git clone),
  2) Unpacks it to a temporary directory,
  3) Walks all files looking for “script” extensions (including .py, .js, .m, .r, etc.),
  4) For each script file, reads its contents and runs the AAAI-24 “perplexity + perturbation” detector:
       a) Compute per-line perplexities under text-davinci-003,
       b) Mask a fraction of high-PPL lines,
       c) Generate perturbed variants via CodeBERT,
       d) Compute (overall_ppl, std_line_ppl, burstiness) on original + variants,
       e) Form a composite score and rank the original among variants,
       f) Label “AI” if original is lowest-score, else “Human.”
  5) Tally total script files, AI vs. Human counts, and compute an AI-ratio summary,
  6) Write a JSON report with:
       • "#AllFiles": total number of script files processed,
       • "#Human":   how many labeled “Human,”
       • "#AI":      how many labeled “AI,”
       • "ai_ratio": rounded ratio,
       • "estimation": “High” / “Medium” / “Low” based on ratio thresholds.
  7) Clean up the temporary directory.

Dependencies:
  - openai (pip install openai>=1.0.0)
  - transformers, torch (pip install transformers torch)
  - requests, tqdm, python-dateutil
"""

import os
import re
import sys
import json
import shutil
import stat
import time
import tempfile
import requests
import tarfile
import torch
import openai
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ─────────────────────────────────────────────────────────────────────────────
# 0) Hardcoded GitHub token and OpenAI key
# ─────────────────────────────────────────────────────────────────────────────

GITHUB_TOKEN ="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"
openai.api_key = "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"

if not openai.api_key:
    raise RuntimeError("Please set OPENAI_API_KEY as an environment variable.")

# ─────────────────────────────────────────────────────────────────────────────
# 1) Script-file extensions (include MATLAB, R, Python, etc.)
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_EXTENSIONS = {
    ".py", ".js", ".java", ".cpp", ".c", ".h", ".cs",
    ".ts", ".rb", ".go", ".php", ".rs", ".sh",
    ".m",      # MATLAB
    ".r",      # R
    ".rmd",    # R Markdown
    ".jl",     # Julia
    ".swift",  # Swift
    ".kt",     # Kotlin
    ".ps1"     # PowerShell
}

# ─────────────────────────────────────────────────────────────────────────────
# 2) Helper: Recursively remove a directory, making files writable if needed
# ─────────────────────────────────────────────────────────────────────────────

def on_rm_error(func, path, _):
    """
    Handler for shutil.rmtree to change file permissions and retry.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# 3) Download & unpack the GitHub repo tarball (no `git clone`)
# ─────────────────────────────────────────────────────────────────────────────

def get_default_branch(owner: str, repo: str, token: str = None) -> str:
    """
    GET https://api.github.com/repos/{owner}/{repo} to find the default branch.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        repo_info = resp.json()
        return repo_info.get("default_branch", "main")
    except Exception:
        return "main"

def strip_dot_git_suffix(s: str) -> str:
    """
    If the string ends with ".git", strip it; else return unchanged.
    """
    return s[:-4] if s.lower().endswith(".git") else s

def download_and_unpack_tarball(owner: str, repo: str, branch: str = "main", token: str = None) -> Tuple[str, str]:
    """
    Download https://api.github.com/repos/{owner}/{repo}/tarball/{branch},
    unpack into a temp dir, and return (tmp_root, top_folder_path).
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    url = f"https://api.github.com/repos/{owner}/{repo}/tarball/{branch}"
    resp = requests.get(url, headers=headers, stream=True)
    resp.raise_for_status()

    tmp_tar = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
    for chunk in resp.iter_content(chunk_size=1 << 20):
        tmp_tar.write(chunk)
    tmp_tar.flush()
    tmp_tar.close()

    tmp_dir = tempfile.mkdtemp(prefix="gh_perplexity_")
    with tarfile.open(tmp_tar.name, "r:gz") as tf:
        tf.extractall(path=tmp_dir)

    entries = os.listdir(tmp_dir)
    if len(entries) != 1:
        raise RuntimeError(f"Expected exactly one top-level directory in {tmp_dir}, got {entries!r}")
    top_folder = os.path.join(tmp_dir, entries[0])

    os.unlink(tmp_tar.name)
    return tmp_dir, top_folder

# ─────────────────────────────────────────────────────────────────────────────
# 4) AAAI-24 Detection: Perplexity + Perturbation
# ─────────────────────────────────────────────────────────────────────────────

def get_perplexity_via_openai(code: str) -> float:
    """
    Compute perplexity of `code` under text-davinci-003 by requesting log-probs
    (echo=True) and converting to cross-entropy perplexity.
    """
    resp = openai.Completion.create(
        model="text-davinci-003",
        prompt=code,
        max_tokens=0,
        echo=True,
        logprobs=0
    )
    token_logprobs = resp.choices[0].logprobs.token_logprobs or []
    logps = [lp for lp in token_logprobs if lp is not None]
    if not logps:
        return float("inf")
    avg_logp = np.mean(logps)
    ppl = float(np.exp(-avg_logp))
    return ppl

def get_line_perplexities(code: str) -> List[Tuple[int, float]]:
    """
    Split `code` into lines and compute perplexity for each non-blank line.
    Returns list of (line_index, perplexity).
    """
    lines = code.splitlines()
    line_ppls: List[Tuple[int, float]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            ppl = get_perplexity_via_openai(line + "\n")
        except Exception:
            ppl = float("inf")
        line_ppls.append((idx, ppl))
        time.sleep(0.5)  # avoid hitting rate limits too fast
    return line_ppls

def mask_high_ppl_lines(code: str, line_ppls: List[Tuple[int, float]], mask_fraction: float) -> str:
    """
    Mask the top `mask_fraction` of lines by perplexity. Each masked line is
    replaced by “[MASK]” tokens of roughly the same length.
    """
    lines = code.splitlines()
    sorted_by_ppl = sorted(line_ppls, key=lambda x: x[1], reverse=True)
    n_to_mask = max(1, int(len(sorted_by_ppl) * mask_fraction))
    mask_indices = {idx for idx, _ in sorted_by_ppl[:n_to_mask]}

    masked_lines: List[str] = []
    for i, line in enumerate(lines):
        if i in mask_indices:
            num_tokens = max(1, len(line.split()))
            masked_lines.append(" ".join(["[MASK]"] * num_tokens))
        else:
            masked_lines.append(line)
    return "\n".join(masked_lines)

class CodeBERTMaskFiller:
    """
    Wraps HuggingFace’s CodeBERT (microsoft/codebert-base-mlm) to fill [MASK] tokens.
    """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base-mlm")
        self.model = AutoModelForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    def fill_masks(self, masked_code: str, top_k: int = 5) -> List[str]:
        """
        Given `masked_code` containing one or more [MASK] tokens, produce up to `top_k`
        variants by picking the k-th most likely token for each mask position.
        """
        inputs = self.tokenizer.encode_plus(masked_code, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        mask_token_id = self.tokenizer.mask_token_id
        seq = inputs["input_ids"].squeeze(0)  # [seq_len]
        mask_positions = (seq == mask_token_id).nonzero(as_tuple=True)[0].tolist()

        variants: List[str] = []
        for k in range(top_k):
            filled = seq.clone()
            for pos in mask_positions:
                topk = torch.topk(logits[0, pos], k + 1)
                chosen = topk.indices[k]
                filled[pos] = chosen
            decoded = self.tokenizer.decode(filled, skip_special_tokens=True)
            variants.append(decoded)
        return variants

def compute_metrics(code: str) -> Tuple[float, float, float]:
    """
    Compute:
      1. overall_ppl = get_perplexity_via_openai(code)
      2. std_line_ppl = standard deviation of per-line PPLs
      3. burstiness = std(line_ppls) / mean(line_ppls)
    """
    try:
        overall_ppl = get_perplexity_via_openai(code)
    except Exception:
        overall_ppl = float("inf")

    line_ppls = get_line_perplexities(code)
    ppls = np.array([p for _, p in line_ppls if np.isfinite(p)])
    if ppls.size == 0:
        return overall_ppl, 0.0, 0.0

    std_ppl = float(np.std(ppls))
    mean_ppl = float(np.mean(ppls))
    burstiness = std_ppl / mean_ppl if mean_ppl > 0 else 0.0
    return overall_ppl, std_ppl, burstiness

def composite_score(metrics: Tuple[float, float, float], weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)) -> float:
    """
    α⋅(1/overall_ppl) + β⋅std_line_ppl + γ⋅burstiness
    """
    overall_ppl, std_ppl, burstiness = metrics
    α, β, γ = weights
    inv_overall = 1.0 / (overall_ppl + 1e-8)
    return α * inv_overall + β * std_ppl + γ * burstiness

def detect_ai_generated(code: str, mask_fraction: float = 0.2, n_variants: int = 5) -> str:
    """
    Perform the AAAI-24 pipeline on a single code snippet. Returns "AI" or "Human".
    """
    line_ppls = get_line_perplexities(code)
    masked = mask_high_ppl_lines(code, line_ppls, mask_fraction)
    filler = CodeBERTMaskFiller()
    variants = filler.fill_masks(masked, top_k=n_variants)

    orig_metrics = compute_metrics(code)
    orig_score = composite_score(orig_metrics)

    variant_scores: List[float] = []
    for var in variants:
        m = compute_metrics(var)
        s = composite_score(m)
        variant_scores.append(s)

    all_scores = variant_scores + [orig_score]
    sorted_indices = np.argsort(all_scores)  # ascending
    orig_index = len(all_scores) - 1
    rank = int(np.where(sorted_indices == orig_index)[0][0])

    return "AI" if rank == 0 else "Human"

def detect_multiple_files(script_paths: List[str],
                          mask_fraction: float = 0.2,
                          n_variants: int = 5) -> Dict[str, str]:
    """
    Given a list of file paths (each containing code), run the
    AAAI-24 pipeline on each and return a dict mapping
       filepath → "AI" or "Human".
    """
    results: Dict[str, str] = {}
    for path in script_paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                code = f.read()
            label = detect_ai_generated(code, mask_fraction=mask_fraction, n_variants=n_variants)
            results[path] = label
        except Exception as e:
            print(f"Warning: failed to classify {path}: {e}")
            continue
    return results

# ─────────────────────────────────────────────────────────────────────────────
# 5) Main: download repo, walk script files, detect AI vs Human, and summarize
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # 5.1) Hardcoded GitHub repo URL (no .git suffix needed)
    repo_url = "https://github.com/SiamakFarshidi/curriculum"

    # 5.2) Parse owner and repo_name from URL
    m = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$", repo_url)
    if not m:
        print("Error: could not parse owner/repo from URL:", repo_url)
        sys.exit(1)
    owner = m.group("owner")
    repo_name = strip_dot_git_suffix(m.group("repo"))

    # 5.3) Discover default branch
    default_branch = get_default_branch(owner, repo_name, GITHUB_TOKEN)
    print(f"Default branch for {owner}/{repo_name} = '{default_branch}'")

    # 5.4) Download & unpack the tarball
    print(f"Downloading {owner}/{repo_name} tarball (branch '{default_branch}') …")
    try:
        tmp_root, clone_dir = download_and_unpack_tarball(owner, repo_name, branch=default_branch, token=GITHUB_TOKEN)
    except Exception as e:
        print("ERROR: failed to download or unpack tarball:", e)
        sys.exit(1)

    try:
        # 5.5) Walk all files and collect script file paths
        script_paths: List[str] = []
        for root, _, files in os.walk(clone_dir):
            if ".git" in root.split(os.sep):
                continue
            for fname in files:
                _, ext = os.path.splitext(fname.lower())
                if ext in SCRIPT_EXTENSIONS:
                    full = os.path.join(root, fname)
                    script_paths.append(full)

        total_scripts = len(script_paths)
        print(f"Found {total_scripts} script files to classify.")

        # 5.6) Classify all script files
        file_labels = detect_multiple_files(script_paths)
        human_count = sum(1 for v in file_labels.values() if v == "Human")
        ai_count    = sum(1 for v in file_labels.values() if v == "AI")

        # 5.7) Summarize into JSON
        total = human_count + ai_count
        ai_ratio = (ai_count / total) if total > 0 else 0.0
        if ai_ratio > 0.66:
            estimation = "High"
        elif ai_ratio > 0.33:
            estimation = "Medium"
        else:
            estimation = "Low"

        result = {
            "#AllFiles": total_scripts,
            "#Files":    total,
            "#Human":    human_count,
            "#AI":       ai_count,
            "ai_ratio":  round(ai_ratio, 3),
            "estimation": estimation
        }

        out_path = Path.cwd() / "repo_ai_detection.json"
        with open(out_path, "w", encoding="utf-8") as jf:
            json.dump(result, jf, indent=2)

        print(f"\nResults written to {out_path.resolve()}")
        print(json.dumps(result, indent=2))

    finally:
        # 5.8) Clean up temporary directory
        shutil.rmtree(tmp_root, onerror=on_rm_error)

if __name__ == "__main__":
    main()
