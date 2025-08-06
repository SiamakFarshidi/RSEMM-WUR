
#!/usr/bin/env python3
"""
detect_ai_code.py

Implements an adapted three‐stage AI‐vs‐Human code detector (originally described in:
“Detecting AI‐Generated Code Assignments Using Perplexity of Large Language Models” (Wang et al., 2023)),
but using a local GPT-2 model to compute perplexity instead of any deprecated OpenAI endpoint.

STAGES:
  1) Per‐line Perplexity Profiling (using GPT-2 from Hugging Face Transformers)
  2) Targeted Masking & Mask‐Filling (using a CodeBERT masked‐LM) on the single highest‐PPL line only
  3) Composite Scoring & Ranking (perplexity, PPL‐std, burstiness)

USAGE:
  - Install dependencies:
      pip install transformers torch numpy tqdm
  - Run on a single code file:
      python detect_ai_code.py path/to/code_file.py
"""

import os
import sys
import math
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModelForMaskedLM
)

# ------------------------------------------------------------------------------
# CONFIGURATION / HYPERPARAMETERS
# ------------------------------------------------------------------------------

# 1) Perplexity Model: GPT-2
PPL_MODEL_NAME = "gpt2"           # small GPT-2 model
PPL_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# 2) Mask‐filling model: CodeBERT (masked-LM)
MASK_FILL_MODEL_NAME = "microsoft/codebert-base"  # pretrained CodeBERT
MASK_FILL_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# 3) Perturbation settings
MASK_PERCENTAGE = 0.05     # fraction of tokens in the selected line to mask
NUM_VARIANTS    = 5        # number of perturbed variants to generate per code file
TOP_P           = 0.9      # nucleus sampling parameter for CodeBERT

# 4) Composite scoring weights (α, β, γ)
WEIGHT_PPL   = 0.6         # weight for full‐file perplexity
WEIGHT_STD   = 0.3         # weight for std of per‐line perplexities
WEIGHT_BURST = 0.1         # weight for burstiness

# ------------------------------------------------------------------------------
# UTILITIES: GPT-2 Perplexity Functions
# ------------------------------------------------------------------------------

class GPT2Perplexity:
    """
    Computes token-level log-probabilities and perplexities using GPT-2,
    truncating inputs that exceed the model's maximum context length.
    """

    def __init__(self, model_name: str = PPL_MODEL_NAME, device: str = None):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.device = device or PPL_DEVICE
        self.model.to(self.device)
        self.model.eval()

        # GPT-2 maximum context length (usually 1024)
        self.max_length = self.tokenizer.model_max_length

    def _compute_neg_log_likelihood(self, input_ids: torch.Tensor) -> float:
        """
        Given token IDs tensor of shape [1, seq_len], compute average negative log-likelihood per token.
        """
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            # outputs.loss is average negative log-likelihood per token
            return outputs.loss.item(), input_ids.size(1)

    def line_perplexity(self, line: str) -> float:
        """
        Computes perplexity of a single line under GPT-2:
          PPL(line) = exp( average negative log-likelihood per token ).
        If the tokenized line exceeds max_length, truncate to max_length tokens.
        Returns float('inf') on empty input.
        """
        text = line.rstrip("\n")
        if text.strip() == "":
            return float("inf")

        # Encode to token IDs (no special tokens)
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 0:
            return float("inf")

        # Truncate if too long
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]

        input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        neg_log_likelihood, token_count = self._compute_neg_log_likelihood(input_ids)
        try:
            ppl = math.exp(neg_log_likelihood)
        except OverflowError:
            ppl = float("inf")
        return ppl

    def full_perplexity(self, code: str) -> float:
        """
        Computes perplexity over the entire code string.
        If tokenized length exceeds max_length, truncate to first max_length tokens.
        """
        text = code.rstrip("\n")
        if text.strip() == "":
            return float("inf")

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 0:
            return float("inf")

        # Truncate if too long
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]

        input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        neg_log_likelihood, token_count = self._compute_neg_log_likelihood(input_ids)
        try:
            ppl = math.exp(neg_log_likelihood)
        except OverflowError:
            ppl = float("inf")
        return ppl

# ------------------------------------------------------------------------------
# UTILITIES: CodeBERT Mask-Filling (for a single line)
# ------------------------------------------------------------------------------

class CodeBERTMaskFiller:
    """
    Wraps a CodeBERT masked-LM for token-level mask filling.
    Provides a method to sample masked tokens in a single line using nucleus (top-p) sampling.
    """

    def __init__(self, model_name: str = MASK_FILL_MODEL_NAME, device: str = MASK_FILL_DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        # CodeBERT maximum context length (512)
        self.max_length = self.tokenizer.model_max_length

    def fill_line_masks(self, masked_line: str, num_samples: int = 1, top_p: float = TOP_P) -> list[str]:
        """
        Given a single line `masked_line` (containing [MASK] tokens), returns a list of
        `num_samples` filled-in lines, sampling each [MASK] via nucleus sampling.
        If tokenized length exceeds max_length, truncate to max_length tokens.
        """
        # Tokenize the masked_line with truncation
        encoding = self.tokenizer(masked_line, return_tensors="pt", truncation=True,
                                  max_length=self.max_length)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        mask_token_id = self.tokenizer.mask_token_id
        # Find positions of [MASK] in this single-line input
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=False).tolist()
        if len(mask_positions) == 0:
            # No [MASK] → just return the line itself repeatedly
            return [masked_line] * num_samples

        variants = []
        for _ in range(num_samples):
            filled_ids = input_ids.clone()
            # Fill each masked position one by one
            for (batch_idx, pos_idx) in mask_positions:
                with torch.no_grad():
                    outputs = self.model(input_ids=filled_ids, attention_mask=attention_mask)
                    logits = outputs.logits[0, pos_idx]  # shape: (vocab_size,)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()

                # Nucleus (top-p) sampling
                sorted_indices = np.argsort(-probs)
                sorted_probs = probs[sorted_indices]
                cumulative_probs = np.cumsum(sorted_probs)
                cutoff = np.searchsorted(cumulative_probs, top_p, side="right")
                top_indices = sorted_indices[: cutoff + 1]
                top_probs = sorted_probs[: cutoff + 1]
                top_probs = top_probs / top_probs.sum()

                sampled_idx = np.random.choice(len(top_indices), p=top_probs)
                sampled_token_id = top_indices[sampled_idx]
                filled_ids[0, pos_idx] = sampled_token_id

            filled_line = self.tokenizer.decode(filled_ids[0], skip_special_tokens=True)
            variants.append(filled_line)

        return variants

# ------------------------------------------------------------------------------
# STAGE 1: Compute Per-Line Perplexities
# ------------------------------------------------------------------------------

def compute_line_perplexities(code: str, ppl_model: GPT2Perplexity) -> list[float]:
    """
    Splits `code` into lines, computes per-line perplexity using GPT-2.
    Returns a list of PPL scores, one per line.
    """
    lines = code.splitlines()
    perp_scores = []
    print("[Stage 1] Computing per-line perplexities…")
    for line in tqdm(lines, desc="Lines", unit="line"):
        ppl = ppl_model.line_perplexity(line)
        perp_scores.append(ppl)
    return perp_scores

# ------------------------------------------------------------------------------
# STAGE 2: Targeted Masking & Generating Variants (on one high-PPL line at a time)
# ------------------------------------------------------------------------------

def mask_line(line: str, mask_pct: float = MASK_PERCENTAGE, tokenizer=None) -> str:
    """
    Given a single `line`, randomly mask `mask_pct` fraction of its tokens.
    Returns the masked_line string (with CodeBERT's [MASK] tokens).
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MASK_FILL_MODEL_NAME)
    tokenized = tokenizer.tokenize(line)
    num_tokens = len(tokenized)
    if num_tokens == 0:
        return line  # nothing to mask

    num_to_mask = max(1, int(num_tokens * mask_pct))
    mask_indices = random.sample(range(num_tokens), num_to_mask)

    masked_tokens = []
    for i, tok in enumerate(tokenized):
        if i in mask_indices:
            masked_tokens.append(tokenizer.mask_token)
        else:
            masked_tokens.append(tok)

    masked_line = tokenizer.convert_tokens_to_string(masked_tokens)
    return masked_line

def generate_variants(code: str, line_perp: list[float], filler: CodeBERTMaskFiller,
                      num_variants: int = NUM_VARIANTS, mask_pct: float = MASK_PERCENTAGE) -> list[str]:
    """
    Given `code` (as a list of lines) and its per-line perplexities `line_perp`,
    generate `num_variants` perturbed variants by:
      1) identifying the single line with max PPL,
      2) masking a fraction of tokens in that line,
      3) filling masks via CodeBERT (nucleus sampling),
      4) replacing that line in the original code to form each variant.
    """
    lines = code.splitlines()
    if not lines or not line_perp:
        return [code] * num_variants

    # Identify index of line with maximum perplexity
    idx_max = int(np.argmax(np.array(line_perp, dtype=float)))
    target_line = lines[idx_max]

    # Mask the target line
    masked_line = mask_line(target_line, mask_pct=mask_pct, tokenizer=filler.tokenizer)

    # Generate filled-in versions of that single line
    filled_lines = filler.fill_line_masks(masked_line, num_samples=num_variants, top_p=TOP_P)

    variants = []
    for filled_line in filled_lines:
        new_lines = lines.copy()
        new_lines[idx_max] = filled_line
        variants.append("\n".join(new_lines))
    return variants

# ------------------------------------------------------------------------------
# STAGE 3: Composite Scoring & Ranking
# ------------------------------------------------------------------------------

def compute_full_perplexity(code: str, ppl_model: GPT2Perplexity) -> float:
    """
    Computes the full-file perplexity of `code` under GPT-2,
    truncating to the model's max length if necessary.
    """
    return ppl_model.full_perplexity(code)

def compute_ppl_std(line_perp: list[float]) -> float:
    """
    Computes the standard deviation of per-line perplexities.
    """
    arr = np.array([p for p in line_perp if not math.isinf(p)], dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.std(arr))

def compute_burstiness(code: str) -> float:
    """
    Simple burstiness: compute token‐frequency variance. Higher variance ⇒ more bursting.
    """
    tokenizer = AutoTokenizer.from_pretrained(MASK_FILL_MODEL_NAME)
    toks = tokenizer.tokenize(code)
    if not toks:
        return 0.0
    freq = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    freqs = np.array(list(freq.values()), dtype=float)
    return float(np.var(freqs))

def composite_score(ppl: float, ppl_std: float, burst: float) -> float:
    """
    Weighted linear combination:
      score = α * ppl + β * ppl_std + γ * burst
    """
    return WEIGHT_PPL * ppl + WEIGHT_STD * ppl_std + WEIGHT_BURST * burst

def detect_ai(code: str, ppl_model: GPT2Perplexity, filler: CodeBERTMaskFiller) -> dict:
    """
    Runs the full detection pipeline on a single `code` string.
    Returns a dict with:
      - original metrics: {perplexity, ppl_std, burstiness, score}
      - variant metrics: list of dicts for each variant
      - verdict: "AI" if original score ≤ min(variant scores), else "Human"
    """
    results = {}

    # Stage 1: Per-line perplexities
    line_perp = compute_line_perplexities(code, ppl_model)

    # Compute original metrics
    orig_ppl      = compute_full_perplexity(code, ppl_model)
    orig_ppl_std  = compute_ppl_std(line_perp)
    orig_burst    = compute_burstiness(code)
    orig_score    = composite_score(orig_ppl, orig_ppl_std, orig_burst)
    results["original"] = {
        "perplexity": orig_ppl,
        "ppl_std":    orig_ppl_std,
        "burstiness": orig_burst,
        "score":      orig_score
    }

    # Stage 2: Generate variants
    variants = generate_variants(code, line_perp, filler,
                                 num_variants=NUM_VARIANTS, mask_pct=MASK_PERCENTAGE)

    # Compute metrics for each variant
    var_metrics = []
    for idx, var_code in enumerate(variants):
        vppl      = compute_full_perplexity(var_code, ppl_model)
        vpll      = compute_line_perplexities(var_code, ppl_model)
        vppl_std  = compute_ppl_std(vpll)
        vburst    = compute_burstiness(var_code)
        vscore    = composite_score(vppl, vppl_std, vburst)
        var_metrics.append({
            "variant_index": idx,
            "perplexity":    vppl,
            "ppl_std":       vppl_std,
            "burstiness":    vburst,
            "score":         vscore
        })
    results["variants"] = var_metrics

    # Stage 3: Decision rule
    variant_scores    = [vm["score"] for vm in var_metrics]
    min_variant_score = min(variant_scores) if variant_scores else float("inf")
    verdict = "AI" if orig_score <= min_variant_score else "Human"
    results["verdict"] = verdict

    return results

# ------------------------------------------------------------------------------
# MAIN: Process a Single File
# ------------------------------------------------------------------------------

def main():
    file_path = r"C:\Users\User\source\repos\SiamakFarshidi\DecisionModelGalaxy\DecisionModelGalaxy\RSEdashboard\Script\detect_ai_code.py"

    if not os.path.isfile(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Initialize models
    print("Loading GPT-2 for perplexity…")
    ppl_model = GPT2Perplexity(model_name=PPL_MODEL_NAME, device=PPL_DEVICE)
    print("Loading CodeBERT for mask-filling…")
    filler = CodeBERTMaskFiller(model_name=MASK_FILL_MODEL_NAME, device=MASK_FILL_DEVICE)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()

    print(f"\nRunning AI-generated code detector on: {file_path}\n")
    detection_results = detect_ai(code, ppl_model, filler)

    print(">>> Original Code Metrics:")
    print(f"    Perplexity      = {detection_results['original']['perplexity']:.3f}")
    print(f"    PPL-Std         = {detection_results['original']['ppl_std']:.3f}")
    print(f"    Burstiness      = {detection_results['original']['burstiness']:.3f}")
    print(f"    Composite Score = {detection_results['original']['score']:.3f}\n")

    print(">>> Variant Metrics (each of the perturbed versions):")
    for vm in detection_results["variants"]:
        idx = vm["variant_index"]
        print(f"  Variant {idx}:")
        print(f"    Perplexity      = {vm['perplexity']:.3f}")
        print(f"    PPL-Std         = {vm['ppl_std']:.3f}")
        print(f"    Burstiness      = {vm['burstiness']:.3f}")
        print(f"    Composite Score = {vm['score']:.3f}\n")

    print(f"*** Verdict: {detection_results['verdict']} (\"AI\" means likely AI-generated) ***\n")


if __name__ == "__main__":
    main()





#file_path = r"C:\Users\User\source\repos\SiamakFarshidi\DecisionModelGalaxy\DecisionModelGalaxy\RSEdashboard\Script\detect_ai_code.py"


#openai_api_key = "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"
