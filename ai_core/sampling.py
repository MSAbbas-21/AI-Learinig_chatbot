"""
Educational Sampling & Softmax Module
=====================================
Explains how raw numbers (Logits) from neural networks are converted into
probabilities and how Temperature, Top-K, and Top-P shape the generated text.

Concepts:
1. Logits: Raw unnormalized confidence scores.
2. Softmax: Converts arbitrary numbers to probabilities summing to 1.0 (100%).
3. Temperature (T): Controls randomness / creativity vs determinism.
4. Top-K: Restricts candidates to the top K most likely words.
5. Top-P (Nucleus): Restricts candidates to the dynamic subset whose cumulative probability >= P.
"""

import math
import random
from typing import List, Dict, Any, Tuple
import numpy as np


def apply_softmax(logits: List[float], temperature: float = 1.0) -> List[float]:
    """
    Computes Softmax with temperature:
    P_i = exp(z_i / T) / sum_j exp(z_j / T)
    """
    temp = max(0.01, float(temperature))
    scaled = np.array(logits, dtype=np.float64) / temp
    
    # Numerical stability trick: subtract max logit before exponentiating
    scaled -= np.max(scaled)
    exp_vals = np.exp(scaled)
    probs = exp_vals / np.sum(exp_vals)
    return [float(p) for p in probs]


def apply_top_k(candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Top-K Filtering: Keep only top K candidates by logit/probability, discard the rest.
    """
    if k <= 0 or k >= len(candidates):
        return candidates

    sorted_cands = sorted(candidates, key=lambda x: x["logit"], reverse=True)
    kept = sorted_cands[:k]
    discarded = sorted_cands[k:]

    results = []
    for c in kept:
        results.append({**c, "kept_by_top_k": True})
    for c in discarded:
        results.append({**c, "kept_by_top_k": False, "probability": 0.0})
    return results


def apply_top_p(candidates: List[Dict[str, Any]], p: float) -> List[Dict[str, Any]]:
    """
    Top-P (Nucleus) Filtering:
    Keep the smallest set of top tokens whose cumulative probability >= P.
    """
    if p >= 1.0:
        return [{**c, "kept_by_top_p": True} for c in candidates]

    # Sort descending by current probability
    sorted_cands = sorted(candidates, key=lambda x: x["probability"], reverse=True)
    
    cum_sum = 0.0
    results = []
    
    for c in sorted_cands:
        if cum_sum < p:
            cum_sum += c["probability"]
            results.append({**c, "kept_by_top_p": True, "cum_prob": round(cum_sum, 4)})
        else:
            results.append({**c, "kept_by_top_p": False, "cum_prob": round(cum_sum, 4), "probability": 0.0})

    return results


def simulate_sampling_pipeline(
    tokens: List[str],
    raw_logits: List[float],
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    seed: int = None
) -> Dict[str, Any]:
    """
    Runs the complete full sampling pipeline step-by-step with educational explanations.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Step 1: Base candidates with raw logits
    candidates = []
    for tok, logit in zip(tokens, raw_logits):
        candidates.append({
            "token": tok,
            "logit": round(float(logit), 3)
        })

    # Sort by raw logits initially
    candidates.sort(key=lambda x: x["logit"], reverse=True)

    # Step 2: Calculate Raw Softmax (T = 1.0)
    raw_probs = apply_softmax([c["logit"] for c in candidates], temperature=1.0)
    for c, p in zip(candidates, raw_probs):
        c["raw_prob"] = round(p, 4)

    # Step 3: Apply Temperature
    temp_probs = apply_softmax([c["logit"] for c in candidates], temperature=temperature)
    for c, p in zip(candidates, temp_probs):
        c["temp_prob"] = round(p, 4)
        c["probability"] = p

    # Step 4: Apply Top-K
    k_filtered = []
    effective_k = min(top_k, len(candidates))
    for i, c in enumerate(candidates):
        is_kept = i < effective_k
        k_filtered.append({**c, "kept_by_top_k": is_kept})

    # Step 5: Apply Top-P on Top-K candidates
    active_for_p = [c for c in k_filtered if c["kept_by_top_k"]]
    # Renormalize among active
    active_sum = sum(c["temp_prob"] for c in active_for_p)
    if active_sum > 0:
        for c in active_for_p:
            c["probability"] = c["temp_prob"] / active_sum

    p_filtered = apply_top_p(active_for_p, top_p)

    # Step 6: Final Renormalization of survivors
    survivors = [c for c in p_filtered if c.get("kept_by_top_p", False)]
    survivor_sum = sum(c["probability"] for c in survivors)
    
    if survivor_sum > 0:
        for c in survivors:
            c["final_prob"] = round(c["probability"] / survivor_sum, 4)
            c["percentage"] = round(c["final_prob"] * 100, 2)
    else:
        # Fallback to top token
        survivors = [candidates[0]]
        survivors[0]["final_prob"] = 1.0
        survivors[0]["percentage"] = 100.0

    # Step 7: Sample a winner
    probs = [c["final_prob"] for c in survivors]
    # Sample from categorical distribution
    chosen_idx = np.random.choice(len(survivors), p=probs)
    winner = survivors[chosen_idx]

    # Educational notes based on parameters
    notes = []
    if temperature < 0.3:
        notes.append("❄️ Very Low Temperature: Distribution is sharpened dramatically. The model will almost always pick the top choice (Greedy Argmax mode). Highly deterministic and repetitive.")
    elif temperature > 1.2:
        notes.append("🔥 High Temperature: Distribution is heavily flattened. Lower probability tokens get a significant chance to be chosen. Very creative, but risks hallucinations or incoherent babbling.")
    else:
        notes.append("⚖️ Balanced Temperature: Good trade-off between coherence and diverse, natural-sounding phrasing.")

    if top_k < len(tokens):
        notes.append(f"✂️ Top-K ({top_k}): Cut off all tokens beyond the top {top_k}, completely eliminating low-scoring nonsense.")

    if top_p < 1.0:
        notes.append(f"🎯 Top-P ({top_p}): Dynamically included only tokens whose cumulative probability reached {int(top_p*100)}%.")

    return {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "candidates": [
            {
                "token": c["token"],
                "logit": c["logit"],
                "raw_prob": c["raw_prob"],
                "temp_prob": c["temp_prob"],
                "final_prob": c.get("final_prob", 0.0),
                "percentage": c.get("percentage", 0.0),
                "is_active": c["token"] in [s["token"] for s in survivors],
                "is_winner": c["token"] == winner["token"]
            }
            for c in candidates
        ],
        "winner": winner["token"],
        "winner_prob": winner.get("final_prob", 0.0),
        "notes": notes
    }
