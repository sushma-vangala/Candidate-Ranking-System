#!/usr/bin/env python3
"""
rank.py — Redrob Hackathon Candidate Ranker (v2)
=================================================
Usage:
    # With precomputed MiniLM embeddings (recommended):
    python rank.py --candidates ./candidates.jsonl \\
                   --embeddings ./embeddings.npy \\
                   --candidate-ids ./candidate_ids.npy \\
                   --jd-embedding ./jd_embedding.npy \\
                   --out ./submission.csv

    # Without embeddings (TF-IDF fallback — still works, less semantic):
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Architecture:
  1. Load candidates + precomputed MiniLM embeddings
  2. Trap detection (honeypots, keyword stuffers, behavioral ghosts)
  3. Hybrid scoring: embedding sim (40%) + skill match (20%) + career fit (20%)
                     + behavioral signals (15%) + education bonus (5%)
  4. Confidence scoring per candidate
  5. Top-100 selection with reasoning and skill checklist
  6. Validated CSV output
"""

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

from ranker.jd_parser import JD_PROFILE
from ranker.scorer import (HybridScorer, compute_skill_match_score,
                           compute_career_fit_score, compute_behavioral_score,
                           compute_education_bonus, compute_confidence)
from ranker.trap_detector import get_trap_multiplier
from ranker.reasoning import generate_reasoning
from ranker.adaptive_weights import compute_adaptive_weights
from ranker.skill_gap import generate_skill_gap_report, format_gap_report_text


def load_candidates(path: str) -> list[dict]:
    p = Path(path)
    print(f"Loading candidates from {p}...")
    candidates = []
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"  Loaded {len(candidates):,} candidates")
    return candidates


def run_pipeline(candidates: list[dict], args) -> list[dict]:
    t0 = time.time()
    n = len(candidates)
    print(f"\n{'='*60}")
    print(f"Redrob Ranker v2 — {n:,} candidates")
    print(f"Weights: {JD_PROFILE['scoring_weights']}")
    print(f"{'='*60}")

    # ── Step 1: Trap detection ────────────────────────────────────────────────
    print("\n[1/4] Trap detection...")
    trap_mults, trap_reasons = [], []
    honeypot_count = 0
    for c in candidates:
        mult, reason = get_trap_multiplier(c)
        trap_mults.append(mult)
        trap_reasons.append(reason)
        if mult == 0.0:
            honeypot_count += 1
    pct = honeypot_count / n * 100
    print(f"  Honeypots detected: {honeypot_count:,} ({pct:.1f}%)")
    print(f"  Keyword stuffers penalized: {sum(1 for r in trap_reasons if 'keyword_stuffer' in r):,}")
    print(f"  Behavioral ghosts penalized: {sum(1 for r in trap_reasons if 'availability' in r):,}")
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ── Step 1b: Compute adaptive weights ────────────────────────────────────
    print("[1b/4] Computing adaptive weights from JD...")
    adapted_weights, fired_signals = compute_adaptive_weights(
        JD_PROFILE["jd_embedding_text"], verbose=True
    )
    # Override scoring_weights in JD_PROFILE for this run
    JD_PROFILE["scoring_weights"] = adapted_weights

    # ── Step 2: Load embeddings / build fallback ──────────────────────────────
    print("\n[2/4] Setting up semantic scorer...")
    scorer = HybridScorer()
    using_embeddings = False

    if args.embeddings and args.candidate_ids and args.jd_embedding:
        using_embeddings = scorer.load_embeddings(
            args.embeddings, args.candidate_ids, args.jd_embedding
        )

    if not using_embeddings:
        print("  Using TF-IDF fallback (run precompute_embeddings.py for better results)")
        scorer.build_tfidf_fallback(candidates)

    print(f"  Semantic mode: {'MiniLM embeddings ✓' if using_embeddings else 'TF-IDF fallback'}")
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ── Step 3: Score all candidates ─────────────────────────────────────────
    print(f"\n[3/4] Scoring {n:,} candidates...")
    results = scorer.score_all(candidates)
    print(f"  Elapsed: {time.time()-t0:.1f}s")

    # ── Step 4: Apply trap penalties + compute confidence ─────────────────────
    print("\n[4/4] Applying trap penalties, computing confidence, ranking...")
    for i, result in enumerate(results):
        tm = trap_mults[i]
        result["trap_mult"] = tm
        result["trap_reason"] = trap_reasons[i]
        result["score"] = result["raw_score"] * tm

        # Confidence score (Problem 8)
        result["confidence"] = compute_confidence(
            result["score"], result["components"], tm
        )

    # Sort: score desc, candidate_id asc for ties
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # Log score distribution
    print(f"  Rank 1:   {results[0]['score']:.4f} ({results[0]['candidate_id']}) "
          f"conf={results[0]['confidence']:.0%}")
    print(f"  Rank 10:  {results[9]['score']:.4f} conf={results[9]['confidence']:.0%}")
    print(f"  Rank 50:  {results[49]['score']:.4f} conf={results[49]['confidence']:.0%}")
    if len(results) >= 100: print(f"  Rank 100: {results[99]['score']:.4f} conf={results[99]['confidence']:.0%}")
    print(f"  Total elapsed: {time.time()-t0:.1f}s")

    return results


def build_submission(results: list[dict], out_path: str):
    """Select top 100, generate reasoning, write CSV."""
    # Ensure 0 honeypots in top 100
    top100 = [r for r in results if r["trap_mult"] > 0.0][:100]
    honeypot_count_top100 = sum(1 for r in results[:100] if r["trap_mult"] == 0.0)
    if honeypot_count_top100 > 0:
        print(f"\n⚠️  Excluded {honeypot_count_top100} honeypots from top 100 (replaced with next-best)")

    assert len(top100) == 100, f"Not enough non-trap candidates! Got {len(top100)}"

    # Honeypot rate check
    hp_rate = sum(1 for r in top100 if r["trap_mult"] == 0.0) / 100
    print(f"\nHoneypot rate: {hp_rate:.1%} "
          f"({'✅ SAFE (<10%)' if hp_rate < 0.10 else '❌ DISQUALIFIED'})")

    # Normalize scores to [0.01, 1.0], keep monotonic
    max_s = top100[0]["score"]
    min_s = top100[-1]["score"]
    rng = max(max_s - min_s, 1e-9)

    rows = []
    for rank, result in enumerate(top100, start=1):
        norm_score = round(0.01 + 0.99 * (result["score"] - min_s) / rng, 6)

        # Ensure monotonic non-increasing
        if rows and norm_score > rows[-1]["score"]:
            norm_score = rows[-1]["score"]

        gap_report = generate_skill_gap_report(
            result["candidate"],
            result.get("matched_skills", []),
            result.get("missing_skills", []),
            result["components"].get("skill_match", 0),
        )
        reasoning = generate_reasoning(
            candidate=result["candidate"],
            rank=rank,
            score=norm_score,
            components=result["components"],
            matched_skills=result.get("matched_skills", []),
            missing_skills=result.get("missing_skills", []),
            confidence=result["confidence"],
            progression_explanation=result.get("progression_explanation", ""),
        )

        rows.append({
            "candidate_id": result["candidate_id"],
            "rank": rank,
            "score": norm_score,
            "reasoning": reasoning,
        })

    # Final monotonic check
    for i in range(1, len(rows)):
        if rows[i]["score"] > rows[i-1]["score"]:
            rows[i]["score"] = rows[i-1]["score"]

    # Write CSV
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Submission written: {out_path}")
    print("\nTop 10 candidates:")
    for r in rows[:10]:
        c = r["candidate_id"]
        p = next((res["candidate"]["profile"] for res in [
            res for res in [] + [{"candidate": {"profile": {}}}]
        ] if res["candidate_id"] == c), {}) if False else {}
        print(f"  Rank {r['rank']:3d} | {r['candidate_id']} | "
              f"score={r['score']:.4f} | {r['reasoning'][:90]}...")


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker v2")
    parser.add_argument("--candidates", default="data/candidates.jsonl")
    parser.add_argument("--embeddings", default="embeddings/embeddings.npy",
                        help="Precomputed candidate embeddings (.npy)")
    parser.add_argument("--candidate-ids", default="embeddings/candidate_ids.npy",
                        help="Candidate IDs matching embeddings (.npy)")
    parser.add_argument("--jd-embedding", default="embeddings/jd_embedding.npy",
                        help="JD embedding (.npy)")
    parser.add_argument("--out", default="outputs/submission.csv")
    args = parser.parse_args()

    t_start = time.time()
    candidates = load_candidates(args.candidates)
    results = run_pipeline(candidates, args)
    build_submission(results, args.out)

    elapsed = time.time() - t_start
    print(f"\n⏱️  Total runtime: {elapsed:.1f}s "
          f"({'✅ within 5-min limit' if elapsed < 300 else '⚠️ OVER 5 MIN'})")


if __name__ == "__main__":
    main()
