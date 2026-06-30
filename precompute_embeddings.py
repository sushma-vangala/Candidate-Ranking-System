#!/usr/bin/env python3
"""
precompute_embeddings.py
========================
Run this ONCE on a machine with internet access and sentence-transformers installed.

    pip install sentence-transformers
    python precompute_embeddings.py --candidates ./candidates.jsonl

Saves:
    embeddings.npy          — shape (N, 384), float32, L2-normalized MiniLM embeddings
    candidate_ids.npy       — shape (N,), string array of candidate_id in same order
    jd_embedding.npy        — shape (1, 384), the JD embedding

These files are loaded by rank.py at scoring time (no network needed).

Model: all-MiniLM-L6-v2 (22 MB, fast, good semantic quality for tech domain)
Why MiniLM: 384-dim, ~14k sentences/sec on CPU, captures synonym relationships
like "sentence-transformers" ↔ "SBERT" ↔ "bi-encoder" that TF-IDF misses.
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("ERROR: sentence-transformers not installed.")
    print("Run: pip install sentence-transformers")
    exit(1)

from ranker.jd_parser import JD_PROFILE


# ── Candidate text builder ────────────────────────────────────────────────────
def build_candidate_text(c: dict) -> str:
    """
    Builds a rich semantic text representation of a candidate.
    Key design choices:
    - Recent roles weighted 3x (most relevant)
    - Advanced/expert skills repeated for emphasis
    - Job descriptions included (not just titles) — captures actual work done
    - Certifications included (e.g., AWS ML Specialty = signal)
    """
    p = c.get("profile", {})
    parts = []

    # Profile headline + summary (core identity)
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))
    parts.append(p.get("current_title", ""))
    parts += [p.get("current_industry", "")]

    # Career history — weight recent roles more
    for i, job in enumerate(c.get("career_history", [])[:6]):
        weight = max(1, 4 - i)  # role 0 = 4x, role 1 = 3x, ..., role 3+ = 1x
        parts += [job.get("title", "")] * weight
        desc = job.get("description", "")
        parts.append(desc[:500])
        parts += [job.get("industry", "")]
        parts += [job.get("company", "")]

    # Skills — weight by proficiency
    for s in c.get("skills", []):
        name = s.get("name", "")
        prof = s.get("proficiency", "")
        weight = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(prof, 1)
        parts += [name] * weight

    # Education
    for edu in c.get("education", []):
        parts += [edu.get("field_of_study", ""), edu.get("degree", "")]

    # Certifications (ML certs are strong signals)
    for cert in c.get("certifications", []):
        parts += [cert.get("name", ""), cert.get("issuer", "")]

    return " ".join(x for x in parts if x).strip()


# ── JD text for embedding ─────────────────────────────────────────────────────
JD_EMBED_TEXT = JD_PROFILE["jd_embedding_text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/candidates.jsonl",
                        help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model name (default: all-MiniLM-L6-v2)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", default=".",
                        help="Directory to save .npy files (default: current dir)")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load candidates ───────────────────────────────────────────────────────
    print(f"Loading candidates from {args.candidates}...")
    p = Path(args.candidates)
    opener = gzip.open if p.suffix == ".gz" else open
    candidates = []
    with opener(p, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"  Loaded {len(candidates):,} candidates")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    model = SentenceTransformer(args.model)
    model.max_seq_length = 256
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")

    # ── Embed JD ─────────────────────────────────────────────────────────────
    print("\nEmbedding JD...")
    jd_emb = model.encode([JD_EMBED_TEXT], normalize_embeddings=True,
                           show_progress_bar=False)
    np.save(out / "embeddings/jd_embedding.npy", jd_emb.astype(np.float32))
    print(f"  JD embedding shape: {jd_emb.shape}")

    # ── Build candidate texts ─────────────────────────────────────────────────
    print("\nBuilding candidate texts...")
    t0 = time.time()
    texts = [build_candidate_text(c) for c in candidates]
    ids = [c["candidate_id"] for c in candidates]
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Embed all candidates in batches ───────────────────────────────────────
    print(f"\nEmbedding {len(texts):,} candidates (batch_size={args.batch_size})...")
    t0 = time.time()
    embs = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,  # L2-normalize → cosine sim = dot product
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(texts)/elapsed:.0f} candidates/sec)")
    print(f"  Embedding matrix: {embs.shape} ({embs.nbytes/1024/1024:.1f} MB)")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(out / "embeddings/embeddings.npy", embs.astype(np.float32))
    np.save(out / "embeddings/candidate_ids.npy", np.array(ids))
    print(f"\n✅ Saved to {out}/")
    print(f"   embeddings.npy      — {embs.shape}")
    print(f"   candidate_ids.npy   — {len(ids):,} IDs")
    print(f"   jd_embedding.npy    — {jd_emb.shape}")
    print(f"\nNext: python rank.py --candidates {args.candidates} --embeddings {out}/embeddings/embeddings.npy ...")


if __name__ == "__main__":
    main()
