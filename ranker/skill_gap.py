"""
skill_gap.py — Skill gap analysis report
=========================================
Original feature: generates a structured skill gap report for each candidate
showing exactly which required skills they have, which they're missing,
and how much each gap cost them in score points.

This gives recruiters actionable information:
  "This candidate is missing BM25 and hybrid search (cost: -0.042 score points).
   If they have 2 weeks to upskill, they become a top-5 fit."

Output per candidate:
  {
    "matched": [{"skill": "embeddings", "match_type": "direct", "depth": 0.92}],
    "missing": [{"skill": "bm25", "score_cost": 0.021, "learnability": "medium"}],
    "coverage_pct": 68.4,
    "skill_score": 0.61,
    "top_gap": "bm25",
    "gap_severity": "moderate"
  }

Learnability ratings (heuristic, domain knowledge):
  High   = tool/library (can learn in days — e.g. switching Pinecone→Qdrant)
  Medium = concept with practical depth (weeks — e.g. BM25 tuning, NDCG eval)
  Low    = fundamental understanding (months — e.g. building retrieval from scratch)
"""

from ranker.jd_parser import JD_PROFILE


# ── Learnability of each must-have skill ──────────────────────────────────────
# High = tool swap (easy), Medium = concept (moderate), Low = foundational (hard)
LEARNABILITY = {
    "embeddings":           "low",    # foundational concept
    "vector search":        "medium", # concept + implementation
    "retrieval":            "low",    # foundational
    "ranking":              "low",    # foundational
    "sentence-transformers":"high",   # library, easy to pick up
    "faiss":                "high",   # library
    "pinecone":             "high",   # managed service, API-level
    "weaviate":             "high",   # managed service
    "qdrant":               "high",   # managed service
    "milvus":               "high",   # library/service
    "opensearch":           "medium", # ops + search concepts
    "elasticsearch":        "medium", # ops + search concepts
    "hybrid search":        "medium", # concept + tuning
    "ndcg":                 "medium", # metric, needs eval framework understanding
    "mrr":                  "medium", # metric
    "map":                  "medium", # metric
    "evaluation framework": "medium", # design + implementation
    "bm25":                 "medium", # algorithm understanding
    "information retrieval":"low",    # foundational
    "python":               "high",   # language, assumed known
    "semantic search":      "medium", # concept + implementation
    "lora":                 "medium", # technique
    "qlora":                "medium", # technique
    "peft":                 "medium", # framework
    "fine-tuning":          "low",    # deep ML concept
    "learning to rank":     "low",    # algorithmic depth
    "xgboost":              "high",   # library
}


def generate_skill_gap_report(candidate: dict,
                               matched_skills: list,
                               missing_skills: list,
                               skill_score: float) -> dict:
    """
    Generates a structured skill gap report for a single candidate.

    Args:
        candidate: full candidate dict
        matched_skills: list of match strings from compute_skill_match_score
        missing_skills: list of miss strings from compute_skill_match_score
        skill_score: the final skill component score [0,1]

    Returns:
        Structured dict with gap analysis.
    """
    must_haves = JD_PROFILE["must_have_skills"]
    n_must = len(must_haves)

    # ── Parse matched skills ──────────────────────────────────────────────────
    parsed_matched = []
    for m in matched_skills:
        if m.startswith("✓"):
            skill_name = m.replace("✓", "").split("(via")[0].strip()
            match_type = "direct"
        elif m.startswith("≈"):
            skill_name = m.replace("≈", "").split("(")[0].strip()
            match_type = "synonym" if "synonym" in m else "description"
        else:
            skill_name = m
            match_type = "unknown"

        parsed_matched.append({
            "skill": skill_name,
            "match_type": match_type,
            "display": m,
        })

    # ── Parse missing skills ──────────────────────────────────────────────────
    # Score cost = how much this gap reduced the skill_match component
    # Approximation: each must-have contributes equally to the must-have coverage score
    must_missing = [m for m in missing_skills if "✗" in m]
    per_skill_cost = (1.0 - skill_score) / max(len(must_missing), 1) * 0.5

    parsed_missing = []
    for m in must_missing:
        skill_name = m.replace("✗", "").strip()
        learn = LEARNABILITY.get(skill_name.lower(), "medium")
        parsed_missing.append({
            "skill": skill_name,
            "score_cost": round(per_skill_cost, 4),
            "learnability": learn,
            "display": m,
        })

    # ── Coverage ──────────────────────────────────────────────────────────────
    n_matched = sum(1 for m in matched_skills if m.startswith("✓") or m.startswith("≈"))
    coverage_pct = round(n_matched / max(n_must, 1) * 100, 1)

    # ── Gap severity ─────────────────────────────────────────────────────────
    n_critical_missing = sum(
        1 for m in parsed_missing
        if m["learnability"] == "low"  # foundational gaps = most severe
    )
    if n_critical_missing >= 3:
        gap_severity = "critical"
    elif n_critical_missing >= 1 or len(parsed_missing) >= 5:
        gap_severity = "moderate"
    elif len(parsed_missing) >= 2:
        gap_severity = "minor"
    else:
        gap_severity = "minimal"

    # ── Top gap (most impactful missing skill) ────────────────────────────────
    top_gap = None
    if parsed_missing:
        # Prioritize low-learnability (foundational) gaps
        low_learn = [m for m in parsed_missing if m["learnability"] == "low"]
        top_gap = (low_learn or parsed_missing)[0]["skill"]

    # ── Upskilling note ───────────────────────────────────────────────────────
    high_learn_missing = [m["skill"] for m in parsed_missing if m["learnability"] == "high"]
    if high_learn_missing and len(parsed_missing) <= 3:
        upskill_note = (f"Gap is primarily in {', '.join(high_learn_missing[:2])} "
                        f"(high learnability — days to upskill). Strong candidate to develop.")
    elif len(parsed_missing) <= 2:
        upskill_note = "Near-complete skill coverage. Minor gaps only."
    elif gap_severity == "critical":
        upskill_note = "Foundational ML/IR gaps present. Significant ramp-up required."
    else:
        upskill_note = f"Missing {len(parsed_missing)} of {n_must} required skills."

    return {
        "matched": parsed_matched,
        "missing": parsed_missing,
        "coverage_pct": coverage_pct,
        "n_matched": n_matched,
        "n_must": n_must,
        "skill_score": round(skill_score, 4),
        "top_gap": top_gap,
        "gap_severity": gap_severity,
        "upskill_note": upskill_note,
        "critical_missing_count": n_critical_missing,
    }


def format_gap_report_text(report: dict, candidate_id: str, rank: int) -> str:
    """Formats a gap report as a readable text block for logging/display."""
    lines = [
        f"── Skill Gap: {candidate_id} (Rank {rank}) ──────────────────",
        f"Coverage: {report['coverage_pct']}% ({report['n_matched']}/{report['n_must']} required skills)",
        f"Severity: {report['gap_severity'].upper()} | Top gap: {report['top_gap'] or 'none'}",
        f"Note: {report['upskill_note']}",
        "",
        "Matched:",
    ]
    for m in report["matched"][:5]:
        lines.append(f"  {m['display']}")
    if len(report["matched"]) > 5:
        lines.append(f"  ... and {len(report['matched'])-5} more")

    lines.append("Missing:")
    for m in report["missing"][:5]:
        lines.append(f"  {m['display']} | learnability: {m['learnability']} | cost: -{m['score_cost']:.4f}")
    if not report["missing"]:
        lines.append("  (none — full must-have coverage)")

    return "\n".join(lines)
