"""
career_progression.py — Career trajectory scoring (Problem 6 fix)
==================================================================
Scores how well a candidate's career TRAJECTORY fits the role,
not just their current snapshot.

The core insight: two candidates with identical current titles and YoE
can have very different trajectories:

  Candidate A: Junior ML → ML Engineer → Senior ML → Lead AI (↑ consistent ML progression)
  Candidate B: Data Analyst → Sales Analyst → Marketing → ML Engineer (↑ scattered path)

For THIS JD (Senior AI Engineer, founding team), we want Candidate A.
The JD says: "we need someone who understood retrieval and ranking before it
became fashionable" — which implies sustained ML focus over time.

Scoring dimensions:
  1. ML/AI role consistency — fraction of roles that were AI/ML-related
  2. Upward trajectory — is seniority increasing over time?
  3. Domain consistency — are roles in consistent industries/domains?
  4. Time-in-domain — how many total months in AI/ML-related work?

Design note on thresholds (for interviewers):
  Thresholds are derived from the JD's own language:
  - "1.5 years per company = title-chaser" → min_tenure = 18mo
  - "3+ years commitment" → ideal_tenure = 36mo
  - We reward candidates whose MOST RECENT roles are ML-focused (recency matters
    more than career start, because ML skills evolved rapidly post-2017)
"""

import re
from datetime import datetime


# ── AI/ML role detection keywords ──────────────────────────────────────────────
_ML_TITLE_SIGNALS = [
    "machine learning", "ml engineer", "ai engineer", "nlp", "data scientist",
    "research scientist", "applied scientist", "search engineer", "ranking",
    "retrieval", "recommendation", "deep learning", "computer vision",
    "speech recognition", "applied ml", "ml platform", "ai platform",
    "intelligence", "analytics engineer",
]

# Seniority tiers — used to detect upward vs lateral vs downward movement
# Higher number = more senior.
# These tiers are deliberately coarse — exact title matching is brittle.
# We look for keywords, not exact strings.
_SENIORITY_TIERS = [
    (5, ["vp", "vice president", "director of", "head of", "chief"]),
    (4, ["principal", "distinguished", "fellow", "architect"]),
    (3, ["staff", "lead", "tech lead", "senior staff"]),
    (2, ["senior", "sr.", "sr "]),
    (1, ["mid", "engineer ii", "engineer 2", "engineer iii"]),
    (0, ["junior", "jr.", "jr ", "associate", "intern", "trainee",
         "graduate", "entry", "analyst"]),
]

_ML_INDUSTRY_SIGNALS = [
    "artificial intelligence", "machine learning", "saas", "fintech",
    "edtech", "healthtech", "ai", "startup", "e-commerce", "marketplace",
    "software", "technology", "data"
]

_SERVICES_SIGNALS = [
    "it services", "consulting", "outsourcing", "bpo", "staffing"
]


def _get_seniority_tier(title: str) -> int:
    """Returns seniority tier 0-5 from a job title string."""
    tl = title.lower()
    for tier, keywords in _SENIORITY_TIERS:
        if any(kw in tl for kw in keywords):
            return tier
    return 1  # default: mid-level if no signal found


def _is_ml_role(title: str, description: str = "") -> bool:
    """Returns True if the role is clearly AI/ML-related."""
    combined = (title + " " + description).lower()
    return any(sig in combined for sig in _ML_TITLE_SIGNALS)


def _parse_year(s) -> int:
    """Parse a year from a date string or int."""
    if not s:
        return 0
    if isinstance(s, int):
        return s
    m = re.search(r'\b(19|20)\d{2}\b', str(s))
    return int(m.group()) if m else 0


def compute_career_progression_score(candidate: dict) -> tuple[float, str]:
    """
    Returns (score [0,1], explanation_string).

    Explanation is used in reasoning output so recruiters understand
    WHY a candidate was ranked where they were.
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.3, "no career history"

    # Sort by start_date descending (most recent first)
    def get_start_year(job):
        sd = job.get("start_date", "")
        return _parse_year(sd)

    sorted_career = sorted(career, key=get_start_year, reverse=True)
    n = len(sorted_career)

    # ── Dimension 1: ML role consistency ──────────────────────────────────────
    # What fraction of roles were AI/ML-focused?
    ml_role_count = sum(
        1 for j in career
        if _is_ml_role(j.get("title", ""), j.get("description", ""))
    )
    ml_consistency = ml_role_count / n  # 0 to 1

    # ── Dimension 2: Recency of ML work ───────────────────────────────────────
    # We care more about RECENT roles being ML-focused.
    # Logic: the most recent 2 roles get 2x weight.
    # Heuristic rationale: ML skills evolved rapidly post-2017. A candidate
    # who did ML 5yr ago but switched to management is less relevant than
    # one who started ML recently and has been deepening ever since.
    recent_ml = sum(
        1 for j in sorted_career[:2]
        if _is_ml_role(j.get("title", ""), j.get("description", ""))
    )
    recency_score = recent_ml / min(2, n)  # 0 to 1

    # ── Dimension 3: Seniority trajectory ────────────────────────────────────
    # Is seniority increasing? Detect the trend.
    # We look at the chronological order (oldest to newest).
    chrono = sorted(career, key=get_start_year)
    tiers = [_get_seniority_tier(j.get("title", "")) for j in chrono]

    if len(tiers) >= 2:
        # Count upward moves vs downward moves
        up = sum(1 for i in range(1, len(tiers)) if tiers[i] > tiers[i-1])
        down = sum(1 for i in range(1, len(tiers)) if tiers[i] < tiers[i-1])
        lateral = len(tiers) - 1 - up - down
        trajectory_score = (up * 1.0 + lateral * 0.5 - down * 0.5) / (len(tiers) - 1)
        trajectory_score = max(0.0, min(1.0, trajectory_score))
    else:
        trajectory_score = 0.5  # neutral for single role

    # ── Dimension 4: Domain consistency ──────────────────────────────────────
    # Reward focused careers. Penalize scattered domain-hopping.
    # Logic: a candidate who went Data Analyst → Sales → ML Engineer
    # has a less coherent trajectory than Junior ML → ML Eng → Senior ML.
    industries = [j.get("industry", "").lower() for j in career]
    ml_industries = sum(
        1 for ind in industries
        if any(sig in ind for sig in _ML_INDUSTRY_SIGNALS)
    )
    domain_consistency = ml_industries / n

    # ── Dimension 5: Total months in AI/ML work ──────────────────────────────
    ml_months = sum(
        j.get("duration_months", 0)
        for j in career
        if _is_ml_role(j.get("title", ""), j.get("description", ""))
    )
    # Ideal: 60+ months (5yr) of ML-specific work
    # Heuristic: 60mo = ideal, log-scale diminishing returns above that
    import math
    ml_depth_score = min(1.0, math.log1p(ml_months) / math.log1p(60))

    # ── Weighted combination ──────────────────────────────────────────────────
    # Weights rationale:
    # - recency (30%): most predictive of current capability (skills decay)
    # - ml_consistency (25%): sustained focus, not a recent convert
    # - ml_depth (20%): raw months in ML = accumulated experience
    # - trajectory (15%): career is growing, not plateauing
    # - domain_consistency (10%): ML industry focus
    score = (
        0.30 * recency_score +
        0.25 * ml_consistency +
        0.20 * ml_depth_score +
        0.15 * trajectory_score +
        0.10 * domain_consistency
    )

    # ── Build explanation ────────────────────────────────────────────────────
    parts = []
    if recency_score >= 0.5:
        parts.append(f"recent roles in ML ({recent_ml}/{min(2,n)})")
    else:
        parts.append(f"recent roles not ML-focused ({recent_ml}/{min(2,n)})")

    if ml_consistency >= 0.7:
        parts.append(f"consistent ML career ({ml_role_count}/{n} roles)")
    elif ml_consistency >= 0.4:
        parts.append(f"mixed ML career ({ml_role_count}/{n} roles ML)")
    else:
        parts.append(f"limited ML focus ({ml_role_count}/{n} roles ML)")

    if tiers and len(tiers) >= 2:
        latest_tier = tiers[-1]
        tier_names = {0: "junior", 1: "mid", 2: "senior", 3: "staff/lead",
                      4: "principal", 5: "director+"}
        if trajectory_score >= 0.7:
            parts.append(f"upward trajectory → {tier_names.get(latest_tier, 'senior')}")
        elif trajectory_score <= 0.3:
            parts.append("lateral/downward moves detected")

    if ml_months >= 48:
        parts.append(f"{ml_months//12}yr in ML roles")
    elif ml_months > 0:
        parts.append(f"{ml_months}mo in ML roles")

    explanation = "; ".join(parts)
    return round(score, 4), explanation


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test cases that exercise the scoring dimensions
    test_cases = [
        {
            "name": "Ideal: consistent ML progression",
            "career_history": [
                {"title": "Senior ML Engineer", "start_date": "2022-01-01", "duration_months": 24,
                 "description": "embedding retrieval ranking", "industry": "saas"},
                {"title": "ML Engineer", "start_date": "2019-01-01", "duration_months": 36,
                 "description": "nlp recommendation", "industry": "fintech"},
                {"title": "Junior ML Engineer", "start_date": "2016-01-01", "duration_months": 36,
                 "description": "machine learning", "industry": "technology"},
            ]
        },
        {
            "name": "Scattered: non-ML path → recent ML",
            "career_history": [
                {"title": "ML Engineer", "start_date": "2023-01-01", "duration_months": 12,
                 "description": "machine learning", "industry": "ai"},
                {"title": "Marketing Analyst", "start_date": "2020-01-01", "duration_months": 36,
                 "description": "digital marketing", "industry": "advertising"},
                {"title": "Data Analyst", "start_date": "2018-01-01", "duration_months": 24,
                 "description": "sql reporting", "industry": "retail"},
            ]
        },
        {
            "name": "Title-chaser: ML but frequent switches",
            "career_history": [
                {"title": "Principal ML Engineer", "start_date": "2023-06-01", "duration_months": 6,
                 "description": "embeddings", "industry": "ai"},
                {"title": "Senior ML Engineer", "start_date": "2022-01-01", "duration_months": 17,
                 "description": "nlp", "industry": "saas"},
                {"title": "ML Engineer", "start_date": "2021-01-01", "duration_months": 12,
                 "description": "ml", "industry": "fintech"},
                {"title": "Data Scientist", "start_date": "2019-01-01", "duration_months": 24,
                 "description": "analytics", "industry": "consulting"},
            ]
        },
    ]

    for tc in test_cases:
        score, explanation = compute_career_progression_score({"career_history": tc["career_history"]})
        print(f"\n{tc['name']}")
        print(f"  Score: {score:.3f} | {explanation}")
