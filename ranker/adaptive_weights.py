"""
adaptive_weights.py — Adaptive scoring weights derived from JD emphasis
=======================================================================
Original feature: instead of fixed weights, we READ the JD text and
adjust component weights based on what the JD actually emphasizes.

How it works:
  We scan the JD for "emphasis signals" — phrases that signal the hiring
  manager cares unusually about a specific dimension. Then we shift weight
  toward that dimension and away from less-emphasized ones.

Examples of adaptation:
  JD says "we need someone available immediately, sub-30-day notice" →
    → behavioral_signals weight increases (availability is critical)

  JD says "we need 3+ years commitment, no title-chasers" →
    → career_progression weight increases (trajectory matters more)

  JD says "production experience with embeddings at scale" (3x repetition) →
    → embedding_similarity weight increases (semantic fit is paramount)

  JD says "strong Python, code quality" (explicitly called out) →
    → skill_match weight increases slightly (hard skills matter here)

Why this is meaningful:
  Two different JDs for "ML Engineer" might weight things very differently.
  A startup JD emphasizing "ship in week 1" cares more about availability.
  A research-adjacent JD emphasizing "evaluation frameworks" cares more
  about semantic depth. Fixed weights ignore this entirely.

This is entirely our own logic — no existing ranking library does this.
"""

import re
from copy import deepcopy


# ── Base weights (fallback if no signals found) ────────────────────────────────
BASE_WEIGHTS = {
    "embedding_similarity": 0.40,
    "skill_match":          0.20,
    "career_fit":           0.20,
    "career_progression":   0.10,
    "behavioral_signals":   0.15,
    "education_bonus":      0.05,
}

# ── Emphasis signal definitions ────────────────────────────────────────────────
# Each signal is: (regex_pattern, component_to_boost, boost_amount, components_to_reduce)
# boost_amount is added to the target component and subtracted proportionally from others.
# We cap any single component at 0.55 and floor at 0.03 to keep the system stable.

EMPHASIS_SIGNALS = [
    # ── Availability / urgency signals → boost behavioral ──────────────────
    {
        "name": "urgency",
        "patterns": [
            r'sub[- ]?30[- ]day\s+notice',
            r'immediate(?:ly)?\s+(?:available|join)',
            r'notice\s+period.*(?:critical|important|key)',
            r'available\s+to\s+start\s+(?:immediately|within)',
            r'we\s+(?:can\s+)?buy\s+out\s+(?:up\s+to\s+)?\d+\s+day',
        ],
        "boost_component": "behavioral_signals",
        "boost": 0.06,
        "reduce_from": ["education_bonus", "skill_match"],
        "rationale": "JD emphasizes availability → behavioral signals weighted higher",
    },

    # ── Long-term commitment signals → boost career progression ───────────
    {
        "name": "commitment",
        "patterns": [
            r'(?:3|three)[+\s]*year[s]?\s+commit',
            r'title[- ]chaser',
            r'switching\s+(?:companies|jobs)\s+every',
            r'plan\s+to\s+be\s+here\s+for',
            r'long[- ]term\s+(?:ownership|commitment|fit)',
            r'we\s+need\s+someone\s+who\s+(?:will\s+)?stay',
        ],
        "boost_component": "career_progression",
        "boost": 0.06,
        "reduce_from": ["education_bonus", "behavioral_signals"],
        "rationale": "JD emphasizes long-term commitment → career trajectory weighted higher",
    },

    # ── Semantic/embedding depth signals → boost embedding similarity ──────
    {
        "name": "semantic_depth",
        "patterns": [
            r'(?:embeddings?|retrieval|ranking).{0,50}(?:embeddings?|retrieval|ranking)',  # repeated
            r'go\s+beyond\s+keyword',
            r'understand(?:ing)?\s+(?:who|what)\s+(?:fits?|matches?)',
            r'semantic\s+understanding',
            r'genuinely\s+(?:fit|match|understand)',
        ],
        "boost_component": "embedding_similarity",
        "boost": 0.05,
        "reduce_from": ["education_bonus", "behavioral_signals"],
        "rationale": "JD emphasizes semantic understanding → embedding similarity weighted higher",
    },

    # ── Hard skills / technical depth signals → boost skill match ─────────
    {
        "name": "hard_skills",
        "patterns": [
            r'strong\s+python',
            r'code\s+quality\s+(?:matters?|is\s+important)',
            r'(?:hands[- ]on|production)\s+experience\s+with\s+(?:specific|named)',
            r'you\s+absolutely\s+need',
            r'hard\s+requirements?',
            r'non[- ]negotiable',
        ],
        "boost_component": "skill_match",
        "boost": 0.04,
        "reduce_from": ["education_bonus", "career_progression"],
        "rationale": "JD emphasizes hard technical skills → skill match weighted higher",
    },

    # ── Career background signals → boost career fit ──────────────────────
    {
        "name": "career_background",
        "patterns": [
            r'product\s+compan(?:y|ies)',
            r'not\s+(?:pure\s+)?(?:research|consulting)',
            r'production\s+(?:deployment|experience)',
            r'shipped\s+to\s+real\s+users',
            r'consulting\s+(?:firm|background|only)',
            r'services?\s+(?:company|background)',
        ],
        "boost_component": "career_fit",
        "boost": 0.05,
        "reduce_from": ["education_bonus", "career_progression"],
        "rationale": "JD emphasizes career background type → career fit weighted higher",
    },

    # ── Education/pedigree signals → boost education ──────────────────────
    {
        "name": "education",
        "patterns": [
            r'(?:tier[- ]?1|iit|nit|bits)\s+(?:institute|college|grad)',
            r'(?:ms|phd|master|doctorate)\s+(?:required|preferred|strongly)',
            r'academic\s+(?:background|credentials?)\s+matter',
        ],
        "boost_component": "education_bonus",
        "boost": 0.05,
        "reduce_from": ["behavioral_signals", "career_progression"],
        "rationale": "JD emphasizes education credentials → education bonus weighted higher",
    },
]


def compute_adaptive_weights(jd_text: str, verbose: bool = False) -> dict:
    """
    Reads JD text, detects emphasis signals, and returns adapted weights.

    Args:
        jd_text: raw job description text
        verbose: if True, prints which signals fired and why

    Returns:
        dict of component → weight, summing to ~1.0
    """
    weights = deepcopy(BASE_WEIGHTS)
    fired_signals = []

    jd_lower = jd_text.lower()

    for signal in EMPHASIS_SIGNALS:
        hit_count = 0
        for pattern in signal["patterns"]:
            matches = re.findall(pattern, jd_lower, re.I)
            hit_count += len(matches)

        if hit_count > 0:
            fired_signals.append({
                "name": signal["name"],
                "hits": hit_count,
                "rationale": signal["rationale"],
            })

            # Apply boost
            boost = signal["boost"] * min(hit_count, 2)  # cap at 2x boost per signal
            boost_target = signal["boost_component"]
            reduce_from = signal["reduce_from"]

            # Add to target
            weights[boost_target] = min(0.55, weights[boost_target] + boost)

            # Distribute reduction proportionally across specified components
            total_reduce = boost
            per_component = total_reduce / len(reduce_from)
            for comp in reduce_from:
                weights[comp] = max(0.03, weights[comp] - per_component)

    # Renormalize to sum to 1.0 (floating point drift correction)
    total = sum(weights.values())
    weights = {k: round(v / total, 4) for k, v in weights.items()}

    if verbose:
        print("\n=== Adaptive Weight Computation ===")
        if fired_signals:
            for sig in fired_signals:
                print(f"  ✓ Signal '{sig['name']}' fired ({sig['hits']} match(es))")
                print(f"    → {sig['rationale']}")
        else:
            print("  No emphasis signals detected — using base weights")
        print(f"\n  Final weights: {weights}")
        total_check = sum(weights.values())
        print(f"  Sum: {total_check:.4f} (should be ~1.0)")

    return weights, fired_signals


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: The actual Redrob JD
    from ranker.jd_parser import _BUNDLED_JD
    print("Test 1: Redrob Senior AI Engineer JD")
    weights, signals = compute_adaptive_weights(_BUNDLED_JD, verbose=True)
    print(f"\nBase:     {BASE_WEIGHTS}")
    print(f"Adapted:  {weights}")
    delta = {k: round(weights[k] - BASE_WEIGHTS[k], 4) for k in weights}
    print(f"Delta:    {delta}")

    # Test 2: A hypothetical urgency-heavy JD
    print("\n\nTest 2: Hypothetical urgency-heavy JD")
    urgency_jd = """
    We need someone available immediately. Sub-30-day notice is critical.
    We can buy out up to 30 days notice. Please only apply if you are
    immediately available or can join within 2 weeks.
    Strong Python required. Code quality matters. Non-negotiable.
    """
    weights2, signals2 = compute_adaptive_weights(urgency_jd, verbose=True)
    print(f"Base:     {BASE_WEIGHTS}")
    print(f"Adapted:  {weights2}")
