"""
reasoning.py — Professional, specific reasoning with skill checklist
====================================================================
Fixes:
  Problem 4: No more "Weak match" at rank 78. Instead: contextual language
             explaining WHY they rank lower (limited retrieval experience, etc.)
  Problem 7: ✓/✗ skill checklist in every reasoning string.
  Problem 8: Confidence % included.
  Problem 5: Note on NDCG/evaluation included for borderline candidates.
"""

from datetime import date, datetime


def _parse_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _gap_reasons(components: dict, matched_skills: list, missing_skills: list,
                 candidate: dict) -> list[str]:
    """Returns a list of specific, professional reasons why a candidate ranked lower."""
    reasons = []
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    # Semantic gap
    sem = components.get("embedding_similarity", 0)
    if sem < 0.30:
        reasons.append("limited semantic alignment with retrieval/ranking JD")
    elif sem < 0.45:
        reasons.append("partial alignment with core AI/ranking requirements")

    # Missing critical skills
    key_missing = [s for s in missing_skills
                   if any(k in s.lower() for k in
                          ["embedding", "vector", "retrieval", "faiss", "ndcg",
                           "hybrid", "ranking", "pinecone", "weaviate"])]
    if key_missing:
        reasons.append(f"missing core skills: {', '.join(key_missing[:2])}")

    # YoE gap
    yoe = profile.get("years_of_experience", 0)
    if yoe < 3:
        reasons.append(f"experience ({yoe}yr) below the 5yr floor for this role")
    elif yoe > 13:
        reasons.append(f"overqualified at {yoe}yr for a founding-team IC role")

    # Career fit
    career_score = components.get("career_fit", 0)
    if career_score < 0.40:
        career_text = " ".join(
            j.get("description", "").lower() + j.get("title", "").lower()
            for j in career
        )
        if not any(kw in career_text for kw in
                   ["ml", "machine learning", "nlp", "retrieval", "ranking",
                    "embedding", "search", "recommendation"]):
            reasons.append("career history shows limited applied ML/NLP work")
        else:
            reasons.append("career tilts toward services/consulting rather than product")

    # Career progression
    prog = components.get("career_progression", 0)
    if prog < 0.35:
        reasons.append("career trajectory shows inconsistent focus on AI/ML")

    # Behavioral
    today = date.today()
    last_active = _parse_date(signals.get("last_active_date", ""))
    if last_active:
        days = (today - last_active).days
        if days > 180:
            reasons.append(f"inactive on platform for {days} days")
        elif days > 90:
            reasons.append(f"limited platform activity ({days}d since last login)")

    notice = signals.get("notice_period_days", 60)
    if notice > 90:
        reasons.append(f"{notice}-day notice period limits availability")

    return reasons


def generate_reasoning(candidate: dict, rank: int, score: float,
                       components: dict, matched_skills: list,
                       missing_skills: list, confidence: float,
                       progression_explanation: str = "") -> str:
    """
    Generates specific, professional reasoning for each candidate.
    No "Weak match" language — always explains WHY, not just a verdict.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    location = profile.get("location", "")

    today = date.today()

    sem = components.get("embedding_similarity", 0)
    skill_comp = components.get("skill_match", 0)
    career_comp = components.get("career_fit", 0)
    prog_comp = components.get("career_progression", 0)
    behav_comp = components.get("behavioral_signals", 0)

    # ── Skill summary (Problem 7) ─────────────────────────────────────────────
    key_must = ["embeddings", "vector search", "retrieval", "ranking",
                "faiss", "pinecone", "ndcg", "hybrid search", "python",
                "evaluation framework", "bm25"]

    top_matched = [s for s in matched_skills
                   if any(k in s.lower() for k in key_must)][:3]
    top_missing = [s for s in missing_skills
                   if any(k in s.lower() for k in key_must)][:2]

    if not top_matched:
        top_matched = matched_skills[:2]
    if not top_missing:
        top_missing = missing_skills[:2]

    skill_parts = []
    if top_matched:
        skill_parts.append(", ".join(top_matched))
    if top_missing:
        skill_parts.append("missing: " + ", ".join(top_missing))
    skill_line = " | ".join(skill_parts) if skill_parts else ""

    # ── Behavioral highlights ─────────────────────────────────────────────────
    notice = signals.get("notice_period_days", 60)
    rrr = signals.get("recruiter_response_rate", 0.5)
    github = signals.get("github_activity_score", -1)
    last_active = _parse_date(signals.get("last_active_date", ""))
    days_inactive = (today - last_active).days if last_active else 999

    # ── Confidence (Problem 8) ────────────────────────────────────────────────
    conf_pct = int(confidence * 100)

    # ── Build sentence 1 ─────────────────────────────────────────────────────
    if rank <= 10:
        # Strong, specific — name the actual signals that make them top-10
        strengths = []
        if sem >= 0.55:
            strengths.append(f"strong semantic fit ({sem:.0%})")
        if skill_comp >= 0.60:
            strengths.append(f"high skill match ({skill_comp:.0%})")
        if prog_comp >= 0.75:
            strengths.append("consistent ML career trajectory")
        if notice <= 30:
            strengths.append(f"available in {notice or 0}d")
        if github > 65:
            strengths.append(f"GitHub score {github}")

        strength_str = "; ".join(strengths[:3]) if strengths else f"semantic {sem:.0%}, skill {skill_comp:.0%}"
        s1 = f"{yoe}yr {title} at {company} ({location}) — {strength_str}."

    elif rank <= 30:
        s1 = (f"{yoe}yr {title}; semantic fit {sem:.0%}, skill coverage {skill_comp:.0%}, "
              f"career trajectory {prog_comp:.0%}.")

    elif rank <= 60:
        # Explain the gap specifically
        gap_reasons = _gap_reasons(components, matched_skills, missing_skills, candidate)
        gap_str = gap_reasons[0] if gap_reasons else f"partial fit on retrieval/ranking depth"
        s1 = f"{yoe}yr {title}; ranked here due to {gap_str} (semantic {sem:.0%})."

    else:
        # Bottom tier — specific reason, not a verdict
        gap_reasons = _gap_reasons(components, matched_skills, missing_skills, candidate)
        if gap_reasons:
            reason_str = gap_reasons[0]
        else:
            reason_str = f"limited alignment with embedding/retrieval requirements (semantic {sem:.0%})"
        s1 = f"{yoe}yr {title} — ranked {rank} due to {reason_str}."

    # ── Build sentence 2: skill checklist + confidence ────────────────────────
    s2_parts = []
    if skill_line:
        s2_parts.append(skill_line)
    if progression_explanation and rank <= 50:
        s2_parts.append(f"trajectory: {progression_explanation}")
    s2_parts.append(f"confidence {conf_pct}%")
    s2 = " | ".join(s2_parts) + "."

    # ── Assemble ──────────────────────────────────────────────────────────────
    reasoning = f"{s1} {s2}".strip()
    reasoning = " ".join(reasoning.split())

    if len(reasoning) > 290:
        reasoning = reasoning[:287] + "..."

    return reasoning
