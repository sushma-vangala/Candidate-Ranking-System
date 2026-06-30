"""
trap_detector.py
Detects three classes of trap candidates described in the challenge docs:
  1. Honeypots   — subtly impossible profiles (timeline contradictions, impossible credentials)
  2. Keyword stuffers — many AI keywords but no real depth in career history
  3. Behavioral ghosts — great on paper but disengaged/unavailable
Returns a penalty multiplier in [0.0, 1.0] where 1.0 = no penalty.
"""

from datetime import date, datetime
import re


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def detect_honeypot(candidate: dict) -> tuple[bool, str]:
    """
    Returns (is_honeypot, reason).
    Checks for impossible timelines and credential contradictions.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])

    yoe = profile.get("years_of_experience", 0)
    today = date.today()

    # ── Check 1: YoE vs education graduation year ──────────────────────────
    grad_years = [e.get("end_year") for e in education if e.get("end_year")]
    if grad_years:
        earliest_grad = min(grad_years)
        max_possible_yoe = today.year - earliest_grad
        if yoe > max_possible_yoe + 2:  # allow 2yr buffer for part-time/internships
            return True, f"YoE {yoe} impossible: graduated {earliest_grad} ({max_possible_yoe} max)"

    # ── Check 2: Company tenure vs company founding (impossible overlap) ────
    # Heuristic: check if stated duration_months >> realistic for start_date
    for job in career:
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date")) or today
        stated_duration = job.get("duration_months", 0)
        if start:
            actual_months = (end.year - start.year) * 12 + (end.month - start.month)
            if stated_duration > actual_months + 3:  # 3-month grace
                return True, f"Duration {stated_duration}mo > actual {actual_months}mo at {job.get('company')}"

    # ── Check 3: Overlapping jobs (same period, two full-time roles) ────────
    active_jobs = [(parse_date(j.get("start_date")), parse_date(j.get("end_date")) or today)
                   for j in career if not j.get("is_current") and parse_date(j.get("start_date"))]
    for i in range(len(active_jobs)):
        for k in range(i + 1, len(active_jobs)):
            s1, e1 = active_jobs[i]
            s2, e2 = active_jobs[k]
            if s1 and s2:
                overlap_start = max(s1, s2)
                overlap_end = min(e1, e2)
                if overlap_start < overlap_end:
                    overlap_months = (overlap_end.year - overlap_start.year) * 12 + \
                                     (overlap_end.month - overlap_start.month)
                    if overlap_months > 6:  # >6 month overlap = suspicious
                        return True, f"Overlapping jobs for {overlap_months} months"

    # ── Check 4: Expert skill with 0 months duration ────────────────────────
    expert_zero = [s for s in skills
                   if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0]
    if len(expert_zero) >= 3:
        return True, f"{len(expert_zero)} 'expert' skills with 0 months used"

    # ── Check 5: Total career months >> stated YoE by large margin ──────────
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    stated_months = yoe * 12
    if total_career_months > stated_months * 1.5 and total_career_months > 24:
        return True, f"Career months {total_career_months} >> stated YoE months {stated_months:.0f}"

    return False, ""


def keyword_stuffer_score(candidate: dict) -> float:
    """
    Returns a penalty [0.0 = stuffed, 1.0 = clean].
    Keyword stuffers have many AI skills but thin career descriptions.
    """
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    ai_keywords = {
        "embeddings", "vector search", "llm", "rag", "fine-tuning", "lora",
        "pinecone", "faiss", "langchain", "openai", "gpt", "bert", "transformer",
        "nlp", "sentence-transformers", "weaviate", "qdrant", "semantic search",
    }

    # How many AI skills listed?
    skill_names_lower = {s.get("name", "").lower() for s in skills}
    ai_skill_count = sum(1 for kw in ai_keywords if kw in " ".join(skill_names_lower))

    # How much real AI content in career descriptions?
    career_text = " ".join(j.get("description", "") for j in career).lower()
    career_ai_mentions = sum(1 for kw in ai_keywords if kw in career_text)

    # Red flag: lots of AI skills listed but career history doesn't mention them
    if ai_skill_count >= 6 and career_ai_mentions <= 1:
        return 0.3  # heavy penalty

    # Check skill endorsement quality (stuffers have 0 endorsements on "advanced" skills)
    advanced_zero_endorsements = [
        s for s in skills
        if s.get("proficiency") in ("advanced", "expert") and s.get("endorsements", 0) == 0
    ]
    if len(advanced_zero_endorsements) > 5:
        return 0.6  # moderate penalty

    # Title vs skills mismatch (marketing manager claiming 10 ML skills)
    title = profile.get("current_title", "").lower()
    non_technical_roles = ["marketing", "sales", "hr", "business analyst", "project manager",
                           "product manager", "recruiter", "account manager"]
    is_non_technical = any(r in title for r in non_technical_roles)
    if is_non_technical and ai_skill_count >= 4:
        return 0.2  # severe penalty

    return 1.0  # clean


def behavioral_availability_score(candidate: dict) -> float:
    """
    Returns availability multiplier [0.0, 1.0].
    A perfect candidate who is unreachable/disengaged gets penalized.
    """
    signals = candidate.get("redrob_signals", {})
    today = date.today()

    score = 1.0

    # ── Recency of last login ──────────────────────────────────────────────
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days_inactive = (today - last_active).days
        if days_inactive > 180:
            score *= 0.4
        elif days_inactive > 90:
            score *= 0.6
        elif days_inactive > 45:
            score *= 0.8
        elif days_inactive <= 14:
            score *= 1.05  # bonus for very recent

    # ── Open to work ──────────────────────────────────────────────────────
    if not signals.get("open_to_work_flag", True):
        score *= 0.75

    # ── Recruiter response rate ────────────────────────────────────────────
    rrr = signals.get("recruiter_response_rate", 0.5)
    if rrr < 0.2:
        score *= 0.6
    elif rrr < 0.4:
        score *= 0.8
    elif rrr > 0.7:
        score *= 1.05

    # ── Interview completion ───────────────────────────────────────────────
    icr = signals.get("interview_completion_rate", 0.5)
    if icr < 0.3:
        score *= 0.7
    elif icr > 0.8:
        score *= 1.05

    return min(score, 1.0)


def get_trap_multiplier(candidate: dict) -> tuple[float, str]:
    """
    Master function. Returns (multiplier, reason_string).
    multiplier 0.0 = discard, 1.0 = no penalty.
    """
    # Step 1: Honeypot check — hard filter
    is_honeypot, honeypot_reason = detect_honeypot(candidate)
    if is_honeypot:
        return 0.0, f"HONEYPOT: {honeypot_reason}"

    # Step 2: Keyword stuffer
    stuffer_mult = keyword_stuffer_score(candidate)

    # Step 3: Behavioral availability
    avail_mult = behavioral_availability_score(candidate)

    total = stuffer_mult * avail_mult
    reasons = []
    if stuffer_mult < 0.9:
        reasons.append(f"keyword_stuffer={stuffer_mult:.2f}")
    if avail_mult < 0.9:
        reasons.append(f"availability={avail_mult:.2f}")

    return total, "; ".join(reasons) if reasons else ""
