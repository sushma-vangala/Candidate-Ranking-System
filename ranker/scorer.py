"""
scorer.py — Hybrid scorer with MiniLM embedding similarity
===========================================================
Fixes addressed:
  Problem 1: MiniLM embeddings (precomputed) → cosine similarity (40% weight)
  Problem 3: Skills derived from JD profile object, not hardcoded
  Problem 4: No employer name penalties — only work content signals  
  Problem 5: Embedding similarity at 40%
  Problem 6: Skill synonym relationships (sentence-transformers ↔ SBERT ↔ bi-encoder)
"""

import math
import numpy as np
from datetime import date, datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from ranker.jd_parser import JD_PROFILE
from ranker.career_progression import compute_career_progression_score


# ── Embedding similarity (Problem 1 + 5 fix) ──────────────────────────────────

def compute_embedding_similarity(candidate_idx: int, emb_matrix: np.ndarray,
                                  jd_emb: np.ndarray) -> float:
    """
    Dot product of L2-normalized vectors = cosine similarity.
    Both embeddings are pre-normalized, so this is O(1) per candidate.
    """
    if emb_matrix is None or jd_emb is None:
        return 0.0
    # Dot product of normalized vectors = cosine similarity
    sim = float(np.dot(jd_emb[0], emb_matrix[candidate_idx]))
    # Cosine sim in [-1, 1] → map to [0, 1]
    return (sim + 1.0) / 2.0


# ── Skill match with synonym awareness (Problem 6 fix) ────────────────────────

def compute_skill_match_score(candidate: dict) -> tuple[float, list, list]:
    """
    Returns (score, matched_skills, missing_skills).
    Checks direct matches AND synonym relationships.
    Fix for Problem 6: "sentence-transformers" matches "SBERT", "bi-encoder", etc.
    Fix for Problem 7: returns matched/missing lists for rich reasoning.
    """
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    career_text = " ".join(
        j.get("description", "").lower() + " " + j.get("title", "").lower()
        for j in career
    )
    profile_text = (
        candidate.get("profile", {}).get("summary", "").lower() + " " +
        candidate.get("profile", {}).get("headline", "").lower()
    )
    all_text = career_text + " " + profile_text

    # Build skill presence map with depth weights
    skill_map = {}
    for s in skills:
        name = s.get("name", "").lower().strip()
        prof = s.get("proficiency", "beginner")
        end = s.get("endorsements", 0)
        dur = s.get("duration_months", 0)

        prof_w = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.85, "expert": 1.0}.get(prof, 0.25)
        end_bonus = min(0.15, math.log1p(end) * 0.025)
        dur_bonus = min(0.10, math.log1p(dur) * 0.012)
        skill_map[name] = min(1.0, prof_w + end_bonus + dur_bonus)

    synonyms = JD_PROFILE["skill_synonyms"]
    must_haves = JD_PROFILE["must_have_skills"]
    nice_haves = JD_PROFILE["nice_to_have_skills"]

    matched_skills = []
    missing_skills = []

    def skill_score_with_match(target: str) -> tuple[float, str]:
        """Returns (score, match_type_string)."""
        tl = target.lower().strip()

        # 1. Direct match in skill list
        if tl in skill_map:
            return skill_map[tl], f"✓ {target}"

        # 2. Partial direct match (e.g., "elasticsearch" matches "elasticsearch 8")
        for sname, sc in skill_map.items():
            if tl in sname or sname in tl:
                return sc * 0.9, f"✓ {target} (via {sname})"

        # 3. Synonym match (Problem 6 fix)
        for sname, sc in skill_map.items():
            syns = synonyms.get(tl, [])
            if any(syn in sname or sname in syn for syn in syns):
                return sc * 0.80, f"≈ {target} (via synonym {sname})"

        # 4. Mentioned in career/profile text
        if tl in all_text:
            return 0.40, f"≈ {target} (in description)"

        # 5. Synonym mentioned in career text
        syns = synonyms.get(tl, [])
        for syn in syns:
            if syn in all_text:
                return 0.30, f"≈ {target} (synonym '{syn}' in description)"

        return 0.0, f"✗ {target}"

    # Score must-haves
    must_scores = []
    for sk in must_haves:
        sc, match_str = skill_score_with_match(sk)
        must_scores.append(sc)
        if sc >= 0.3:
            matched_skills.append(match_str)
        else:
            missing_skills.append(match_str)

    must_coverage = sum(s > 0.3 for s in must_scores) / len(must_haves)
    must_avg = sum(must_scores) / len(must_haves)
    must_score = 0.65 * must_coverage + 0.35 * must_avg

    # Score nice-to-haves (bonus only)
    nice_scores = []
    for sk in nice_haves:
        sc, _ = skill_score_with_match(sk)
        nice_scores.append(sc)
    nice_bonus = min(0.18, sum(s > 0.3 for s in nice_scores) / max(1, len(nice_haves)) * 0.18)

    final_score = min(1.0, must_score + nice_bonus)
    return final_score, matched_skills, missing_skills


# ── Career fit (Problem 4 fix: no employer name penalties) ────────────────────

def compute_career_fit_score(candidate: dict) -> float:
    """
    Scores based on WORK CONTENT, not employer names.
    A TCS engineer doing real ML ranking work is NOT penalized.
    A startup engineer doing pure CRUD work IS penalized.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    score = 0.45  # baseline

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "").lower()
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()

    # ── YoE band ─────────────────────────────────────────────────────────────
    ideal_min = JD_PROFILE["ideal_yoe_min"]
    ideal_max = JD_PROFILE["ideal_yoe_max"]
    soft_min = JD_PROFILE["yoe_soft_min"]
    hard_min = JD_PROFILE["yoe_hard_min"]
    soft_max = JD_PROFILE["yoe_soft_max"]

    if ideal_min <= yoe <= ideal_max:
        score += 0.15
    elif soft_min <= yoe < ideal_min:
        score += 0.06
    elif ideal_max < yoe <= soft_max:
        score += 0.04  # slight over-qual concern but not a dealbreaker
    elif yoe < hard_min:
        score -= 0.12
    else:
        score -= 0.02  # very over-qualified

    # ── Title quality signals ────────────────────────────────────────────────
    strong_titles = ["ml engineer", "ai engineer", "machine learning engineer",
                     "nlp engineer", "search engineer", "ranking engineer",
                     "retrieval engineer", "recommendation", "applied scientist",
                     "research scientist", "applied research"]
    good_titles = ["data scientist", "senior engineer", "staff engineer",
                   "principal engineer", "software engineer ml", "backend ml"]
    ok_titles = ["software engineer", "backend engineer", "data engineer",
                 "research engineer", "platform engineer"]

    if any(t in title for t in strong_titles):
        score += 0.15
    elif any(t in title for t in good_titles):
        score += 0.08
    elif any(t in title for t in ok_titles):
        score += 0.03

    # Red flag: explicitly excluded roles from JD
    for rf in JD_PROFILE["red_flag_role_types"]:
        if rf in title:
            score -= 0.25
            break

    # ── Work content in career (THE fix for Problem 4) ───────────────────────
    # Score based on WHAT they did, not WHERE they worked
    career_text = " ".join(
        j.get("description", "").lower() + " " + j.get("title", "").lower()
        for j in career
    )

    # ML/AI content in actual work
    ml_work_signals = [
        "embedding", "retrieval", "ranking", "recommendation", "search",
        "nlp", "natural language", "machine learning", "deep learning",
        "model training", "model deployment", "feature engineering",
        "vector", "similarity", "transformer", "bert", "fine-tun",
    ]
    ml_hits = sum(1 for sig in ml_work_signals if sig in career_text)
    score += min(0.15, ml_hits * 0.015)

    # Pure CV without NLP: only penalize if NO NLP signals
    cv_signals = JD_PROFILE["cv_only_domains"]
    nlp_signals = ["nlp", "natural language", "text", "embedding", "retrieval",
                   "ranking", "recommendation", "search"]
    has_cv = any(sig in career_text for sig in cv_signals)
    has_nlp = any(sig in career_text for sig in nlp_signals)
    if has_cv and not has_nlp:
        score -= 0.08  # CV-only background

    # ── Production vs research signal ────────────────────────────────────────
    # JD explicitly says "not pure research"
    prod_signals = ["deployed", "production", "launched", "shipped", "users",
                    "serving", "api", "inference", "latency", "scale", "pipeline"]
    research_signals = ["arxiv", "paper", "published", "phd", "academic",
                        "laboratory", "lab", "research institute"]
    prod_hits = sum(1 for sig in prod_signals if sig in career_text)
    research_hits = sum(1 for sig in research_signals if sig in career_text)
    if prod_hits > research_hits:
        score += 0.06
    elif research_hits > prod_hits * 2:
        score -= 0.05  # pure research signal

    # ── Tenure signal (JD wants committed engineers, not title-chasers) ──────
    if career:
        durations = [j.get("duration_months", 0) for j in career if j.get("duration_months", 0) > 0]
        if durations:
            avg_tenure = sum(durations) / len(durations)
            if avg_tenure >= JD_PROFILE["ideal_avg_tenure_months"]:
                score += 0.08
            elif avg_tenure < JD_PROFILE["min_avg_tenure_months"]:
                score -= 0.06  # frequent job-hopper

    # ── Location ─────────────────────────────────────────────────────────────
    loc_combined = location + " " + country
    if any(loc in loc_combined for loc in JD_PROFILE["preferred_locations"]):
        score += 0.06
    elif "india" in loc_combined:
        score += 0.02

    # ── AI/ML career progression (multiple AI roles = real depth) ────────────
    ai_role_count = sum(
        1 for j in career
        if any(kw in j.get("title", "").lower()
               for kw in ["ml", "ai", "machine learning", "nlp", "data science",
                          "search", "ranking", "retrieval", "recommendation", "scientist"])
    )
    if ai_role_count >= 3:
        score += 0.10
    elif ai_role_count == 2:
        score += 0.06
    elif ai_role_count == 1:
        score += 0.03

    return max(0.0, min(1.0, score))


# ── Behavioral score ───────────────────────────────────────────────────────────

def compute_behavioral_score(candidate: dict) -> float:
    """Scores from the 23 redrob_signals. Unchanged from original."""
    signals = candidate.get("redrob_signals", {})
    today = date.today()
    score, wt = 0.0, 0.0

    def add(w, v):
        nonlocal score, wt
        score += w * max(0.0, min(1.0, float(v))); wt += w

    add(0.08, signals.get("profile_completeness_score", 50) / 100.0)

    try:
        la = datetime.strptime(signals.get("last_active_date", ""), "%Y-%m-%d").date()
        add(0.15, max(0.0, 1.0 - (today - la).days / 365.0))
    except Exception:
        add(0.15, 0.5)

    add(0.12, 1.0 if signals.get("open_to_work_flag") else 0.2)
    add(0.15, signals.get("recruiter_response_rate", 0.5))
    add(0.08, max(0.0, 1.0 - signals.get("avg_response_time_hours", 24) / 72.0))
    add(0.12, signals.get("interview_completion_rate", 0.5))

    g = signals.get("github_activity_score", -1)
    add(0.10, 0.3 if g == -1 else g / 100.0)

    n = signals.get("notice_period_days", 60)
    hard_max = JD_PROFILE["notice_period_hard_max_days"]
    if n > hard_max:
        notice_score = 0.15
    elif n > 60:
        notice_score = 0.45
    elif n > 30:
        notice_score = 0.65
    elif n <= 0:
        notice_score = 1.0
    else:
        notice_score = 0.88
    add(0.08, notice_score)

    add(0.05, 1.0 if (signals.get("verified_email") and signals.get("verified_phone")) else 0.5)

    asc = signals.get("skill_assessment_scores", {})
    add(0.07, (sum(asc.values()) / len(asc) / 100.0) if asc else 0.5)

    return score / wt if wt > 0 else 0.5


# ── Education bonus ────────────────────────────────────────────────────────────

def compute_education_bonus(candidate: dict) -> float:
    education = candidate.get("education", [])
    if not education:
        return 0.3
    tier_w = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.55, "tier_4": 0.35, "unknown": 0.40}
    rel_fields = ["computer science", "electrical", "mathematics", "statistics",
                  "data science", "artificial intelligence", "machine learning",
                  "information technology", "software", "engineering"]
    best = 0.0
    for edu in education:
        ts = tier_w.get(edu.get("tier", "unknown"), 0.4)
        field = edu.get("field_of_study", "").lower()
        es = ts * (1.1 if any(r in field for r in rel_fields) else 0.8)
        best = max(best, es)
    return min(1.0, best)


# ── Confidence score (Problem 8 fix) ─────────────────────────────────────────

def compute_confidence(score: float,
                       component_scores: dict,
                       trap_mult: float) -> float:

    if trap_mult < 0.5:
        return 0.10

    comp_values = [v for v in component_scores.values() if v is not None]

    if not comp_values:
        return 0.50

    mean = sum(comp_values) / len(comp_values)
    variance = sum((v - mean) ** 2 for v in comp_values) / len(comp_values)
    std_dev = variance ** 0.5

    # Better base confidence
    base = 0.50 + score * 0.50

    # Mild disagreement penalty
    agreement_factor = max(0.75, 1.0 - std_dev * 0.60)

    trap_factor = 0.80 if trap_mult < 1.0 else 1.0

    confidence = base * agreement_factor * trap_factor

    return round(min(0.98, confidence), 3)

# ── Main hybrid scorer ─────────────────────────────────────────────────────────

class HybridScorer:
    """
    Uses precomputed MiniLM embeddings for semantic similarity (40%),
    combined with skill match, career fit, behavioral, and education signals.

    If embeddings are not available (e.g., first run without precompute),
    falls back to TF-IDF (graceful degradation).
    """

    def __init__(self):
        self.emb_matrix = None
        self.jd_emb = None
        self.candidate_id_to_idx = {}
        self.tfidf_fallback = None
        self.jd_tfidf = None
        self.candidate_tfidf = None
        self.weights = JD_PROFILE["scoring_weights"]

    def load_embeddings(self, emb_path: str, ids_path: str, jd_path: str):
        """Load precomputed MiniLM embeddings."""
        import os
        if not all(os.path.exists(p) for p in [emb_path, ids_path, jd_path]):
            print("  ⚠️  Embedding files not found — falling back to TF-IDF")
            return False

        print(f"  Loading embeddings from {emb_path}...")
        self.emb_matrix = np.load(emb_path)   # (N, 384)
        self.jd_emb = np.load(jd_path)         # (1, 384)
        ids = np.load(ids_path, allow_pickle=True)
        self.candidate_id_to_idx = {cid: i for i, cid in enumerate(ids)}
        print(f"  Loaded {self.emb_matrix.shape[0]:,} embeddings "
              f"(dim={self.emb_matrix.shape[1]})")
        return True

    def build_tfidf_fallback(self, candidates: list[dict]):
        """TF-IDF fallback when embeddings not available."""
        from ranker.precompute_embeddings_compat import build_candidate_text_light
        print("  Building TF-IDF fallback index...")
        JD_TEXT = JD_PROFILE["jd_embedding_text"]
        texts = [JD_TEXT] + [build_candidate_text_light(c) for c in candidates]
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=8000,
                              sublinear_tf=True, min_df=3, stop_words="english")
        matrix = vec.fit_transform(texts)
        self.tfidf_fallback = vec
        self.jd_tfidf = matrix[0]
        self.candidate_tfidf = matrix[1:]
        print(f"  TF-IDF matrix: {self.candidate_tfidf.shape}")

    def get_embedding_score(self, candidate_id: str, fallback_idx: int) -> float:
        """Get semantic similarity score for one candidate."""
        if self.emb_matrix is not None and candidate_id in self.candidate_id_to_idx:
            idx = self.candidate_id_to_idx[candidate_id]
            # Dot product of normalized vectors = cosine similarity
            sim = float(np.dot(self.jd_emb[0], self.emb_matrix[idx]))
            return (sim + 1.0) / 2.0  # map [-1,1] to [0,1]
        elif self.candidate_tfidf is not None:
            sim = sk_cosine(self.jd_tfidf, self.candidate_tfidf[fallback_idx]).item()
            return float(sim)
        return 0.0

    def score_all(self, candidates: list[dict]) -> list[dict]:
        """
        Score all candidates. Returns list of result dicts with all components.
        """
        results = []
        w = self.weights

        # Pre-compute TF-IDF sims if using fallback (vectorized)
        if self.candidate_tfidf is not None:
            tfidf_sims = sk_cosine(self.jd_tfidf, self.candidate_tfidf).flatten()
        else:
            tfidf_sims = None

        for i, candidate in enumerate(candidates):
            cid = candidate["candidate_id"]

            # Semantic score
            if self.emb_matrix is not None and cid in self.candidate_id_to_idx:
                idx = self.candidate_id_to_idx[cid]
                sim = float(np.dot(self.jd_emb[0], self.emb_matrix[idx]))
                semantic = (sim + 1.0) / 2.0
            elif tfidf_sims is not None:
                semantic = float(tfidf_sims[i])
            else:
                semantic = 0.0

            skill, matched, missing = compute_skill_match_score(candidate)
            career = compute_career_fit_score(candidate)
            behavioral = compute_behavioral_score(candidate)
            edu = compute_education_bonus(candidate)

            prog, prog_explanation = compute_career_progression_score(candidate)

            components = {
                "embedding_similarity": semantic,
                "skill_match": skill,
                "career_fit": career,
                "career_progression": prog,
                "behavioral_signals": behavioral,
                "education_bonus": edu,
            }

            raw_score = (
                w["embedding_similarity"] * semantic +
                w["skill_match"] * skill +
                w["career_fit"] * career +
                w.get("career_progression", 0.0) * prog +
                w["behavioral_signals"] * behavioral +
                w["education_bonus"] * edu
            )

            results.append({
                "candidate_id": cid,
                "candidate": candidate,
                "raw_score": raw_score,
                "components": components,
                "matched_skills": matched,
                "missing_skills": missing,
                "progression_explanation": prog_explanation,
                "trap_mult": 1.0,  # filled in by main pipeline
                "score": raw_score,
            })

        return results
