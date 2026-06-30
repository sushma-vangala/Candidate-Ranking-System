"""
precompute_embeddings_compat.py
Lightweight text builder for TF-IDF fallback (no sentence-transformers needed).
"""

def build_candidate_text_light(c: dict) -> str:
    p = c.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", ""), p.get("current_title", ""),
             p.get("current_industry", "")]
    for i, job in enumerate(c.get("career_history", [])[:5]):
        w = max(1, 3 - i)
        parts += [job.get("title", "")] * w + [job.get("description", "")]
    for s in c.get("skills", []):
        prof = s.get("proficiency", "")
        n = {"expert": 3, "advanced": 2, "intermediate": 1}.get(prof, 1)
        parts += [s.get("name", "")] * n
    for edu in c.get("education", []):
        parts += [edu.get("field_of_study", "")]
    return " ".join(x for x in parts if x)
