"""
app.py — Streamlit Score Breakdown Visualizer
=============================================
Run with: streamlit run app.py

Shows:
  1. Upload candidates + JD → run ranker
  2. Score breakdown per candidate (radar chart + bar chart)
  3. Skill gap analysis per candidate
  4. Adaptive weight explanation (why weights shifted for this JD)
  5. Full ranked table with filters

This is the sandbox required for submission (Section 10.5).
"""

import json
import gzip
import io
import time
import csv
from pathlib import Path

from docx import Document


import streamlit as st
import numpy as np

# ── Page config 
st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports (lazy to keep startup fast) 
import sys
sys.path.insert(0, str(Path(__file__).parent))

from ranker.jd_parser import build_jd_profile, JD_PROFILE
from ranker.scorer import HybridScorer, compute_skill_match_score, \
    compute_career_fit_score, compute_behavioral_score, compute_education_bonus, \
    compute_confidence
from ranker.trap_detector import get_trap_multiplier
from ranker.reasoning import generate_reasoning
from ranker.career_progression import compute_career_progression_score
from ranker.adaptive_weights import compute_adaptive_weights, BASE_WEIGHTS
from ranker.skill_gap import generate_skill_gap_report, format_gap_report_text


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("🎯 Redrob Ranker")
st.sidebar.markdown("**Intelligent Candidate Discovery & Ranking**")
st.sidebar.divider()

mode = st.sidebar.radio(
    "Mode",
    ["Ranked Results", "Upload Candidates"],
    index=0,
)

st.sidebar.divider()
st.sidebar.markdown("**Scoring weights**")
st.sidebar.caption("Adapted automatically from JD text")


# ── Load candidates ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_top100_candidates():
    """
    Load the pre-generated Top 100 candidates.
    """

    file_path = Path(__file__).parent / "data" / "top100_candidates.json"

    if not file_path.exists():
        st.error("data/top100_candidates.json not found.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    return candidates



@st.cache_data(show_spinner=False)
def load_uploaded_candidates(file_bytes, filename):
    candidates = []
    if filename.endswith(".gz"):
        with gzip.open(io.BytesIO(file_bytes), "rt") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    else:
        text = file_bytes.decode("utf-8")
        for line in text.splitlines():
            if line.strip():
                candidates.append(json.loads(line))
    return candidates


# ── Run ranker ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_ranker(candidates_json: str, jd_text: str):
    """Cache-friendly wrapper. candidates_json is JSON string for hashing."""
    candidates = json.loads(candidates_json)
    profile = build_jd_profile(jd_text=jd_text)
    adapted_weights, fired_signals = compute_adaptive_weights(jd_text)
    profile["scoring_weights"] = adapted_weights

    scorer = HybridScorer()
    scorer.build_tfidf_fallback(candidates)
    results = scorer.score_all(candidates)

    for i, r in enumerate(results):
        tm, tr = get_trap_multiplier(r["candidate"])
        r["trap_mult"] = tm
        r["trap_reason"] = tr
        r["score"] = r["raw_score"] * tm
        r["confidence"] = compute_confidence(r["score"], r["components"], tm)
        prog, prog_exp = compute_career_progression_score(r["candidate"])
        r["components"]["career_progression"] = prog
        r["progression_explanation"] = prog_exp
        gap = generate_skill_gap_report(
            r["candidate"],
            r.get("matched_skills", []),
            r.get("missing_skills", []),
            r["components"]["skill_match"],
        )
        r["gap_report"] = gap

    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))

    # Keep only Top 100
    results = results[:100]

    # Assign ranks and reasoning only for Top 100
    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["reasoning"] = generate_reasoning(
            r["candidate"],
            r["rank"],
            r["score"],
            r["components"],
            r.get("matched_skills", []),
            r.get("missing_skills", []),
            r["confidence"],
            r.get("progression_explanation", ""),
        )

    return results, adapted_weights, fired_signals


# ── Radar chart (pure SVG — no external deps) ─────────────────────────────────
def radar_svg(scores: dict, size: int = 260) -> str:
    """Generates an SVG radar chart for component scores."""
    import math
    labels = list(scores.keys())
    values = [min(1.0, max(0.0, v)) for v in scores.values()]
    n = len(labels)
    cx, cy, r = size // 2, size // 2, size // 2 - 40
    angles = [math.pi / 2 - 2 * math.pi * i / n for i in range(n)]

    def point(val, angle, radius=r):
        x = cx + radius * val * math.cos(angle)
        y = cy - radius * val * math.sin(angle)
        return x, y

    # Grid rings
    rings = ""
    for ring in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{point(ring, a)[0]:.1f},{point(ring, a)[1]:.1f}" for a in angles)
        rings += f'<polygon points="{pts}" fill="none" stroke="#e0e0e0" stroke-width="1"/>'

    # Axes
    axes = ""
    for a in angles:
        x, y = point(1.0, a)
        axes += f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#e0e0e0" stroke-width="1"/>'

    # Data polygon
    data_pts = " ".join(f"{point(v, a)[0]:.1f},{point(v, a)[1]:.1f}" for v, a in zip(values, angles))
    data = f'<polygon points="{data_pts}" fill="rgba(99,102,241,0.25)" stroke="#6366f1" stroke-width="2"/>'

    # Dots
    dots = ""
    for v, a in zip(values, angles):
        x, y = point(v, a)
        dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#6366f1"/>'

    # Labels
    label_els = ""
    short = {
        "embedding_similarity": "Semantic",
        "skill_match": "Skills",
        "career_fit": "Career",
        "career_progression": "Progress",
        "behavioral_signals": "Behavioral",
        "education_bonus": "Education",
    }
    for lbl, a in zip(labels, angles):
        x, y = point(1.28, a)
        anchor = "middle"
        if x < cx - 10:
            anchor = "end"
        elif x > cx + 10:
            anchor = "start"
        display = short.get(lbl, lbl)
        label_els += (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
                      f'font-size="11" fill="#374151" font-family="sans-serif">{display}</text>')

    return f"""
    <svg width="{size}" height="{size}" xmlns="http://www.w3.org/2000/svg">
      {rings}{axes}{data}{dots}{label_els}
    </svg>
    """


# ── Bar chart (pure SVG) ───────────────────────────────────────────────────────
def bar_svg(scores: dict, weights: dict, width: int = 340, row_h: int = 32) -> str:
    labels = list(scores.keys())
    short = {
        "embedding_similarity": "Semantic (MiniLM)",
        "skill_match": "Skill match",
        "career_fit": "Career fit",
        "career_progression": "Career trajectory",
        "behavioral_signals": "Behavioral signals",
        "education_bonus": "Education bonus",
    }
    height = row_h * len(labels) + 20
    bar_max_w = width - 160

    rows = ""
    for i, lbl in enumerate(labels):
        val = min(1.0, max(0.0, scores.get(lbl, 0)))
        w_pct = weights.get(lbl, 0)
        bar_w = int(val * bar_max_w)
        y = i * row_h + 10
        color = "#6366f1" if val >= 0.5 else ("#f59e0b" if val >= 0.3 else "#ef4444")
        display = short.get(lbl, lbl)
        rows += f"""
        <text x="0" y="{y+20}" font-size="11" fill="#374151" font-family="sans-serif">{display}</text>
        <rect x="145" y="{y+6}" width="{bar_w}" height="18" fill="{color}" rx="3"/>
        <text x="{145+bar_w+5}" y="{y+20}" font-size="11" fill="#6b7280" font-family="sans-serif">{val:.0%} (wt {w_pct:.0%})</text>
        """

    return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">{rows}</svg>"""


# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("🎯 Redrob Intelligent Candidate Ranker")
st.caption("Hybrid ranking: MiniLM embeddings · Skill synonyms · Career trajectory · Adaptive weights")

# ── Load data ──────────────────────────────────────────────────────────────────
if mode == "Ranked Results":
    candidates = load_top100_candidates()
    if not candidates:
        st.error("data/top100_candidates.json not found.")
        st.stop()
    jd_text = JD_PROFILE["jd_embedding_text"]
    st.info("Top 100 Candidate Recommendations")

else:
    col1, col2 = st.columns(2)
    with col1:
        cand_file = st.file_uploader("Upload candidates (.jsonl or .jsonl.gz)", type=["jsonl", "gz"])
    with col2:
        jd_file = st.file_uploader("Upload job description (.docx, .txt or .md)",type=["docx", "txt", "md"])

    if not cand_file:
        st.info("Upload a candidates file to get started. You can use `top100_candidates.json` from the bundle.")
        st.stop()

    candidates = load_uploaded_candidates(cand_file.read(), cand_file.name)
    if jd_file:
        if jd_file.name.endswith(".docx"):
            doc = Document(jd_file)
            jd_text = "\n".join(p.text for p in doc.paragraphs)
        else:
            jd_text = jd_file.read().decode("utf-8")
    else:
        jd_text = JD_PROFILE["jd_embedding_text"]
        st.success(f"Loaded {len(candidates):,} candidates")

# ── Run ranker ─────────────────────────────────────────────────────────────────
with st.spinner(f"Ranking {len(candidates):,} candidates..."):
    t0 = time.time()
    results, adapted_weights, fired_signals = run_ranker(
        json.dumps([c for c in candidates]), jd_text
    )
    elapsed = time.time() - t0

st.success(f"Loaded Top {len(results)} ranked candidates")

# ── Adaptive weights sidebar display ──────────────────────────────────────────
for comp, w in adapted_weights.items():
    base = BASE_WEIGHTS.get(comp, 0)
    delta = w - base
    label = comp.replace("_", " ").title()
    arrow = f" ↑{delta:+.0%}" if delta > 0.005 else (f" ↓{delta:+.0%}" if delta < -0.005 else "")
    st.sidebar.metric(label, f"{w:.0%}", arrow if arrow else None)

if fired_signals:
    st.sidebar.divider()
    st.sidebar.markdown("**Why weights adapted:**")
    for sig in fired_signals:
        st.sidebar.caption(f"• {sig['rationale']}")

# ── Main content tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏆 Rankings", "🔍 Candidate Deep Dive", "📊 Score Distribution"])

# ── Tab 1: Rankings table ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Top Candidates")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_confidence = st.slider("Min confidence %", 0, 100, 0, 5)
    with col2:
        gap_filter = st.selectbox("Gap severity", ["All", "minimal", "minor", "moderate", "critical"])
    with col3:
        show_n = st.selectbox("Show top N", [10, 25, 50, 100], index=1)

    filtered = [
        r for r in results
        if int(r["confidence"] * 100) >= min_confidence
        and (gap_filter == "All" or r.get("gap_report", {}).get("gap_severity") == gap_filter)
    ][:show_n]

    if not filtered:
        st.warning("No candidates match the current filters.")
    else:
        for r in filtered:
            profile = r["candidate"].get("profile", {})
            gap = r.get("gap_report", {})
            signals = r["candidate"].get("redrob_signals", {})

            severity_color = {
                "minimal": "🟢", "minor": "🟡",
                "moderate": "🟠", "critical": "🔴"
            }.get(gap.get("gap_severity", "moderate"), "⚪")

            trap_badge = " 🚨" if r["trap_mult"] == 0.0 else (" ⚠️" if r["trap_mult"] < 0.8 else "")

            with st.expander(
                f"Rank {r['rank']:3d} | {r['candidate_id']}{trap_badge} | "
                f"{profile.get('current_title','?')} at {profile.get('current_company','?')} | "
                f"Score: {r['score']:.3f} | Confidence: {int(r['confidence']*100)}% | "
                f"Skill gap: {severity_color} {gap.get('gap_severity','?')}"
            ):
                col_left, col_right = st.columns([1, 1])

                with col_left:
                    st.markdown("**Score breakdown**")
                    st.components.v1.html(
                        radar_svg(r["components"]),
                        height=270, scrolling=False
                    )

                with col_right:
                    st.markdown("**Component scores**")
                    st.components.v1.html(
                        bar_svg(r["components"], adapted_weights),
                        height=220, scrolling=False
                    )

                st.markdown(f"**Reasoning:** {r['reasoning']}")

                if gap:
                    st.markdown(f"**Skill coverage:** {gap['coverage_pct']}% "
                                f"({gap['n_matched']}/{gap['n_must']} required) | "
                                f"{gap['upskill_note']}")

                if r["trap_mult"] < 1.0 and r["trap_reason"]:
                    st.warning(f"⚠️ Trap signal: {r['trap_reason']}")

# ── Tab 2: Candidate deep dive ─────────────────────────────────────────────────
with tab2:
    st.subheader("Candidate Deep Dive")

    candidate_options = {
        f"Rank {r['rank']}: {r['candidate_id']} — {r['candidate']['profile'].get('current_title','?')}": r
        for r in results[:50]
    }
    selected_label = st.selectbox("Select candidate", list(candidate_options.keys()))
    r = candidate_options[selected_label]
    profile = r["candidate"].get("profile", {})
    signals = r["candidate"].get("redrob_signals", {})
    gap = r.get("gap_report", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Score", f"{r['score']:.4f}")
    col2.metric("Confidence", f"{int(r['confidence']*100)}%")
    col3.metric("Skill coverage", f"{gap.get('coverage_pct', 0):.0f}%")
    col4.metric("Trap multiplier", f"{r['trap_mult']:.2f}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### Score Radar")
        st.components.v1.html(radar_svg(r["components"], size=300), height=310)

    with col_r:
        st.markdown("### Skill Gap Analysis")
        if gap.get("matched"):
            st.markdown("**✓ Matched skills:**")
            for m in gap["matched"][:8]:
                st.markdown(f"  - {m['display']} `{m['match_type']}`")
        if gap.get("missing"):
            st.markdown("**✗ Missing skills:**")
            for m in gap["missing"][:6]:
                learn_emoji = {"high": "📗", "medium": "📙", "low": "📕"}.get(m["learnability"], "")
                st.markdown(f"  - {m['display']} {learn_emoji} learnability: `{m['learnability']}`  "
                            f"  score cost: `-{m['score_cost']:.4f}`")

    st.markdown("### Career History")
    for job in r["candidate"].get("career_history", [])[:4]:
        st.markdown(f"**{job.get('title')}** at {job.get('company')} "
                    f"({job.get('start_date','?')} – {job.get('end_date','present')}, "
                    f"{job.get('duration_months',0)}mo)")
        st.caption(job.get("description", "")[:200] + "...")

    st.markdown(f"**Reasoning:** {r['reasoning']}")
    if r.get("progression_explanation"):
        st.caption(f"Career trajectory: {r['progression_explanation']}")

# ── Tab 3: Score distribution ──────────────────────────────────────────────────
with tab3:
    st.subheader("Score Distribution")

    scores = [r["score"] for r in results]
    confidences = [r["confidence"] for r in results]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Top score", f"{max(scores):.4f}")
        st.metric("Median score", f"{sorted(scores)[len(scores)//2]:.4f}")
        st.metric("Score at rank 10", f"{results[min(9,len(results)-1)]['score']:.4f}")
        if len(results) >= 100:
            st.metric("Score at rank 100", f"{results[99]['score']:.4f}")

    with col2:
        trap_count = sum(1 for r in results if r["trap_mult"] == 0.0)
        penalized = sum(1 for r in results if 0 < r["trap_mult"] < 1.0)
        st.metric("Honeypots detected", trap_count)
        st.metric("Partially penalized", penalized)
        st.metric("Clean candidates", len(results) - trap_count - penalized)

    st.markdown("### Adaptive Weight Explanation")
    if fired_signals:
        st.markdown("The following JD signals caused weights to shift from baseline:")
        for sig in fired_signals:
            st.info(f"**{sig['name'].title()}** ({sig['hits']} match(es)): {sig['rationale']}")
    else:
        st.info("No strong emphasis signals detected. Base weights used.")

    weight_data = {
        "Component": list(adapted_weights.keys()),
        "Adapted weight": [f"{v:.0%}" for v in adapted_weights.values()],
        "Base weight": [f"{BASE_WEIGHTS.get(k,0):.0%}" for k in adapted_weights.keys()],
        "Delta": [f"{adapted_weights[k]-BASE_WEIGHTS.get(k,0):+.0%}" for k in adapted_weights.keys()],
    }
    st.table(weight_data)

    st.markdown("### Gap Severity Distribution")
    severity_counts = {}
    for r in results:
        sev = r.get("gap_report", {}).get("gap_severity", "unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    for sev, count in sorted(severity_counts.items()):
        emoji = {"minimal":"🟢","minor":"🟡","moderate":"🟠","critical":"🔴"}.get(sev,"⚪")
        st.markdown(f"{emoji} **{sev.title()}**: {count} candidates")
