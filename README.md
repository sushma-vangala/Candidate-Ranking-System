# Redrob Hackathon — Intelligent Candidate Ranker 

**Challenge:** Intelligent Candidate Discovery & Ranking  
**Task:** Rank 100,000 candidates against a Senior AI Engineer JD.  
**Runtime:** ~50 seconds on CPU · No network at rank time · Passes `validate_submission.py`

---

## Architecture

```
candidates.jsonl (100K)
        │
        ▼
[1] JD Parser (regex-based, deterministic)
    Reads job_description text → extracts must-haves, nice-to-haves,
    YoE range, locations, salary band, red flags, notice preference.
    Works on any JD — no hardcoded skill lists.
        │
        ▼
[2] Trap Detection (rule-based, ~12s for 100K)
    ├── Honeypots       — impossible timelines, credential contradictions
    ├── Keyword stuffers — AI skills listed but absent from career descriptions
    └── Behavioral ghosts — inactive 180+d, <20% response rate, <30% interview rate
        │
        ▼
[3] Hybrid Scoring (5 components, weighted)
    ├── MiniLM Embedding Similarity (40%)  ← precomputed, loaded at rank time
    ├── Skill Match with Synonym Awareness (20%)
    ├── Career Fit on Work Content (20%)
    ├── Career Progression Trajectory (10%)
    └── Behavioral Platform Signals (15%) + Education Bonus (5%)
        │
        ▼
[4] Confidence Score per Candidate
    Inter-component agreement → confidence % shown in reasoning
        │
        ▼
[5] Top 100 → CSV with ✓/✗ skill checklist + confidence % per candidate
```

---

## Scoring weights — rationale

| Component | Weight | Why this weight |
|-----------|--------|-----------------|
| Embedding similarity (MiniLM) | **40%** | The challenge explicitly asks to go beyond keyword matching. Semantic similarity is the only signal that captures "Recommendation Systems Engineer at Swiggy" as a fit for "retrieval/ranking" without requiring exact vocabulary overlap. It is the primary discriminator between genuinely aligned candidates and keyword stuffers. |
| Skill match | **20%** | Must-haves matter, but keywords can be gamed. We apply synonym awareness (SBERT ↔ sentence-transformers, FAISS ↔ ANN index) to reduce gaming surface. Weight is kept below embedding because skill lists are self-reported and unverified. |
| Career fit | **20%** | The JD explicitly excludes consulting-only and pure-research backgrounds. This component scores work *content* (ML signals in descriptions, production vs research language, AI role history) not employer names. Equal weight to skill match because the JD's disqualifier section is unusually specific. |
| Career progression | **10%** | Original feature. Rewards sustained ML focus over time (Junior ML → ML Eng → Senior ML scores higher than Data Analyst → Sales → ML Engineer). Kept at 10% because it is a supporting signal, not a primary discriminator. |
| Behavioral signals | **15%** | The JD says explicitly: "a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available." This weight directly follows that instruction. |
| Education bonus | **5%** | The JD does not mention education requirements. Kept as a small tiebreaker only. |

**On weight selection:** These values were chosen based on the JD's emphasis and refined through manual inspection of 50 sample candidates to ensure top-10 results passed a face-validity check. They are not learned from data. If relevance labels were available, a learning-to-rank model (XGBoost/LightGBM) could replace this manual weighting — see Limitations section.

---

## JD Parser — why regex, not an LLM

The parser uses regex section detection and domain-aware pattern matching, not an LLM. This is a deliberate engineering choice, not a limitation:

- **The hackathon requires CPU-only ranking with no network.** An LLM call at parse time would violate both constraints.
- **JDs follow predictable recruiting patterns.** Section headers like "Things you absolutely need" and "Things we'd like" are structurally consistent enough for regex to handle reliably.
- **Regex is deterministic and reproducible.** Given the same JD text, it always produces the same profile. An LLM parser introduces non-determinism that is hard to debug in a competition setting.
- **It generalises to any JD.** The parser works on any job description text — call `build_jd_profile("new_jd.md")` and the scorer adapts automatically. No Python editing required.

---

## Career Progression Scorer — design notes

`career_progression.py` scores five dimensions of career trajectory:

| Dimension | Weight within component | Rationale |
|-----------|------------------------|-----------|
| Recency of ML work | 30% | ML skills evolved rapidly post-2017. Recent ML focus is more predictive of current capability than ML work from 8 years ago. |
| ML role consistency | 25% | Sustained focus ("understood retrieval before it was fashionable" — from JD) preferred over recent conversion. |
| Total months in ML | 20% | Raw accumulated experience in the domain. |
| Seniority trajectory | 15% | Upward movement signals growth; lateral/downward signals plateau. |
| Domain consistency | 10% | ML-adjacent industries (SaaS, fintech, AI startups) preferred over scattered industry history. |

These are heuristic weights chosen to prioritise recent, sustained AI/ML experience for this specific role. They were validated against the 50-candidate sample set and adjusted until rankings matched human intuition on obvious cases (clear ML engineers ranked above clear non-ML engineers).

---

## Trap Detection — threshold rationale

All thresholds are documented inline in `trap_detector.py`. Key ones:

- **6-month job overlap = honeypot signal.** Two full-time roles overlapping by more than 6 months is practically impossible. 3-month grace handles contractor-to-FTE transitions and notice period overlaps.
- **YoE vs graduation year tolerance = 2 years.** Allows for part-time work, internships, and gap years without false-positive honeypot detection.
- **180 days inactive = behavioral ghost.** A candidate who hasn't logged in for 6 months is unlikely to respond to recruiter outreach, regardless of profile quality.
- **Keyword stuffer threshold: ≥6 AI skills + ≤1 career mention.** Someone listing 6+ AI skills with zero mention of those skills in any job description is almost certainly padding their profile.

---

## On evaluation metrics (NDCG, MRR, MAP)

The JD mentions NDCG, MRR, and MAP as required experience. These metrics require *relevance labels* — ground-truth judgments of which candidates are actually good fits for a role.

**The competition dataset contains no relevance labels.** This means supervised ranking metrics cannot be computed directly on this dataset.

Our pipeline is designed to support these metrics when labels are available:

```python
# When you have relevance labels:
from sklearn.metrics import ndcg_score
import numpy as np

# scores[i] = our model score, labels[i] = human relevance (0/1/2)
ndcg = ndcg_score([labels], [scores], k=10)
```

If the evaluation team has a hidden labeled test set, our scores will be evaluated against it. Our confidence scores also provide a proxy for label certainty — high-confidence rankings are more likely to align with human judgment.

---

## How to run

### Requirements
- Python 3.9+, CPU only, ~2 GB RAM, no GPU needed

### Install
```bash
pip3 install scikit-learn numpy scipy
pip3 install sentence-transformers   
```

### Step 1 — Precompute embeddings (once, needs internet, ~8 min on CPU)
```bash
python3 precompute_embeddings.py --candidates ./candidates.jsonl
# Saves: embeddings.npy, candidate_ids.npy, jd_embedding.npy
```

### Step 2 — Rank (no internet needed, ~50 seconds)
```bash
python3 rank.py \
  --candidates ./candidates.jsonl \
  --embeddings ./embeddings.npy \
  --candidate-ids ./candidate_ids.npy \
  --jd-embedding ./jd_embedding.npy \
  --out ./submission.csv
```

### Step 3 — Validate
```bash
python3 validate_submission.py submission.csv
```

### Fallback (no precomputed embeddings)
```bash
python3 rank.py --candidates ./candidates.jsonl --out ./submission.csv

```

---

## File structure

```
IndiaRuns/
│
├── rank.py
├── app.py
├── precompute_embeddings.py
├── validate_submission.py
├── requirements.txt
├── README.md
├── submission_metadata.yaml
│
├── data/
│   ├── sample_candidates.json
│   ├── job_description.md
│   ├── candidate_schema.json
│   ├── redrob_signals_doc.md
│   ├── sample_submission.csv
│   └── submission_spec.md
│
├── embeddings/
│   ├── embeddings.npy
│   ├── candidate_ids.npy
│   └── jd_embedding.npy
│
├── outputs/
│   └── submission.csv
│
└── ranker/
    ├── __init__.py
    ├── jd_parser.py
    ├── scorer.py
    ├── trap_detector.py
    ├── career_progression.py
    ├── adaptive_weights.py
    ├── skill_gap.py
    ├── reasoning.py
    └── precompute_embeddings_compat.py
```

---

## Limitations

We believe in stating limitations clearly. These are the known weaknesses of this system:

**1. Role-Specific Optimization**
The current adaptive scoring strategy is optimized for AI/ML engineering roles. Extending the system with automatically learned role-specific configurations would further improve performance across diverse domains such as frontend, data analytics, and product management.

**2. Advanced JD Understanding**
The current JD parser uses deterministic rule-based parsing for speed and reproducibility under CPU-only constraints. Future versions can incorporate lightweight LLM-based semantic parsing to better understand highly descriptive or unconventional job descriptions.

## Results

• Ranked 100,000 candidates in approximately 50 seconds on CPU

• Generated Top-100 ranked candidates with explainable reasoning

• Adaptive weighting based on Job Description emphasis

• Semantic candidate matching using MiniLM embeddings

• Interactive Streamlit dashboard with ranking analytics

• Submission passes validate_submission.py successfully
