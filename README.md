# CTIOS Enrollment Risk Pipeline
### Clinical Trial Intelligence Operating System — Predictive Intelligence Layer
**Z'KORA LLC** | Built by Sophia Obamije

---

## What This Is

A machine learning pipeline that predicts which clinical trials are at risk of enrollment failure — before it happens.

Trained on **500 real trials** pulled live from the ClinicalTrials.gov API, the model identifies patterns in protocol complexity, enrollment size, and eligibility criteria that predict whether a trial will be terminated or withdrawn.

This pipeline powers the **SiteRadar** module of the CTIOS platform.

---

## Results

| Metric | Score |
|---|---|
| AUC Score | **0.888** |
| Cross-Validation AUC | 0.826 ± 0.091 |
| Overall Accuracy | 85% |
| Training Data | 500 real ClinicalTrials.gov trials |

---

## What The Model Learned

The two strongest predictors of trial failure — discovered by the model from real data:

1. **Enrollment size (56%)** — larger trials fail more often
2. **Eligibility complexity (30%)** — stricter inclusion/exclusion criteria = harder recruitment

These findings align with what clinical operations teams already know intuitively, which validates that the model is learning real signal rather than noise.

---

## Pipeline Architecture
```
ClinicalTrials.gov API
        ↓
  fetch_trials.py        ← pulls 500 real trials via REST API
        ↓
  Feature Engineering    ← enrollment size, eligibility complexity,
                            phase risk, sponsor type, start year
        ↓
  sklearn Pipeline       ← StandardScaler + GradientBoostingClassifier
        ↓
  Evaluation             ← AUC, cross-validation, classification report
        ↓
  real_model.pkl         ← saved model ready for production inference
        ↓
  score_trials()         ← scores new trials 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW
```

---

## Files

| File | What It Does |
|---|---|
| `fetch_trials.py` | Pulls real trial data from ClinicalTrials.gov API v2 |
| `build_model.py` | Feature engineering, pipeline training, evaluation |
| `real_trials.csv` | 500 real trials downloaded from ClinicalTrials.gov |
| `real_model.pkl` | Trained model (generated when you run build_model.py) |

---

## How To Run It
```bash
# Install dependencies
pip install requests pandas scikit-learn joblib

# Pull fresh data from ClinicalTrials.gov
python3 fetch_trials.py

# Train the model and see results
python3 build_model.py
```

---

## Clinical Context

Enrollment failure is one of the most expensive problems in drug development. Roughly 80% of clinical trials fail to meet enrollment timelines, costing sponsors an average of $8M per month in delays.

This pipeline gives clinical operations teams an early warning system — flagging at-risk trials before delays become failures.

---

## Connection To CTIOS

This repository is the **predictive intelligence backend** for [CTIOS](https://harmonyhavenai.com) — the Clinical Trial Intelligence Operating System built by Z'KORA LLC.

The SiteRadar module in CTIOS is designed to surface these risk scores to clinical teams in real time.

---

*Built with Python, scikit-learn, and the ClinicalTrials.gov API v2*
*© 2025 Z'KORA LLC — Proprietary and Confidential*
