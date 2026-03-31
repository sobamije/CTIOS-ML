from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd

app = FastAPI(title="CTIOS Enrollment Risk API", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
model = joblib.load("real_model.pkl")

class TrialInput(BaseModel):
    phase: str = "PHASE2"
    enrollment: float = 100
    eligibility_words: int = 200
    sponsor_class: str = "INDUSTRY"
    enrollment_type: str = "ESTIMATED"
    start_year: int = 2020
    study_type: str = "INTERVENTIONAL"
    site_count: int = 5
    country_count: int = 1
    is_oncology: int = 0
    is_rare: int = 0

def engineer(t):
    pr = {"EARLY_PHASE1":5,"PHASE1":4,"PHASE1, PHASE2":3,"PHASE2":2,"PHASE2, PHASE3":1.5,"PHASE3":1,"PHASE4":0.5,"NA":2.5,"":2.5}
    e = t.enrollment
    y = t.start_year
    return pd.DataFrame([{
        "phase_risk": pr.get(t.phase.upper(), 2.5),
        "enrollment_log": np.log1p(e),
        "large_trial": int(e > 100),
        "eligibility_words": t.eligibility_words,
        "high_complexity": int(t.eligibility_words > 300),
        "is_industry": int(t.sponsor_class == "INDUSTRY"),
        "is_academic": int(t.sponsor_class in ["OTHER","NETWORK"]),
        "enrollment_estimated": int(t.enrollment_type == "ESTIMATED"),
        "start_year": y,
        "is_interventional": int(t.study_type == "INTERVENTIONAL"),
        "zero_enrollment": int(e == 0),
        "enrollment_per_year": e / (2024 - min(y, 2023) + 1),
        "site_log": np.log1p(t.site_count),
        "country_count": t.country_count,
        "is_multinational": int(t.country_count > 1),
        "is_oncology": t.is_oncology,
        "is_rare": t.is_rare,
        "multi_site": int(t.site_count > 10),
    }])

@app.get("/")
def root():
    return {"system":"CTIOS Enrollment Risk API","owner":"Z KORA LLC","status":"online","version":"3.0.0"}


def apply_guardrails(risk_pct: float, trial) -> tuple:
    flags = []
    score = risk_pct

    # Rule 1: Early phase with unrealistic enrollment
    if trial.phase.upper() in ["EARLY_PHASE1","PHASE1"] and trial.enrollment > 200:
        score = max(score, 75.0)
        flags.append("Early phase trial with unrealistic enrollment target")

    # Rule 2: Zero enrollment with estimated type
    if trial.enrollment == 0 and trial.enrollment_type == "ESTIMATED":
        score = max(score, 65.0)
        flags.append("Zero enrollment with estimated count — high withdrawal risk")

    # Rule 3: Too few sites for large enrollment
    if trial.site_count < 3 and trial.enrollment > 300:
        score = max(score, 70.0)
        flags.append("Insufficient sites for enrollment target")

    # Rule 4: Rare disease with complex eligibility
    if trial.is_rare == 1 and trial.eligibility_words > 800:
        score = min(100.0, score + 20.0)
        flags.append("Rare disease with highly complex eligibility criteria")

    # Rule 5: Academic sponsor with large enrollment
    if trial.sponsor_class == "OTHER" and trial.enrollment > 400:
        score = min(100.0, score + 15.0)
        flags.append("Academic sponsor with large enrollment target")

    # Rule 6: Very high eligibility complexity
    if trial.eligibility_words > 1200:
        score = min(100.0, score + 10.0)
        flags.append("Extremely complex eligibility criteria")

    return round(score, 1), flags

@app.post("/score")
def score_trial(trial: TrialInput):
    features = engineer(trial)
    ml_score = round(float(model.predict_proba(features)[0][1]) * 100, 1)
    final_score, guardrail_flags = apply_guardrails(ml_score, trial)

    if final_score >= 70:
        flag,action = "HIGH","Immediate intervention recommended."
    elif final_score >= 40:
        flag,action = "MEDIUM","Schedule proactive check-in within 2 weeks."
    else:
        flag,action = "LOW","Standard monitoring sufficient."

    return {
        "risk_score":      final_score,
        "ml_score":        ml_score,
        "risk_flag":       flag,
        "action":          action,
        "guardrail_flags": guardrail_flags,
        "model":           "GradientBoosting v3 + Guardrails · AUC 0.826 · 1105 real trials",
        "inputs":          trial.dict()
    }

    features = engineer(trial)

