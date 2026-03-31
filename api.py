from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="CTIOS Enrollment Risk API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("/Users/sophiaobamije/ctios-ml/real_model.pkl")

class TrialInput(BaseModel):
    phase: str = "PHASE2"
    enrollment: float = 100
    eligibility_words: int = 200
    sponsor_class: str = "INDUSTRY"
    enrollment_type: str = "ESTIMATED"
    start_year: int = 2020
    study_type: str = "INTERVENTIONAL"

def engineer(t: TrialInput) -> pd.DataFrame:
    phase_risk = {
        "EARLY_PHASE1":5,"PHASE1":4,"PHASE1, PHASE2":3,
        "PHASE2":2,"PHASE2, PHASE3":1.5,"PHASE3":1,
        "PHASE4":0.5,"NA":2.5,"":2.5,
    }
    enrollment = t.enrollment
    start_year = t.start_year
    return pd.DataFrame([{
        "phase_risk":           phase_risk.get(t.phase.upper(), 2.5),
        "enrollment_log":       np.log1p(enrollment),
        "large_trial":          int(enrollment > 100),
        "eligibility_words":    t.eligibility_words,
        "high_complexity":      int(t.eligibility_words > 300),
        "is_industry":          int(t.sponsor_class == "INDUSTRY"),
        "is_academic":          int(t.sponsor_class in ["OTHER","NETWORK"]),
        "enrollment_estimated": int(t.enrollment_type == "ESTIMATED"),
        "start_year":           start_year,
        "is_interventional":    int(t.study_type == "INTERVENTIONAL"),
        "zero_enrollment":      int(enrollment == 0),
        "enrollment_per_year":  enrollment / (2024 - min(start_year, 2023) + 1),
    }])

@app.get("/")
def root():
    return {"system":"CTIOS Enrollment Risk API","owner":"Z'KORA LLC","status":"online","version":"2.0.0"}

@app.post("/score")
def score_trial(trial: TrialInput):
    features = engineer(trial)
    risk_pct = round(float(model.predict_proba(features)[0][1]) * 100, 1)
    if risk_pct >= 70:
        flag,emoji,action = "HIGH","🔴","Immediate intervention recommended."
    elif risk_pct >= 40:
        flag,emoji,action = "MEDIUM","🟡","Schedule proactive check-in within 2 weeks."
    else:
        flag,emoji,action = "LOW","🟢","Standard monitoring sufficient."
    return {
        "risk_score": risk_pct,
        "risk_flag":  flag,
        "emoji":      emoji,
        "action":     action,
        "model":      "GradientBoosting v2 · AUC 0.799 · 1105 real trials",
        "inputs":     trial.dict()
    }
