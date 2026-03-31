import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# ── LOAD REAL DATA ─────────────────────────────────────────────────────
df = pd.read_csv("~/ctios-ml/real_trials.csv")
print(f"Loaded {len(df)} real trials from ClinicalTrials.gov")

# ── FILTER TO KNOWN OUTCOMES ONLY ──────────────────────────────────────
# We can only train on trials where we KNOW what happened
# Ongoing trials (RECRUITING, ACTIVE, etc.) get filtered out
known = df[df["status"].isin(["COMPLETED", "TERMINATED", "WITHDRAWN"])].copy()
print(f"Trials with known outcomes: {len(known)}")
print(f"  COMPLETED:  {(known['status']=='COMPLETED').sum()}")
print(f"  TERMINATED: {(known['status']=='TERMINATED').sum()}")
print(f"  WITHDRAWN:  {(known['status']=='WITHDRAWN').sum()}")

# ── CREATE TARGET LABEL ────────────────────────────────────────────────
# 1 = trial failed (terminated or withdrawn)
# 0 = trial succeeded (completed)
known["failed"] = known["status"].isin(["TERMINATED", "WITHDRAWN"]).astype(int)
print(f"\nFailure rate: {known['failed'].mean()*100:.1f}% of trials with known outcomes")

# ── FEATURE ENGINEERING ────────────────────────────────────────────────
print("\nEngineering features from real data...")

# Phase — encode as risk level (earlier phase = more experimental = riskier)
phase_risk = {
    "EARLY_PHASE1": 5,
    "PHASE1": 4,
    "PHASE1, PHASE2": 3,
    "PHASE2": 2,
    "PHASE2, PHASE3": 1.5,
    "PHASE3": 1,
    "PHASE4": 0.5,
    "NA": 2.5,
    "": 2.5,
}
known["phase_risk"] = known["phase"].map(phase_risk).fillna(2.5)

# Enrollment size — larger trials are harder to fill
known["enrollment"] = pd.to_numeric(known["enrollment"], errors="coerce").fillna(0)
known["large_trial"] = (known["enrollment"] > 100).astype(int)
known["enrollment_log"] = np.log1p(known["enrollment"])  # log scale handles outliers

# Eligibility complexity — more words = stricter criteria = harder to enroll
known["eligibility_words"] = known["eligibility_words"].fillna(0)
known["high_complexity"] = (known["eligibility_words"] > known["eligibility_words"].median()).astype(int)

# Sponsor type — industry (pharma) vs academic medical center
known["is_industry"] = (known["sponsor_class"] == "INDUSTRY").astype(int)

# Enrollment type — ESTIMATED means they're guessing (riskier than ACTUAL)
known["enrollment_estimated"] = (known["enrollment_type"] == "ESTIMATED").astype(int)

# Start year — older trials had different success rates
known["start_year"] = pd.to_datetime(
    known["start_date"], errors="coerce"
).dt.year.fillna(2015)

FEATURES = [
    "phase_risk",
    "enrollment_log",
    "large_trial",
    "eligibility_words",
    "high_complexity",
    "is_industry",
    "enrollment_estimated",
    "start_year",
]

X = known[FEATURES]
y = known["failed"]

print(f"  {len(FEATURES)} features ready")
print(f"  {len(X)} trials ready for training\n")

# ── BUILD AND TRAIN PIPELINE ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ))
])

print("Training on real ClinicalTrials.gov data...")
pipeline.fit(X_train, y_train)

# ── EVALUATE ───────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
cv  = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")

print(f"\n=== REAL DATA RESULTS ===")
print(f"AUC Score:     {auc:.3f}  (trained on REAL trials)")
print(f"Cross-Val AUC: {cv.mean():.3f} ± {cv.std():.3f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Completed','Failed'])}")

# ── FEATURE IMPORTANCE ─────────────────────────────────────────────────
print("=== WHAT THE MODEL LEARNED ===")
importances = pipeline.named_steps["model"].feature_importances_
feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=False)
for feat, score in feat_imp.items():
    bar = "█" * int(score * 50)
    print(f"  {feat:<25} {bar} {score:.3f}")

# ── SAVE ───────────────────────────────────────────────────────────────
joblib.dump(pipeline, "/Users/sophiaobamije/ctios-ml/real_model.pkl")
print(f"\n✅ Real model saved → /Users/sophiaobamije/ctios-ml/real_model.pkl")
print("✅ This model was trained on actual ClinicalTrials.gov outcomes.")
