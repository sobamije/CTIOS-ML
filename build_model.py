
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib

df = pd.read_csv("/Users/sophiaobamije/ctios-ml/real_trials.csv")
print(f"Loaded {len(df)} trials")

known = df[df["status"].isin(["COMPLETED","TERMINATED","WITHDRAWN"])].copy()
print(f"Known outcomes: {len(known)}")
print(f"  COMPLETED:  {(known['status']=='COMPLETED').sum()}")
print(f"  TERMINATED: {(known['status']=='TERMINATED').sum()}")
print(f"  WITHDRAWN:  {(known['status']=='WITHDRAWN').sum()}")

known["failed"] = known["status"].isin(["TERMINATED","WITHDRAWN"]).astype(int)
print(f"Failure rate: {known['failed'].mean()*100:.1f}%")

phase_risk = {
    "EARLY_PHASE1":5,"PHASE1":4,"PHASE1, PHASE2":3,
    "PHASE2":2,"PHASE2, PHASE3":1.5,"PHASE3":1,
    "PHASE4":0.5,"NA":2.5,"":2.5,
}
known["phase_risk"]        = known["phase"].map(phase_risk).fillna(2.5)
known["enrollment"]        = pd.to_numeric(known["enrollment"], errors="coerce").fillna(0)
known["enrollment_log"]    = np.log1p(known["enrollment"])
known["large_trial"]       = (known["enrollment"] > 100).astype(int)
known["eligibility_words"] = known["eligibility_words"].fillna(0)
known["high_complexity"]   = (known["eligibility_words"] > known["eligibility_words"].median()).astype(int)
known["is_industry"]       = (known["sponsor_class"]=="INDUSTRY").astype(int)
known["is_academic"]       = known["sponsor_class"].isin(["OTHER","NETWORK"]).astype(int)
known["enrollment_estimated"] = (known["enrollment_type"]=="ESTIMATED").astype(int)
known["start_year"]        = pd.to_datetime(known["start_date"], errors="coerce").dt.year.fillna(2015)
known["is_interventional"] = (known["study_type"]=="INTERVENTIONAL").astype(int)
known["zero_enrollment"]   = (known["enrollment"]==0).astype(int)
known["enrollment_per_year"] = known["enrollment"] / (2024 - known["start_year"].clip(upper=2023) + 1)
known["site_count"]        = known["site_count"].fillna(0)
known["country_count"]     = known["country_count"].fillna(0)
known["is_multinational"]  = known["is_multinational"].fillna(0)
known["is_oncology"]       = known["is_oncology"].fillna(0)
known["is_rare"]           = known["is_rare"].fillna(0)
known["multi_site"]        = (known["site_count"] > 10).astype(int)
known["site_log"]          = np.log1p(known["site_count"])

FEATURES = [
    "phase_risk",
    "enrollment_log",
    "large_trial",
    "eligibility_words",
    "high_complexity",
    "is_industry",
    "is_academic",
    "enrollment_estimated",
    "start_year",
    "is_interventional",
    "zero_enrollment",
    "enrollment_per_year",
    "site_log",
    "country_count",
    "is_multinational",
    "is_oncology",
    "is_rare",
    "multi_site",
]

X = known[FEATURES]
y = known["failed"]
print(f"{len(FEATURES)} features | {len(X)} trials")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
weights = compute_sample_weight("balanced", y_train)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, random_state=42, subsample=0.8,
    ))
])

print("Training V3 model...")
pipeline.fit(X_train, y_train, model__sample_weight=weights)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_prob)
cv  = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc")

print(f"\n=== V3 MODEL RESULTS ===")
print(f"AUC Score:     {auc:.3f}")
print(f"Cross-Val AUC: {cv.mean():.3f} +/- {cv.std():.3f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Completed','Failed'])}")

print("=== WHAT THE MODEL LEARNED ===")
importances = pipeline.named_steps["model"].feature_importances_
feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=False)
for feat, score in feat_imp.items():
    bar = "X" * int(score * 50)
    print(f"  {feat:<25} {bar} {score:.3f}")

joblib.dump(pipeline, "/Users/sophiaobamije/ctios-ml/real_model.pkl")
print(f"\nV3 model saved.")
