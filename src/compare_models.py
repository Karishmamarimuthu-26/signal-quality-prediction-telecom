"""
compare_models.py
Trains Linear/Logistic Regression and Random Forest baselines alongside
XGBoost so the "why XGBoost over X" interview answer is backed by real numbers.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

df = pd.read_csv("/home/claude/signal-quality-prediction/data/telecom_signal_quality.csv")
for col in ["rssi_dbm", "sinr_db", "connected_users"]:
    df[col] = df[col].fillna(df[col].median())

feature_cols = [
    "distance_to_cell_km", "rsrp_dbm", "rssi_dbm", "rsrq_db", "sinr_db",
    "interference_dbm", "network_load_pct", "connected_users", "handovers_last_hr",
]
X = df[feature_cols]
le = LabelEncoder()
y = le.fit_transform(df["signal_quality_label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

results = {}

# Logistic Regression (needs scaling)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_s, y_train)
pred = lr.predict(X_test_s)
results["Logistic Regression"] = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
}

# Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
results["Random Forest"] = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
}

# XGBoost (same config as train_model.py)
xgb_clf = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.1,
    objective="multi:softprob", eval_metric="mlogloss", random_state=42,
)
xgb_clf.fit(X_train, y_train)
pred = xgb_clf.predict(X_test)
results["XGBoost"] = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
}

with open("/home/claude/signal-quality-prediction/results/model_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
