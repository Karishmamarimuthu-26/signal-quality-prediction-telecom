"""
train_model.py

End-to-end ML pipeline: load -> clean -> feature engineer -> train XGBoost
(classifier, predicting signal quality band) -> evaluate -> feature importance.

Also trains an XGBoost REGRESSOR on the continuous signal_quality_score,
since interviewers may ask about either framing (classification into bands
vs. regression on a continuous KPI).
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
import xgboost as xgb

RESULTS = {}

# ---------- 1. Load ----------
df = pd.read_csv("/home/claude/signal-quality-prediction/data/telecom_signal_quality.csv")

# ---------- 2. Clean ----------
# Median imputation for numeric KPI columns with missingness (robust to outliers,
# appropriate for skewed dBm/count distributions vs. mean imputation)
for col in ["rssi_dbm", "sinr_db", "connected_users"]:
    df[col] = df[col].fillna(df[col].median())

# ---------- 3. Feature engineering ----------
feature_cols = [
    "distance_to_cell_km", "rsrp_dbm", "rssi_dbm", "rsrq_db", "sinr_db",
    "interference_dbm", "network_load_pct", "connected_users", "handovers_last_hr",
]
X = df[feature_cols]

le = LabelEncoder()
y_class = le.fit_transform(df["signal_quality_label"])
y_reg = df["signal_quality_score"]

# ---------- 4. Train/test split (stratified, 80/20) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# ---------- 5. Train XGBoost Classifier ----------
clf = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,      # L2 regularization
    reg_alpha=0.1,       # L1 regularization
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

acc = accuracy_score(y_test, pred)
f1_macro = f1_score(y_test, pred, average="macro")
f1_weighted = f1_score(y_test, pred, average="weighted")
report = classification_report(y_test, pred, target_names=le.classes_, output_dict=True)
cm = confusion_matrix(y_test, pred).tolist()

RESULTS["classification"] = {
    "accuracy": round(acc, 4),
    "f1_macro": round(f1_macro, 4),
    "f1_weighted": round(f1_weighted, 4),
    "classes": le.classes_.tolist(),
    "confusion_matrix": cm,
    "per_class_report": report,
}

fi = dict(zip(feature_cols, clf.feature_importances_.tolist()))
fi_sorted = dict(sorted(fi.items(), key=lambda x: -x[1]))
RESULTS["classification"]["feature_importance"] = {k: round(v, 4) for k, v in fi_sorted.items()}

# ---------- 6. Train XGBoost Regressor (continuous quality score, 0-100) ----------
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
reg = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    objective="reg:squarederror",
    random_state=42,
)
reg.fit(Xr_train, yr_train)
pred_r = reg.predict(Xr_test)

mae = mean_absolute_error(yr_test, pred_r)
rmse = np.sqrt(mean_squared_error(yr_test, pred_r))
r2 = r2_score(yr_test, pred_r)

fi_reg = dict(zip(feature_cols, reg.feature_importances_.tolist()))
fi_reg_sorted = dict(sorted(fi_reg.items(), key=lambda x: -x[1]))

RESULTS["regression"] = {
    "mae": round(mae, 3),
    "rmse": round(rmse, 3),
    "r2": round(r2, 4),
    "feature_importance": {k: round(v, 4) for k, v in fi_reg_sorted.items()},
}

with open("/home/claude/signal-quality-prediction/results/metrics.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print("=== CLASSIFICATION (quality band) ===")
print("Accuracy:", RESULTS["classification"]["accuracy"])
print("F1 (macro):", RESULTS["classification"]["f1_macro"])
print("F1 (weighted):", RESULTS["classification"]["f1_weighted"])
print("\nTop features (classifier):")
for k, v in list(fi_sorted.items())[:5]:
    print(f"  {k}: {v:.4f} ({v*100:.1f}%)")

print("\n=== REGRESSION (continuous quality score 0-100) ===")
print("MAE:", RESULTS["regression"]["mae"])
print("RMSE:", RESULTS["regression"]["rmse"])
print("R2:", RESULTS["regression"]["r2"])
print("\nTop features (regressor):")
for k, v in list(fi_reg_sorted.items())[:5]:
    print(f"  {k}: {v:.4f} ({v*100:.1f}%)")

clf.save_model("/home/claude/signal-quality-prediction/results/xgb_classifier.json")
reg.save_model("/home/claude/signal-quality-prediction/results/xgb_regressor.json")
