# Signal Quality Prediction Using ML (Telecom Dataset)

Predicting mobile network signal quality from radio KPIs using XGBoost —
built to explore ML pipelines relevant to telecom network optimization and
analytics (e.g. the kind of work done at Nokia).

## 1. Problem
Predict signal quality (both as a **4-class band**: Poor / Fair / Good /
Excellent, and as a **continuous 0–100 score**) from radio-level network
KPIs, so degradation can be flagged and explained before it affects users.

## 2. Data
`data/telecom_signal_quality.csv` — 15,000 samples, 9 features.

**Source:** synthetically generated (`src/generate_dataset.py`), modeled
directly on real **3GPP TS 36.214** LTE/5G KPI definitions and their known
statistical relationships (RSRP ≈ f(distance), RSSI = RSRP + interference,
SINR = RSRP − interference, RSRQ = RSRP − RSSI, etc.), with Gaussian noise
and 2% missingness injected per column to mimic real drive-test logs.

*Why synthetic:* labeled operator KPI datasets with signal-quality ground
truth are proprietary and not freely redistributable. Public academic
alternatives exist (e.g. SRFG's **LTE-4G-HIGHWAY-DRIVE-TESTS-SALZBURG**
dataset, or the **Vienna 4G/5G Drive-Test Dataset**, both real drive-test
corpora from Austria) but require registration/licensing, so this project
reproduces the same feature relationships to keep the full pipeline
runnable and reproducible by anyone who clones the repo.
**Be upfront about this if asked** — it's a legitimate and common approach
for portfolio projects, and it means every number in this README is real,
not invented.

## 3. Features
| Feature | Definition |
|---|---|
| `rsrp_dbm` | Reference Signal Received Power — average power of one reference signal (dBm, ~-140 to -44). Best single indicator of raw coverage strength. |
| `rssi_dbm` | Received Signal Strength Indicator — total wideband power incl. signal + interference + noise (dBm). |
| `rsrq_db` | Reference Signal Received Quality = RSRP − RSSI (dB). Captures interference relative to signal. |
| `sinr_db` | Signal-to-Interference-plus-Noise Ratio (dB). The strongest single predictor of usable link quality. |
| `interference_dbm` | Estimated interference power, correlated with cell load. |
| `network_load_pct` | PRB (Physical Resource Block) utilization — how busy the serving cell is. |
| `distance_to_cell_km` | Distance from UE to serving cell. |
| `connected_users` | Number of users concurrently connected to the cell. |
| `handovers_last_hr` | Handover count in the last hour — proxy for mobility / cell-edge behavior. |

## 4. Model: XGBoost
Gradient-boosted decision tree ensemble. Trained both as a classifier
(quality band) and a regressor (continuous score) — see `src/train_model.py`.

**Why XGBoost over alternatives — with numbers from this run** (`src/compare_models.py`):

| Model | Accuracy | F1 (macro) |
|---|---|---|
| Logistic Regression | 0.861 | 0.691 |
| Random Forest | 0.857 | 0.631 |
| **XGBoost** | **0.864** | **0.720** |

- **vs. Linear/Logistic Regression:** signal quality is a non-linear
  function of its inputs (e.g. SINR's effect on quality saturates at high
  values and collapses sharply below a threshold). Linear models can't
  capture that curvature or interactions (e.g. high load only hurts
  quality when SINR is already marginal); trees can.
- **vs. Random Forest:** XGBoost's boosting corrects prior trees' errors
  sequentially rather than averaging independent trees, plus built-in L1/L2
  regularization (`reg_alpha`, `reg_lambda`) — measurably better macro-F1
  here, which matters because quality bands are imbalanced (few
  "Excellent" samples).
- **vs. Neural network:** on ~15K tabular rows with mixed-scale numeric
  features, a deep net offers no accuracy advantage and loses the
  interpretable, per-feature importance a telecom analyst needs to explain
  *why* quality dropped — not just predict that it did.

## 5. Feature Importance (the answer to "top 3 features")
From the trained XGBoost **classifier** (`results/metrics.json`,
`results/feature_importance.png`):

1. **SINR — 54.0%** — by far the dominant driver. Makes physical sense:
   SINR already folds together signal strength *and* interference/noise,
   so it's the most direct proxy for "can this link actually be decoded
   cleanly."
2. **RSRP — 15.4%** — raw coverage strength, second most important once
   SINR is accounted for.
3. **RSRQ — 8.8%** — adds interference-relative-to-signal information not
   fully captured by RSRP alone.
4. Network load (6.2%) and interference (5.0%) matter less directly
   because their effect is already largely mediated through SINR.

*(Regressor importances are similar but even more concentrated on SINR —
75.8% — since the continuous score responds more smoothly to it.)*

**Takeaway to say out loud:** "SINR alone explains over half the variance
in predicted quality, which lines up with telecom theory — it's the KPI
that already combines signal strength and interference into one number.
That told me interference management and cell-edge SINR, not just raw
coverage (RSRP), is the leverage point for improving perceived quality."

## 6. Evaluation
Two framings, both evaluated on a held-out 20% test set:

**Classification (quality band):**
- Accuracy: **86.4%**
- F1 (macro): **0.720** — reported because classes are imbalanced (Poor/Fair
  are common, Excellent is rare); macro-F1 avoids the metric being dominated
  by the majority classes.
- F1 (weighted): **0.863**

**Regression (continuous 0–100 score):**
- MAE: **3.32**
- RMSE: **4.16**
- R²: **0.910**

Full classification report + confusion matrix: `results/metrics.json`.

## 7. Connection to Nokia / network optimization
This started as an extension of BSNL internship exposure — working
adjacent to live network operations data — into a structured ML pipeline:
raw KPI logs → cleaning → feature engineering → a model that doesn't just
predict "quality is dropping" but *explains why* via feature importance.
That explainability piece is what maps onto Nokia's network optimization
and analytics domain: a NOC engineer doesn't just want a quality score,
they want to know whether to look at interference mitigation, capacity
(load), or coverage (RSRP) — which is exactly what feature importance
ranks for them here.

## Pipeline (end to end)
```
data → generate_dataset.py (or real KPI export)
     → clean: median imputation for missing RSSI/SINR/user-count (robust to skewed dBm dists)
     → feature engineering: 9 radio KPIs
     → train/test split: 80/20, stratified by quality band, random_state=42
     → model: XGBoost (300 trees, depth 5, lr 0.05, L1+L2 reg)
     → evaluate: accuracy, macro/weighted F1 (classifier); MAE/RMSE/R² (regressor)
     → feature importance: gain-based, from the trained model
```

## Handling missing/noisy data
2% of `rssi_dbm`, `sinr_db`, `connected_users` are missing (simulating
dropped drive-test samples). Used **median imputation** rather than mean —
dBm and count distributions are skewed, so the median is more robust to
outliers than the mean.

## Deploying this in a real telecom ops setting
- Batch-score cell-level KPI exports on a schedule (e.g. every 15 min from
  the OSS/NMS), write flagged "Poor"/"Fair" cells to a NOC dashboard.
- Retrain periodically as network topology/load patterns shift (concept
  drift) — track feature importance drift as an early signal something
  changed operationally.
- Pair the classifier (band, for alerting) with the regressor (continuous
  score, for trending) — this repo trains both for exactly that reason.

## XGBoost vs. gradient boosting in general
Gradient boosting is the general algorithm: build trees sequentially, each
fitting the *residual error* of the ensemble so far, combined via gradient
descent on a loss function. **XGBoost** is a specific, optimized
implementation of that idea: adds explicit L1/L2 regularization terms to
the objective (reduces overfitting vs. vanilla GBM), uses a second-order
(Newton) approximation of the loss for faster/more accurate splits,
supports parallelized tree construction, and handles missing values
natively during split-finding.

## Project structure
```
signal-quality-prediction/
├── data/telecom_signal_quality.csv       # generated dataset
├── src/
│   ├── generate_dataset.py               # dataset generation
│   ├── train_model.py                    # main pipeline: clean, train, evaluate
│   └── compare_models.py                 # XGBoost vs LR vs RF baseline comparison
├── results/
│   ├── metrics.json                      # full metrics + feature importances
│   ├── model_comparison.json             # baseline comparison numbers
│   ├── feature_importance.png            # chart
│   ├── xgb_classifier.json               # saved model
│   └── xgb_regressor.json                # saved model
└── README.md
```

## Reproduce
```bash
pip install xgboost scikit-learn pandas numpy matplotlib
python src/generate_dataset.py
python src/train_model.py
python src/compare_models.py
```
