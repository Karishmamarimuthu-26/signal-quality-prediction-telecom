"""
generate_dataset.py

Generates a synthetic telecom network KPI dataset modeled on real 3GPP
LTE/5G signal parameter definitions and their known statistical relationships
(RSRP, RSRQ, RSSI, SINR, PRB utilization / network load, distance-to-cell,
interference). Public labeled "signal quality" telecom datasets with ground
truth are not freely redistributable (operator data is proprietary; academic
drive-test sets like SRFG's LTE-4G-HIGHWAY-DRIVE-TESTS-SALZBURG or the
Vienna 4G/5G Drive-Test Dataset require registration). This generator
reproduces the same feature relationships (3GPP TS 36.214 definitions) so the
pipeline, model, and evaluation are fully real and reproducible end-to-end.

Ground-truth signal quality label is derived from a realistic formula
combining RSRP, SINR, RSRQ, load and interference with added noise, then
binned into 4 quality classes (Poor/Fair/Good/Excellent) -- mirroring how
operators bucket KPIs into quality bands for NOC dashboards.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 15000

# --- Core radio features (3GPP-standard ranges) ---
distance_km = np.random.exponential(scale=0.45, size=N).clip(0.02, 3.5)       # distance to serving cell
rsrp = -44 - (20 * np.log10(distance_km * 1000 + 1)) + np.random.normal(0, 4, N)   # RSRP dBm (~ -140 to -44)
rsrp = rsrp.clip(-140, -44)

rssi = rsrp + np.random.normal(18, 3, N)                                       # RSSI = RSRP + wideband noise/interference (dBm)
rssi = rssi.clip(-110, -40)

network_load_pct = np.random.beta(2, 3, N) * 100                               # PRB utilization / cell load (%)
interference_dbm = -110 + (network_load_pct / 100) * 25 + np.random.normal(0, 3, N)  # interference rises with load

sinr = rsrp - interference_dbm - np.random.normal(5, 2, N)                     # SINR proxy (dB)
sinr = sinr.clip(-10, 30)

rsrq = (rsrp - rssi) + np.random.normal(0, 1, N)                               # RSRQ (dB), ~ -19.5 to -3
rsrq = rsrq.clip(-19.5, -3)

handovers_last_hr = np.random.poisson(1.5 + distance_km, N)                    # mobility/edge-of-cell proxy
num_connected_users = np.random.poisson(20 + network_load_pct * 0.6, N)

# --- Ground-truth signal quality score (0-100), realistic weighted combo ---
quality_score = (
    0.35 * ((rsrp + 140) / 96 * 100) +          # RSRP normalized
    0.30 * ((sinr + 10) / 40 * 100) +           # SINR normalized
    0.20 * ((rsrq + 19.5) / 16.5 * 100) +       # RSRQ normalized
    0.15 * (100 - network_load_pct)             # load penalty
    + np.random.normal(0, 4, N)
).clip(0, 100)

def bucket(q):
    if q < 35: return "Poor"
    if q < 55: return "Fair"
    if q < 75: return "Good"
    return "Excellent"

quality_label = np.array([bucket(q) for q in quality_score])

df = pd.DataFrame({
    "distance_to_cell_km": distance_km.round(3),
    "rsrp_dbm": rsrp.round(2),
    "rssi_dbm": rssi.round(2),
    "rsrq_db": rsrq.round(2),
    "sinr_db": sinr.round(2),
    "interference_dbm": interference_dbm.round(2),
    "network_load_pct": network_load_pct.round(2),
    "connected_users": num_connected_users,
    "handovers_last_hr": handovers_last_hr,
    "signal_quality_score": quality_score.round(2),
    "signal_quality_label": quality_label,
})

# Inject realistic missingness/noise (as in real drive-test logs)
for col in ["rssi_dbm", "sinr_db", "connected_users"]:
    idx = np.random.choice(N, size=int(N * 0.02), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv("/home/claude/signal-quality-prediction/data/telecom_signal_quality.csv", index=False)
print(df.head())
print("\nShape:", df.shape)
print("\nLabel distribution:\n", df["signal_quality_label"].value_counts())
print("\nMissing values:\n", df.isna().sum())
