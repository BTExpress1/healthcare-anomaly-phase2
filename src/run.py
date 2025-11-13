import os, duckdb, numpy as np, pandas as pd
from dotenv import load_dotenv; load_dotenv()
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import plotly.express as px

# determine project root (parent of src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


DB = os.getenv("DUCKDB_PATH", "../healthcare-stream-phase1/data/warehouse.duckdb")
W = int(os.getenv("WINDOW_DAYS", "7")); MIN_ROWS = int(os.getenv("MIN_ROWS","2000"))
con = duckdb.connect(DB)

# 1) Pull features
q = f"""
WITH d AS (
  SELECT DATE_TRUNC('day', event_ts)::DATE AS d, provider_id,
         COUNT(*) AS claims_cnt,
         AVG(allowed_amt) AS avg_allowed,
         STDDEV_SAMP(allowed_amt) AS std_allowed
  FROM claims_events   
  GROUP BY 1,2

),
w AS (
  SELECT d, provider_id, claims_cnt, avg_allowed, std_allowed,
         AVG(claims_cnt) OVER (PARTITION BY provider_id ORDER BY d
             ROWS BETWEEN {W} PRECEDING AND CURRENT ROW) AS ma_claims,
         AVG(avg_allowed) OVER (PARTITION BY provider_id ORDER BY d
             ROWS BETWEEN {W} PRECEDING AND CURRENT ROW) AS ma_allowed
  FROM d
)
SELECT * FROM w ORDER BY d, provider_id;
"""
df = con.sql(q).df()
if len(df) < MIN_ROWS:
    print("Not enough rows yet. Populate more Phase-1 data."); raise SystemExit(0)

# 2) Train unsupervised model (simple, reproducible)
feat = df[["claims_cnt","avg_allowed","std_allowed","ma_claims","ma_allowed"]].fillna(0)
iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
df["score"] = iso.fit_predict(feat)  # -1 = anomaly
df["anomaly"] = (df["score"] == -1)

# 3) Save artifacts
# os.makedirs("artifacts", exist_ok=True)
df.to_parquet(os.path.join(ARTIFACTS_DIR, "anomalies.parquet"))
df.loc[df["anomaly"]].to_csv(os.path.join(ARTIFACTS_DIR, "anomaly_examples.csv"), index=False)

# 4) Tiny viz for LinkedIn
top = (df.groupby("d")["anomaly"].mean()*100).reset_index(name="anomaly_rate_pct")
fig = px.line(top, x="d", y="anomaly_rate_pct", title="Daily Anomaly Rate (%)")
fig.write_html(os.path.join(ARTIFACTS_DIR, "anomalies_trend.html"), include_plotlyjs="cdn")
print("Wrote artifacts/anomalies.parquet, anomaly_examples.csv, anomaly_trend.html")

# 5) Local Outlier Factor (LOF) comparison - Train the unsupervised model using LocalOutlierFactor. 

feat = df[["claims_cnt","avg_allowed","std_allowed","ma_claims","ma_allowed"]].fillna(0)

try:
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.02, novelty=False)
    lof_labels = lof.fit_predict(feat)            # -1 = anomaly, 1 = normal
    df["anomaly_lof"] = (lof_labels == -1)

    # Save LOF artifacts
    df.to_parquet(os.path.join(ARTIFACTS_DIR, "anomalies_lof.parquet"))
    df.loc[df["anomaly_lof"]].to_csv(os.path.join(ARTIFACTS_DIR, "anomaly_examples_lof.csv"), index=False)

    # Viz
    top_lof = (df.groupby("d")["anomaly_lof"].mean()*100).reset_index(name="anomaly_rate_pct")
    fig = px.line(top_lof, x="d", y="anomaly_rate_pct", title="Daily Anomaly Rate (%) — LOF")
    fig.write_html(os.path.join(ARTIFACTS_DIR, "anomaly_trend_lof.html"), include_plotlyjs="cdn")

    print("Wrote artifacts/anomalies_lof.parquet, anomaly_examples_lof.csv, anomaly_trend_lof.html")
except Exception as e:
    print(f"[yellow]LOF failed/skipped: {e}[/yellow]")

