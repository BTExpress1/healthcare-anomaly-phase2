import os, duckdb, numpy as np, pandas as pd
from dotenv import load_dotenv; load_dotenv()
from sklearn.ensemble import IsolationForest
import plotly.express as px

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
  FROM claims_events GROUP BY 1,2
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
os.makedirs("artifacts", exist_ok=True)
df.to_parquet("artifacts/anomalies.parquet")
df.loc[df["anomaly"]].to_csv("artifacts/anomaly_examples.csv", index=False)

# 4) Tiny viz for LinkedIn
top = (df.groupby("d")["anomaly"].mean()*100).reset_index(name="anomaly_rate_pct")
fig = px.line(top, x="d", y="anomaly_rate_pct", title="Daily Anomaly Rate (%)")
fig.write_html("artifacts/anomaly_trend.html", include_plotlyjs="cdn")
print("Wrote artifacts/anomalies.parquet, anomaly_examples.csv, anomaly_trend.html")
