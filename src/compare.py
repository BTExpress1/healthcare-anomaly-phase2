import os
import numpy as np
import pandas as pd
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# ----- paths -----
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

load_dotenv(os.path.join(ROOT_DIR, ".env"))
DB = os.getenv("DUCKDB_PATH", os.path.join(ROOT_DIR, "../healthcare-stream-phase1/data/warehouse.duckdb"))
W = int(os.getenv("WINDOW_DAYS", "7"))

# ----- pull features from DuckDB (same contract as run.py) -----
con = duckdb.connect(DB)
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
df = con.sql(q).df().fillna(0)

# ----- read model outputs (full frames), keep only flagged rows -----
iso_full = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "anomalies.parquet"))
lof_full = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "anomalies_lof.parquet"))

iso = iso_full.loc[iso_full["anomaly"], ["d", "provider_id"]].assign(anomaly_iso=True)
lof = lof_full.loc[lof_full["anomaly_lof"], ["d", "provider_id"]].assign(anomaly_lof=True)

# ----- join back to features -----
m = (
    df.merge(iso, on=["d", "provider_id"], how="left")
      .merge(lof, on=["d", "provider_id"], how="left")
)
m["anomaly_iso"] = m["anomaly_iso"].fillna(False)
m["anomaly_lof"] = m["anomaly_lof"].fillna(False)

# ===== 1) Daily anomaly rate (ISO vs LOF) =====
daily = (
    m.groupby("d")[["anomaly_iso", "anomaly_lof"]].mean().mul(100).reset_index()
      .rename(columns={"anomaly_iso": "ISO %", "anomaly_lof": "LOF %"})
)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=daily["d"], y=daily["ISO %"], mode="lines", name="ISO"))
fig1.add_trace(go.Scatter(x=daily["d"], y=daily["LOF %"], mode="lines", name="LOF"))
fig1.update_layout(
    title="Daily Anomaly Rate: ISO vs LOF",
    xaxis_title="Date",
    yaxis_title="% of provider-days flagged"
)
fig1.write_html(os.path.join(ARTIFACTS_DIR, "compare_daily_rate.html"), include_plotlyjs="cdn")

# ===== 2) Overlap summary =====
both   = (m["anomaly_iso"] & m["anomaly_lof"]).sum()
iso_o  = (m["anomaly_iso"] & ~m["anomaly_lof"]).sum()
lof_o  = (~m["anomaly_iso"] & m["anomaly_lof"]).sum()
either = (m["anomaly_iso"] | m["anomaly_lof"]).sum()
jacc   = (both / either) if either else 0.0
pd.DataFrame({
    "metric": ["ISO only", "LOF only", "Both", "Either", "Jaccard overlap"],
    "value":  [iso_o,       lof_o,      both,  either,   round(jacc, 4)]
}).to_csv(os.path.join(ARTIFACTS_DIR, "compare_overlap_summary.csv"), index=False)

# ===== 3) Top providers by anomaly count (bar) =====
prov = (
    m.groupby("provider_id")[["anomaly_iso", "anomaly_lof"]].sum()
     .sort_values("anomaly_iso", ascending=False).head(15).reset_index()
)
figp = px.bar(
    prov, x="provider_id", y=["anomaly_iso", "anomaly_lof"],
    title="Top 15 Providers by Anomaly Count (ISO vs LOF)", barmode="group"
)
figp.write_html(os.path.join(ARTIFACTS_DIR, "compare_top_providers.html"), include_plotlyjs="cdn")

# ===== 4) Scatter (sampled) with jitter + log axes =====
sample = m.sample(min(8000, len(m)), random_state=42).copy()
sample["cc_jitter"] = sample["claims_cnt"] + np.random.uniform(-0.05, 0.05, len(sample))
sample["flag"] = np.select(
    [sample["anomaly_iso"] & sample["anomaly_lof"],
     sample["anomaly_iso"],
     sample["anomaly_lof"]],
    ["Both", "ISO only", "LOF only"],
    default="None"
)
fig2 = px.scatter(
    sample, x="cc_jitter", y="avg_allowed", color="flag",
    title="ISO vs LOF flags (sampled provider-days)",
    hover_data=["d", "provider_id"]
)
fig2.update_xaxes(title="claims_cnt (jittered)", type="log")
fig2.update_yaxes(title="avg_allowed (log)", type="log")
fig2.write_html(os.path.join(ARTIFACTS_DIR, "compare_scatter.html"), include_plotlyjs="cdn")

print("Wrote: compare_daily_rate.html, compare_overlap_summary.csv, compare_top_providers.html, compare_scatter.html")
