🩺 Healthcare Anomaly Detection (Phase 2)

Objective: Compare two unsupervised learning models — Isolation Forest and Local Outlier Factor (LOF) — for anomaly detection on simulated healthcare claims data, using a local, cloud-free pipeline built with Python and DuckDB.

📘 Project Overview

This phase builds on Phase 1 – Healthcare Stream Simulation
, where streaming claims data were generated and stored in DuckDB.
Phase 2 introduces machine learning–based anomaly detection, helping uncover unusual provider behavior and claims patterns that could represent fraud, abuse, or operational anomalies.

🧠 Skills Demonstrated

Unsupervised Machine Learning – Isolation Forest & Local Outlier Factor comparison

Feature Engineering – aggregating claims per day and provider with rolling metrics

Data Visualization & Interpretation – Plotly dashboards comparing model outputs

Local Data Pipeline Design – DuckDB querying, ETL scripts, artifact management

Reproducible Experiment Setup – .env configuration and lightweight orchestration

⚙️ Quick Start
1️⃣ Clone and enter

git clone https://github.com/BTExpress1/healthcare-anomaly-phase2.git
cd healthcare-anomaly-phase2

2️⃣ Set up environment

python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # adjust paths if Phase 1 repo is elsewhere

3️⃣ Run analysis

python src/run.py

4️⃣ Compare models

python src/compare.py

Results are saved under:
/artifacts
 ├── anomalies.parquet
 ├── anomalies_lof.parquet
 ├── anomaly_trend.html
 ├── anomaly_trend_lof.html
 ├── compare_daily_rate.html
 ├── compare_top_providers.html
 └── compare_overlap_summary.csv

Open the .html files in your browser for interactive visuals.

📊 Key Findings

Isolation Forest isolates global outliers (rare, large-impact anomalies).

Local Outlier Factor highlights local density shifts (context-based anomalies).

Both overlap on ≈ X % of flagged provider-days, offering complementary risk signals.

🔗 Related Projects

Healthcare Stream Phase 1

🖼 Alt-Text (for LinkedIn visual)

Line chart comparing daily anomaly rates detected by Isolation Forest (blue) and Local Outlier Factor (red) on healthcare claims data from 2008 – 2011, illustrating how different algorithms capture unique risk patterns.

🏷 Hashtags

#DataScience #MachineLearning #AnomalyDetection #HealthcareAnalytics #Python #UnsupervisedLearning #AIInHealthcare #DataDrivenInsights
