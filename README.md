
# MicroClog — Predicting Drain Clog Hotspots (Starter Kit)

This repository is a **self-contained starter** that lets you **train**, **score**, and **visualize** clog risk for drains using synthetic data. 
It includes:
- Tabular ML pipeline (Gradient Boosting) for hotspot prediction
- Classical CV pipeline using scikit-image features (HOG) for photo-based "clog evidence" classification
- Risk fusion + NDRI computation
- Minimal digital twin simulation (NetworkX)
- Flask API for programmatic access + automated work orders with a hashed log (blockchain-like ledger)
- Streamlit dashboard for risk maps & analytics
- Sample synthetic dataset to get started immediately

> Replace synthetic data under `data/` with real municipal datasets as you acquire them.

## Quickstart

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Regenerate synthetic data
python scripts/generate_sample_data.py

# 4) Train the tabular model
python ml/train_tabular.py

# 5) Train the (optional) image model (needs images under data/images/{clog,clean}) 
python ml/train_cv.py  # safe to skip if you don't have images yet

# 6) Produce risk scores & NDRI
python risk_engine/fuse_and_score.py

# 7) Run the API
python api/app.py

# 8) Run the dashboard
streamlit run dashboard/streamlit_app.py
```

## Files & Folders

- `data/` — CSVs for drains, cleaning logs, rainfall forecast, citizen reports, overflow history.
- `ml/train_tabular.py` — trains GradientBoosting model, saves `models/tabular_model.pkl`.
- `ml/train_cv.py` — classic-vision (HOG+LogReg) pipeline; saves `models/cv_model.pkl` if images available.
- `risk_engine/fuse_and_score.py` — fuses latest data to compute risk for each drain; writes `reports/risk_scores.csv` and ward-level `reports/ndri.csv`.
- `risk_engine/digital_twin.py` — simple graph-based overflow simulation for what-if analysis.
- `api/app.py` — Flask endpoints for scores, citizen reports, and auto-generated work orders with a tamper-evident ledger (`api/chain.json`).
- `dashboard/streamlit_app.py` — interactive UI for city planners.
- `scripts/generate_sample_data.py` — re-creates a realistic synthetic dataset.

## Swapping in Real Data

- Replace `data/drains.csv` with columns at least: `drain_id, lat, lon, ward_id, last_cleaned_date`.
- Replace `data/cleaning_logs.csv` with: `drain_id, date, method, crew_id, duration_min`.
- Replace `data/rainfall_forecast.csv` with: `ward_id, date, rainfall_mm`.
- Replace `data/citizen_reports.csv` with: `report_id, date, lat, lon, ward_id, drain_id(optional), issue_type, text, image_path`.
- Replace `data/overflow_history.csv` with: `drain_id, date, severity`.

## License
Apache-2.0 for easy adoption.
