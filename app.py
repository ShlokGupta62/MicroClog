# backend_predict_model_final.py
"""
Final backend for MicroClog — copy/paste ready.

Endpoints:
 - GET /api/top_drains?n=5[&demo=1]  -> returns top N drains sorted by risk_score desc
 - GET /api/predict[?demo=1]         -> runs prediction, saves High work orders, returns {"work_orders_count","work_orders"}
 - GET /api/work_orders/latest       -> returns {"filename","data"} (stable latest file)
 - GET /work_orders.json             -> raw array (latest)
 - GET / (and static assets from same folder) -> serves your Admin Dashboard HTML
"""

import os, json, time, traceback
from pathlib import Path
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
from joblib import load
from flask import Flask, jsonify, send_from_directory, make_response, request
from flask_cors import CORS

# ------ CONFIG (edit only if paths differ) ------
DRAINS_CSV = Path(r"C:\Users\Ujan\Downloads\MicroClog_starter_1\MicroClog_starter\microclog\data\drains.csv")
FLOOD_HISTORY_CSV = Path(r"C:\Users\Ujan\Downloads\MicroClog_starter_1\MicroClog_starter\microclog\data\overflow_history.csv")
CLEANING_LOGS_CSV = Path(r"C:\Users\Ujan\Downloads\MicroClog_starter_1\MicroClog_starter\microclog\data\cleaning_logs.csv")

MODEL_PATH = Path(r"C:\Users\Ujan\Downloads\MicroClog_starter_1\MicroClog_starter\models\model_joblib.pkl")
RAINFALL_MODEL_PATH = Path(r"C:\Users\Ujan\Downloads\MicroClog_starter_1\MicroClog_starter\models\weather_models\kolkata.joblib")

WORK_ORDERS_DIR = Path(r"C:\Users\Ujan\Downloads\MicroClog_starter_1\MicroClog_starter\outputs\work_orders")
WORK_ORDERS_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_HTML = Path(r"C:\Users\Ujan\Downloads\MicroClog_starter_1\MicroClog_starter\frontend\Admin Dashboard.html")
CUSTOM_HTML_DIR = CUSTOM_HTML.parent

OPENWEATHER_KEY = "cea6d99f576eb5ba0f94a6b032fda0b3"
OPENWEATHER_CACHE = Path("cache/ow"); OPENWEATHER_CACHE.mkdir(parents=True, exist_ok=True)

RAIN_CAP = 150.0
HISTORY_CAP = 10.0
CLEANING_CAP_DAYS = 180.0
PAUSE_BETWEEN_CALLS = 1.0
KOLKATA_LAT, KOLKATA_LON = 22.5726, 88.3639

# thresholds used only to classify continuous scores (not used to pick top drains)
HIGH_THR = 0.55
MED_THR = 0.30

# If the model predicts no positives, fallback to returning this many top drains as High
FALLBACK_TOP_K_IF_NO_POSITIVES = 10

app = Flask(__name__, static_folder=None)
CORS(app)

def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def _tb():
    return traceback.format_exc()

# Weather helpers (cached)
def cache_path_for_point(lat, lon, ref_date):
    safe = f"{lat:.4f}_{lon:.4f}_{ref_date}".replace(".", "_")
    return OPENWEATHER_CACHE / (safe + ".json")

def fetch_openweather_forecast(lat, lon, force_refresh=False):
    if not OPENWEATHER_KEY:
        return None
    ref_date = datetime.utcnow().date().isoformat()
    cache_file = cache_path_for_point(lat, lon, ref_date)
    if cache_file.exists() and not force_refresh:
        try:
            return json.load(open(cache_file, "r", encoding="utf-8"))
        except Exception:
            pass
    try:
        url = ("https://api.openweathermap.org/data/2.5/forecast"
               f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric")
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        json.dump(data, open(cache_file, "w", encoding="utf-8"), indent=2)
        time.sleep(PAUSE_BETWEEN_CALLS)
        return data
    except Exception as e:
        print("[warn] OpenWeather fetch failed:", e)
        return None

def max_next24_from_forecast_json(forecast_json):
    if not forecast_json:
        return 0.0
    now = datetime.utcnow()
    cutoff = now + pd.Timedelta(hours=24)
    vals = []
    for item in forecast_json.get("list", []):
        t = pd.to_datetime(item.get("dt_txt", None), errors="coerce")
        if pd.isna(t): continue
        if now < t <= cutoff:
            vals.append(float(item.get("rain", {}).get("3h", 0.0)))
    return max(vals) if vals else 0.0

# Rainfall prediction using your Prophet model
def predict_rainfall_with_prophet(lat, lon):
    """Predict rainfall using your trained Prophet model for the nearest station."""
    try:
        if not RAINFALL_MODEL_PATH.exists():
            print("[warn] Rainfall model not found:", RAINFALL_MODEL_PATH)
            return None, "rainfall_model_not_found"
        
        # Load the rainfall model
        rainfall_model_data = load(RAINFALL_MODEL_PATH)
        prophet_model = rainfall_model_data['model']
        
        # Make prediction for next 24 hours
        future = prophet_model.make_future_dataframe(periods=2, freq='D')
        forecast = prophet_model.predict(future)
        
        # Get tomorrow's forecast
        tomorrow = pd.Timestamp.now(tz='UTC').normalize() + pd.Timedelta(days=1)
        tomorrow_forecast = forecast[forecast['ds'] == tomorrow]
        
        if not tomorrow_forecast.empty:
            rainfall_mm = float(tomorrow_forecast['yhat'].iloc[0])
            return max(0.0, rainfall_mm), "prophet_model"
        else:
            # Fallback: use the last prediction
            rainfall_mm = float(forecast.tail(1)['yhat'].iloc[0])
            return max(0.0, rainfall_mm), "prophet_model_fallback"
            
    except Exception as e:
        print("[warn] Prophet rainfall prediction failed:", e)
        return None, f"prophet_error: {str(e)}"

# ---------- data building ----------
def build_master_df():
    for p in (DRAINS_CSV, FLOOD_HISTORY_CSV, CLEANING_LOGS_CSV):
        if not p.exists():
            raise FileNotFoundError(f"Missing CSV: {p}")
    drains = pd.read_csv(DRAINS_CSV, dtype=str)
    fh = pd.read_csv(FLOOD_HISTORY_CSV, dtype=str)
    logs = pd.read_csv(CLEANING_LOGS_CSV, dtype=str)

    fh['severity'] = pd.to_numeric(fh.get('severity', 1), errors='coerce').fillna(1.0)
    flood_counts = fh.groupby('drain_id')['severity'].sum().reset_index().rename(columns={'severity':'flood_count'})

    logs['date'] = pd.to_datetime(logs['date'], errors='coerce')
    recent_clean = logs.groupby('drain_id')['date'].max().reset_index().rename(columns={'date':'last_cleaned_log'})

    df = drains.merge(flood_counts, on='drain_id', how='left').merge(recent_clean, on='drain_id', how='left')
    df['flood_count'] = df['flood_count'].fillna(0).astype(float)

    if 'last_cleaned_date' in df.columns:
        df['last_cleaned'] = df['last_cleaned_date'].fillna(df['last_cleaned_log'])
    else:
        df['last_cleaned'] = df['last_cleaned_log']
    df['last_cleaned'] = pd.to_datetime(df['last_cleaned'], errors='coerce')
    df['days_since_clean'] = (pd.Timestamp.now() - df['last_cleaned']).dt.days
    df['days_since_clean'] = df['days_since_clean'].fillna(9999).astype(float)

    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
    return df

def build_features(df, use_weather=True, force_rainfall=None):
    # Use Prophet model for rainfall prediction instead of OpenWeather
    prophet_rainfall, prophet_source = predict_rainfall_with_prophet(KOLKATA_LAT, KOLKATA_LON)
    
    rows = []
    for _, row in df.iterrows():
        drain_id = row['drain_id']
        ward_id = row.get('ward_id')
        lat = None if pd.isna(row['lat']) else float(row['lat'])
        lon = None if pd.isna(row['lon']) else float(row['lon'])
        flood_count = float(row['flood_count'])
        days_since_clean = None if pd.isna(row['days_since_clean']) else float(row['days_since_clean'])

        if force_rainfall is not None:
            rainfall_mm = float(force_rainfall); source = "DEMO_FORCE"
        elif prophet_rainfall is not None:
            rainfall_mm = prophet_rainfall; source = prophet_source
        else:
            # Fallback to OpenWeather if Prophet fails
            forecast = None
            if use_weather and (lat is not None and lon is not None):
                forecast = fetch_openweather_forecast(lat, lon)
            if forecast:
                rainfall_mm = max_next24_from_forecast_json(forecast); source = "OpenWeather_point"
            else:
                rainfall_mm = 0.0; source = "fallback_zero"

        rain_index = min(rainfall_mm / RAIN_CAP, 1.0)
        history_index = min(flood_count / HISTORY_CAP, 1.0)
        cleaning_index = 1.0 if days_since_clean is None else min(days_since_clean / CLEANING_CAP_DAYS, 1.0)

        rows.append({
            "drain_id": drain_id,
            "ward_id": ward_id,
            "lat": lat,
            "lon": lon,
            "rainfall_24h_mm": round(float(rainfall_mm),4),
            "rain_source": source,
            "flood_count": float(flood_count),
            "days_since_clean": None if days_since_clean is None else int(days_since_clean),
            "rain_index": round(rain_index,4),
            "history_index": round(history_index,4),
            "cleaning_index": round(cleaning_index,4)
        })
    enriched = pd.DataFrame(rows)
    X = enriched[["rain_index","history_index","cleaning_index"]].fillna(0.0)
    return X, enriched

# ---------- model utils ----------
def load_model_safe():
    if not MODEL_PATH.exists():
        print("[warn] model file not found:", MODEL_PATH)
        return None
    try:
        m = load(MODEL_PATH)
        print("[info] loaded model:", MODEL_PATH)
        return m
    except Exception as e:
        print("[error] failed to load model:", e)
        return None

def model_scores_to_01(model, X):
    """
    Return score array in [0,1] for each row.
    Tries predict_proba positive class; if not available uses predict (0/1 or continuous).
    Continuous outputs are min-max normalized to [0,1].
    """
    if model is None:
        return None
    # predict_proba preferred
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            # find positive class index
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                pos_idx = None
                if 1 in classes:
                    pos_idx = classes.index(1)
                elif "1" in classes:
                    pos_idx = classes.index("1")
                else:
                    pos_idx = int(np.argmax(probs.mean(axis=0)))
            else:
                pos_idx = 1 if probs.shape[1] > 1 else 0
            scores = probs[:, pos_idx].astype(float)
            scores = np.clip(scores, 0.0, 1.0)
            return scores
        except Exception as e:
            print("[warn] predict_proba failed:", e)
    # fallback predict
    try:
        preds = model.predict(X)
        preds = np.array(preds, dtype=float)
        uniques = np.unique(preds)
        if set(uniques).issubset({0.0,1.0}):
            return preds
        # continuous -> normalize
        mn, mx = preds.min(), preds.max()
        if mx == mn:
            # if constant but >0 treat as 1 else 0
            return np.ones_like(preds) if mn > 0 else np.zeros_like(preds)
        norm = (preds - mn) / (mx - mn)
        return norm
    except Exception as e:
        print("[warn] model.predict failed:", e)
        return None

def save_workorders(work_orders, prefix="work_orders"):
    WORK_ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    tsfile = WORK_ORDERS_DIR / f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    latest = WORK_ORDERS_DIR / "work_orders_latest.json"
    try:
        with open(tsfile, "w", encoding="utf-8") as fh:
            json.dump(work_orders, fh, indent=2)
        with open(latest, "w", encoding="utf-8") as fh:
            json.dump(work_orders, fh, indent=2)
    except Exception as e:
        print("[error] saving workorders failed:", e)

# ---------- main pipeline ----------
def produce_predictions_and_workorders(use_weather=True, force_rainfall=None, fallback_top_k=FALLBACK_TOP_K_IF_NO_POSITIVES):
    df = build_master_df()
    X, enriched = build_features(df, use_weather=use_weather, force_rainfall=force_rainfall)

    model = load_model_safe()
    scores = None
    if model is not None:
        scores = model_scores_to_01(model, X)

    if scores is None:
        # heuristic score same as earlier pipeline
        enriched["risk_score"] = enriched.apply(lambda r: 0.5 * r["rain_index"] + 0.3 * r["history_index"] + 0.2 * r["cleaning_index"], axis=1).astype(float)
    else:
        enriched["risk_score"] = scores.astype(float)

    # If model produced no positives, ensure there are top items by picking top-K
    positive_count = int((enriched["risk_score"] > HIGH_THR).sum())
    if positive_count == 0 and len(enriched) > 0:
        k = min(fallback_top_k, len(enriched))
        enriched = enriched.sort_values("risk_score", ascending=False).reset_index(drop=True)
        # bump top-k to above HIGH_THR if necessary
        for i in range(k):
            if enriched.at[i, "risk_score"] <= HIGH_THR:
                enriched.at[i, "risk_score"] = float(HIGH_THR + 0.01)

    # map to risk_level for convenience
    enriched["risk_level"] = enriched["risk_score"].apply(lambda v: "High" if v > HIGH_THR else ("Medium" if v > MED_THR else "Low"))
    # save CSV summary (optional)
    try:
        enriched.to_csv("auto_predictions_hybrid.csv", index=False)
    except Exception:
        pass

    # Build work orders only for High
    high_df = enriched[enriched["risk_level"] == "High"]
    work_orders = []
    ts = int(time.time())
    for _, r in high_df.iterrows():
        work_orders.append({
            "work_order_id": f"WO_{r['drain_id']}_{ts}",
            "drain_id": r["drain_id"],
            "ward_id": r.get("ward_id"),
            "lat": None if pd.isna(r["lat"]) else float(r["lat"]),
            "lon": None if pd.isna(r["lon"]) else float(r["lon"]),
            "status": "Pending",
            "created_at": now_utc_iso(),
            "risk_score": float(r["risk_score"])
        })

    save_workorders(work_orders)
    return enriched, work_orders

# ---------- endpoints ----------
@app.route("/api/top_drains", methods=["GET"])
def api_top_drains():
    """
    Query: /api/top_drains?n=5[&demo=1]
    Returns: {"top": [ {drain_id, ward_id, lat, lon, flood_count, days_since_clean, rainfall_24h_mm, risk_score, risk_level}, ... ]}
    Sorted: High -> Medium -> Low, within group by score desc OR simply by score desc if prefer.
    """
    try:
        n = int(request.args.get("n", 5))
        demo = request.args.get("demo", "0") in ("1","true","True")
        force_rain = 120.0 if demo else None
        enriched, _ = produce_predictions_and_workorders(use_weather=True, force_rainfall=force_rain)
        if enriched is None or len(enriched) == 0:
            return jsonify({"top": []})
        # prioritize High first, then score desc
        order_map = {"High": 0, "Medium": 1, "Low": 2}
        enriched["order_group"] = enriched["risk_level"].map(order_map).fillna(3)
        enriched_sorted = enriched.sort_values(["order_group", "risk_score"], ascending=[True, False])
        top = enriched_sorted.head(n)
        out = []
        for _, r in top.iterrows():
            out.append({
                "drain_id": r.get("drain_id"),
                "ward_id": r.get("ward_id"),
                "lat": None if pd.isna(r.get("lat")) else float(r.get("lat")),
                "lon": None if pd.isna(r.get("lon")) else float(r.get("lon")),
                "flood_count": int(r.get("flood_count", 0)),
                "days_since_clean": None if pd.isna(r.get("days_since_clean")) else int(r.get("days_since_clean")),
                "rainfall_24h_mm": float(r.get("rainfall_24h_mm", 0.0)),
                "risk_score": float(r.get("risk_score", 0.0)),
                "risk_level": r.get("risk_level")
            })
        return jsonify({"top": out})
    except Exception as e:
        print("[error] /api/top_drains:", e)
        print(_tb())
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["GET"])
def api_predict():
    demo = request.args.get("demo", "0") in ("1","true","True")
    try:
        _, work_orders = produce_predictions_and_workorders(use_weather=True, force_rainfall=(120.0 if demo else None))
        return jsonify({"work_orders_count": len(work_orders), "work_orders": work_orders})
    except Exception as e:
        print("[error] /api/predict failed:", e)
        print(_tb())
        return jsonify({"error": str(e)}), 500

@app.route("/api/work_orders/latest", methods=["GET"])
def api_workorders_latest():
    latest = WORK_ORDERS_DIR / "work_orders_latest.json"
    if latest.exists():
        with open(latest, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return jsonify({"filename": latest.name, "data": data})
    files = sorted(WORK_ORDERS_DIR.glob("work_orders_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return jsonify({"error":"no work_orders"}), 404
    with open(files[0], "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return jsonify({"filename": files[0].name, "data": data})

@app.route("/work_orders.json", methods=["GET"])
def workorders_json_alias():
    latest = WORK_ORDERS_DIR / "work_orders_latest.json"
    if latest.exists():
        with open(latest, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return jsonify(data)
    files = sorted(WORK_ORDERS_DIR.glob("work_orders_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return jsonify([]), 404
    with open(files[0], "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return jsonify(data)

@app.route("/api/work_orders/list", methods=["GET"])
def api_workorders_list():
    files = sorted(WORK_ORDERS_DIR.glob("work_orders_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonify([{"filename": f.name, "mtime": f.stat().st_mtime} for f in files])

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path:
        candidate = CUSTOM_HTML_DIR / path
        if candidate.exists() and candidate.is_file():
            return send_from_directory(str(CUSTOM_HTML_DIR), path)
    if CUSTOM_HTML.exists():
        return send_from_directory(str(CUSTOM_HTML_DIR), CUSTOM_HTML.name)
    return make_response("<h3>Admin Dashboard not found. Update CUSTOM_HTML path.</h3>", 404)

if __name__ == "__main__":
    print("[info] Starting backend_predict_model_final")
    print("MODEL_PATH:", MODEL_PATH)
    print("RAINFALL_MODEL_PATH:", RAINFALL_MODEL_PATH)
    app.run(host="127.0.0.1", port=5000, debug=True)