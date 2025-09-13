import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS  # <-- NEW

# ---------- Training-time helper to combine title & description --------------
def combine_title_desc(df: pd.DataFrame):
    t = df.get("Title")
    d = df.get("Description")
    if t is None or d is None:
        t = df["Title"] if "Title" in df else pd.Series([""] * len(df))
        d = df["Description"] if "Description" in df else pd.Series([""] * len(df))
    return (t.fillna("").astype(str) + " " + d.fillna("").astype(str))

# ---------- Configuration ----------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "price_model.joblib")
TITLE_COL  = "Title"
DESC_COL   = "Description"
CAT_COL    = "Category"
COND_COL   = "Condition"

# ---------- Initialize Flask app --------------------------------------------
app = Flask(__name__)
# If behind a proxy (Render/Heroku), keep correct scheme/host for any redirects
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Enable CORS for all routes (adjust origins as needed)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},  # put your frontend origin instead of "*", if you want
    supports_credentials=False,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ---------- Load model -------------------------------------------------------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

# ---------- Helpers ----------------------------------------------------------
def to_row(payload: dict) -> dict:
    return {
        TITLE_COL: payload.get("Title", "") or "",
        DESC_COL:  payload.get("Description", "") or "",
        CAT_COL:   payload.get("Category", "") or "",
        COND_COL:  payload.get("Condition", "") or "",
    }

# ---------- Healthcheck ------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

# ---------- Predict ----------------------------------------------------------
@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Preflight will be auto-handled by flask-cors headers; return early
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        data = request.get_json(force=True, silent=False)
        if not isinstance(data, dict):
            return jsonify({"message": "Body must be a JSON object."}), 400

        row = to_row(data)
        df = pd.DataFrame([row])

        # If your pipeline uses combine_title_desc, ensure the function name is importable
        y = model.predict(df)[0]
        # Make sure it’s JSON serializable
        y_num = float(np.asarray(y).item())

        return jsonify({"predicted_price": y_num}), 200

    except Exception as e:
        return jsonify({"message": f"Inference failed: {str(e)}"}), 400

# ---------- Entrypoint -------------------------------------------------------
if __name__ == "__main__":
    # For local dev only; production should use gunicorn
    app.run(host="0.0.0.0", port=8000, debug=True)
