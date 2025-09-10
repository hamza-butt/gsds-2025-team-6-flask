import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix

# ---------- Training-time helper function to combine title and description -----
def combine_title_desc(df):
    t = df.get("Title")
    d = df.get("Description")
    if t is None or d is None:
        t = df["Title"] if "Title" in df else pd.Series([""] * len(df))
        d = df["Description"] if "Description" in df else pd.Series([""] * len(df))
    return (t.fillna("").astype(str) + " " + d.fillna("").astype(str))

# ---------- Configuration ----------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "price_model.joblib") 
TITLE_COL = "Title"
DESC_COL  = "Description"
CAT_COL   = "Category"
COND_COL  = "Condition"

# ---------- Initialize Flask app --------------------------------------------
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  

try:
    model = joblib.load(MODEL_PATH) 
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}") 

# ---------- Helper Functions -----------------------------------------------
def to_row(payload: dict) -> dict:
    return {
        TITLE_COL: payload.get("Title", "") or "", 
        DESC_COL:  payload.get("Description", "") or "", 
        CAT_COL:   payload.get("Category", "") or "", 
        COND_COL:  payload.get("Condition", "") or "", 
    }

# ---------- Flask Route ------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=False) 
        if not isinstance(data, dict):
            return jsonify({"message": "Body must be a JSON object."}), 400 
        
        row = to_row(data)
        df = pd.DataFrame([row]) 
        y = model.predict(df)[0]  
        return jsonify({"predicted_price": y}), 200 

    except Exception as e:
        return jsonify({"message": f"Inference failed: {str(e)}"}), 400

# ---------- Start the Flask Application -------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
