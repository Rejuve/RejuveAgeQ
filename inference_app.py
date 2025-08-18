import logging
import pandas as pd
from flask import Flask, request, jsonify
from autogluon.tabular import TabularPredictor

# --- App & logging ---
app = Flask(__name__)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# --- Load predictor once (at startup) ---
# Point to your trained predictor directory (where predictor.pkl / models/ live)
PREDICTOR_PATH = "checkpoint/survey_model"  
PREDICTOR = TabularPredictor.load(PREDICTOR_PATH)     # heavy load happens once

def _to_dataframe(payload) -> pd.DataFrame:
    """Accepts dict (single sample) or list[dict] (batch) and returns a DataFrame."""
    if isinstance(payload, dict):
        return pd.DataFrame([payload])
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return pd.DataFrame(payload)
    raise ValueError("Expecting a JSON object or a list of JSON objects.")

@app.route("/predict", methods=["POST"])
def predict_route():
    logging.info("Started processing request.")
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "code": 400,
            "biological_age": None,
            "message": "Invalid input: provide JSON object or list of objects."
        }), 400

    try:
        df = _to_dataframe(data)
        df = df.reindex(columns=PREDICTOR.features())
        # If your model is a regressor:
        y_pred = PREDICTOR.predict(df)             # pd.Series
        # If you need prediction intervals (optional):
        # y_pred, y_std = PREDICTOR.predict(df, as_pandas=True, return_std=True)

        # Single vs batch response
        if len(y_pred) == 1:
            result = {
                "code": 200,
                "biological_age": float(y_pred.iloc[0]),
                "message": "ok",
            }
        else:
            result = {
                "code": 200,
                "biological_age": [float(v) for v in y_pred.to_list()],
                "message": "ok",
                "n": len(y_pred),
            }

        logging.info("Finished processing request.")
        return jsonify(result), 200

    except Exception as e:
        logging.exception("Prediction failed.")
        return jsonify({
            "code": 500,
            "biological_age": None,
            "message": f"Prediction error: {type(e).__name__}: {e}"
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    # Optional: warm the model on first /health call by touching PREDICTOR if you lazy-load it
    return jsonify(status="Flask App is running, all good :) "), 200

if __name__ == "__main__":
    # For multi-process servers like gunicorn, consider --preload to load once then fork
    # gunicorn app:app -w 2 -b 0.0.0.0:5001 --preload
    app.run(host="0.0.0.0", port=5001)
