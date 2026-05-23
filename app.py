"""Flask backend for Habiba's Heart Attack Risk Predictor.

Loads the leak-safe scikit-learn pipeline (imputer + scaler + model live
inside one saved file), the ordered feature list, the tuned probability
threshold, and a metadata JSON with encoder maps and UI defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request


ROOT = Path(__file__).resolve().parent

app = Flask(__name__)

# ---- Load saved artefacts (produced by the notebook) ----------------------
model = joblib.load(ROOT / "heart_model.pkl")
feature_columns = joblib.load(ROOT / "heart_features.pkl")
threshold = float(joblib.load(ROOT / "heart_threshold.pkl"))

metadata_path = ROOT / "heart_metadata.json"
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    encoders = metadata.get("encoders", {})
    AGE_MAP = encoders.get("AGE_MAP", {})
    GEN_MAP = encoders.get("GEN_MAP", {})
    SMOKE_MAP = encoders.get("SMOKE_MAP", {})
    DIABETES_MAP = encoders.get("DIABETES_MAP", {})
else:
    # Fallback: same encoders the notebook uses, in case metadata.json is missing.
    AGE_MAP = {age: i for i, age in enumerate([
        "Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39",
        "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59",
        "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79",
        "Age 80 or older",
    ])}
    GEN_MAP = {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
    SMOKE_MAP = {
        "Never smoked": 0,
        "Former smoker": 1,
        "Current smoker - now smokes some days": 2,
        "Current smoker - now smokes every day": 3,
    }
    DIABETES_MAP = {
        "No": 0,
        "No, pre-diabetes or borderline diabetes": 1,
        "Yes, but only during pregnancy (female)": 1,
        "Yes": 2,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # The HTML form may send numeric flags as strings ('1' / '0'); coerce
        # everything to numbers because the saved pipeline was trained on ints.
        row = {
            "Sex": 1 if str(data["sex"]).strip() == "Male" else 0,
            "AgeCategory": int(AGE_MAP[data["age_category"]]),
            "GeneralHealth": int(GEN_MAP[data["general_health"]]),
            "SmokerStatus": int(SMOKE_MAP[data["smoker_status"]]),
            "HadDiabetes": _coerce_diabetes(data["had_diabetes"]),
            "PhysicalHealthDays": float(data["physical_health_days"]),
            "MentalHealthDays": float(data["mental_health_days"]),
            "SleepHours": float(data["sleep_hours"]),
            "BMI": float(data["bmi"]),
            "PhysicalActivities": _flag(data["physical_activities"]),
            "AlcoholDrinkers": _flag(data["alcohol_drinkers"]),
            "HadAngina": _flag(data["had_angina"]),
            "HadStroke": _flag(data["had_stroke"]),
            "HadCOPD": _flag(data["had_copd"]),
            "HadKidneyDisease": _flag(data["had_kidney_disease"]),
            "HadArthritis": _flag(data["had_arthritis"]),
            "DifficultyWalking": _flag(data["difficulty_walking"]),
            "ChestScan": _flag(data["chest_scan"]),
        }

        df = pd.DataFrame([row])[feature_columns]
        proba = float(model.predict_proba(df)[0, 1])

        prediction = int(proba >= threshold)
        risk_level = "high" if prediction == 1 else "low"

        return jsonify({
            "success": True,
            "probability": round(proba * 100, 2),
            "prediction": prediction,
            "risk": risk_level,
            "threshold": round(threshold * 100, 2),
        })

    except Exception as exc:  # noqa: BLE001 - surface the message to the UI
        return jsonify({"success": False, "error": str(exc)}), 400


def _flag(value) -> int:
    """Accept 1/0, '1'/'0', True/False, 'Yes'/'No' from the form."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "yes", "true"}:
        return 1
    return 0


def _coerce_diabetes(value) -> int:
    """The form sends a 0/1 flag for had_diabetes; map to the model's 0/1/2 scale."""
    if isinstance(value, str) and value in DIABETES_MAP:
        return int(DIABETES_MAP[value])
    return 2 if _flag(value) == 1 else 0


if __name__ == "__main__":
    app.run(debug=True, port=5000)
