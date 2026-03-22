"""Flask routes - pages and prediction API."""

import os
import numpy as np
from flask import Blueprint, render_template, request, jsonify, send_from_directory
import config as cfg
from src.utils import load_json, load_pkl

main_bp = Blueprint("main", __name__)

# cache loaded models and data so we only read from disk once
_cache = {}


def _summary():
    if "summary" not in _cache:
        _cache["summary"] = load_json(cfg.SUMMARY_PATH)
    return _cache["summary"]


def _model(name):
    if name not in _cache:
        _cache[name] = load_pkl(os.path.join(cfg.MODEL_DIR, name))
    return _cache[name]


@main_bp.route("/plots/<path:filename>")
def serve_plot(filename):
    return send_from_directory(cfg.PLOT_DIR, filename)


# -- Pages --

@main_bp.route("/")
def overview():
    s = _summary()
    return render_template("overview.html", ds=s["dataset"], active="overview")


@main_bp.route("/eda")
def eda():
    s = _summary()
    return render_template("eda.html",
                           plots=s["eda_plots"],
                           model_plots=s.get("plots", []),
                           active="eda")


@main_bp.route("/prediction")
def prediction():
    s = _summary()
    return render_template("prediction.html",
                           models_info=s["models"],
                           improvement=s["improvement"],
                           active="prediction")


# -- Prediction API --

@main_bp.route("/predict", methods=["POST"])
def predict():
    """Accept form JSON, return fraud prediction."""
    data = request.get_json(force=True)

    model_files = {
        "rf_baseline": "rf_baseline.pkl",
        "logistic":    "logistic_regression.pkl",
        "rf_improved": "rf_improved.pkl",
    }
    model_file = model_files.get(data.get("model_choice"), "rf_improved.pkl")
    model = _model(model_file)
    scaler = _model("scaler.pkl")
    encoders = _model("label_encoders.pkl")
    feature_names = _model("feature_names.pkl")

    # threshold only applies to the improved model
    threshold = 0.5
    if data.get("model_choice") == "rf_improved":
        try:
            threshold = _model("optimal_threshold.pkl")
        except Exception:
            threshold = 0.5

    # build feature vector
    binary = {"M": 1, "F": 0, "Y": 1, "N": 0}
    age = int(data.get("age", 35))
    family_size = int(data.get("family_size", 2))
    income = float(data.get("income", 150000))
    years_emp = int(data.get("years_employed", 5))

    row = {
        "GENDER":           binary[data.get("gender", "M")],
        "CAR":              binary[data.get("car", "N")],
        "REALITY":          binary[data.get("reality", "Y")],
        "NO_OF_CHILD":      int(data.get("no_children", 0)),
        "FAMILY_TYPE":      0,
        "HOUSE_TYPE":       0,
        "WORK_PHONE":       int(data.get("work_phone", 0)),
        "PHONE":            int(data.get("phone", 0)),
        "E_MAIL":           int(data.get("email", 0)),
        "FAMILY SIZE":      family_size,
        "BEGIN_MONTH":      int(data.get("begin_month", 12)),
        "AGE":              age,
        "YEARS_EMPLOYED":   years_emp,
        "INCOME":           income,
        "INCOME_TYPE":      0,
        "EDUCATION_TYPE":   0,
        "INCOME_PER_MEMBER": income / max(family_size, 1),
        "AGE_GROUP":        0,
        "EMPLOYMENT_RATIO": years_emp / max(age, 1),
        "COMM_SCORE":       (int(data.get("work_phone", 0))
                             + int(data.get("phone", 0))
                             + int(data.get("email", 0))),
    }

    # encode categorical fields
    cat_vals = {
        "FAMILY_TYPE":    data.get("family_type", "Married"),
        "HOUSE_TYPE":     data.get("house_type", "House / apartment"),
        "INCOME_TYPE":    data.get("income_type", "Working"),
        "EDUCATION_TYPE": data.get("education_type", "Secondary / secondary special"),
    }
    for col, val in cat_vals.items():
        le = encoders.get(col)
        if le and val in le.classes_:
            row[col] = int(le.transform([val])[0])

    # encode age group
    age_bins = [(25, "Young"), (35, "Adult"), (45, "Middle"),
                (55, "Senior"), (999, "Elderly")]
    ag = next(lbl for cut, lbl in age_bins if age <= cut)
    le_ag = encoders.get("AGE_GROUP")
    if le_ag and ag in le_ag.classes_:
        row["AGE_GROUP"] = int(le_ag.transform([ag])[0])

    # align to model's expected feature order and scale
    vec = np.array([row.get(f, 0) for f in feature_names]).reshape(1, -1)
    vec_scaled = scaler.transform(vec)

    prob = model.predict_proba(vec_scaled)[0]
    pred = int(prob[1] >= threshold)

    return jsonify({
        "prediction":  pred,
        "probability": round(float(prob[1]) * 100, 1),
        "threshold":   round(threshold, 2),
        "label":       "Fraud" if pred else "Legitimate",
    })
