"""Flask routes - pages and prediction API."""

import os
import numpy as np
from flask import Blueprint, render_template, request, jsonify, send_from_directory
import config as cfg
from src.utils import load_json, load_pkl, map_age_group

main_bp = Blueprint("main", __name__)

# Cache loaded model files so we only read from disk once
_model_cache = {}


def _summary():
    # Remove the caching check to ensure fresh data on every refresh
    return load_json(cfg.SUMMARY_PATH)


def _model(name):
    if name not in _model_cache:
        _model_cache[name] = load_pkl(os.path.join(cfg.MODEL_DIR, name))
    return _model_cache[name]


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
    """Accept form JSON, return fraud prediction using Pipeline models."""
    data = request.get_json(force=True)

    model_files = {
        "rf_baseline": "rf_baseline.pkl",
        "logistic":    "logistic_regression.pkl",
        "rf_improved": "rf_improved.pkl",
    }
    model_choice = data.get("model_choice", "rf_improved")
    # Load the full pipeline!
    model_pipeline = _model(model_files.get(model_choice, "rf_improved.pkl"))
    
    # Threshold handling
    threshold = 0.5
    if model_choice == "rf_improved":
        try:
            threshold = _model("optimal_threshold.pkl")
        except:
            threshold = 0.5

    # Process and Map Binary inputs
    row_dict, age = _extract_row(data)
    
    # Convert to DataFrame and enforce column order
    import pandas as pd
    df_single = pd.DataFrame([row_dict])
    df_single = df_single[cfg.ALL_FEATURE_COLS]
    
    # Prediction
    prob = model_pipeline.predict_proba(df_single)[0]
    pred = int(prob[1] >= threshold)

    return jsonify({
        "prediction":  pred,
        "probability": round(float(prob[1]) * 100, 1),
        "threshold":   round(threshold, 2),
        "label":       "Fraud" if pred else "Legitimate",
    })


def _extract_row(data):
    """Internal helper to convert JSON data into a feature-ready dictionary for the Pipeline."""
    age = int(data.get("age", 35))
    income = float(data.get("income", 150000))
    family_size = int(data.get("family_size", 2))
    years_emp = int(data.get("years_employed", 5))

    # Binary features mapping
    gender = cfg.BINARY_MAP.get(data.get("gender", "F"), 0)
    car = cfg.BINARY_MAP.get(data.get("car", "N"), 0)
    reality = cfg.BINARY_MAP.get(data.get("reality", "Y"), 1)

    # Communication channels count
    comm_score = (int(data.get("work_phone", 0)) +
                  int(data.get("phone", 0)) +
                  int(data.get("email", 0)))

    return {
        "GENDER":           gender,
        "CAR":              car,
        "REALITY":          reality,
        "NO_OF_CHILD":      int(data.get("no_children", 0)),
        "FAMILY_TYPE":      data.get("family_type", "Married"),
        "HOUSE_TYPE":       data.get("house_type", "House / apartment"),
        "WORK_PHONE":       int(data.get("work_phone", 0)),
        "PHONE":            int(data.get("phone", 0)),
        "E_MAIL":           int(data.get("email", 0)),
        "FAMILY SIZE":      family_size,
        "BEGIN_MONTH":      int(data.get("begin_month", 12)),
        "AGE":              age,
        "YEARS_EMPLOYED":   years_emp,
        "INCOME":           income,
        "INCOME_TYPE":      data.get("income_type", "Working"),
        "EDUCATION_TYPE":   data.get("education_type", "Higher education"),
        "INCOME_PER_MEMBER": income / max(family_size, 1),
        "AGE_GROUP":        map_age_group(age),
        "EMPLOYMENT_RATIO": years_emp / max(age, 1),
        "COMM_SCORE":       comm_score,
    }, age
