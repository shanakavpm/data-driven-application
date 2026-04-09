"""
Module for training and evaluating credit card fraud detection models.
Implements a leakage-free pipeline with GridSearchCV and threshold optimization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import config as cfg
from src.utils import section, sub, save_pkl


def build_and_evaluate(df):
    """Run the full modelling pipeline and return metrics for JSON export."""
    section("TASK 3: MODEL BUILDING")

    # 1. Feature Selection & Binary Mapping
    df = df.copy()
    for col in cfg.BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map(cfg.BINARY_MAP)

    # Use the definitive feature list from config
    X = df[cfg.ALL_FEATURE_COLS]
    y = df[cfg.TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE, stratify=y,
    )
    print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # 2. Define Preprocessing Transformer
    # Extract categories from maps to ensure correct order
    edu_cats = sorted(cfg.ORDINAL_MAPS["EDUCATION_TYPE"], key=cfg.ORDINAL_MAPS["EDUCATION_TYPE"].get)
    age_cats = sorted(cfg.ORDINAL_MAPS["AGE_GROUP"], key=cfg.ORDINAL_MAPS["AGE_GROUP"].get)

    preprocessor = ColumnTransformer(
        transformers=[
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cfg.NOMINAL_COLS),
            ("ord", OrdinalEncoder(categories=[edu_cats, age_cats], 
                                   handle_unknown="use_encoded_value", unknown_value=-1), cfg.ORDINAL_COLS),
            ("num", StandardScaler(), cfg.NUMERICAL_COLS),
        ],
        remainder="passthrough"
    )

    def create_pipeline(classifier):
        """Encapsulate preprocessing and model into a single leakage-free Pipeline."""
        return ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=cfg.RANDOM_STATE)),
            ("clf", classifier)
        ])

    results = {}

    # Model 1: Random Forest Baseline
    rf_base_pipe = create_pipeline(RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=cfg.RANDOM_STATE, n_jobs=1,
    ))
    results["Random Forest (Baseline)"] = _train_evaluate(
        "Random Forest (Baseline)", rf_base_pipe, X_train, y_train, X_test, y_test
    )

    # Model 2: Logistic Regression
    lr_pipe = create_pipeline(LogisticRegression(
        class_weight="balanced", max_iter=1000,
        random_state=cfg.RANDOM_STATE, solver="lbfgs",
    ))
    results["Logistic Regression"] = _train_evaluate(
        "Logistic Regression", lr_pipe, X_train, y_train, X_test, y_test
    )

    # Improvement 1: GridSearchCV on Random Forest
    sub("Improvement 1 - Hyperparameter Tuning (GridSearchCV)")
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_leaf': [1, 2]
    }
    
    rf_pipe = create_pipeline(RandomForestClassifier(
        class_weight="balanced", random_state=cfg.RANDOM_STATE
    ))
    
    print(f"  Running GridSearchCV ({len(param_grid['clf__n_estimators'])*len(param_grid['clf__max_depth'])*len(param_grid['clf__min_samples_leaf'])} combos x 3 folds)...")
    
    gs = GridSearchCV(
        rf_pipe, param_grid, cv=3, scoring='f1', n_jobs=1, verbose=0
    )
    gs.fit(X_train, y_train)
    
    rf_tuned_pipe = gs.best_estimator_
    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV F1  : {gs.best_score_:.4f} (Leakage-protected)")
    
    results["RF + GridSearchCV"] = _train_evaluate(
        "RF + GridSearchCV", rf_tuned_pipe, X_train, y_train, X_test, y_test
    )

    # Improvement 2: Threshold Optimisation (Methodologically Rigorous)
    sub("Improvement 2 - Threshold Optimisation (Tuned on Train CV)")
    # Get cross-validated probabilities on training set to find optimal threshold
    y_train_proba = cross_val_predict(
        rf_tuned_pipe, X_train, y_train, cv=3, method="predict_proba", n_jobs=1
    )[:, 1]
    
    opt_thresh, opt_f1_train = _find_optimal_threshold(y_train, y_train_proba)
    print(f"  Optimal threshold (from Train CV) : {opt_thresh:.2f}")
    print(f"  Expected F1 on Train (CV)         : {opt_f1_train:.4f}")

    # Evaluate on Test Set once using the locked threshold
    y_test_proba = rf_tuned_pipe.predict_proba(X_test)[:, 1]
    y_test_pred_opt = (y_test_proba >= opt_thresh).astype(int)
    
    results["RF + Tuned Threshold"] = _metrics_dict(
        rf_tuned_pipe, y_test, y_test_pred_opt, y_test_proba
    )
    _print_report("RF + Tuned Threshold", results["RF + Tuned Threshold"], y_test, y_test_pred_opt)

    _print_comparison(results)

    # Feature names after transformation
    feature_names = _get_feature_names(preprocessor, X_train)

    # save plots and model files
    plot_files = _save_model_plots(results, y_test, feature_names, rf_tuned_pipe.named_steps["clf"])

    sub("Saving model files")
    save_pkl(rf_base_pipe,   os.path.join(cfg.MODEL_DIR, "rf_baseline.pkl"))
    save_pkl(lr_pipe,        os.path.join(cfg.MODEL_DIR, "logistic_regression.pkl"))
    save_pkl(rf_tuned_pipe,  os.path.join(cfg.MODEL_DIR, "rf_improved.pkl"))
    # The preprocessor is inside the pipeline, but we save individual components if needed for references
    save_pkl(feature_names,  os.path.join(cfg.MODEL_DIR, "feature_names.pkl"))
    save_pkl(opt_thresh,     os.path.join(cfg.MODEL_DIR, "optimal_threshold.pkl"))
    print("  All model files saved to models/")

    metrics_export = {
        name: {k: v for k, v in m.items() if k not in ["model", "y_pred", "y_prob"]}
        for name, m in results.items()
    }
    return {
        "models": metrics_export,
        "improvement": {
            "best_params": gs.best_params_,
            "optimal_threshold": opt_thresh,
            "optimal_f1": opt_f1_train,
        },
        "feature_names": feature_names,
        "plots": plot_files,
    }


def _get_feature_names(column_transformer, X):
    """Extract feature names after transformation."""
    new_names = []
    for name, trans, cols in column_transformer.transformers_:
        if name == "remainder":
            # For passthrough columns
            if trans == "passthrough":
                rem_cols = [X.columns[i] for i in cols]
                new_names.extend(rem_cols)
            continue
        
        if hasattr(trans, "get_feature_names_out"):
            # OneHot, etc
            names = trans.get_feature_names_out(cols)
            new_names.extend(names)
        else:
            # Scaler, Ordinal usually don't rename or we keep original
            new_names.extend(cols)
    return new_names


# --- Training and evaluation ---

def _train_evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = (model.predict_proba(X_te)[:, 1]
              if hasattr(model, "predict_proba") else None)

    result = _metrics_dict(model, y_te, y_pred, y_prob)
    _print_report(name, result, y_te, y_pred)
    return result


def _metrics_dict(model, y_true, y_pred, y_prob):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob) if y_prob is not None else 0,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "model":     model,
    }


def _print_report(name, m, y_true, y_pred):
    print(f"\n  {'-' * 50}")
    print(f"  {name}")
    print(f"  {'-' * 50}")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k:12s}: {m[k]:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Non-Fraud','Fraud'])}")


def _print_comparison(results):
    section("MODEL COMPARISON")
    hdr = f"  {'Model':<28s} {'Acc':>8s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'AUC':>8s}"
    print(hdr)
    print(f"  {'-' * 68}")
    for n, r in results.items():
        print(f"  {n:<28s} {r['accuracy']:>8.4f} {r['precision']:>8.4f} "
              f"{r['recall']:>8.4f} {r['f1']:>8.4f} {r['roc_auc']:>8.4f}")

    print("\n  OBSERVATIONS:")
    print("  - Accuracy is misleadingly high due to class imbalance (~98% non-fraud).")
    print("  - Precision vs Recall are conflicting: LR has higher recall")
    print("    but very low precision; RF has better precision but lower recall.")
    print("  - Threshold optimisation (protected by CV) boosts recall & F1.")
    print("  - Best CV F1 is now leakage-protected and reflects true generalization.")


# --- Threshold optimisation ---

def _find_optimal_threshold(y_true, y_proba):
    """
    Sweep thresholds 0.05..0.95 and pick the one that maximises F1.
    This is the key improvement: instead of a fixed 0.5 cut-off,
    we choose the threshold that best balances precision and recall
    for the fraud class.
    """
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(best_t, 2), round(best_f1, 4)


# --- Model evaluation plots ---

def _save_model_plots(results, y_test, feature_names, rf_model):
    files = []

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, r in results.items():
        if r["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
            ax.plot(fpr, tpr, lw=2,
                    label=f"{name} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title("ROC Curves", fontweight="bold", pad=20)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = "roc_curves.png"
    fig.savefig(os.path.join(cfg.PLOT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    files.append(fname)
    print(f"  Saved: {fname}")

    # confusion matrices
    n_models = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    for i, (name, r) in enumerate(results.items()):
        ax = axes_flat[i]
        cm = confusion_matrix(y_test, r["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    annot_kws={"size": 16, "weight": "bold"}, 
                    xticklabels=["Legit", "Fraud"],
                    yticklabels=["Legit", "Fraud"],
                    cbar_kws={"shrink": 0.8})
        short = name[:28] + "..." if len(name) > 30 else name
        ax.set_title(short, fontsize=12, fontweight="bold", pad=15)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_xlabel("Predicted", fontsize=11)
    
    # Remove any extra empty subplots if n_models < 4
    for j in range(i + 1, 4):
        fig.delaxes(axes_flat[j])

    fig.suptitle("Model Evaluation: Confusion Matrices", fontweight="bold", fontsize=16, y=1.02)
    fig.tight_layout(pad=4.0)
    fname = "confusion_matrices.png"
    fig.savefig(os.path.join(cfg.PLOT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    files.append(fname)
    print(f"  Saved: {fname}")

    # feature importance
    fig, ax = plt.subplots(figsize=(9, 7))
    imp = rf_model.feature_importances_
    idx = np.argsort(imp)
    ax.barh(np.array(feature_names)[idx], imp[idx],
            color="#3498db", edgecolor="white")
    ax.set_title("Feature Importance (RF Improved)", fontweight="bold", pad=20)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fname = "feature_importance.png"
    fig.savefig(os.path.join(cfg.PLOT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    files.append(fname)
    print(f"  Saved: {fname}")

    # performance comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    w = 0.8 / len(results)
    palette = sns.color_palette("Set2", len(results))
    for i, (name, r) in enumerate(results.items()):
        vals = [r[m] for m in metrics]
        ax.bar(x + i * w, vals, w, label=name, color=palette[i], edgecolor="white")
    ax.set_xticks(x + w * (len(results) - 1) / 2)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1.2)  # Extra headroom for legend
    ax.set_title("Model Performance Comparison", fontweight="bold", pad=20)
    ax.set_ylabel("Score")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fname = "model_comparison.png"
    fig.savefig(os.path.join(cfg.PLOT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    files.append(fname)
    print(f"  Saved: {fname}")

    return files
