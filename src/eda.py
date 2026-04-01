"""Task 2a & 2c - Data understanding and EDA visualisations."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg
from src.utils import section, sub

sns.set_theme(style="whitegrid", palette="muted")

COLORS = {
    "safe": "#2ecc71", "fraud": "#e74c3c",
    "male": "#3498db", "female": "#e91e63",
}


# --- Task 2a: Data understanding ---

def show_data_understanding(df):
    """Print dataset stats and return a summary dict for the web GUI."""
    section("TASK 2a: DATA UNDERSTANDING")

    sub("Shape & Types")
    print(f"  Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
    print(df.dtypes.to_string())

    sub("Unique Values")
    for col in df.columns:
        print(f"  {col:20s}: {df[col].nunique()}")

    sub("Statistical Summary - Numerical")
    print(df.describe().to_string())

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if cat_cols:
        sub("Statistical Summary - Categorical")
        print(df[cat_cols].describe().to_string())

    sub("Missing Values")
    missing = df.isnull().sum()
    miss_cols = missing[missing > 0]
    if miss_cols.empty:
        print("  None")
    else:
        for col, n in miss_cols.items():
            print(f"  {col}: {n} ({n / len(df) * 100:.2f}%)")
    print(f"  Total missing: {missing.sum()}")

    sub("Target Distribution")
    t = df[cfg.TARGET_COL].value_counts()
    print(f"  Non-fraud (0): {t.get(0, 0):,}  ({t.get(0, 0) / len(df) * 100:.2f}%)")
    print(f"  Fraud     (1): {t.get(1, 0):,}  ({t.get(1, 0) / len(df) * 100:.2f}%)")

    return _build_summary(df, missing)


# --- Task 2c: Visualisations ---

def generate_visualisations(df):
    """Create all charts, save to plots/, return filenames grouped by type."""
    section("TASK 2c: VISUALISATIONS")

    filenames = {"univariate": [], "bivariate": [], "multivariate": []}

    # univariate
    sub("Univariate Analysis")
    filenames["univariate"] += [
        _plot_target(df),
    ]

    # bivariate
    sub("Bivariate Analysis")
    filenames["bivariate"] += [
        _plot_box_by_target(df, "INCOME", "Income by Fraud Status"),
        _plot_box_by_target(df, "AGE", "Age by Fraud Status"),
        _plot_fraud_rate(df, "GENDER", "Fraud Rate by Gender"),
        _plot_fraud_rate(df, "FAMILY_TYPE", "Fraud Rate by Family Type"),
    ]

    # multivariate
    sub("Multivariate Analysis")
    filenames["multivariate"] += [
        _plot_corr_heatmap(df),
        _plot_box_multi(df, "HOUSE_TYPE", "INCOME",
                        "Income by House Type & Fraud"),
    ]

    return filenames


# --- Plot helpers ---

def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.PLOT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")
    return name


def _plot_target(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    vc = df[cfg.TARGET_COL].value_counts()
    bars = ax.bar(["Non-Fraud (0)", "Fraud (1)"], vc.values,
                  color=[COLORS["safe"], COLORS["fraud"]], edgecolor="white")
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width()/2, v + (vc.max() * 0.01),
                f"{v:,}\n({v/len(df)*100:.1f}%)",
                ha="center", fontweight="bold", fontsize=10)
    
    ax.set_ylim(0, vc.max() * 1.25)  # Add 25% space at the top
    ax.set_title("Target Variable Distribution", fontweight="bold", pad=20)
    ax.set_ylabel("Count")
    return _save(fig, "target_distribution.png")


def _plot_box_by_target(df, col, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    df.boxplot(column=col, by=cfg.TARGET_COL, ax=ax,
               boxprops=dict(color="#2c3e50"),
               medianprops=dict(color="red"))
    ax.set_title(title, fontweight="bold", pad=20)
    ax.set_xlabel("Target (0 = Legitimate, 1 = Fraud)")
    ax.set_ylabel(col)
    plt.suptitle("")
    return _save(fig, f"{col.lower()}_by_target.png")


def _plot_fraud_rate(df, col, title):
    fig, ax = plt.subplots(figsize=(9, 4))
    ct = pd.crosstab(df[col], df[cfg.TARGET_COL], normalize="index") * 100
    ct.plot(kind="bar", ax=ax,
            color=[COLORS["safe"], COLORS["fraud"]], edgecolor="white")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 115)  # Give room above 100% bars if they exist
    ax.legend(["Non-Fraud", "Fraud"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    return _save(fig, f"fraud_rate_{col.lower()}.png")


def _plot_corr_heatmap(df):
    fig, ax = plt.subplots(figsize=(12, 10))
    num = df.select_dtypes(include=[np.number])
    # Drop irrelevant or zero-variance columns for a cleaner heatmap
    to_drop = ["ID", "FLAG_MOBIL", "WORK_PHONE", "PHONE"]
    num = num.drop(columns=[c for c in to_drop if c in num.columns])
    
    corr = num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlBu_r", center=0, ax=ax, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.6})
    ax.set_title("Correlation Heatmap", fontweight="bold")
    return _save(fig, "correlation_heatmap.png")


def _plot_box_multi(df, group, value, title):
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df, x=group, y=value, hue=cfg.TARGET_COL,
                palette={0: COLORS["safe"], 1: COLORS["fraud"]}, ax=ax)
    ax.set_title(title, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    return _save(fig, f"{value.lower()}_by_{group.lower()}_target.png")


# --- Summary builder for the web app ---

def _build_summary(df, missing):
    cols_info = []
    for c in df.columns:
        cols_info.append({
            "name": c,
            "dtype": str(df[c].dtype),
            "unique": int(df[c].nunique()),
            "missing": int(df[c].isnull().sum()),
            "description": cfg.COLUMN_DESCRIPTIONS.get(c, ""),
        })

    target = df[cfg.TARGET_COL].value_counts()
    return {
        "total_rows": int(len(df)),
        "total_columns": int(df.shape[1]),
        "fraud_count": int(target.get(1, 0)),
        "fraud_pct": round(target.get(1, 0) / len(df) * 100, 2),
        "columns": cols_info,
        "missing_total": int(missing.sum()),
        "sample_rows": df.head(10).fillna("").to_dict(orient="records"),
    }
