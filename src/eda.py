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
        _plot_bar(df, "GENDER", "Gender Distribution",
                  [COLORS["female"], COLORS["male"]]),
        _plot_hist(df, "INCOME", "#9b59b6", "Income Distribution"),
        _plot_hist(df, "AGE", "#1abc9c", "Age Distribution"),
        _plot_hbar(df, "INCOME_TYPE", "Income Type Distribution"),
        _plot_hbar(df, "EDUCATION_TYPE", "Education Type Distribution"),
        _plot_pie(df, "FAMILY_TYPE", "Family Type Distribution"),
    ]

    # bivariate
    sub("Bivariate Analysis")
    filenames["bivariate"] += [
        _plot_box_by_target(df, "INCOME", "Income by Fraud Status"),
        _plot_box_by_target(df, "AGE", "Age by Fraud Status"),
        _plot_fraud_rate(df, "GENDER", "Fraud Rate by Gender"),
        _plot_fraud_rate(df, "FAMILY_TYPE", "Fraud Rate by Family Type"),
        _plot_scatter_target(df, "AGE", "INCOME", "Age vs Income"),
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
        ax.text(b.get_x() + b.get_width()/2, v + 80,
                f"{v:,}\n({v/len(df)*100:.1f}%)",
                ha="center", fontweight="bold", fontsize=11)
    ax.set_title("Target Variable Distribution", fontweight="bold")
    ax.set_ylabel("Count")
    return _save(fig, "01_target_distribution.png")


def _plot_bar(df, col, title, colors=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    vc = df[col].value_counts()
    c = colors or sns.color_palette("Set2", len(vc))
    ax.bar(vc.index, vc.values, color=c, edgecolor="white")
    for i, (_, v) in enumerate(vc.items()):
        ax.text(i, v + 80, f"{v:,}", ha="center", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Count")
    return _save(fig, f"{col.lower()}_distribution.png")


def _plot_hist(df, col, color, title):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(df[col], bins=40, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(df[col].mean(), color="red", ls="--",
               label=f"Mean: {df[col].mean():,.1f}")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.legend()
    return _save(fig, f"{col.lower()}_distribution.png")


def _plot_hbar(df, col, title):
    fig, ax = plt.subplots(figsize=(9, 4))
    vc = df[col].value_counts()
    colors = sns.color_palette("Set2", len(vc))
    ax.barh(vc.index, vc.values, color=colors, edgecolor="white")
    for bar, v in zip(ax.patches, vc.values):
        ax.text(v + 30, bar.get_y() + bar.get_height()/2,
                f"{v:,}", va="center", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Count")
    return _save(fig, f"{col.lower()}_distribution.png")


def _plot_pie(df, col, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    vc = df[col].value_counts()
    ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%",
           colors=sns.color_palette("pastel"), startangle=140)
    ax.set_title(title, fontweight="bold")
    return _save(fig, f"{col.lower()}_distribution.png")


def _plot_box_by_target(df, col, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    df.boxplot(column=col, by=cfg.TARGET_COL, ax=ax,
               boxprops=dict(color="#2c3e50"),
               medianprops=dict(color="red"))
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Target (0 = Legitimate, 1 = Fraud)")
    ax.set_ylabel(col)
    plt.suptitle("")
    return _save(fig, f"{col.lower()}_by_target.png")


def _plot_fraud_rate(df, col, title):
    fig, ax = plt.subplots(figsize=(9, 4))
    ct = pd.crosstab(df[col], df[cfg.TARGET_COL], normalize="index") * 100
    ct.plot(kind="bar", ax=ax,
            color=[COLORS["safe"], COLORS["fraud"]], edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.legend(["Non-Fraud", "Fraud"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    return _save(fig, f"fraud_rate_{col.lower()}.png")


def _plot_scatter_target(df, x, y, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    sample = df.sample(min(5000, len(df)), random_state=cfg.RANDOM_STATE)
    for label, c, m, a, s in [(0, COLORS["safe"], "o", 0.3, 10),
                               (1, COLORS["fraud"], "x", 0.9, 30)]:
        sub_df = sample[sample[cfg.TARGET_COL] == label]
        ax.scatter(sub_df[x], sub_df[y], c=c, marker=m, alpha=a, s=s,
                   label="Fraud" if label else "Non-Fraud")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    return _save(fig, f"{x.lower()}_vs_{y.lower()}_scatter.png")


def _plot_corr_heatmap(df):
    fig, ax = plt.subplots(figsize=(13, 9))
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlBu_r", center=0, ax=ax, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.7})
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
