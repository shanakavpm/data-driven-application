"""Task 2b - Preprocessing: missing values, outliers, duplicates, feature engineering."""

import numpy as np
import pandas as pd
import config as cfg
from src.utils import section, sub


def preprocess(df):
    """Full preprocessing pipeline. Returns a cleaned DataFrame."""
    section("TASK 2b: DATA PREPROCESSING")
    df = df.copy()

    df = _fill_missing(df)
    df = _remove_duplicates(df)
    df = _cap_outliers(df, cfg.NUMERIC_OUTLIER_COLS)
    df = _engineer_features(df)

    print(f"\n  Preprocessed shape: {df.shape}")
    return df


def _fill_missing(df):
    sub("Handling Missing Values")
    before = df.isnull().sum().sum()
    print(f"  Missing before: {before}")

    for col in df.columns:
        n_miss = df[col].isnull().sum()
        if n_miss == 0:
            continue
        if df[col].dtype in ("float64", "int64"):
            fill = df[col].median()
            df[col] = df[col].fillna(fill)
            print(f"  {col}: filled {n_miss} NaN with median ({fill})")
        else:
            fill = df[col].mode()[0]
            df[col] = df[col].fillna(fill)
            print(f"  {col}: filled {n_miss} NaN with mode ({fill})")

    print(f"  Missing after : {df.isnull().sum().sum()}")
    return df


def _remove_duplicates(df):
    sub("Removing Duplicates")
    n = df.duplicated().sum()
    print(f"  Duplicates found: {n}")
    if n > 0:
        df = df.drop_duplicates()
        print(f"  Rows after removal: {df.shape[0]:,}")
    return df


def _cap_outliers(df, columns):
    """Cap values outside 1.5x IQR to the fence values."""
    sub("Outlier Capping (IQR)")
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_out:
            df[col] = df[col].clip(lower=lo, upper=hi)
            print(f"  {col}: capped {n_out} outliers -> [{lo:.0f}, {hi:.0f}]")
    return df


def _engineer_features(df):
    sub("Feature Engineering")

    df["INCOME_PER_MEMBER"] = df["INCOME"] / df["FAMILY SIZE"].replace(0, 1)
    print("  Created: INCOME_PER_MEMBER")

    df["AGE_GROUP"] = pd.cut(
        df["AGE"], bins=[0, 25, 35, 45, 55, 100],
        labels=["Young", "Adult", "Middle", "Senior", "Elderly"],
    )
    print("  Created: AGE_GROUP")

    df["EMPLOYMENT_RATIO"] = df["YEARS_EMPLOYED"] / df["AGE"].replace(0, 1)
    print("  Created: EMPLOYMENT_RATIO")

    df["COMM_SCORE"] = df["WORK_PHONE"] + df["PHONE"] + df["E_MAIL"]
    print("  Created: COMM_SCORE")

    return df
