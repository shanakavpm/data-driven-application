"""Shared helper functions."""

import json
import numpy as np
import pandas as pd
import joblib


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def sub(title):
    print(f"\n--- {title} ---")


def save_pkl(obj, path):
    joblib.dump(obj, path)


def load_pkl(path):
    return joblib.load(path)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def encode_binary(df, columns, mapping):
    df = df.copy()
    for col in columns:
        df[col] = df[col].map(mapping)
    return df


def _json_default(obj):
    """Handle numpy types when writing JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
