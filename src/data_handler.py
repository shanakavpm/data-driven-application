"""Task 1 - Load and merge the two CSV datasets."""

import pandas as pd
import config as cfg
from src.utils import section


def load_and_merge():
    """
    Read both CSVs, drop the unnamed index column,
    inner-merge on ID/User, drop the redundant User column.
    """
    section("TASK 1: DATA HANDLING")

    df1 = pd.read_csv(cfg.DATA1_PATH, na_values=cfg.NA_VALUES)
    df2 = pd.read_csv(cfg.DATA2_PATH, na_values=cfg.NA_VALUES)
    print(f"  Dataset 1 : {df1.shape}  |  Dataset 2 : {df2.shape}")

    unnamed = [c for c in df1.columns if "Unnamed" in str(c) or c == ""]
    if unnamed:
        df1 = df1.drop(columns=unnamed)

    df = pd.merge(df1, df2,
                  left_on=cfg.MERGE_LEFT_KEY,
                  right_on=cfg.MERGE_RIGHT_KEY,
                  how="inner")
    df = df.drop(columns=[cfg.MERGE_RIGHT_KEY])

    print(f"  Merged    : {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df
