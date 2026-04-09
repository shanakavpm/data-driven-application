"""Task 1 - Load and merge the two CSV datasets."""

import pandas as pd
import config as cfg
from src.utils import section


def load_and_merge():
    """
    Read both CSVs, verify primary key uniqueness,
    inner-merge on ID/User, and confirm record counts.
    """
    section("TASK 1: DATA HANDLING")

    df1 = pd.read_csv(cfg.DATA1_PATH, na_values=cfg.NA_VALUES)
    df2 = pd.read_csv(cfg.DATA2_PATH, na_values=cfg.NA_VALUES)
    
    # 1. Uniqueness check (Academic Rigour)
    u1, u2 = df1[cfg.MERGE_LEFT_KEY].nunique(), df2[cfg.MERGE_RIGHT_KEY].nunique()
    print(f"  Dataset 1 : {df1.shape} | Unique IDs: {u1}")
    print(f"  Dataset 2 : {df2.shape} | Unique Users: {u2}")
    
    if u1 != len(df1) or u2 != len(df2):
        print("  WARNING: Duplicate keys found in input datasets!")

    unnamed = [c for c in df1.columns if "Unnamed" in str(c) or c == ""]
    if unnamed:
        df1 = df1.drop(columns=unnamed)

    # 2. Merge with verification
    df = pd.merge(df1, df2,
                  left_on=cfg.MERGE_LEFT_KEY,
                  right_on=cfg.MERGE_RIGHT_KEY,
                  how="inner")
    
    match_rate = (len(df) / len(df1)) * 100
    print(f"  Merged    : {df.shape[0]:,} rows | Match Rate: {match_rate:.2f}%")
    
    df = df.drop(columns=[cfg.MERGE_RIGHT_KEY])
    return df
