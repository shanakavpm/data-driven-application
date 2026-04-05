"""Centralised configuration - paths and constants."""

import os

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATA1_PATH = os.path.join(DATA_DIR, "Credit_Card_Dataset_2025_Sept_1.csv")
DATA2_PATH = os.path.join(DATA_DIR, "Credit_Card_Dataset_2025_Sept_2.csv")

PLOT_DIR   = os.path.join(BASE_DIR, "plots")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

SUMMARY_PATH      = os.path.join(OUTPUT_DIR, "summary.json")
CLEANED_DATA_PATH = os.path.join(OUTPUT_DIR, "cleaned_dataset.csv")

# dataset constants
MERGE_LEFT_KEY  = "ID"
MERGE_RIGHT_KEY = "User"
TARGET_COL      = "TARGET"
NA_VALUES       = ["NA", "na", "N/A", ""]

# feature groups
BINARY_COLS = ["GENDER", "CAR", "REALITY"]
BINARY_MAP  = {"M": 1, "F": 0, "Y": 1, "N": 0}

CATEGORICAL_COLS = ["FAMILY_TYPE", "HOUSE_TYPE", "INCOME_TYPE",
                    "EDUCATION_TYPE", "AGE_GROUP"]

NUMERIC_OUTLIER_COLS = ["INCOME", "NO_OF_CHILD", "FAMILY SIZE",
                        "BEGIN_MONTH", "AGE", "YEARS_EMPLOYED"]

DROP_COLS = ["ID", "FLAG_MOBIL"]

# model constants
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 3

# column descriptions shown in the UI
COLUMN_DESCRIPTIONS = {
    "ID":             "Unique transaction identifier",
    "GENDER":         "Gender of cardholder (M / F)",
    "CAR":            "Car ownership (Y / N)",
    "REALITY":        "Property ownership (Y / N)",
    "NO_OF_CHILD":    "Number of children",
    "INCOME":         "Annual income",
    "INCOME_TYPE":    "Employment category",
    "EDUCATION_TYPE": "Education level",
    "FAMILY_TYPE":    "Marital / family status",
    "HOUSE_TYPE":     "Housing type",
    "FLAG_MOBIL":     "Mobile phone flag (always 1)",
    "WORK_PHONE":     "Has work phone (0 / 1)",
    "PHONE":          "Has home phone (0 / 1)",
    "E_MAIL":         "Has e-mail (0 / 1)",
    "FAMILY SIZE":    "Number of family members",
    "BEGIN_MONTH":    "Month of account opening (relative)",
    "AGE":            "Cardholder age in years",
    "YEARS_EMPLOYED": "Years of employment",
    "TARGET":         "Fraud indicator (1 = Fraud, 0 = Legitimate)",
}

# make sure output folders exist
for _dir in (PLOT_DIR, MODEL_DIR, OUTPUT_DIR):
    os.makedirs(_dir, exist_ok=True)
