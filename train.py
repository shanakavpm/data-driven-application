"""
train.py - Run the full pipeline: import, explore, preprocess, build models.

Usage:
    python train.py
"""
import warnings
warnings.filterwarnings("ignore")

import config as cfg
from src.data_handler  import load_and_merge
from src.eda           import show_data_understanding, generate_visualisations
from src.preprocessor  import preprocess
from src.model_builder import build_and_evaluate
from src.utils         import save_json


def main():
    print("\n--- CREDIT CARD FRAUD DETECTION ---\n")

    # Task 1: load and merge datasets
    df_raw = load_and_merge()

    # Task 2a: data understanding
    dataset_summary = show_data_understanding(df_raw)

    # Task 2b: preprocessing
    df_clean = preprocess(df_raw)
    df_clean.to_csv(cfg.CLEANED_DATA_PATH, index=False)
    print(f"  Cleaned data saved to {cfg.CLEANED_DATA_PATH}")

    # Task 2c: EDA visualisations
    eda_plots = generate_visualisations(df_clean)

    # Task 3: model training and evaluation
    model_results = build_and_evaluate(df_clean)

    # save summary for the Flask app
    summary = {
        "dataset":   dataset_summary,
        "eda_plots": eda_plots,
        **model_results,
    }
    save_json(summary, cfg.SUMMARY_PATH)
    print(f"  Summary JSON saved to {cfg.SUMMARY_PATH}")
    print("\n--- ALL TASKS COMPLETED ---\n")


if __name__ == "__main__":
    main()
