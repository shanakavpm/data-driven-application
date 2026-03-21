"""
CMP7005 Assignment – Credit Card Fraud Detection
=================================================
This script performs:
  Task 1: Data Handling – Import and merge datasets
  Task 2: Exploratory Data Analysis (EDA) – Understanding, preprocessing, visualisation
  Task 3: Model Building – Random Forest & Logistic Regression with improvements
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
import joblib

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA1_PATH = os.path.join(BASE_DIR, "Credit_Card_Dataset_2025_Sept_1.csv")
DATA2_PATH = os.path.join(BASE_DIR, "Credit_Card_Dataset_2025_Sept_2.csv")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =============================================================================
# TASK 1 – DATA HANDLING
# =============================================================================
def task1_data_handling():
    """Import and merge the two datasets."""
    print("=" * 70)
    print("TASK 1: DATA HANDLING")
    print("=" * 70)

    # Import datasets – treat 'NA' strings as NaN
    df1 = pd.read_csv(DATA1_PATH, na_values=["NA", "na", "N/A", ""])
    df2 = pd.read_csv(DATA2_PATH, na_values=["NA", "na", "N/A", ""])

    print(f"\nDataset 1 shape: {df1.shape}")
    print(f"Dataset 1 columns: {list(df1.columns)}")
    print(f"\nDataset 2 shape: {df2.shape}")
    print(f"Dataset 2 columns: {list(df2.columns)}")

    # Drop the unnamed index column from Dataset 1 if present
    if df1.columns[0] == "" or "Unnamed" in str(df1.columns[0]):
        df1 = df1.drop(columns=[df1.columns[0]])
        print("\nDropped unnamed index column from Dataset 1.")

    # Merge on primary key: ID (Dataset 1) <-> User (Dataset 2)
    print("\nMerging datasets on ID (Dataset 1) = User (Dataset 2) ...")
    df = pd.merge(df1, df2, left_on="ID", right_on="User", how="inner")

    # Drop the redundant 'User' column after merge
    df = df.drop(columns=["User"])

    print(f"\n✅ Merged dataset shape: {df.shape}")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"\nMerged columns ({df.shape[1]}):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")

    return df


# =============================================================================
# TASK 2a – FUNDAMENTAL DATA UNDERSTANDING
# =============================================================================
def task2a_data_understanding(df):
    """Demonstrate understanding of the data."""
    print("\n" + "=" * 70)
    print("TASK 2a: FUNDAMENTAL DATA UNDERSTANDING")
    print("=" * 70)

    # Basic shape
    print(f"\n--- Shape ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Data types
    print(f"\n--- Data Types ---")
    print(df.dtypes.to_string())

    # Info
    print(f"\n--- Dataset Info ---")
    print(df.info())

    # Unique values
    print(f"\n--- Unique Values per Column ---")
    for col in df.columns:
        print(f"   {col:20s}: {df[col].nunique()} unique values")

    # Statistical summary – numerical
    print(f"\n--- Statistical Summary (Numerical) ---")
    print(df.describe().to_string())

    # Statistical summary – categorical
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        print(f"\n--- Statistical Summary (Categorical) ---")
        print(df[cat_cols].describe().to_string())

    # Missing values
    print(f"\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Count": missing, "Percentage (%)": missing_pct})
    print(missing_df[missing_df["Count"] > 0].to_string() if missing.sum() > 0 else "   No missing values found.")
    print(f"\nTotal missing values: {missing.sum()}")

    # Target distribution
    print(f"\n--- Target Distribution ---")
    target_counts = df["TARGET"].value_counts()
    print(f"   Non-fraud (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(df)*100:.2f}%)")
    print(f"   Fraud    (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(df)*100:.2f}%)")

    # Interpretation
    print("\n📊 INTERPRETATION:")
    print("   • The dataset contains 25,134 transactions with 19 features.")
    print("   • There are 12 numerical and 7 categorical features.")
    print("   • The target variable is highly imbalanced – only ~1.6% are fraudulent.")
    print("   • Missing values are present in FAMILY SIZE and YEARS_EMPLOYED (as 'NA' strings).")
    print("   • This imbalance will need to be addressed before model building.")


# =============================================================================
# TASK 2b – DATA PREPROCESSING
# =============================================================================
def task2b_preprocessing(df):
    """Handle missing values, outliers, duplicates, and feature engineering."""
    print("\n" + "=" * 70)
    print("TASK 2b: DATA PREPROCESSING")
    print("=" * 70)

    df = df.copy()

    # ── 1. Handle Missing Values ──────────────────────────────────────────
    print("\n--- Handling Missing Values ---")
    missing_before = df.isnull().sum().sum()
    print(f"   Missing values before: {missing_before}")

    # FAMILY SIZE – fill with median
    if df["FAMILY SIZE"].isnull().sum() > 0:
        median_fs = df["FAMILY SIZE"].median()
        df["FAMILY SIZE"] = df["FAMILY SIZE"].fillna(median_fs)
        print(f"   FAMILY SIZE: filled {df['FAMILY SIZE'].isnull().sum()} remaining NaN with median ({median_fs})")

    # YEARS_EMPLOYED – fill with median
    if df["YEARS_EMPLOYED"].isnull().sum() > 0:
        median_ye = df["YEARS_EMPLOYED"].median()
        df["YEARS_EMPLOYED"] = df["YEARS_EMPLOYED"].fillna(median_ye)
        print(f"   YEARS_EMPLOYED: filled NaN with median ({median_ye})")

    # Fill any other remaining missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].median())
                print(f"   {col}: filled with median")
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                print(f"   {col}: filled with mode")

    missing_after = df.isnull().sum().sum()
    print(f"   Missing values after: {missing_after}")

    # ── 2. Remove Duplicate Rows ─────────────────────────────────────────
    print("\n--- Handling Duplicates ---")
    dups_before = df.duplicated().sum()
    print(f"   Duplicate rows found: {dups_before}")
    if dups_before > 0:
        df = df.drop_duplicates()
        print(f"   Rows after removing duplicates: {df.shape[0]}")

    # ── 3. Handle Outliers (IQR method for numerical columns) ────────────
    print("\n--- Outlier Detection (IQR Method) ---")
    numerical_cols = ["INCOME", "NO_OF_CHILD", "FAMILY SIZE", "BEGIN_MONTH", "AGE", "YEARS_EMPLOYED"]
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            print(f"   {col}: {outliers} outliers detected (range: [{lower:.1f}, {upper:.1f}])")
            # Cap outliers instead of removing (to preserve data)
            df[col] = df[col].clip(lower=lower, upper=upper)
            print(f"      → Capped to [{lower:.1f}, {upper:.1f}]")

    # ── 4. Feature Engineering ───────────────────────────────────────────
    print("\n--- Feature Engineering ---")

    # Income per family member
    df["INCOME_PER_MEMBER"] = df["INCOME"] / df["FAMILY SIZE"].replace(0, 1)
    print("   Created: INCOME_PER_MEMBER = INCOME / FAMILY SIZE")

    # Age group
    df["AGE_GROUP"] = pd.cut(df["AGE"], bins=[0, 25, 35, 45, 55, 100],
                              labels=["Young", "Adult", "Middle", "Senior", "Elderly"])
    print("   Created: AGE_GROUP (Young / Adult / Middle / Senior / Elderly)")

    # Employment ratio
    df["EMPLOYMENT_RATIO"] = df["YEARS_EMPLOYED"] / df["AGE"].replace(0, 1)
    print("   Created: EMPLOYMENT_RATIO = YEARS_EMPLOYED / AGE")

    print(f"\n✅ Preprocessed dataset shape: {df.shape}")
    return df


# =============================================================================
# TASK 2c – STATISTICAL ANALYSIS & VISUALISATION
# =============================================================================
def task2c_analysis_visualisation(df):
    """Perform univariate, bivariate, multivariate analysis with visualisations."""
    print("\n" + "=" * 70)
    print("TASK 2c: STATISTICAL ANALYSIS & VISUALISATION")
    print("=" * 70)

    # ── UNIVARIATE ANALYSIS ──────────────────────────────────────────────
    print("\n--- Univariate Analysis ---")

    # 1. Target distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    target_counts = df["TARGET"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(["Non-Fraud (0)", "Fraud (1)"], target_counts.values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, target_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                f"{val}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax.set_title("Target Variable Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "01_target_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 01_target_distribution.png")

    # 2. Gender distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    gender_counts = df["GENDER"].value_counts()
    ax.bar(gender_counts.index, gender_counts.values, color=["#3498db", "#e91e63"], edgecolor="white")
    for i, (idx, val) in enumerate(gender_counts.items()):
        ax.text(i, val + 100, f"{val}", ha="center", fontweight="bold")
    ax.set_title("Gender Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "02_gender_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 02_gender_distribution.png")

    # 3. Income distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["INCOME"], bins=50, color="#9b59b6", edgecolor="white", alpha=0.85)
    ax.axvline(df["INCOME"].mean(), color="red", linestyle="--", label=f"Mean: {df['INCOME'].mean():,.0f}")
    ax.axvline(df["INCOME"].median(), color="blue", linestyle="--", label=f"Median: {df['INCOME'].median():,.0f}")
    ax.set_title("Income Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Income")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "03_income_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 03_income_distribution.png")

    # 4. Age distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["AGE"], bins=30, color="#1abc9c", edgecolor="white", alpha=0.85)
    ax.axvline(df["AGE"].mean(), color="red", linestyle="--", label=f"Mean: {df['AGE'].mean():.1f}")
    ax.set_title("Age Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "04_age_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 04_age_distribution.png")

    # 5. Income Type distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    income_type_counts = df["INCOME_TYPE"].value_counts()
    colors_it = sns.color_palette("Set2", len(income_type_counts))
    bars = ax.barh(income_type_counts.index, income_type_counts.values, color=colors_it, edgecolor="white")
    for bar, val in zip(bars, income_type_counts.values):
        ax.text(val + 50, bar.get_y() + bar.get_height()/2, f"{val}", va="center", fontweight="bold")
    ax.set_title("Income Type Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "05_income_type_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 05_income_type_distribution.png")

    # 6. Education Type distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    edu_counts = df["EDUCATION_TYPE"].value_counts()
    colors_ed = sns.color_palette("Set3", len(edu_counts))
    ax.barh(edu_counts.index, edu_counts.values, color=colors_ed, edgecolor="white")
    for i, (idx, val) in enumerate(edu_counts.items()):
        ax.text(val + 30, i, f"{val}", va="center", fontweight="bold")
    ax.set_title("Education Type Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "06_education_type_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 06_education_type_distribution.png")

    # 7. Family Type distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    fam_counts = df["FAMILY_TYPE"].value_counts()
    ax.pie(fam_counts.values, labels=fam_counts.index, autopct="%1.1f%%",
           colors=sns.color_palette("pastel"), startangle=140)
    ax.set_title("Family Type Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "07_family_type_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 07_family_type_distribution.png")

    # ── BIVARIATE ANALYSIS ───────────────────────────────────────────────
    print("\n--- Bivariate Analysis ---")

    # 8. Income by Target
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column="INCOME", by="TARGET", ax=ax,
               boxprops=dict(color="#2c3e50"), medianprops=dict(color="red"))
    ax.set_title("Income by Fraud Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Target (0=Non-Fraud, 1=Fraud)")
    ax.set_ylabel("Income")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "08_income_by_target.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 08_income_by_target.png")

    # 9. Age by Target
    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column="AGE", by="TARGET", ax=ax,
               boxprops=dict(color="#2c3e50"), medianprops=dict(color="red"))
    ax.set_title("Age by Fraud Status", fontsize=14, fontweight="bold")
    ax.set_xlabel("Target (0=Non-Fraud, 1=Fraud)")
    ax.set_ylabel("Age")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "09_age_by_target.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 09_age_by_target.png")

    # 10. Gender vs. Target
    fig, ax = plt.subplots(figsize=(8, 5))
    ct = pd.crosstab(df["GENDER"], df["TARGET"], normalize="index") * 100
    ct.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="white")
    ax.set_title("Fraud Rate by Gender", fontsize=14, fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Gender")
    ax.legend(["Non-Fraud", "Fraud"], loc="upper right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "10_fraud_rate_by_gender.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 10_fraud_rate_by_gender.png")

    # 11. Family Type vs. Target
    fig, ax = plt.subplots(figsize=(10, 5))
    ct2 = pd.crosstab(df["FAMILY_TYPE"], df["TARGET"], normalize="index") * 100
    ct2.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="white")
    ax.set_title("Fraud Rate by Family Type", fontsize=14, fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.legend(["Non-Fraud", "Fraud"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "11_fraud_rate_by_family_type.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 11_fraud_rate_by_family_type.png")

    # 12. Scatter: Age vs. Income coloured by Target
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter_df = df.sample(min(5000, len(df)), random_state=42)
    ax.scatter(scatter_df[scatter_df["TARGET"] == 0]["AGE"],
               scatter_df[scatter_df["TARGET"] == 0]["INCOME"],
               alpha=0.3, s=10, c="#2ecc71", label="Non-Fraud")
    ax.scatter(scatter_df[scatter_df["TARGET"] == 1]["AGE"],
               scatter_df[scatter_df["TARGET"] == 1]["INCOME"],
               alpha=0.8, s=30, c="#e74c3c", label="Fraud", marker="x")
    ax.set_title("Age vs Income (by Fraud Status)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_ylabel("Income")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "12_age_vs_income_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 12_age_vs_income_scatter.png")

    # ── MULTIVARIATE ANALYSIS ────────────────────────────────────────────
    print("\n--- Multivariate Analysis ---")

    # 13. Correlation Heatmap
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, ax = plt.subplots(figsize=(14, 10))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, ax=ax, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap (Numerical Features)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "13_correlation_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 13_correlation_heatmap.png")

    # 14. Pairplot for key features (subsample for speed)
    key_features = ["AGE", "INCOME", "YEARS_EMPLOYED", "NO_OF_CHILD", "TARGET"]
    sample = df[key_features].sample(min(2000, len(df)), random_state=42)
    g = sns.pairplot(sample, hue="TARGET", palette={0: "#2ecc71", 1: "#e74c3c"},
                     diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15})
    g.fig.suptitle("Pairplot of Key Features by Fraud Status", y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "14_pairplot_key_features.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 14_pairplot_key_features.png")

    # 15. House Type vs Income by Target
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df, x="HOUSE_TYPE", y="INCOME", hue="TARGET",
                palette={0: "#2ecc71", 1: "#e74c3c"}, ax=ax)
    ax.set_title("Income by House Type & Fraud Status", fontsize=14, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "15_income_by_house_target.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 15_income_by_house_target.png")

    # Interpretation
    print("\n📊 KEY INSIGHTS FROM ANALYSIS:")
    print("   • The dataset is highly imbalanced (98.4% non-fraud vs 1.6% fraud).")
    print("   • Income distributions are similar between fraud and non-fraud groups.")
    print("   • Age does not show a strong separation between fraud classes.")
    print("   • FAMILY SIZE and NO_OF_CHILD show weak positive correlation.")
    print("   • FLAG_MOBIL is constant (all = 1) – provides no discriminative power.")
    print("   • Married individuals form the largest group across family types.")
    print("   • Most applicants are Working class or Commercial associates.")


# =============================================================================
# TASK 3 – MODEL BUILDING
# =============================================================================
def task3_model_building(df):
    """Build, evaluate, and improve two ML models."""
    print("\n" + "=" * 70)
    print("TASK 3: MODEL BUILDING")
    print("=" * 70)

    df_model = df.copy()

    # ── Encoding ─────────────────────────────────────────────────────────
    print("\n--- Encoding Categorical Variables ---")

    # Binary encoding
    binary_map = {"M": 1, "F": 0, "Y": 1, "N": 0}
    for col in ["GENDER", "CAR", "REALITY"]:
        df_model[col] = df_model[col].map(binary_map)
        print(f"   {col}: binary encoded (M/Y→1, F/N→0)")

    # Label encode multi-class categoricals
    label_encoders = {}
    cat_cols = ["FAMILY_TYPE", "HOUSE_TYPE", "INCOME_TYPE", "EDUCATION_TYPE", "AGE_GROUP"]
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        label_encoders[col] = le
        print(f"   {col}: label encoded ({len(le.classes_)} classes)")

    # ── Feature Selection ────────────────────────────────────────────────
    print("\n--- Feature Selection ---")

    # Remove ID and FLAG_MOBIL (constant)
    drop_cols = ["ID", "FLAG_MOBIL"]
    for c in drop_cols:
        if c in df_model.columns:
            df_model = df_model.drop(columns=[c])
            print(f"   Dropped: {c}")

    X = df_model.drop(columns=["TARGET"])
    y = df_model["TARGET"]

    feature_names = X.columns.tolist()
    print(f"\n   Features used ({len(feature_names)}): {feature_names}")
    print(f"   Target distribution: {dict(y.value_counts())}")

    # ── Train-Test Split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n   Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")

    # ── Feature Scaling ──────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Handle Class Imbalance with SMOTE ────────────────────────────────
    print("\n--- Applying SMOTE for Class Imbalance ---")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"   Before SMOTE: {dict(pd.Series(y_train).value_counts())}")
    print(f"   After  SMOTE: {dict(pd.Series(y_train_res).value_counts())}")

    # ── Helper: Evaluate Model ───────────────────────────────────────────
    results = {}

    def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        auc = roc_auc_score(y_te, y_prob) if y_prob is not None else 0

        print(f"\n   {'─' * 50}")
        print(f"   {name}")
        print(f"   {'─' * 50}")
        print(f"   Accuracy : {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall   : {rec:.4f}")
        print(f"   F1-Score : {f1:.4f}")
        print(f"   ROC-AUC  : {auc:.4f}")
        print(f"\n   Confusion Matrix:")
        cm = confusion_matrix(y_te, y_pred)
        print(f"   {cm}")
        print(f"\n   Classification Report:")
        print(classification_report(y_te, y_pred, target_names=["Non-Fraud", "Fraud"]))

        results[name] = {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "roc_auc": auc, "model": model,
            "y_pred": y_pred, "y_prob": y_prob
        }
        return model

    # ══ MODEL 1: Random Forest (Baseline) ════════════════════════════════
    print("\n" + "=" * 50)
    print("MODEL 1: Random Forest Classifier (Baseline)")
    print("=" * 50)

    rf_base = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    evaluate_model("Random Forest (Baseline)", rf_base, X_train_res, y_train_res, X_test_scaled, y_test)

    # ══ MODEL 2: Logistic Regression ═════════════════════════════════════
    print("\n" + "=" * 50)
    print("MODEL 2: Logistic Regression")
    print("=" * 50)

    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs"
    )
    evaluate_model("Logistic Regression", lr_model, X_train_res, y_train_res, X_test_scaled, y_test)

    # ══ IMPROVEMENT: Random Forest with GridSearchCV ═════════════════════
    print("\n" + "=" * 50)
    print("IMPROVEMENT: Random Forest with Hyperparameter Tuning (GridSearchCV)")
    print("=" * 50)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    print("\n   Performing GridSearchCV (this may take a few minutes)...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train_res, y_train_res)
    print(f"   Best Parameters: {grid_search.best_params_}")
    print(f"   Best CV F1-Score: {grid_search.best_score_:.4f}")

    rf_improved = grid_search.best_estimator_
    evaluate_model("Random Forest (Improved)", rf_improved, X_train_res, y_train_res, X_test_scaled, y_test)

    # ── Model Comparison Summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n   {'Model':<30s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'ROC-AUC':>10s}")
    print(f"   {'─'*80}")
    for name, r in results.items():
        print(f"   {name:<30s} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}")

    print("\n📊 OBSERVATIONS:")
    print("   • Accuracy may appear high for all models due to class imbalance.")
    print("   • Recall (sensitivity) for the fraud class is more important than accuracy,")
    print("     since failing to detect fraud has higher cost than false alarms.")
    print("   • The trade-off between precision and recall creates conflicting observations:")
    print("     a model can be very accurate overall but miss many fraudulent cases.")
    print("   • SMOTE helps improve recall by balancing the training data.")
    print("   • The improved Random Forest with hyperparameter tuning typically achieves")
    print("     better F1 and recall compared to the baseline.")

    # ── Save plots ───────────────────────────────────────────────────────
    # 16. ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, r in results.items():
        if r["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
            ax.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title("ROC Curves – Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "16_roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("\n   Saved: 16_roc_curves.png")

    # 17. Confusion Matrices side by side
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, r) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, r["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    plt.suptitle("Confusion Matrices – All Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "17_confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 17_confusion_matrices.png")

    # 18. Feature Importance (Random Forest Improved)
    fig, ax = plt.subplots(figsize=(10, 8))
    importances = rf_improved.feature_importances_
    idx = np.argsort(importances)
    ax.barh(np.array(feature_names)[idx], importances[idx], color="#3498db", edgecolor="white")
    ax.set_title("Feature Importance (Random Forest – Improved)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "18_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 18_feature_importance.png")

    # 19. Model Performance Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(metrics))
    width = 0.25
    colors_bar = ["#3498db", "#e74c3c", "#2ecc71"]
    for i, (name, r) in enumerate(results.items()):
        vals = [r[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=name, color=colors_bar[i], edgecolor="white")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"])
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "19_model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("   Saved: 19_model_comparison.png")

    # ── Save Models ──────────────────────────────────────────────────────
    print("\n--- Saving Models ---")
    joblib.dump(rf_base, os.path.join(MODEL_DIR, "rf_baseline.pkl"))
    joblib.dump(lr_model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    joblib.dump(rf_improved, os.path.join(MODEL_DIR, "rf_improved.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
    joblib.dump(results, os.path.join(MODEL_DIR, "results.pkl"))

    # Save the best params for the Streamlit app
    joblib.dump(grid_search.best_params_, os.path.join(MODEL_DIR, "best_params.pkl"))

    print("   ✅ All models and artifacts saved to 'models/' directory.")

    return results, feature_names, scaler, label_encoders


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n" + "🔷" * 35)
    print("   CREDIT CARD FRAUD DETECTION – CMP7005 ASSIGNMENT")
    print("🔷" * 35 + "\n")

    # Task 1
    df = task1_data_handling()

    # Task 2a
    task2a_data_understanding(df)

    # Task 2b
    df = task2b_preprocessing(df)

    # Save cleaned dataset
    cleaned_path = os.path.join(BASE_DIR, "cleaned_dataset.csv")
    df.to_csv(cleaned_path, index=False)
    print(f"\n✅ Cleaned dataset saved to: {cleaned_path}")

    # Task 2c
    task2c_analysis_visualisation(df)

    # Task 3
    task3_model_building(df)

    print("\n" + "🔷" * 35)
    print("   ALL TASKS COMPLETED SUCCESSFULLY!")
    print("🔷" * 35 + "\n")
