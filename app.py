"""
CMP7005 Assignment – Credit Card Fraud Detection
=================================================
Task 4: Multi-page Streamlit Application with GUI
Pages:
  1. Data Overview
  2. Exploratory Data Analysis (EDA)
  3. Modelling & Prediction
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA1_PATH = os.path.join(BASE_DIR, "Credit_Card_Dataset_2025_Sept_1.csv")
DATA2_PATH = os.path.join(BASE_DIR, "Credit_Card_Dataset_2025_Sept_2.csv")
CLEANED_PATH = os.path.join(BASE_DIR, "cleaned_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")


# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }

    .main-header p {
        color: rgba(255,255,255,0.85);
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-3px);
    }

    .metric-card h3 {
        color: #2c3e50;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-card .value {
        color: #667eea;
        font-size: 2rem;
        font-weight: 700;
    }

    .insight-box {
        background: linear-gradient(135deg, #e8f8f5 0%, #d5f5e3 100%);
        border-left: 4px solid #27ae60;
        padding: 1rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
        font-size: 0.95rem;
    }

    .warning-box {
        background: linear-gradient(135deg, #fdf2e9 0%, #fce4d6 100%);
        border-left: 4px solid #e67e22;
        padding: 1rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }

    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
    }

    .stSidebar [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        color: white;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .prediction-result {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
    }

    .fraud {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }

    .not-fraud {
        background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_raw_data():
    """Load and merge the raw datasets."""
    df1 = pd.read_csv(DATA1_PATH, na_values=["NA", "na", "N/A", ""])
    df2 = pd.read_csv(DATA2_PATH, na_values=["NA", "na", "N/A", ""])
    # Drop unnamed index
    if df1.columns[0] == "" or "Unnamed" in str(df1.columns[0]):
        df1 = df1.drop(columns=[df1.columns[0]])
    # Merge
    df = pd.merge(df1, df2, left_on="ID", right_on="User", how="inner")
    df = df.drop(columns=["User"])
    return df1, df2, df


@st.cache_data
def load_cleaned_data():
    """Load the cleaned/preprocessed dataset."""
    if os.path.exists(CLEANED_PATH):
        return pd.read_csv(CLEANED_PATH)
    return None


@st.cache_resource
def load_models():
    """Load saved models and artifacts."""
    models = {}
    try:
        models["rf_baseline"] = joblib.load(os.path.join(MODEL_DIR, "rf_baseline.pkl"))
        models["logistic_regression"] = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
        models["rf_improved"] = joblib.load(os.path.join(MODEL_DIR, "rf_improved.pkl"))
        models["scaler"] = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        models["label_encoders"] = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
        models["feature_names"] = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
        models["results"] = joblib.load(os.path.join(MODEL_DIR, "results.pkl"))
        models["best_params"] = joblib.load(os.path.join(MODEL_DIR, "best_params.pkl"))
    except Exception as e:
        st.warning(f"⚠️ Could not load some models. Please run `python main.py` first. Error: {e}")
    return models


# ─── Sidebar Navigation ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Navigation")
    st.markdown("---")
    page = st.radio(
        "Select a Section",
        ["🏠 Data Overview", "📊 Exploratory Data Analysis", "🤖 Modelling & Prediction"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    **CMP7005 Assignment**

    Credit Card Fraud Detection using
    Machine Learning.

    - 25,134 transactions
    - 19 features
    - Binary classification
    - ~1.6% fraud rate
    """)
    st.markdown("---")
    st.markdown("*Built with Streamlit*")


# =============================================================================
# PAGE 1: DATA OVERVIEW
# =============================================================================
if page == "🏠 Data Overview":
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Credit Card Fraud Detection</h1>
        <p>CMP7005 Assignment – Data Overview & Understanding</p>
    </div>
    """, unsafe_allow_html=True)

    df1, df2, df_merged = load_raw_data()

    # ── Key Metrics ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(df_merged):,}")
    with col2:
        st.metric("Features", f"{df_merged.shape[1]}")
    with col3:
        fraud_count = df_merged["TARGET"].sum()
        st.metric("Fraudulent", f"{int(fraud_count):,}")
    with col4:
        fraud_pct = fraud_count / len(df_merged) * 100
        st.metric("Fraud Rate", f"{fraud_pct:.2f}%")

    st.markdown("---")

    # ── Dataset Details ──────────────────────────────────────────────────
    st.subheader("📁 Dataset Structure")

    tab1, tab2, tab3 = st.tabs(["Merged Dataset", "Dataset 1 (Raw)", "Dataset 2 (Raw)"])

    with tab1:
        st.markdown("**Merged Dataset** – Combined using `ID` (Dataset 1) = `User` (Dataset 2)")
        st.dataframe(df_merged.head(20), use_container_width=True, height=400)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Data Types**")
            dtype_df = pd.DataFrame({
                "Column": df_merged.columns,
                "Type": df_merged.dtypes.values,
                "Non-Null Count": df_merged.count().values,
                "Unique Values": [df_merged[c].nunique() for c in df_merged.columns],
            })
            st.dataframe(dtype_df, use_container_width=True, hide_index=True)

        with c2:
            st.markdown("**Missing Values**")
            missing = df_merged.isnull().sum()
            missing_pct = (missing / len(df_merged) * 100).round(2)
            miss_df = pd.DataFrame({
                "Column": missing.index,
                "Missing Count": missing.values,
                "Missing %": missing_pct.values,
            })
            miss_df = miss_df[miss_df["Missing Count"] > 0].reset_index(drop=True)
            if len(miss_df) > 0:
                st.dataframe(miss_df, use_container_width=True, hide_index=True)
            else:
                st.success("No missing values!")

    with tab2:
        st.markdown(f"**Dataset 1** – Shape: {df1.shape}")
        st.dataframe(df1.head(15), use_container_width=True)

    with tab3:
        st.markdown(f"**Dataset 2** – Shape: {df2.shape}")
        st.dataframe(df2.head(15), use_container_width=True)

    st.markdown("---")

    # ── Statistical Summary ──────────────────────────────────────────────
    st.subheader("📈 Statistical Summary")
    tab_num, tab_cat = st.tabs(["Numerical Features", "Categorical Features"])
    with tab_num:
        st.dataframe(df_merged.describe().round(2), use_container_width=True)
    with tab_cat:
        cat_cols = df_merged.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            st.dataframe(df_merged[cat_cols].describe(), use_container_width=True)

    # ── Target Distribution ──────────────────────────────────────────────
    st.subheader("🎯 Target Variable Distribution")
    c1, c2 = st.columns([1, 1])
    with c1:
        target_counts = df_merged["TARGET"].value_counts().reset_index()
        target_counts.columns = ["Target", "Count"]
        target_counts["Label"] = target_counts["Target"].map({0: "Non-Fraud", 1: "Fraud"})
        fig = px.pie(target_counts, values="Count", names="Label",
                     color="Label",
                     color_discrete_map={"Non-Fraud": "#2ecc71", "Fraud": "#e74c3c"},
                     hole=0.4)
        fig.update_layout(height=400, margin=dict(t=30, b=0, l=0, r=0))
        fig.update_traces(textinfo="percent+value", textfont_size=14)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("""
        <div class="insight-box">
            <strong>📊 Key Insight:</strong><br>
            The dataset is <strong>highly imbalanced</strong>. Only <strong>1.67%</strong> of
            transactions are fraudulent (420 out of 25,134). This imbalance requires special
            handling techniques such as <strong>SMOTE</strong> (Synthetic Minority Over-sampling
            Technique) during model training to prevent the model from being biased toward the
            majority class.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ Note on Accuracy:</strong><br>
            A naive classifier that always predicts "Non-Fraud" would achieve ~98.3% accuracy.
            Therefore, we must focus on metrics like <strong>Recall</strong>, <strong>F1-Score</strong>,
            and <strong>ROC-AUC</strong> rather than just accuracy.
        </div>
        """, unsafe_allow_html=True)


    # ── Column Descriptions ──────────────────────────────────────────────
    st.subheader("📋 Feature Descriptions")
    feature_info = pd.DataFrame({
        "Feature": ["ID", "GENDER", "CAR", "REALITY", "NO_OF_CHILD", "INCOME",
                     "INCOME_TYPE", "EDUCATION_TYPE", "FAMILY_TYPE", "HOUSE_TYPE",
                     "FLAG_MOBIL", "WORK_PHONE", "PHONE", "E_MAIL", "FAMILY SIZE",
                     "BEGIN_MONTH", "AGE", "YEARS_EMPLOYED", "TARGET"],
        "Type": ["Numerical", "Categorical", "Categorical", "Categorical", "Numerical",
                 "Numerical", "Categorical", "Categorical", "Categorical", "Categorical",
                 "Numerical", "Numerical", "Numerical", "Numerical", "Numerical",
                 "Numerical", "Numerical", "Numerical", "Binary"],
        "Description": [
            "Unique identifier for each transaction",
            "Gender of the cardholder (M/F)",
            "Whether the cardholder owns a car (Y/N)",
            "Whether the cardholder owns property (Y/N)",
            "Number of children",
            "Annual income of the cardholder",
            "Type of income (Working, Commercial associate, etc.)",
            "Education level of the cardholder",
            "Family status (Married, Single, etc.)",
            "Type of housing (House/apartment, Rented, etc.)",
            "Mobile phone flag (1=Yes)",
            "Work phone flag (1=Yes, 0=No)",
            "Home phone flag (1=Yes, 0=No)",
            "Email flag (1=Yes, 0=No)",
            "Number of family members",
            "Month of account opening (relative)",
            "Age of the cardholder in years",
            "Years of employment",
            "Fraud indicator (1=Fraud, 0=Not Fraud)",
        ],
    })
    st.dataframe(feature_info, use_container_width=True, hide_index=True, height=700)


# =============================================================================
# PAGE 2: EXPLORATORY DATA ANALYSIS
# =============================================================================
elif page == "📊 Exploratory Data Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Exploratory Data Analysis</h1>
        <p>Interactive visualisations and statistical insights</p>
    </div>
    """, unsafe_allow_html=True)

    df_clean = load_cleaned_data()
    if df_clean is None:
        _, _, df_clean = load_raw_data()
        st.warning("⚠️ Cleaned dataset not found. Showing raw merged data. Run `python main.py` first.")

    # ── Analysis Type Selector ───────────────────────────────────────────
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis", "Saved Plots Gallery"],
    )

    if analysis_type == "Univariate Analysis":
        st.subheader("📊 Univariate Analysis")
        st.markdown("Examining individual feature distributions.")

        col1, col2 = st.columns(2)

        with col1:
            # Target distribution
            st.markdown("#### Target Distribution")
            tc = df_clean["TARGET"].value_counts().reset_index()
            tc.columns = ["Target", "Count"]
            tc["Label"] = tc["Target"].map({0: "Non-Fraud", 1: "Fraud"})
            fig = px.bar(tc, x="Label", y="Count", color="Label",
                         color_discrete_map={"Non-Fraud": "#2ecc71", "Fraud": "#e74c3c"},
                         text="Count")
            fig.update_layout(height=400, showlegend=False)
            fig.update_traces(textposition="outside", textfont_size=14)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Gender distribution
            st.markdown("#### Gender Distribution")
            gc = df_clean["GENDER"].value_counts().reset_index()
            gc.columns = ["Gender", "Count"]
            fig = px.bar(gc, x="Gender", y="Count", color="Gender",
                         color_discrete_map={"M": "#3498db", "F": "#e91e63"},
                         text="Count")
            fig.update_layout(height=400, showlegend=False)
            fig.update_traces(textposition="outside", textfont_size=14)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            # Income distribution
            st.markdown("#### Income Distribution")
            fig = px.histogram(df_clean, x="INCOME", nbins=50,
                               color_discrete_sequence=["#9b59b6"])
            fig.add_vline(x=df_clean["INCOME"].mean(), line_dash="dash", line_color="red",
                          annotation_text=f"Mean: {df_clean['INCOME'].mean():,.0f}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            # Age distribution
            st.markdown("#### Age Distribution")
            fig = px.histogram(df_clean, x="AGE", nbins=30,
                               color_discrete_sequence=["#1abc9c"])
            fig.add_vline(x=df_clean["AGE"].mean(), line_dash="dash", line_color="red",
                          annotation_text=f"Mean: {df_clean['AGE'].mean():.1f}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        col5, col6 = st.columns(2)
        with col5:
            # Income Type
            st.markdown("#### Income Type Distribution")
            it_c = df_clean["INCOME_TYPE"].value_counts().reset_index()
            it_c.columns = ["Type", "Count"]
            fig = px.bar(it_c, y="Type", x="Count", orientation="h",
                         color="Type", color_discrete_sequence=px.colors.qualitative.Set2,
                         text="Count")
            fig.update_layout(height=400, showlegend=False)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            # Family Type
            st.markdown("#### Family Type Distribution")
            ft_c = df_clean["FAMILY_TYPE"].value_counts().reset_index()
            ft_c.columns = ["Type", "Count"]
            fig = px.pie(ft_c, values="Count", names="Type",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <strong>💡 Univariate Insights:</strong>
            <ul>
                <li>The target is heavily imbalanced (~98.3% non-fraud).</li>
                <li>Female cardholders outnumber males in the dataset.</li>
                <li>Income is right-skewed with most values between 100K-250K.</li>
                <li>Age is roughly normally distributed, centred around 35-45 years.</li>
                <li>Most cardholders are classified as 'Working' income type.</li>
                <li>Married individuals form the largest family type group.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    elif analysis_type == "Bivariate Analysis":
        st.subheader("📊 Bivariate Analysis")
        st.markdown("Examining relationships between pairs of variables.")

        col1, col2 = st.columns(2)

        with col1:
            # Income by Target
            st.markdown("#### Income by Fraud Status")
            fig = px.box(df_clean, x="TARGET", y="INCOME",
                         color="TARGET",
                         color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                         labels={"TARGET": "Fraud Status"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Age by Target
            st.markdown("#### Age by Fraud Status")
            fig = px.box(df_clean, x="TARGET", y="AGE",
                         color="TARGET",
                         color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                         labels={"TARGET": "Fraud Status"})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            # Gender vs Target
            st.markdown("#### Fraud Rate by Gender")
            ct = pd.crosstab(df_clean["GENDER"], df_clean["TARGET"], normalize="index").reset_index()
            ct.columns = ["Gender", "Non-Fraud", "Fraud"]
            ct_melt = ct.melt(id_vars="Gender", var_name="Status", value_name="Proportion")
            fig = px.bar(ct_melt, x="Gender", y="Proportion", color="Status",
                         barmode="group",
                         color_discrete_map={"Non-Fraud": "#2ecc71", "Fraud": "#e74c3c"},
                         text=ct_melt["Proportion"].apply(lambda x: f"{x:.1%}"))
            fig.update_layout(height=400)
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            # Family Type vs Target
            st.markdown("#### Fraud Rate by Family Type")
            ct2 = pd.crosstab(df_clean["FAMILY_TYPE"], df_clean["TARGET"], normalize="index").reset_index()
            ct2.columns = ["Family Type", "Non-Fraud", "Fraud"]
            fig = px.bar(ct2, x="Family Type", y="Fraud",
                         color_discrete_sequence=["#e74c3c"],
                         text=ct2["Fraud"].apply(lambda x: f"{x:.1%}"))
            fig.update_layout(height=400, yaxis_title="Fraud Proportion")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        # Scatter: Age vs Income
        st.markdown("#### Age vs Income (by Fraud Status)")
        sample = df_clean.sample(min(5000, len(df_clean)), random_state=42)
        fig = px.scatter(sample, x="AGE", y="INCOME", color="TARGET",
                         color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                         opacity=0.5, labels={"TARGET": "Fraud Status"})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <strong>💡 Bivariate Insights:</strong>
            <ul>
                <li>Income distributions are similar between fraud and non-fraud groups, suggesting income alone is not a strong fraud predictor.</li>
                <li>Age distributions also overlap significantly between classes.</li>
                <li>Males show a slightly higher fraud rate compared to females.</li>
                <li>Civil marriage and Separated family types show slightly higher fraud proportions.</li>
                <li>No single feature can clearly separate fraud from non-fraud cases, highlighting the need for multivariate models.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    elif analysis_type == "Multivariate Analysis":
        st.subheader("📊 Multivariate Analysis")
        st.markdown("Examining complex relationships across multiple variables.")

        # Correlation Heatmap
        st.markdown("#### Correlation Heatmap (Numerical Features)")
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        corr = df_clean[num_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdYlBu_r",
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            zmin=-1, zmax=1,
        ))
        fig.update_layout(height=700, width=900, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

        # Income by House Type and Target
        st.markdown("#### Income Distribution by House Type & Fraud Status")
        fig = px.box(df_clean, x="HOUSE_TYPE", y="INCOME", color="TARGET",
                     color_discrete_map={0: "#2ecc71", 1: "#e74c3c"})
        fig.update_layout(height=500, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        # Education Type vs Income by Target
        st.markdown("#### Income by Education Type & Fraud Status")
        fig = px.box(df_clean, x="EDUCATION_TYPE", y="INCOME", color="TARGET",
                     color_discrete_map={0: "#2ecc71", 1: "#e74c3c"})
        fig.update_layout(height=500, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
            <strong>💡 Multivariate Insights:</strong>
            <ul>
                <li>NO_OF_CHILD and FAMILY SIZE show a strong positive correlation (0.89), which is expected.</li>
                <li>AGE and YEARS_EMPLOYED are positively correlated – older people tend to have more work experience.</li>
                <li>FLAG_MOBIL has zero correlation with all features (constant value = 1).</li>
                <li>The TARGET variable has very weak correlations with individual features, confirming that no single feature is a strong predictor of fraud.</li>
                <li>This suggests that fraud detection requires complex models that can capture non-linear interactions between features.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    elif analysis_type == "Saved Plots Gallery":
        st.subheader("🖼️ Saved Plots Gallery")
        st.markdown("All plots generated during the analysis (from `main.py`).")

        if os.path.exists(PLOT_DIR):
            plot_files = sorted([f for f in os.listdir(PLOT_DIR) if f.endswith(".png")])
            if plot_files:
                for i in range(0, len(plot_files), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        idx = i + j
                        if idx < len(plot_files):
                            with col:
                                plot_name = plot_files[idx].replace(".png", "").replace("_", " ").title()
                                st.markdown(f"**{plot_name}**")
                                st.image(os.path.join(PLOT_DIR, plot_files[idx]), use_container_width=True)
            else:
                st.info("No plots found. Run `python main.py` first to generate plots.")
        else:
            st.info("📁 Plots directory not found. Run `python main.py` to generate all analysis plots.")


# =============================================================================
# PAGE 3: MODELLING & PREDICTION
# =============================================================================
elif page == "🤖 Modelling & Prediction":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Modelling & Prediction</h1>
        <p>Select a model, provide inputs, and predict fraud likelihood</p>
    </div>
    """, unsafe_allow_html=True)

    models = load_models()

    if not models or "rf_baseline" not in models:
        st.error("❌ Models not found! Please run `python main.py` first to train and save models.")
        st.stop()

    # ── Model Performance Comparison ─────────────────────────────────────
    st.subheader("📊 Model Performance Comparison")

    results = models.get("results", {})
    if results:
        perf_data = []
        for name, r in results.items():
            perf_data.append({
                "Model": name,
                "Accuracy": f"{r['accuracy']:.4f}",
                "Precision": f"{r['precision']:.4f}",
                "Recall": f"{r['recall']:.4f}",
                "F1-Score": f"{r['f1']:.4f}",
                "ROC-AUC": f"{r['roc_auc']:.4f}",
            })
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True, hide_index=True)

        # Performance bar chart
        fig = go.Figure()
        metrics_list = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        colors = ["#3498db", "#e74c3c", "#2ecc71"]
        for i, (name, r) in enumerate(results.items()):
            fig.add_trace(go.Bar(
                name=name,
                x=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                y=[r[m] for m in metrics_list],
                marker_color=colors[i % len(colors)],
                text=[f"{r[m]:.3f}" for m in metrics_list],
                textposition="outside",
            ))
        fig.update_layout(
            barmode="group", height=450,
            title="Model Performance Comparison",
            yaxis_range=[0, 1.15],
            yaxis_title="Score",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Observations
        st.markdown("""
        <div class="insight-box">
            <strong>📊 Observations on Model Metrics:</strong>
            <ul>
                <li><strong>Accuracy appears high for all models</strong> due to class imbalance – a model predicting all as non-fraud would achieve ~98% accuracy. This is misleading.</li>
                <li><strong>Precision vs Recall trade-off (conflicting observation):</strong> High precision means fewer false alarms but may miss real fraud. High recall catches more fraud but increases false positives. These metrics often conflict.</li>
                <li><strong>F1-Score</strong> balances precision and recall, making it a better metric for imbalanced datasets.</li>
                <li><strong>ROC-AUC</strong> provides a threshold-independent evaluation of model performance.</li>
                <li><strong>SMOTE</strong> was used to balance the training data, which typically improves recall at some cost to precision.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Model Improvement Explanation ────────────────────────────────────
    st.subheader("🚀 Model Improvement")

    best_params = models.get("best_params", {})
    col_imp1, col_imp2 = st.columns([1, 1])
    with col_imp1:
        st.markdown("""
        <div class="model-card">
            <h4>🔧 Improvement Technique: Hyperparameter Tuning (GridSearchCV)</h4>
            <p>The baseline Random Forest was improved using <strong>GridSearchCV</strong> with
            <strong>Stratified 3-Fold Cross-Validation</strong> optimising for <strong>F1-Score</strong>:</p>
            <ul>
                <li><code>n_estimators</code>: [100, 200, 300] – Number of trees</li>
                <li><code>max_depth</code>: [10, 20, None] – Maximum tree depth</li>
                <li><code>min_samples_split</code>: [2, 5] – Min samples to split a node</li>
                <li><code>min_samples_leaf</code>: [1, 2] – Min samples at leaf node</li>
            </ul>
            <p>This exhaustive search explores <strong>36 parameter combinations</strong> × 3 folds = <strong>108 model fits</strong>
            to find the optimal configuration.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_imp2:
        st.markdown("**Best Parameters Found:**")
        if best_params:
            for k, v in best_params.items():
                st.markdown(f"- `{k}`: **{v}**")
        else:
            st.info("Run `python main.py` to see best parameters.")

        st.markdown("""
        <div class="warning-box">
            <strong>Why GridSearchCV?</strong><br>
            Hyperparameter tuning systematically explores the parameter space to find the
            combination that maximises model performance. Unlike manual tuning, it evaluates
            every combination using cross-validation, ensuring the result is robust and not
            overfit to a specific train/test split.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Prediction Interface ─────────────────────────────────────────────
    st.subheader("🔮 Make a Prediction")

    model_choice = st.selectbox(
        "Select Model",
        ["Random Forest (Baseline)", "Logistic Regression", "Random Forest (Improved) ⭐"],
        index=2,
    )

    model_key_map = {
        "Random Forest (Baseline)": "rf_baseline",
        "Logistic Regression": "logistic_regression",
        "Random Forest (Improved) ⭐": "rf_improved",
    }

    selected_model = models[model_key_map[model_choice]]
    scaler = models["scaler"]
    label_encoders = models["label_encoders"]
    feature_names = models["feature_names"]

    st.markdown("**Enter transaction details:**")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        gender = st.selectbox("Gender", ["M", "F"])
        car = st.selectbox("Car Owner", ["Y", "N"])
        reality = st.selectbox("Property Owner", ["Y", "N"])
        no_children = st.number_input("Number of Children", 0, 10, 0)
        family_size = st.number_input("Family Size", 1, 15, 2)
        income = st.number_input("Annual Income", 10000, 2000000, 150000, step=10000)

    with col_b:
        income_type = st.selectbox("Income Type", [
            "Working", "Commercial associate", "State servant", "Pensioner", "Student"
        ])
        education = st.selectbox("Education Type", [
            "Secondary / secondary special", "Higher education",
            "Incomplete higher", "Lower secondary", "Academic degree"
        ])
        family_type = st.selectbox("Family Type", [
            "Married", "Single / not married", "Civil marriage",
            "Separated", "Widow"
        ])
        house_type = st.selectbox("House Type", [
            "House / apartment", "Municipal apartment", "With parents",
            "Rented apartment", "Co-op apartment", "Office apartment"
        ])

    with col_c:
        work_phone = st.selectbox("Work Phone", [0, 1])
        phone = st.selectbox("Home Phone", [0, 1])
        email = st.selectbox("Email", [0, 1])
        begin_month = st.number_input("Begin Month", 0, 60, 12)
        age = st.number_input("Age", 18, 80, 35)
        years_employed = st.number_input("Years Employed", 0, 50, 5)

    if st.button("🔍 Predict Fraud", type="primary", use_container_width=True):
        # Prepare input
        binary_map = {"M": 1, "F": 0, "Y": 1, "N": 0}
        input_data = {
            "GENDER": binary_map[gender],
            "CAR": binary_map[car],
            "REALITY": binary_map[reality],
            "NO_OF_CHILD": no_children,
            "FAMILY_TYPE": 0,
            "HOUSE_TYPE": 0,
            "WORK_PHONE": work_phone,
            "PHONE": phone,
            "E_MAIL": email,
            "FAMILY SIZE": family_size,
            "BEGIN_MONTH": begin_month,
            "AGE": age,
            "YEARS_EMPLOYED": years_employed,
            "INCOME": income,
            "INCOME_TYPE": 0,
            "EDUCATION_TYPE": 0,
            "INCOME_PER_MEMBER": income / max(family_size, 1),
            "AGE_GROUP": 0,
            "EMPLOYMENT_RATIO": years_employed / max(age, 1),
        }

        # Encode categoricals
        cat_encode_map = {
            "FAMILY_TYPE": family_type,
            "HOUSE_TYPE": house_type,
            "INCOME_TYPE": income_type,
            "EDUCATION_TYPE": education,
        }
        for col, val in cat_encode_map.items():
            le = label_encoders.get(col)
            if le and val in le.classes_:
                input_data[col] = le.transform([val])[0]
            else:
                input_data[col] = 0

        # Encode AGE_GROUP
        if age <= 25:
            ag = "Young"
        elif age <= 35:
            ag = "Adult"
        elif age <= 45:
            ag = "Middle"
        elif age <= 55:
            ag = "Senior"
        else:
            ag = "Elderly"
        le_ag = label_encoders.get("AGE_GROUP")
        if le_ag and ag in le_ag.classes_:
            input_data["AGE_GROUP"] = le_ag.transform([ag])[0]

        # Build feature vector – ensure correct order
        feature_vector = []
        for f in feature_names:
            feature_vector.append(input_data.get(f, 0))

        feature_arr = np.array(feature_vector).reshape(1, -1)
        feature_scaled = scaler.transform(feature_arr)

        prediction = selected_model.predict(feature_scaled)[0]
        probability = selected_model.predict_proba(feature_scaled)[0]

        st.markdown("---")

        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-result fraud">
                ⚠️ FRAUD DETECTED<br>
                <span style="font-size: 1rem;">Probability: {probability[1]:.1%} fraud | {probability[0]:.1%} legitimate</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-result not-fraud">
                ✅ LEGITIMATE TRANSACTION<br>
                <span style="font-size: 1rem;">Probability: {probability[0]:.1%} legitimate | {probability[1]:.1%} fraud</span>
            </div>
            """, unsafe_allow_html=True)

        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability[1] * 100,
            title={"text": "Fraud Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#e74c3c" if prediction == 1 else "#2ecc71"},
                "steps": [
                    {"range": [0, 30], "color": "#d5f5e3"},
                    {"range": [30, 70], "color": "#fdebd0"},
                    {"range": [70, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ── Saved Model Plots ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Model Evaluation Plots")

    plot_tabs = st.tabs(["ROC Curves", "Confusion Matrices", "Feature Importance", "Performance Comparison"])

    with plot_tabs[0]:
        roc_path = os.path.join(PLOT_DIR, "16_roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.info("Run `python main.py` to generate this plot.")

    with plot_tabs[1]:
        cm_path = os.path.join(PLOT_DIR, "17_confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.info("Run `python main.py` to generate this plot.")

    with plot_tabs[2]:
        fi_path = os.path.join(PLOT_DIR, "18_feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)
        else:
            st.info("Run `python main.py` to generate this plot.")

    with plot_tabs[3]:
        mc_path = os.path.join(PLOT_DIR, "19_model_comparison.png")
        if os.path.exists(mc_path):
            st.image(mc_path, use_container_width=True)
        else:
            st.info("Run `python main.py` to generate this plot.")
