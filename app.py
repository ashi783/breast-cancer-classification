"""
Breast Cancer Classification - Streamlit Web Application

This app provides an interactive interface to explore different classification models
trained on the Breast Cancer Wisconsin dataset.

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
Source: UCI Machine Learning Repository
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Force dark theme regardless of system settings */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
        color: #e0e0e0 !important;
    }
    
    /* Force all text to be light colored */
    .stApp, .stApp p, .stApp span, .stApp li, .stApp label,
    .stApp .stMarkdown, .stApp .stMarkdown p,
    .stApp div[data-testid="stText"],
    .stApp .element-container,
    .stApp .stAlert p,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #e0e0e0 !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e) !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e0e0e0 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a2e !important;
        border-radius: 10px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #a0a0a0 !important;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        color: #00d2ff !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        color: #00d2ff !important;
        background-color: #1a1a2e !important;
    }
    details summary span {
        color: #00d2ff !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #303050 !important;
    }
    
    /* Selectbox and input styling */
    .stSelectbox label, .stFileUploader label {
        color: #e0e0e0 !important;
    }
    
    /* Blockquote styling */
    blockquote {
        border-left: 4px solid #ff416c !important;
        background-color: rgba(255, 65, 108, 0.1) !important;
        padding: 10px 15px !important;
        border-radius: 0 8px 8px 0 !important;
    }
    blockquote p {
        color: #e0e0e0 !important;
    }
    
    /* Metric cards from streamlit */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00d2ff !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
    }
    
    /* Info, warning, success, error boxes */
    .stAlert {
        background-color: rgba(26, 26, 46, 0.8) !important;
        border-radius: 10px !important;
    }
    
    .main-header {
        font-size: 72px !important;
        font-weight: 900 !important;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00d2ff) !important;
        background-size: 200% auto;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center;
        margin-bottom: 5px;
        padding: 30px 0 10px 0;
        letter-spacing: 3px;
        line-height: 1.1;
    }
    .sub-header {
        font-size: 26px !important;
        font-weight: bold !important;
        color: #00d2ff !important;
        margin-top: 20px;
        border-left: 4px solid #00d2ff;
        padding-left: 15px;
    }
    .tagline {
        font-size: 20px !important;
        color: #b0b0b0 !important;
        text-align: center;
        margin-bottom: 35px;
        font-style: italic;
        letter-spacing: 1px;
    }
    .metric-container {
        background: linear-gradient(135deg, #1a1a2e, #16213e) !important;
        border: 1px solid #00d2ff;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.2);
        transition: transform 0.3s ease;
    }
    .metric-container:hover {
        transform: translateY(-5px);
    }
    .metric-value {
        font-size: 36px !important;
        font-weight: bold !important;
        color: #00d2ff !important;
    }
    .metric-label {
        font-size: 14px !important;
        color: #a0a0a0 !important;
        margin-top: 5px;
    }
    .priority-high {
        background: linear-gradient(90deg, #ff416c, #ff4b2b) !important;
        color: white !important;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
    .priority-medium {
        background: linear-gradient(90deg, #f7971e, #ffd200) !important;
        color: #1a1a2e !important;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
    .winner-banner {
        background: linear-gradient(90deg, #FFD700, #FFA500) !important;
        color: #1a1a2e !important;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(255, 215, 0, 0.4);
    }
    .insight-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e) !important;
        border-left: 4px solid #34e89e;
        padding: 15px 20px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
        color: #e0e0e0 !important;
    }
    .insight-box b {
        color: #ffffff !important;
    }
    .param-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e) !important;
        border: 1px solid #3a7bd5;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(58, 123, 213, 0.15);
    }
    .param-card h4 {
        color: #00d2ff !important;
        margin-bottom: 15px;
        font-size: 18px;
    }
    .param-item {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #303050;
        color: #e0e0e0 !important;
    }
    .param-name {
        color: #a0a0a0 !important;
        font-size: 14px;
    }
    .param-value {
        color: #34e89e !important;
        font-weight: bold;
        font-size: 14px;
    }
    .tuning-badge {
        background: linear-gradient(90deg, #34e89e, #0f9b0f) !important;
        color: white !important;
        padding: 3px 12px;
        border-radius: 15px;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
        margin: 3px;
    }
    .default-badge {
        background: linear-gradient(90deg, #636e72, #b2bec3) !important;
        color: white !important;
        padding: 3px 12px;
        border-radius: 15px;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
        margin: 3px;
    }
    .prediction-benign {
        background: linear-gradient(135deg, #0f3443, #34e89e) !important;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(52, 232, 158, 0.3);
        color: white !important;
    }
    .prediction-malignant {
        background: linear-gradient(135deg, #3d0000, #ff416c) !important;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(255, 65, 108, 0.3);
        color: white !important;
    }
    .upload-section {
        background: linear-gradient(135deg, #1a1a2e, #16213e) !important;
        border: 2px dashed #00d2ff;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============================
# Data Loading Functions
# ============================

@st.cache_resource
def load_models():
    """Load all trained models and scaler."""
    models = {}
    model_names = [
        'logistic_regression', 'decision_tree', 'k-nearest_neighbors',
        'naive_bayes', 'random_forest', 'xgboost'
    ]
    for name in model_names:
        try:
            with open(f'model/{name}_model.pkl', 'rb') as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            st.error(f"Model file not found: {name}_model.pkl")
    try:
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        st.error("Scaler file not found!")
        scaler = None
    return models, scaler


@st.cache_data
def load_metrics():
    """Load model performance metrics."""
    try:
        return pd.read_csv('model/model_metrics.csv')
    except FileNotFoundError:
        st.error("Metrics file not found!")
        return None


@st.cache_data
def load_hyperparameters():
    """Load best hyperparameters from tuning."""
    try:
        return pd.read_csv('model/best_hyperparameters.csv')
    except FileNotFoundError:
        return None


@st.cache_data
def load_dataset():
    """Load the breast cancer dataset."""
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names, data.target_names


@st.cache_data
def generate_test_csv():
    """Generate a sample test CSV for download."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    test_df = X_test.copy()
    test_df['target'] = y_test.values
    return test_df


# ============================
# Visualization Functions
# ============================

def plot_confusion_matrix(y_true, y_pred, target_names):
    """Plot styled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 20, 'weight': 'bold'},
                linewidths=2, linecolor='#16213e')
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', color='#00d2ff')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold', color='#00d2ff')
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', color='#00d2ff', pad=20)
    ax.tick_params(colors='#a0a0a0')
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color('#a0a0a0')
    cbar.ax.tick_params(colors='#a0a0a0')
    plt.tight_layout()
    return fig


def display_classification_report(y_true, y_pred, target_names):
    """Display classification report as formatted table."""
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    return pd.DataFrame(report).transpose().round(4)


def get_best_models(metrics_df):
    """Get best model for each metric with tiebreakers."""
    metrics_list = ['Recall', 'Accuracy', 'AUC Score', 'Precision', 'F1 Score', 'MCC']
    tiebreakers = ['AUC Score', 'Accuracy', 'F1 Score']
    best_models = {}
    for metric in metrics_list:
        max_val = metrics_df[metric].max()
        tied = metrics_df[metrics_df[metric] == max_val]
        if len(tied) > 1:
            for tb in tiebreakers:
                if (tb != metric) and (tb in tied.columns):
                    best_idx = tied[tb].idxmax()
                    tied = tied.loc[[best_idx]]
                    break
        best_idx = tied.index[0]
        best_models[metric] = {
            'model': metrics_df.loc[best_idx, 'Model'],
            'score': metrics_df.loc[best_idx, metric]
        }
    return best_models


def plot_radar_chart(metrics_df, model_name):
    """Plot radar chart for a specific model."""
    metrics_cols = ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC']
    model_data = metrics_df[metrics_df['Model'] == model_name][metrics_cols].values.flatten()
    angles = np.linspace(0, 2 * np.pi, len(metrics_cols), endpoint=False).tolist()
    model_data = np.concatenate((model_data, [model_data[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.plot(angles, model_data, 'o-', linewidth=2, color='#00d2ff', label=model_name)
    ax.fill(angles, model_data, alpha=0.25, color='#00d2ff')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_cols, fontsize=11, fontweight='bold', color='#e0e0e0')
    ax.set_ylim(0, 1.05)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(colors='#a0a0a0')
    ax.grid(color='#303050', linewidth=0.5)
    ax.spines['polar'].set_color('#303050')
    ax.set_title(f'{model_name} - Performance Profile', fontsize=14,
                 fontweight='bold', color='#00d2ff', pad=30)
    plt.tight_layout()
    return fig


def plot_comparison_chart(metrics_df, metric_to_plot):
    """Plot bar chart comparison across models."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    colors = ['#00d2ff', '#3a7bd5', '#34e89e', '#ff416c', '#f7971e', '#9b59b6']
    bars = ax.bar(metrics_df['Model'], metrics_df[metric_to_plot], color=colors,
                  edgecolor='#16213e', linewidth=1.5, width=0.6)
    best_idx = metrics_df[metric_to_plot].idxmax()
    bars[best_idx].set_edgecolor('#FFD700')
    bars[best_idx].set_linewidth(3)
    ax.set_xlabel('Model', fontsize=13, fontweight='bold', color='#a0a0a0')
    ax.set_ylabel(metric_to_plot, fontsize=13, fontweight='bold', color='#a0a0a0')
    ax.set_title(f'{metric_to_plot} Comparison Across Models',
                fontsize=16, fontweight='bold', color='#00d2ff', pad=15)
    min_val = metrics_df[metric_to_plot].min()
    ax.set_ylim([max(0, min_val - 0.05), 1.05])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
               f'{height:.4f}', ha='center', va='bottom', fontweight='bold',
               color='#e0e0e0', fontsize=10)
    ax.tick_params(colors='#a0a0a0')
    ax.spines['bottom'].set_color('#303050')
    ax.spines['left'].set_color('#303050')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, color='#303050')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    return fig


def get_model_params(params_df, model_name):
    """Get hyperparameters for a specific model."""
    if params_df is None:
        return {}
    row = params_df[params_df['Model'] == model_name]
    if row.empty:
        return {}
    return row.iloc[0].drop('Model').dropna().to_dict()


# Default sklearn parameters for comparison
DEFAULT_PARAMS = {
    'Logistic Regression': {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'},
    'Decision Tree': {'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2},
    'K-Nearest Neighbors': {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'minkowski', 'p': 2},
    'Naive Bayes': {'var_smoothing': 1e-9},
    'Random Forest': {'n_estimators': 100, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'max_features': 'sqrt'},
    'XGBoost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.3, 'subsample': 1.0, 'colsample_bytree': 1.0, 'gamma': 0}
}


# ============================
# Main App
# ============================
def main():
    """Main application function."""

    # Header
    st.markdown('<h2 class="main-header">🏥 Breast Cancer Classification System</h2>',
                unsafe_allow_html=True)
    st.markdown('<p class="tagline">Leveraging Machine Learning with Hyperparameter Tuning for Early Cancer Detection</p>',
                unsafe_allow_html=True)

    # Introduction
    st.markdown("""
    <h3 class="sub-header">📊 Problem Statement</h3>
    """, unsafe_allow_html=True)
    st.markdown("""
    This application demonstrates a machine learning solution for **breast cancer diagnosis**
    using the **Breast Cancer Wisconsin (Diagnostic)** dataset from the UCI Machine Learning Repository.

    The goal is to classify tumors as **Malignant** (cancerous) or **Benign** (non-cancerous)
    based on features computed from digitized images of fine needle aspirate (FNA) of breast mass.

    > ⚠️ **Clinical Insight:** In cancer diagnosis, **Recall (Sensitivity)** is the most critical metric
    > because missing a cancer case (False Negative) is far more dangerous than a false alarm (False Positive).
    """)

    with st.expander("📋 Dataset Information"):
        st.markdown("""
        **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset  
        **Source:** UCI Machine Learning Repository  
        **Features:** 30 numeric features computed from cell nuclei characteristics  
        **Target Classes:** 0 = Malignant (212 samples), 1 = Benign (357 samples)  
        **Total Samples:** 569 instances  
        
        **Training Approach:**
        - 80/20 Train-Test Split (Stratified)
        - StandardScaler for feature normalization
        - GridSearchCV with 5-Fold StratifiedKFold for hyperparameter tuning
        - Optimized for F1 Score (balances Recall & Precision)
        """)

    # Load everything
    X, y, feature_names, target_names = load_dataset()
    models, scaler = load_models()
    metrics_df = load_metrics()
    params_df = load_hyperparameters()

    # Sidebar - Model Selection
    st.sidebar.markdown("## 🎯 Navigation")
    st.sidebar.markdown("---")

    model_display_names = {
        'logistic_regression': 'Logistic Regression',
        'decision_tree': 'Decision Tree',
        'k-nearest_neighbors': 'K-Nearest Neighbors',
        'naive_bayes': 'Naive Bayes',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost'
    }

    selected_model_key = st.sidebar.selectbox(
        "🔬 Choose a Classification Model:",
        options=list(model_display_names.keys()),
        format_func=lambda x: model_display_names[x]
    )

    selected_model_name = model_display_names[selected_model_key]
    selected_model = models[selected_model_key]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Selected Model")
    st.sidebar.success(f"**{selected_model_name}**")

    if metrics_df is not None:
        model_row = metrics_df[metrics_df['Model'] == selected_model_name]
        if not model_row.empty:
            st.sidebar.markdown("### 📊 Quick Stats (Test Set)")
            st.sidebar.metric("Recall", f"{model_row['Recall'].values[0]:.4f}")
            st.sidebar.metric("Accuracy", f"{model_row['Accuracy'].values[0]:.4f}")
            st.sidebar.metric("F1 Score", f"{model_row['F1 Score'].values[0]:.4f}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **👨‍💻 Developed for**  
    ML Assignment 2
    
    **📚 Dataset Source**  
    UCI ML Repository
    
    **🔧 Tuning Method**  
    GridSearchCV + StratifiedKFold
    """)

    # ==========================================
    # Tabs
    # ==========================================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📤 Upload & Predict",
        "🏆 Best Model Insights",
        "📊 Model Performance",
        "🔧 Hyperparameter Tuning",
        "🎯 Model Comparison",
        "📋 Detailed Reports"
    ])

    # ==========================================
    # Tab 1: Upload & Predict
    # ==========================================
    with tab1:
        st.markdown('<p class="sub-header">📤 Upload Test Data & Predict</p>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>📋 Instructions:</b> Upload a CSV file containing test data from the Breast Cancer Wisconsin dataset.
        The file should contain the 30 feature columns and optionally a <code>target</code> column for evaluation.
        If no target column is present, the app will show predictions only.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Download sample CSV
        col_dl, col_info = st.columns([1, 2])
        with col_dl:
            test_csv = generate_test_csv()
            csv_data = test_csv.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Test CSV",
                data=csv_data,
                file_name="breast_cancer_test_data.csv",
                mime="text/csv",
                help="Download a sample test dataset (20% holdout) to try the upload feature"
            )
        with col_info:
            st.info("💡 **Tip:** Download the sample CSV above to test the upload feature. "
                    "It contains the 20% test split with 114 samples, 30 features + target column.")

        st.markdown("---")

        # File uploader
        uploaded_file = st.file_uploader(
            "📂 Upload your test data CSV file",
            type=['csv'],
            help="Upload a CSV with 30 feature columns from the Breast Cancer Wisconsin dataset"
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {uploaded_df.shape[0]} rows × {uploaded_df.shape[1]} columns")

                with st.expander("👀 Preview Uploaded Data"):
                    st.dataframe(uploaded_df.head(10), use_container_width=True)

                # Separate features and target
                has_target = 'target' in uploaded_df.columns

                if has_target:
                    y_uploaded = uploaded_df['target'].values
                    X_uploaded = uploaded_df.drop('target', axis=1).values
                    st.info(f"✅ Target column detected — evaluation metrics will be shown. "
                           f"Classes: Malignant={np.sum(y_uploaded==0)}, Benign={np.sum(y_uploaded==1)}")
                else:
                    X_uploaded = uploaded_df.values
                    y_uploaded = None
                    st.warning("⚠️ No 'target' column found — predictions only (no evaluation metrics).")

                # Validate feature count
                if X_uploaded.shape[1] != 30:
                    st.error(f"❌ Expected 30 features, got {X_uploaded.shape[1]}. "
                            f"Please check your CSV format.")
                else:
                    # Scale and predict
                    X_uploaded_scaled = scaler.transform(X_uploaded)
                    y_pred_uploaded = selected_model.predict(X_uploaded_scaled)

                    st.markdown("---")
                    st.markdown(f"### 🔮 Predictions using **{selected_model_name}**")

                    # Prediction summary
                    n_malignant = np.sum(y_pred_uploaded == 0)
                    n_benign = np.sum(y_pred_uploaded == 1)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Samples", len(y_pred_uploaded))
                    with col2:
                        st.markdown(f"""
                        <div class="prediction-malignant">
                            <b>🔴 Malignant (Cancerous)</b><br>
                            <span style="font-size: 36px; font-weight: bold;">{n_malignant}</span><br>
                            <span>({n_malignant/len(y_pred_uploaded)*100:.1f}%)</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"""
                        <div class="prediction-benign">
                            <b>🟢 Benign (Non-cancerous)</b><br>
                            <span style="font-size: 36px; font-weight: bold;">{n_benign}</span><br>
                            <span>({n_benign/len(y_pred_uploaded)*100:.1f}%)</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Results table
                    st.markdown("### 📋 Prediction Results")
                    results_table = uploaded_df.copy()
                    results_table['Prediction'] = ['Malignant' if p == 0 else 'Benign' for p in y_pred_uploaded]
                    if has_target:
                        results_table['Actual'] = ['Malignant' if t == 0 else 'Benign' for t in y_uploaded]
                        results_table['Correct'] = ['✅' if p == t else '❌' for p, t in zip(y_pred_uploaded, y_uploaded)]
                    st.dataframe(results_table, use_container_width=True)

                    # Evaluation metrics (only if target exists)
                    if has_target:
                        st.markdown("---")
                        st.markdown("### 📊 Evaluation Metrics on Uploaded Data")

                        acc = accuracy_score(y_uploaded, y_pred_uploaded)
                        prec = precision_score(y_uploaded, y_pred_uploaded)
                        rec = recall_score(y_uploaded, y_pred_uploaded)
                        f1 = f1_score(y_uploaded, y_pred_uploaded)
                        mcc = matthews_corrcoef(y_uploaded, y_pred_uploaded)
                        if hasattr(selected_model, 'predict_proba'):
                            y_proba = selected_model.predict_proba(X_uploaded_scaled)[:, 1]
                            auc = roc_auc_score(y_uploaded, y_proba)
                        else:
                            auc = roc_auc_score(y_uploaded, y_pred_uploaded)

                        st.markdown("#### 🔴 Critical Metric")
                        rc1, rc2, rc3 = st.columns([2, 1, 1])
                        with rc1:
                            st.markdown(f"""
                            <div class="metric-container" style="border-color: #ff416c;">
                                <span class="priority-high">MOST IMPORTANT</span>
                                <br><br>
                                🔴 <span style="color: #a0a0a0;">Recall (Sensitivity)</span>
                                <br>
                                <span class="metric-value" style="font-size: 48px; color: #ff416c;">{rec:.4f}</span>
                                <br>
                                <span class="metric-label">Ability to detect actual cancer cases</span>
                            </div>
                            """, unsafe_allow_html=True)
                        with rc2:
                            st.metric("🎪 Precision", f"{prec:.4f}")
                        with rc3:
                            st.metric("⚖️ F1 Score", f"{f1:.4f}")

                        mc1, mc2, mc3 = st.columns(3)
                        with mc1:
                            st.metric("📈 AUC Score", f"{auc:.4f}")
                        with mc2:
                            st.metric("🎯 Accuracy", f"{acc:.4f}")
                        with mc3:
                            st.metric("📊 MCC", f"{mcc:.4f}")

                        # Confusion Matrix on uploaded data
                        st.markdown("---")
                        st.markdown("### 🔍 Confusion Matrix (Uploaded Data)")
                        uc1, uc2 = st.columns([2, 1])
                        with uc1:
                            fig = plot_confusion_matrix(y_uploaded, y_pred_uploaded, target_names)
                            st.pyplot(fig)
                        with uc2:
                            cm = confusion_matrix(y_uploaded, y_pred_uploaded)
                            tn, fp, fn, tp = cm.ravel()
                            st.markdown(f"""
                            <div class="metric-container">
                                ✅ <b>True Negative:</b> <span style="color: #34e89e; font-size: 24px;">{tn}</span>
                                <br><span class="metric-label">Correctly identified Malignant</span>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"""
                            <div class="metric-container">
                                ✅ <b>True Positive:</b> <span style="color: #34e89e; font-size: 24px;">{tp}</span>
                                <br><span class="metric-label">Correctly identified Benign</span>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"""
                            <div class="metric-container" style="border-color: #f7971e;">
                                ⚠️ <b>False Positive:</b> <span style="color: #f7971e; font-size: 24px;">{fp}</span>
                                <br><span class="metric-label">False Alarms (Type I)</span>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(f"""
                            <div class="metric-container" style="border-color: #ff416c;">
                                ❌ <b>False Negative:</b> <span style="color: #ff416c; font-size: 24px;">{fn}</span>
                                <br><span class="metric-label">Missed Cancer Cases ⚠️</span>
                            </div>
                            """, unsafe_allow_html=True)

                        # Classification Report on uploaded data
                        st.markdown("---")
                        st.markdown("### 📋 Classification Report (Uploaded Data)")
                        report_df = display_classification_report(y_uploaded, y_pred_uploaded, target_names)
                        st.dataframe(
                            report_df.style.format("{:.4f}")
                                .background_gradient(cmap='YlGnBu', subset=['precision', 'recall', 'f1-score'])
                                .set_properties(**{'text-align': 'center'}),
                            use_container_width=True
                        )

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please ensure the CSV has the correct format with 30 feature columns.")

        else:
            st.markdown("""
            <div class="upload-section">
                <p style="font-size: 20px; color: #00d2ff;">📂 No file uploaded yet</p>
                <p style="color: #a0a0a0;">Upload a CSV file above or download the sample test data to get started.</p>
                <p style="color: #a0a0a0; font-size: 13px;">
                Expected: 30 feature columns + optional 'target' column<br>
                Features: mean radius, mean texture, mean perimeter, ... (30 features)
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================
    # Tab 2: Best Model Insights
    # ==========================================
    with tab2:
        st.markdown('<p class="sub-header">🏆 Best Model Insights & Recommendations</p>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>🔑 Key Insight:</b> For breast cancer classification, <b>Recall (Sensitivity)</b> is the
        most critical metric. A model with high recall ensures that most actual cancer cases are
        correctly identified, minimizing the risk of missing a diagnosis.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        if metrics_df is not None:
            best_models = get_best_models(metrics_df)

            priority_metrics = [
                ('Recall', '🔴', 'CRITICAL', 'Minimizes missed cancer cases (False Negatives)'),
                ('Precision', '🟠', 'HIGH', 'Minimizes false alarms (False Positives)'),
                ('F1 Score', '🟡', 'HIGH', 'Balance between Precision and Recall'),
                ('AUC Score', '🔵', 'MEDIUM', 'Overall ability to distinguish classes'),
                ('Accuracy', '🟢', 'MEDIUM', 'Overall correctness of predictions'),
                ('MCC', '🟣', 'MEDIUM', 'Balanced measure for imbalanced datasets'),
            ]

            # Winner Banner
            best_recall = best_models['Recall']
            max_recall = metrics_df['Recall'].max()
            tied_models = metrics_df[metrics_df['Recall'] == max_recall]['Model'].tolist()

            if len(tied_models) > 1:
                tied_text = " & ".join(tied_models)
                st.markdown(f"""
                <div class="winner-banner">
                    🏆 TOP MODELS FOR RECALL: {tied_text.upper()}
                    <br>
                    <span style="font-size: 16px;">Highest Recall Score: {max_recall:.4f} — Best at detecting cancer cases</span>
                    <br>
                    <span style="font-size: 14px;">Primary Recommendation: {best_recall['model']} (best AUC as tiebreaker)</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="winner-banner">
                    🏆 RECOMMENDED MODEL: {best_recall['model'].upper()}
                    <br>
                    <span style="font-size: 16px;">Highest Recall Score: {best_recall['score']:.4f} — Best at detecting cancer cases</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("### 📋 Best Model for Each Metric (Ranked by Clinical Priority)")
            st.markdown("")

            for i in range(0, len(priority_metrics), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(priority_metrics):
                        metric_name, emoji, priority, description = priority_metrics[i + j]
                        best = best_models[metric_name]
                        with cols[j]:
                            priority_class = "priority-high" if priority in ['CRITICAL', 'HIGH'] else "priority-medium"
                            st.markdown(f"""
                            <div class="metric-container">
                                <span class="{priority_class}">{priority} PRIORITY</span>
                                <br><br>
                                {emoji} <span style="color: #a0a0a0; font-size: 14px;">{metric_name}</span>
                                <br>
                                <span class="metric-value">{best['score']:.4f}</span>
                                <br>
                                <span style="color: #34e89e; font-size: 16px; font-weight: bold;">{best['model']}</span>
                                <br>
                                <span class="metric-label">{description}</span>
                            </div>
                            """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("---")

            st.markdown("### 🕸️ Performance Profile - Recommended Model")
            fig = plot_radar_chart(metrics_df, best_recall['model'])
            st.pyplot(fig)

            st.markdown("---")
            st.markdown("### 💡 Key Observations")

            st.markdown(f"""
            <div class="insight-box">
            <b>1. Best for Recall (Most Important):</b> <b>{best_models['Recall']['model']}</b>
            achieves a recall of <b>{best_models['Recall']['score']:.4f}</b>, correctly identifying
            <b>{best_models['Recall']['score']*100:.2f}%</b> of all cancer cases.
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="insight-box">
            <b>2. Best Overall Accuracy:</b> <b>{best_models['Accuracy']['model']}</b>
            achieves <b>{best_models['Accuracy']['score']*100:.2f}%</b> accuracy on the test set.
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="insight-box">
            <b>3. Best AUC Score:</b> <b>{best_models['AUC Score']['model']}</b>
            with AUC of <b>{best_models['AUC Score']['score']:.4f}</b> shows the best ability
            to distinguish between malignant and benign tumors.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.warning(f"""
            ⚕️ **Clinical Recommendation:**
            For deployment in a clinical setting, **{best_models['Recall']['model']}** is recommended
            as it minimizes the chance of missing a cancer diagnosis. In healthcare,
            **it's better to have a false alarm than to miss a real case.**
            """)

    # ==========================================
    # Tab 3: Model Performance
    # ==========================================
    with tab3:
        st.markdown(f'<p class="sub-header">📊 Performance Metrics: {selected_model_name}</p>',
                    unsafe_allow_html=True)

        if metrics_df is not None:
            model_metrics = metrics_df[metrics_df['Model'] == selected_model_name]

            if not model_metrics.empty:
                recall_val = model_metrics['Recall'].values[0]
                precision_val = model_metrics['Precision'].values[0]
                f1_val = model_metrics['F1 Score'].values[0]
                auc_val = model_metrics['AUC Score'].values[0]
                accuracy_val = model_metrics['Accuracy'].values[0]
                mcc_val = model_metrics['MCC'].values[0]
                cv_f1 = model_metrics['Best CV F1'].values[0] if 'Best CV F1' in model_metrics.columns else None

                st.markdown("#### 🔴 Critical Metric")
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"""
                    <div class="metric-container" style="border-color: #ff416c;">
                        <span class="priority-high">MOST IMPORTANT</span>
                        <br><br>
                        🔴 <span style="color: #a0a0a0;">Recall (Sensitivity)</span>
                        <br>
                        <span class="metric-value" style="font-size: 48px; color: #ff416c;">{recall_val:.4f}</span>
                        <br>
                        <span class="metric-label">Ability to detect actual cancer cases</span>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.metric("🎪 Precision", f"{precision_val:.4f}")
                with col3:
                    st.metric("⚖️ F1 Score", f"{f1_val:.4f}")

                st.markdown("")
                st.markdown("#### 📊 Other Metrics")
                col4, col5, col6, col7 = st.columns(4)
                with col4:
                    st.metric("📈 AUC Score", f"{auc_val:.4f}")
                with col5:
                    st.metric("🎯 Accuracy", f"{accuracy_val:.4f}")
                with col6:
                    st.metric("📊 MCC", f"{mcc_val:.4f}")
                with col7:
                    if cv_f1 is not None:
                        st.metric("🔄 Best CV F1", f"{cv_f1:.4f}")

                st.info("ℹ️ **Note:** Metrics from 20% test holdout. CV F1 from hyperparameter tuning.")

                st.markdown("---")
                st.markdown(f"### 🕸️ Performance Profile: {selected_model_name}")
                fig = plot_radar_chart(metrics_df, selected_model_name)
                st.pyplot(fig)

            st.markdown("---")
            with st.expander("ℹ️ Understanding the Metrics"):
                st.markdown("""
                | Metric | Description | Why it matters |
                |--------|-------------|----------------|
                | **Recall** | % of actual positives correctly identified | 🔴 Missing cancer is dangerous |
                | **Precision** | % of predicted positives that are correct | Reduces false alarms |
                | **F1 Score** | Harmonic mean of Precision & Recall | Balances both concerns |
                | **AUC Score** | Area Under ROC Curve | Overall discrimination ability |
                | **Accuracy** | % of all correct predictions | General correctness |
                | **MCC** | Matthews Correlation Coefficient | Balanced for imbalanced data |
                | **Best CV F1** | Cross-validation F1 during tuning | Generalization ability |
                """)

    # ==========================================
    # Tab 4: Hyperparameter Tuning
    # ==========================================
    with tab4:
        st.markdown(f'<p class="sub-header">🔧 Hyperparameter Tuning: {selected_model_name}</p>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>🔧 Tuning Strategy:</b> All models were tuned using <b>GridSearchCV</b> with
        <b>5-Fold Stratified Cross-Validation</b>, optimizing for <b>F1 Score</b>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        if params_df is not None:
            tuned_params = get_model_params(params_df, selected_model_name)
            default_params = DEFAULT_PARAMS.get(selected_model_name, {})

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🎯 Tuned Parameters (GridSearchCV)")
                if tuned_params:
                    params_html = f'<div class="param-card"><h4>✅ {selected_model_name}</h4>'
                    for param, value in tuned_params.items():
                        default_val = default_params.get(param, 'N/A')
                        changed = str(value) != str(default_val)
                        badge = '<span class="tuning-badge">TUNED</span>' if changed else '<span class="default-badge">DEFAULT</span>'
                        params_html += f'''
                        <div class="param-item">
                            <span class="param-name">{param}</span>
                            <span class="param-value">{value} {badge}</span>
                        </div>'''
                    params_html += '</div>'
                    st.markdown(params_html, unsafe_allow_html=True)

            with col2:
                st.markdown("### 📋 Default Parameters (sklearn)")
                if default_params:
                    defaults_html = f'<div class="param-card"><h4>📦 {selected_model_name} (Defaults)</h4>'
                    for param, value in default_params.items():
                        defaults_html += f'''
                        <div class="param-item">
                            <span class="param-name">{param}</span>
                            <span class="param-value">{value}</span>
                        </div>'''
                    defaults_html += '</div>'
                    st.markdown(defaults_html, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📊 Parameter Changes Summary")
            if tuned_params and default_params:
                changes = []
                for param, tuned_val in tuned_params.items():
                    default_val = default_params.get(param, 'N/A')
                    changed = str(tuned_val) != str(default_val)
                    changes.append({
                        'Parameter': param,
                        'Default': str(default_val),
                        'Tuned': str(tuned_val),
                        'Changed': '✅ Yes' if changed else '➖ No'
                    })
                st.dataframe(pd.DataFrame(changes), use_container_width=True)

                num_changed = sum(1 for c in changes if c['Changed'] == '✅ Yes')
                if num_changed == 0:
                    st.info(f"🔍 GridSearchCV confirmed default parameters are optimal for {selected_model_name}.")
                else:
                    st.success(f"🔧 {num_changed}/{len(changes)} parameters were tuned to improve performance.")

            st.markdown("---")
            st.markdown("### 🗂️ All Models - Tuned Hyperparameters")

            for _, row in params_df.iterrows():
                model_name = row['Model']
                params = row.drop('Model').dropna().to_dict()
                defaults = DEFAULT_PARAMS.get(model_name, {})

                with st.expander(f"🔧 {model_name}"):
                    if params:
                        pc1, pc2 = st.columns(2)
                        param_items = list(params.items())
                        mid = (len(param_items) + 1) // 2
                        with pc1:
                            for p, v in param_items[:mid]:
                                changed = str(v) != str(defaults.get(p, 'N/A'))
                                icon = "🟢" if changed else "⚪"
                                st.markdown(f"{icon} **{p}:** `{v}` {'*(tuned)*' if changed else '*(default)*'}")
                        with pc2:
                            for p, v in param_items[mid:]:
                                changed = str(v) != str(defaults.get(p, 'N/A'))
                                icon = "🟢" if changed else "⚪"
                                st.markdown(f"{icon} **{p}:** `{v}` {'*(tuned)*' if changed else '*(default)*'}")

    # ==========================================
    # Tab 5: Model Comparison
    # ==========================================
    with tab5:
        st.markdown('<p class="sub-header">🎯 All Models Comparison</p>',
                    unsafe_allow_html=True)

        if metrics_df is not None:
            display_cols = ['Model', 'Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC']
            if 'Best CV F1' in metrics_df.columns:
                display_cols.append('Best CV F1')

            format_dict = {col: '{:.4f}' for col in display_cols if col != 'Model'}
            styled_df = metrics_df[display_cols].style.format(format_dict).background_gradient(
                cmap='YlGnBu', subset=[c for c in display_cols if c != 'Model']
            ).set_properties(**{'text-align': 'center'})
            st.dataframe(styled_df, use_container_width=True)

            st.markdown("---")
            st.markdown("### 📊 Visual Comparison")

            metric_to_plot = st.selectbox(
                "Select metric to compare:",
                options=['Recall', 'Accuracy', 'AUC Score', 'Precision', 'F1 Score', 'MCC'],
                index=0
            )
            fig = plot_comparison_chart(metrics_df, metric_to_plot)
            st.pyplot(fig)

            best_idx = metrics_df[metric_to_plot].idxmax()
            best_model = metrics_df.loc[best_idx, 'Model']
            best_score = metrics_df.loc[best_idx, metric_to_plot]
            st.markdown(f"""
            <div class="winner-banner">
                🏆 Best for {metric_to_plot}: {best_model} ({best_score:.4f})
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🕸️ Radar Chart Comparison")
            rc1, rc2 = st.columns(2)
            with rc1:
                m1 = st.selectbox("Model 1:", metrics_df['Model'].tolist(), index=0)
                st.pyplot(plot_radar_chart(metrics_df, m1))
            with rc2:
                m2 = st.selectbox("Model 2:", metrics_df['Model'].tolist(), index=5)
                st.pyplot(plot_radar_chart(metrics_df, m2))

    # ==========================================
    # Tab 6: Detailed Reports
    # ==========================================
    with tab6:
        st.markdown('<p class="sub-header">📋 Detailed Reports</p>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>📋 Note:</b> Reports below use the <b>full dataset (569 samples)</b> for comprehensive
        visualization. For test-set metrics, see <b>Model Performance</b> or <b>Upload & Predict</b> tabs.
        </div>
        """, unsafe_allow_html=True)

        X_scaled = scaler.transform(X)
        y_pred_full = selected_model.predict(X_scaled)

        rt1, rt2 = st.tabs(["🔍 Confusion Matrix", "📋 Classification Report"])

        with rt1:
            st.markdown(f"### Confusion Matrix: {selected_model_name}")
            fc1, fc2 = st.columns([2, 1])
            with fc1:
                fig = plot_confusion_matrix(y, y_pred_full, target_names)
                st.pyplot(fig)
            with fc2:
                cm = confusion_matrix(y, y_pred_full)
                tn, fp, fn, tp = cm.ravel()
                st.markdown(f"""
                <div class="metric-container">
                    ✅ <b>True Negative:</b> <span style="color: #34e89e; font-size: 24px;">{tn}</span>
                    <br><span class="metric-label">Correctly identified Malignant</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-container">
                    ✅ <b>True Positive:</b> <span style="color: #34e89e; font-size: 24px;">{tp}</span>
                    <br><span class="metric-label">Correctly identified Benign</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-container" style="border-color: #f7971e;">
                    ⚠️ <b>False Positive:</b> <span style="color: #f7971e; font-size: 24px;">{fp}</span>
                    <br><span class="metric-label">False Alarms (Type I)</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="metric-container" style="border-color: #ff416c;">
                    ❌ <b>False Negative:</b> <span style="color: #ff416c; font-size: 24px;">{fn}</span>
                    <br><span class="metric-label">Missed Cancer Cases ⚠️ CRITICAL</span>
                </div>
                """, unsafe_allow_html=True)

        with rt2:
            st.markdown(f"### Classification Report: {selected_model_name}")
            report_df = display_classification_report(y, y_pred_full, target_names)
            st.dataframe(
                report_df.style.format("{:.4f}")
                    .background_gradient(cmap='YlGnBu', subset=['precision', 'recall', 'f1-score'])
                    .set_properties(**{'text-align': 'center'}),
                use_container_width=True
            )
            with st.expander("ℹ️ Understanding the Classification Report"):
                st.markdown("""
                | Metric | Formula | Description |
                |--------|---------|-------------|
                | **Precision** | TP / (TP + FP) | How many predicted positives are correct |
                | **Recall** | TP / (TP + FN) | How many actual positives were found |
                | **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of Precision & Recall |
                | **Support** | — | Number of actual occurrences per class |
                """)


if __name__ == "__main__":
    main()