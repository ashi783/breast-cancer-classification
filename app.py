"""
Breast Cancer Classification - Streamlit Web Application

This app provides an interactive interface to explore different classification models
trained on the Breast Cancer Wisconsin dataset.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
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

# Custom CSS for unique styling
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    
    /* Header styling */
    .main-header {
        font-size: 48px;
        font-weight: bold;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
        padding: 20px 0;
    }
    
    .sub-header {
        font-size: 26px;
        font-weight: bold;
        color: #00d2ff;
        margin-top: 20px;
        border-left: 4px solid #00d2ff;
        padding-left: 15px;
    }
    
    .tagline {
        font-size: 18px;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 30px;
        font-style: italic;
    }
    
    /* Metric cards */
    .metric-container {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
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
        font-size: 36px;
        font-weight: bold;
        color: #00d2ff;
    }
    
    .metric-label {
        font-size: 14px;
        color: #a0a0a0;
        margin-top: 5px;
    }
    
    /* Priority badge */
    .priority-high {
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
    
    .priority-medium {
        background: linear-gradient(90deg, #f7971e, #ffd200);
        color: #1a1a2e;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
    }
    
    /* Winner banner */
    .winner-banner {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        color: #1a1a2e;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(255, 215, 0, 0.4);
    }
    
    /* Insight box */
    .insight-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 4px solid #34e89e;
        padding: 15px 20px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
        color: #e0e0e0;
    }
    
    /* Param card */
    .param-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #3a7bd5;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(58, 123, 213, 0.15);
    }
    
    .param-card h4 {
        color: #00d2ff;
        margin-bottom: 15px;
        font-size: 18px;
    }
    
    .param-item {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #303050;
        color: #e0e0e0;
    }
    
    .param-name {
        color: #a0a0a0;
        font-size: 14px;
    }
    
    .param-value {
        color: #34e89e;
        font-weight: bold;
        font-size: 14px;
    }
    
    /* Tuning badge */
    .tuning-badge {
        background: linear-gradient(90deg, #34e89e, #0f9b0f);
        color: white;
        padding: 3px 12px;
        border-radius: 15px;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
        margin: 3px;
    }
    
    .default-badge {
        background: linear-gradient(90deg, #636e72, #b2bec3);
        color: white;
        padding: 3px 12px;
        border-radius: 15px;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
        margin: 3px;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px;
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
    for model_name in model_names:
        try:
            with open(f'model/{model_name}_model.pkl', 'rb') as file:
                models[model_name] = pickle.load(file)
        except FileNotFoundError:
            st.error(f"Model file not found: {model_name}_model.pkl")
    try:
        with open('model/scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
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
    """Display classification report as a formatted table."""
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return report_df.round(4)


def get_best_models(metrics_df):
    """Get best model for each metric, breaking ties with AUC Score then Accuracy."""
    metrics_list = ['Recall', 'Accuracy', 'AUC Score', 'Precision', 'F1 Score', 'MCC']
    tiebreakers = ['AUC Score', 'Accuracy', 'F1 Score']
    best_models = {}
    for metric in metrics_list:
        max_val = metrics_df[metric].max()
        tied = metrics_df[metrics_df[metric] == max_val]
        if len(tied) > 1:
            for tb in tiebreakers:
                if tb != metric:
                    best_idx = tied[tb].idxmax()
                    tied = tied.loc[[best_idx]]
                    break
            best_idx = tied.index[0]
        else:
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
    num_vars = len(metrics_cols)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
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
    """Plot styled bar chart comparison."""
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
    """Get hyperparameters for a specific model, excluding NaN values."""
    if params_df is None:
        return {}
    row = params_df[params_df['Model'] == model_name]
    if row.empty:
        return {}
    params = row.iloc[0].drop('Model').dropna().to_dict()
    return params


# ============================
# Default Parameters Reference
# ============================
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
    st.markdown('<p class="main-header">🏥 Breast Cancer Classification System</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="tagline">Leveraging Machine Learning with Hyperparameter Tuning for Early Cancer Detection</p>',
                unsafe_allow_html=True)

    # Introduction
    st.markdown("""
    ### 📊 Problem Statement
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

    # Sidebar
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
            st.sidebar.markdown("### 📊 Quick Stats")
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

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏆 Best Model Insights",
        "📊 Model Performance",
        "🔧 Hyperparameter Tuning",
        "🔍 Confusion Matrix",
        "📋 Classification Report",
        "🎯 Model Comparison"
    ])

    # Scale data
    X_scaled = scaler.transform(X)
    y_pred = selected_model.predict(X_scaled)

    # ==========================================
    # Tab 1: Best Model Insights
    # ==========================================
    with tab1:
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
                    🏆 RECOMMENDED MODELS (TIED): {tied_text.upper()}
                    <br>
                    <span style="font-size: 16px;">Highest Recall Score: {max_recall:.4f} — Best at detecting cancer cases</span>
                    <br>
                    <span style="font-size: 14px;">Tiebreaker (AUC Score): {best_recall['model']} selected as primary recommendation</span>
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
            achieves a recall of <b>{best_models['Recall']['score']:.4f}</b>, meaning it correctly
            identifies <b>{best_models['Recall']['score']*100:.2f}%</b> of all cancer cases.
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
    # Tab 2: Model Performance
    # ==========================================
    with tab2:
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
            else:
                recall_val = precision_val = f1_val = auc_val = accuracy_val = mcc_val = 0.0
                cv_f1 = None

            # Recall first
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

            st.info("ℹ️ **Note:** Test set metrics (20% holdout). CV F1 is the cross-validation score during hyperparameter tuning.")

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
    # Tab 3: Hyperparameter Tuning (NEW)
    # ==========================================
    with tab3:
        st.markdown(f'<p class="sub-header">🔧 Hyperparameter Tuning: {selected_model_name}</p>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>🔧 Tuning Strategy:</b> All models were tuned using <b>GridSearchCV</b> with
        <b>5-Fold Stratified Cross-Validation</b>, optimizing for <b>F1 Score</b>.
        This ensures a balance between Recall (detecting cancer) and Precision (avoiding false alarms).
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
                else:
                    st.info("No hyperparameters available for this model.")

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

            # Show what changed
            st.markdown("### 📊 Parameter Changes Summary")
            if tuned_params and default_params:
                changes = []
                for param, tuned_val in tuned_params.items():
                    default_val = default_params.get(param, 'N/A')
                    changed = str(tuned_val) != str(default_val)
                    changes.append({
                        'Parameter': param,
                        'Default Value': str(default_val),
                        'Tuned Value': str(tuned_val),
                        'Changed': '✅ Yes' if changed else '➖ No'
                    })
                changes_df = pd.DataFrame(changes)
                st.dataframe(changes_df, use_container_width=True)

                num_changed = sum(1 for c in changes if c['Changed'] == '✅ Yes')
                total = len(changes)
                if num_changed == 0:
                    st.info(f"🔍 **Result:** GridSearchCV confirmed that default parameters are already optimal for {selected_model_name} on this dataset.")
                else:
                    st.success(f"🔧 **Result:** {num_changed}/{total} parameters were tuned to improve {selected_model_name} performance.")

            st.markdown("---")

            # All models hyperparameters overview
            st.markdown("### 🗂️ All Models - Tuned Hyperparameters")

            for _, row in params_df.iterrows():
                model_name = row['Model']
                params = row.drop('Model').dropna().to_dict()
                defaults = DEFAULT_PARAMS.get(model_name, {})

                with st.expander(f"🔧 {model_name}"):
                    if params:
                        param_cols = st.columns(2)
                        param_items = list(params.items())
                        mid = (len(param_items) + 1) // 2

                        with param_cols[0]:
                            for param, value in param_items[:mid]:
                                default_val = defaults.get(param, 'N/A')
                                changed = str(value) != str(default_val)
                                icon = "🟢" if changed else "⚪"
                                st.markdown(f"{icon} **{param}:** `{value}` {'*(tuned)*' if changed else '*(default)*'}")

                        with param_cols[1]:
                            for param, value in param_items[mid:]:
                                default_val = defaults.get(param, 'N/A')
                                changed = str(value) != str(default_val)
                                icon = "🟢" if changed else "⚪"
                                st.markdown(f"{icon} **{param}:** `{value}` {'*(tuned)*' if changed else '*(default)*'}")
                    else:
                        st.info("Default parameters used.")

    # ==========================================
    # Tab 4: Confusion Matrix
    # ==========================================
    with tab4:
        st.markdown(f'<p class="sub-header">🔍 Confusion Matrix: {selected_model_name}</p>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = plot_confusion_matrix(y, y_pred, target_names)
            st.pyplot(fig)

        with col2:
            cm = confusion_matrix(y, y_pred)
            tn, fp, fn, tp = cm.ravel()

            st.markdown("### 📊 Breakdown")
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
                <br><span class="metric-label">Incorrectly predicted Benign (Type I)</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-container" style="border-color: #ff416c;">
                ❌ <b>False Negative:</b> <span style="color: #ff416c; font-size: 24px;">{fn}</span>
                <br><span class="metric-label">Missed Cancer Cases (Type II) ⚠️ CRITICAL</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("ℹ️ Understanding the Confusion Matrix"):
            st.markdown("""
            In medical diagnosis, **False Negatives** are particularly critical:
            - A **False Negative** means a patient with cancer is told they're healthy ❌
            - A **False Positive** means a healthy patient gets further testing ⚠️
            
            **In healthcare, it's better to have a false alarm than to miss a real case.**
            """)

    # ==========================================
    # Tab 5: Classification Report
    # ==========================================
    with tab5:
        st.markdown(f'<p class="sub-header">📋 Classification Report: {selected_model_name}</p>',
                    unsafe_allow_html=True)

        report_df = display_classification_report(y, y_pred, target_names)

        st.dataframe(
            report_df.style.format("{:.4f}")
                .background_gradient(cmap='YlGnBu', subset=['precision', 'recall', 'f1-score'])
                .set_properties(**{'text-align': 'center'}),
            use_container_width=True
        )

        st.markdown("---")
        with st.expander("ℹ️ Understanding the Classification Report"):
            st.markdown("""
            | Metric | Formula | Description |
            |--------|---------|-------------|
            | **Precision** | TP / (TP + FP) | How many predicted positives are correct |
            | **Recall** | TP / (TP + FN) | How many actual positives were found |
            | **F1-Score** | 2 × (P × R) / (P + R) | Harmonic mean of Precision & Recall |
            | **Support** | — | Number of actual occurrences per class |
            """)

    # ==========================================
    # Tab 6: Model Comparison
    # ==========================================
    with tab6:
        st.markdown('<p class="sub-header">🎯 All Models Comparison</p>',
                    unsafe_allow_html=True)

        if metrics_df is not None:
            display_cols = ['Model', 'Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score', 'MCC']
            if 'Best CV F1' in metrics_df.columns:
                display_cols.append('Best CV F1')

            format_dict = {col: '{:.4f}' for col in display_cols if col != 'Model'}

            styled_df = metrics_df[display_cols].style.format(format_dict).background_gradient(
                cmap='YlGnBu',
                subset=[c for c in display_cols if c != 'Model']
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
                🏆 Best Model for {metric_to_plot}: {best_model} ({best_score:.4f})
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            st.markdown("### 🕸️ Radar Chart Comparison")
            col1, col2 = st.columns(2)

            with col1:
                model1 = st.selectbox("Select Model 1:", metrics_df['Model'].tolist(), index=0)
                fig1 = plot_radar_chart(metrics_df, model1)
                st.pyplot(fig1)

            with col2:
                model2 = st.selectbox("Select Model 2:", metrics_df['Model'].tolist(), index=5)
                fig2 = plot_radar_chart(metrics_df, model2)
                st.pyplot(fig2)


if __name__ == "__main__":
    main()