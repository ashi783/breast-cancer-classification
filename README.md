# 🏥 Breast Cancer Classification Project

**Author:** Ashutosh Joshi
**ID:** 2025AA05402  
**Email:** 2025aa05402@wilp.bits-pilani.ac.in  
**Institution:** BITS Pilani — WILP

---

## Problem Statement
Breast cancer is one of the most common cancers among women worldwide. Early detection and accurate diagnosis are crucial for effective treatment. This project aims to build a machine learning classification system to predict whether a tumor is **Malignant** (cancerous) or **Benign** (non-cancerous) based on features derived from digitized images of fine needle aspirate (FNA) of breast mass.

In cancer diagnosis, **Recall (Sensitivity)** is the most critical metric because missing a cancer case (False Negative) is far more dangerous than a false alarm (False Positive).

## Dataset Description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository / sklearn built-in dataset
- **Instances:** 569 (212 Malignant, 357 Benign)
- **Features:** 30 numeric features computed from cell nuclei measurements
- **Feature Categories:** Mean, Standard Error, and Worst (largest) values for:
  - Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension
- **Target:** Binary classification — Malignant (0) or Benign (1)
- **Preprocessing:** StandardScaler applied for feature normalization
- **Train/Test Split:** 80/20 with stratification (random_state=42)

## Models Used
All models were trained with **GridSearchCV** hyperparameter tuning (5-fold cross-validation, optimized for F1 score):

1. **Logistic Regression** — Linear model with regularization
2. **Decision Tree** — Tree-based non-linear classifier
3. **K-Nearest Neighbors (KNN)** — Instance-based learning
4. **Naive Bayes** — Probabilistic classifier based on Bayes' theorem
5. **Random Forest (Ensemble)** — Bagging ensemble of decision trees
6. **XGBoost (Ensemble)** — Gradient boosted decision trees

## Comparison of Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9957 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9035 | 0.9358 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |
| K-Nearest Neighbors | 0.9649 | 0.9714 | 0.9595 | 0.9861 | 0.9726 | 0.9245 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9561 | 0.9924 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| XGBoost (Ensemble) | 0.9561 | 0.9950 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | **Best overall performer.** Achieved the highest Accuracy (98.25%), F1 Score (0.9861), and MCC (0.9623). Despite being a simple linear model, it outperformed all complex models after hyperparameter tuning. This suggests the dataset is largely linearly separable. |
| Decision Tree | **Weakest performer.** Lowest scores across all metrics — Accuracy (90.35%), AUC (0.9358), and MCC (0.7969). Prone to overfitting on training data and struggles to generalize well. The relatively low Recall (90.28%) makes it risky for cancer diagnosis where missing cases is critical. |
| K-Nearest Neighbors | **Strong Recall performer.** Tied for the highest Recall (98.61%) with Logistic Regression and XGBoost, meaning it rarely misses cancer cases. Good F1 (0.9726) and MCC (0.9245), but slightly lower AUC (0.9714) compared to ensemble models. |
| Naive Bayes | **Moderate performer with high AUC.** Despite the strong independence assumption, achieved a good AUC (0.9868), indicating excellent class separation in probability scores. However, Accuracy (92.98%) and MCC (0.8492) are lower, suggesting calibration issues in hard predictions. |
| Random Forest (Ensemble) | **Solid ensemble performer.** High AUC (0.9924) shows strong ranking ability. Good balance across all metrics with Recall (97.22%) and F1 (0.9655). Slightly outperformed by Logistic Regression and XGBoost, but remains a reliable and robust choice. |
| XGBoost (Ensemble) | **Top-tier model with best AUC.** Achieved the highest AUC (0.9950) and tied for the highest Recall (98.61%). Excellent for minimizing missed cancer diagnoses. Same Accuracy as Random Forest (95.61%) but better probability calibration shown by superior AUC. |

## Instructions to Run the Project Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/ashi783/breast-cancer-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd breast-cancer-classification
   ```
3. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Deployment on Streamlit Cloud
1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"** and select the repository
4. Set main file path as `app.py`
5. Click **"Deploy"**

## Live Demo
🔗 [Streamlit App](https://breast-cancer-classification-7w2p9kudtgfdonjcfjkgge.streamlit.app/)

## Project Structure
```
breast-cancer-classification/
├── .streamlit/
│   └── config.toml
├── data/
│   └── wdbc.data
├── model/
│   ├── train_models.py
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── k-nearest_neighbors_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   ├── model_metrics.csv
│   └── best_hyperparameters.csv
├── app.py
├── requirements.txt
└── README.md
```