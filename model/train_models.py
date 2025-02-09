"""
Breast Cancer Classification - Model Training Script

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
Source: UCI Machine Learning Repository
Access: sklearn.datasets.load_breast_cancer()
URL: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

Dataset Properties:
- Classification Type: Binary (Malignant=0, Benign=1)
- Features: 30 numeric features
- Instances: 569 samples
- Repository: UCI Machine Learning Repository

This script trains 6 classification models with hyperparameter tuning
and saves them along with evaluation metrics.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def load_and_prepare_data():
    """
    Load the Breast Cancer Wisconsin dataset and split into train/test sets.
    """
    print("Loading Breast Cancer Wisconsin dataset...")
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target distribution: Malignant={np.sum(y==0)}, Benign={np.sum(y==1)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, feature_names


def standardize_features(X_train, X_test):
    """
    Standardize features using StandardScaler.
    """
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def get_models_and_params():
    """
    Define models with their hyperparameter search spaces.
    
    Returns:
        Dictionary of model configs with model object and param grid
    """
    models_config = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=10000),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan'],
                'p': [1, 2]
            }
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {
                'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2, 0.3]
            }
        }
    }

    return models_config


def evaluate_model(model, X_test, y_test, y_pred):
    """
    Calculate all evaluation metrics for a model.
    """
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = y_pred

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC Score': roc_auc_score(y_test, y_pred_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    return metrics


def train_and_evaluate_models(models_config, X_train, X_test, y_train, y_test):
    """
    Train all models with hyperparameter tuning using GridSearchCV
    and evaluate their performance.
    
    Uses StratifiedKFold with 5 folds and optimizes for F1 Score
    (balances Recall and Precision - prevents degenerate models
    while still prioritizing cancer detection).
    """
    print("\n" + "=" * 60)
    print("Training Models with Hyperparameter Tuning (GridSearchCV)")
    print("Scoring Metric: F1 Score (balances Recall & Precision)")
    print("=" * 60)

    # Cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    trained_models = {}
    results = []
    best_params_all = {}

    for model_name, config in models_config.items():
        print(f"\n{'─' * 50}")
        print(f"🔄 Tuning {model_name}...")
        print(f"{'─' * 50}")

        model = config['model']
        param_grid = config['params']

        # Calculate total combinations
        total_combos = 1
        for key, values in param_grid.items():
            total_combos *= len(values)
        print(f"  Search space: {total_combos} combinations")

        # Use GridSearchCV with F1 as scoring metric
        # F1 balances Recall and Precision - prevents degenerate models
        # while still rewarding high recall for cancer detection
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=0,
            refit=True
        )

        # Fit the grid search
        grid_search.fit(X_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_

        print(f"  ✓ Best CV F1 Score: {best_cv_score:.4f}")
        print(f"  ✓ Best Parameters:")
        for param, value in best_params.items():
            print(f"    - {param}: {value}")

        # Predict on test set
        y_pred = best_model.predict(X_test)

        # Evaluate
        metrics = evaluate_model(best_model, X_test, y_test, y_pred)
        metrics['Model'] = model_name
        metrics['Best CV F1'] = best_cv_score
        results.append(metrics)
        trained_models[model_name] = best_model
        best_params_all[model_name] = best_params

        # Print test metrics
        print(f"\n  📊 Test Set Results:")
        print(f"    Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"    AUC Score: {metrics['AUC Score']:.4f}")
        print(f"    Precision: {metrics['Precision']:.4f}")
        print(f"    Recall:    {metrics['Recall']:.4f}")
        print(f"    F1 Score:  {metrics['F1 Score']:.4f}")
        print(f"    MCC:       {metrics['MCC']:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df[['Model', 'Accuracy', 'AUC Score', 'Precision',
                              'Recall', 'F1 Score', 'MCC', 'Best CV F1']]

    return trained_models, results_df, best_params_all


def save_models_and_results(trained_models, scaler, results_df, best_params_all):
    """
    Save trained models, scaler, evaluation results, and best hyperparameters.
    """
    print("\n" + "=" * 60)
    print("Saving Models and Results")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Saving to directory: {script_dir}")

    # Save each model
    for model_name, model in trained_models.items():
        filename = os.path.join(script_dir, f"{model_name.replace(' ', '_').lower()}_model.pkl")
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"✓ Saved: {filename}")

    # Save scaler
    scaler_path = os.path.join(script_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)
    print(f"✓ Saved: {scaler_path}")

    # Save results as CSV
    csv_path = os.path.join(script_dir, 'model_metrics.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")

    # Save best hyperparameters
    params_df = pd.DataFrame([
        {'Model': name, **params}
        for name, params in best_params_all.items()
    ])
    params_path = os.path.join(script_dir, 'best_hyperparameters.csv')
    params_df.to_csv(params_path, index=False)
    print(f"✓ Saved: {params_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


def main():
    """
    Main function to execute the training pipeline.
    """
    print("\n" + "=" * 60)
    print("BREAST CANCER CLASSIFICATION - MODEL TRAINING")
    print("With Hyperparameter Tuning (GridSearchCV)")
    print("=" * 60)

    # Step 1: Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()

    # Step 2: Standardize features
    X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)

    # Step 3: Get models and hyperparameter grids
    models_config = get_models_and_params()

    # Step 4: Train and evaluate with tuning
    trained_models, results_df, best_params_all = train_and_evaluate_models(
        models_config, X_train_scaled, X_test_scaled, y_train, y_test
    )

    # Step 5: Display results
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON (After Hyperparameter Tuning)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Step 6: Save models and results
    save_models_and_results(trained_models, scaler, results_df, best_params_all)

    print("\n✅ All models trained with hyperparameter tuning and saved successfully!")


if __name__ == "__main__":
    main()