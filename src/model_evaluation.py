import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_models(y_true, y_pred_dict):
    metrics = {}
    
    for model_name, y_pred in y_pred_dict.items():
        metrics[model_name] = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1 Score': f1_score(y_true, y_pred),
            'ROC AUC': roc_auc_score(y_true, y_pred)
        }
    
    return pd.DataFrame(metrics).T

def save_evaluation_results(metrics_df, filepath='model_evaluation_results.csv'):
    metrics_df.to_csv(filepath)