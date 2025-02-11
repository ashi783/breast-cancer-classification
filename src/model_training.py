from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import pandas as pd
import pickle

def load_data(filepath):
    data = pd.read_csv(filepath)
    X = data.drop('diagnosis', axis=1)
    y = data['diagnosis'].map({'M': 1, 'B': 0})  # Mapping M to 1 and B to 0
    return X, y

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models, X_test, y_test

def save_best_model(trained_models, X_test, y_test):
    best_model_name = None
    best_accuracy = 0

    for name, model in trained_models.items():
        accuracy = model.score(X_test, y_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name

    best_model = trained_models[best_model_name]
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    return best_model_name, best_accuracy

def main():
    X, y = load_data('data/breast_cancer.csv')
    trained_models, X_test, y_test = train_models(X, y)
    best_model_name, best_accuracy = save_best_model(trained_models, X_test, y_test)
    print(f"Best Model: {best_model_name} with accuracy: {best_accuracy:.2f}")

if __name__ == "__main__":
    main()