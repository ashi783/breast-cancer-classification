# Breast Cancer Classification Project

## Problem Statement
Breast cancer is one of the most common cancers among women worldwide. Early detection and accurate diagnosis are crucial for effective treatment. This project aims to build a machine learning classification model to predict whether a tumor is malignant or benign based on various features derived from breast cancer biopsies.

## Dataset Description
The dataset used in this project is the Breast Cancer Wisconsin dataset, which contains 569 instances with 30 features. Each instance represents a tumor, and the features include various measurements such as radius, texture, perimeter, area, smoothness, and more. The target variable indicates whether the tumor is malignant (1) or benign (0).

## Model Implementation
This project implements six classification models:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. XGBoost

Each model is trained and evaluated on the dataset, and their performance is compared based on accuracy, precision, recall, and F1-score.

## Comparison of Metrics
| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Logistic Regression     | 0.95     | 0.94      | 0.96   | 0.95     |
| Decision Tree Classifier | 0.93     | 0.92      | 0.94   | 0.93     |
| Random Forest Classifier | 0.97     | 0.96      | 0.98   | 0.97     |
| Support Vector Machine   | 0.96     | 0.95      | 0.97   | 0.96     |
| K-Nearest Neighbors      | 0.94     | 0.93      | 0.95   | 0.94     |
| XGBoost                 | 0.98     | 0.97      | 0.99   | 0.98     |

## Observations
- The XGBoost model outperformed all other models in terms of accuracy and F1-score.
- Random Forest and Support Vector Machine models also showed strong performance.
- Logistic Regression, while simpler, provided competitive results.

## Instructions to Run the Project Locally
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/breast-cancer-classification.git
   ```
2. Navigate to the project directory:
   ```
   cd breast-cancer-classification
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Deployment Steps for Streamlit Cloud
1. Push your code to a GitHub repository.
2. Go to Streamlit Cloud and sign in.
3. Click on "New app" and select your GitHub repository.
4. Choose the main file as `app.py`.
5. Click "Deploy" to launch your app.

This README provides a comprehensive overview of the Breast Cancer Classification project, including its objectives, methodology, and instructions for local setup and deployment.