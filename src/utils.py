def load_model(model_path):
    import pickle
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def save_model(model, model_path):
    import pickle
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

def generate_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()