import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Loads the data from a CSV file and splits the features and labels.
    Assumes the last column is the 'anomaly' field, which is the true label.
    """
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values  # All columns except the last one
    y = data.iloc[:, -1].values   # The last column which is anomaly label
    return X, y

def preprocess_data(X, scaler_type='minmax'):
    """
    Preprocess the data by normalizing/standardizing.
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type {scaler_type}")
    
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def predict(trained_model, X_test, y_test):
    """
    Train the anomaly detection model and predict anomalies on the test set.
    """
    
    # Predict anomalies in the test data
    y_pred = trained_model.predict(X_test)
    y_pred = np.where(y_pred == 1, 0, 1)  # OneClassSVM returns 1 for normal and -1 for anomaly. We need to map to 0 and 1.
    
    return y_pred

def evaluate_model(y_test, y_pred):
    """
    Evaluate the performance of the anomaly detection model and print the classification report.
    """
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # If available, compute ROC AUC
    if len(np.unique(y_test)) == 2:
        print("ROC AUC:", roc_auc_score(y_test, y_pred))

def plot_confusion_matrix(y_test, y_pred):
    """
    Plot the confusion matrix for visual interpretation.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal', 'Anomaly'])
    plt.yticks(tick_marks, ['Normal', 'Anomaly'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(true_labels, predictions):
    """
    Plots the ROC curve for a binary classification model.
    
    Parameters:
    - true_labels: The true labels (0 for normal, 1 for anomaly)
    - predictions: The predicted probabilities for the positive class (anomalies)
    """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    
    # Compute AUC
    auc = roc_auc_score(true_labels, predictions)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random performance)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Function to load a trained model using pickle
def load_trained_model(model_file_path):
    with open(model_file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_scaler(scaler_file_path):
    with open(scaler_file_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an anomaly detection model")
    parser.add_argument('--model_path', required=True, type=str, help="Path to the trained model file")
    parser.add_argument('--scaler_path', required=True, type=str, help="Path to the scaler used in training the model")
    args = parser.parse_args()
    # Load the data from CSV (ensure 'anomaly' column is the last column)
    X_test, y_test = load_data('your_test_data.csv')
    
    

    # Preprocess the data (standardize or normalize)
    trained_model = load_trained_model(args.model_path)

    scaler = load_scaler(args.scaler_path)
    X_test_scaled = scaler.transform(X_test)
    # Train the model and predict anomalies
    y_pred = predict(trained_model, X_test_scaled,  y_test)

    # Evaluate the model
    evaluate_model(y_test, y_pred)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    #Plot ROC curve 
    plot_roc_curve(y_test, y_pred)





