import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from scipy.stats import mode
import matplotlib.pyplot as plt
import glob 
import os

def load_anomalous_data(file_paths, directory=False):
    """
    Loads the data from a CSV file and creates a label vector Y where all entries are 1.
    Assumes all rows in the file are anomalous examples.
    """

    if directory:
        file_paths = glob.glob(os.path.join(file_paths[0], "*.csv"))
        if len(file_paths) == 0:
            raise RuntimeError("No CSV files found in directory.")

    anomalous_data = None
    for file_path in file_paths:
        data = pd.read_csv(file_path, header=None)
        
        X = data.values  # All columns
        
        if anomalous_data is not None:
            anomalous_data = np.vstack((anomalous_data, X))
        else:
            anomalous_data = X

    Y = np.ones(len(anomalous_data))         # Label 1 for each row
    return anomalous_data, Y

def load_good_data(file_paths, directory=False):
    """
    Loads the data from multiple CSV files and creates a label vector Y where all entries are 0.
    Assumes all rows in the file are good examples.
    """
    if directory:
        file_paths = glob.glob(os.path.join(file_paths[0], "*.csv"))
        if len(file_paths) == 0:
            raise RuntimeError("No CSV files found in directory.")

    good_data = None
    for file_path in file_paths:
        data = pd.read_csv(file_path, header=None)
        
        X = data.values  # All columns
        
        if good_data is not None:
            good_data = np.vstack((good_data, X))
        else:
            good_data = X

    Y = np.zeros(len(good_data))         # Label 0 for each row
    return good_data, Y



def predict(trained_model, X_test):
    """
    Take the already trained anomaly detection model and predict anomalies on the test set.
    """
    
    # Predict anomalies in the test data
    y_pred = trained_model.predict(X_test)
    y_pred = np.where(y_pred == 1, 0.0, 1.0)  # OneClassSVM returns 1 for normal and -1 for anomaly. We need to map to 0 and 1.
    
    return y_pred

def evaluate_model(y_test, y_pred):
    """
    Evaluate the performance of the anomaly detection model and print the classification report.
    """
    print(y_test.shape, y_pred.shape)
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

def ensemble_voting(model_path, scaler_path, X_test):
    file_paths = glob.glob(os.path.join(model_path, "batch_model_*.pkl"))
    if len(file_paths) == 0:
        raise RuntimeError("No CSV files found in directory.")

    scaler = load_scaler(scaler_path)
    y_pred = []
    for file_path in file_paths:
        trained_model = load_trained_model(file_path)


        flattened_x_test = X_test.flatten()
        reshaped = flattened_x_test.reshape(-1, 1)
        X_test_scaled = scaler.transform(reshaped)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        # Train the model and predict anomalies
        y_pred.append(predict(trained_model, X_test_scaled))
    
    # Convert to numpy array for easier manipulation
    y_pred_array = np.array(y_pred)  # Shape: (num_models, num_samples)

    # Majority vote across models
    ensemble_prediction, _ = mode(y_pred_array, axis=0)

    # Flatten the result
    ensemble_prediction = ensemble_prediction.flatten()

    return ensemble_prediction

    


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an anomaly detection model")
    parser.add_argument('--model_path', required=True, type=str, help="Path to the trained model file")
    parser.add_argument('--scaler_path', required=True, type=str, help="Path to the scaler used in training the model")
    parser.add_argument('-e', '--ensemble_voting', action='store_true')
    args = parser.parse_args()
    # Load the data from CSV (ensure 'anomaly' column is the last column)
    X_test, y_test = load_anomalous_data(['evaluation/data/'], directory=True)
    X_test2, y_test2 = load_good_data(['training_data/original_data/vectorized_data.csv'])

    # Concatenate the feature matrices and labels
    X_combined = np.vstack((X_test, X_test2))    # Stack vertically: (n1+n2, cols)
    y_combined = np.concatenate((y_test, y_test2))  # (n1+n2,)

    # Generate a shuffled index array
    indices = np.random.permutation(len(X_combined))

    # Shuffle both X and y using the same indices to preserve alignment
    X_shuffled = X_combined[indices]
    y_shuffled = y_combined[indices]

    X_test = X_shuffled
    y_test = y_shuffled
    # Preprocess the data (standardize or normalize)

    if not args.ensemble_voting:
        trained_model = load_trained_model(args.model_path)

        scaler = load_scaler(args.scaler_path)

        flattened_x_test = X_test.flatten()
        reshaped = flattened_x_test.reshape(-1, 1)
        X_test_scaled = scaler.transform(reshaped)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        # Train the model and predict anomalies
        y_pred = predict(trained_model, X_test_scaled)
    else:
        y_pred = ensemble_voting(args.model_path, args.scaler_path, X_test)

    # Evaluate the model
    evaluate_model(y_test, y_pred)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    #Plot ROC curve 
    plot_roc_curve(y_test, y_pred)





