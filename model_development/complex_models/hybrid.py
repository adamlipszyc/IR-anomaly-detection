# Install necessary packages
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense




def load_data(file_path):
    """
    Load the vectorized training data from a CSV file.
    
    Args:
    - file_path (str): Path to the CSV file containing the data.
    
    Returns:
    - data (pd.DataFrame): Loaded training data in a pandas DataFrame.
    """
    data = pd.read_csv(file_path, header=None)
    return data

def pca_reduction(data, n_components=50):
    """
    Apply PCA for dimensionality reduction.
    
    Args:
    - data (pd.DataFrame): Input data to be reduced.
    - n_components (int): Number of principal components to retain.
    
    Returns:
    - reduced_data (np.ndarray): Data after PCA transformation.
    """
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data

def autoencoder_model(input_dim, encoding_dim=32):
    """
    Build an autoencoder model for dimensionality reduction.
    
    Args:
    - input_dim (int): Input dimension (number of features).
    - encoding_dim (int): Dimension of the encoded representation.
    
    Returns:
    - model (keras.Model): The compiled autoencoder model.
    """
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    return autoencoder

def fit_autoencoder(data, epochs=50, batch_size=256):
    """
    Train an autoencoder model on the data.
    
    Args:
    - data (np.ndarray): The data to train the autoencoder.
    - epochs (int): Number of epochs to train.
    - batch_size (int): Batch size for training.
    
    Returns:
    - autoencoder (keras.Model): The trained autoencoder model.
    """
    input_dim = data.shape[1]
    autoencoder = autoencoder_model(input_dim)
    autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(data, data))
    
    return autoencoder

def hybrid_pca_isolation_forest(data, n_components=50, n_estimators=100):
    """
    Apply PCA for dimensionality reduction and then use Isolation Forest for anomaly detection.
    
    Args:
    - data (pd.DataFrame): The input data.
    - n_components (int): Number of principal components.
    - n_estimators (int): Number of estimators for Isolation Forest.
    
    Returns:
    - predictions (np.ndarray): Anomaly detection results (-1 for anomaly, 1 for normal).
    """
    reduced_data = pca_reduction(data, n_components)
    isolation_forest = IsolationForest(n_estimators=n_estimators)
    isolation_forest.fit(reduced_data)
    predictions = isolation_forest.predict(reduced_data)
    return predictions

def hybrid_autoencoder_isolation_forest(data, epochs=50, batch_size=256, n_estimators=100):
    """
    Apply Autoencoder for dimensionality reduction and then use Isolation Forest for anomaly detection.
    
    Args:
    - data (pd.DataFrame): The input data.
    - epochs (int): Number of epochs to train the autoencoder.
    - batch_size (int): Batch size for training.
    - n_estimators (int): Number of estimators for Isolation Forest.
    
    Returns:
    - predictions (np.ndarray): Anomaly detection results (-1 for anomaly, 1 for normal).
    """
    autoencoder = fit_autoencoder(data, epochs, batch_size)
    encoded_data = autoencoder.predict(data)
    isolation_forest = IsolationForest(n_estimators=n_estimators)
    isolation_forest.fit(encoded_data)
    predictions = isolation_forest.predict(encoded_data)
    return predictions

def hybrid_pca_lof(data, n_components=50, n_neighbors=20):
    """
    Apply PCA for dimensionality reduction and then use Local Outlier Factor (LOF) for anomaly detection.
    
    Args:
    - data (pd.DataFrame): The input data.
    - n_components (int): Number of principal components.
    - n_neighbors (int): Number of neighbors for LOF.
    
    Returns:
    - predictions (np.ndarray): Anomaly detection results (-1 for anomaly, 1 for normal).
    """
    reduced_data = pca_reduction(data, n_components)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    predictions = lof.fit_predict(reduced_data)
    return predictions

def hybrid_autoencoder_lof(data, epochs=50, batch_size=256, n_neighbors=20):
    """
    Apply Autoencoder for dimensionality reduction and then use Local Outlier Factor (LOF) for anomaly detection.
    
    Args:
    - data (pd.DataFrame): The input data.
    - epochs (int): Number of epochs to train the autoencoder.
    - batch_size (int): Batch size for training.
    - n_neighbors (int): Number of neighbors for LOF.
    
    Returns:
    - predictions (np.ndarray): Anomaly detection results (-1 for anomaly, 1 for normal).
    """
    autoencoder = fit_autoencoder(data, epochs, batch_size)
    encoded_data = autoencoder.predict(data)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    predictions = lof.fit_predict(encoded_data)
    return predictions

def hybrid_pca_ocsvm(data, n_components=50, nu=0.1, kernel="rbf"):
    """
    Apply PCA for dimensionality reduction and then use One-Class SVM for anomaly detection.
    
    Args:
    - data (pd.DataFrame): The input data.
    - n_components (int): Number of principal components.
    - nu (float): An upper bound on the fraction of margin errors.
    - kernel (str): The kernel type for SVM.
    
    Returns:
    - predictions (np.ndarray): Anomaly detection results (-1 for anomaly, 1 for normal).
    """
    reduced_data = pca_reduction(data, n_components)
    ocsvm = OneClassSVM(nu=nu, kernel=kernel)
    ocsvm.fit(reduced_data)
    predictions = ocsvm.predict(reduced_data)
    return predictions

def hybrid_autoencoder_ocsvm(data, epochs=50, batch_size=256, nu=0.1, kernel="rbf"):
    """
    Apply Autoencoder for dimensionality reduction and then use One-Class SVM for anomaly detection.
    
    Args:
    - data (pd.DataFrame): The input data.
    - epochs (int): Number of epochs to train the autoencoder.
    - batch_size (int): Batch size for training.
    - nu (float): An upper bound on the fraction of margin errors.
    - kernel (str): The kernel type for SVM.
    
    Returns:
    - predictions (np.ndarray): Anomaly detection results (-1 for anomaly, 1 for normal).
    """
    autoencoder = fit_autoencoder(data, epochs, batch_size)
    encoded_data = autoencoder.predict(data)
    ocsvm = OneClassSVM(nu=nu, kernel=kernel)
    ocsvm.fit(encoded_data)
    predictions = ocsvm.predict(encoded_data)
    return predictions


#USING PCA + Auto encoder 

# ==== STEP 1: Dimensionality Reduction (PCA) ====
def apply_pca(X, n_components=50):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X), pca

# ==== STEP 2: Autoencoder ====
def build_autoencoder(input_dim, encoding_dim=32):
    model = Sequential([
        Dense(encoding_dim, activation='relu', input_shape=(input_dim,)),
        Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ==== STEP 3: Hybrid Model Evaluation ====
def hybrid_score(autoencoder, pca_X, X):
    reconstructed = autoencoder.predict(X)
    reconstruction_error = np.mean((X - reconstructed)**2, axis=1)

    iforest = IsolationForest(contamination=0.05)
    iforest.fit(pca_X)
    iso_scores = iforest.decision_function(pca_X)

    # Normalize and combine scores
    hybrid = (reconstruction_error - reconstruction_error.min()) / (reconstruction_error.max() - reconstruction_error.min())
    iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())

    return 0.5 * hybrid + 0.5 * (1 - iso_scores)

# Example Usage
# X is your (n_samples x 1100) matrix
# X_pca, pca_model = apply_pca(X, 50)
# ae = build_autoencoder(input_dim=1100)
# ae.fit(X, X, epochs=20, batch_size=32)
# anomaly_scores = hybrid_score(ae, X_pca, X)
