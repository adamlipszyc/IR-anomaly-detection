import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def load_data(file_path):
    """
    Load the data from a CSV file.
    
    Args:
    - file_path (str): Path to the CSV file containing the data.
    
    Returns:
    - data (np.ndarray): Loaded data.
    """
    data = pd.read_csv(file_path, header=None).values
    return data

def build_autoencoder(input_dim, encoding_dim=32):
    """
    Build and compile an autoencoder model for anomaly detection.
    
    Args:
    - input_dim (int): The dimension of input data.
    - encoding_dim (int): The dimension of the encoded representation.
    
    Returns:
    - autoencoder (keras.Model): The autoencoder model.
    """
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')
    
    return autoencoder

def train_autoencoder(data, encoding_dim=32, epochs=50, batch_size=256):
    """
    Train an autoencoder on the data.
    
    Args:
    - data (np.ndarray): The data to train the autoencoder.
    - encoding_dim (int): Dimension of the encoded representation.
    - epochs (int): Number of epochs for training.
    - batch_size (int): Batch size for training.
    
    Returns:
    - autoencoder (keras.Model): The trained autoencoder model.
    """
    input_dim = data.shape[1]
    autoencoder = build_autoencoder(input_dim, encoding_dim)
    autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, shuffle=True)
    return autoencoder

def compute_reconstruction_error(autoencoder, data):
    """
    Compute the reconstruction error for each data point.
    
    Args:
    - autoencoder (keras.Model): The trained autoencoder.
    - data (np.ndarray): The data for which to calculate the reconstruction error.
    
    Returns:
    - reconstruction_error (np.ndarray): The reconstruction error for each data point.
    """
    reconstructed = autoencoder.predict(data)
    reconstruction_error = np.mean(np.square(data - reconstructed), axis=1)
    return reconstruction_error

def plot_reconstruction_error(reconstruction_error, threshold=None):
    """
    Plot the reconstruction error and optionally display an anomaly detection threshold.
    
    Args:
    - reconstruction_error (np.ndarray): The reconstruction error for each data point.
    - threshold (float): Optional threshold value for anomaly detection.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(reconstruction_error, bins=50, alpha=0.7)
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    
    if threshold is not None:
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
        plt.legend()
    
    plt.show()

def detect_anomalies(reconstruction_error, threshold):
    """
    Detect anomalies based on the reconstruction error threshold.
    
    Args:
    - reconstruction_error (np.ndarray): The reconstruction error for each data point.
    - threshold (float): The threshold above which data points are considered anomalies.
    
    Returns:
    - anomalies (np.ndarray): An array with -1 for anomalies and 1 for normal data points.
    """
    anomalies = np.where(reconstruction_error > threshold, -1, 1)
    return anomalies

def main():
    # Load the data
    file_path = 'training_data/vectorized_data.csv'  # Replace with your actual data file path
    data = load_data(file_path)
    
    # Standardize the data (important for neural networks)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Train the autoencoder
    autoencoder = train_autoencoder(data_scaled, encoding_dim=32, epochs=50, batch_size=256)
    
    # Compute reconstruction error on the training data
    reconstruction_error = compute_reconstruction_error(autoencoder, data_scaled)
    
    # Plot the reconstruction error
    plot_reconstruction_error(reconstruction_error)
    
    # Set a threshold based on the distribution of reconstruction errors
    threshold = np.percentile(reconstruction_error, 95)  # You can adjust this percentile
    print(f"Anomaly detection threshold set at: {threshold}")
    
    # Detect anomalies
    anomalies = detect_anomalies(reconstruction_error, threshold)
    print(f"Detected anomalies: {np.sum(anomalies == -1)} anomalies detected.")
    
    # Optionally: Evaluate the results if ground truth is available
    # For demonstration purposes, we will assume that ground truth labels are available in a separate file.
    # ground_truth = load_data('ground_truth_labels.csv')  # 1 for normal, -1 for anomaly
    # roc_score = roc_auc_score(ground_truth, reconstruction_error)
    # print(f"ROC AUC Score: {roc_score}")

if __name__ == "__main__":
    main()
