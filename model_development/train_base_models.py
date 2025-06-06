import numpy as np
import pandas as pd
import pickle
import argparse 
import random
import os
import glob
import sys
import argparse
import logging
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from multiprocessing import Process, Queue, cpu_count
from .config import ORIGINAL_DATA_FILE_PATH

# Configure root logger to use Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    datefmt="[%H:%M:%S]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)

    

def min_max_normalize_data(array, scaler_file_paths):
    """
    Normalize numerical columns using Min-Max scaling.
    Default is to normalize all numerical columns.
    """
    array_reshaped = array.reshape(-1, 1)
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(array_reshaped).flatten()

    for scaler_file_path in scaler_file_paths:
        with open(scaler_file_path, 'wb') as file:
            pickle.dump(scaler, file)
        
        logger.info("Scaler saved: %s", scaler_file_path)
    

    return normalized_array

# Function to standardize the data
def standardize_data(array):
    """
    Standardize numerical columns to have mean=0 and variance=1.
    Default is to standardize all numerical columns.
    """

    array_reshaped = array.reshape(-1, 1)
    scaler = StandardScaler()
    standardized_array = scaler.fit_transform(array_reshaped).flatten()
    return standardized_array


#need to perform hyper-parameter tuning 
def train_one_class_svm(data):
    """
    Train One-Class SVM model.
    :param data: Preprocessed data
    :return: Trained One-Class SVM model
    """
    model = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
    model.fit(data)
    return model

def train_isolation_forest(data):
    """
    Train Isolation Forest model.
    :param data: Preprocessed data
    :return: Trained Isolation Forest model
    """
    model = IsolationForest(contamination=0.1)
    #contamination is what proportion of the data the model should expect to be anomalous during testing 
    #this has no effect during training
    model.fit(data)
    return model

def train_lof(data, n_neighbors=20, contamination=0.05):
    """
    Train the Local Outlier Factor (LOF) model for anomaly detection.
    
    Args:
    - data (pd.DataFrame): The input data, where each row is a 1100-dimensional vector.
    - n_neighbors (int): The number of neighbors to use for LOF. Default is 20.
    - contamination (float): The proportion of outliers in the data. Default is 0.05.
    
    Returns:
    - lof_model (LocalOutlierFactor): The trained LOF model.
    """
    lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    lof_model.fit(data)
    return lof_model

def save_model(model, filepath, num_rows, train_indices = {}):
    """
    Save the model to the specified file path using pickle
    """
    with open(filepath, "wb") as file:
            pickle.dump(model, file)
    
    logger.info("Saved model: %s | Trained on %s rows", filepath, num_rows)

    if train_indices:
        indices_destination = filepath[:-4] + "_indices.json"
        # Save used indices
        with open(indices_destination, 'w') as file:
            json.dump(train_indices, file)
        
        logger.info("Saved model indices: %s", indices_destination)

# === CONFIG ===
DATA_DIR = "data_augmentation/augmented_data_without_noise/" 
SAMPLES_PER_FILE = 20                # Rows to sample per file
NUM_BATCHES = 10                         # Total number of models to train
OUTPUT_MODEL_PREFIX = "batch_model"      # Prefix for saved models
SEED = 42                                # Reproducibility


def worker(input_queue: Queue, output_queue: Queue, first_row_sampled_set: set[str]):
    """Worker function to read and sample CSV files from the input queue."""
    while True:
        file_path = input_queue.get()
        if file_path is None:
            break  # Sentinel value to shut down
        try:
            df = pd.read_csv(file_path, header=None)
            if len(df) > SAMPLES_PER_FILE:
                df = df.sample(SAMPLES_PER_FILE, random_state=random.randint(0, 99999))
            output_queue.put(df)

            # Check if first row (index 0) is in the sampled dataframe
            if 0 in df.index:
                first_row_sampled_set.add(file_path)
        except Exception as e:
            print(f"Skipping {file_path} due to error: {e}")
        finally:
            output_queue.put(None)  # Signal that one file is done (success or skip)

def process_files_in_parallel(csv_files, num_workers=None):
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    input_queue = Queue()
    output_queue = Queue()
    first_row_sampled_set = set()
    batch_data = []

    # Start workers
    workers = []
    for _ in range(num_workers):
        p = Process(target=worker, args=(input_queue, output_queue, first_row_sampled_set))
        p.start()
        workers.append(p)

    # Enqueue file paths
    for file_path in csv_files:
        input_queue.put(file_path)

    # Send sentinel values to stop workers
    for _ in range(num_workers):
        input_queue.put(None)

    # Collect data with progress bar
    with tqdm(total=len(csv_files), desc="Processing files") as pbar:
        finished_files = 0
        while finished_files < len(csv_files):
            result = output_queue.get()
            if result is None:
                finished_files += 1
                pbar.update(1)
            else:
                batch_data.append(result)

    # Clean up
    for p in workers:
        p.join()
    

    print("Done with files")
    return (batch_data, first_row_sampled_set)


def train_model(stats, df, model_path, isolation_forest = False, one_svm = False, lof = False, train_indices: dict = {}) -> None: 
    
    

    training_data = df.values.copy() 

    np.random.shuffle(training_data)

    

    scaler_name = f"scaler_{model_path}.pkl"
    scaler_destinations = []
    if isolation_forest:
        scaler_destinations.append(f"models/isolation_forest/{scaler_name}")
    
    if one_svm:
        scaler_destinations.append(f"models/one_svm/{scaler_name}")
    
    if lof:
        scaler_destinations.append(f"models/LOF/{scaler_name}")
    
    #flatten our data set into one large 1D array 
    flattened_training_data = training_data.flatten()
    
    #normalize our entire dataset 
    normalized_data = min_max_normalize_data(flattened_training_data, scaler_destinations)

    #reshape the 1d array back to its original shape
    reshaped_data = normalized_data.reshape(training_data.shape)
    
    length = len(df)

    if isolation_forest:
        #Train the model
        model = train_isolation_forest(reshaped_data)

        destination = f"models/isolation_forest/{model_path}.pkl"


        save_model(model, destination, length, train_indices)
        

        stats["Isolation Forest model build"] = "Success"

    if one_svm:
        #Train the model
        model = train_one_class_svm(reshaped_data)

        destination = f"models/one_svm/{model_path}.pkl"

        save_model(model, destination, length, train_indices)

        stats["One-SVM model build"] = "Success"
        

    if lof:
        #Train the model
        model = train_lof(reshaped_data)

        destination = f"models/LOF/{model_path}.pkl"

        save_model(model, destination, length, train_indices)

        stats["Local Outlier Factor model build"] = "Success"   

def make_summary(stats: dict) -> None:
    """
    Display a summary table of generation outcomes.
    """
    console = Console()
    table = Table(title="Scenario Generation Summary")
    table.add_column("Task", style="cyan")
    table.add_column("Status", style="magenta", justify="right")
    for task, status in stats.items():
        table.add_row(task, status)
    console.print(table)


def main() -> None:
    """
    This will train a specified model, by first transforming the data into a large 
    1D array then normalizing - so all data is scaled the same, and then training 
    on this normalised array. 

    Can train ensemble voting or regular 

    """
    # Configure Rich-powered logging
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    log = logging.getLogger("main")

    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--one_svm', action='store_true')
    parser.add_argument('-i', '--isolation_forest', action='store_true')
    parser.add_argument('-l', '--local_outlier', action='store_true')
    parser.add_argument('-a', '--train_augmented', action='store_true')

    args = parser.parse_args()

    if not args.one_svm and not args.isolation_forest and not args.local_outlier:
        parser.error("You must specify at least one of -o, -i or -l")

    try:
        stats = {}
        train_indices = {}

        if args.train_augmented:
            # === GATHER FILES ===
            csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
            if len(csv_files) == 0:
                raise RuntimeError("No CSV files found in directory.")

            random.seed(SEED)

            # === TRAIN MODELS ===
            for batch_idx in range(NUM_BATCHES):
                print(f"Starting batch {batch_idx + 1}/{NUM_BATCHES}...")

                batch_data = []

                batch_data, first_row_set = process_files_in_parallel(csv_files)
                
                indexes = [int(str(os.path.basename(file_path)).removeprefix("row_").removesuffix(".csv")) for file_path in first_row_set]

                train_indices[ORIGINAL_DATA_FILE_PATH] = indexes

                print("Done with files")
                if not batch_data:
                    print(f"No data collected for batch {batch_idx}")
                    continue

                df = pd.concat(batch_data, ignore_index=True)

                model_path = f"{OUTPUT_MODEL_PREFIX}_{batch_idx:03d}"

                train_model(stats, df, model_path, args.isolation_forest, args.one_svm, args.local_outlier, train_indices)
                
                make_summary(stats)

            return
        

        df = pd.read_csv(ORIGINAL_DATA_FILE_PATH, header=None)

        #Randomly select 80% of the rows and leave 20% as a held-out test set
        indices = np.random.choice(df.index, size=int(0.8 * len(df)), replace=False)
        train_data = df.loc[indices]
        train_indices[ORIGINAL_DATA_FILE_PATH] = indices.tolist()
        df = train_data

        model_path = "model"

        train_model(stats, df, model_path, args.isolation_forest, args.one_svm, args.local_outlier, train_indices)

        make_summary(stats)
       
    except Exception as e:
        log.exception("Unexpected failure")
        sys.exit(99)


if __name__ == "__main__":
    main()
        
