import os
import random
import glob
import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from itertools import product
from sklearn.metrics import fbeta_score
from .preprocessor import Preprocessor
from .base_model_trainer import BaseModelTrainer
from .encoder_trainer import EncoderTrainer
from .data_loader import DataLoader 
from data_augmentation.data_augmenter import DataAugmenter
from .config import AUGMENTED_DATA_DIR, OUTPUT_MODEL_PREFIX, SAMPLES_PER_FILE, NUM_BATCHES, SEED, TRAINING_TESTS_SPLITS_DIRECTORY, HYPERPARAMETER_FILEPATH, VALIDATION_DATA_DIR
from training_test_splits.data_split_generation.config import NUM_SPLITS
from log.utils import catch_and_log, make_summary
from training_test_splits.data_split_generation.test_data_split_generator import TestDataSplitGenerator
from evaluation.metrics_evaluator import MetricsEvaluator
from plotting.utils import boxplot

class TrainingManager:
    def __init__(self, args):
        self.args = args
        self.models = []
        self.hyperparameter_tuning = args.hyperparameter_tuning
        if self.args.one_svm:
            self.models.append("one_svm")
        if self.args.isolation_forest:
            self.models.append("isolation_forest")
        if self.args.local_outlier:
            self.models.append("LOF")
        if self.args.autoencoder is not None:
            self.models.append("autoencoder")

        self.hyperparameters = self.load_hyperparameter_config()  # Load hyperparameters
        self.encoder_name = self.args.encoder
        self.encoding_dim = self.args.encoding_dim
        self.stats = {}
        self.train_indices = {}
        self.data_loader = DataLoader(AUGMENTED_DATA_DIR, SAMPLES_PER_FILE)
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_hyperparameter_config(file_path=HYPERPARAMETER_FILEPATH):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @catch_and_log(Exception, "Loading validation data")
    def get_validation_data(self):
        # Load all anomaly files in the directory
        anomaly_dfs = []
        for file in Path(VALIDATION_DATA_DIR).glob("*_validation.csv"):
            df = pd.read_csv(file, header=None)
            anomaly_dfs.append(df)
        
        anomalies = pd.concat(anomaly_dfs, axis=0).reset_index(drop=True)
        return anomalies


    @catch_and_log(Exception, "Training complex model with hyperparameter tuning")
    def train_model_with_hyperparameter_tuning(self):
        best_config = None
        best_performance = -float('inf')  # To track the best performance
        best_threshold = 0.5  # Initialize with a default threshold

        for model_type in self.models:
            # Get hyperparameter combinations for the current model
            param_grid = self.hyperparameters.get(model_type, {})
            
            # Generate all combinations of hyperparameters
            all_combinations = list(product(*param_grid.values()))
            
            # Create a DataFrame to track performances
            performance_data = []
            f2_scores = []

            for params in all_combinations:
                # Create a dict of current parameter combination
                param_dict = dict(zip(param_grid.keys(), params))

                # Store this config in the model directory
                models_dir = "models/"
                model_config = f"{'_'.join(f'{k}_{v}' for k, v in param_dict.items())}/"
                model_config_dir = f"{model_type}/hyperparameter_tuning/{model_config}"

                if self.args.augment_techniques is not None:
                    augmented_type = "_".join(self.args.augment_techniques) + f"_{self.args.augment_factor}/"

                    augmented_dir = "augmented/" + augmented_type 
                else:
                    augmented_dir = ""
                
                avg_val_performance = 0

                # Perform training and validation for each split
                all_reconstruction_errors = []
                all_ground_truths = []

                dir_prefix = models_dir

                split_results = []

                for split in range(1, NUM_SPLITS + 1):
                    dir_suffix = augmented_dir + f"split_{split}/"
                    split_dir = os.path.join(TRAINING_TESTS_SPLITS_DIRECTORY, f"split_{split}")
                    training_data_location = os.path.join(split_dir, "train.csv")
                    df = self.data_loader.load_original_data(training_data_location)

                    # Train-validation split logic
                    # Retrieve anomalous entries for validation set
                    anomalous_df = self.get_validation_data()

                    train_data, val_data, val_data_95 = TestDataSplitGenerator().generate_splits(df, anomalous_df, n_splits=1, train_size=0.9, save_to_csv=False)

                    X_val_test = val_data.iloc[:, :-1].values.astype(float)  # all columns except last
                    y_val_test = val_data.iloc[:, -1].values.astype(int)     # last column = label (0 or 1)

                    self.logger.info(f"Validation test data shape: {X_val_test.shape}, Labels: {np.bincount(y_val_test)}")


                    if self.args.augment_techniques != ["none"] and self.args.augment_factor is not None:
                        train_data = DataAugmenter().augment_dataset(train_data, self.args.augment_techniques, self.args.augment_factor)

                    # Preprocess data
                    preprocessor = Preprocessor()
                    X_train_scaled = preprocessor.fit_transform(train_data.values)
                    X_val_test_scaled = preprocessor.transform(X_val_test)

                    if self.encoder_name is not None:
                        dir_prefix += f"hybrid/{self.encoder_name}/"
                        encoder_model_dir = dir_prefix + model_config_dir + dir_suffix
                        encoder_path = os.path.join(encoder_model_dir, "encoder.pkl")
                        encoder = EncoderTrainer(self.encoder_name, self.encoding_dim, self.stats)
                        encoded_data = encoder.run(X_train_scaled, encoder_path)
                        X_scaled = encoded_data

                    model_dir = dir_prefix + model_config_dir + dir_suffix
                    scaler_path = model_dir + "scaler.pkl"
                    preprocessor.save(scaler_path)

                    # Train model with current hyperparameters
                    trainer = BaseModelTrainer(model_type, param_dict, self.stats)
                    model_path = model_dir + "model.pkl"
                    model = trainer.run(X_train_scaled, model_path, self.train_indices, return_model=True)

                    # Get the reconstruction errors for this validation split
                    reconstruction_errors = self.get_reconstruction_errors(model, X_val_test_scaled)
                    # Assuming the label is in 'label' column

                    all_reconstruction_errors.append(reconstruction_errors)
                    all_ground_truths.append(y_val_test)

                    



                # Combine all reconstruction errors and ground truths across splits
                all_reconstruction_errors_combined = np.concatenate(all_reconstruction_errors)
                all_ground_truths_combined = np.concatenate(all_ground_truths)

                # Optimize threshold across all validation splits
                optimized_threshold, avg_f2, predictions = self.optimize_threshold(all_reconstruction_errors_combined, all_ground_truths_combined)

                # Compute per-split F2 using the global threshold
                for i, (split_errors, split_truths) in enumerate(zip(all_reconstruction_errors, all_ground_truths)):
                    # Apply global threshold
                    split_preds = (split_errors >= optimized_threshold).astype(int)
                    
                    # Compute F2 score
                    f2 = fbeta_score(split_truths, split_preds, beta=2, pos_label=1)
                    
                    f2_scores.append({
                        "model": model_config,
                        "split": i,
                        "f2_score": f2
                    })

                performance_data.append((param_dict, avg_f2, optimized_threshold))

                # Track best performance and configuration
                if avg_f2 > best_performance:
                    best_performance = avg_f2
                    best_config = param_dict
                    best_threshold = optimized_threshold

            
            hyperprameter_results = pd.DataFrame(f2_scores)
            x = "model" 
            y = "f2_score"
            boxplot(hyperprameter_results, x, y, dir_path= "hyperparameter_tuning_results/" + dir_prefix + model_type, fifty_fifty=True)

            # Log and display the best hyperparameter configuration for the model
            self.logger.info(f"Best configuration for {model_type}: {best_config} with F1 score: {best_performance} and threshold: {best_threshold}")

            # Now evaluate on the test set with the best configuration
            best_config["threshold"] = best_threshold
            self.model_args = best_config
            self.train_single_model()

    def get_reconstruction_errors(self, model, X_val_scaled):
        # Reconstruct data using the autoencoder model and return reconstruction errors
        reconstruction_errors = model.predict(X_val_scaled)
        return reconstruction_errors

    @catch_and_log(Exception, "Optimising threshold")
    def optimize_threshold(self, reconstruction_errors, ground_truth):
        """
        Optimize the threshold for anomaly detection.
        This function finds the optimal threshold based on the range of reconstruction errors in the validation set.
        """
        # Get the minimum and maximum reconstruction errors
        min_error = np.min(reconstruction_errors)
        max_error = np.max(reconstruction_errors)

        # Normalize the threshold range to be between min and max error
        # We will search for thresholds in the normalized range [0, 1] and then map it back to the original range.
        thresholds = np.linspace(0.0, 1.0, 100)  # 100 thresholds between 0 and 1

        best_f2 = -float('inf')
        best_threshold = 0.5  # Default threshold
        best_predictions = None

        for threshold in thresholds:
            # Map the threshold back to the original error scale
            adaptive_threshold = min_error + threshold * (max_error - min_error)

            # Apply the adaptive threshold to classify as normal or anomalous
            predictions = (reconstruction_errors > adaptive_threshold).astype(int)

            # Calculate F2 score for each row (example) and then average it
           
            f2_scores = [fbeta_score(gt, pred, beta=2, pos_label=1) for gt, pred in zip(ground_truth, predictions)]
            avg_f2 = np.mean(f2_scores)

            
            # Update best F1 score and threshold
            if avg_f2 > best_f2:
                best_f2 = avg_f2
                best_threshold = adaptive_threshold
                best_predictions = predictions

        return best_threshold, best_f2, best_predictions


    @catch_and_log(Exception, "Training regular model")
    def train_single_model(self):

        model_name = f"model"

        if self.args.augment_techniques is not None:
            augmented_type = "_".join(self.args.augment_techniques) + f"_{self.args.augment_factor}/"

            augmented_dir = "augmented/" + augmented_type 
        else:
            augmented_dir = ""


        for split in range(1, NUM_SPLITS + 1):
            split_dir = os.path.join(TRAINING_TESTS_SPLITS_DIRECTORY, f"split_{split}")
            training_data_location = os.path.join(split_dir, "train.csv")

            df = self.data_loader.load_original_data(training_data_location)

            train_data = df

            if self.args.augment_techniques != ["none"] and self.args.augment_factor is not None:
                train_data = DataAugmenter().augment_dataset(train_data, self.args.augment_techniques, self.args.augment_factor)
                
            # # Split train/test
            # indices = np.random.choice(df.index, size=int(0.8 * len(df)), replace=False)
            # train_data = df.loc[indices]
            # self.train_indices[ORIGINAL_DATA_FILE_PATH] = indices.tolist()

            dir_suffix = augmented_dir + f"split_{split}/"

            dir_prefix = "models/"

            X_train = train_data.values.copy()


            # Preprocess once
            preprocessor = Preprocessor()
            X_scaled = preprocessor.fit_transform(X_train)

            if self.encoder_name is not None:
                dir_prefix += f"hybrid/{self.encoder_name}_{self.encoding_dim}/"
                encoder_model_dir = dir_prefix + f"shared_encoder/split_{split}/"
                encoder_path = os.path.join(encoder_model_dir, "encoder.pkl")
                encoder = EncoderTrainer(self.encoder_name, self.encoding_dim, self.stats)
                encoded_data = encoder.run(X_scaled, encoder_path)
                X_scaled = encoded_data
            
        
            
            scaler_path = "scalers/" + dir_suffix + "scaler.pkl"
            preprocessor.save(scaler_path)

            # Train each selected model
            for model_type in self.models:
                model_dir = dir_prefix + f"{model_type}/"

                if model_type == "autoencoder":
                    # Prepare hyperparameters to save in JSON format
                    hyperparameters = {
                        "lr": self.model_args["lr"],
                        "batch_size": self.model_args["batch_size"],
                        "num_epochs": self.model_args["num_epochs"],
                        "encoding_dim": self.model_args["encoding_dim"]
                    }

                    # Create model_dir if it doesn't exist
                    os.makedirs(model_dir, exist_ok=True)

                    # Define the path for the config JSON file
                    config_file_path = os.path.join(model_dir, "hyperparameters.json")

                    # Save the hyperparameters to a JSON file
                    with open(config_file_path, 'w') as json_file:
                        json.dump(hyperparameters, json_file, indent=4)  # indent for better readability

                    # model_dir += f"{self.model_args["lr"]}_{self.model_args["batch_size"]}_{self.model_args["num_epochs"]}_{self.model_args["encoding_dim"]}/"
                model_dir += dir_suffix
                # preprocessor.save(model_dir + f"scaler_{model_name}.pkl")
                trainer = BaseModelTrainer(model_type, self.model_args, self.stats)
                model_path = model_dir + f"{model_name}.pkl"
                trainer.run(X_scaled, model_path, self.train_indices)

    @catch_and_log(Exception, "Training ensemble models")
    def train_ensemble_batches(self):

        for split in range(1, NUM_SPLITS + 1):
            csv_files = glob.glob(os.path.join(AUGMENTED_DATA_DIR, "*.csv"))
            if not csv_files:
                raise RuntimeError("No CSV files found in augmented data directory.")

            random.seed(SEED)

            for batch_idx in range(NUM_BATCHES):
                self.logger.info(f"\n=== Starting batch {batch_idx + 1}/{NUM_BATCHES} ===")

                batch_data, first_row_set = self.data_loader.load_batch_parallel(csv_files)

                if not batch_data:
                    self.logger.info(f"No data collected for batch {batch_idx}")
                    continue

                df = pd.concat(batch_data, ignore_index=True)
                # Store first-row indices
                # indexes = [
                #     int(str(os.path.basename(file_path)).removeprefix("row_").removesuffix(".csv"))
                #     for file_path in first_row_set
                # ]
                # self.train_indices[ORIGINAL_DATA_FILE_PATH] = indexes

                # Preprocess once per batch
                preprocessor = Preprocessor()
                
                X_scaled = preprocessor.fit_transform(df.values.copy())


                model_name = f"{OUTPUT_MODEL_PREFIX}_{batch_idx:03d}"
                for model_type in self.models:
                    model_dir = f"models/{model_type}/ensemble/split_{split}/"
                    preprocessor.save(model_dir + f"scaler_{model_path}.pkl")
                    trainer = BaseModelTrainer(model_type, self.stats)
                    model_path = model_dir + f"{model_name}.pkl"
                    trainer.run(X_scaled, model_path, self.train_indices)
            
            


    def run(self):
        if self.args.train_ensemble:
            self.train_ensemble_batches()
            make_summary("Training Model Summary", self.stats)
            return 
    
        if self.args.hyperparameter_tuning:
            self.train_model_with_hyperparameter_tuning()
        else:
            self.train_single_model()

        make_summary("Training Model Summary", self.stats)
