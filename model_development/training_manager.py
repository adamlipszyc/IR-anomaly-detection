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
from sklearn.model_selection import train_test_split
from .preprocessor import Preprocessor
from .base_model_trainer import BaseModelTrainer
from .encoder_trainer import EncoderTrainer
from .data_loader import DataLoader 
from data_augmentation.data_augmenter import DataAugmenter
from .config import AUGMENTED_DATA_DIR, OUTPUT_MODEL_PREFIX, SAMPLES_PER_FILE, NUM_BATCHES, SEED, TRAINING_TESTS_SPLITS_DIRECTORY, HYPERPARAMETER_FILEPATH, VALIDATION_DATA_DIR, base_models, OPTIMAL_HYPERPARAMETER_FILEPATH
from training_test_splits.data_split_generation.config import NUM_SPLITS
from log.utils import catch_and_log, make_summary
from training_test_splits.data_split_generation.test_data_split_generator import TestDataSplitGenerator
from evaluation.metrics_evaluator import MetricsEvaluator
from plotting.utils import boxplot




supervised_models = {"cnn_supervised_1d", "cnn_supervised_2d", "lstm"}

class TrainingManager:
    def __init__(self, args):
        self.args = args
        self.models = []
        if self.args.one_svm:
            self.models.append("one_svm")
        if self.args.isolation_forest:
            self.models.append("isolation_forest")
        if self.args.local_outlier:
            self.models.append("LOF")
        if self.args.autoencoder:
            self.models.append("autoencoder")
        if self.args.anogan:
            self.models.append("anogan")
        if self.args.cnn_anogan:
            self.models.append("cnn_anogan")
        if self.args.cnn_supervised_2d:
            self.models.append("cnn_supervised_2d")
        if self.args.cnn_supervised_1d:
            self.models.append("cnn_supervised_1d")
        if self.args.lstm:
            self.models.append("lstm")

        self.model_args = args.model_args

        self.hyperparameters = self.load_hyperparameter_config()  # Load hyperparameters
        self.encoder_name = self.args.encoder
        self.encoding_dim = self.args.encoding_dim
        self.stats = {}
        self.train_indices = {}
        self.data_loader = DataLoader(AUGMENTED_DATA_DIR, SAMPLES_PER_FILE)
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_hyperparameter_config(self, file_path=HYPERPARAMETER_FILEPATH):
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
        
        if self.args.augment_techniques is not None:
            augmented_type = "_".join(self.args.augment_techniques) + f"_{self.args.augment_factor}/"

            augmented_dir = "augmented/" + augmented_type 
        else:
            augmented_dir = ""

        for model_type in self.models:
            best_config = None
            best_performance = -float('inf')  # To track the best performance
            best_threshold = 0.5  # Initialize with a default threshold

            if not self.args.hyperparameter_tuning:
                self.train_single_model(model_type)
                continue
            # Get hyperparameter combinations for the current model
            param_grid = self.hyperparameters.get(model_type, {}) 

            if self.encoder_name is not None:
                param_grid = {
                    **param_grid,
                    **self.hyperparameters.get("hybrid", {})
                }
            

            # Generate all combinations of hyperparameters
            all_combinations = list(product(*param_grid.values()))
            
            # Create a DataFrame to track performances
            performance_data = []
            f2_scores = []

            for params in all_combinations:
                # Create a dict of current parameter combination
                param_dict = dict(zip(param_grid.keys(), params))

                print("Testing configuration", json.dumps(param_dict))
                # Store this config in the model directory
                models_dir = "models/"
                model_config = f"{'_'.join(f'{k}_{v}' for k, v in param_dict.items())}/"
                model_config_dir = f"{model_type}/hyperparameter_tuning/{augmented_dir}{model_config}"

                
                avg_val_performance = 0

                # Perform training and validation for each split
                all_reconstruction_errors = []
                all_ground_truths = []

                dir_prefix = models_dir

                split_results = []

                param_dict_copy = param_dict.copy()

                if self.encoder_name is not None:
                    param_dict_copy.pop("encoding_dim")
                    

                for split in range(1, NUM_SPLITS + 1):
                    dir_suffix = f"split_{split}/"
                    split_dir = os.path.join(TRAINING_TESTS_SPLITS_DIRECTORY, f"split_{split}")
                    training_data_location = os.path.join(split_dir, "train.csv")
                    df = self.data_loader.load_original_data(training_data_location)

                    if model_type in supervised_models:
                        anomalous_entries_path = os.path.join(split_dir, "train_anomalies.csv")
                        anomalous_entries_df = self.data_loader.load_original_data(anomalous_entries_path)
                        
                        # Label the data
                        df["label"] = 0  # Good data
                        anomalous_entries_df["label"] = 1  # Anomalous data

                        # Combine and shuffle deterministically
                        combined_df = pd.concat([df, anomalous_entries_df], axis=0, ignore_index=True)
                        train_data, val_data = train_test_split(
                            combined_df,
                            train_size=0.9,
                            random_state=split,  # Ensures reproducibility across runs
                            stratify=combined_df["label"]  # keeps class balance
                        )
                        
                        
                    else:
                        # Train-validation split logic
                        # Retrieve anomalous entries for validation set
                        anomalous_df = self.get_validation_data()

                        [(train_data, val_data,  val_data_95)] = TestDataSplitGenerator().generate_splits(df, anomalous_df, n_splits=1, train_size=0.9)
                        train_labels = None

                    

                    X_val_test = val_data.iloc[:, :-1].values.astype(float)  # all columns except last
                    y_val_test = val_data.iloc[:, -1].values.astype(int)     # last column = label (0 or 1)

                    self.logger.info(f"Validation test data shape: {X_val_test.shape}, Labels: {np.bincount(y_val_test)}")


                    if self.args.augment_techniques != ["none"] and self.args.augment_factor is not None:
                        
                        train_data = DataAugmenter().augment_dataset(train_data, self.args.augment_techniques, self.args.augment_factor)
                    
                    if model_type in supervised_models:
                        train_labels = train_data.iloc[:,-1]
                        train_data = train_data.iloc[:,:-1]

                    self.logger.info(f"Training data shape: {train_data.values.shape}, Labels: {np.bincount(train_labels) if train_labels is not None else "None"}")

                    # Preprocess data
                    preprocessor = Preprocessor()
                    X_train_scaled = preprocessor.fit_transform(train_data.values)
                    X_val_test_scaled = preprocessor.transform(X_val_test)

                    if self.encoder_name is not None:
                        dir_prefix = models_dir + f"hybrid/{self.encoder_name}/"
                        encoder_model_dir = dir_prefix + model_config_dir + dir_suffix
                        encoder_path = os.path.join(encoder_model_dir, "encoder.pkl")
                        encoder_trainer = EncoderTrainer(self.encoder_name, param_dict["encoding_dim"], self.stats)
                        encoded_data, encoder = encoder_trainer.run(X_train_scaled, encoder_path, return_encoder=True)
                        X_train_scaled = encoded_data
                        self.logger.info(f"Train shape after encoding: {X_train_scaled.shape}")
                        X_val_test_scaled = encoder.encode(X_val_test_scaled)
                        self.logger.info(f"Validation shape after encoding: {X_val_test_scaled.shape}")

                    model_dir = dir_prefix + model_config_dir + dir_suffix
                    scaler_path = model_dir + "scaler.pkl"
                    preprocessor.save(scaler_path)

                    # Train model with current hyperparameters
                    trainer = BaseModelTrainer(model_type, param_dict_copy, self.stats)
                    model_path = model_dir + "model.pkl"
                    model = trainer.run(X_train_scaled, model_path, y=train_labels, train_indices=self.train_indices, return_model=True)

                    # Get the reconstruction errors for this validation split
                    reconstruction_errors = self.get_reconstruction_errors(model, X_val_test_scaled)
                    # Assuming the label is in 'label' column

                    all_reconstruction_errors.append(reconstruction_errors)
                    all_ground_truths.append(y_val_test)
                
                
                # Optimize threshold across all validation splits
                optimized_threshold, avg_f2 = self.optimize_threshold(all_reconstruction_errors, all_ground_truths)

                f2_raw_scores = []
                # Compute per-split F2 using the global threshold
                for i, (split_errors, split_truths) in enumerate(zip(all_reconstruction_errors, all_ground_truths)):

                    
                    # Apply global threshold
                    split_preds = (split_errors >= optimized_threshold).astype(int)
                    
                    # Compute F2 score
                    f2 = fbeta_score(split_truths, split_preds, beta=2, pos_label=1)
                    
                    f2_raw_scores.append(f2)
                    result = {
                        "model": model_config,
                        "split": i,
                        "f2_score": f2,
                        **param_dict, 
                    }
                    f2_scores.append(result)
                    


                # Track best performance and configuration
                if avg_f2 > best_performance:
                    best_performance = avg_f2
                    best_config = param_dict
                    best_threshold = optimized_threshold

            

            hyperprameter_results = pd.DataFrame(f2_scores)
            if 'hidden_dims' in hyperprameter_results.columns:
                hyperprameter_results['hidden_dims'] = hyperprameter_results['hidden_dims'].astype(str)

            if 'out_channels' in hyperprameter_results.columns:
                hyperprameter_results['out_channels'] = hyperprameter_results['out_channels'].astype(str)

            # Log and display the best hyperparameter configuration for the model
            self.logger.info(f"Best configuration for {model_type}: {best_config} with F2 score: {best_performance} and threshold: {best_threshold}")
            hyperparameter_results_dir = "hyperparameter_tuning_results/" + dir_prefix + model_type + "/" + augmented_dir
            
            os.makedirs(hyperparameter_results_dir, exist_ok=True)
            hyperprameter_results.to_csv(hyperparameter_results_dir + "/results.csv")
            
            x = "model" 
            y = "f2_score"
            boxplot(hyperprameter_results, x, y, dir_path=hyperparameter_results_dir, fifty_fifty=True, title=f"Hyperparameter Comparison ({self.encoder_name + " + " if self.encoder_name is not None else ""}{model_type})", xlabel="Hyper-parameter configuration")
            for k in best_config.keys():
                boxplot(hyperprameter_results, k, y, dir_path=hyperparameter_results_dir, fifty_fifty=True, title=f"Hyperparmeter {k} comparison ({self.encoder_name + " + " if self.encoder_name is not None else ""}{model_type})", xlabel=f"{k}")

            # Now evaluate on the test set with the best configuration
            if best_threshold is not None:
                best_config["threshold"] = best_threshold
            
            self.model_args = best_config
            self.train_single_model(model_type)

    @catch_and_log(Exception, "Obtaining reconstruction errors")
    def get_reconstruction_errors(self, model, X_val_scaled):
        # Reconstruct data using the autoencoder model and return reconstruction errors
        reconstruction_errors, _ = model.predict(X_val_scaled)
        return reconstruction_errors

    @catch_and_log(Exception, "Optimising threshold")
    def optimize_threshold(self, splits_reconstruction_errors, splits_ground_truths):
        """
        Optimize the threshold for anomaly detection.
        This function finds the optimal threshold based on the range of reconstruction errors in the validation set.
        """
        # Combine all reconstruction errors and ground truths across splits
        all_reconstruction_errors_combined = np.concatenate(splits_reconstruction_errors)
        all_ground_truths_combined = np.concatenate(splits_ground_truths)

        # Get the minimum and maximum reconstruction errors
        min_error = np.min(all_reconstruction_errors_combined)
        max_error = np.max(all_reconstruction_errors_combined)

        self.logger.info("Min error: %s, Max error: %s", min_error, max_error)

        # Normalize the threshold range to be between min and max error
        # We will search for thresholds in the normalized range [0, 1] and then map it back to the original range.
        thresholds = np.linspace(0.0, 1.0, 10000)  # 1000 thresholds between 0 and 1

        best_f2 = -float('inf')
        best_threshold = 0.5  # Default threshold
        best_predictions = None

        for threshold in thresholds:
            # Map the threshold back to the original error scale
            adaptive_threshold = min_error + threshold * (max_error - min_error)

            f2_scores = []
            for reconstruction_errors, ground_truth in zip(splits_reconstruction_errors, splits_ground_truths):


                # Apply the adaptive threshold to classify as normal or anomalous
                predictions = (reconstruction_errors >= adaptive_threshold).astype(int)

                # Calculate F2 score for each row (example) and then average it
            
                f2_scores.append(fbeta_score(ground_truth, predictions, beta=2, pos_label=1))

            avg_f2 = np.mean(f2_scores)

            
            # Update best F1 score and threshold
            if avg_f2 > best_f2:
                best_f2 = avg_f2
                best_threshold = adaptive_threshold

        self.logger.info("Best threshold: %s, with f2 score: %s", best_threshold, best_f2)

        return best_threshold, best_f2


    @catch_and_log(Exception, "Training regular model")
    def train_single_model(self, model_type):

        model_name = f"model"

        if self.args.augment_techniques is not None:
            augmented_type = "_".join(self.args.augment_techniques) + f"_{self.args.augment_factor}/"

            augmented_dir = "augmented/" + augmented_type 
        else:
            augmented_dir = ""

        args_dict_copy = self.model_args.copy()

        if self.encoder_name is not None:
            args_dict_copy.pop("encoding_dim")

        for split in range(1, NUM_SPLITS + 1):
            split_dir = os.path.join(TRAINING_TESTS_SPLITS_DIRECTORY, f"split_{split}")
            training_data_location = os.path.join(split_dir, "train.csv")

            df = self.data_loader.load_original_data(training_data_location)

            if model_type in supervised_models:
                anomalous_entries_path = os.path.join(split_dir, "train_anomalies.csv")
                anomalous_entries_df = self.data_loader.load_original_data(anomalous_entries_path)
                
                # Label the data
                df["label"] = 0  # Good data
                anomalous_entries_df["label"] = 1  # Anomalous data

                # Combine and shuffle deterministically
                combined_df = pd.concat([df, anomalous_entries_df], axis=0, ignore_index=True)
                print("Data shape: ", combined_df.shape)
                shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
                train_data = shuffled_df
            else:
                train_data = df
                train_labels = None

            if self.args.augment_techniques != ["none"] and self.args.augment_factor is not None:
                train_data = DataAugmenter().augment_dataset(train_data, self.args.augment_techniques, self.args.augment_factor)
            
            if model_type in supervised_models:
                train_labels = train_data.iloc[:,-1]
                train_data = train_data.iloc[:,:-1]
            

            dir_suffix = augmented_dir + f"split_{split}/"

            dir_prefix = "models/"

            X_train = train_data.values.copy()


            # Preprocess once
            preprocessor = Preprocessor()
            X_scaled = preprocessor.fit_transform(X_train)

            if self.encoder_name is not None:
                encoding_dim = self.model_args["encoding_dim"]
                dir_prefix += f"hybrid/{self.encoder_name}/"
                encoder_model_dir = dir_prefix +  f"{model_type}/" + dir_suffix
                encoder_path = os.path.join(encoder_model_dir, "encoder.pkl")
                encoder = EncoderTrainer(self.encoder_name, encoding_dim, self.stats)
                encoded_data = encoder.run(X_scaled, encoder_path)
                X_scaled = encoded_data
            
        
            supervised_scaler = "supervised/" if model_type in supervised_models else ""
            scaler_path = "scalers/" + supervised_scaler + dir_suffix + "scaler.pkl"
            preprocessor.save(scaler_path)

            # Train selected model
            
            model_dir = dir_prefix + f"{model_type}/"

            if self.model_args:
                # Prepare hyperparameters to save in JSON format

                hyperparameter_dir = model_dir + augmented_dir

                # Create model_dir if it doesn't exist
                os.makedirs(hyperparameter_dir, exist_ok=True)

                # Define the path for the config JSON file
                config_file_path = os.path.join(hyperparameter_dir, "hyperparameters.json")

                # Save the hyperparameters to a JSON file
                with open(config_file_path, 'w') as json_file:
                    json.dump(self.model_args, json_file, indent=4)  # indent for better readability

                # model_dir += f"{self.model_args["lr"]}_{self.model_args["batch_size"]}_{self.model_args["num_epochs"]}_{self.model_args["encoding_dim"]}/"
            model_dir += dir_suffix
            # preprocessor.save(model_dir + f"scaler_{model_name}.pkl")
            trainer = BaseModelTrainer(model_type, args_dict_copy, self.stats)
            model_path = model_dir + f"{model_name}.pkl"
            trainer.run(X_scaled, model_path, y=train_labels, train_indices=self.train_indices)

    @catch_and_log(Exception, "Training ensemble models")
    def train_ensemble_batches(self):

        #DEPRACATED - DO NOT USE#

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
            raise ValueError("Train ensemble is depracated")

            # self.train_ensemble_batches()
            # make_summary("Training Model Summary", self.stats)
            # return 

        
        self.train_model_with_hyperparameter_tuning()
            
    

        make_summary("Training Model Summary", self.stats)
