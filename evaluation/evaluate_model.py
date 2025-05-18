import argparse
import numpy as np
import json
import glob 
import os
import logging
import sys
from rich.logging import RichHandler
from log.utils import make_summary, catch_and_log
from .config import OUTPUT_DIR, MODELS_DIR
from .data_loader import DataLoader
from .metrics_evaluator import MetricsEvaluator
from .model_handler import ModelHandler


class AnomalyDetectionEvaluator():
    def __init__(self, args, logger: logging.Logger = None):
        

        self.model_path = MODELS_DIR / args.model_name
        if args.augmented_dir_name:
            self.model_path = self.model_path / "augmented" / args.augmented_dir_name
        if args.ensemble_voting:
            self.model_path = self.model_path / "ensemble"

        self.model_path = self.model_path / f"split_{args.split}" 
        if not args.ensemble_voting:
            self.model_path = self.model_path / "model.pkl"

        self.ensemble = args.ensemble_voting
        self.model_name = args.model_name
        self.data_loader = DataLoader(args.split, args.fifty_fifty)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.stats = {}
        self.split = args.split 
        self.fifty_fifty = args.fifty_fifty

    @catch_and_log(Exception, "Rebasing directory")
    def rebase_dir_path(self, original_path, old_root, new_root, remove_file_name: bool = False):
        """
        Replace the base directory of a path and remove the filename.

        Args:
            original_path (str): Full original path including filename (e.g., 'models/one_svm/split_1/model.pkl').
            old_root (str): Base directory to remove (e.g., 'models').
            new_root (str): Directory to prepend (e.g., 'outputs').

        Returns:
            str: Updated directory path without the filename.
        """
        # Normalize paths to handle slashes consistently
        original_path = os.path.normpath(original_path)
        old_root = os.path.normpath(old_root)

        # Remove the file name
        dir_path = original_path
        if remove_file_name:
            dir_path = os.path.dirname(original_path)

        # Get relative path from old root
        rel_path = os.path.relpath(dir_path, start=old_root)

        # Construct new path
        new_path = os.path.join(new_root, rel_path)

        return new_path

    @catch_and_log(Exception, "Carrying out ensemble voting")
    def ensemble_voting(self, X_test):
        file_paths = glob.glob(os.path.join(self.model_path, "batch_model_*.pkl"))
        if len(file_paths) == 0:
            raise RuntimeError("No CSV files found in directory.")

        
        y_pred = []
        for file_path in file_paths:

            model_handler = ModelHandler(file_path)

            X_test_scaled = model_handler.prepare_data(X_test)

            # Train the model and predict anomalies
            y_pred.append(model_handler.predict(X_test_scaled))

            self.stats[file_path] = "Success"
        
        # Convert to numpy array for easier manipulation
        y_pred_array = np.array(y_pred)  # Shape: (num_models, num_samples)

        # Majority vote across models
        ensemble_prediction = np.max(y_pred_array, axis=0)
        # ensemble_prediction, _ = mode(y_pred_array, axis=0)

        # Flatten the result
        ensemble_prediction = ensemble_prediction.flatten()

        return ensemble_prediction
    
    @catch_and_log(Exception, "Retrieving used indices")
    def get_used_indices(self):
        used_indices = {}
        if not self.ensemble:
            used_indices_file_path = f"{self.model_path[:-4]}_indices.json"
            with open(used_indices_file_path, "r") as file:
                used_indices = json.load(file)
        else:
            file_paths = glob.glob(os.path.join(self.model_path, "batch_model_*.json"))
            for used_indices_file_path in file_paths:
                with open(used_indices_file_path, "r") as file:
                    data: dict[str, list[int]] = json.load(file)
                    for path, indices in data.items():
                        if path in used_indices:
                            used_indices[path] = list(set(used_indices[path]) | set(indices))
                        else:
                            used_indices[path] = indices
        
        return used_indices

    @catch_and_log(Exception, "Evaluating model")
    def evaluate_model(self, return_metrics: bool = False):
        
        # used_indices = self.get_used_indices()

        # X_test2, y_test2 = self.load_good_data(GOOD_DATA, used_indices=used_indices)

        # # Load the data from CSV (ensure 'anomaly' column is the last column)
        # X_test, y_test = self.load_anomalous_data(ANOMALOUS_DATA, directory=True, num_samples=len(X_test2))

        # # Concatenate the feature matrices and labels
        # X_combined = np.vstack((X_test, X_test2))    # Stack vertically: (n1+n2, cols)
        # y_combined = np.concatenate((y_test, y_test2))  # (n1+n2,)

        
        X_test, y_test = self.data_loader.load_labeled_test_data()

       
        # Generate a shuffled index array
        indices = np.random.permutation(len(X_test))

        # Shuffle both X and y using the same indices to preserve alignment
        X_shuffled = X_test[indices]
        y_shuffled = y_test[indices]

        X_test = X_shuffled
        y_test = y_shuffled
        # Preprocess the data (standardize or normalize)
       
        if not self.ensemble:
            
            model_handler = ModelHandler(self.model_path)

            X_test_scaled = model_handler.prepare_data(X_test)

            # Predict anomalies
            y_pred = model_handler.predict(X_test_scaled)

            self.stats[self.model_path] = "Success"
        else:
            #Carry out ensemble voting
            y_pred = self.ensemble_voting(X_test)

        output_dir = self.rebase_dir_path(self.model_path, MODELS_DIR, OUTPUT_DIR, remove_file_name=(not self.ensemble))
        output_dir = os.path.join(output_dir, "test_50_50" if self.fifty_fifty else "test_95_5")


        metrics_evaluator = MetricsEvaluator(y_test, y_pred, output_dir)

        # Evaluate the model
        metrics = metrics_evaluator.generate_evaluation_metrics(return_metrics)

        #Save evaluation metrics to excel file
        metrics_evaluator.export_evaluation_to_excel()

        # Plot confusion matrix
        metrics_evaluator.plot_confusion_matrix()

        #Plot ROC curve 
        metrics_evaluator.plot_roc_curve()

        if return_metrics:
            return metrics

        make_summary("Model Evaluation Summary", self.stats)

        return None


        


def main() -> None:
    # Logging config with Rich
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

    log = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Evaluate an anomaly detection model")
    parser.add_argument('--model_name', required=True, type=str, choices=["one_svm", "LOF", "isolation_forest"], help="Name of model algorithm")
    # parser.add_argument('--scaler_path', required=True, type=str, help="Path to the scaler used in training the model")
    parser.add_argument('--split', required=True, type=int, choices=range(1,6), help="Which test data split")
    parser.add_argument('-50/50', "--fifty_fifty", action='store_true')
    parser.add_argument('-e', '--ensemble_voting', action='store_true')
    parser.add_argument('-a', '--augmented_dir_name', type=str, default='', help="Augmented directory name specifying technique and factor")
    args = parser.parse_args()

    # if args.ensemble_voting and not args.augment_data:
    #     parser.error("Must use augmented data for ensemble voting")

    evaluator = AnomalyDetectionEvaluator(args)


    try:
        evaluator.evaluate_model()
    except Exception as e:
        log.exception("Unexpected failure")
        sys.exit(99)

    

if __name__ == "__main__":
    main()
    