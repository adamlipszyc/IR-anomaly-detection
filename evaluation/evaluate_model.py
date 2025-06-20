import argparse
import numpy as np
import json
import glob 
import os
import logging
import sys
from rich.logging import RichHandler
from log.utils import make_summary, catch_and_log
from .config import RESULTS_DIR, MODELS_DIR, SCALERS_DIR
from .data_loader import DataLoader
from .metrics_evaluator import MetricsEvaluator
from .model_handler import ModelHandler
from .encoder_handler import EncoderHandler
from training_test_splits.data_split_generation.config import NUM_SPLITS

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

thresholding = {"autoencoder", "anogan", "cnn_anogan", "cnn_supervised_1d", "cnn_supervised_2d"}
supervised_models = {"cnn_supervised_1d", "cnn_supervised_2d", "lstm"}

class AnomalyDetectionEvaluator():
    def __init__(self, args, logger: logging.Logger = None):
        
        self.model_path = MODELS_DIR 
        if args.encoder is not None:
            self.model_path = self.model_path / "hybrid" / f"{args.encoder}"

        self.model_path = self.model_path / args.model_name

        

        self.scaler_path = SCALERS_DIR 
        if args.model_name in supervised_models:
            self.scaler_path = self.scaler_path / "supervised"
        if args.augmented_dir_name:
            self.model_path = self.model_path / "augmented" / args.augmented_dir_name
            self.scaler_path = self.scaler_path / "augmented" / args.augmented_dir_name

        if args.ensemble_voting:
            self.model_path = self.model_path / "ensemble"
        

        # self.model_path = self.model_path / f"split_{args.split}" 
        # if not args.ensemble_voting:
        #     self.model_path = self.model_path / "model.pkl"

        self.encoder_name = args.encoder
        self.ensemble = args.ensemble_voting
        self.model_name = args.model_name
        # self.data_loader = DataLoader(args.split, args.fifty_fifty)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.stats = {}
        # self.split = args.split 
        # self.fifty_fifty = args.fifty_fifty

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
        #DEPRECATED DO NOT USE#
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
    

    def plot_tsne_encoded(self, data, labels, path, anomalous_label=1, predicted_labels=None):
        print("Performing t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(data)

        # Create DataFrame
        tsne_df = pd.DataFrame(data=tsne_results, columns=['tsne_component_1', 'tsne_component_2'])
        tsne_df['true_label'] = labels if isinstance(labels, np.ndarray) else np.array(labels)

        if predicted_labels is not None:
            tsne_df['pred_label'] = predicted_labels if isinstance(predicted_labels, np.ndarray) else np.array(predicted_labels)

            def classify(row):
                if row['true_label'] == anomalous_label:
                    return 'TP' if row['pred_label'] == anomalous_label else 'FN'
                else:
                    return 'FP' if row['pred_label'] == anomalous_label else 'TN'

            tsne_df['classification'] = tsne_df.apply(classify, axis=1)

            palette = {'TP': 'red', 'FN': 'orange', 'FP': 'purple', 'TN': 'blue'}
            legend_title = 'Prediction Outcome'
        else:
            tsne_df['classification'] = tsne_df['true_label'].apply(
                lambda x: 'Anomalous' if x == anomalous_label else 'Normal'
            )
            palette = {'Normal': 'blue', 'Anomalous': 'red'}
            legend_title = 'Label'

        # Plotting
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            x='tsne_component_1',
            y='tsne_component_2',
            hue='classification',
            palette=palette,
            data=tsne_df,
            legend='full',
            alpha=0.7
        )
        plt.title('t-SNE Visualization of Anomalies vs. Normals')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True)
        plt.legend(title=legend_title)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        # plt.show()

    @catch_and_log(Exception, "Evaluating model")
    def evaluate_model(self):
        
        # used_indices = self.get_used_indices()

        # X_test2, y_test2 = self.load_good_data(GOOD_DATA, used_indices=used_indices)

        # # Load the data from CSV (ensure 'anomaly' column is the last column)
        # X_test, y_test = self.load_anomalous_data(ANOMALOUS_DATA, directory=True, num_samples=len(X_test2))

        # # Concatenate the feature matrices and labels
        # X_combined = np.vstack((X_test, X_test2))    # Stack vertically: (n1+n2, cols)
        # y_combined = np.concatenate((y_test, y_test2))  # (n1+n2,)

        for split in range(1, NUM_SPLITS + 1):
            split_dir = f"split_{split}" 
            model_dir = self.model_path / split_dir
            scaler_dir = self.scaler_path / split_dir

            for fifty_fifty in [True, False]:
                data_loader = DataLoader(split, fifty_fifty)

                X_test, y_test = data_loader.load_labeled_test_data()

                # Generate a shuffled index array
                indices = np.random.permutation(len(X_test))

                # Shuffle both X and y using the same indices to preserve alignment
                X_shuffled = X_test[indices]
                y_shuffled = y_test[indices]

                X_test = X_shuffled
                y_test = y_shuffled
                # Preprocess the data (standardize or normalize)
            
                if not self.ensemble:
                    model_path = model_dir / "model.pkl"
                   
                    scaler_path = scaler_dir / "scaler.pkl"
                    
                    model_handler = ModelHandler(model_path, self.model_name, scaler_path)

                    X_test_scaled = model_handler.prepare_data(X_test)

                    if self.encoder_name is not None:
                        self.logger.info("Hybrid model detected")

                        encoder_path = model_dir / "encoder.pkl"

                        encoder_handler = EncoderHandler(encoder_path, self.encoder_name)

                        X_test_scaled = encoder_handler.encode(X_test_scaled)

                        path = self.rebase_dir_path(model_path, MODELS_DIR, RESULTS_DIR, remove_file_name=(not self.ensemble))
                        path = os.path.join(path, "test_50_50" if fifty_fifty else "test_95_5")  
                        path = os.path.join(path, "tsne_encoded_data.png")

                        self.plot_tsne_encoded(X_test_scaled, y_test, path)
                    # Predict anomalies

                    y_pred, y_scores = model_handler.predict(X_test_scaled, threshold=True)

                    self.stats[self.model_path] = "Success"
                else:
                    #Carry out ensemble voting
                    # results = self.ensemble_voting(X_test)
                    self.logger.error("Not implemented")

                output_dir = self.rebase_dir_path(model_path, MODELS_DIR, RESULTS_DIR, remove_file_name=(not self.ensemble))
                output_dir = os.path.join(output_dir, "test_50_50" if fifty_fifty else "test_95_5")       

                self.plot_tsne_encoded(X_test, y_test, os.path.join(output_dir, "tsne_results.png"), predicted_labels=y_pred)

                metrics_evaluator = MetricsEvaluator(y_test, y_pred, y_scores, output_dir)

                # Evaluate the model
                metrics_evaluator.generate_evaluation_metrics()

                #Save evaluation metrics to excel file
                metrics_evaluator.export_evaluation_to_excel()

                # Plot confusion matrix
                metrics_evaluator.plot_confusion_matrix()

                #Plot ROC curve 
                metrics_evaluator.plot_roc_curve()


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
    parser.add_argument('--model_name', required=True, type=str, choices=["one_svm", "LOF", "isolation_forest", "autoencoder", "anogan", "cnn_anogan", "cnn_supervised_2d", "cnn_supervised_1d", "lstm"], help="Name of model algorithm")
    parser.add_argument('--encoder', type=str, choices=["autoencoder", "pca"], help="Which encoder to use for hybrid models")

    parser.add_argument('-e', '--ensemble_voting', action='store_true')
    parser.add_argument('-a', '--augmented_dir_name', type=str, default='', help="Augmented directory name specifying technique and factor")
    args = parser.parse_args()


    evaluator = AnomalyDetectionEvaluator(args)


    try:
        evaluator.evaluate_model()
    except Exception as e:
        log.exception("Unexpected failure")
        sys.exit(99)

    

if __name__ == "__main__":
    main()
    