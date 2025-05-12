import argparse
import pickle
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import glob 
import os
import logging
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from scipy.stats import mode
from rich.logging import RichHandler
from log.utils import make_summary, catch_and_log
from .config import BASE_OUTPUT_FILE_PATH, GOOD_DATA, ANOMALOUS_DATA, CLASS_SIZE



class AnomalyDetectionEvaluator():
    def __init__(self, model_path: str, ensemble: bool = False, logger: logging.Logger = None):
        self.model_path = model_path
        self.ensemble = ensemble
        model_name = os.path.basename(os.path.dirname(self.model_path))
        name_suffix = "_ensemble" if self.ensemble else ""
        model_name += name_suffix
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.X_test = None
        self.y_test = None
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.stats = {}

    @catch_and_log(Exception, "Loading anomalous data")
    def load_anomalous_data(self, file_paths, directory=False, num_samples=None):
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

            np.random.shuffle(X)
            
            if anomalous_data is not None:
                anomalous_data = np.vstack((anomalous_data, X))
            else:
                anomalous_data = X
        
        # If num_samples is set and smaller than total, randomly select subset
        if num_samples is not None and num_samples < len(anomalous_data):
            indices = np.random.choice(len(anomalous_data), size=num_samples, replace=False)
            anomalous_data = anomalous_data[indices]


        Y = np.ones(len(anomalous_data)) # Label 1 for each row
        return anomalous_data, Y

    @catch_and_log(Exception, "Loading good data")
    def load_good_data(self, file_paths, directory=False, used_indices={}):
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

            if file_path in used_indices:
                data = data[~data.index.isin(used_indices[file_path])]
            
            X = data.values  # All columns

            np.random.shuffle(X)

            if good_data is not None:
                good_data = np.vstack((good_data, X))
            else:
                good_data = X
        
        if self.ensemble:
            indices = np.random.choice(len(good_data), size=CLASS_SIZE, replace=False)
            good_data = good_data[indices]

        Y = np.zeros(len(good_data))         # Label 0 for each row
        return good_data, Y


    @catch_and_log(Exception, "Carrying out prediction")
    def predict(self, trained_model, X_test):
        """
        Take the already trained anomaly detection model and predict anomalies on the test set.
        Output: 1 for anomaly, 0 for normal.
        """
        
        # Predict anomalies in the test data
        y_pred = trained_model.predict(X_test)
        y_pred = np.where(y_pred == 1, 0.0, 1.0)  # OneClassSVM returns 1 for normal and -1 for anomaly. We need to map to 0 and 1.
        
        return y_pred

    @catch_and_log(Exception, "Generating evaluation metrics")
    def generate_evaluation_metrics(self, y_test, y_pred):
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

    @catch_and_log(Exception, "Exporting evaluation to excel")
    def export_evaluation_to_excel(self, y_test, y_pred, output_dir=BASE_OUTPUT_FILE_PATH):
        

        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"evaluation_metrics.xlsx")

        # Metrics summary
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) == 2 else None
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        summary_df = pd.DataFrame([{
            "Model": self.model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": auc,
            "True Positives": tp,
            "False Positives": fp,
            "True Negatives": tn,
            "False Negatives": fn,
        }])

        # Classification report
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True, zero_division=0)).transpose()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

        # Write all to Excel
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            report_df.to_excel(writer, sheet_name="Classification Report")
            cm_df.to_excel(writer, sheet_name="Confusion Matrix")

        self.logger.info("[✓] Evaluation report written to: %s", file_path)

    @catch_and_log(Exception, "Plotting confusion matrix")
    def plot_confusion_matrix(self, y_test, y_pred, output_dir=BASE_OUTPUT_FILE_PATH):
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
        plt.savefig(os.path.join(output_dir, "Confusion_Matrix.png"))
        plt.show()

    @catch_and_log(Exception, "Plotting ROC curve")
    def plot_roc_curve(self, true_labels, predictions, output_dir=BASE_OUTPUT_FILE_PATH):
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
        plt.savefig(os.path.join(output_dir, "ROC.png"))
        plt.show()

    # Function to load a trained model using pickle
    @catch_and_log(Exception, "Loading trained model")
    def load_trained_model(self, model_file_path):
        with open(model_file_path, 'rb') as file:
            model = pickle.load(file)
        return model

    @catch_and_log(Exception, "Loading scaler")
    def load_scaler(self, scaler_file_path):
        with open(scaler_file_path, 'rb') as file:
            scaler = pickle.load(file)
        
        return scaler

    @catch_and_log(Exception, "Getting scaler path")
    def get_scaler_path(self, file_path="") -> str:
        """
        Given a model path like 'models/one_svm/batch_model_....pkl',
        return the corresponding scaler path with 'scaler_' prefixed to the filename.
        """
        path = file_path if file_path else self.model_path
        dir_path, filename = os.path.split(path)
        scaler_filename = f"scaler_{filename}"
        scaler_path = os.path.join(dir_path, scaler_filename)
        self.logger.info("Scaler path found: %s", scaler_path)
        return scaler_path

    @catch_and_log(Exception, "Carrying out ensemble voting")
    def ensemble_voting(self, X_test):
        file_paths = glob.glob(os.path.join(self.model_path, "batch_model_*.pkl"))
        if len(file_paths) == 0:
            raise RuntimeError("No CSV files found in directory.")

        
        y_pred = []
        for file_path in file_paths:
            trained_model = self.load_trained_model(file_path)
            scaler = self.load_scaler(self.get_scaler_path(file_path)) 

            flattened_x_test = X_test.flatten()
            reshaped = flattened_x_test.reshape(-1, 1)
            X_test_scaled = scaler.transform(reshaped)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            # Train the model and predict anomalies
            y_pred.append(self.predict(trained_model, X_test_scaled))

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
    def evaluate_model(self):
        
        used_indices = self.get_used_indices()

        X_test2, y_test2 = self.load_good_data(GOOD_DATA, used_indices=used_indices)

        # Load the data from CSV (ensure 'anomaly' column is the last column)
        X_test, y_test = self.load_anomalous_data(ANOMALOUS_DATA, directory=True, num_samples=len(X_test2))

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
       
        if not self.ensemble:
            #Load the trained model
            trained_model = self.load_trained_model(self.model_path)

            #Load the corresponding scaler
            scaler = self.load_scaler(self.get_scaler_path())

            #Reshape the test data to prepare for scaling
            flattened_x_test = X_test.flatten()
            reshaped = flattened_x_test.reshape(-1, 1)

            X_test_scaled = scaler.transform(reshaped)
            X_test_scaled = X_test_scaled.reshape(X_test.shape)

            # Train the model and predict anomalies
            y_pred = self.predict(trained_model, X_test_scaled)

            self.stats[self.model_path] = "Success"
        else:
            #Carry out ensemble voting
            y_pred = self.ensemble_voting(X_test)

        
        # Evaluate the model
        self.generate_evaluation_metrics(y_test, y_pred)

        #Get results directory 
        output_dir = os.path.join(BASE_OUTPUT_FILE_PATH, f"{self.model_name}")

        #Save evaluation metrics to excel file
        self.export_evaluation_to_excel(y_test, y_pred, output_dir)

        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, output_dir)

        #Plot ROC curve 
        self.plot_roc_curve(y_test, y_pred, output_dir)

        

        make_summary("Model Evaluation Summary", self.stats)


        


def main() -> None:
    # Logging config with Rich
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )

    parser = argparse.ArgumentParser(description="Evaluate an anomaly detection model")
    parser.add_argument('--model_path', required=True, type=str, help="Path to the trained model file")
    # parser.add_argument('--scaler_path', required=True, type=str, help="Path to the scaler used in training the model")
    parser.add_argument('-e', '--ensemble_voting', action='store_true')
    args = parser.parse_args()

    evaluator = AnomalyDetectionEvaluator(args.model_path, ensemble=args.ensemble_voting)

    evaluator.evaluate_model()

    

if __name__ == "__main__":
    main()
   