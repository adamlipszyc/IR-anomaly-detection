import logging 
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    fbeta_score
)
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from log.utils import catch_and_log
from .config import RESULTS_DIR


class MetricsEvaluator:
    def __init__(self, y_test, y_pred, y_scores, output_dir):
       self.logger = logging.getLogger(self.__class__.__name__)
       self.y_test = y_test
       self.y_pred = y_pred 
       self.y_scores = y_scores
       self.output_dir = output_dir
       #TODO add way to store result of best threshold think of metric that really encompasses this i.e 2:1 -> f2:f5


    
    @catch_and_log(Exception, "Generating evaluation metrics")
    def generate_evaluation_metrics(self, return_metrics: bool = False):
        """
        Evaluate the performance of the anomaly detection model and print the classification report.
        """
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print("Accuracy:", accuracy)
        print("Classification Report:")
        print(classification_report(self.y_test, self.y_pred))
        # beta > 1 means recall is weighted more than precision
        fbeta2 = fbeta_score(self.y_test, self.y_pred, beta=2, pos_label=1)
        print("F2 Score (recall-weighted):", fbeta2)
        fbeta5 = fbeta_score(self.y_test, self.y_pred, beta=5, pos_label=1)
        print("F5 Score (heavily recall-weighted):", fbeta5)
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, self.y_pred))
        
        # If available, compute ROC AUC
        roc = None
        if len(np.unique(self.y_test)) == 2:
            roc = roc_auc_score(self.y_test, self.y_scores)
            print("ROC AUC:", roc)

        if return_metrics:
            return pd.DataFrame([{
                "accuracy": accuracy,
                "f2_score": fbeta2,
                "f5_score": fbeta5,
                "roc_auc": roc
            }])

        return None

    @catch_and_log(Exception, "Exporting evaluation to excel")
    def export_evaluation_to_excel(self):
        

        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, f"evaluation_metrics.xlsx")

        # Metrics summary
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, zero_division=0)
        recall = recall_score(self.y_test, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_test, self.y_pred, zero_division=0)
        fbeta2 = fbeta_score(self.y_test, self.y_pred, beta=2, pos_label=1)
        fbeta5 = fbeta_score(self.y_test, self.y_pred, beta=5, pos_label=1)
        auc = roc_auc_score(self.y_test, self.y_scores) if len(np.unique(self.y_test)) == 2 else None
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred).ravel()

        summary_df = pd.DataFrame([{
            "Model": os.path.relpath(self.output_dir, start=RESULTS_DIR),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "F2 Score (Recall-Weighted)": fbeta2,
            "F5 Score (Heavily Recall-Weighted)": fbeta5,
            "AUC": auc,
            "True Positives": tp,
            "False Positives": fp,
            "True Negatives": tn,
            "False Negatives": fn,
        }])

        # Classification report
        report_df = pd.DataFrame(classification_report(self.y_test, self.y_pred, output_dict=True, zero_division=0)).transpose()

        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])

        # Write all to Excel
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            report_df.to_excel(writer, sheet_name="Classification Report")
            cm_df.to_excel(writer, sheet_name="Confusion Matrix")

        self.logger.info("[✓] Evaluation report written to: %s", file_path)

    @catch_and_log(Exception, "Plotting confusion matrix")
    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix for visual interpretation.
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
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
        plt.savefig(os.path.join(self.output_dir, "Confusion_Matrix.png"))
        # plt.show()

    @catch_and_log(Exception, "Plotting ROC curve")
    def plot_roc_curve(self):
        """
        Plots the ROC curve for a binary classification model.
        
        Parameters:
        - self.y_test: The true labels (0 for normal, 1 for anomaly)
        - self.y_pred: The predicted probabilities for the positive class (anomalies)
        """
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_scores)
        
        # Compute AUC
        auc = roc_auc_score(self.y_test, self.y_scores)
        
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
        plt.savefig(os.path.join(self.output_dir, "ROC.png"))
        # plt.show()