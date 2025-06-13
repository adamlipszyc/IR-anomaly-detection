import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_and_plot_confusion_matrices(results_dir):
    for root, _, files in os.walk(results_dir):
        if 'evaluation_metrics.xlsx' in files and 'Confusion_Matrix.png' in files:
            xlsx_path = os.path.join(root, 'evaluation_metrics.xlsx')

            try:
                # Read the Confusion Matrix sheet
                df = pd.read_excel(xlsx_path, sheet_name='Confusion Matrix', index_col=0)

                # Ensure we get a numpy array in correct shape
                cm = df.to_numpy()
                
                # Plot confusion matrix with annotations
                plt.figure(figsize=(6, 6))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()

                tick_marks = np.arange(cm.shape[0])
                classes = ['Normal', 'Anomaly']
                plt.xticks(tick_marks, classes)
                plt.yticks(tick_marks, classes)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')

                # Add text annotations
                thresh = cm.max() / 2
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, format(cm[i, j], 'd'),
                                 ha="center", va="center",
                                 color="white" if cm[i, j] > thresh else "black")

                plt.tight_layout()
                save_path = os.path.join(root, 'Confusion_Matrix.png')
                plt.savefig(save_path)
                plt.close()
                print(f"Saved annotated confusion matrix to: {save_path}")
            except Exception as e:
                print(f"Failed to process {xlsx_path}: {e}")

find_and_plot_confusion_matrices("evaluation/results/")
