from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# # Assuming `data` is your 2D numpy array
# tsne = TSNE(n_components=2)  # Set n_components to 3 for 3D visualization

# for i in range(1, 6):
        

#     df = pd.read_csv('training_test_splits/vectorized_data.csv')

#     data = df.to_numpy()

#     reduced_data = tsne.fit_transform(data)

#     # Plotting
#     plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=1)
#     plt.title("t-SNE Visualization (2D)")
#     plt.xlabel("t-SNE 1")
#     plt.ylabel("t-SNE 2")
#     plt.savefig("data_plot.png")


import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_tsne_anomalies(base_data_path, anomalous_label=1):
    """
    Reads CSV test data from multiple folders, performs t-SNE, and plots
    anomalous vs. good data with different colors.

    Args:
        base_data_path (str): The path to the directory containing the 'split X' folders.
        label_column_name (str): The name of the column containing the labels (default: 'label').
                                 Assumes this is the last column if not specified.
        anomalous_label: The value in the label column that denotes an anomaly.
                         (e.g., 1, 'anomaly', True).
    """

    print(f"Reading data from: {base_data_path}")

    # Read data from each split folder
    for i in range(1, 6):
        folder_name = f"split_{i}"
        folder_path = os.path.join(base_data_path, folder_name)
      
        # Assuming CSV files are directly in the split folders. 
        # If there are multiple CSVs per folder, you might need another loop.
        for file_name in ["test_50_50.csv", "test_95_5.csv"]:
           
            file_path = os.path.join(folder_path, file_name)
            print(f"  Reading {file_path}")
            try:
                df = pd.read_csv(file_path)

                features = df.iloc[:, :-1]
                labels = df.iloc[:, -1]
                
                print(f"Features shape: {features.shape}")
                print(f"Labels shape: {labels.shape}")
                
                # Perform t-SNE
                print("Performing t-SNE dimensionality reduction...")
                tsne = TSNE(n_components=2) # You can tune perplexity and n_iter
                tsne_results = tsne.fit_transform(features)

                # Create a DataFrame for plotting
                tsne_df = pd.DataFrame(data=tsne_results, columns=['tsne_component_1', 'tsne_component_2'])
                tsne_df['label'] = labels.values # Add labels back to the t-SNE results DataFrame

                # Map labels for plotting
                tsne_df['anomaly_status'] = tsne_df['label'].apply(
                    lambda x: 'Anomalous' if x == anomalous_label else 'Normal'
                )

                # Plotting
                plt.figure(figsize=(12, 10))
                sns.scatterplot(
                    x='tsne_component_1',
                    y='tsne_component_2',
                    hue='anomaly_status',
                    palette={'Normal': 'blue', 'Anomalous': 'red'}, # Customize colors as needed
                    data=tsne_df,
                    legend='full',
                    alpha=0.7
                )
                plt.title('t-SNE Visualization of Normal vs. Anomalous Data')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.grid(True)
                path = f"training_test_splits/visualization/split_{i}/{file_name[:-4]}"
                os.makedirs(path, exist_ok=True)
                plt.savefig(path)
                # plt.show()


            except Exception as e:
                print(f"Error reading {file_path}: {e}")


if __name__ == "__main__":
    
    base_path = 'training_test_splits/'
   
    plot_tsne_anomalies(
        base_data_path=base_path,
        anomalous_label=1
    )