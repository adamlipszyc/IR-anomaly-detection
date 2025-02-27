from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Assuming `data` is your 2D numpy array
tsne = TSNE(n_components=2)  # Set n_components to 3 for 3D visualization

df = pd.read_csv('training_data/vectorized_data.csv')

data = df.to_numpy()

reduced_data = tsne.fit_transform(data)

# Plotting
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=1)
plt.title("t-SNE Visualization (2D)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.savefig("data_plot.png")
