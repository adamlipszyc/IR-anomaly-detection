import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from log.utils import catch_and_log
from .config import NORMAL_PATH, OUTPUT_DIR, ANOMALY_DIR


class TestDataSplitGenerator:

    def __init__(self, stats: dict):
        self.stats = stats
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @catch_and_log(Exception, "Loading data")
    def load_data(self, normal_path, anomalies_dir):
        # Load original (normal) data
        normal = pd.read_csv(normal_path, header=None)

        # Load all anomaly files in the directory
        anomaly_dfs = []
        for file in Path(anomalies_dir).glob("*.csv"):
            df = pd.read_csv(file, header=None)
            anomaly_dfs.append(df)
        
        anomalies = pd.concat(anomaly_dfs, axis=0).reset_index(drop=True)
        return normal, anomalies
    

    @catch_and_log(Exception, "Generating splits")
    def generate_splits(self, normal_df, anomaly_df, output_dir, n_splits=5, seed=42):
        np.random.seed(seed)
        os.makedirs(output_dir, exist_ok=True)

        for i in range(1, n_splits + 1):

            # Create subdirectory for each split
            split_dir = os.path.join(output_dir , f"split_{i}")
            os.makedirs(split_dir, exist_ok=True)

            # Split normal data into 80% train, 20% test (534 exact)
            normal_train, normal_test = train_test_split(normal_df, train_size=0.8, random_state=seed + i, shuffle=True)
            normal_test_labeled = normal_test.copy()
            normal_test_labeled["label"] = 0
            test_normal_size = len(normal_test)


            # === Create 50/50 test split ===
            n_anomalies_50 = test_normal_size  # 534 normal, 534 anomalies
            anomalies_50 = anomaly_df.sample(n=n_anomalies_50, replace=True, random_state=seed + i)
            anomalies_50_labeled = anomalies_50.copy()
            anomalies_50_labeled["label"] = 1

            test_50_50 = pd.concat([normal_test_labeled, anomalies_50_labeled], axis=0).sample(frac=1, random_state=seed + i)


            # === Create 95/5 test split ===
            n_anomalies_5 = int(test_normal_size * 0.05 / 0.95) 
            anomalies_5 = anomaly_df.sample(n=n_anomalies_5, replace=True, random_state=seed + i)
            anomalies_5_labeled = anomalies_5.copy()
            anomalies_5_labeled["label"] = 1

            test_95_5 = pd.concat([normal_test_labeled, anomalies_5_labeled], axis=0).sample(frac=1, random_state=seed + i)

            # === Save splits ===
            train_path = os.path.join(split_dir, "train.csv")
            test_50_path = os.path.join(split_dir, "test_50_50.csv")
            test_95_path = os.path.join(split_dir, "test_95_5.csv")

            normal_train.to_csv(train_path, header=False, index=False)
            test_50_50.to_csv(test_50_path, header=False, index=False)
            test_95_5.to_csv(test_95_path, header=False, index=False)

            self.logger.info(f"Split {i} saved: train ({len(normal_train)}), test_50_50 ({len(test_50_50)}), test_95_5 ({len(test_95_5)})")

            self.stats[f"Split {i}"] = "Success"
    
    def generate(self):
        normal_df, anomaly_df = self.load_data(NORMAL_PATH, ANOMALY_DIR)
        self.generate_splits(normal_df, anomaly_df, OUTPUT_DIR)


   
