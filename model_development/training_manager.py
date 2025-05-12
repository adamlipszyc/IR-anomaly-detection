import os
import random
import glob
import numpy as np
import pandas as pd
import logging
from .preprocessor import Preprocessor
from .base_model_trainer import BaseModelTrainer
from .data_loader import DataLoader 
from .config import ORIGINAL_DATA_FILE_PATH, AUGMENTED_DATA_DIR, OUTPUT_MODEL_PREFIX, SAMPLES_PER_FILE, NUM_BATCHES, SEED
from log.utils import catch_and_log, make_summary

class TrainingManager:
    def __init__(self, args):
        self.args = args
        self.stats = {}
        self.train_indices = {}
        self.data_loader = DataLoader(AUGMENTED_DATA_DIR, SAMPLES_PER_FILE)
        self.logger = logging.getLogger(self.__class__.__name__)


    def selected_models(self):
        models = []
        if self.args.one_svm:
            models.append("one_svm")
        if self.args.isolation_forest:
            models.append("isolation_forest")
        if self.args.local_outlier:
            models.append("LOF")
        return models

    @catch_and_log(Exception, "Training regular model")
    def train_single_model(self):
        df = self.data_loader.load_original_data(ORIGINAL_DATA_FILE_PATH)

        # Split train/test
        indices = np.random.choice(df.index, size=int(0.8 * len(df)), replace=False)
        train_data = df.loc[indices]
        self.train_indices[ORIGINAL_DATA_FILE_PATH] = indices.tolist()

        # Preprocess once
        preprocessor = Preprocessor()
        X_scaled = preprocessor.fit_transform(train_data.values)
        

        model_path = "model"
        # Train each selected model
        for model_type in self.selected_models():
            preprocessor.save(f"models/{model_type}/scaler_{model_path}.pkl")
            trainer = BaseModelTrainer(model_type, self.stats)
            trainer.run(X_scaled, model_path, self.train_indices)

    @catch_and_log(Exception, "Training ensemble models")
    def train_ensemble_batches(self):
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
            indexes = [
                int(str(os.path.basename(file_path)).removeprefix("row_").removesuffix(".csv"))
                for file_path in first_row_set
            ]
            self.train_indices[ORIGINAL_DATA_FILE_PATH] = indexes

            # Preprocess once per batch
            preprocessor = Preprocessor()
            X_scaled = preprocessor.fit_transform(df.values)

            model_path = f"{OUTPUT_MODEL_PREFIX}_{batch_idx:03d}"
            for model_type in self.selected_models():
                preprocessor.save(f"models/{model_type}/scaler_{model_path}.pkl")
                trainer = BaseModelTrainer(model_type, self.stats)
                trainer.run(X_scaled, model_path, self.train_indices)


    def run(self):
        if self.args.train_augmented:
            self.train_ensemble_batches()
        else:
            self.train_single_model()
        make_summary("Training Model Summary", self.stats)
