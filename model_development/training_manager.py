import os
import random
import glob
import numpy as np
import pandas as pd
import logging
from .preprocessor import Preprocessor
from .base_model_trainer import BaseModelTrainer
from .data_loader import DataLoader 
from data_augmentation.data_augmenter import DataAugmenter
from .config import AUGMENTED_DATA_DIR, OUTPUT_MODEL_PREFIX, SAMPLES_PER_FILE, NUM_BATCHES, SEED, TRAINING_TESTS_SPLITS_DIRECTORY
from training_test_splits.data_split_generation.config import NUM_SPLITS
from log.utils import catch_and_log, make_summary

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

        self.encoder = self.args.encoder
        self.encoding_dim = self.args.encoding_dim
        self.stats = {}
        self.train_indices = {}
        self.data_loader = DataLoader(AUGMENTED_DATA_DIR, SAMPLES_PER_FILE)
        self.logger = logging.getLogger(self.__class__.__name__)




    @catch_and_log(Exception, "Training regular model")
    def train_single_model(self):

        model_name = f"model"

        if self.args.augment_techniques is not None:
            augmented_type = "_".join(self.args.augment_techniques) + f"_{self.args.augment_factor}/"

            augmented_dir = "augmented/" + augmented_type 
        else:
            augmented_dir = ""


        for split in range(1, NUM_SPLITS + 1):
            split_dir = os.path.join(TRAINING_TESTS_SPLITS_DIRECTORY, f"split_{split}")
            training_data_location = os.path.join(split_dir, "train.csv")

            df = self.data_loader.load_original_data(training_data_location)

            train_data = df

            if self.args.augment_techniques != ["none"] and self.args.augment_factor is not None:
                train_data = DataAugmenter().augment_dataset(train_data, self.args.augment_techniques, self.args.augment_factor)
                
            # # Split train/test
            # indices = np.random.choice(df.index, size=int(0.8 * len(df)), replace=False)
            # train_data = df.loc[indices]
            # self.train_indices[ORIGINAL_DATA_FILE_PATH] = indices.tolist()


            # Preprocess once
            preprocessor = Preprocessor()
            X_scaled = preprocessor.fit_transform(train_data.values.copy())
            
            
            dir_suffix = augmented_dir + f"split_{split}/"
            # Train each selected model
            for model_type in self.models:
                model_dir = f"models/{f"hybrid/{self.encoder}" if self.encoder is not None else ""}{model_type}/{dir_suffix}"
                preprocessor.save(model_dir + f"scaler_{model_name}.pkl")
                trainer = BaseModelTrainer(model_type, self.stats, self.encoder, self.encoding_dim)
                model_path = model_dir + f"{model_name}.pkl"
                encoder_path = model_dir + f"{self.encoder}.pkl"
                trainer.run(X_scaled, model_path, encoder_path, self.train_indices)

    @catch_and_log(Exception, "Training ensemble models")
    def train_ensemble_batches(self):

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
            self.train_ensemble_batches()
            make_summary("Training Model Summary", self.stats)
            return 
    
        
        self.train_single_model()

        make_summary("Training Model Summary", self.stats)
