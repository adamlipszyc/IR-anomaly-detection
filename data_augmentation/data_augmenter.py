import os
import random
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple
from log.utils import catch_and_log



class DataAugmenter:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def augment_magnitude(self, vector: pd.Series) -> List[pd.Series]:
        """Randomly scales nonzero values"""

        augmented_results = []
        min_scale, max_scale = 0.1, 5

        # Generates 10 new entries by randomly sampling a scale in range 0.1-5 and
        # multiplying our original vector by this
        for _ in range(10):
            scaling_factor = np.random.uniform(min_scale, max_scale)
            # Randomly rounds the vector to a whole number approximately 50% of the time 
            isRounded = np.random.uniform(0, 1) > 0.5
            result = vector * scaling_factor
            if isRounded:
                result = result.round(0)
            else:
                # Randomly selects the number of decimal places to round to 
                decimals = np.floor(np.random.uniform(1, 8)).astype(int)
                result = result.round(decimals)
            augmented_results.append(result)

        return augmented_results

    def augment_shift(self, vector: pd.Series, max_results: int = 99999999) -> List[np.ndarray]:
        """ 
        Shifts vector according to first non-zero value and last-non zero 
        ensuring time frame between positions is preserved 
        but the actual day the positions fall on can change 
        """
        # find first and last non-zero value 
        first_nonzero_index = (vector != 0).idxmax()
        last_nonzero_index = vector.where(vector != 0).last_valid_index()
        
        augmented_results = []
        possible_left_shifts = first_nonzero_index
        if possible_left_shifts:       
            for val in range(2, min(possible_left_shifts, 2* max_results) + 1, 2):
                augmented_results.append(np.roll(vector, -val))

        possible_right_shifts = len(vector) - last_nonzero_index - 1
        if possible_right_shifts:
            for val in range(2, min(possible_right_shifts, 2 * max_results) + 1, 2):
                augmented_results.append(np.roll(vector, val))
        
        return augmented_results

    @catch_and_log(Exception, "Adding noise (fast)")
    def augment_noise_fast(self, vector: pd.Series, noise_level: float = 0.1, num_augmentations: int = 1) -> List[pd.Series]:
        """Fast version of adding small noise to zero value pairs."""
        print("adding noise (fast)")
        vector: np.ndarray = np.array(vector)

        if len(vector) % 2 != 0:
            raise ValueError("The series length must be even.")

        # Reshape to (num_pairs, 2)
        pairs = vector.reshape(-1, 2)
        augmented_rows = []

        for _ in range(num_augmentations):
            modified = pairs.copy()

            # Find all-zero pairs
            zero_mask = np.all(modified == 0.0, axis=1)
            zero_indices = np.where(zero_mask)[0]

            if len(zero_indices) > 0:
                num_to_modify = random.randint(1, len(zero_indices))
                selected = np.random.choice(zero_indices, size=num_to_modify, replace=False)

                # Random noise per selected pair
                noise_values = np.array([
                    round(np.random.uniform(-noise_level, noise_level), 
                          int(np.floor(np.random.uniform(0, 8))))
                    for _ in selected
                ])

                modified[selected, 0] += noise_values
                modified[selected, 1] += noise_values

            augmented_rows.append(pd.Series(modified.flatten()))

        return augmented_rows

    def augment_noise_vectorized(self, vector: pd.Series, noise_level: float = 0.1, num_augmentations: int = 1) -> List[pd.Series]:
        """Extremely fast, fully vectorized noise augmentation."""
        
        vector = np.asarray(vector)
        if len(vector) % 2 != 0:
            raise ValueError("Vector length must be even.")
        
        n_pairs = len(vector) // 2
        pairs = vector.reshape(n_pairs, 2)

        # Find all-zero pairs
        zero_mask = np.all(pairs == 0.0, axis=1)
        zero_indices = np.where(zero_mask)[0]

        # Early return if no zero pairs
        if len(zero_indices) == 0:
            return [pd.Series(vector.copy()) for _ in range(num_augmentations)]

        # Preallocate output: shape (num_augmentations, n_pairs, 2)
        all_augmented = np.tile(pairs, (num_augmentations, 1)).reshape(num_augmentations, n_pairs, 2)

        # For each augmentation, randomly pick subset of zero_pairs and add noise
        for i in range(num_augmentations):
            # Choose subset of zero indices
            num_to_modify = np.random.randint(1, len(zero_indices) + 1)
            selected = np.random.choice(zero_indices, size=num_to_modify, replace=False)

            # Generate vectorized noise
            decimal_places = np.floor(np.random.uniform(0, 8, size=num_to_modify)).astype(int)
            raw_noise = np.random.uniform(-noise_level, noise_level, size=num_to_modify)
            noise = np.array([np.round(val, dec) for val, dec in zip(raw_noise, decimal_places)])

            # Add noise to both elements of each selected pair
            all_augmented[i, selected, :] += noise[:, np.newaxis]

        # Flatten each augmented array and convert to pd.Series
        return [pd.Series(all_augmented[i].flatten()) for i in range(num_augmentations)]

    @catch_and_log(Exception, "Adding noise")
    def augment_noise(self, vector: pd.Series, noise_level: float = 0.1, num_augmentations: int = 1) -> List[pd.Series]:
        """Adds small noise to zero values. NEEDS TO BE SPED UP""" 
        print("adding noise")
        augmented_rows = []
        
        # Ensure series has an even number of elements
        if len(vector) % 2 != 0:
            raise ValueError("The series length must be even.")
        
        # Get pairs of values
        pairs = [vector[i:i+2].reset_index(drop=True) for i in range(0, len(vector), 2)]
        
        for _ in range(num_augmentations):
            # Make a copy of the pairs to modify
            modified_pairs = [pair.copy() for pair in pairs]
            
            # Randomly select how many pairs to modify
            num_pairs_to_modify = random.randint(1, len(pairs))
            selected_pairs = random.sample(range(len(pairs)), num_pairs_to_modify)
            
            # Add noise to selected pairs where both values are zero
            for index in selected_pairs:
                pair = modified_pairs[index]
                if pair[0] == 0.0 and pair[1] == 0.0:
                    noise = np.round(np.random.uniform(-noise_level, noise_level), np.floor(np.random.uniform(0, 8)).astype(int))
                    pair[0] += noise
                    pair[1] += noise
            
            # Flatten the modified pairs and add to the augmented rows
            augmented_rows.append(pd.concat(modified_pairs, ignore_index=True))
        
        # Return the augmented data
        return augmented_rows

    @catch_and_log(Exception, "Processing a single row")
    def process_row(self, row_id: int, row_data: pd.Series) -> None:
        """Apply augmentation to a single row and write to a temp CSV."""
        
        
        augmented_data = pd.concat([pd.DataFrame([row_data]), pd.DataFrame(self.augment_magnitude(row_data))], ignore_index=True)
        
        shift_results = []

        for index, row in augmented_data.iterrows():
            shift_results.append(pd.DataFrame(self.augment_shift(row)))
        
        augmented_data = pd.concat([augmented_data, pd.concat(shift_results, ignore_index=True)], ignore_index=True)

        # print("adding noise")                             
        # noise_results = []
        # for index, row in augmented_data.iterrows():
        #     noise_results.append(pd.DataFrame(self.augment_noise_vectorized(row)))
        
        # augmented_data = pd.concat([augmented_data, pd.concat(noise_results, ignore_index=True)], ignore_index=True)
        
        return augmented_data
    

    @catch_and_log(Exception, "Augmenting dataset")
    def augment_dataset(
        self,
        data: pd.DataFrame,
        techniques: List[str],
        factor: int
    ) -> pd.DataFrame:
        """
        Augments the dataset using the specified techniques until target_size is reached.
        """
        augmented = [data]
        target_size = len(data) * factor
        num_needed = target_size - len(data)
        per_sample = num_needed // len(data)

        has_label = "label" in data.columns
        feature_cols = data.columns.drop("label") if has_label else data.columns
        
        for index, row in data.iterrows():
            augmented_samples = []

            row_features = row[feature_cols]

            augmented_samples.append(row_features.copy())

            if "magnitude" in techniques:
                augmented_samples += self.augment_magnitude(row_features)

            if "shift" in techniques:
                augmented_samples += [pd.Series(x) for x in self.augment_shift(row_features, max_results=per_sample)]

            if "noise" in techniques:
                augmented_samples += self.augment_noise_vectorized(row_features)

            # Only keep what we need
            sampled = random.sample(augmented_samples, min(per_sample, len(augmented_samples)))

            # Convert to DataFrame and reattach label if present
            if has_label:
                label_value = row["label"]
                df_sampled = pd.DataFrame(sampled)
                df_sampled["label"] = label_value
                augmented.append(df_sampled)
            else:
                augmented.append(pd.DataFrame(sampled))


        augmented_df = pd.concat(augmented, ignore_index=True).iloc[:target_size]

        # Ensure label is the last column
        if "label" in augmented_df.columns:
            label_col = augmented_df.pop("label")
            augmented_df["label"] = label_col  
        
        return augmented_df



        

