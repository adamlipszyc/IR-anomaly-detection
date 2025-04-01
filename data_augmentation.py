import numpy as np
import pandas as pd
import random

def augment_magnitude(vector):
    """Randomly scales nonzero values while keeping them above 5."""
    augmented_results = []
    min_scale, max_scale = 0.1, 5

    #Generates 1000 new entries by randomly sampling a scale in range 0.1-5 and
    #mulitplying our original vector by this
    for i in range(1000):
        scaling_factors = np.random.uniform(min_scale, max_scale, size=vector.shape)
        augmented_results.append(vector * scaling_factors)

    return augmented_results

def augment_shift(vector):
    """ 
        Shifts vector according to first non-zero value and last-non zero 
        ensuring time frame between positions is preserved 
        but the actual day the positions fall on can change 
    """
    # find first and last non-zero value 
    first_nonzero_index = (vector != 0).idxmax()
    last_nonzero_index = vector.where(vector != 0).last_valid_index()
    
    augmented_results = []
    if first_nonzero_index:
        possible_left_shifts = first_nonzero_index
        possible_right_shifts = len(vector) - last_nonzero_index - 1
        for val in range(1, possible_left_shifts):
            augmented_results.append(np.roll(vector, -val))

        for val in range(1, possible_right_shifts):
            augmented_results.append(np.roll(vector, val))
    
    return augmented_results

def augment_noise(vector, noise_level=0.1, num_augmentations=100):
    """Adds small noise to zero values.""" 
    
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
                noise = np.random.uniform(-noise_level, noise_level)
                pair[0] += noise
                pair[1] += noise
        
        # Flatten the modified pairs and add to the augmented rows
        augmented_rows.append(pd.concat(modified_pairs, ignore_index=True))
    
    # Return the augmented data
    return augmented_rows

def augment_data():
    """Generates augmented data by applying transformations."""

    file_path = "training_data/vectorized_data.csv"
    df = pd.read_csv(file_path, header=None) #load data from csv file 

    augmented_data = df
    jobs = [1, 2, 3]
    func = None
    for i in jobs:
        if i == 1:
            func = augment_magnitude
        elif i == 2:
            func = augment_shift 
        else:
            func = augment_noise

        new_entries = []
        for index, row in augmented_data.iterrows():
            if i == 3:
                new_entries.append(func(row))
            else:
                new_entries += func(row)
        
        augmented_data = pd.concat([augmented_data, pd.DataFrame(new_entries)], ignore_index=True)

    output_file_path = "training_data/augmented_data.csv"
    augmented_data.to_csv(output_file_path, index=False, header=False)    
    


if __name__ == "__main__":
    # augment_data()
    print(augment_shift(pd.read_csv("training_data/vectorized_data.csv", header=None).iloc[0]))

