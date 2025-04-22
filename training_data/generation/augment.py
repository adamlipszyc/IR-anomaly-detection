import numpy as np
import pandas as pd
import random
import multiprocessing as mp
import os
import csv
from tqdm import tqdm  # For the progress bar
import time 

def augment_magnitude(vector):
    """Randomly scales nonzero values """
    # print("magnitudes")
    augmented_results = []
    min_scale, max_scale = 0.1, 5

    #Generates 10 new entries by randomly sampling a scale in range 0.1-5 and
    #mulitplying our original vector by this
    for i in range(10):
        scaling_factor = np.random.uniform(min_scale, max_scale)
        #Randomly rounds the vector to a whole number approximately 50% of the time 
        isRounded = np.random.uniform(0, 1) > 0.5
        result = vector * scaling_factor
        if isRounded:
            result = result.round(0)
        else:
            #Randomly selects the number of decimal places to round to 
            decimals = np.floor(np.random.uniform(1, 8)).astype(int)
            result = result.round(decimals)
        augmented_results.append(result)

    return augmented_results

def augment_shift(vector):
    """ 
        Shifts vector according to first non-zero value and last-non zero 
        ensuring time frame between positions is preserved 
        but the actual day the positions fall on can change 
    """
    # print("shifting")
    # find first and last non-zero value 
    first_nonzero_index = (vector != 0).idxmax()
    last_nonzero_index = vector.where(vector != 0).last_valid_index()
    
    augmented_results = []
    if first_nonzero_index:
        possible_left_shifts = first_nonzero_index
        possible_right_shifts = len(vector) - last_nonzero_index - 1
        for val in range(2, possible_left_shifts, 2):
            augmented_results.append(np.roll(vector, -val))

        for val in range(2, possible_right_shifts, 2):
            augmented_results.append(np.roll(vector, val))
    
    return augmented_results


def augment_noise_fast(vector, noise_level=0.1, num_augmentations=1):
    """Fast version of adding small noise to zero value pairs."""
    print("adding noise (fast)")
    vector = np.array(vector)

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


def augment_noise_vectorized(vector, noise_level=0.1, num_augmentations=1):
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


def augment_noise(vector, noise_level=0.1, num_augmentations=1):
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




# File paths
INPUT_FILE = "training_data/original_data/vectorized_data.csv"   # Your input CSV file
TEMP_DIR = "augmented_data"   # Temp directory
FINAL_OUTPUT_FILE = "training_data/augmented_data/augmented_data.csv"  # Merged output file

os.makedirs(TEMP_DIR, exist_ok=True)


def process_row(row_id, row_data):
    """Apply augmentation to a single row and write to a temp CSV."""
    temp_file = os.path.join(TEMP_DIR, f"row_{row_id}.csv")
    
    print("augmenting magnitude")
    augmented_data = pd.concat([pd.DataFrame([row_data]), pd.DataFrame(augment_magnitude(row_data))], ignore_index=True)
    print("shifting results")
    shift_results = []
    for index, row in augmented_data.iterrows():
        shift_results.append(pd.DataFrame(augment_shift(row)))
    
    augmented_data = pd.concat([augmented_data, pd.concat(shift_results, ignore_index=True)], ignore_index=True)

    print("adding noise")                             
    noise_results = []
    for index, row in augmented_data.iterrows():
        noise_results.append(pd.DataFrame(augment_noise_vectorized(row)))
    
    augmented_data = pd.concat([augmented_data, pd.concat(noise_results, ignore_index=True)], ignore_index=True)

    augmented_data.to_csv(temp_file, index=False, header=False)


def worker(queue, progress_queue):
    """Worker function: reads from the queue and processes rows."""
    while True:
        task = queue.get()
        if task is None:  # Stop condition
            break
        row_id, row_data = task
        process_row(row_id, row_data)
        progress_queue.put(1)  # Report completion of a task


def merge_files():
    """Merge all temporary CSV files into a single output CSV."""
    with open(FINAL_OUTPUT_FILE, "w", newline="") as fout:
        writer = csv.writer(fout)

        for temp_file in sorted(os.listdir(TEMP_DIR)):  # Ensure correct order
            with open(os.path.join(TEMP_DIR, temp_file), "r", newline="") as f:
                reader = csv.reader(f)
                writer.writerows(reader)  # Append each row

    # # Cleanup temporary files
    # for temp_file in os.listdir(TEMP_DIR):
    #     os.remove(os.path.join(TEMP_DIR, temp_file))
    # os.rmdir(TEMP_DIR)

def main():
    # Count rows in the input file for progress tracking
    total_rows = sum(1 for _ in open(INPUT_FILE))  # Count lines in the CSV
    num_workers = mp.cpu_count() - 1  # Use available CPU cores
    queue = mp.Queue()
    progress_queue = mp.Queue()  # For updating progress bar

    workers = [mp.Process(target=worker, args=(queue, progress_queue)) for _ in range(num_workers)]

    # Start workers
    for w in workers:
        w.start()

    # Set up progress bar
    progress_bar = tqdm(total=total_rows, desc="Processing rows", unit="row")

    # # Read CSV line-by-line and push to queue (no header)
    # with open(INPUT_FILE, "r") as f:
    #     reader = csv.reader(f)
    #     for row_id, row_data in enumerate(reader):
    #         queue.put((row_id, list(map(float, row_data))))  # Convert to float if needed

    df = pd.read_csv(INPUT_FILE, header=None)

    for row_id, row in df.iterrows():
        queue.put((row_id, row))

    # Stop workers
    for _ in workers:
        queue.put(None)

    # Track progress updates from the workers
    completed_tasks = 0
    while completed_tasks < total_rows:
        progress_queue.get()  # Wait for a task completion
        completed_tasks += 1
        progress_bar.update(1)  # Update progress bar

    # Join workers after completion
    for w in workers:
        w.join()

    # Merge temporary files
    # merge_files()

    # Close progress bar
    progress_bar.close()

if __name__ == "__main__":
    # main()
    merge_files()
    # pd.DataFrame(augment_shift(pd.read_csv("training_data/vectorized_data.csv", header=None).iloc[0])).to_csv("practice/shift.csv", index=False, header=False)

    # row = pd.read_csv("training_data/vectorized_data.csv", header=None).iloc[0]
    # start = time.time()
    # process_row(1, row)
    # end = time.time()
    # print("Time taken: ", end - start, " seconds")


    # start = time.time()
    # augment_noise(row)
    # end = time.time()
    # print("slow took: ", end - start, " seconds")
    # start = time.time()
    # pd.DataFrame(augment_noise_vectorized(row)).to_csv("practice/noise_vectorized.csv", index=False, header=False)
    # end = time.time()
    # print("fast took: ", end - start, " seconds")




# def augment_data():
#     """Generates augmented data by applying magnitude, noise and shift transformations."""

#     file_path = "training_data/vectorized_data.csv"
#     df = pd.read_csv(file_path, header=None) #load data from csv file 

#     augmented_data = df
#     jobs = [1, 2, 3]
#     func = None
#     for i in jobs:
#         if i == 1:
#             func = augment_magnitude
#         elif i == 2:
#             func = augment_shift 
#         else:
#             func = augment_noise

#         new_entries = []
#         for index, row in augmented_data.iterrows():
#             new_entries += func(row)
        
#         augmented_data = pd.concat([augmented_data, pd.DataFrame(new_entries)], ignore_index=True)

#     output_file_path = "training_data/augmented_data.csv"
#     augmented_data.to_csv(output_file_path, index=False, header=False)    
    


# if __name__ == "__main__":
#     augment_data()

