import logging 
import os 
import csv 
import pandas as pd
import multiprocessing as mp
from .data_augmenter import DataAugmenter
from log.utils import catch_and_log
from tqdm import tqdm

# File paths
INPUT_FILE = "training_data/original_data/vectorized_data.csv"   # Your input CSV file
TEMP_DIR = "data_augmentation/augmented_data_without_noise_shift"   # Temp directory
FINAL_OUTPUT_FILE = "data_augmentation/augmented_data/augmented_data.csv"  # Merged output file

os.makedirs(TEMP_DIR, exist_ok=True)

def process_wrapper(row_id, row_data):
    return DataAugmenter().process_row(row_id, row_data)

class AugmentedDataGenerator():

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def worker(queue: mp.Queue, progress_queue: mp.Queue) -> None:
        """Worker function: reads from the queue and processes rows."""
        while True:
            task = queue.get()
            if task is None:  # Stop condition
                break
            row_id, row_data = task
            augmented_data = process_wrapper(row_id, row_data)
            temp_file = os.path.join(TEMP_DIR, f"row_{row_id}.csv")
            augmented_data.to_csv(temp_file, index=False, header=False)
            progress_queue.put(1)  # Report completion of a task

    @staticmethod
    @catch_and_log(Exception, "Merging files")
    def merge_files() -> None:
        """Merge all temporary CSV files into a single output CSV."""
        with open(FINAL_OUTPUT_FILE, "w", newline="") as fout:
            writer = csv.writer(fout)

            for temp_file in sorted(os.listdir(TEMP_DIR)):  # Ensure correct order
                with open(os.path.join(TEMP_DIR, temp_file), "r", newline="") as f:
                    reader = csv.reader(f)
                    writer.writerows(reader)


    def generate(self) -> None:
        # Count rows in the input file for progress tracking
        total_rows = sum(1 for _ in open(INPUT_FILE))  # Count lines in the CSV
        num_workers = mp.cpu_count() - 1  # Use available CPU cores
        queue = mp.Queue()
        progress_queue = mp.Queue()  # For updating progress bar

        workers = [mp.Process(target=self.worker, args=(queue, progress_queue)) for _ in range(num_workers)]

        # Start workers
        for w in workers:
            w.start()

        # Set up progress bar
        progress_bar = tqdm(total=total_rows, desc="Processing rows", unit="row")

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