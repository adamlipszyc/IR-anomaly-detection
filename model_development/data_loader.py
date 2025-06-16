
import pandas as pd
from multiprocessing import Process, Queue, cpu_count
from tqdm import tqdm
import random
from log.utils import catch_and_log
import logging

class DataLoader:
    def __init__(self, data_dir: str, samples_per_file: int):
        self.data_dir = data_dir
        self.samples_per_file = samples_per_file
        self.logger = logging.getLogger(self.__class__.__name__)

    def _worker(self, input_queue: Queue, output_queue: Queue, shared_set: set):
        """
        Worker process to read and sample files from input_queue.
        """
        while True:
            file_path = input_queue.get()
            if file_path is None:
                break  # Exit signal

            try:
                df = pd.read_csv(file_path, header=None)
                if len(df) > self.samples_per_file:
                    df = df.sample(self.samples_per_file,  random_state=random.randint(0, 99999))
                output_queue.put(df)

                # Track if row 0 was included
                if 0 in df.index:
                    shared_set.add(file_path)

            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")
            finally:
                output_queue.put(None)  # Mark completion

    @catch_and_log(Exception, "Loading batch in parallel")
    def load_batch_parallel(self, csv_files: list[str] = None, num_workers: int = None):
        """
        Process multiple files in parallel and return batch DataFrames and sampled first-row file list.
        """
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)

        input_queue = Queue()
        output_queue = Queue()
        first_row_sampled_set = set()  

        # Start workers
        workers = []
        for _ in range(num_workers):
            p = Process(target=self._worker, args=(input_queue, output_queue, first_row_sampled_set))
            p.start()
            workers.append(p)

        # Enqueue file paths
        for file_path in csv_files:
            input_queue.put(file_path)

        # Send sentinel values to stop workers
        for _ in range(num_workers):
            input_queue.put(None)
        
        batch_data = []
        # Collect data with progress bar
        with tqdm(total=len(csv_files), desc="Processing files") as pbar:
            finished_files = 0
            while finished_files < len(csv_files):
                result = output_queue.get()
                if result is None:
                    finished_files += 1
                    pbar.update(1)
                else:
                    batch_data.append(result)

        # Clean up
        for p in workers:
            p.join()
        

        self.logger.info("Done with files")
        return (batch_data, first_row_sampled_set)

    @staticmethod
    def load_original_data(filepath: str) -> pd.DataFrame:
        """
        Load the original unaugmented dataset.
        """
        return pd.read_csv(filepath, header=None)
