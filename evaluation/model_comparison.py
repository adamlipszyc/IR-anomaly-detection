import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging 
import sys 
from rich.logging import RichHandler
from argparse import Namespace
import re
from typing import Tuple
from tqdm import tqdm
from model_development.training_manager import TrainingManager
from evaluation.evaluate_model import AnomalyDetectionEvaluator
from log.utils import catch_and_log
from plotting.utils import boxplot
from .config import COMPARISON_DIR, RESULTS_DIR


num_splits = 5



@catch_and_log(Exception, "Loading results from Excel")
def load_results_from_excels(results_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rebuilds the results_50 and results_95 DataFrames from Excel files.

    Args:
        results_dir (str): Base directory where evaluation results are stored

    Returns:
        Tuple of DataFrames: (results_50, results_95)
    """
    results_50 = []
    results_95 = []

    count = 0

    for root, _, files in os.walk(results_dir):
        for file in files:
            if file == "evaluation_metrics.xlsx":
                path = os.path.join(root, file)

                # Extract metadata from path
                # e.g. "evaluation/results/isolation_forest/split_1/test_50_50/evaluation_metrics.xlsx"
                rel_path = os.path.relpath(path, results_dir)
                parts = rel_path.split(os.sep)

                if len(parts) < 4:
                    continue  # Skip malformed paths

                if parts[1] in {"augmented", "ensemble"}:
                    continue # Skip augmented results

                count += 1

                try:
                    
                    split = int(parts[-3][-1])
                    split_type = parts[-2].removeprefix("test_")  
                    model = parts[0]
                except Exception as e:
                    print(f"Skipping {path}: {e}")
                    continue

                try:
                    summary_df = pd.read_excel(path, sheet_name="Summary")
                    row = summary_df.iloc[0].to_dict()

                    # Rename human-friendly metrics to standardized keys
                    row["f2_score"] = row.pop("F2 Score (Recall-Weighted)", None)
                    row["f5_score"] = row.pop("F5 Score (Heavily Recall-Weighted)", None)

                    row.update({
                        "model": model,
                        "split": split
                    })

                    if split_type == "50_50":
                        results_50.append(row)
                    elif split_type == "95_5":
                        results_95.append(row)
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    continue

    print(f"Collected results from: {count} files")
    return pd.DataFrame(results_50), pd.DataFrame(results_95)

def main() -> None:

    # run_experiments()

    sns.set_theme(style="whitegrid")

    results_df_50, results_df_95 = load_results_from_excels(RESULTS_DIR)

    f2_metric = "f2_score"
    f5_metric = "f5_score"

    x = "model"
    ys = [f2_metric, f5_metric]
    results = [(results_df_50, True), (results_df_95, False)]

    for y in ys:
        for result, fifty_fifty in results:
            boxplot(result, x, y, dir_path=COMPARISON_DIR, fifty_fifty=fifty_fifty)
    

if __name__ == "__main__":

    # Logging config with Rich
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    log = logging.getLogger(__name__)

    try:
        main()
    except Exception as e:
        log.exception("Unexpected failure")
        sys.exit(99)



