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
from plotting.utils import lineplot, boxplot

techniques_list = [
    ["none"],
    ["magnitude"],
    ["shift"],
    ["noise"],
    ["magnitude", "shift"],
    ["magnitude", "noise"],
    ["shift", "noise"],
    ["magnitude", "shift", "noise"]
]

num_splits = 5

size_factors = [1, 2, 3, 4, 5]  # e.g., 2133, 4266, ..., 10665

EXPERIMENT_DIR = "data_augmentation/experiment_results/"

RESULTS_DIR = "evaluation/results/isolation_forest/augmented/"

@catch_and_log(Exception, "Running experiment")
def run_experiments():
    """
    Runs all experiments and stores results.
    """

    total_iterations = num_splits * len(techniques_list) * len(size_factors)
    progress = tqdm(total=total_iterations, desc="Running experiments")


    for techniques in techniques_list:
        for factor in size_factors:

            training_args = {
                "augment_factor": factor,
                "augment_techniques": techniques,
                "isolation_forest": True,
                "one_svm": False,
                "local_outlier": False,
                "train_ensemble": False,
            }

            training_args = Namespace(**training_args)

            manager = TrainingManager(training_args)


            #Train model and save it
            manager.run()

            evaluation_args = {
                "model_name": "isolation_forest",
                "ensemble_voting": False,
                "augmented_dir_name": "_".join(techniques) + f"_{factor}"
            }

            evaluator = AnomalyDetectionEvaluator(Namespace(**evaluation_args))

            evaluator.evaluate_model()


            progress.update(1)

    progress.close()

    return None 



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
                # e.g. "evaluation/results/isolation_forest/augmented/magnitude_shift_2/50-50/evaluation_metrics.xlsx"
                rel_path = os.path.relpath(path, results_dir)
                parts = rel_path.split(os.sep)

                if len(parts) < 4:
                    continue  # Skip malformed paths

                count += 1

                try:
                    augmented_dir = parts[-4]  #  "magnitude_shift_2"
                    split = int(parts[-3][-1])
                    split_type = parts[-2].removeprefix("test_")  
                    techniques, factor = "_".join(augmented_dir.split("_")[:-1]), int(augmented_dir.split("_")[-1])
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
                        "techniques": techniques.replace("_", "+"),
                        "factor": factor,
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

    x = "factor"
    ys = [f2_metric, f5_metric]
    results = [(results_df_50, True), (results_df_95, False)]

    for y in ys:
        for result, fifty_fifty in results:
            lineplot(result, 
                    x, 
                    y, 
                    hue="techniques", 
                    title=f"Model {y} vs Augmentation {x.capitalize()}", 
                    xlabel="Training Set Increase Size Factor (Augmented)",
                    ylabel=f"{y[:2]} Score",
                    dir_path=EXPERIMENT_DIR,
                    fifty_fifty=fifty_fifty
                    )
            
            boxplot(result, "techniques", y, dir_path=EXPERIMENT_DIR, fifty_fifty=fifty_fifty)
    

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



