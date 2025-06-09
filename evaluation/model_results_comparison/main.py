import pandas as pd
import seaborn as sns
import os
import logging 
import sys 
import argparse
from rich.logging import RichHandler
from typing import Tuple
from log.utils import catch_and_log
from plotting.utils import boxplot
from ..config import RESULTS_DIR
from .config import COMPARISON_DIR
from model_development.config import base_models

num_splits = 5


shorten = {
    "autoencoder": "AE",
    "isolation_forest": "IF",
    "one_svm": "OSVM",
    "LOF": "LOF",
    "pca": "PCA",
    "cnn_anogan": "CNN_ANOGAN",
    "anogan": "MLP_ANOGAN",
    "cnn_supervised_2d": "CNN_CLASSIFIER",
    "lstm": "LSTM_CLASSIFIER"
}



@catch_and_log(Exception, "Loading results from Excel")
def load_results_from_excels(results_dir: str, base: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
                
                if "old" in rel_path:
                    continue #Skip old results


                if parts[0] == "hybrid":
                    if parts[1] == "pre_scale":
                        continue 

                    if base:
                        continue # only comparing base models, skip

                    if len(parts) < 6:
                        continue
                    encoder = parts[1]
                    model = shorten[encoder] + f"+" + shorten[parts[2]]

                else:
                    model = shorten[parts[0]]
                    

                if parts[1] in {"augmented", "ensemble"}:
                    continue # Skip augmented results

                count += 1

                try:
                    split = int(parts[-3][-1])
                    test_type = parts[-2].removeprefix("test_")  
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

                    if test_type == "50_50":
                        results_50.append(row)
                    elif test_type == "95_5":
                        results_95.append(row)
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    continue

    print(f"Collected results from: {count} files")
    return pd.DataFrame(results_50), pd.DataFrame(results_95)

def main(base: bool) -> None:

    # run_experiments()

    sns.set_theme(style="whitegrid")

    results_df_50, results_df_95 = load_results_from_excels(RESULTS_DIR, base)

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

    parser = argparse.ArgumentParser(description="Compare models")
    parser.add_argument('--base', action='store_true', help="Name of model algorithm")

    args = parser.parse_args()
    try:
        main(args.base)
    except Exception as e:
        log.exception("Unexpected failure")
        sys.exit(99)



