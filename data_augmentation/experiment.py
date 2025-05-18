import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging 
import sys 
from rich.logging import RichHandler
from tqdm import tqdm
from model_development.training_manager import TrainingManager
from evaluation.evaluate_model import AnomalyDetectionEvaluator
from argparse import Namespace

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

EXPERIMENT_DIR = "data_augmentation/results/"

def run_experiments():
    """
    Runs all experiments and stores results.
    """
    results_50 = []
    results_95 = []

    total_iterations = num_splits * len(techniques_list) * len(size_factors)
    progress = tqdm(total=total_iterations, desc="Running experiments")


    for split in range(1, num_splits + 1):
        for techniques in techniques_list:
            for factor in size_factors:

                training_args = {
                    "augment_factor": factor,
                    "augment_techniques": techniques,
                    "isolation_forest": True,
                    "one_svm": False,
                    "local_outlier": False,
                    "train_ensemble": False,
                    "split": split
                }

                training_args = Namespace(**training_args)

                manager = TrainingManager(training_args)


                #Train model and save it
                manager.run()

                evaluation_args = {
                    "model_name": "isolation_forest",
                    "split": split,
                    "fifty_fifty": True,
                    "ensemble_voting": False,
                    "augmented_dir_name": "_".join(techniques) + f"_{factor}"
                }

                evaluator = AnomalyDetectionEvaluator(Namespace(**evaluation_args))

                metrics = evaluator.evaluate_model(return_metrics=True)

                results_50.append({
                    "split": split,
                    "techniques": "+".join(techniques),
                    "factor": factor,
                    **metrics
                })

                evaluation_args["fifty_fifty"] = False
                evaluator = AnomalyDetectionEvaluator(Namespace(**evaluation_args))

                metrics = evaluator.evaluate_model(return_metrics=True)

                results_95.append({
                    "split": split,
                    "techniques": "+".join(techniques),
                    "factor": factor,
                    **metrics
                })

                progress.update(1)

    progress.close()

    return (pd.DataFrame(results_50), pd.DataFrame(results_95))



def plot_metric_vs_factor(results_df, metric: str, fifty_fifty: bool = True):
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=results_df,
        x="factor",
        y=f"{metric}_score",
        hue="techniques",
        estimator="mean",
        ci="sd",
    )

    plt.title(f"Model {metric} vs Augmentation Factor")
    plt.xlabel("Training Set Increase Size Factor (Augmented)")
    plt.ylabel(f"{metric} Score")
    plt.legend(title="Techniques")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(EXPERIMENT_DIR, f"{metric}_vs_augmentation_factor_{"50-50" if fifty_fifty else "95-5"}.png"))


def plot_metric_vs_techniques(results_df, metric: str, fifty_fifty: bool = True):
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=results_df,
        x="techniques",
        y=f"{metric}_score"
    )

    plt.title("Technique Comparison: Accuracy Across Splits")
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(EXPERIMENT_DIR, f"{metric}_vs_techniques_{"50-50" if fifty_fifty else "95-5"}.png"))


def main() -> None:

    results_df_50, results_df_95 =  run_experiments()

    sns.set_theme(style="whitegrid")

    plot_metric_vs_factor(results_df_50, "f2")
    
    plot_metric_vs_factor(results_df_50, "f5")

    plot_metric_vs_factor(results_df_95, "f2", fifty_fifty = False)

    plot_metric_vs_factor(results_df_95, "f5", fifty_fifty = False)

    plot_metric_vs_techniques(results_df_50, "f2")

    plot_metric_vs_techniques(results_df_50, "f5")

    plot_metric_vs_techniques(results_df_95, "f2", fifty_fifty= False)

    plot_metric_vs_techniques(results_df_95, "f5", fifty_fifty= False)


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

    

