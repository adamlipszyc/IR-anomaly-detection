import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import logging 
import sys 
from rich.logging import RichHandler
from log.utils import catch_and_log




@catch_and_log(Exception, "Plotting boxplot")
def boxplot(results_df: pd.DataFrame, x: str, y: str, dir_path: str = None, fifty_fifty: bool = True):
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=results_df,
        x=x,
        y=y
    )

    plt.title(f"{x.capitalize()} Comparison: {y.capitalize()} Across Splits")
    plt.xticks(rotation=45)
    plt.ylabel(y)
    plt.tight_layout()
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{y}_vs_{x}_{"50_50" if fifty_fifty else "95_5"}.png")
        plt.savefig(path)
    # plt.show()
    plt.close()


@catch_and_log(Exception, "Plotting line plot")
def lineplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str = None,
    estimator: str = "mean",
    errorbar: str = "sd",
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    dir_path: str = None,
    show: bool = False,
    fifty_fifty: bool = True
):
    """
    General-purpose line plot with optional grouping and error bars.

    Args:
        data (pd.DataFrame): Data to plot.
        x (str): Column for x-axis.
        y (str): Column for y-axis.
        hue (str, optional): Column to group lines by color.
        estimator (str): Aggregation function (e.g., 'mean').
        errorbar (str): Error bar (e.g., 'sd', 'ci', None).
        title (str): Title of the plot.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        save_path (str): File path to save the plot.
        show (bool): Whether to display the plot.
    """
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        estimator=estimator,
        errorbar=errorbar
    )

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    if hue:
        plt.legend(title=hue.capitalize())

    plt.tight_layout()

    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{y}_vs_{x}_{"50_50" if fifty_fifty else "95_5"}.png")
        plt.savefig(path)

    if show:
        plt.show()

    plt.close()


