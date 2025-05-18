
"""
Configuration constants for evaluation file paths.
"""
from pathlib import Path

ORIGINAL_FILE_PATH = 'training_data/original_data/vectorized_data.csv'

GOOD_DATA = [ORIGINAL_FILE_PATH]

ANOMALOUS_DATA = ['evaluation/data/']

OUTPUT_DIR = "evaluation/results"

CLASS_SIZE = 534

TEST_DATA_DIR = "training_test_splits/"


# Base directory for generated scenarios
MODELS_DIR: Path = Path("models")