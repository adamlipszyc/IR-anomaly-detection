# config.py
"""
Configuration constants for scenario generation file paths.
"""
from pathlib import Path

# Base directory for generated scenarios
DATA_DIR: Path = Path("evaluation/anomalous_data/scenarios")

# Permutation example file paths
BASE_PERMUTATION_DIR: Path = DATA_DIR.parent
SIMPLE_SCENARIOS: Path = BASE_PERMUTATION_DIR / "simple_scenarios.csv"
COMPLEX_SCENARIOS: Path = BASE_PERMUTATION_DIR / "complex_scenarios.csv"