# config.py
"""
Configuration constants for scenario generation file paths.
"""
from pathlib import Path

# Base directory for generated scenarios
DATA_DIR: Path = Path("evaluation/anomalous_data/scenarios")

# Scenario CSV file paths
FILE_SCEN_4: Path = DATA_DIR / "generated_scen_4.csv"
FILE_SCEN_9: Path = DATA_DIR / "generated_scen_9.csv"
FILE_SCEN_10: Path = DATA_DIR / "generated_scen_10.csv"
FILE_SCEN_11: Path = DATA_DIR / "generated_scen_11.csv"
FILE_SCEN_PERM: Path = DATA_DIR / "generated_scen_perm.csv"
