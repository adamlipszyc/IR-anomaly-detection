# config.py
"""
Configuration constants for scenario generation file paths.
"""
from pathlib import Path

# Base directory for generated scenarios
DATA_DIR: Path = Path("evaluation/data/scenarios")

# Scenario CSV file paths
FILE_SCEN_4: Path = DATA_DIR / "generated_scen_4.csv"
FILE_SCEN_9: Path = DATA_DIR / "generated_scen_9.csv"
FILE_SCEN_10: Path = DATA_DIR / "generated_scen_10.csv"
FILE_SCEN_11: Path = DATA_DIR / "generated_scen_11.csv"

# Permutation example file paths
BASE_PERMUTATION_DIR: Path = DATA_DIR.parent
SIMPLE_SCENARIOS: Path = BASE_PERMUTATION_DIR / "simple_scenarios.csv"
SIMPLE_SCENARIOS_WITH_REAL: Path = BASE_PERMUTATION_DIR / "simple_scenarios_with_real.csv"
REPEAT_SCENARIOS: Path = BASE_PERMUTATION_DIR / "repeat_scenarios.csv"
REPEAT_SCENARIOS_WITH_REAL: Path = BASE_PERMUTATION_DIR / "repeat_scenarios_with_real.csv"