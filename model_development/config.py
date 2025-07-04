
ORIGINAL_DATA_FILE_PATH = "training_data/original_data/vectorized_data.csv"
TRAINING_TESTS_SPLITS_DIRECTORY = "training_test_splits/"

AUGMENTED_DATA_DIR = "data_augmentation/augmented_data_without_noise/" 
SAMPLES_PER_FILE = 20                # Rows to sample per file
NUM_BATCHES = 10                         # Total number of models to train
OUTPUT_MODEL_PREFIX = "batch_model"      # Prefix for saved models
SEED = 42                                # Reproducibility

HYPERPARAMETER_FILEPATH = "model_development/hyperparameters.json"

OPTIMAL_HYPERPARAMETER_FILEPATH = "model_development/optimal_hyperparameters.json"

VALIDATION_DATA_DIR = "evaluation/anomalous_data/"

base_models = {"one_svm", "isolation_forest", "LOF"}