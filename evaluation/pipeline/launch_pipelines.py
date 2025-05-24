import logging
import sys
from rich.logging import RichHandler
from itertools import product
from tqdm import tqdm
from .pipeline_config import PipelineConfig
from .evaluation_pipeline_runner import EvaluationPipelineRunner


def autoencoder_parameters():
    base_models = ["autoencoder"] #"isolation_forest", "one_svm", "LOF"]
    lrs = [0.1, 0.01, 0.001]
    batch_sizes = [32, 64, 128, 256]
    num_epochs_choices = [10, 20, 50]
    encoders = [None] #"autoencoder", "pca"]
    encoding_dims = [16, 32, 64, 128, 256]  # If None, will be ignored
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    augment_techniques_list = [
        None,
        # ["none"],
        # ["magnitude"],
        # ["shift"],
        # ["noise"],
        # ["magnitude", "shift"]
    ]
    augment_factors = [None] #1, 2, 3, 4, 5]

def launch_pipelines():
    base_models = ["autoencoder"] #"isolation_forest", "one_svm", "LOF"]
    lrs = [0.1, 0.01, 0.001]
    batch_sizes = [32, 64, 128, 256]
    num_epochs_choices = [10, 20, 50]
    encoders = [None] #"autoencoder", "pca"]
    encoding_dims = [16, 32, 64, 128, 256]  # If None, will be ignored
    augment_techniques_list = [
        None,
        # ["none"],
        # ["magnitude"],
        # ["shift"],
        # ["noise"],
        # ["magnitude", "shift"]
    ]
    augment_factors = [None] #1, 2, 3, 4, 5]
    threshold = True

    configs = []
    for encoder, encoding_dim, techniques, factor, lr, batch_size, num_epochs in product(
        encoders, encoding_dims, augment_techniques_list, augment_factors, lrs, batch_sizes, num_epochs_choices
    ):
        # if encoder is None and encoding_dim is not None:
        #     continue  # skip invalid combo

        # if techniques is None and factor is not None:
        #     continue  # skip invalid combo

        config = PipelineConfig(
            base_models=base_models,
            encoder=encoder,
            encoding_dim=encoding_dim,
            augment_techniques=techniques,
            augment_factor=factor,
            lr=lr,
            batch_size=batch_size,
            num_epochs=num_epochs,
            threshold=threshold
        )
        configs.append(config)

    print(f"Launching {len(configs)} experiments")

    progress = tqdm(total=len(configs), desc="Running experiments")
    for config in configs:
        runner = EvaluationPipelineRunner(config)
        runner.run()
        progress.update(1)
    progress.close()


if __name__ == "__main__":
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    log = logging.getLogger(__name__)

    try:
        launch_pipelines()
    except Exception as e:
        log.exception("Unexpected failure during experiment run")
        sys.exit(1)
