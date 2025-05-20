from argparse import Namespace
from model_development.training_manager import TrainingManager
from evaluation.evaluate_model import AnomalyDetectionEvaluator
from .pipeline_config import PipelineConfig

class EvaluationPipelineRunner:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self):
        train_args = {
            "one_svm": "one_svm" in self.config.base_models ,
            "isolation_forest": "isolation_forest" in self.config.base_models,
            "local_outlier": "LOF" in self.config.base_models,
            "train_ensemble": self.config.train_ensemble,
            "encoder": self.config.encoder,
            "encoding_dim": self.config.encoding_dim,
            "augment_techniques": self.config.augment_techniques,
            "augment_factor": self.config.augment_factor,
        }

        manager = TrainingManager(Namespace(**train_args))
        manager.run()

        augmented_dir_name = (
            "_".join(self.config.augment_techniques) + f"_{self.config.augment_factor}"
            if self.config.augment_techniques is not None and self.config.augment_factor is not None
            else None
        )

        for base_model in self.config.base_models:
            eval_args = {
                "model_name": base_model,
                "ensemble_voting": self.config.train_ensemble,
                "augmented_dir_name": augmented_dir_name,
                "encoder": self.config.encoder,
                "encoding_dim": self.config.encoding_dim,
            }

            evaluator = AnomalyDetectionEvaluator(Namespace(**eval_args))
            evaluator.evaluate_model()
