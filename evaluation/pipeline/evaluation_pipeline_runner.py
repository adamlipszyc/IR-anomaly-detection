import logging
from argparse import Namespace
from model_development.training_manager import TrainingManager
from evaluation.evaluate_model import AnomalyDetectionEvaluator
from .pipeline_config import PipelineConfig

class EvaluationPipelineRunner:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self):
        
        model_args = {}
        if self.config.lr:
            model_args["lr"] = self.config.lr
        if self.config.encoding_dim:
            model_args["encoding_dim"] = self.config.encoding_dim
        if self.config.batch_size:
            model_args["batch_size"] = self.config.batch_size
        if self.config.num_epochs:
            model_args["num_epochs"] = self.config.num_epochs



        train_args = {
            "one_svm": "one_svm" in self.config.base_models ,
            "isolation_forest": "isolation_forest" in self.config.base_models,
            "local_outlier": "LOF" in self.config.base_models,
            "autoencoder": "autoencoder" in self.config.base_models,
            "train_ensemble": self.config.train_ensemble,
            "encoder": self.config.encoder,
            "encoding_dim": self.config.encoding_dim,
            "augment_techniques": self.config.augment_techniques,
            "augment_factor": self.config.augment_factor,
            "model_args": model_args
        }

        manager = TrainingManager(Namespace(**train_args))
        manager.run()

        augmented_dir_name = (
            "_".join(self.config.augment_techniques) + f"_{self.config.augment_factor}"
            if self.config.augment_techniques is not None and self.config.augment_factor is not None
            else None
        )

        hyper_parameters = [self.config.lr, self.config.batch_size, self.config.num_epochs, self.config.encoding_dim]
        if any(param is None for param in hyper_parameters):
            hyperparameter_dir = None
        else:
            hyper_parameters = [str(hyperparameter) for hyperparameter in hyper_parameters]
            hyperparameter_dir = "_".join(hyper_parameters) 

        
        for base_model in self.config.base_models:
            
            eval_args = {
                "model_name": base_model,
                "ensemble_voting": self.config.train_ensemble,
                "augmented_dir_name": augmented_dir_name,
                "encoder": self.config.encoder,
                "encoding_dim": self.config.encoding_dim,
                "hyperparameter_dir": hyperparameter_dir,
                "threshold": self.config.threshold
            }

            evaluator = AnomalyDetectionEvaluator(Namespace(**eval_args))
            evaluator.evaluate_model()
