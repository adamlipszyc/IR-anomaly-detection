from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PipelineConfig:
    base_models: List[str]                          # Required
    encoder: Optional[str] = None            # "autoencoder", "pca", or None
    encoding_dim: Optional[int] = None       # Used only if encoder is not None
    augment_techniques: Optional[List[str]] = None  # e.g. ["shift", "magnitude"]
    augment_factor: Optional[int] = None     # Integer size factor
    train_ensemble: bool = False
    lr: Optional[float] = None 
    batch_size: Optional[int] = None
    num_epochs: Optional[int] = None

