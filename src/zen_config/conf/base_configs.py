"""Configuration Schema

- Defines base configs for each group in main config.yaml
- ConfigStore provides type validation, debugging, parameter sharing etc.
- Enables high flexibility of experimentation through detailed
configuration parameters

Author: David Linden
Date: 301441CJun23
"""

from dataclasses import dataclass, field
from typing import List

from omegaconf import MISSING


@dataclass
class DataConfig:
    train_files: List[str] = MISSING
    val_files: List[str] = MISSING
    context_length: int = MISSING
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 1234


@dataclass
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0

    # For demonstrating parameter relationships
    warmup_steps: int = 1000
    total_steps: int = 50000
    min_lr_ratio: float = 0.1


@dataclass
class ModelConfig:
    vocab_size: int = MISSING
    dim_model: int = MISSING
    num_heads: int = MISSING
    num_layers: int = 6
    dropout: float = 0.1
    dim_feedforward: int = MISSING
    context_length: int = MISSING  # This will be linked to DataConfig


@dataclass
class TrainingConfig:
    output_dir: str = MISSING
    max_steps: int = 50000
    save_every: int = 5000
    eval_every: int = 1000
    seed: int = 1234
    precision: str = "16-mixed"
    devices: int = 1
    accelerator: str = "gpu"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = "default"
    debug: bool = False
