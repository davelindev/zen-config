"""Configuration Schema using Pydantic

- Defines base configs for each group in main config.yaml
- Pydantic models provide type validation, debugging, parameter sharing etc.
- Enables high flexibility of experimentation through detailed
configuration parameters

Author: David Lindevelt
Date: 30May24
"""

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, validator


class EmbeddingConfig(BaseModel):
    """Configuration for embedding layer."""

    model_config = ConfigDict(validate_assignment=True)

    num_embeddings: int = Field(..., gt=0, description="Size of the vocabulary")
    embedding_dim: int = Field(
        default=1024, gt=0, description="Dimension of embeddings"
    )
    padding_idx: Optional[int] = Field(
        default=None, ge=0, description="Padding token index"
    )
    max_norm: Optional[Annotated[float, Field(strict=True, gt=0)]] = Field(
        default=None, description="Maximum norm for embeddings"
    )
    norm_type: float = Field(
        default=2.0, gt=0, description="Norm type for max_norm renormalization"
    )
    scale_grad_by_freq: bool = Field(
        default=False, description="Scale gradients by word frequency"
    )
    sparse: bool = Field(default=False, description="Whether to use sparse gradients")
    freeze: bool = Field(
        default=False, alias="_freeze", description="Whether to freeze embeddings"
    )

    @field_validator("padding_idx")
    @classmethod
    def validate_padding_idx(cls, v: Optional[int], info) -> Optional[int]:
        if v is not None and "num_embeddings" in info.data:
            if v >= info.data["num_embeddings"]:
                raise ValueError(
                    f"padding_idx must be less than {info.data['num_embeddings']}"
                )
        return v


class DataConfig(BaseModel):
    """Data configuration for training and validation.

    Equivalent to the DataConfig dataclass in the original implementation.
    """

    train_files: List[str]
    val_files: List[str]
    context_length: int
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 1234


class OptimizerConfig(BaseModel):
    """Optimizer configuration parameters.

    Equivalent to the OptimizerConfig dataclass in the original implementation.
    """

    lr: float = 1e-3
    weight_decay: float = 0.0

    # For demonstrating parameter relationships
    warmup_steps: int = 1000
    total_steps: int = 50000
    min_lr_ratio: float = 0.1


class ModelConfig(BaseModel):
    """Model architecture configuration.

    Equivalent to the ModelConfig dataclass in the original implementation.
    """

    vocab_size: int
    dim_model: int
    num_heads: int
    num_layers: int = 6
    dropout: float = 0.1
    dim_feedforward: Optional[Union[int, str]] = (
        None  # Will be calculated from dim_model
    )
    context_length: Optional[Union[int, str]] = None  # Will be linked to DataConfig
    text_embedding: EmbeddingConfig = Field(
        default_factory=lambda: EmbeddingConfig(
            num_embeddings=32_000, embedding_dim=512
        ),
        description="Text embedding configuration",
    )

    model_config = ConfigDict(serialize_by_alias=True)

    # Validation example
    @validator("num_heads")
    def validate_heads(cls, v, values):
        if "dim_model" in values and values["dim_model"] % v != 0:
            raise ValueError(f"num_heads must divide {values['dim_model']}")
        return v


class TrainingConfig(BaseModel):
    """Training process configuration.

    Equivalent to the TrainingConfig dataclass in the original implementation.
    """

    output_dir: str
    max_steps: int = 50000
    save_every: int = 5000
    eval_every: int = 1000
    seed: int = 1234
    precision: str = "16-mixed"
    devices: int = 1
    accelerator: Literal["cpu", "gpu", "tpu"] = "gpu"


class Config(BaseModel):
    """Main configuration that combines all sub-configurations.

    Equivalent to the Config dataclass in the original implementation.
    """

    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    experiment_name: str = "default"
    debug: bool = False

    # Allow for extra fields in the config to support hydra-specific fields
    class Config:
        extra = "allow"
