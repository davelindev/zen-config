"""Configuration Store using hydra_zen

This module sets up the hydra_zen config store with our pydantic models.
It creates the equivalent of the Hydra ConfigStore but using hydra_zen's
more Pythonic API.

Author: David Lindevelt
Date: 30May24
"""

from hydra_zen import make_custom_builds_fn, store
from omegaconf import OmegaConf

from zen_config.hydra_zen.conf.config_schemas import (
    Config,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)

# Register OmegaConf resolvers for value interpolation
# Note: These will be available globally in the application
OmegaConf.clear_resolvers()
OmegaConf.register_new_resolver("sum", lambda x, y: x + y)
OmegaConf.register_new_resolver("multiply", lambda a, b: a * b)

# Create a builds function that converts pydantic models to hydra_zen builds
builds_fn = make_custom_builds_fn(populate_full_signature=True)

# Register base configurations
store(builds_fn(DataConfig), group="data", name="base_data")
store(builds_fn(ModelConfig), group="model", name="base_model")
store(builds_fn(OptimizerConfig), group="optimizer", name="base_optimizer")
store(builds_fn(TrainingConfig), group="training", name="base_training")

# Store the main config
store(builds_fn(Config), name="base_config")
