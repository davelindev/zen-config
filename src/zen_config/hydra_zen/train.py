"""Main training script that uses hydra_zen for configuration

This is the entry point for the application, which loads configuration
using hydra_zen and hydra.

Author: David Lindevelt
Date: 30May24
"""

import logging

import hydra
from hydra_zen import instantiate, to_yaml
from hydra_zen.third_party.pydantic import pydantic_parser
from omegaconf import DictConfig

# Import the config store to register all configurations
from zen_config.hydra_zen.conf import config_store

# Set up logging
log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="./conf/", config_name="config")
def main(cfg: DictConfig):
    """Main entry point for the application.

    Args:
        cfg: The configuration object loaded by Hydra.
    """
    # Print the resolved configuration
    print("=" * 80)
    print("Configuration (hydra_zen version):")
    print("=" * 80)

    print(to_yaml(cfg, resolve=True))

    cfg = instantiate(cfg, _target_wrapper_=pydantic_parser)
    print(type(cfg.model))
    print(cfg.model)


if __name__ == "__main__":
    main()
