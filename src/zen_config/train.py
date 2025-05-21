import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from zen_config.conf import base_configs as bc



OmegaConf.clear_resolvers()
OmegaConf.register_new_resolver("sum", lambda x, y: x + y)
OmegaConf.register_new_resolver("multiply", lambda a, b: a * b)


csi = ConfigStore.instance()
csi.store(name="base_config", node=bc.Config)
csi.store(group="data", name="base_data", node=bc.DataConfig)
csi.store(group="model", name="base_model", node=bc.ModelConfig)
csi.store(group="optimizer", name="base_optimizer", node=bc.OptimizerConfig)
csi.store(group="training", name="base_training", node=bc.TrainingConfig)


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()

