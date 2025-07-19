import hydra

from il_lib.utils.training_utils import seed_everywhere
from il_lib.utils.config_utils import omegaconf_to_dict
from il_lib.training import Trainer


@hydra.main(config_name="base_config", config_path="il_lib/configs", version_base="1.1")
def main(cfg):
    cfg.seed = seed_everywhere(cfg.seed)
    trainer_ = Trainer(cfg)
    trainer_.trainer.loggers[-1].log_hyperparams(omegaconf_to_dict(cfg))
    trainer_.fit()
    trainer_.test()


if __name__ == "__main__":
    main()
