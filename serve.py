import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from il_lib.policies import ResidualPolicy
from il_lib.utils.config_utils import register_omegaconf_resolvers
from il_lib.utils.training_utils import load_state_dict, load_torch
from omegaconf import OmegaConf
from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
import os
import sys


def main():
    # Initialize Hydra with logging disabled for serve mode
    config_dir = os.path.join(os.path.dirname(__file__), "il_lib/configs")
    config_dir = os.path.abspath(config_dir)
    
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        # Compose config with Hydra logging disabled
        overrides = sys.argv[1:] + [
            "hydra.output_subdir=null",
            "hydra.run.dir=.",
            "hydra/job_logging=none",
            "hydra/hydra_logging=none"
        ]
        cfg = compose(config_name="base_config", overrides=overrides)
        
        register_omegaconf_resolvers()
        OmegaConf.resolve(cfg)
        OmegaConf.set_struct(cfg, False)
        policy = instantiate(cfg.module, _recursive_=False)
        ckpt = load_torch(
            cfg.ckpt_path,
            map_location="cpu",
        )
        if isinstance(policy, ResidualPolicy):
            incompatible_keys = policy.load_state_dict(ckpt["state_dict"], strict=False)
            missing_keys = [
                key for key in incompatible_keys.missing_keys
                if not key.startswith("_base_policy.")
            ]
            unexpected_keys = [
                key for key in incompatible_keys.unexpected_keys
                if not key.startswith("_base_policy.")
            ]
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    "Residual checkpoint load mismatch.\n"
                    f"Missing keys: {missing_keys}\n"
                    f"Unexpected keys: {unexpected_keys}"
                )
        else:
            load_state_dict(
                policy,
                ckpt["state_dict"],
                strict=True
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = policy.to(device)
        policy.eval()
        # instantiate wrapper for policy
        policy_wrapper = instantiate(cfg.policy_wrapper)
        policy_wrapper.policy = policy
        server = WebsocketPolicyServer(
            policy=policy_wrapper,
            host=cfg.get("host", "0.0.0.0"),
            port=cfg.get("port", 8001),
        )
        try:
            server.serve_forever()
        finally:
            if hasattr(policy_wrapper, "close"):
                policy_wrapper.close()


if __name__ == "__main__":
    main()
