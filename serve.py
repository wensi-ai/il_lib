
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from il_lib.utils.training_utils import load_state_dict, load_torch
from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
from il_lib.utils.config_utils import register_omegaconf_resolvers


@hydra.main(config_name="base_config", config_path="il_lib/configs", version_base="1.1")
def main(cfg):
    register_omegaconf_resolvers()
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    policy = instantiate(cfg.module, _recursive_=False)
    ckpt = load_torch(
        "/home/svl/Documents/Files/behavior/ckpt/wbvima_radio/last_2.pth",
        # "/home/svl/Research/il_lib/outputs/2025-08-01/20-54-51/diffusion_rgb_unet_PickPlaceTask_20250801-205451/ckpt/last.pth",
        map_location="cpu",
    )
    load_state_dict(
        policy,
        ckpt["state_dict"],
        strict=True
    )
    policy = policy.to("cuda")
    policy.eval()
    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=8000,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
