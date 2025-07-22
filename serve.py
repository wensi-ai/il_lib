
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from il_lib.utils.training_utils import load_state_dict, load_torch
from omnigibson.learning.utils.network_utils import WebsocketPolicyServer


@hydra.main(config_name="base_config", config_path="il_lib/configs", version_base="1.1")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    policy = instantiate(cfg.module, _recursive_=False)
    ckpt = load_torch(
        "/home/svl/Research/il_lib/outputs/2025-07-20/18-40-26/diffusion_rgb_transformer_turning_on_radio_20250720-184026/ckpt/last.pth",
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
