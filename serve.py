import hydra
import torch
import time
import traceback
import websockets
from copy import deepcopy
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from il_lib.policies import ResidualPolicy
from il_lib.utils.config_utils import register_omegaconf_resolvers
from il_lib.utils.training_utils import load_state_dict, load_torch
from omegaconf import OmegaConf
from omnigibson.learning.utils.network_utils import Packer, WebsocketPolicyServer, unpackb
import os
import sys


DEFAULT_BASE_POLICY_PORT = 8002

def _to_msgpackable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: _to_msgpackable(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_to_msgpackable(v) for v in value]
    return value


class PolicyStateWebsocketPolicyServer(WebsocketPolicyServer):
    async def _handler(self, websocket):
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                result = unpackb(await websocket.recv(), strict_map_key=False)
                if "reset" in result:
                    self._policy.reset()
                    continue

                obs = deepcopy(result)
                infer_time = time.monotonic()
                action = self._policy.act(obs)
                infer_time = time.monotonic() - infer_time

                response = {"action": action.cpu().numpy()}
                if hasattr(self._policy, "get_current_state"):
                    policy_state = self._policy.get_current_state()
                    if policy_state is not None:
                        response["policy_state"] = _to_msgpackable(policy_state)
                response["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(response))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                break
            except Exception:
                error = traceback.format_exc()
                print(error, file=sys.stderr, flush=True)
                await websocket.send(error)


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
        server = PolicyStateWebsocketPolicyServer(
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
