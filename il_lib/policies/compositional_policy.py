from typing import Any, Dict, List, Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from il_lib.policies.diffusion_policy import DiffusionPolicy
from il_lib.policies.policy_base import BasePolicy
from il_lib.utils.array_tensor_utils import get_batch_size


class CompositionalPolicy(BasePolicy):
    """
    Product-of-experts sampler over two normal DiffusionPolicy instances.

    At every denoising step this combines the two policies' score/noise
    predictions as:

        eps = (1 - p_intervention) * eps_base + p_intervention * eps_corrector

    For epsilon-prediction DDIM this is the score corresponding to
    base_policy ** (1 - p) * corrector_policy ** p, up to the shared diffusion
    noise scale.
    """

    is_sequence_policy = True

    def __init__(
        self,
        *,
        base_policy: DiffusionPolicy | DictConfig | Dict[str, Any],
        corrector_policy: DiffusionPolicy | DictConfig | Dict[str, Any],
        intervention_probe: torch.nn.Module | DictConfig | Dict[str, Any],
        intervention_prop_keys: Optional[List[str]] = None,
        intervention_probability_key: Optional[str] = None,
        num_denoise_steps_per_inference: Optional[int] = None,
        noise_scheduler_step_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        base_kwargs = {
            key: kwargs.pop(key)
            for key in ("online_eval", "policy_wrapper", "robot_type")
            if key in kwargs
        }
        super().__init__(**base_kwargs)
        self.base_policy = self._instantiate_if_config(base_policy)
        self.corrector_policy = self._instantiate_if_config(corrector_policy)
        self.intervention_probe = self._instantiate_if_config(intervention_probe)

        self._validate_policy_compatibility()
        self._intervention_prop_keys = (
            intervention_prop_keys
            if intervention_prop_keys is not None
            else list(getattr(self.base_policy, "_prop_keys", []))
        )
        self._intervention_probability_key = intervention_probability_key
        self.num_denoise_steps_per_inference = (
            num_denoise_steps_per_inference
            if num_denoise_steps_per_inference is not None
            else self.base_policy.num_denoise_steps_per_inference
        )
        self.noise_scheduler_step_kwargs = (
            noise_scheduler_step_kwargs
            if noise_scheduler_step_kwargs is not None
            else dict(getattr(self.base_policy, "noise_scheduler_step_kwargs", {}))
        )

        self.action_dim = self.base_policy.action_dim
        self.horizon = self.base_policy.horizon
        self.num_latest_obs = self.base_policy.num_latest_obs
        self.deployed_action_steps = self.base_policy.deployed_action_steps
        self.save_hyperparameters(ignore=["base_policy", "corrector_policy", "intervention_probe"])

    def forward(self, obs, noisy_traj, diffusion_timesteps):
        intervention_prob = self._intervention_probability(obs)
        base_pred = self.base_policy.forward(obs, noisy_traj, diffusion_timesteps)
        corrector_pred = self.corrector_policy.forward(obs, noisy_traj, diffusion_timesteps)
        return self._blend_predictions(base_pred, corrector_pred, intervention_prob)

    @torch.no_grad()
    def act(self, obs: dict) -> torch.Tensor:
        obs = self.base_policy.process_data(obs, extract_action=False)
        intervention_prob = self._intervention_probability(obs)
        batch_size = get_batch_size(obs, strict=True)
        noisy_traj = torch.randn(
            size=(batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        scheduler = self.base_policy.noise_scheduler
        scheduler.set_timesteps(self.num_denoise_steps_per_inference)

        for timestep in scheduler.timesteps:
            base_pred = self.base_policy.forward(obs, noisy_traj, timestep)
            corrector_pred = self.corrector_policy.forward(obs, noisy_traj, timestep)
            pred = self._blend_predictions(base_pred, corrector_pred, intervention_prob)
            noisy_traj = scheduler.step(
                pred,
                timestep,
                noisy_traj,
                **self.noise_scheduler_step_kwargs,
            ).prev_sample

        action = noisy_traj[:, self.num_latest_obs - 1 :].clone().cpu()
        return self.base_policy._denormalize_action(action)

    def reset(self) -> None:
        self.base_policy.reset()
        self.corrector_policy.reset()

    def policy_training_step(self, batch, batch_idx):
        raise NotImplementedError("CompositionalPolicy is an inference-only policy.")

    def policy_evaluation_step(self, batch, batch_idx):
        raise NotImplementedError("CompositionalPolicy is an inference-only policy.")

    def configure_optimizers(self):
        raise NotImplementedError("CompositionalPolicy is an inference-only policy.")

    def _instantiate_if_config(self, value):
        if isinstance(value, (DictConfig, dict)):
            return instantiate(value, _recursive_=False)
        return value

    def _validate_policy_compatibility(self) -> None:
        required = ("action_dim", "horizon", "num_latest_obs")
        for attr in required:
            if getattr(self.base_policy, attr) != getattr(self.corrector_policy, attr):
                raise ValueError(
                    f"base_policy.{attr} and corrector_policy.{attr} must match: "
                    f"{getattr(self.base_policy, attr)} != {getattr(self.corrector_policy, attr)}"
                )
        base_prediction_type = getattr(self.base_policy.noise_scheduler.config, "prediction_type", None)
        corrector_prediction_type = getattr(
            self.corrector_policy.noise_scheduler.config,
            "prediction_type",
            None,
        )
        if base_prediction_type != corrector_prediction_type:
            raise ValueError(
                "base_policy and corrector_policy must use the same scheduler prediction_type: "
                f"{base_prediction_type} != {corrector_prediction_type}"
            )

    def _blend_predictions(
        self,
        base_pred: torch.Tensor,
        corrector_pred: torch.Tensor,
        intervention_prob: torch.Tensor,
    ) -> torch.Tensor:
        weight = intervention_prob.to(device=base_pred.device, dtype=base_pred.dtype)
        while weight.dim() < base_pred.dim():
            weight = weight.unsqueeze(-1)
        return (1.0 - weight) * base_pred + weight * corrector_pred

    @torch.no_grad()
    def _intervention_probability(self, obs: dict) -> torch.Tensor:
        if self._intervention_probability_key is not None:
            prob = obs[self._intervention_probability_key]
            return torch.as_tensor(prob, device=self.device, dtype=self.dtype).reshape(-1)

        probe_inputs = self._build_probe_inputs(obs)
        logits_or_prob = self.intervention_probe(probe_inputs)
        prob = logits_or_prob
        if not self._looks_like_probability(prob):
            prob = torch.sigmoid(prob)
        prob = prob.to(device=self.device, dtype=self.dtype).reshape(-1).clamp(0.0, 1.0)
        self._print_intervention_probability(prob)
        return prob

    def _print_intervention_probability(self, prob: torch.Tensor) -> None:
        scores = prob.detach().cpu().flatten().tolist()
        formatted = ", ".join(f"{score:.4f}" for score in scores)
        print(f"[CompositionalPolicy] intervention_score={formatted}", flush=True)

    def _looks_like_probability(self, value: torch.Tensor) -> bool:
        if not torch.is_floating_point(value):
            return False
        detached = value.detach()
        return bool(torch.all((0.0 <= detached) & (detached <= 1.0)))

    def _build_probe_inputs(self, obs: dict) -> dict:
        input_keys = set(getattr(self.intervention_probe, "input_keys", []))
        feature_keys = set(getattr(self.intervention_probe, "_features", []))
        requested_inputs = input_keys | feature_keys
        if not requested_inputs:
            return obs

        inputs = {}
        if "proprioception" in requested_inputs:
            prop_obs = []
            for prop_key in self._intervention_prop_keys:
                if "/" in prop_key:
                    group, key = prop_key.split("/", 1)
                    prop_obs.append(obs[group][key])
                else:
                    prop_obs.append(obs[prop_key])
            inputs["proprioception"] = self._format_probe_sequence_input(
                "proprioception",
                torch.cat(prop_obs, dim=-1),
            )
        if "rgb" in requested_inputs:
            if "rgb" in obs:
                inputs["rgb"] = obs["rgb"]
            else:
                inputs["rgb"] = {
                    key.rsplit("::", 1)[0]: value.float() / 255.0
                    for key, value in obs.items()
                    if key.endswith("::rgb")
                }
        if "rgbd" in requested_inputs and "rgbd" in obs:
            inputs["rgbd"] = obs["rgbd"]
        if "pcd" in requested_inputs and "pcd" in obs:
            inputs["pcd"] = obs["pcd"]
        if "task" in requested_inputs and "task" in obs:
            inputs["task"] = self._format_probe_sequence_input("task", obs["task"])
        for key in ("action_history", "base_action_chunk"):
            if key in requested_inputs and key in obs:
                inputs[key] = self._format_probe_sequence_input(key, obs[key])
        if "base_action_chunk" in requested_inputs and "base_action_chunk" not in inputs:
            inputs["base_action_chunk"] = self._sample_base_action_chunk_for_probe(obs)
        return inputs

    def _format_probe_sequence_input(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if value.dim() == 2:
            value = value.unsqueeze(1)

        steps = self._probe_input_steps(key, default=value.shape[1])
        if value.shape[1] == steps:
            return value
        if value.shape[1] > steps:
            return value[:, :steps]

        pad = value[:, -1:].expand(-1, steps - value.shape[1], *value.shape[2:])
        return torch.cat([value, pad], dim=1)

    def _sample_base_action_chunk_for_probe(self, obs: dict) -> torch.Tensor:
        batch_size = get_batch_size(obs, strict=True)
        noisy_traj = torch.randn(
            size=(batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        scheduler = self.base_policy.noise_scheduler
        scheduler.set_timesteps(self.num_denoise_steps_per_inference)

        for timestep in scheduler.timesteps:
            pred = self.base_policy.forward(obs, noisy_traj, timestep)
            noisy_traj = scheduler.step(
                pred,
                timestep,
                noisy_traj,
                **self.noise_scheduler_step_kwargs,
            ).prev_sample

        chunk = noisy_traj[:, self.num_latest_obs - 1 :]
        steps = self._probe_input_steps("base_action_chunk", default=chunk.shape[1])
        if chunk.shape[1] >= steps:
            return chunk[:, :steps].detach()

        pad = chunk[:, -1:].expand(-1, steps - chunk.shape[1], -1)
        return torch.cat([chunk, pad], dim=1).detach()

    def _probe_input_steps(self, key: str, default: int) -> int:
        input_configs = getattr(self.intervention_probe, "input_configs", {})
        if key in input_configs and "steps" in input_configs[key]:
            return int(input_configs[key]["steps"])

        input_steps = getattr(self.intervention_probe, "input_steps", {})
        if key in input_steps:
            return int(input_steps[key])

        return int(default)
