from __future__ import annotations
from typing import Literal, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from il_lib.utils.shape_utils import check_shape
from il_lib.utils.training_utils import classify_accuracy
from il_lib.nn.common import build_mlp


__all__ = [
    "Categorical",
    "CategoricalHead",
    "CategoricalNet",
    "MultiCategorical",
    "MultiCategoricalHead",
    "MultiCategoricalNet",
    "DiagonalGaussian",
    "DiagonalGaussianHead",
    "DiagonalGaussianNet",
    "SquashedGaussian",
    "SquashedGaussianHead",
    "SquashedGaussianNet",
    "MixtureOfGaussian",
    "GMMHead",
]


# ==================== SAC ====================
class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1, eps: float = 0):
        super().__init__(cache_size=cache_size)
        self._eps = eps  # to avoid NaN at inverse (atanh)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        if self._eps:
            y = y.clamp(-1 + self._eps, 1 - self._eps)
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedGaussian(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale, atanh_eps: Optional[float] = None):
        """
        Args:
            atanh_eps: clip atanh(action between [-1+eps, 1-eps]). If the action is
                exactly -1 or exactly 1, its log_prob will be inf/NaN
        """
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform(eps=atanh_eps)]
        super().__init__(self.base_dist, transforms)

    def log_prob(self, actions):
        # We assume independent action dims, so the probs are additive
        # clip the actions to avoid NaN
        return super().log_prob(actions).sum(-1)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def mode(self):
        return self.mean

    def entropy(self):
        raise NotImplementedError(
            "no analytical form, entropy must be estimated from -log_prob.mean()"
        )


class SquashedGaussianHead(nn.Module):
    def __init__(
        self,
        process_log_std: Literal["expln", "scale", "clip", "none", None] = "expln",
        log_std_bounds: tuple[float, float] = (-10, 2),
        atanh_eps: float = 1e-6,
    ):
        """
        Output dim should be action_dim*2, because it will be chunked into (mean, log_std)

        Args:
          process_log_std: different methods to process raw log_std value from NN output
              before sending to SquashedNormal
            - "expln": trick introduced by "State-Dependent Exploration for Policy
               Gradient Methods", Schmidhuber group: https://people.idsia.ch/~juergen//ecml2008rueckstiess.pdf
               also appears in stable-baselines-3:
               https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/distributions.py
               This trick works out of box with PPO and doesn't need other hypers
            - "scale": first apply tanh() to log_std, and then scale to `log_std_bounds`
               used in some SAC implementations: https://github.com/denisyarats/drq
               WARNING: you may see NaN with this mode
            - "clip": simply clips log_std to `log_std_bounds`, used in the original SAC
               WARNING: you may see NaN with this mode
            - "none"/None: do nothing and directly pass log_std.exp() to SquashedNormal
               WARNING: you may see NaN with this mode
          atanh_eps: clip actions between [-1+eps, 1-eps]. If the action is
              exactly -1 or exactly 1, its log_prob will be inf/NaN
        """
        super().__init__()

        assert process_log_std in ["expln", "scale", "clip", "none", None]
        self._process_log_std = process_log_std
        self._log_std_bounds = log_std_bounds
        self._atanh_eps = atanh_eps

    def forward(self, x: torch.Tensor) -> SquashedGaussian:
        mean, log_std = x.chunk(2, dim=-1)
        # Used in SAC: rescale log_std inside [log_std_min, log_std_max]
        # WARNING: does not work for PPO somehow
        if self._process_log_std == "exp_ln":
            below_threshold = log_std.exp() * (log_std <= 0)
            safe_log_std = log_std * (log_std > 0) + 1e-6
            above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        elif self._process_log_std == "scale":
            log_std = torch.tanh(log_std)
            log_std_min, log_std_max = self._log_std_bounds
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            std = log_std.exp()
        elif self._process_log_std == "clip":
            log_std = log_std.clip(*self._log_std_bounds)
            std = log_std.exp()
        else:
            std = log_std.exp()

        return SquashedGaussian(loc=mean, scale=std, atanh_eps=self._atanh_eps)


def _build_mlp_distribution_net(
    input_dim: int,
    *,
    output_dim: int,
    hidden_dim: int,
    hidden_depth: int,
    activation: str | Callable = "relu",
    norm_type: Literal["batchnorm", "layernorm"] | None = None,
    last_layer_gain: float | None = 0.01,
):
    """
    Use orthogonal initialization to initialize the MLP policy

    Args:
        last_layer_gain: orthogonal initialization gain for the last FC layer.
            you may want to set it to a small value (e.g. 0.01) to have the
            Gaussian centered around 0.0 in the beginning.
            Set to None to use the default gain (dependent on the NN activation)
    """

    mlp = build_mlp(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        activation=activation,
        weight_init="orthogonal",
        bias_init="zeros",
        norm_type=norm_type,
    )
    if last_layer_gain:
        assert last_layer_gain > 0
        nn.init.orthogonal_(mlp[-1].weight, gain=last_layer_gain)
    return mlp


class SquashedGaussianNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        action_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        last_layer_gain: float | None = 0.01,
        process_log_std: Literal["expln", "scale", "clip", "none", None] = "expln",
        log_std_bounds: tuple[float, float] = (-10, 2),
        atanh_eps: float = 1e-6,
    ):
        """
        Use orthogonal initialization to initialize the MLP policy

        Args:
            last_layer_gain: orthogonal initialization gain for the last FC layer.
                you may want to set it to a small value (e.g. 0.01) to have the
                Gaussian centered around 0.0 in the beginning.
                Set to None to use the default gain (dependent on the NN activation)
        """
        super().__init__()

        self.mlp = _build_mlp_distribution_net(
            input_dim=input_dim,
            output_dim=action_dim * 2,  # mean and log_std
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
            last_layer_gain=last_layer_gain,
        )
        self.head = SquashedGaussianHead(
            process_log_std=process_log_std,
            log_std_bounds=log_std_bounds,
            atanh_eps=atanh_eps,
        )

    def forward(self, x):
        return self.head(self.mlp(x))


# ==================== PPO ====================
# ---------------- Categorical -----------------


class Categorical(torch.distributions.Categorical):
    """
    Mostly interface changes, add mode() function, no real difference from Categorical
    """

    def mode(self):
        return self.logits.argmax(dim=-1)

    def imitation_loss(self, actions, reduction="mean"):
        """
        actions: groundtruth actions from expert
        """
        assert actions.dtype == torch.long
        if self.logits.ndim == 3:
            assert actions.ndim == 2
            assert self.logits.shape[:2] == actions.shape
            return F.cross_entropy(
                self.logits.reshape(-1, self.logits.shape[-1]),
                actions.reshape(-1),
                reduction=reduction,
            )
        return F.cross_entropy(self.logits, actions, reduction=reduction)

    def imitation_accuracy(self, actions, mask=None, reduction="mean", scale_100=False):
        if self.logits.ndim == 3:
            assert actions.ndim == 2
            assert self.logits.shape[:2] == actions.shape
            if mask is not None:
                assert mask.ndim == 2
                assert self.logits.shape[:2] == mask.shape
            actions = actions.reshape(-1)
            if mask is not None:
                mask = mask.reshape(-1)
            return classify_accuracy(
                self.logits.reshape(-1, self.logits.shape[-1]),
                actions,
                mask=mask,
                reduction=reduction,
                scale_100=scale_100,
            )
        return classify_accuracy(
            self.logits, actions, mask=mask, reduction=reduction, scale_100=scale_100
        )

    def random_actions(self):
        """
        Generate a completely random action, NOT the same as sample(), more like
        action_space.sample()
        """
        return torch.randint(
            low=0,
            high=self.logits.size(-1),
            size=self.logits.size()[:-1],
            device=self.logits.device,
        )


class CategoricalHead(nn.Module):
    def forward(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=x)


class CategoricalNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        action_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        last_layer_gain: float | None = 0.01,
    ):
        """
        Use orthogonal initialization to initialize the MLP policy

        Args:
            last_layer_gain: orthogonal initialization gain for the last FC layer.
                you may want to set it to a small value (e.g. 0.01) to make the
                Categorical close to uniform random at the beginning.
                Set to None to use the default gain (dependent on the NN activation)
        """
        super().__init__()
        self.mlp = _build_mlp_distribution_net(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
            last_layer_gain=last_layer_gain,
        )
        self.head = CategoricalHead()

    def forward(self, x):
        return self.head(self.mlp(x))


# ---------------- MultiCategorical -----------------
class MultiCategorical(torch.distributions.Distribution):
    def __init__(self, logits, action_dims: list[int]):
        assert logits.dim() == 2, logits.shape
        super().__init__(batch_shape=logits[:1], validate_args=False)
        self._action_dims = tuple(action_dims)
        assert logits.size(1) == sum(
            self._action_dims
        ), f"sum of action dims {self._action_dims} != {logits.size(1)}"
        self._dists = [
            Categorical(logits=split)
            for split in torch.split(logits, action_dims, dim=1)
        ]

    def log_prob(self, actions):
        return torch.stack(
            [
                dist.log_prob(action)
                for dist, action in zip(self._dists, torch.unbind(actions, dim=1))
            ],
            dim=1,
        ).sum(dim=1)

    def entropy(self):
        return torch.stack([dist.entropy() for dist in self._dists], dim=1).sum(dim=1)

    def sample(self, sample_shape=torch.Size()):
        assert sample_shape == torch.Size()
        return torch.stack([dist.sample() for dist in self._dists], dim=1)

    def mode(self):
        return torch.stack(
            [torch.argmax(dist.probs, dim=1) for dist in self._dists], dim=1
        )

    @property
    def mean(self) -> torch.Tensor:
        return self.mode()

    def imitation_loss(self, actions, weights=None, reduction="mean"):
        """
        Args:
            actions: groundtruth actions from expert
            weights: weight the imitation loss from each component in MultiDiscrete
            reduction: "mean" or "none"

        Returns:
            one torch float
        """
        assert actions.dtype == torch.long
        check_shape(actions, [None, len(self._action_dims)])
        assert reduction in ["mean", "none"]
        if weights is None:
            weights = [1.0] * len(self._dists)
        else:
            assert len(weights) == len(self._dists)

        aggregate = sum if reduction == "mean" else list
        return aggregate(
            dist.imitation_loss(a, reduction=reduction) * w
            for dist, a, w in zip(self._dists, torch.unbind(actions, dim=1), weights)
        )

    def imitation_accuracy(self, actions, mask=None, reduction="mean", scale_100=False):
        """
        Returns:
            a 1D tensor of accuracies between 0 and 1 as float
        """
        return [
            dist.imitation_accuracy(
                a, mask=mask, reduction=reduction, scale_100=scale_100
            )
            for dist, a in zip(self._dists, torch.unbind(actions, dim=1))
        ]

    def random_actions(self):
        return torch.stack([dist.random_actions() for dist in self._dists], dim=-1)


class MultiCategoricalHead(nn.Module):
    def __init__(self, action_dims: list[int]):
        super().__init__()
        self._action_dims = tuple(action_dims)

    def forward(self, x: torch.Tensor) -> MultiCategorical:
        return MultiCategorical(logits=x, action_dims=self._action_dims)

    def extra_repr(self) -> str:
        return f"action_dims={list(self._action_dims)}"


class MultiCategoricalNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        action_dims: list[int],
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        last_layer_gain: float | None = 0.01,
    ):
        """
        Use orthogonal initialization to initialize the MLP policy
        Split head, does not share the NN weights

        Args:
            last_layer_gain: orthogonal initialization gain for the last FC layer.
                you may want to set it to a small value (e.g. 0.01) to make the
                Categorical close to uniform random at the beginning.
                Set to None to use the default gain (dependent on the NN activation)
        """
        super().__init__()
        self.mlps = nn.ModuleList()
        for action in action_dims:
            net = _build_mlp_distribution_net(
                input_dim=input_dim,
                output_dim=action,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
                activation=activation,
                norm_type=norm_type,
                last_layer_gain=last_layer_gain,
            )
            self.mlps.append(net)
        self.head = MultiCategoricalHead(action_dims)

    def forward(self, x):
        return self.head(torch.cat([mlp(x) for mlp in self.mlps], dim=1))


# ---------------- Gaussian -----------------
class DiagonalGaussian(torch.distributions.Normal):
    def log_prob(self, actions):
        # We assume independent action dims, so the probs are additive
        return super().log_prob(actions).sum(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class DiagonalGaussianHead(nn.Module):
    def __init__(self, action_dim: int, initial_log_std: float = -1.0):
        super().__init__()
        self._action_dim = action_dim
        self.log_std = torch.nn.Parameter(
            torch.ones((action_dim,)) * initial_log_std, requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> DiagonalGaussian:
        return DiagonalGaussian(loc=x, scale=self.log_std.exp())

    def extra_repr(self) -> str:
        return f"action_dim={self._action_dim}"


class DiagonalGaussianNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        action_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        last_layer_gain: float | None = 0.01,
        initial_log_std: float = 0.0,
    ):
        """
        Use orthogonal initialization to initialize the MLP policy

        Args:
            last_layer_gain: orthogonal initialization gain for the last FC layer.
                you may want to set it to a small value (e.g. 0.01) to make the
                Categorical close to uniform random at the beginning.
                Set to None to use the default gain (dependent on the NN activation)
        """
        super().__init__()
        self.mlp = _build_mlp_distribution_net(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
            last_layer_gain=last_layer_gain,
        )
        self.head = DiagonalGaussianHead(action_dim, initial_log_std=initial_log_std)

    def forward(self, x):
        return self.head(self.mlp(x))


# ---------------- GMM -----------------
class MixtureOfGaussian:
    def __init__(
        self,
        logits: torch.Tensor,
        means: torch.Tensor,
        scales: torch.Tensor,
        min_std: float = 0.0001,
        low_noise_eval: bool = True,
    ):
        """
        logits: (..., n_modes)
        means: (..., n_modes, dim)
        scales: (..., n_modes, dim)
        """
        assert logits.dim() + 1 == means.dim() == scales.dim()
        assert logits.shape[-1] == means.shape[-2] == scales.shape[-2]
        assert means.shape == scales.shape

        self._logits = logits
        self._means = torch.tanh(means)
        self._scales = scales
        self._min_std = min_std
        self._low_noise_eval = low_noise_eval

    def mode(self):
        # assume mode will only be called during eval
        if self._low_noise_eval:
            scales = torch.ones_like(self._means) * 1e-4
            component_distribution = torch.distributions.Normal(
                loc=self._means, scale=scales
            )
            component_distribution = torch.distributions.Independent(
                component_distribution, 1
            )
            dist = torch.distributions.MixtureSameFamily(
                mixture_distribution=torch.distributions.Categorical(
                    logits=self._logits
                ),
                component_distribution=component_distribution,
            )
            return dist.sample()
        else:
            # return the mean of the most probable component
            one_hot = F.one_hot(
                self._logits.argmax(dim=-1), self._logits.shape[-1]
            ).unsqueeze(-1)
            return (self._means * one_hot).sum(dim=-2)

    def imitation_loss(self, actions, reduction="mean"):
        """
        NLL loss
        actions: (..., dim)
        """

        batch_dims = self._logits.shape[:-1]
        logits = self._logits.reshape(-1, self._logits.shape[-1])
        means = self._means.reshape(-1, *self._means.shape[-2:])
        scales = self._scales.reshape(-1, *self._scales.shape[-2:])

        scales = F.softplus(scales) + self._min_std
        component_distribution = torch.distributions.Normal(loc=means, scale=scales)
        component_distribution = torch.distributions.Independent(
            component_distribution, 1
        )
        dist = torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(logits=logits),
            component_distribution=component_distribution,
        )
        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dist.batch_shape) == 1

        assert actions.shape[:-1] == batch_dims
        actions = actions.reshape(-1, actions.shape[-1])
        log_probs = dist.log_prob(actions)  # (...,), note that action dim is summed
        log_probs = log_probs.reshape(*batch_dims)
        if reduction == "mean":
            return -log_probs.mean()
        elif reduction == "sum":
            return -log_probs.sum()
        elif reduction == "none":
            return -log_probs

    def imitation_accuracy(self, actions, mask=None, reduction="mean"):
        """
        L1 distance between mode and actions

        actions: (..., dim)
        mask: (...,)
        """
        if mask is not None:
            assert mask.shape == actions.shape[:-1]

        scales = torch.ones_like(self._means) * 1e-4
        component_distribution = torch.distributions.Normal(
            loc=self._means, scale=scales
        )
        component_distribution = torch.distributions.Independent(
            component_distribution, 1
        )
        dist = torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(logits=self._logits),
            component_distribution=component_distribution,
        )
        mode = dist.mean  # (..., dim)
        loss = (actions - mode).abs().sum(-1)  # (...)
        # we want accuracy, higher is better, so we negate the loss
        loss = -loss
        if mask is not None:
            loss *= mask
        if reduction == "mean":
            if mask is not None:
                return loss.sum() / mask.sum()
            else:
                return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        elif reduction == "none":
            return loss


class GMMHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        n_modes: int = 5,
        action_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        mean_mlp_last_layer_gain: float | None = 0.01,
        low_noise_eval: bool = True,
    ):
        super().__init__()
        self._logits_mlp = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=n_modes,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
        )
        self._mean_mlp = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=n_modes * action_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
        )
        if mean_mlp_last_layer_gain is not None:
            assert mean_mlp_last_layer_gain > 0
            nn.init.orthogonal_(
                self._mean_mlp[-1].weight, gain=mean_mlp_last_layer_gain
            )
        self._scale_mlp = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=n_modes * action_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
        )
        self._n_modes = n_modes
        self._action_dim = action_dim

        self._low_noise_eval = low_noise_eval

    def forward(self, x: torch.Tensor) -> MixtureOfGaussian:
        logits = self._logits_mlp(x)
        mean = self._mean_mlp(x)  # (..., n_modes * action_dim)
        scale = self._scale_mlp(x)  # (..., n_modes * action_dim)
        mean = mean.reshape(*mean.shape[:-1], self._n_modes, self._action_dim)
        scale = scale.reshape(*scale.shape[:-1], self._n_modes, self._action_dim)

        assert logits.shape[-1] == self._n_modes
        assert mean.shape[-2:] == (self._n_modes, self._action_dim)
        assert scale.shape[-2:] == (self._n_modes, self._action_dim)
        return MixtureOfGaussian(
            logits, mean, scale, low_noise_eval=self._low_noise_eval
        )

    @property
    def action_dim(self):
        return self._action_dim