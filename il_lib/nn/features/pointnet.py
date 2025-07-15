import torch
import torch.nn as nn
from il_lib.nn.common import build_mlp
from il_lib.optim import default_optimizer_groups
from il_lib.utils.convert_utils import any_to_torch_tensor


class PointNetSimplified(nn.Module):
    def __init__(
        self,
        *,
        point_channels: int = 3,
        output_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "gelu",
    ):
        super().__init__()
        self._mlp = build_mlp(
            input_dim=point_channels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
            activation=activation,
        )
        self.output_dim = output_dim

    def forward(self, x):
        """
        x: (..., points, point_channels)
        """
        x = any_to_torch_tensor(x)
        x = self._mlp(x)  # (..., points, output_dim)
        x = torch.max(x, dim=-2)[0]  # (..., output_dim)
        return x


class PointNet(nn.Module):
    def __init__(
        self,
        *,
        n_coordinates: int = 3,
        n_color: int = 3,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_depth: int = 2,
        activation: str = "gelu",
        subtract_mean: bool = False,
    ):
        super().__init__()
        pn_in_channels = n_coordinates + n_color
        if subtract_mean:
            pn_in_channels += n_coordinates
        self.pointnet = PointNetSimplified(
            point_channels=pn_in_channels,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
        )
        self.subtract_mean = subtract_mean
        self.output_dim = self.pointnet.output_dim

    def forward(self, x):
        """
        x["xyz"]: (..., points, coordinates)
        x["rgb"]: (..., points, color)
        """
        xyz = x["xyz"]
        rgb = x["rgb"]
        point = any_to_torch_tensor(xyz)
        if self.subtract_mean:
            mean = torch.mean(point, dim=-2, keepdim=True)  # (..., 1, coordinates)
            mean = torch.broadcast_to(mean, point.shape)  # (..., points, coordinates)
            point = point - mean
            point = torch.cat([point, mean], dim=-1)  # (..., points, 2 * coordinates)
        rgb = any_to_torch_tensor(rgb)
        x = torch.cat([point, rgb], dim=-1)
        return self.pointnet(x)

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        pg, pids = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=["ee_embd_layer.*"],
        )
        return pg, pids


class UncoloredPointNet(nn.Module):
    def __init__(
        self,
        *,
        n_coordinates: int = 3,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_depth: int = 2,
        activation: str = "gelu",
        subtract_mean: bool = False,
    ):
        super().__init__()
        pn_in_channels = n_coordinates
        if subtract_mean:
            pn_in_channels += n_coordinates
        self.pointnet = PointNetSimplified(
            point_channels=pn_in_channels,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
        )
        self.subtract_mean = subtract_mean
        self.output_dim = self.pointnet.output_dim

    def forward(self, x):
        """
        x["xyz"]: (..., points, coordinates)
        """
        xyz = x["xyz"]
        point = any_to_torch_tensor(xyz)
        if self.subtract_mean:
            mean = torch.mean(point, dim=-2, keepdim=True)  # (..., 1, coordinates)
            mean = torch.broadcast_to(mean, point.shape)  # (..., points, coordinates)
            point = point - mean
            point = torch.cat([point, mean], dim=-1)  # (..., points, 2 * coordinates)
        return self.pointnet(point)

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        pg, pids = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
            no_decay_filter=["ee_embd_layer.*"],
        )
        return pg, pids