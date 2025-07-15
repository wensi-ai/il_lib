import os
from typing import List, Optional, Union

import torch.nn as nn
from torchvision.models import ResNet18_Weights
from torchvision import transforms
import il_lib.utils as U
from il_lib.utils.array_tensor_utils import any_concat
from il_lib.utils.training_utils import load_torch, load_state_dict
from il_lib.nn.features.resnet import resnet18
from einops import rearrange
from il_lib.optim import default_optimizer_groups


class MultiviewResNet18(nn.Module):
    def __init__(
        self,
        views: List[str],
        *,
        resnet_output_dim: int,
        token_dim: int,
        load_pretrained: bool = True,
        pretrained_ckpt_path: Optional[str] = None,
        enable_random_crop: bool = True,
        random_crop_size: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__()
        self._views = views
        self._resnet = resnet18(output_dim=resnet_output_dim)
        if load_pretrained:
            assert pretrained_ckpt_path is not None and os.path.exists(
                pretrained_ckpt_path
            )
            ckpt = load_torch(pretrained_ckpt_path, map_location="cpu")
            del ckpt["fc.weight"]
            del ckpt["fc.bias"]
            load_state_dict(self._resnet, ckpt, strict=False)
        self._output_fc = nn.Linear(len(views) * resnet_output_dim, token_dim)
        self.output_dim = token_dim

        train_transforms, eval_transforms = [], []
        if enable_random_crop:
            train_transforms.append(transforms.RandomCrop(random_crop_size))
            eval_transforms.append(transforms.CenterCrop(random_crop_size))
        train_transforms.append(ResNet18_Weights.DEFAULT.transforms())
        eval_transforms.append(ResNet18_Weights.DEFAULT.transforms())
        self._train_transforms = transforms.Compose(train_transforms)
        self._eval_transforms = transforms.Compose(eval_transforms)

    def forward(self, x):
        """
        x: a dict with keys in self._views and values of shape (B, L, C, H, W)
        """
        assert set(x.keys()) == set(self._views)
        B, L = x[self._views[0]].shape[:2]
        x = {
            k: rearrange(v, "B L C H W -> (B L) C H W").contiguous()
            for k, v in x.items()
        }
        x = {
            k: self._train_transforms(v) if self.training else self._eval_transforms(v)
            for k, v in x.items()
        }
        resnet_output = {
            k: self._resnet(v) for k, v in x.items()
        }  # dict of (B * L, resnet_output_dim)
        multiview_output = any_concat(
            [resnet_output[k] for k in self._views],
            dim=-1,
        )  # (B * L, len(views) * resnet_output_dim)
        flattened_output = self._output_fc(multiview_output)  # (B * L, token_dim)
        output = rearrange(flattened_output, "(B L) E -> B L E", B=B, L=L).contiguous()
        return output

    def get_optimizer_groups(self, weight_decay, lr_layer_decay, lr_scale=1.0):
        pg, pids = default_optimizer_groups(
            self,
            weight_decay=weight_decay,
            lr_scale=lr_scale,
        )
        return pg, pids