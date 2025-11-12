import il_lib.nn.features.resnet as resnet_lib
import torch
import torch.nn as nn
from einops import rearrange
from functools import partial
from il_lib.utils.array_tensor_utils import any_concat
from il_lib.utils.training_utils import load_state_dict
from il_lib.optim import default_optimizer_groups
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from typing import List, Optional, Union


class MultiviewResNet18(nn.Module):
    def __init__(
        self,
        backbone: str,
        views: List[str],
        *,
        resnet_output_dim: int,
        token_dim: Optional[int] = None,
        use_shared_backbone: bool = True,
        load_pretrained: bool = True,
        include_depth: bool = False,
        enable_random_crop: bool = True,
        random_crop_size: Optional[Union[int, List[int]]] = None,
        return_last_spatial_map: bool = False
    ):
        super().__init__()
        self._views = views
        self._use_shared_backbone = use_shared_backbone
        if use_shared_backbone:
            self._resnet = getattr(resnet_lib, backbone)(output_dim=resnet_output_dim, return_last_spatial_map=return_last_spatial_map)
        else:
            self._resnet = nn.ModuleDict({
                view: getattr(resnet_lib, backbone)(output_dim=resnet_output_dim, return_last_spatial_map=return_last_spatial_map)
                for view in views
            })
        if load_pretrained:
            ckpt = torch.hub.load_state_dict_from_url(url=ResNet18_Weights.DEFAULT.url, map_location="cpu")
            del ckpt["fc.weight"]
            del ckpt["fc.bias"]
            resnet_to_load = [self._resnet] if use_shared_backbone else list(self._resnet.values())
            for resnet in resnet_to_load:
                load_state_dict(resnet, ckpt, strict=False)
                if include_depth:
                    rgbd_conv1 = nn.Conv2d(
                        4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                    )
                    rgbd_conv1.weight.data[:, :3, :, :] = resnet.conv1.weight.data
                    rgbd_conv1.weight.data[:, 3, :, :] = 0.0
                    resnet.conv1 = rgbd_conv1
        self.return_last_spatial_map = return_last_spatial_map
        if not return_last_spatial_map:
            assert token_dim is not None, "token_dim must be specified when return_last_spatial_map is False!"
            self._output_fc = nn.Linear(len(views) * resnet_output_dim, token_dim)
            self.output_dim = token_dim

        train_transforms, eval_transforms = [], []
        if enable_random_crop:
            train_transforms.append(transforms.RandomCrop(random_crop_size))
            eval_transforms.append(transforms.CenterCrop(random_crop_size))
        if include_depth:
            # We do not normalize depth
            train_transforms.append(
                partial(ResNet18_Weights.DEFAULT.transforms, mean=(0.485, 0.456, 0.406, 0.0), std=(0.229, 0.224, 0.225, 1.0))()
            )
            eval_transforms.append(
                partial(ResNet18_Weights.DEFAULT.transforms, mean=(0.485, 0.456, 0.406, 0.0), std=(0.229, 0.224, 0.225, 1.0))()
            )
        else:
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
        if self._use_shared_backbone:
            resnet_output = {
                k: self._resnet(v) for k, v in x.items()
            }  # dict of (B * L, **resnet_output_dim)
        else:
            resnet_output = {
                k: self._resnet[k](v) for k, v in x.items()
            }  # dict of (B * L, **resnet_output_dim)
        if self.return_last_spatial_map:
            return resnet_output
        else:
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
