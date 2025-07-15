import torch
import torch.nn as nn
import torchvision.models as _models
from il_lib.nn.common import get_activation
from il_lib.nn.common import Conv2dWS
from il_lib.utils.config_utils import register_class
from typing import Callable


__all__ = ["create_resnet", "get_resnet_class", "get_all_resnet_names"]


def get_resnet_class(model_name):
    import il_lib.nn.features.resnet as _res

    if hasattr(_res, model_name):
        return getattr(_res, model_name)
    elif hasattr(_models, model_name):
        return getattr(_models, model_name)
    else:
        raise NotImplementedError(f"Unknown model: {model_name}")


def create_resnet(model_name, **kwargs):
    return get_resnet_class(model_name)(**kwargs)


def get_all_resnet_names():
    import il_lib.nn.features.resnet as _res

    return [a for a in _res.__dict__ if a.startswith("res")]


def _conv_op(ws) -> Callable:
    return Conv2dWS if ws else nn.Conv2d


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, ws=False):
    """
    3x3 convolution with padding
    ws: weight standardization
    """
    return _conv_op(ws)(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1, ws=False):
    """1x1 convolution"""
    return _conv_op(ws)(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        nonlinearity="relu",
        ws=False,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, ws=ws)
        self.bn1 = norm_layer(planes)
        self.relu = get_activation(nonlinearity)()
        self.conv2 = conv3x3(planes, planes, ws=ws)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class LightBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        nonlinearity="relu",
        ws=False,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, ws=ws)
        self.bn1 = norm_layer(planes)
        self.relu = get_activation(nonlinearity)()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        nonlinearity="relu",
        ws=False,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, ws=ws)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, ws=ws)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, ws=ws)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = get_activation(nonlinearity)()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        output_dim=1000,
        base_width=64,
        zero_init_residual=True,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        nonlinearity="relu",
        ws=False,
        return_last_spatial_map=False,
    ):
        """
        Args:
            zero_init_residual: WARNING: default is True for our implementation, but
                False for torchvision's ResNet.
            return_last_spatial_map: if True, returns the last conv map [512, f_H, f_W]
                if False, returns the features after the final FC layer
        """
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.output_dim = output_dim
        self.inplanes = base_width
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.width_per_group = width_per_group
        self.ws = ws
        self.conv1 = _conv_op(ws)(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.nonlinearity = nonlinearity
        self.relu = get_activation(nonlinearity)()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, base_width, layers[0])
        self.layer2 = self._make_layer(
            block,
            base_width * 2,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            base_width * 4,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            base_width * 8,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        if return_last_spatial_map:
            self.avgpool, self.fc = nn.Identity(), nn.Identity()
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(base_width * 8 * block.expansion, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, LightBasicBlock):
                    nn.init.constant_(m.bn1.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, ws=self.ws),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.width_per_group,
                previous_dilation,
                norm_layer=norm_layer,
                nonlinearity=self.nonlinearity,
                ws=self.ws,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.width_per_group,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    nonlinearity=self.nonlinearity,
                    ws=self.ws,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        DEBUG = False
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if DEBUG:
            print(x.size())
        x = self.layer2(x)
        if DEBUG:
            print(x.size())
        x = self.layer3(x)
        if DEBUG:
            print(x.size())
        x = self.layer4(x)

        if isinstance(self.avgpool, nn.Identity):
            return x

        if DEBUG:
            print(x.size())
        x = self.avgpool(x)
        if DEBUG:
            print(x.size())
        x = torch.flatten(x, 1)
        if DEBUG:
            print(x.size())
        x = self.fc(x)

        return x


# work around for pytorch DDP, cannot pickle lambda
class GroupNorm32(nn.GroupNorm):
    """
    num_groups=32 is the default that Kaiming used for regular ResNets in GN paper
    """

    def __init__(self, num_channels):
        super().__init__(num_groups=32, num_channels=num_channels, affine=True)


class GroupNorm16(nn.GroupNorm):
    def __init__(self, num_channels):
        super().__init__(num_groups=16, num_channels=num_channels, affine=True)


def _resnet_basic_gn(
    num_layers,
    base_width,
    nonlinearity="swish",
    *,
    output_dim,
    ws=False,
    block=BasicBlock,
    **kwargs,
):
    """
    Args:
        base_width: 64 is the regular ResNet family
    """
    if base_width in [64, 128]:
        gn_layer = GroupNorm32
    elif base_width in [16, 32]:
        gn_layer = GroupNorm16
    else:
        raise NotImplementedError("only support base_width 16, 32, 64, 128")

    if num_layers == 9:
        layers = [1, 1, 1, 1]
    elif num_layers == 18:
        layers = [2, 2, 2, 2]
    else:
        raise NotImplementedError
    return ResNet(
        block,
        layers,
        output_dim=output_dim,
        base_width=base_width,
        norm_layer=gn_layer,
        nonlinearity=nonlinearity,
        ws=ws,
        **kwargs,
    )


@register_class
def resnet9w32(output_dim, **kwargs):
    return ResNet(
        BasicBlock,
        [1, 1, 1, 1],
        output_dim=output_dim,
        base_width=32,
        nonlinearity="relu",
        ws=False,
        **kwargs,
    )


@register_class
def resnet18(output_dim, **kwargs):
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        output_dim=output_dim,
        base_width=64,
        nonlinearity="relu",
        ws=False,
        **kwargs,
    )


@register_class
def resnet9_gn(output_dim, **kwargs):
    return _resnet_basic_gn(9, 64, "relu", output_dim=output_dim, **kwargs)


@register_class
def resnet9_gn_ws(output_dim, **kwargs):
    "swish with learnable beta"
    return _resnet_basic_gn(9, 64, "relu", output_dim=output_dim, ws=True, **kwargs)


@register_class
def resnet18w32_gn_ws(output_dim, **kwargs):
    return _resnet_basic_gn(18, 32, "relu", output_dim=output_dim, ws=True, **kwargs)


@register_class
def resnet18_gn(output_dim, **kwargs):
    return _resnet_basic_gn(18, 64, "relu", output_dim=output_dim, **kwargs)


@register_class
def resnet18_gns(output_dim, **kwargs):
    return _resnet_basic_gn(18, 64, "swish", output_dim=output_dim, **kwargs)


@register_class
def resnet18_gn_ws(output_dim, **kwargs):
    "swish with learnable beta"
    return _resnet_basic_gn(18, 64, "relu", output_dim=output_dim, ws=True, **kwargs)