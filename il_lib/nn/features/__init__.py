from .simple import Embedding, Identity
from .fusion import SimpleFeatureFusion, ObsTokenizer
from .multiview_resnet18 import MultiviewResNet18
from .pointnet import PointNet, UncoloredPointNet


__all__ = [
    "Embedding",
    "Identity",
    "SimpleFeatureFusion",
    "ObsTokenizer",
    "MultiviewResNet18",
    "PointNet",
    "UncoloredPointNet",
]

try:
    from .min_vit import MinVit, MultiviewMinVit
except ModuleNotFoundError as exc:
    if exc.name != f"{__name__}.min_vit":
        raise
else:
    __all__.extend(["MinVit", "MultiviewMinVit"])
