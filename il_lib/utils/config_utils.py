from copy import deepcopy
from omegaconf import OmegaConf
from il_lib.utils.functional_utils import is_sequence, is_mapping, call_once, meta_decorator
from il_lib.utils.print_utils import to_scientific_str


_NO_INSTANTIATE = "__no_instantiate__"  # return config as-is

_CLASS_REGISTRY = {}  # for instantiation


@call_once(on_second_call="noop")
def register_omegaconf_resolvers():
    OmegaConf.register_new_resolver("eval", eval)
    
    # try each key until the key exists. Useful for multiple classes that have different
    # names for the same key
    def _try_key(cfg, *keys):
        for k in keys:
            if k in cfg:
                return cfg[k]
        raise KeyError(f"no key in {keys} is valid")

    OmegaConf.register_new_resolver("trykey", _try_key)
    # replace `resnet.gn.ws` -> `resnet_gn_ws`, because omegaconf doesn't support
    # keys with dots. Useful for generating run name with dots
    OmegaConf.register_new_resolver("underscore_to_dots", lambda s: s.replace("_", "."))

    def _no_instantiate(cfg):
        cfg = deepcopy(cfg)
        cfg[_NO_INSTANTIATE] = True
        return cfg

    OmegaConf.register_new_resolver("no_instantiate", _no_instantiate)


def omegaconf_to_dict(cfg, resolve: bool = True, enum_to_str: bool = False):
    """
    Convert arbitrary nested omegaconf objects to primitive containers

    WARNING: cannot use tree lib because it gets confused on DictConfig and ListConfig
    """
    kw = dict(resolve=resolve, enum_to_str=enum_to_str)
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, **kw)
    elif is_sequence(cfg):
        return type(cfg)(omegaconf_to_dict(c, **kw) for c in cfg)
    elif is_mapping(cfg):
        return {k: omegaconf_to_dict(c, **kw) for k, c in cfg.items()}
    else:
        return cfg


def omegaconf_save(cfg, *paths: str, resolve: bool = True):
    """
    Save omegaconf to yaml
    """
    from il_lib.utils.file_utils import f_join

    OmegaConf.save(cfg, f_join(*paths), resolve=resolve)


@meta_decorator
def register_class(cls, alias=None):
    """
    Decorator
    """
    assert callable(cls)
    _CLASS_REGISTRY[cls.__name__] = cls
    if alias:
        assert is_sequence(alias)
        for a in alias:
            _CLASS_REGISTRY[str(a)] = cls
    return cls

