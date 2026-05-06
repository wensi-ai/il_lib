# my_app/hydra_plugins/my_plugin.py
import os
from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


def _optional_repo_path(env_var: str, suffix: str) -> str | None:
    root = os.environ.get(env_var)
    if not root:
        return None
    candidate = Path(root).expanduser().resolve() / suffix
    return str(candidate) if candidate.exists() else None


def _discover_omnigibson_configs() -> str | None:
    try:
        import omnigibson as og

        return f"{og.__path__[0]}/learning/configs"
    except Exception:
        return _optional_repo_path(
            "OMNIGIBSON_REPO_ROOT",
            "omnigibson/learning/configs",
        )


def _discover_iiil_configs() -> str | None:
    try:
        import iiil

        return f"{iiil.__path__[0]}/configs"
    except Exception:
        return _optional_repo_path("IIIL_REPO_ROOT", "iiil/configs")

class SearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Append your custom search path (priority: after Hydra default)
        og_configs = _discover_omnigibson_configs()
        if og_configs is not None:
            search_path.append("il_lib", og_configs)

        iiil_configs = _discover_iiil_configs()
        if iiil_configs is not None:
            search_path.append("iiil", iiil_configs)
