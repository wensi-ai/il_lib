# my_app/hydra_plugins/my_plugin.py
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
import omnigibson as og
iiil = None
try:
    import iiil
except ImportError:
    pass

class SearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Append your custom search path (priority: after Hydra default)
        search_path.append("il_lib", f"{og.__path__[0]}/learning/configs")
        if iiil is not None:
            search_path.append("iiil", f"{iiil.__path__[0]}/configs")
