__all__ = [
    "ACT",
    "BaseChunkDiffusionPolicy",
    "BaseChunkFlowMatchingPolicy",
    "BaseChunkPolicy",
    "BC_RNN",
    "CompositionalPolicy",
    "DiffusionPolicy",
    "GatedDiffusionPolicy",
    "InterventionClassifier",
    "ResidualDiffusionPolicy",
    "ResidualPolicy",
    "SimpleResidualPolicy",
    "WBVIMA",
]


def __getattr__(name):
    if name == "ACT":
        from .act_policy import ACT

        return ACT
    if name == "BaseChunkDiffusionPolicy":
        from .base_chunk_diffusion_policy import BaseChunkDiffusionPolicy

        return BaseChunkDiffusionPolicy
    if name == "BaseChunkFlowMatchingPolicy":
        from .base_chunk_flow_matching_policy import BaseChunkFlowMatchingPolicy

        return BaseChunkFlowMatchingPolicy
    if name == "BaseChunkPolicy":
        from .base_chunk_policy import BaseChunkPolicy

        return BaseChunkPolicy
    if name == "BC_RNN":
        from .bcrnn_policy import BC_RNN

        return BC_RNN
    if name == "CompositionalPolicy":
        from .compositional_policy import CompositionalPolicy

        return CompositionalPolicy
    if name == "DiffusionPolicy":
        from .diffusion_policy import DiffusionPolicy

        return DiffusionPolicy
    if name == "GatedDiffusionPolicy":
        from .gated_diffusion_policy import GatedDiffusionPolicy

        return GatedDiffusionPolicy
    if name == "InterventionClassifier":
        from .intervention_classifier import InterventionClassifier

        return InterventionClassifier
    if name == "ResidualDiffusionPolicy":
        from .residual_diffusion_policy import ResidualDiffusionPolicy

        return ResidualDiffusionPolicy
    if name == "ResidualPolicy":
        from .residual_policy import ResidualPolicy

        return ResidualPolicy
    if name == "SimpleResidualPolicy":
        from .simple_residual_policy import SimpleResidualPolicy

        return SimpleResidualPolicy
    if name == "WBVIMA":
        from .wbvima_policy import WBVIMA

        return WBVIMA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
