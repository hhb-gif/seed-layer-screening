"""Screening steps for seed layer pipeline."""

from .base import BaseStep

__all__ = [
    "BaseStep",
    "StabilityStep",
    "LatticeStep",
    "AdsorptionStep",
    "DiffusionStep",
]


def __getattr__(name: str):
    """Lazy imports for concrete step classes.

    These modules depend on mp-api which may not be installed,
    so we defer their import until actually accessed.
    """
    if name == "StabilityStep":
        from .stability import StabilityStep
        return StabilityStep
    elif name == "LatticeStep":
        from .lattice import LatticeStep
        return LatticeStep
    elif name == "AdsorptionStep":
        from .adsorption import AdsorptionStep
        return AdsorptionStep
    elif name == "DiffusionStep":
        from .diffusion import DiffusionStep
        return DiffusionStep
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
