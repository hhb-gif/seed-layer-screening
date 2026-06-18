"""Calculator factory and base class."""

from typing import Any, Dict

from .base import CalculatorBase


def create_calculator(config: Dict[str, Any]) -> CalculatorBase:
    """Create calculator instance from config.

    Args:
        config: Calculator configuration with 'type' and 'kwargs'

    Returns:
        CalculatorBase implementation

    Raises:
        ValueError: If calculator type is unknown
    """
    calc_type = config.get("type", "chgnet")
    kwargs = config.get("kwargs", {})

    if calc_type == "chgnet":
        from .chgnet import CHGNetCalculator
        return CHGNetCalculator(**kwargs)
    elif calc_type == "mace":
        from .mace import MACECalculator
        return MACECalculator(**kwargs)
    else:
        raise ValueError(f"Unknown calculator type: {calc_type}")


__all__ = ["CalculatorBase", "create_calculator"]
