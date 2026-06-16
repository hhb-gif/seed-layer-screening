import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from seed_layer.calculators.base import CalculatorBase


def test_calculator_base_is_abstract():
    """Test that CalculatorBase cannot be instantiated directly."""
    with pytest.raises(TypeError):
        CalculatorBase()


def test_create_calculator_factory():
    """Test factory function creates correct calculator type."""
    from seed_layer.calculators import create_calculator

    # Mock chgnet imports to avoid heavy dependency
    mock_struct_optimizer = MagicMock()
    with patch.dict('sys.modules', {
        'chgnet': MagicMock(),
        'chgnet.model': MagicMock(StructOptimizer=mock_struct_optimizer),
    }):
        config = {"type": "chgnet", "kwargs": {}}
        calc = create_calculator(config)
        assert isinstance(calc, CalculatorBase)


def test_create_calculator_unknown_type():
    """Test factory raises ValueError for unknown calculator type."""
    from seed_layer.calculators import create_calculator

    with pytest.raises(ValueError, match="Unknown calculator type"):
        create_calculator({"type": "unknown_calc", "kwargs": {}})
