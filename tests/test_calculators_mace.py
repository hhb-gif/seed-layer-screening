"""Tests for MACE calculator."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from seed_layer.calculators.base import CalculatorBase


def _mock_mace_module():
    """Create a mock mace module hierarchy."""
    mock_mace = MagicMock()
    mock_mace_mp = MagicMock()
    mock_mace.calculators.mace_mp = mock_mace_mp
    return mock_mace, mock_mace_mp


def test_create_mace_calculator():
    """Test factory creates MACE calculator."""
    from seed_layer.calculators import create_calculator

    mock_mace, mock_mace_mp = _mock_mace_module()
    with patch.dict("sys.modules", {
        "mace": mock_mace,
        "mace.calculators": mock_mace.calculators,
    }):
        config = {
            "type": "mace",
            "kwargs": {"model": "medium-mpa-0", "device": "cpu"},
        }
        calc = create_calculator(config)
        assert isinstance(calc, CalculatorBase)
        mock_mace_mp.assert_called_once_with(
            model="medium-mpa-0", device="cpu", default_dtype="float64", dispersion=False
        )


def test_mace_get_energy():
    """Test MACE get_energy returns a float."""
    mock_mace, mock_mace_mp = _mock_mace_module()
    mock_ase_calc = MagicMock()
    mock_mace_mp.return_value = mock_ase_calc

    with patch.dict("sys.modules", {
        "mace": mock_mace,
        "mace.calculators": mock_mace.calculators,
    }):
        from seed_layer.calculators.mace import MACECalculator
        calc = MACECalculator(model="medium-mpa-0", device="cpu")

    mock_atoms = MagicMock()
    mock_atoms.get_potential_energy.return_value = -3.5

    with patch.object(calc, "_to_ase", return_value=mock_atoms):
        result = calc.get_energy(MagicMock())
        assert isinstance(result, float)
        assert result == -3.5


def test_mace_get_forces():
    """Test MACE get_forces returns array of correct shape."""
    mock_mace, mock_mace_mp = _mock_mace_module()
    mock_ase_calc = MagicMock()
    mock_mace_mp.return_value = mock_ase_calc

    with patch.dict("sys.modules", {
        "mace": mock_mace,
        "mace.calculators": mock_mace.calculators,
    }):
        from seed_layer.calculators.mace import MACECalculator
        calc = MACECalculator(model="medium-mpa-0", device="cpu")

    mock_atoms = MagicMock()
    mock_atoms.get_forces.return_value = np.array([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]])

    with patch.object(calc, "_to_ase", return_value=mock_atoms):
        result = calc.get_forces(MagicMock())
        assert result.shape == (2, 3)


def test_mace_relax():
    """Test MACE relax returns correct dict structure."""
    mock_mace, mock_mace_mp = _mock_mace_module()
    mock_ase_calc = MagicMock()
    mock_mace_mp.return_value = mock_ase_calc

    # Mock both mace and ase modules
    mock_ase = MagicMock()
    mock_ase_filters = MagicMock()
    mock_ase_optimize = MagicMock()

    mock_bfgs_cls = MagicMock()
    mock_opt = MagicMock()
    mock_bfgs_cls.return_value = mock_opt
    mock_ase_optimize.BFGS = mock_bfgs_cls

    mock_ecf = MagicMock()
    mock_ase_filters.ExpCellFilter = mock_ecf

    with patch.dict("sys.modules", {
        "mace": mock_mace,
        "mace.calculators": mock_mace.calculators,
        "ase": mock_ase,
        "ase.filters": mock_ase_filters,
        "ase.optimize": mock_ase_optimize,
    }):
        from seed_layer.calculators.mace import MACECalculator
        calc = MACECalculator(model="medium-mpa-0", device="cpu")

    mock_atoms = MagicMock()
    mock_atoms.get_potential_energy.return_value = -5.0
    mock_structure = MagicMock()

    with (
        patch.object(calc, "_to_ase", return_value=mock_atoms),
        patch.object(calc, "_to_pmg", return_value=mock_structure),
        patch.dict("sys.modules", {
            "mace": mock_mace,
            "mace.calculators": mock_mace.calculators,
            "ase": mock_ase,
            "ase.filters": mock_ase_filters,
            "ase.optimize": mock_ase_optimize,
        }),
    ):
        result = calc.relax(MagicMock(), fmax=0.05, steps=100)

    assert "final_structure" in result
    assert "energy" in result
    assert "trajectory" in result
    assert result["energy"] == -5.0
    mock_opt.run.assert_called_once_with(fmax=0.05, steps=100)
