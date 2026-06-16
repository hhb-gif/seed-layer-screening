"""Tests for IO module."""

import pytest
import json
import tempfile
from pathlib import Path
from seed_layer.io import (
    save_structure_cif,
    save_structure_xyz,
    save_json,
    create_material_dir,
    create_miller_dir,
)


def test_save_json():
    """Test saving JSON data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = {"energy": -10.5, "passed": True}
        path = Path(tmpdir) / "test.json"
        save_json(data, path)

        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["energy"] == -10.5
        assert loaded["passed"] is True


def test_create_material_dir():
    """Test creating material directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        mat_dir = create_material_dir(base, "mp-12345")

        assert mat_dir.exists()
        assert mat_dir.name == "mp-12345"


def test_create_miller_dir():
    """Test creating miller index directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        miller_dir = create_miller_dir(base, (1, 1, 0))

        assert miller_dir.exists()
        assert miller_dir.name == "(110)"
