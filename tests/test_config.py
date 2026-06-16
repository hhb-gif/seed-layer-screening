import pytest
import yaml
import tempfile
from pathlib import Path
from seed_layer.config import load_config, PipelineConfig


def test_load_config_from_yaml():
    """Test loading config from YAML file."""
    config_data = {
        "api": {"mp_api_key": "test-key"},
        "screening": {
            "energy_above_hull_max": 0.10,
            "n_elements": [2, 3],
            "elements_to_exclude": ["Pt", "Pd"],
        },
        "calculator": {"type": "chgnet", "kwargs": {}},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    config = load_config(config_path)

    assert config.api["mp_api_key"] == "test-key"
    assert config.screening["energy_above_hull_max"] == 0.10
    assert config.calculator["type"] == "chgnet"


def test_env_variable_substitution():
    """Test that ${VAR} is replaced with environment variable."""
    import os
    os.environ["TEST_API_KEY"] = "my-secret-key"

    config_data = {
        "api": {"mp_api_key": "${TEST_API_KEY}"},
        "calculator": {"type": "chgnet", "kwargs": {}},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    config = load_config(config_path)
    assert config.api["mp_api_key"] == "my-secret-key"

    del os.environ["TEST_API_KEY"]


def test_pipeline_config_dataclass():
    """Test PipelineConfig creation."""
    config = PipelineConfig(
        api={"mp_api_key": "test"},
        screening={"energy_above_hull_max": 0.1},
        calculator={"type": "chgnet", "kwargs": {}},
    )

    assert config.api["mp_api_key"] == "test"
    assert config.calculator["type"] == "chgnet"
