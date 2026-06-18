import os
import pytest
import yaml
import tempfile
from pathlib import Path
from seed_layer.config import load_config, save_config, PipelineConfig, _sanitize_api


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


# ---------------------------------------------------------------------------
# save_config tests
# ---------------------------------------------------------------------------

def test_save_config_roundtrip():
    """load -> save -> load should produce equivalent config values."""
    original_data = {
        "api": {"mp_api_key": "secret-abc", "base_url": "https://api.example.com"},
        "working_ion": "Na",
        "ref_structure_id": "mp-1234",
        "ref_miller": [1, 0, 0],
        "screening": {
            "energy_above_hull_max": 0.08,
            "n_elements": [2, 4],
            "elements_to_exclude": ["Pt"],
        },
        "calculator": {"type": "m3gnet", "kwargs": {"stress_weight": 0.1}},
        "relaxation": {"fmax": 0.02, "steps": 1000},
        "interface": {
            "max_metal_layers": 3,
            "slab_thickness": 6.0,
            "vacuum": 12.0,
            "fmax": 0.03,
            "steps": 300,
        },
        "scoring": {"weights": {"adsorption": 0.6, "diffusion": 0.4}},
        "output": {"dir": "./results"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(original_data, f)
        src_path = f.name

    try:
        cfg = load_config(src_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            dst_path = f.name

        try:
            save_config(cfg, dst_path)
            cfg2 = load_config(dst_path)

            # Numeric / scalar fields survive roundtrip
            assert cfg2.working_ion == "Na"
            assert cfg2.ref_structure_id == "mp-1234"
            assert tuple(cfg2.ref_miller) == (1, 0, 0)
            assert cfg2.screening["energy_above_hull_max"] == pytest.approx(0.08)
            assert cfg2.calculator == {"type": "m3gnet", "kwargs": {"stress_weight": 0.1}}
            assert cfg2.interface["max_metal_layers"] == 3
            assert cfg2.scoring["weights"]["adsorption"] == pytest.approx(0.6)
        finally:
            os.unlink(dst_path)
    finally:
        os.unlink(src_path)


def test_save_config_api_key_not_written():
    """API keys and tokens must NOT appear in the saved YAML file."""
    config_data = {
        "api": {
            "mp_api_key": "super-secret",
            "other_token": "do-not-leak",
            "base_url": "https://safe.example.com",
        },
        "calculator": {"type": "chgnet", "kwargs": {}},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        src_path = f.name

    try:
        cfg = load_config(src_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            dst_path = f.name

        try:
            save_config(cfg, dst_path)

            with open(dst_path, "r", encoding="utf-8") as f:
                saved_text = f.read()

            # Secrets must be absent
            assert "super-secret" not in saved_text
            assert "do-not-leak" not in saved_text

            # Non-secret values survive
            assert "https://safe.example.com" in saved_text
        finally:
            os.unlink(dst_path)
    finally:
        os.unlink(src_path)


def test_save_config_special_characters_roundtrip():
    """Special characters (unicode, colons, quotes) survive save/load."""
    config_data = {
        "working_ion": "锂",          # Chinese character
        "ref_structure_id": "mp-5678",
        "output": {
            "dir": "结果/目录",
            "comment": '含冒号: 和引号"的值',
        },
        "calculator": {"type": "chgnet", "kwargs": {}},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f, allow_unicode=True)
        src_path = f.name

    try:
        cfg = load_config(src_path)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            dst_path = f.name

        try:
            save_config(cfg, dst_path)
            cfg2 = load_config(dst_path)

            assert cfg2.working_ion == "锂"
            assert cfg2.output["dir"] == "结果/目录"
            assert cfg2.output["comment"] == '含冒号: 和引号"的值'
        finally:
            os.unlink(dst_path)
    finally:
        os.unlink(src_path)


def test_sanitize_api_filters_secrets():
    """_sanitize_api removes keys containing 'key', 'token', 'secret', 'password'."""
    raw = {
        "mp_api_key": "secret",
        "other_token": "secret",
        "some_secret": "secret",
        "db_password": "secret",
        "base_url": "https://example.com",
        "timeout": 30,
    }
    filtered = _sanitize_api(raw)
    assert "base_url" in filtered
    assert "timeout" in filtered
    assert len(filtered) == 2
    for forbidden in ("mp_api_key", "other_token", "some_secret", "db_password"):
        assert forbidden not in filtered
