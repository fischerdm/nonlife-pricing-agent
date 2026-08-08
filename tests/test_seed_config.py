import yaml

from core.seed_config import load_distillation_seed, load_feature_seed


def test_load_feature_seed_missing_file_returns_none(tmp_path):
    assert load_feature_seed(tmp_path / "feature_seed.yaml") is None


def test_load_distillation_seed_missing_file_returns_none(tmp_path):
    assert load_distillation_seed(tmp_path / "distillation_seed.yaml") is None


def test_load_feature_seed_parses_with_default_temperature(tmp_path):
    path = tmp_path / "feature_seed.yaml"
    path.write_text(yaml.dump({
        "numeric": [{"name": "vehicle_value", "description": "d", "approved": True}],
        "categorical": [{
            "name": "vehicle_brand", "description": "d", "n_clusters": 8,
            "temperature": 0.0, "grouping": {"A": ["x"]},
        }],
    }))

    seed = load_feature_seed(path)

    assert seed.numeric[0].name == "vehicle_value"
    assert seed.numeric[0].temperature == 0.3  # default when omitted
    assert seed.categorical[0].temperature == 0.0
    assert seed.categorical[0].grouping == {"A": ["x"]}


def test_load_distillation_seed_parses_with_default_temperature(tmp_path):
    path = tmp_path / "distillation_seed.yaml"
    path.write_text(yaml.dump({
        "commercially_excluded": [{"name": "driver_occupation", "rationale": "regulatory"}],
    }))

    seed = load_distillation_seed(path)

    assert seed.commercially_excluded[0].name == "driver_occupation"
    assert seed.commercially_excluded[0].temperature == 0.0  # default when omitted


def test_load_feature_seed_empty_file_returns_empty_seed(tmp_path):
    path = tmp_path / "feature_seed.yaml"
    path.write_text("")

    seed = load_feature_seed(path)

    assert seed.numeric == []
    assert seed.categorical == []
