"""Unit tests for core.run_scope. No LLM calls, no real repo-root filesystem access —
`repo_root` is monkeypatched to a tmp_path for every test that touches disk."""
import pandas as pd
import pytest
import yaml

import core.run_scope as run_scope
from core.run_scope import RunConfigError, create_run, open_run, validate_config


@pytest.fixture(autouse=True)
def fake_repo_root(tmp_path, monkeypatch):
    monkeypatch.setattr(run_scope, "repo_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def template(tmp_path):
    """A minimal starting project_config.yaml template, same shape create_run copies."""
    (tmp_path / "config").mkdir()
    path = tmp_path / "config" / "project_config.example.yaml"
    path.write_text(yaml.dump({
        "data": {"path": "x", "sep": ",", "target_col": "t", "exposure_col": "e", "objective": "gamma"},
        "llm": {"model": "m", "temperature": 0.2},
    }))
    return path


# ── path derivation ──────────────────────────────────────────────────────────

def test_run_root_and_derived_paths(tmp_path):
    config_path = tmp_path / "my_run" / "config" / "project_config.yaml"

    assert run_scope.run_root(config_path) == tmp_path / "my_run"
    assert run_scope.reports_dir(config_path) == tmp_path / "my_run" / "reports"
    assert run_scope.sessions_dir(config_path) == tmp_path / "my_run" / "reports" / "sessions"
    assert run_scope.drafts_dir(config_path) == tmp_path / "my_run" / "reports" / "drafts"
    assert run_scope.decisions_log_path(config_path) == tmp_path / "my_run" / "reports" / "actuary_decisions.csv"


def test_gbm_model_path_uses_only_the_filename(tmp_path):
    config_path = tmp_path / "my_run" / "config" / "project_config.yaml"

    # A stray directory prefix left over from the pre-run-scoping config value
    # must not resolve against the CWD — only the filename survives.
    assert (
        run_scope.gbm_model_path(config_path, "reports/gbm_model.txt")
        == tmp_path / "my_run" / "reports" / "gbm_model.txt"
    )
    assert run_scope.gbm_model_path(config_path) == tmp_path / "my_run" / "reports" / "gbm_model.txt"


# ── active_run.yaml pointer ──────────────────────────────────────────────────

def test_get_active_run_name_raises_when_pointer_missing(tmp_path):
    with pytest.raises(RunConfigError):
        run_scope.get_active_run_name()


def test_get_active_run_name_raises_when_name_field_missing(tmp_path):
    (tmp_path / "active_run.yaml").write_text(yaml.dump({}))
    with pytest.raises(RunConfigError):
        run_scope.get_active_run_name()


def test_default_config_path_resolves_via_pointer(tmp_path):
    (tmp_path / "active_run.yaml").write_text(yaml.dump({"name": "my_run"}))
    assert run_scope.default_config_path() == tmp_path / "my_run" / "config" / "project_config.yaml"


# ── create_run / open_run ────────────────────────────────────────────────────

def test_create_run_scaffolds_and_activates(tmp_path, template):
    name = create_run("acme", template=template)

    assert name.startswith("acme_")
    run_dir = tmp_path / name
    assert (run_dir / "config" / "project_config.yaml").exists()
    assert (run_dir / "reports" / "sessions").is_dir()
    for kind in ("initial", "modified", "finalized"):
        assert (run_dir / "reports" / "drafts" / kind).is_dir()
    assert run_scope.get_active_run_name() == name


def test_create_run_applies_only_the_given_data_overrides(tmp_path, template):
    name = create_run("acme", template=template, dataset_path="data/real.csv", target_col="t2")

    config = yaml.safe_load((tmp_path / name / "config" / "project_config.yaml").read_text())
    assert config["data"]["path"] == "data/real.csv"
    assert config["data"]["target_col"] == "t2"
    # Untouched fields keep the template's own values.
    assert config["data"]["exposure_col"] == "e"
    assert config["data"]["objective"] == "gamma"


def test_create_run_with_no_overrides_leaves_template_data_block_untouched(tmp_path, template):
    name = create_run("acme", template=template)

    config = yaml.safe_load((tmp_path / name / "config" / "project_config.yaml").read_text())
    template_config = yaml.safe_load(template.read_text())
    assert config["data"] == template_config["data"]


def test_create_run_refuses_to_overwrite_an_existing_folder(tmp_path, template, monkeypatch):
    # Force two calls to collide on the same timestamp — timestamp uniqueness
    # already makes this practically impossible in real use, but the guard
    # should still refuse rather than silently reuse the folder.
    fixed_now = run_scope.datetime(2026, 1, 1, 0, 0, 0)
    monkeypatch.setattr(run_scope, "datetime", type("_FixedDT", (), {"now": staticmethod(lambda: fixed_now)}))

    create_run("dup", template=template)
    with pytest.raises(RunConfigError):
        create_run("dup", template=template)


def test_open_run_activates_existing_run(tmp_path, template):
    name = create_run("acme", template=template)
    create_run("other", template=template)  # switches active to "other"
    assert run_scope.get_active_run_name() != name

    open_run(name)
    assert run_scope.get_active_run_name() == name


def test_open_run_raises_for_missing_run(tmp_path):
    with pytest.raises(RunConfigError):
        open_run("does_not_exist")


# ── validate_config ──────────────────────────────────────────────────────────

def _write_dataset(path, columns):
    pd.DataFrame({c: [1, 2] for c in columns}).to_csv(path, index=False)


def _config(tmp_path, **data_overrides):
    run_dir = tmp_path / "a_run"
    (run_dir / "config").mkdir(parents=True)
    config_path = run_dir / "config" / "project_config.yaml"
    data = {"path": str(run_dir / "data.csv"), "sep": ",", "target_col": "t",
            "exposure_col": "e", "objective": "gamma"}
    data.update(data_overrides)
    config_path.write_text(yaml.dump({"data": data}))
    return config_path


def test_validate_config_raises_when_run_folder_missing(tmp_path):
    with pytest.raises(RunConfigError):
        validate_config(tmp_path / "nope" / "config" / "project_config.yaml")


def test_validate_config_raises_when_project_config_missing(tmp_path):
    run_dir = tmp_path / "a_run"
    (run_dir / "config").mkdir(parents=True)
    with pytest.raises(RunConfigError):
        validate_config(run_dir / "config" / "project_config.yaml")


def test_validate_config_raises_when_required_data_key_missing(tmp_path):
    run_dir = tmp_path / "a_run"
    (run_dir / "config").mkdir(parents=True)
    config_path = run_dir / "config" / "project_config.yaml"
    config_path.write_text(yaml.dump({"data": {"path": "x"}}))

    with pytest.raises(RunConfigError):
        validate_config(config_path)


def test_validate_config_raises_when_dataset_file_missing(tmp_path):
    config_path = _config(tmp_path)  # data.path points at a dataset that doesn't exist
    with pytest.raises(RunConfigError):
        validate_config(config_path)


def test_validate_config_passes_with_no_seeds(tmp_path):
    config_path = _config(tmp_path)
    _write_dataset(config_path.parent.parent / "data.csv", ["t", "e", "vehicle_age"])

    assert validate_config(config_path) == []


def test_validate_config_raises_when_target_col_not_a_real_column(tmp_path):
    # The exact mistake a fresh create_run leaves behind if --target-col
    # isn't passed and the template's placeholder is never hand-edited.
    config_path = _config(tmp_path, target_col="not_a_column")
    _write_dataset(config_path.parent.parent / "data.csv", ["t", "e", "vehicle_age"])

    with pytest.raises(RunConfigError, match="target_col"):
        validate_config(config_path)


def test_validate_config_raises_when_exposure_col_not_a_real_column(tmp_path):
    config_path = _config(tmp_path, exposure_col="not_a_column")
    _write_dataset(config_path.parent.parent / "data.csv", ["t", "e", "vehicle_age"])

    with pytest.raises(RunConfigError, match="exposure_col"):
        validate_config(config_path)


def test_validate_config_warns_on_seed_entry_naming_an_absent_column(tmp_path):
    config_path = _config(tmp_path)
    _write_dataset(config_path.parent.parent / "data.csv", ["t", "e", "vehicle_age"])
    (config_path.parent / "feature_seed.yaml").write_text(yaml.dump({
        "numeric": [{"name": "does_not_exist"}],
        "categorical": [],
    }))

    warnings = validate_config(config_path)
    assert len(warnings) == 1
    assert "does_not_exist" in warnings[0]
    assert "feature_seed.yaml" in warnings[0]
