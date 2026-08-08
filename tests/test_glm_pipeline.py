import yaml

from core.glm_pipeline import proposal_from_glm_config, save_glm_checkpoint
from core.schemas import GLMProposal, GLMTerm

DATA_CFG = {"target_col": "total_premium", "exposure_col": "total_exposure", "objective": "gamma"}


def _proposal():
    return GLMProposal(terms=[
        GLMTerm(name="driver_age", term_type="main", rationale="r", approved=True),
        GLMTerm(name="vehicle_age", term_type="main", rationale="r", approved=False),
        GLMTerm(
            name="driver_age:vehicle_age", term_type="interaction", h_statistic=0.3,
            rationale="r", approved=True,
        ),
    ])


def test_proposal_from_glm_config_missing_file_returns_none(tmp_path):
    assert proposal_from_glm_config(tmp_path / "glm_config.yaml") is None


def test_proposal_from_glm_config_no_terms_returns_none(tmp_path):
    path = tmp_path / "glm_config.yaml"
    path.write_text(yaml.dump({"glm": {"terms": []}}))
    assert proposal_from_glm_config(path) is None


def test_save_then_load_round_trips_approved_terms(tmp_path):
    path = tmp_path / "glm_config.yaml"
    formula = save_glm_checkpoint(path, DATA_CFG, _proposal())

    assert formula == "total_premium ~ driver_age + driver_age:vehicle_age"

    reloaded = proposal_from_glm_config(path)
    assert reloaded is not None
    assert {t.name for t in reloaded.terms} == {"driver_age", "driver_age:vehicle_age"}
    assert reloaded.formula == formula


def test_save_glm_checkpoint_only_persists_approved_terms(tmp_path):
    path = tmp_path / "glm_config.yaml"
    save_glm_checkpoint(path, DATA_CFG, _proposal())

    saved = yaml.safe_load(path.read_text())
    names = {t["name"] for t in saved["glm"]["terms"]}
    assert names == {"driver_age", "driver_age:vehicle_age"}
    assert "vehicle_age" not in names
