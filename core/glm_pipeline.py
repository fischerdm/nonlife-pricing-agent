"""Shared GLM distillation checkpoint logic, used by both the Orchestrator and the dashboard."""
from pathlib import Path

import yaml

from core.schemas import GLMProposal, GLMTerm
from tools.glm_tools import build_formula


def proposal_from_glm_config(glm_config_path: Path) -> GLMProposal | None:
    """Reconstruct a GLMProposal from glm_config.yaml, or None if no checkpoint exists."""
    if not glm_config_path.exists():
        return None
    with open(glm_config_path) as f:
        glm_cfg = yaml.safe_load(f) or {}
    terms = glm_cfg.get("glm", {}).get("terms", [])
    if not terms:
        return None
    return GLMProposal(
        terms=[GLMTerm(**t) for t in terms],
        formula=glm_cfg["glm"].get("formula"),
    )


def save_glm_checkpoint(glm_config_path: Path, data_cfg: dict, proposal: GLMProposal) -> str:
    """Write approved GLM terms + formula to glm_config.yaml. Returns the built formula."""
    approved_terms = [t for t in proposal.terms if t.approved is True]
    formula = build_formula(data_cfg["target_col"], approved_terms)
    proposal.formula = formula

    glm_cfg: dict = {}
    if glm_config_path.exists():
        with open(glm_config_path) as f:
            glm_cfg = yaml.safe_load(f) or {}

    glm_cfg.setdefault("glm", {})
    glm_cfg["glm"]["objective"] = data_cfg["objective"]
    glm_cfg["glm"]["link"] = "log"
    glm_cfg["glm"]["terms"] = [
        {k: v for k, v in t.model_dump().items() if v is not None}
        for t in proposal.terms
        if t.approved is True
    ]
    glm_cfg["glm"]["formula"] = formula

    with open(glm_config_path, "w") as f:
        yaml.dump(glm_cfg, f, allow_unicode=True, sort_keys=False)
    print(f"GLM checkpoint saved to {glm_config_path}")
    return formula
