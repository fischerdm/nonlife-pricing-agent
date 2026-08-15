"""Optional actuary-authored seed configs that prime a fresh agent draft.

Seeds are hand-edited YAML, same as how `project_config.yaml` can already be
pre-populated to skip a stage — there is no seed-authoring UI. Both loaders
return None when the file is absent, so callers throughout the pipeline treat
"no seed" and "seed not yet implemented" identically. See CLAUDE.md and the
`seed_config_design` memory for the full design.
"""
from pathlib import Path

import yaml

from core.schemas import DistillationSeed, FeatureSeed

FEATURE_SEED_FILENAME = "feature_seed.yaml"
DISTILLATION_SEED_FILENAME = "distillation_seed.yaml"


def load_feature_seed(path: Path) -> FeatureSeed | None:
    if not path.exists():
        return None
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return FeatureSeed(**raw)


def load_distillation_seed(path: Path) -> DistillationSeed | None:
    if not path.exists():
        return None
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return DistillationSeed(**raw)
