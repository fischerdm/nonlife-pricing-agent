from pathlib import Path

import yaml

from core.schemas import CategoricalFeatureConfig, FeatureProposal, NumericFeatureConfig


def proposal_from_config(config: dict) -> FeatureProposal:
    """Reconstruct a FeatureProposal from the project_config.yaml checkpoint."""
    features = config.get("features", {})
    numeric = [NumericFeatureConfig(**f) for f in features.get("numeric", [])]
    categorical = [CategoricalFeatureConfig(**f) for f in features.get("categorical", [])]
    return FeatureProposal(numeric=numeric, categorical=categorical)


def save_feature_checkpoint(config_path: Path, config: dict, proposal: FeatureProposal) -> None:
    """Write approved features back into the full project config dict and persist it."""
    config["features"] = {
        "numeric": [_feature_to_dict(f) for f in proposal.numeric if f.approved is True],
        "categorical": [_feature_to_dict(f) for f in proposal.categorical if f.approved is True],
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    print(f"Feature checkpoint saved to {config_path}")


def _feature_to_dict(feat: NumericFeatureConfig | CategoricalFeatureConfig) -> dict:
    return {k: v for k, v in feat.model_dump().items() if v is not None}
