from pathlib import Path

import pandas as pd
import yaml

from agents.feature_selection_agent import _EXCLUDE_ALWAYS, FeatureSelectionAgent
from agents.grouping_agent import GroupingAgent
from core.llm_client import LLMClient
from core.schemas import (
    CategoricalFeatureConfig,
    CategoryCluster,
    FeatureProposal,
    GroupingResponse,
    NumericFeatureConfig,
)


def generate_draft(
    llm: LLMClient,
    df: pd.DataFrame,
    data_cfg: dict,
    grouping_cfg: dict,
) -> FeatureProposal:
    """Run feature selection, then immediately group every resulting categorical.

    This is the actuary-facing "one combined step": feature selection and
    grouping happen together as a single first draft, with no gate in between.
    """
    fs_agent = FeatureSelectionAgent(llm)
    proposal = fs_agent.propose(
        df=df,
        target_col=data_cfg["target_col"],
        exposure_col=data_cfg["exposure_col"],
        objective=data_cfg["objective"],
    )

    grp_agent = GroupingAgent(llm, min_exposure=grouping_cfg.get("min_exposure", 500))
    for cat in proposal.categorical:
        response = grp_agent.group(
            df=df,
            col_name=cat.name,
            exposure_col=data_cfg["exposure_col"],
            n_clusters=cat.n_clusters,
            claim_freq_col=data_cfg.get("claim_freq_col"),
        )
        cat.grouping = {c.cluster_name: c.elements for c in response.clusters}

    return proposal


def refine_draft(
    llm: LLMClient,
    df: pd.DataFrame,
    data_cfg: dict,
    grouping_cfg: dict,
    previous: FeatureProposal,
    remarks: dict[str, str],
) -> FeatureProposal:
    """Refine a draft with actuary remarks, re-grouping only what changed.

    Order matters: feature-selection refinement runs first, since it may change
    a categorical's n_clusters or promote a variable out of `excluded` — the
    grouping step below must use the post-refine values, not the stale ones.
    """
    fs_agent = FeatureSelectionAgent(llm)
    updated = fs_agent.refine(
        df=df,
        previous_proposal=previous,
        actuary_remarks=remarks,
        objective=data_cfg["objective"],
        target_col=data_cfg["target_col"],
        exposure_col=data_cfg["exposure_col"],
    )

    grp_agent = GroupingAgent(llm, min_exposure=grouping_cfg.get("min_exposure", 500))
    prev_cats_by_name = {c.name: c for c in previous.categorical}

    for cat in updated.categorical:
        prev_cat = prev_cats_by_name.get(cat.name)
        prev_grouping = prev_cat.grouping if prev_cat else None
        has_remark = cat.name in remarks

        if not has_remark and prev_grouping:
            # Untouched — carry the prior grouping forward explicitly rather
            # than trusting the LLM to echo a field it wasn't asked about.
            cat.grouping = prev_grouping
            continue

        if prev_grouping is None:
            # Newly approved or promoted from `excluded` — no prior grouping
            # to refine from, so generate one from scratch.
            response = grp_agent.group(
                df=df,
                col_name=cat.name,
                exposure_col=data_cfg["exposure_col"],
                n_clusters=cat.n_clusters,
                claim_freq_col=data_cfg.get("claim_freq_col"),
            )
        else:
            prev_response = GroupingResponse(clusters=[
                CategoryCluster(cluster_name=k, elements=v, rationale="")
                for k, v in prev_grouping.items()
            ])
            response = grp_agent.refine(
                df=df,
                col_name=cat.name,
                exposure_col=data_cfg["exposure_col"],
                n_clusters=cat.n_clusters,
                previous_response=prev_response,
                actuary_remarks={cat.name: remarks[cat.name]},
                claim_freq_col=data_cfg.get("claim_freq_col"),
            )

        cat.grouping = {c.cluster_name: c.elements for c in response.clusters}

    return updated


def proposal_from_config(config: dict, df: pd.DataFrame | None = None) -> FeatureProposal:
    """Reconstruct a FeatureProposal from the project_config.yaml checkpoint.

    project_config.yaml only ever persists *approved* features — the agent's
    `excluded` list is never written there. If `df` is given, the excluded
    list is reconstructed deterministically as every dataset column that
    isn't already approved, so a "revise" pass never hides a column from
    the actuary just because it was previously dropped.
    """
    features = config.get("features", {})
    numeric = [NumericFeatureConfig(**f) for f in features.get("numeric", [])]
    categorical = [CategoricalFeatureConfig(**f) for f in features.get("categorical", [])]

    excluded: list[str] = []
    if df is not None:
        data_cfg = config["data"]
        approved_names = {f.name for f in numeric} | {f.name for f in categorical}
        always_exclude = {data_cfg["target_col"], data_cfg["exposure_col"]} | _EXCLUDE_ALWAYS
        excluded = [c for c in df.columns if c not in approved_names and c not in always_exclude]

    return FeatureProposal(numeric=numeric, categorical=categorical, excluded=excluded)


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
