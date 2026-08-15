from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from agents.feature_selection_agent import _EXCLUDE_ALWAYS, FeatureSelectionAgent
from agents.grouping_agent import OTHER_RESIDUAL, GroupingAgent
from core.llm_client import LLMClient
from core.schemas import (
    CategoricalFeatureConfig,
    CategoryCluster,
    FeatureProposal,
    FeatureSeed,
    GroupingResponse,
    NumericFeatureConfig,
)

DRAFTS_DIR = Path("reports/drafts")


def generate_draft(
    llm: LLMClient,
    df: pd.DataFrame,
    data_cfg: dict,
    grouping_cfg: dict,
    seed: FeatureSeed | None = None,
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
        seed=seed,
    )

    grp_agent = GroupingAgent(llm, min_exposure=grouping_cfg.get("min_exposure", 500))
    for cat in proposal.categorical:
        if cat.grouping is not None:
            # Already carries a seeded grouping — no agent call needed.
            continue
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
    seed: FeatureSeed | None = None,
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
        seed=seed,
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

    # Defensive carry-forward: don't trust the LLM to echo exclusion_rationale/
    # excluded_description for columns the actuary's remarks didn't touch.
    for col in updated.excluded:
        if col not in updated.exclusion_rationale and col in previous.exclusion_rationale:
            updated.exclusion_rationale[col] = previous.exclusion_rationale[col]
        if col not in updated.excluded_description and col in previous.excluded_description:
            updated.excluded_description[col] = previous.excluded_description[col]

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
    exclusion_rationale: dict[str, str] = {}
    excluded_description: dict[str, str] = {}
    if df is not None:
        data_cfg = config["data"]
        approved_names = {f.name for f in numeric} | {f.name for f in categorical}
        always_exclude = {data_cfg["target_col"], data_cfg["exposure_col"]} | _EXCLUDE_ALWAYS
        excluded = [c for c in df.columns if c not in approved_names and c not in always_exclude]
        # project_config.yaml never persists the agent's original exclusion_rationale/
        # excluded_description either, so until a real agent proposal touches these
        # columns, fall back to a generic reason plus a locally-computed data profile
        # in place of an actual actuarial description.
        exclusion_rationale = {c: "Not yet reviewed by the agent." for c in excluded}
        excluded_description = {c: _describe_column(df, c) for c in excluded}

    return FeatureProposal(
        numeric=numeric, categorical=categorical, excluded=excluded,
        exclusion_rationale=exclusion_rationale, excluded_description=excluded_description,
    )


def apply_groupings(df: pd.DataFrame, proposal: FeatureProposal) -> pd.DataFrame:
    """Replace each approved categorical column with its grouped cluster values.

    Shared by the Orchestrator and the dashboard so GBM training and GLM fitting
    always see the same grouped columns, regardless of which caller applies them.
    """
    df = df.copy()
    for cat_feat in proposal.categorical:
        if cat_feat.approved and cat_feat.grouping:
            mapping = {
                val: cluster_name
                for cluster_name, values in cat_feat.grouping.items()
                for val in values
            }
            df[cat_feat.name] = df[cat_feat.name].map(mapping).fillna(OTHER_RESIDUAL)
    return df


def reconcile_membership(
    draft: FeatureProposal,
    checkbox_state: dict[str, bool],
    df: pd.DataFrame,
    comments: dict[str, str] | None = None,
) -> FeatureProposal:
    """Recompute which list (numeric/categorical/excluded) each variable belongs to,
    purely from the actuary's current checkbox state.

    The agent never controls placement, only content (description/grouping/
    rationale) for whatever's already placed — unchecking a variable moves it to
    `excluded` even with no comment, and checking an `excluded` one promotes it.
    Call this before any agent refine call (so its previous_proposal_json reflects
    true membership) and again after the refine call returns, using the same
    `checkbox_state` — a defense-in-depth backstop, same pattern as the seed-config
    locks, so an agent response can never move a variable regardless of what it
    returns.
    """
    comments = comments or {}
    kept_numeric: list[NumericFeatureConfig] = []
    kept_categorical: list[CategoricalFeatureConfig] = []
    newly_excluded: list[str] = []

    def _sort(feat: NumericFeatureConfig | CategoricalFeatureConfig, kept: list) -> None:
        if checkbox_state.get(feat.name, False):
            feat.approved = True
            kept.append(feat)
        else:
            newly_excluded.append(feat.name)
            draft.excluded_description[feat.name] = feat.description
            draft.exclusion_rationale[feat.name] = feat.actuary_note or "Actuary excluded this round."

    for feat in draft.numeric:
        _sort(feat, kept_numeric)
    for feat in draft.categorical:
        _sort(feat, kept_categorical)

    still_excluded: list[str] = []
    for col in draft.excluded:
        if not checkbox_state.get(col, False):
            still_excluded.append(col)
            continue
        # Promoted — dtype decides which list it joins; description seeded from the
        # data profile until a remark gives the agent a chance to write a real one.
        description = _describe_column(df, col) if col in df.columns else ""
        note = comments.get(col) or None
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            kept_numeric.append(NumericFeatureConfig(
                name=col, description=description, approved=True, actuary_note=note,
            ))
        else:
            kept_categorical.append(CategoricalFeatureConfig(
                name=col, description=description, approved=True, actuary_note=note,
            ))
        draft.exclusion_rationale.pop(col, None)
        draft.excluded_description.pop(col, None)

    draft.numeric = kept_numeric
    draft.categorical = kept_categorical
    draft.excluded = still_excluded + newly_excluded
    return draft


# ── Draft snapshots (disk-backed, never overwritten, two kinds) ─────────────────
#
# "initial" — a genuinely fresh, actuary-untouched LLM proposal (only written by
# an explicit "regenerate from scratch" action). "modified" — an actuary-edited
# draft, snapshotted after every non-finalize Update round. Kept as separate,
# fully browsable histories (no pruning, no single "latest" pointer) rather than
# one undifferentiated cache — a snapshot's kind is exactly what it sounds like,
# never inferred after the fact.

def save_draft_snapshot(proposal: FeatureProposal, kind: str) -> Path:
    """Persist a draft snapshot to reports/drafts/<kind>/feature_draft_<timestamp>.yaml.

    Each call writes a new, distinctly-timestamped file — never overwrites a prior
    snapshot of either kind. Microsecond precision avoids collisions on rapid
    consecutive calls.
    """
    assert kind in ("initial", "modified"), f"unknown snapshot kind: {kind!r}"
    kind_dir = DRAFTS_DIR / kind
    kind_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = kind_dir / f"feature_draft_{timestamp}.yaml"
    with open(path, "w") as f:
        yaml.dump(proposal.model_dump(), f, allow_unicode=True, sort_keys=False)
    print(f"Draft snapshot ({kind}) saved to {path}")
    return path


def list_draft_snapshots(kind: str) -> list[Path]:
    """All saved snapshots of one kind, newest first."""
    assert kind in ("initial", "modified"), f"unknown snapshot kind: {kind!r}"
    kind_dir = DRAFTS_DIR / kind
    if not kind_dir.exists():
        return []
    return sorted(kind_dir.glob("feature_draft_*.yaml"), reverse=True)


def load_draft_snapshot(path: Path) -> FeatureProposal:
    """Load one specific snapshot file (from `list_draft_snapshots`)."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return FeatureProposal(**data)


def _describe_column(df: pd.DataFrame, col: str) -> str:
    series = df[col]
    null_pct = round(float(series.isnull().mean() * 100), 2)
    if pd.api.types.is_numeric_dtype(series):
        return (
            f"Numeric — mean {series.mean():.2f}, range [{series.min():.2f}, {series.max():.2f}], "
            f"{null_pct}% null."
        )
    n_unique = int(series.nunique())
    return f"Categorical — {n_unique} unique values, {null_pct}% null."


def save_feature_checkpoint(config_path: Path, config: dict, proposal: FeatureProposal) -> bool:
    """Write approved features back into the full project config dict and persist it.

    Returns True if the approved feature set (names + categorical groupings) differs
    from what was previously checkpointed. In that case any existing GBM/GLM checkpoints
    are cleared first — otherwise a stale feature set could silently survive into
    downstream training, since both stages skip re-running when a checkpoint exists.
    """
    old_features = config.get("features", {})
    old_signature = _feature_signature(
        [f["name"] for f in old_features.get("numeric", [])],
        {f["name"]: f.get("grouping") for f in old_features.get("categorical", [])},
    )

    approved_numeric = [f for f in proposal.numeric if f.approved is True]
    approved_categorical = [f for f in proposal.categorical if f.approved is True]
    new_signature = _feature_signature(
        [f.name for f in approved_numeric],
        {f.name: f.grouping for f in approved_categorical},
    )

    invalidated = False
    if old_signature != new_signature and (config.get("gbm_output") or _glm_terms_exist(config_path)):
        invalidate_downstream_checkpoints(config_path, config)
        invalidated = True

    config["features"] = {
        "numeric": [_feature_to_dict(f) for f in approved_numeric],
        "categorical": [_feature_to_dict(f) for f in approved_categorical],
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    print(f"Feature checkpoint saved to {config_path}")
    return invalidated


def _feature_signature(numeric_names: list[str], categorical_groupings: dict[str, dict | None]) -> dict:
    """Canonical snapshot of an approved feature set, for change detection."""
    return {"numeric": sorted(numeric_names), "categorical": categorical_groupings}


def _glm_terms_exist(config_path: Path) -> bool:
    glm_cfg_path = config_path.parent / "glm_config.yaml"
    if not glm_cfg_path.exists():
        return False
    with open(glm_cfg_path) as f:
        glm_cfg = yaml.safe_load(f) or {}
    return bool(glm_cfg.get("glm", {}).get("terms"))


def invalidate_downstream_checkpoints(config_path: Path, config: dict) -> None:
    """Clear GBM + GLM checkpoints when the approved feature set changes underneath them."""
    config.pop("gbm_output", None)
    glm_cfg_path = config_path.parent / "glm_config.yaml"
    if not glm_cfg_path.exists():
        return
    with open(glm_cfg_path) as f:
        glm_cfg = yaml.safe_load(f) or {}
    if glm_cfg.get("glm", {}).get("terms"):
        glm_cfg["glm"]["terms"] = []
        glm_cfg["glm"]["formula"] = None
        with open(glm_cfg_path, "w") as f:
            yaml.dump(glm_cfg, f, allow_unicode=True, sort_keys=False)
        print(f"GLM checkpoint cleared at {glm_cfg_path} — feature set changed.")


def _feature_to_dict(feat: NumericFeatureConfig | CategoricalFeatureConfig) -> dict:
    return {k: v for k, v in feat.model_dump().items() if v is not None}
