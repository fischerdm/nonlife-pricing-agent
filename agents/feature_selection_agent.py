import json

import pandas as pd

from core.llm_client import LLMClient
from core.schemas import FeatureProposal

_EXCLUDE_ALWAYS = {"insured_id", "year"}


class FeatureSelectionAgent:
    """Profile dataset columns, propose a feature list, refine based on actuary remarks."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def propose(
        self,
        df: pd.DataFrame,
        target_col: str,
        exposure_col: str,
        objective: str,
    ) -> FeatureProposal:
        exclude = {target_col, exposure_col} | _EXCLUDE_ALWAYS
        profiles = self._profile_columns(df, exclude)
        return self.llm.call_template(
            agent_name="feature_selection",
            section="proposal",
            response_model=FeatureProposal,
            objective=objective,
            target_col=target_col,
            exposure_col=exposure_col,
            column_profiles_json=json.dumps(profiles, indent=2),
        )

    def refine(
        self,
        previous_proposal: FeatureProposal,
        actuary_remarks: dict[str, str],
        objective: str,
        target_col: str,
        exposure_col: str,
    ) -> FeatureProposal:
        return self.llm.call_template(
            agent_name="feature_selection",
            section="refinement",
            response_model=FeatureProposal,
            objective=objective,
            target_col=target_col,
            exposure_col=exposure_col,
            previous_proposal_json=json.dumps(previous_proposal.model_dump(), indent=2),
            actuary_remarks_json=json.dumps(actuary_remarks, indent=2),
        )

    def _profile_columns(self, df: pd.DataFrame, exclude: set[str]) -> list[dict]:
        profiles = []
        for col in df.columns:
            if col in exclude:
                continue
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            profile: dict = {
                "name": col,
                "dtype": "numeric" if is_numeric else "categorical",
                "null_pct": round(float(df[col].isnull().mean() * 100), 2),
                "n_unique": int(df[col].nunique()),
            }
            if is_numeric:
                profile.update({
                    "min": round(float(df[col].min()), 4),
                    "max": round(float(df[col].max()), 4),
                    "mean": round(float(df[col].mean()), 4),
                    "std": round(float(df[col].std()), 4),
                })
            else:
                profile["top_values"] = {
                    str(k): int(v)
                    for k, v in df[col].value_counts().head(10).items()
                }
            profiles.append(profile)
        return profiles
