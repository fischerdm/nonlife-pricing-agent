import json

import pandas as pd

from core.llm_client import LLMClient
from core.schemas import GroupingResponse

GROUPING_PROMPT = """
You are a senior actuary. Group these categorical values of the variable '{col_name}'
into exactly {n_clusters} risk-homogeneous clusters for a Non-Life pricing model.

Data (value, exposure count{freq_hint}):
{values_json}

Base groupings on actuarial risk logic — driving behavior, occupational exposure,
vehicle characteristics, etc. — NOT alphabetical or arbitrary similarity.
Every value listed must appear in exactly one cluster.

Respond ONLY with valid JSON — no preamble:
{{
  "clusters": [
    {{
      "cluster_name": "<SNAKE_CASE_NAME>",
      "elements": ["value1", "value2"],
      "rationale": "<actuarial justification>"
    }}
  ]
}}
"""

OTHER_RESIDUAL = "Other_Residual"


class GroupingAgent:
    def __init__(self, llm_client: LLMClient, min_exposure: int = 500):
        self.llm = llm_client
        self.min_exposure = min_exposure

    def group(
        self,
        df: pd.DataFrame,
        col_name: str,
        exposure_col: str,
        n_clusters: int,
        claim_freq_col: str | None = None,
    ) -> dict[str, str]:
        value_stats = self._compute_value_stats(df, col_name, exposure_col, claim_freq_col)
        freq_hint = ", claim frequency" if claim_freq_col else ""

        prompt = GROUPING_PROMPT.format(
            col_name=col_name,
            n_clusters=n_clusters,
            freq_hint=freq_hint,
            values_json=json.dumps(value_stats, indent=2),
        )
        response: GroupingResponse = self.llm.call(prompt, GroupingResponse)
        return self._build_mapping(response)

    def apply_grouping(
        self, df: pd.DataFrame, col_name: str, mapping: dict[str, str]
    ) -> pd.Series:
        return df[col_name].map(mapping).fillna(OTHER_RESIDUAL)

    def _compute_value_stats(
        self,
        df: pd.DataFrame,
        col_name: str,
        exposure_col: str,
        claim_freq_col: str | None,
    ) -> list[dict]:
        agg: dict = {exposure_col: "sum"}
        if claim_freq_col:
            agg[claim_freq_col] = "mean"

        stats = df.groupby(col_name).agg(agg).reset_index()
        stats = stats[stats[exposure_col] >= self.min_exposure]

        records = []
        for _, row in stats.iterrows():
            entry: dict = {
                "value": row[col_name],
                "exposure": round(float(row[exposure_col]), 1),
            }
            if claim_freq_col:
                entry["claim_frequency"] = round(float(row[claim_freq_col]), 4)
            records.append(entry)

        return records

    def _build_mapping(self, response: GroupingResponse) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for cluster in response.clusters:
            for element in cluster.elements:
                mapping[element] = cluster.cluster_name
        return mapping
