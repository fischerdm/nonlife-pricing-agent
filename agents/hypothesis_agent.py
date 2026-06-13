import json

from core.llm_client import LLMClient
from core.schemas import FeatureMetadata, HypothesisResponse

HYPOTHESIS_PROMPT = """
You are a senior actuary specializing in Non-Life insurance pricing.

Analyze the following features for a {target_type} model (target: {target_col}):
{features_json}

Identify the {n} most statistically and actuarially meaningful pairwise interactions.
For each interaction, explain WHY it creates a non-additive risk effect that exceeds
the sum of individual feature effects.

Respond ONLY with a valid JSON object — no preamble, no markdown:
{{
  "hypotheses": [
    {{
      "feature_a": "<name>",
      "feature_b": "<name>",
      "operation": "multiply | divide | ratio_a_over_b | ratio_b_over_a",
      "new_feature_name": "<snake_case_name>",
      "rationale": "<actuarial justification>"
    }}
  ]
}}
"""


class HypothesisAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(
        self,
        features: list[FeatureMetadata],
        target_col: str,
        target_type: str = "frequency",
        n_hypotheses: int = 5,
    ) -> HypothesisResponse:
        features_json = json.dumps([f.model_dump() for f in features], indent=2)
        prompt = HYPOTHESIS_PROMPT.format(
            target_type=target_type,
            target_col=target_col,
            features_json=features_json,
            n=n_hypotheses,
        )
        return self.llm.call(prompt, HypothesisResponse)
