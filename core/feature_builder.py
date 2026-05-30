import numpy as np
import pandas as pd

from core.schemas import InteractionHypothesis


def build_interaction_feature(
    df: pd.DataFrame, hypothesis: InteractionHypothesis
) -> pd.Series:
    a = df[hypothesis.feature_a].astype(float)
    b = df[hypothesis.feature_b].astype(float)

    if hypothesis.operation == "multiply":
        result = a * b
    elif hypothesis.operation in ("divide", "ratio_a_over_b"):
        result = a / b.replace(0, np.nan)
    elif hypothesis.operation == "ratio_b_over_a":
        result = b / a.replace(0, np.nan)
    else:
        raise ValueError(f"Unknown operation: {hypothesis.operation}")

    return result.rename(hypothesis.new_feature_name)
