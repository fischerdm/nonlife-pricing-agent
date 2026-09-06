"""Unit tests for the pure (non-Streamlit) helpers in dashboard/glm_workbench.py.

The workbench itself is UI code and not otherwise unit tested (see other
dashboard/*.py modules) — but a couple of its helpers are plain functions with
no Streamlit dependency, and one of them (`_approved_feature_names`) turned out
to hide a real bug that unit tests would have caught immediately, found instead
only by driving the dashboard live (see the `glm_distillation_workbench` memory
note). Worth a regression test precisely because it's easy to get subtly wrong
again without one.
"""
from dashboard.glm_workbench import _approved_feature_names


def test_approved_feature_names_excludes_rejected_features():
    cfg = {"features": {
        "numeric": [
            {"name": "driver_age", "approved": True},
            {"name": "vehicle_age", "approved": False},
        ],
        "categorical": [
            {"name": "policy_type", "approved": True},
            {"name": "business_type", "approved": False},
        ],
    }}

    assert _approved_feature_names(cfg) == ["driver_age", "policy_type"]
