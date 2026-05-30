# CLAUDE.md — nonlife-pricing-agent

Full architecture spec is in **CLAUDE_CODE_HANDOFF.md**. This file is the quick operational brief.

## What this project is

Agentic Python framework for Non-Life insurance pricing model distillation:
LLM generates feature interaction hypotheses → LightGBM validates them → actuary approves via CLI gate → approved features enter a GLM.

## Implementation status

| Phase | Status | Key files |
|-------|--------|-----------|
| 1 — Core pipeline (hypothesis → validate → approve) | **Done** | `core/`, `agents/hypothesis_agent.py`, `agents/orchestrator.py`, `dashboard/approval_gate.py` |
| 2 — Categorical grouping | **Done** | `agents/grouping_agent.py` |
| 3 — GBM training + SHAP + distillation | **Stubs** — `raise NotImplementedError` | `agents/gbm_agent.py`, `agents/distillation_agent.py`, `tools/shap_tools.py`, `tools/glm_tools.py` |
| 4 — Streamlit dashboard + reports | **Not started** | `tools/reporting.py` (stub) |

**Do not start Phase 3 before Phases 1 + 2 are tested end-to-end with real data.**

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env          # add ANTHROPIC_API_KEY
pytest                        # all Phase 1+2 tests, no LLM calls needed
```

Drop a Parquet file at `data/training_data.parquet` and run via the orchestrator:

```python
from agents.orchestrator import Orchestrator
Orchestrator("config/project_config.yaml").run()
```

## Key design decisions

- **LLM model:** `claude-sonnet-4-6` (default in `LLMClient`). The handoff doc referenced an older model ID — this is the correct current one.
- **Prompt caching:** enabled on the actuary system prompt in `core/llm_client.py` — reduces cost on repeated runs.
- **Feature metadata → LLM, never raw rows:** cost + data privacy constraint. Pass `FeatureMetadata` objects, not DataFrames.
- **Temperature 0.2:** deterministic enough for actuarial reasoning, slight variation improves hypothesis diversity across runs.
- **Exposure-weighted LightGBM:** `w = df[exposure_col]` passed as `weight=` to LightGBM datasets. Industry standard for frequency/severity modeling.
- **Pydantic for all LLM outputs:** if the LLM returns malformed JSON, a clear `ValidationError` surfaces immediately — no silent failures.
- **Human gate is non-optional:** `dashboard/approval_gate.py` runs before any feature enters the final model. Regulatory requirement.

## Config

Edit `config/project_config.yaml` to set data path, feature list, LLM model/temperature, and validation thresholds. No code changes needed for a new dataset.

## LangGraph compatibility

All agents are stateless classes with typed inputs/outputs. Wrapping them in LangGraph nodes is straightforward when the time comes.
