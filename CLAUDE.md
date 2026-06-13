# CLAUDE.md — nonlife-pricing-agent

Full architecture spec is in **CLAUDE_CODE_HANDOFF.md**. This file is the quick operational brief.

## What this project is

Agentic Python tool that supports actuaries in Non-Life insurance pricing model development.
The goal is to distill a GBM into an interpretable GLM, with a human-in-the-loop at every key decision.

**Pipeline:**
```
Actuary sets target + exposure
        ↓
Feature Selection Agent
  → profiles all columns, LLM proposes feature list with actuarial rationale
  → actuary reviews variable by variable (approve / reject / remark)
  → remarks feed back to LLM for a revised proposal (loop until confirmed)
  → approved list saved to project_config.yaml as checkpoint
        ↓
Grouping Agent (for approved categoricals)
  → LLM clusters high-cardinality variables into risk-homogeneous groups
  → actuary reviews and refines
        ↓
GBM trains on approved features  [Phase 3 — stub]
        ↓
SHAP + H-statistics rank interactions  [Phase 3 — stub]
  → LLM proposes GLM terms with actuarial rationale
  → actuary reviews term by term (same loop)
  → approved terms saved to glm_config.yaml
        ↓
GLM fitted on approved terms  [Phase 3 — stub]
```

## Implementation status

| Component | Status | Key files |
|-----------|--------|-----------|
| Feature selection agent + actuary gate | **Done** | `agents/feature_selection_agent.py`, `dashboard/approval_gate.py` |
| Prompt template system | **Done** | `prompts/feature_selection.yaml`, `prompts/grouping.yaml`, `prompts/distillation.yaml` |
| Categorical grouping agent | **Done** | `agents/grouping_agent.py` |
| Orchestrator with checkpoint logic | **Done** | `agents/orchestrator.py` |
| Pydantic schemas | **Done** | `core/schemas.py` |
| LLM client with prompt caching + templates | **Done** | `core/llm_client.py` |
| GBM training + SHAP | **Stub** | `agents/gbm_agent.py`, `tools/shap_tools.py` |
| GLM distillation agent + gate | **Stub** | `agents/distillation_agent.py`, `tools/glm_tools.py` |
| Streamlit dashboard | **Not started** | `tools/reporting.py` |

**Next step: implement GBM agent (Phase 3).**

## Dataset

Motor insurance portfolio — DOI: 10.17632/sw4jmdb2sm.1
- 354,140 rows × 47 variables, semicolon-delimited CSV
- Target: `total_premium` (Gamma GLM), 49 zero-premium rows filtered
- Exposure: `total_exposure` (policy-years)
- File: `data/Dataset of motor insurance portfolio.csv`

## Setup

Python 3.12.8 (pinned in `.python-version`, managed via pyenv).

```bash
pyenv local 3.12.8
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # add ANTHROPIC_API_KEY
pytest                        # unit tests, no LLM calls needed
```

Run the pipeline:

```python
from agents.orchestrator import Orchestrator
Orchestrator("config/project_config.yaml").run()
```

## Config files

| File | Purpose |
|------|---------|
| `config/project_config.yaml` | Data settings, LLM, validation params. Feature list added here after actuary approval (checkpoint). |
| `config/glm_config.yaml` | GLM terms and formula. Populated after distillation gate (checkpoint). |

The actuary can pre-populate either config to skip the agent proposal step.

## Key design decisions

- **GBM-first, not hypothesis-first:** the GBM reveals what the data says; the LLM and actuary then decide what goes into the GLM. More grounded than speculative hypothesis generation.
- **Prompts in `prompts/` YAML files:** separated from code, easy to iterate without touching Python. Each file has named sections (`proposal`, `refinement`) used by the corresponding agent.
- **Actuary-in-the-loop at every stage:** feature selection, grouping, and GLM term selection all have an approve/reject/remark gate. Remarks loop back to the LLM for refinement.
- **Checkpoint pattern:** approved decisions are written back to YAML. Re-running skips already-approved stages.
- **Pydantic for all LLM outputs:** malformed JSON surfaces as a clear `ValidationError` immediately.
- **Prompt caching:** system prompt cached in `LLMClient` — reduces cost on repeated calls.
- **Temperature 0.2:** stable enough for actuarial reasoning, slight variation improves proposal diversity.
- **LangGraph-compatible:** all agents are stateless classes with typed inputs/outputs.
