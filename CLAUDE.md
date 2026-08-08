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
GBM trains on approved features
  → LightGBM with MSE on log(premium/exposure) — standard log-rate target
  → Friedman H-statistics rank pairwise interactions among top-N features
  → interactions + model saved as checkpoint in project_config.yaml
        ↓
Distillation Agent
  → LLM proposes main effects (all approved features) + pairwise interaction terms
  → actuary reviews term by term (same loop)
  → approved terms + patsy formula saved to glm_config.yaml
        ↓
GLM fitted on approved terms
  → Gamma GLM with log link, log(exposure) offset via statsmodels
  → coefficients, deviance explained, AIC printed
  → post-fit coefficient review gate: actuary reviews each term's sign/significance,
    rejects suppressor variables or sparse levels; rejected terms are dropped and the
    model is automatically refit until no rejections remain
  → rating factors table: exp(coef) per parameter as multiplicative relativities
```

Every gate above can be worked either via the CLI (`dashboard/approval_gate.py`, terminal A/R/N/S prompts) or the Streamlit dashboard's Layer 2 interactive workbenches — both read/write the same YAML checkpoints.

**Two separate deliverables, not one pipeline with one output:** the GBM (technical/risk price, explores what the data says) and the distilled GLM (commercial tariff, explains effects). A variable can be GBM-relevant but GLM-ineligible (not available at quote time, regulatory, business strategy) — that's why GLM main-effect inclusion needs its own actuary gate even though the variable's GBM inclusion was already decided upstream, not a shared flag.

## Implementation status

| Component | Status | Key files |
|-----------|--------|-----------|
| Feature selection agent + actuary gate | **Done** | `agents/feature_selection_agent.py`, `dashboard/approval_gate.py` |
| Prompt template system | **Done** | `prompts/feature_selection.yaml`, `prompts/grouping.yaml`, `prompts/distillation.yaml` |
| Categorical grouping agent + actuary gate | **Done** | `agents/grouping_agent.py`, `dashboard/approval_gate.py` |
| Orchestrator with checkpoint logic | **Done** | `agents/orchestrator.py` |
| Pydantic schemas | **Done** | `core/schemas.py` |
| LLM client with prompt caching + templates | **Done** | `core/llm_client.py` |
| GBM training + H-statistics | **Done** | `agents/gbm_agent.py`, `tools/shap_tools.py` |
| GLM distillation agent + gate | **Done** | `agents/distillation_agent.py`, `dashboard/approval_gate.py` |
| GLM fitting + diagnostics | **Done** | `tools/glm_tools.py`, `dashboard/approval_gate.py` |
| Session logging | **Done** | `core/session_logger.py` |
| Shared checkpoint/training logic (CLI + dashboard, one impl each) | **Done** | `core/feature_pipeline.py`, `core/gbm_pipeline.py`, `core/glm_pipeline.py` |
| Streamlit dashboard — Layer 1 (read-only) | **Done** | `dashboard/streamlit_app.py` |
| Streamlit dashboard — Layer 2: Feature & Grouping Workbench | **Done** | `dashboard/feature_workbench.py` |
| Streamlit dashboard — Layer 2: in-dashboard GBM retrain | **Done** | `dashboard/gbm_workbench.py` |
| Streamlit dashboard — Layer 2: GLM Distillation Workbench | **Done** | `dashboard/glm_workbench.py` |
| Streamlit dashboard — Layer 2: GLM coefficient review | **Done** | `dashboard/glm_coef_workbench.py` |
| Shared dashboard session state (LLM client, cached df, logger) | **Done** | `dashboard/_session.py` |

**Full CLI pipeline proven end-to-end (2026-06-20). Streamlit dashboard Layer 2 completed for all four actuary gates (2026-08-08, branch `orchestration-and-presentation`).**

**Known issues, not yet fixed** (both `dashboard/feature_workbench.py` and `dashboard/glm_workbench.py`):
- A freshly agent-proposed variable/term's `approved` field is left `None` (the prompt templates never ask the LLM for it), and the card's `default_checked = bool(feat.approved)` renders that as unchecked — agent-proposed and agent-excluded items look identical on first render.
- Tab placement is agent-response-driven, not actuary-decision-driven: an uncommented uncheck produces no remark, so it can silently revert on the next refine triggered by an unrelated comment. Agreed fix: recompute grouping from the actuary's submitted checkbox state every round; show the agent's original recommendation as a badge instead.
- The GLM Results tab still sources rating factors from the last session-log event rather than a checkpoint — same staleness class the GBM tab had before this session's fix (checkpoint now carries `feature_importances` alongside `interactions`), not yet applied here.

**Next planned work:** `config/feature_seed.yaml` / `config/distillation_seed.yaml` — optional actuary-authored priors (partial feature lists / groupings, and a `commercially_excluded` list for GLM) so an actuary's existing knowledge doesn't need re-deriving every run. Each entry carries a `temperature` (0.0 enforced code-level — never sent to the agent at all, for regulatory/business facts; >0.0 prompt-level flexibility for the agent to propose alternatives) and an `updated_at` staleness timestamp (not full version history, see below).

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
| `config/project_config.yaml` | Data settings, LLM, GBM params, validation params. Feature list and GBM interactions added here after approval (checkpoints). |
| `config/glm_config.yaml` | GLM terms and formula. Populated after distillation gate (checkpoint). |

The actuary can pre-populate either config to skip the agent proposal step.

## Key design decisions

- **GBM-first, not hypothesis-first:** the GBM reveals what the data says; the LLM and actuary then decide what goes into the GLM. More grounded than speculative hypothesis generation.
- **H-statistics only (not SHAP interaction values):** Friedman H-statistics are fast to compute on large datasets. Full SHAP interaction values on 354k rows are prohibitively expensive. H-stats are sufficient for ranking interactions for the distillation agent.
- **Top-N feature cutoff for H-statistics:** only pairs among the top-N features (by LightGBM gain importance) are evaluated. Configurable via `gbm.top_n_features` in project_config.yaml.
- **No hyperparameter tuning (Optuna):** the GBM is an instrument for finding interactions, not the deliverable. Reasonable defaults + early stopping produce correct interaction rankings without the 30+ min tuning overhead.
- **Log-rate target for GBM:** MSE on `log(total_premium / total_exposure)` — the annualised pure premium rate. This dataset stores the **earned (pro-rata) premium**, confirmed empirically: `corr(total_premium, total_exposure) ≈ 0.60` and dividing by exposure reduces the CV (0.77 → 0.57). If a dataset stores the annual tariff premium instead, exposure and premium would be uncorrelated and no division would be needed.
- **Prompts in `prompts/` YAML files:** separated from code, easy to iterate without touching Python. Each file has named sections (`proposal`, `refinement`) used by the corresponding agent. `feature_selection.yaml` and `distillation.yaml` are actively used; `grouping.yaml` exists but the grouping agent uses inline prompts (`GROUPING_PROMPT` / `GROUPING_REFINE_PROMPT` in `agents/grouping_agent.py`).
- **Pairwise interactions only in the GLM:** H-statistics are inherently pairwise, so the distillation agent is restricted to proposing main effects and two-feature interactions. Three-way interactions are not proposed automatically — the actuary can add them via the remark loop if needed.
- **Actuary-in-the-loop at every stage:** feature selection, grouping, and GLM term selection all have an approve/reject/remark gate. Remarks loop back to the LLM for refinement.
- **Checkpoint pattern:** approved decisions are written back to YAML. Re-running skips already-approved stages.
- **Pydantic for all LLM outputs:** malformed JSON surfaces as a clear `ValidationError` immediately.
- **Prompt caching:** system prompt cached in `LLMClient` — reduces cost on repeated calls.
- **Temperature 0.2:** stable enough for actuarial reasoning, slight variation improves proposal diversity.
- **LangGraph-compatible:** all agents are stateless classes with typed inputs/outputs.
- **Post-fit coefficient review gate (GLM):** after the initial fit, the actuary reviews every term's coefficient sign, p-value, and CI. A rejected term is dropped and the model is immediately refit — the loop repeats until a clean pass. This catches suppressor variables and levels with sparse data that the distillation gate (which reviews LLM proposals, not fitted parameters) cannot detect.
- **Rating factors as relativities, not log-coefficients:** `exp(coef)` per parameter is shown as a multiplicative relativity (base = 1.0 for the reference level). This is the direct pricing output actuaries use.
- **No explainerdashboard:** explainerdashboard runs a separate Dash server and doesn't natively support statsmodels GLMs. Diagnostics for the GLM (deviance residuals, Q-Q) and GBM (SHAP summary) will be rendered natively in the Streamlit dashboard using matplotlib/shap, keeping it as a single-pane-of-glass UI.
- **max_tokens=8192:** the distillation agent's JSON response (main effects + interaction terms for all approved features) can exceed 4,000 tokens. The client is set to the model maximum of 8,192.
- **Session logging (JSONL):** every pipeline run writes a JSONL file to `reports/sessions/`. Each event (proposal, decision, remark, GBM metrics, GLM fit) is flushed immediately so partial runs are preserved. This is the data source for the dashboard's Layer 1 read-only views (Audit Trail, and — for now — GLM Results; see "Known issues" above).
- **Dashboard layered build:** Layer 1 is a read-only Streamlit viewer of session logs. Layer 2 (complete, 2026-08-08) replaces terminal gates with Streamlit interactive forms, one workbench per gate, all writing the same checkpoints the CLI reads.
- **Checkpoints over session-log events for anything the dashboard can retrain:** the GBM checkpoint (`gbm_output` in `project_config.yaml`) stores `feature_importances` alongside `interactions` specifically so the dashboard never has to fall back to a possibly-stale session-log event to render current results. Applied to GBM; not yet applied to the GLM's fitted results (see "Known issues").
- **Checkpoint invalidation on feature-set change:** finalizing the Feature Workbench with a different approved feature set (names or categorical groupings, compared via `_feature_signature`) clears any existing `gbm_output`/`glm_config.yaml` terms, since both downstream stages otherwise skip re-running whenever *any* checkpoint exists — a stale feature set could silently survive into training without this.
- **Shared `core/*_pipeline.py` modules:** checkpoint read/write and training/fitting logic for features, GBM, and GLM each live in one module (`core/feature_pipeline.py`, `core/gbm_pipeline.py`, `core/glm_pipeline.py`), used identically by `agents/orchestrator.py` and the dashboard workbenches. `dashboard/_session.py` similarly shares the LLM client, cached dataframe, and session logger across every Layer 2 tab in one browser session, so one dashboard visit produces one coherent session log regardless of which workbenches get used.
