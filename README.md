# Non-Life Pricing Agent

Agentic Python tool that supports actuaries in Non-Life insurance pricing model development.
It distills a GBM into an interpretable GLM, with a human-in-the-loop at every key decision.

## What it does

1. **Feature selection** — an LLM profiles the dataset and proposes which variables to include, with actuarial rationale per variable. The actuary reviews each one, approves, rejects, or leaves a remark. Remarks loop back to the LLM for a revised proposal.
2. **Categorical grouping** — high-cardinality variables (e.g. vehicle brand) are clustered into risk-homogeneous groups by the LLM. The actuary reviews and refines.
3. **GBM training** — LightGBM trains on the approved feature set (MSE on log-rate target). Friedman H-statistics rank pairwise interactions among the top-N most important features. All parameters are configurable in `config/project_config.yaml` under `gbm:`.
4. **Distillation** — the ranked interaction list is sent to the LLM, which proposes which interactions are actuarially defensible GLM terms. The actuary reviews term by term.
5. **GLM fitting** — statsmodels GLM is fitted on the approved terms. A post-fit coefficient review gate lets the actuary reject any term whose sign is wrong or whose p-value is unacceptable; rejected terms are dropped and the model is automatically refit. A rating factors table then shows `exp(coef)` as multiplicative relativities (base level = 1.0), the direct pricing output.

All actuary decisions are saved to YAML checkpoints. Re-running the pipeline skips already-approved stages.

Every run also writes a structured JSONL session log to `reports/sessions/` — capturing every LLM proposal, actuary decision, remark, GBM metric, and GLM fit result. This is the data source for the dashboard.

## Setup

**Prerequisites:** [pyenv](https://github.com/pyenv/pyenv) must be installed. On macOS: `brew install pyenv`.

```bash
pyenv local 3.12.8
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add ANTHROPIC_API_KEY
pytest                 # unit tests, no LLM calls needed
```

## Usage

```python
from agents.orchestrator import Orchestrator
Orchestrator("config/project_config.yaml").run()
```

Edit `config/project_config.yaml` to set the data path, target variable, exposure column, and LLM settings. No code changes needed for a new dataset.

The pipeline is interactive: at each stage the actuary reviews proposals in the terminal (`[A]pprove / [R]eject / [N]ote / [S]kip`). Remarks loop back to the LLM for a revised proposal. Approved decisions are checkpointed to YAML so re-runs skip completed stages.

**Seed configs** (optional): `config/feature_seed.yaml` and `config/distillation_seed.yaml` let an actuary pre-fill known priors — a preferred grouping, a variable that must never appear in the GLM — without skipping the agent proposal entirely. See the `.example.yaml` versions of each for the shape; per-entry `temperature` controls how much license the agent has to deviate (`0.0` is enforced in code, the agent never even sees that entry).

## Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Dark theme by default. Read-only session viewer (Overview, GBM, GLM Results, Audit Trail tabs) plus four interactive workbenches that replace the terminal gates with Streamlit forms — same checkpoints either way, so CLI and dashboard can be used interchangeably run to run:

- **Sidebar pipeline graph** — one node per stage (Feature Selection, GBM Training, GLM Distillation, GLM Fitting), solid and filled once that stage has run, dashed while still pending *or* while stale — has run, but an upstream stage has since finalized/trained a newer version than the one this stage actually used (staleness cascades: one stale ancestor dashes everything built on top of it). Hovering a dashed node names which case applies. Reads the same completion flags as the Audit Trail, plus the same lineage data ("Built from") that view already tracks.
- **Feature & Grouping Workbench** — combined feature selection + categorical grouping in one screen. Cards per variable (Numerical / Categorical / Not Proposed tabs) default checked unless explicitly rejected; checking or unchecking a card and clicking "Update" moves it between tabs immediately — tab placement is always actuary-owned, the agent's response can never move a variable. Each card keeps a comment history (not a single overwritten note): "💾" saves a comment and clears the box with no LLM call, and Claude's replies show inline with its logo, newest first. "Update" sends every not-yet-sent comment to the agent for a revised draft — a card the actuary *didn't* comment on this round keeps its exact description/data-quality-note/grouping regardless of what that round's LLM call returned for it, so only what was actually remarked on ever changes; "Finalize" locks the card view (same layout, checkboxes/comments disabled) and writes the checkpoint — "Re-open" loads it back into an editable draft. Finalizing with a changed feature set auto-invalidates any existing GBM/GLM checkpoints. Every draft (a fresh agent proposal, an Update round, or a Finalize) snapshots to disk under `reports/drafts/{initial,modified,finalized}/`, browsable and restorable from a "Load a saved snapshot" section.
- **GBM tab** — a "(Re)train GBM" button trains LightGBM and computes H-statistics synchronously in-dashboard, no CLI trip needed. A picker lets the actuary train on any finalized feature snapshot, not just the currently active one — picking an older one restores it as current first.
- **GLM Distillation Workbench** — same card/tab/comment/finalize pattern as the feature workbench (Main Effects / Interactions / Not Proposed tabs; unchecking a term moves it to Not Proposed rather than leaving it stranded, checked; each card cross-references which interactions use a main effect, or which main effects an interaction is built from), for GLM main effects and pairwise interactions ranked by the GBM's H-statistics. Gated on a GBM checkpoint existing. Two ways to introduce a term the agent hasn't proposed: a "➕ Add a new interaction" card (pick two included main effects, no LLM call) and a "💬 Message to the agent" box for anything more open-ended. A "GBM run to distill from" picker lets the actuary regenerate from any past GBM run, not just the current one. If a feature was de-approved since the draft was built (e.g. after retraining GBM without it), Re-open, Regenerate, loading a snapshot, and every Update/Finalize all auto-exclude any term still referencing it — it lands in Not Proposed with a note explaining why, rather than silently riding along into a fit formula; re-approving the feature later lifts the exclusion with no special handling needed. Same minimal-diff guarantee as the Feature Workbench: a term not remarked on this round keeps its exact rationale and H-statistic, no matter what that round's LLM call returned for it.
- **GLM coefficient review** (in the GLM Results tab) — a "GLM Distillation version to fit" picker (current checkpoint or any finalized snapshot) feeds the fit; then Keep/Reject cards per term (coefficient, exp(coef), p-value, CI, optional rejection note); rejecting refits automatically until every remaining term is kept. The GLM Results tab's Main Effects/Interactions views show each term's rationale and full comment history (with the Claude logo, not just the current round's note). A "Fit History" table lists every completed fit — Deviance Explained, AIC, term counts, source — since AIC only means something compared against other fits on the same target, not as a single number.
- **Audit Trail → Model Lineage** — traces the current rating factors back through GLM Distillation → GBM run → feature snapshot, one row per stage, each showing which version it was actually built from and that version's position in its own history ("latest", "2nd latest", ...) — so it's immediately clear whether a stage used the newest available upstream input or a deliberately older one.

The sidebar's staleness warning above (dashed for stale, not just pending) covers the pipeline-graph view; the GLM Results tab itself still sources its numbers from the last completed fit regardless of whether it's stale — see the code comment at its "Fit History" table for the distinction.

## Dataset conventions and exposure

How exposure enters the model depends on what is stored in the dataset — and this is dataset-specific:

**Frequency models** (Poisson, claim counts): unambiguous. The target is an integer count of claims, and `E[claims] = exposure × frequency(X)`. Exposure is always an offset: `log(exposure)` added to the linear predictor, or equivalently `init_score = log(exposure)` in LightGBM.

**Pure premium / severity models** (Gamma): depends on what is stored.
- *Earned premium stored* (pro-rata): the premium reflects the actual policy duration. A 6-month policy has roughly half the premium of a full-year policy at the same risk. Exposure and premium will be positively correlated. The annual rate is `total_premium / total_exposure`, and the GBM target is `log(total_premium / total_exposure)`.
- *Annual premium stored* (tariff premium): the premium is the full-year price regardless of duration. Exposure and premium will be uncorrelated. No division by exposure needed; the GBM target is `log(total_premium)` directly, and exposure may be used as an observation weight for credibility.

**This dataset** (motor portfolio, DOI: 10.17632/sw4jmdb2sm.1) stores the **earned premium**: `corr(total_premium, total_exposure) ≈ 0.60`, and dividing by exposure reduces the coefficient of variation (0.77 → 0.57), confirming the pro-rata convention. The GBM therefore targets `log(total_premium / total_exposure)` — the log of the annualised pure premium rate.
