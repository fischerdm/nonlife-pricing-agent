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
| Shared GLM distillation draft-generation logic (CLI + dashboard, one impl) | **Done** | `core/distillation_pipeline.py` |
| Seed configs (`feature_seed.yaml` / `distillation_seed.yaml`) | **Done** | `core/seed_config.py`, `agents/feature_selection_agent.py`, `agents/distillation_agent.py` |

**Full CLI pipeline proven end-to-end (2026-06-20). Streamlit dashboard Layer 2 completed for all four actuary gates (2026-08-08, branch `orchestration-and-presentation`). Seed configs completed (2026-08-08, branch `seed-config`). Feature & Grouping Workbench review loop redesigned after live testing (2026-08-15, branch `seed-config`) — see below. Finalized checkpoint snapshots and actuary/agent comment history added (2026-08-30, branch `finalized-snapshots-and-comment-history`) — see below. GLM Distillation Workbench ported to the same pattern, plus a GBM finalized-version picker (2026-08-30, branch `distillation-workbench-redesign`) — see below. Cross-stage lineage (GBM run history, a version picker at every training/fitting stage, an Audit Trail "Model Lineage" view) and several GLM Results/Audit Trail cleanups added on the same branch (2026-08-30 through 2026-09-05) — see below. Dashboard visual redesign — dark theme, icons, sidebar pipeline graph, a Fit History table (2026-09-06, branch `dashboard-layout`) — see below.**

**Feature & Grouping Workbench (`dashboard/feature_workbench.py`) — fixed 2026-08-15, was "not yet fixed" below for weeks:**
- Numeric/categorical cards now default checked unless explicitly rejected (`feat.approved is not False`), not `bool(feat.approved)` — a freshly-proposed-but-undecided variable no longer renders identically to a rejected one.
- Tab placement (numeric/categorical/excluded) is purely actuary-owned: `core/feature_pipeline.py::reconcile_membership()` recomputes it from the submitted checkbox state on every Update or Finalize, applied both before *and* after any agent refine call — the agent structurally cannot move a variable regardless of what it returns, not just prompt-instructed not to.
- Finalize now shows the same card layout locked (checkboxes/comments `disabled=True`) instead of a plain table; "Revise current selection" renamed "Re-open".

**Extended 2026-08-30 — finalized snapshots + comment history:**
- Every draft snapshots to disk under `reports/drafts/{initial,modified,finalized}/` — `initial` only from an explicit "Regenerate from scratch" (LLM output isn't deterministic, so this is the only way back to a specific past agent take), `modified` after every Update round or saved comment, `finalized` once per Finalize (`project_config.yaml` only ever holds the *current* checkpoint — this is the full history). Browsable via a "Load a saved snapshot" section (three dropdowns), which warns before discarding an unsaved in-progress draft.
- Comments accumulate instead of overwriting: `NumericFeatureConfig`/`CategoricalFeatureConfig` gained `comment_history` (`core/schemas.py::CommentEntry`), shown newest-first above each card's comment box, labeled "👤 Actuary" / "Claude" (with the icon at `icons/claude-ai.svg`, see `icons/README.md` for provenance). A per-card "💾 save comment" `st.form_submit_button` appends and clears the box immediately with no LLM call; Update/Finalize collect every not-yet-`sent` entry as that variable's remark. `actuary_note` is now purely transient (the agent's reply for the current round) — `core/feature_pipeline.py::refine_draft` merges it into history and clears it. Legacy checkpoints/snapshots with only a bare `actuary_note` migrate automatically on load (`_migrate_legacy_note`) rather than losing that content.
- Defense-in-depth lesson from this pass: an early version of the refine prompt sent every variable's `comment_history` as an empty `[]`, which was enough for the model to try "helpfully" filling it with a hallucinated flat string, failing validation. Fixed two ways — the prompt now omits the key entirely (`agents/feature_selection_agent.py::_proposal_dict_for_prompt`), and the schemas gained a defensive `field_validator` that discards malformed `comment_history` content rather than failing the whole response. Same pattern as the seed-config locks: prompt-level reduction *and* a code-level guarantee, never one alone.

**GLM Distillation Workbench (`dashboard/glm_workbench.py`) — ported to the Feature Workbench pattern, 2026-08-30:**
- Same checkbox-default fix: `term.approved is not False`, not `bool(term.approved)`.
- Tab placement (Main Effects vs. Interactions) is data/actuary-owned, not agent-owned: `core/distillation_pipeline.py::reconcile_terms()` derives `term_type` structurally from the term's own name (`":"`-joined → interaction, else main) on every Update/Finalize, applied both before *and* after any agent refine call. The one deliberate exception is `"polynomial"`, which only survives for a term the actuary explicitly remarked on that round — an unprompted agent reclassification never sticks. `approved` is likewise always forced from the submitted checkbox state, same defense-in-depth as `reconcile_membership`.
- Same locked-view-on-finalize, "Re-open", and three-kind (`initial`/`modified`/`finalized`) disk-backed snapshot browser as the Feature Workbench — sharing `reports/drafts/`'s kind subfolders via a `glm_draft_` filename prefix (`core/distillation_pipeline.py::save_glm_draft_snapshot` et al.) so the two workbenches' pickers never mix up each other's history.
- Same comment-history mechanic: `GLMTerm` gained `comment_history` (`core/schemas.py::CommentEntry`), rendered via a new shared `dashboard/_comments.py` (`render_comment_history`, `claude_logo_data_uri`) — extracted out of `feature_workbench.py` so both workbenches use one implementation instead of two copies of the icon-loading/history-rendering logic. `core/distillation_pipeline.py::refine_glm_draft` does the same carry-forward/mark-sent/append-agent-reply/clear-transient-note bookkeeping as `feature_pipeline.py::refine_draft`.
- New: two ways to introduce an interaction the agent hasn't proposed, since (unlike the Feature Workbench, where every dataset column already has a card) there's no pre-existing card for an interaction nobody has suggested yet. A "➕ Add a new interaction" card at the bottom of the Interactions tab — two dropdowns over currently-included main effects, appends instantly via `core/distillation_pipeline.py::add_manual_interaction`, no LLM call, mirroring how unchecking a feature costs nothing in the Feature Workbench. And a general "💬 Message to the agent" box at the top of the form, for anything more open-ended than naming two known features — sent as `general_remark` (`DistillationAgent.refine`'s new parameter, threaded into `prompts/distillation.yaml`'s `refinement` section) alongside, but kept distinct from, the per-term `actuary_remarks` dict.

**GLM Distillation Workbench — fixed and extended further, 2026-08-30/09-05:**
- **Widget-key collision, found via live Playwright testing:** `feature_workbench.py` and `glm_workbench.py` used the identical, unprefixed key pattern (`comment_{name}_{round}`, `iter{N}_include/save_{name}`) for their card widgets. Streamlit executes every tab's body on every script run regardless of which is visible, and a GLM main-effect term is always named after its underlying feature — so this collided (`StreamlitDuplicateElementKey`) on the very first render of any main effect. Fixed by giving each workbench its own prefix (`wb_*` / `glm_*`).
- **Third tab, "Not Proposed":** lists approved features with no main-effect term at all, GBM-ranked pairs (from the `gbm_output` checkpoint) never proposed as an interaction, *and* — the actual tab-placement fix — any term the actuary unchecked. `_render_cards` sorts by `approved` before `term_type`, so unchecking a term moves it here instead of leaving it, still unchecked, in its original tab; the full term (rationale, comment history, H-statistic) survives the move intact, unlike the Feature Workbench's excluded tab which collapses to a bare column name. Re-checking it there (no LLM call) restores it on the next Update.
- **Cross-reference notes on every card:** a Main Effects card notes which interactions (if any) currently use it; an Interactions card always names its two constituent main effects and additionally flags one that's currently excluded. Both recomputed fresh every render.

**GBM training (`dashboard/gbm_workbench.py`) — finalized-version picker added, 2026-08-30:** training now runs against one explicitly-picked finalized feature checkpoint rather than implicitly whatever's current in `project_config.yaml`. A selectbox lists every `reports/drafts/finalized/` snapshot (same history the Feature Workbench writes on every Finalize) plus a default "Current checkpoint" option; picking an older snapshot restores it as the active checkpoint first (`feature_pipeline.py::save_feature_checkpoint`, same invalidation semantics as re-finalizing the workbench itself) and then trains on it.

**Cross-stage lineage — 2026-08-30/09-05, closing the gap the paragraph above used to describe as deferred:** every stage now runs against an explicitly-picked version of its upstream input, and the Audit Trail makes the resulting chain visible:
- **GBM has no actuary review loop** (every Train/Retrain is immediately usable, unlike Feature Selection/GLM Distillation), so rather than a parallel snapshot-file history, `core/gbm_pipeline.py::list_gbm_runs()` reads GBM's run history back from the session log's own `gbm_complete` events, which already carry everything needed (`interactions`, `feature_importances`).
- The GLM Distillation Workbench's locked view gained a "GBM run to distill from" picker (mirroring the GBM tab's own feature-set picker) — only affects "Regenerate from scratch"; a historical pick restores it as the active `gbm_output` checkpoint first.
- The GLM Fitting step (`glm_coef_workbench.py`) gained a "GLM Distillation version to fit" picker, same current-vs-finalized-snapshot pattern.
- Each stage logs which upstream version it actually used — `feature_source`/`gbm_source`/`distillation_source`, each a `{"kind", "ts"[, "label", "path"]}` dict; `core/snapshot_utils.py` (`snapshot_ts`, `format_ts`) is the one shared place that parses a snapshot filename's or session-log event's timestamp into the plain, comparable form used for this.
- The Audit Trail's "Model Lineage" section (`dashboard/streamlit_app.py`) shows one row per completed stage (GBM Training, GLM Distillation (finalized), GLM Fit) with a "Built from" column in the consistent format `"<Upstream Stage>: <timestamp> (<position>)"`. **The position, not `"(current)"`, is the actual information:** `"current"` always coincides with `"latest"` by construction (Finalize and every picker's own restore branch are the only writers, and both always make the picked version the new current one) — so it told the actuary nothing they couldn't already infer. `"latest"` / `"2nd latest"` / `"3rd latest"` (looked up by position against that stage's own newest-first history) answers the question that matters: was this deliberately built from an older version, and how far back. Falls back gracefully (blank) if the referenced version has since been deleted from its history.
- The old "Stage" decision-log block (`feature_decision`/`grouping_decision`/`glm_term_decision`/`glm_coef_decision`) was removed from the Audit Trail entirely (2026-09-05) — the first three event types are logged *only* by the terminal `approval_gate.py` flow, never by the Streamlit workbenches (only `glm_coef_decision` is shared between both), so this table only ever showed the original CLI proof-of-concept session's data regardless of any dashboard-driven work. If CLI-driven per-decision auditing is needed again, it needs reworking against the current data model (snapshots + comment history), not a revival of this table.
- The GLM Results tab's Main Effects/Interactions views were also changed from a plain dataframe to a card list (reusing `dashboard/_comments.py::render_comment_history`) so a term's notes show the real Claude logo instead of a 🤖 emoji placeholder — a dataframe cell can't render inline HTML/images at all. The column itself was renamed "Notes" (was "Actuary Note") and now sources the union of `comment_history` and any lingering transient `actuary_note`, instead of just the latter (which goes blank the instant a refine round runs — the actual notes for a term finalized with zero Update rounds are the distillation agent's own initial-proposal caveats, only ever written for interactions, never main effects — that's why Main Effects notes are empty by design, not staleness). The "Relativity Chart" sub-tab was removed — it plotted exactly the `exp(coef)` values already in the Rating Factors table, just as a scatter plot.

**Dashboard visual redesign — 2026-09-06, branch `dashboard-layout`:** pure UI/UX pass, no pipeline logic changes.
- Dark theme by default: `.streamlit/config.toml` pins `[theme] base = "dark"`. Trade-off: this removes Streamlit's own System/Light/Dark picker from the `⋮` menu — a configured theme is fixed, not switchable at runtime.
- Emoji swapped for Material Symbols icons (`:material/monitoring:`, `:material/model_training:`, `:material/functions:`, `:material/refresh:`) via `page_icon=`/`icon=` params.
- Sidebar "Pipeline Stages" tick list replaced with a vertical node/connector graph (`render_pipeline_graph`) — solid filled node + solid connector for a completed stage, dashed hollow node + dashed connector for a pending one, driven by the same session-log completion flags the Audit Trail reads. Categorical Grouping dropped from the graph (4 nodes, not 5) at the actuary's request.
- Fixed a comment-field alignment bug in both `feature_workbench.py` and `glm_workbench.py`: the comment history and comment box were rendering full-width, flush with the checkbox column instead of nested under the description column — `_feature_card`/`_render_term_card` now wrap that tail in `with c2:`.
- Rating Factors table: "Relativity" renamed to "Coefficient" (still `exp(coef)`); pastel color-coding by magnitude removed — a coefficient's practical size depends on the feature's level too, not the coefficient alone.
- The lone "AIC" tile (Overview and GLM Results) replaced with a "Fit History" table in GLM Results: AIC only means something next to other AICs from models fit on the same target, so every completed fit (each `rating_factors` event, logged only once the coefficient-review loop fully converges) now gets a row — Deviance Explained, AIC, actual term counts, and which Distillation snapshot it was built from. `_fmt_ts`/`_fmt_built_from` and the three snapshot-history lists were hoisted out of the Audit Trail block to module level so this table and "Model Lineage" share one implementation.
- Fixed a term-counting bug found while building that table: counting *fitted coefficient rows* (one per dummy-coded level) rather than formula terms made a single interaction term between two categoricals look like "2 interactions" and inflated main-effect counts. `_param_base_term()` strips each side's `[T.<level>]` suffix before rejoining and deduping, so counts now match `glm_config.yaml`'s own term list.

**Known issues, not yet fixed:**
- The GLM Results tab still sources rating factors from the last session-log event rather than a checkpoint — same staleness class the GBM tab had before an earlier session's fix (checkpoint now carries `feature_importances` alongside `interactions`), not yet applied here.
- **Generalized 2026-09-06:** the Fit History table above made this concrete — a fit run before 31 more interactions were later approved in Distillation showed as silently mismatched against the current config, with nothing flagging it. This generalizes to every stage (GBM vs. approved features, GLM Distillation vs. its GBM run, GLM Fit vs. approved terms), not just this one tab. Scoped into its own future feature branch rather than patched here — see the memory note `staleness_detection_deferred`.

**Next planned work:** cross-stage staleness detection (its own feature branch, not yet started — see `staleness_detection_deferred`): warn when a stage's last completed run no longer matches its current upstream approved input, building on the version-tracking (`feature_source`/`gbm_source`/`distillation_source`) already logged by the cross-stage lineage work above. Also: an `updated_at` staleness badge in both workbenches (seed configs already carry the timestamp — just not surfaced in the UI yet); origin badges (🔒 actuary / 🤖 agent / 🚫 agent-against). The GLM distillation workbench's "+" add-interaction and general-remark boxes are a first pass at the actuary-driven-interaction UX — expect further iteration on the exact interaction once it's used live (e.g. whether the general remark should get its own comment-history treatment instead of being one-shot). Full pipeline versioning (tag a finalize, branch/compare cascades) is still deliberately deferred — the version *pickers* built this session are a step toward it, not a substitute (see the memory note `pipeline_versioning_deferred`).

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
| `config/feature_seed.yaml` *(optional)* | Actuary-authored priors for a fresh feature/grouping draft. See `config/feature_seed.example.yaml`. |
| `config/distillation_seed.yaml` *(optional)* | Actuary-authored `commercially_excluded` list for GLM main effects. See `config/distillation_seed.example.yaml`. |

The actuary can pre-populate `project_config.yaml`/`glm_config.yaml` to skip the agent
proposal step entirely, or use the two seed files above to prime — not skip — a
fresh proposal with known priors (see "Key design decisions" below).

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
- **Shared `core/*_pipeline.py` modules:** checkpoint read/write and training/fitting logic for features, GBM, and GLM each live in one module (`core/feature_pipeline.py`, `core/gbm_pipeline.py`, `core/glm_pipeline.py`, `core/distillation_pipeline.py`), used identically by `agents/orchestrator.py` and the dashboard workbenches. `dashboard/_session.py` similarly shares the LLM client, cached dataframe, and session logger across every Layer 2 tab in one browser session, so one dashboard visit produces one coherent session log regardless of which workbenches get used.
- **Seed configs (`core/seed_config.py`):** optional, hand-edited YAML priors (`feature_seed.yaml`, `distillation_seed.yaml`) distinct from the checkpoint configs above — a seed is a stable, rarely-changing input, not an accumulated/invalidated result. Per-entry `temperature` (0.0–1.0) governs how much license the *agent* has to deviate, never the actuary's own override power: `temperature == 0.0` is enforced in `FeatureSelectionAgent`/`DistillationAgent` at the code level — the column/feature is stripped from what's profiled and sent to the LLM before the call, so it structurally cannot be reconsidered, not just prompt-instructed not to be. `temperature > 0.0` stays visible to the LLM with a prompt note on how much license it has to suggest an alternative. An actuary remark on a locked entry during a refine round lets it through for that round only — the lock restrains the agent's *unprompted* behavior, never an explicit actuary instruction. No seed-authoring UI yet (hand-edit the YAML, same as pre-populating `project_config.yaml`) and no `updated_at` staleness badge in the dashboard yet (see "Next planned work").
- **Comment history is code-owned, never the LLM's to populate:** `NumericFeatureConfig`/`CategoricalFeatureConfig.comment_history` accumulates across rounds; `actuary_note` is a transient single field the LLM writes its reply into for the current round only, immediately folded into history and cleared by `core/feature_pipeline.py::refine_draft`. The refine prompt never shows the model `comment_history` at all — same rationale as the seed-config locks, applied to a new surface: don't just ask the model not to touch something, structurally keep it from seeing the field in the first place, and defend with a schema-level validator in case a future prompt change reintroduces it.
- **Snapshot kinds are declared, never inferred:** `initial`/`modified`/`finalized` under `reports/drafts/` are written by exactly the action that produces that kind of draft (a fresh LLM call, an Update round, a Finalize) — there's no logic anywhere that guesses a snapshot's kind after the fact from its content.
