# Non-Life Pricing Agent

Agentic Python tool that supports actuaries in Non-Life insurance pricing model development.
It distills a GBM into an interpretable GLM, with a human-in-the-loop at every key decision.

## What it does

1. **Feature selection** — an LLM profiles the dataset and proposes which variables to include, with actuarial rationale per variable. The actuary reviews each one, approves, rejects, or leaves a remark. Remarks loop back to the LLM for a revised proposal.
2. **Categorical grouping** — high-cardinality variables (e.g. vehicle brand) are clustered into risk-homogeneous groups by the LLM. The actuary reviews and refines.
3. **GBM training** — LightGBM trains on the approved feature set (MSE on log-rate target). Friedman H-statistics rank pairwise interactions among the top-N most important features. All parameters are configurable in `config/project_config.yaml` under `gbm:`.
4. **Distillation** — the ranked interaction list is sent to the LLM, which proposes which interactions are actuarially defensible GLM terms. The actuary reviews term by term.
5. **GLM fitting** — statsmodels GLM is fitted on the approved terms.

All actuary decisions are saved to YAML checkpoints. Re-running the pipeline skips already-approved stages.

## Setup

**Prerequisites:** [pyenv](https://github.com/pyenv/pyenv) must be installed. On macOS: `brew install pyenv`.

```bash
pyenv local 3.12.8
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # add ANTHROPIC_API_KEY
```

## Usage

```python
from agents.orchestrator import Orchestrator
Orchestrator("config/project_config.yaml").run()
```

Edit `config/project_config.yaml` to set the data path, target variable, exposure column, and LLM settings. No code changes needed for a new dataset.

## Dataset conventions and exposure

How exposure enters the model depends on what is stored in the dataset — and this is dataset-specific:

**Frequency models** (Poisson, claim counts): unambiguous. The target is an integer count of claims, and `E[claims] = exposure × frequency(X)`. Exposure is always an offset: `log(exposure)` added to the linear predictor, or equivalently `init_score = log(exposure)` in LightGBM.

**Pure premium / severity models** (Gamma): depends on what is stored.
- *Earned premium stored* (pro-rata): the premium reflects the actual policy duration. A 6-month policy has roughly half the premium of a full-year policy at the same risk. Exposure and premium will be positively correlated. The annual rate is `total_premium / total_exposure`, and the GBM target is `log(total_premium / total_exposure)`.
- *Annual premium stored* (tariff premium): the premium is the full-year price regardless of duration. Exposure and premium will be uncorrelated. No division by exposure needed; the GBM target is `log(total_premium)` directly, and exposure may be used as an observation weight for credibility.

**This dataset** (motor portfolio, DOI: 10.17632/sw4jmdb2sm.1) stores the **earned premium**: `corr(total_premium, total_exposure) ≈ 0.60`, and dividing by exposure reduces the coefficient of variation (0.77 → 0.57), confirming the pro-rata convention. The GBM therefore targets `log(total_premium / total_exposure)` — the log of the annualised pure premium rate.
