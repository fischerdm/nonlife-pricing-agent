# Non-Life Pricing Agent

Agentic Python tool that supports actuaries in Non-Life insurance pricing model development.
It distills a GBM into an interpretable GLM, with a human-in-the-loop at every key decision.

## What it does

1. **Feature selection** — an LLM profiles the dataset and proposes which variables to include, with actuarial rationale per variable. The actuary reviews each one, approves, rejects, or leaves a remark. Remarks loop back to the LLM for a revised proposal.
2. **Categorical grouping** — high-cardinality variables (e.g. vehicle brand) are clustered into risk-homogeneous groups by the LLM. The actuary reviews and refines.
3. **GBM training** — LightGBM trains on the approved feature set.
4. **Distillation** — SHAP interaction values and H-statistics rank pairwise interactions. The LLM proposes which interactions are actuarially defensible GLM terms. The actuary reviews term by term.
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
