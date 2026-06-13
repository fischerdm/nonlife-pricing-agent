# Agentic AI – Non-Life Pricing: Project Plan & Architecture

> **Handoff document for Claude Code**
> Language: English (code, docs, comments, variable names — everything)
> Status: Ready for implementation

---

## 1. Project Vision

Build a modular, agentic Python framework that automates the most time-consuming actuarial tasks in Non-Life pricing model development:

1. **Hypothesis generation** – LLM proposes statistically and actuarially justified feature interactions
2. **Categorical grouping** – LLM clusters high-cardinality categorical variables into risk-meaningful groups
3. **Automated validation loop** – Python tests each LLM hypothesis against a LightGBM benchmark
4. **Actuary dashboard** – Human-in-the-loop approval gate before any feature enters the final model
5. **GBM-to-GLM distillation** – Dedicated agent pipeline to extract interpretable GLM structure from a GBM

The framework is a **private side project** and should be self-contained, runnable locally, and designed for eventual integration into a larger agentic pricing platform (LangGraph-based).

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Pricing Analyst                          │
│              (provides: data path, target, config)              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                            │
│         Reads config, dispatches sub-agents, collects           │
│         results, manages the actuary approval gate              │
└──────┬──────────────────┬──────────────────────────┬────────────┘
       │                  │                          │
       ▼                  ▼                          ▼
┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐
│  Hypothesis  │  │  Categorical     │  │   GBM Agent           │
│  Agent       │  │  Grouping Agent  │  │   (train, SHAP,       │
│              │  │                  │  │    H-statistic,       │
│  LLM → JSON  │  │  LLM → mapping   │  │    PDP/ICE)           │
│  interactions│  │  dict → pandas   │  └──────────┬────────────┘
└──────┬───────┘  └──────┬───────────┘             │
       │                 │                          ▼
       ▼                 ▼               ┌───────────────────────┐
┌──────────────────────────────────┐     │   Distillation Agent  │
│       Validation Engine          │     │                       │
│  - Build feature (multiply/div)  │     │  GBM insights → GLM   │
│  - Train LightGBM with/without   │◄────│  interaction finder   │
│  - Compare Poisson/Gamma deviance│     │  GLM spec builder     │
│  - Rank by gain + deviance delta │     └──────────┬────────────┘
└──────────────────┬───────────────┘                │
                   │                                │
                   ▼                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Actuary Approval Gate                          │
│                                                                  │
│  Feature Name │ LLM Rationale │ Deviance Δ │ Gain Rank │ Action │
│  ─────────────┼───────────────┼────────────┼───────────┼─────── │
│  age_x_power  │ "Young driver"│  -0.41%    │    12     │ ✓ / ✗  │
│  ...          │ ...           │  ...       │   ...     │ ✓ / ✗  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
nonlife-pricing-agent/
│
├── README.md
├── pyproject.toml                  # dependencies (uv or pip)
├── .env.example                    # ANTHROPIC_API_KEY placeholder
│
├── config/
│   └── project_config.yaml         # data path, target col, exposure col,
│                                   # LLM model, temperature, max hypotheses
│
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py             # top-level agent: dispatches, collects
│   ├── hypothesis_agent.py         # LLM → interaction hypotheses
│   ├── grouping_agent.py           # LLM → categorical cluster mappings
│   ├── gbm_agent.py                # train GBM, SHAP, H-stat, PDP
│   └── distillation_agent.py       # GBM → GLM interaction extraction
│
├── core/
│   ├── __init__.py
│   ├── llm_client.py               # thin wrapper around Anthropic SDK
│   ├── feature_builder.py          # builds interaction columns in DataFrame
│   ├── validator.py                # LightGBM A/B test, deviance delta
│   └── schemas.py                  # Pydantic models for all LLM outputs
│
├── tools/
│   ├── __init__.py
│   ├── shap_tools.py               # SHAP values, interaction values
│   ├── glm_tools.py                # statsmodels GLM wrapper
│   └── reporting.py                # Markdown/CSV report generator
│
├── dashboard/
│   └── approval_gate.py            # CLI or Streamlit approval interface
│
├── tests/
│   ├── test_hypothesis_agent.py
│   ├── test_grouping_agent.py
│   ├── test_validator.py
│   └── fixtures/
│       └── sample_features.json
│
└── notebooks/
    └── 01_exploration.ipynb        # scratch for manual testing
```

---

## 4. Core Components – Detailed Specification

### 4.1 `core/schemas.py` — Pydantic Models

All LLM outputs are validated through Pydantic before any downstream code runs. This prevents silent failures and makes the API contract explicit.

```python
from pydantic import BaseModel, field_validator
from typing import Literal

class FeatureMetadata(BaseModel):
    name: str
    dtype: Literal["numeric", "categorical"]
    description: str

class InteractionHypothesis(BaseModel):
    feature_a: str
    feature_b: str
    operation: Literal["multiply", "divide", "ratio_a_over_b", "ratio_b_over_a"]
    new_feature_name: str
    rationale: str  # actuarial justification from LLM

class HypothesisResponse(BaseModel):
    hypotheses: list[InteractionHypothesis]

class CategoryCluster(BaseModel):
    cluster_name: str
    elements: list[str]
    rationale: str

class GroupingResponse(BaseModel):
    clusters: list[CategoryCluster]

class ValidationResult(BaseModel):
    hypothesis: InteractionHypothesis
    deviance_delta_pct: float       # negative = improvement
    gain_rank: int                  # feature importance rank in LightGBM
    baseline_deviance: float
    new_deviance: float
    approved: bool | None = None    # set by actuary
```

### 4.2 `core/llm_client.py` — Anthropic Wrapper

```python
import json
from anthropic import Anthropic
from typing import Type
from pydantic import BaseModel

class LLMClient:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514",
                 temperature: float = 0.2):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def call(self, prompt: str, response_model: Type[BaseModel]) -> BaseModel:
        """Call Claude and parse response into a Pydantic model."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        raw_text = response.content[0].text
        # Strip potential markdown code fences
        clean = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
        return response_model.model_validate(json.loads(clean))
```

**Design principles:**
- Single method `call()` — no complexity hidden inside the client
- Temperature default `0.2` — deterministic enough for actuarial reasoning, not zero (allows slight variation for diverse hypothesis sets)
- Pydantic validation is the safety net; if parsing fails, a clear error surfaces immediately

### 4.3 `agents/hypothesis_agent.py` — Interaction Hypothesis Generator

**Prompt design (key decisions from Gemini conversation):**
- Pass only feature **metadata**, never raw data (cost + privacy)
- Explicitly ask for actuarial/physical/behavioral justification
- Constrain output count (e.g., top 5) to control API cost per run
- Request JSON-only response, no preamble

```python
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
```

### 4.4 `agents/grouping_agent.py` — Categorical Clustering

**Key design decisions (from Gemini conversation):**
- Send `(value, count)` pairs — never raw rows
- Let analyst specify `n_clusters` in config
- Include option to provide historical claim frequencies as additional signal
- Fallback: unmapped values → `"Other_Residual"` group

```python
GROUPING_PROMPT = """
You are a senior actuary. Group these categorical values of the variable '{col_name}' 
into exactly {n_clusters} risk-homogeneous clusters for a Non-Life pricing model.

Data (value, exposure count{freq_hint}):
{values_json}

Base groupings on actuarial risk logic — driving behavior, occupational exposure, 
vehicle characteristics, etc. — NOT alphabetical or arbitrary similarity.

Respond ONLY with valid JSON — no preamble:
{{
  "clusters": [
    {{
      "cluster_name": "<SNAKE_CASE_NAME>",
      "elements": ["value1", "value2"],
      "rationale": "<actuarial justification>"
    }}
  ]
}}
"""
```

**Python mapping after LLM response:**
```python
def build_mapping_dict(grouping_response: GroupingResponse) -> dict[str, str]:
    mapping = {}
    for cluster in grouping_response.clusters:
        for element in cluster.elements:
            mapping[element] = cluster.cluster_name
    return mapping

# Apply to DataFrame
df["occupation_grouped"] = df["occupation_raw"].map(mapping_dict).fillna("Other_Residual")
```

### 4.5 `core/validator.py` — LightGBM A/B Test

For each hypothesis:
1. Build the interaction column in a copy of the DataFrame
2. Train LightGBM **with** the new feature (full feature set)
3. Compare against baseline model **without** the new feature
4. Report: deviance delta %, feature gain rank

```python
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

class Validator:
    def __init__(self, config: dict):
        self.lgb_params = {
            "objective": config.get("objective", "poisson"),  # or "gamma"
            "metric": "poisson",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
        }
        self.n_rounds = config.get("n_rounds", 300)

    def validate(self, df, feature_cols, target_col,
                 exposure_col, new_feature_col) -> dict:
        """Train baseline and candidate model, return metrics."""
        X_base = df[feature_cols]
        X_cand = df[feature_cols + [new_feature_col]]
        y = df[target_col]
        w = df[exposure_col]  # exposure weights

        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X_cand, y, w, test_size=0.2, random_state=42
        )

        # Baseline (without new feature)
        base_dev = self._train_and_score(
            X_train[feature_cols], y_train, w_train,
            X_val[feature_cols], y_val, w_val
        )

        # Candidate (with new feature)
        cand_dev, gain_rank = self._train_and_score(
            X_train, y_train, w_train,
            X_val, y_val, w_val,
            return_importance=True,
            target_col=new_feature_col
        )

        return {
            "baseline_deviance": base_dev,
            "candidate_deviance": cand_dev,
            "deviance_delta_pct": (cand_dev - base_dev) / base_dev * 100,
            "gain_rank": gain_rank,
        }
```

### 4.6 `agents/distillation_agent.py` — GBM → GLM

This is the more complex agent. Pipeline:

1. **GBM Agent** trains model, computes SHAP interaction values and H-statistics
2. **Distillation Agent** receives ranked interaction list → asks LLM to confirm which are actuarially justifiable → builds GLM formula
3. **Actuary gate** approves final GLM terms

```python
# Distillation Agent prompt (simplified)
DISTILLATION_PROMPT = """
You are a senior actuary reviewing the output of a Gradient Boosting Machine 
for a {lob} pricing model.

The model identified the following pairwise interactions (ranked by H-statistic):
{interactions_json}

Your tasks:
1. Select the interactions that are actuarially justifiable and regulatorily defensible.
2. For each selected interaction, specify whether it should enter the GLM as:
   - A product term (feature_a * feature_b)
   - A ratio term (feature_a / feature_b)  
   - A binned interaction (segment feature_a into bands first)
3. Flag any interaction that appears spurious or data-driven only.

Respond ONLY with valid JSON:
{{
  "approved_interactions": [...],
  "rejected_interactions": [...],
  "glm_formula_terms": ["feature_a:feature_b", ...]
}}
"""
```

---

## 5. Configuration (`config/project_config.yaml`)

```yaml
data:
  path: "data/training_data.parquet"
  target_col: "claim_count"          # or claim_amount
  exposure_col: "exposure_years"
  objective: "poisson"               # poisson | gamma | tweedie

features:
  numeric:
    - name: "driver_age"
      description: "Age of youngest driver"
    - name: "vehicle_power_kw"
      description: "Vehicle engine power in kW"
    - name: "vehicle_age"
      description: "Age of vehicle in years"
  categorical:
    - name: "occupation"
      description: "Policyholder occupation"
      n_clusters: 4
    - name: "regional_class"
      description: "Geographic risk region"
      n_clusters: 5

llm:
  model: "claude-sonnet-4-20250514"
  temperature: 0.2
  max_hypotheses: 5                  # number of interactions to generate per run

validation:
  objective: "poisson"
  n_rounds: 300
  min_deviance_improvement_pct: 0.1  # threshold: below this → auto-reject
  min_exposure_per_cluster: 500      # minimum exposure for categorical groups

output:
  report_path: "reports/"
  approval_log: "reports/actuary_decisions.csv"
```

---

## 6. Actuary Approval Gate (`dashboard/approval_gate.py`)

### Phase 1: CLI (implement first)

```
╔══════════════════════════════════════════════════════════════════╗
║              ACTUARY APPROVAL GATE – Session 2024-12            ║
╠══════════════════════════════════════════════════════════════════╣
║ Feature              │ Rationale               │ Δ Dev  │ Rank  ║
╠══════════════════════════════════════════════════════════════════╣
║ driver_age_x_kw      │ Young driver in high-   │ -0.41% │   12  ║
║                      │ powered vehicle creates │        │       ║
║                      │ disproportionate risk   │        │       ║
╠══════════════════════════════════════════════════════════════════╣
║ [A]pprove  [R]eject  [S]kip  [Q]uit                            ║
╚══════════════════════════════════════════════════════════════════╝
```

### Phase 2: Streamlit dashboard (optional, post-MVP)

Simple web interface with approve/reject buttons and notes field. Saves decisions to `actuary_decisions.csv`.

---

## 7. Implementation Phases

### Phase 1 — Core Pipeline (MVP)

**Goal:** End-to-end working loop for hypothesis generation and validation.

| Task | File | Priority |
|------|------|----------|
| Pydantic schemas | `core/schemas.py` | P0 |
| Anthropic wrapper | `core/llm_client.py` | P0 |
| Feature builder | `core/feature_builder.py` | P0 |
| LightGBM validator | `core/validator.py` | P0 |
| Hypothesis agent | `agents/hypothesis_agent.py` | P0 |
| CLI approval gate | `dashboard/approval_gate.py` | P0 |
| Orchestrator (basic) | `agents/orchestrator.py` | P0 |
| Config loader | `config/` + YAML | P0 |

### Phase 2 — Categorical Grouping

| Task | File | Priority |
|------|------|----------|
| Grouping agent | `agents/grouping_agent.py` | P1 |
| Volume check + fallback | inside grouping agent | P1 |
| Optional: claim freq as signal | config flag | P1 |

### Phase 3 — GBM/GLM Distillation

| Task | File | Priority |
|------|------|----------|
| GBM agent (train + SHAP) | `agents/gbm_agent.py` | P2 |
| H-statistic computation | `tools/shap_tools.py` | P2 |
| Distillation agent | `agents/distillation_agent.py` | P2 |
| GLM builder | `tools/glm_tools.py` | P2 |

### Phase 4 — Polish

| Task | Priority |
|------|----------|
| Streamlit approval dashboard | P3 |
| Markdown/HTML report generator | P3 |
| Full test suite | P3 |
| LangGraph integration (future) | P4 |

---

## 8. Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **Anthropic Claude (API)** | Superior reasoning for domain-specific logic; structured JSON output; pay-per-token (no idle EC2 cost) |
| **Pydantic for all LLM output** | Silent parsing failures are unacceptable in a regulated context; Pydantic surfaces errors immediately |
| **Feature metadata → LLM, never raw data** | Cost control + data privacy; LLM needs column names and descriptions, not individual rows |
| **Temperature 0.2 (not 0.0)** | Fully deterministic (0.0) can produce repetitive hypotheses across runs; slight variation improves coverage |
| **Exposure-weighted LightGBM** | Industry standard for frequency/severity modeling; validates hypotheses under the actual modeling objective |
| **Human gate before GLM** | Regulatory requirement in most markets; architecture makes this non-optional, not an afterthought |
| **Modular agents, not a monolith** | Each agent can be tested, improved, or replaced independently; hypothesis agent ≠ distillation agent |
| **LangGraph-compatible design** | Agents are stateless functions with typed inputs/outputs → trivial to wrap in LangGraph nodes later |

---

## 9. Dependencies (`pyproject.toml`)

```toml
[project]
name = "nonlife-pricing-agent"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "anthropic>=0.30.0",
    "pydantic>=2.7.0",
    "lightgbm>=4.3.0",
    "shap>=0.45.0",
    "statsmodels>=0.14.0",    # for GLM
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "scikit-learn>=1.5.0",
    "pyyaml>=6.0.0",
    "python-dotenv>=1.0.0",
    "rich>=13.0.0",            # CLI formatting
]

[project.optional-dependencies]
dashboard = ["streamlit>=1.35.0"]
dev = ["pytest>=8.0.0", "ruff>=0.4.0"]
```

---

## 10. Immediate Next Steps for Claude Code

Start with these files **in this order**:

1. `pyproject.toml` + `.env.example`
2. `core/schemas.py` — all Pydantic models
3. `core/llm_client.py` — Anthropic wrapper
4. `config/project_config.yaml` — with example values
5. `core/feature_builder.py` — interaction column construction
6. `core/validator.py` — LightGBM A/B test
7. `agents/hypothesis_agent.py` — prompt + LLM call + schema validation
8. `dashboard/approval_gate.py` — CLI gate
9. `agents/orchestrator.py` — wire everything together
10. `tests/` — at minimum test schemas and validator

**Do NOT start with the distillation agent** — that requires a working GBM pipeline first. Build and test phases 1 and 2 before touching the GBM/GLM components.

---

## 11. What Was Taken from the Gemini Conversation

The following ideas from the Gemini chat were incorporated directly:

- ✅ **Feature metadata as LLM input** (not raw data) — adopted as core constraint
- ✅ **JSON-only LLM responses** — enforced via Pydantic at the schema layer
- ✅ **`(value, count)` pairs for categorical clustering** — adopted in grouping agent design
- ✅ **Actuary dashboard with Approved/Rejected columns** — implemented as approval gate
- ✅ **`pandas.map()` for applying cluster mappings** — standard pattern in `grouping_agent.py`
- ✅ **`fillna("Other_Residual")` fallback** — handles LLM omissions gracefully
- ✅ **`temperature=0.2`** for stable, reproducible groupings — configured in LLM client

The following was **intentionally not adopted**:
- ❌ Gemini suggested OpenAI as alternative — project uses Anthropic exclusively
- ❌ Manual web UI workflow for hypothesis generation — project uses API from day one
