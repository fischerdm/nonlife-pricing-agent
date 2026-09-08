"""Run-scoped config/reports layout.

Config (`project_config.yaml`, `glm_config.yaml`, optional seed YAMLs) and
`reports/` live under a single top-level `<name>/` folder per dataset +
target configuration, e.g.:

    <name>/
      config/
        project_config.yaml
        glm_config.yaml
        feature_seed.yaml        # optional
        distillation_seed.yaml   # optional
      reports/
        sessions/
        drafts/{initial,modified,finalized}/
        gbm_model.txt
        actuary_decisions.csv

A run switches to a different dataset/target by switching which `<name>/`
is active — never by editing config/reports in place. `active_run.yaml`,
at the repo root (a sibling of every `<name>/`, so it's findable before any
name is known), holds a single field naming the active run:

    name: motor_portfolio_20260620_103855

`<name>` is a human-chosen, human-typed string — there is no dashboard UI,
auto-naming, or auto-detection here. `default_config_path()` is the only
function that consults `active_run.yaml`; every other function below takes
an explicit `config_path` and never touches the pointer, so the same
derivation works identically for the active run, a deliberately reopened
old run, or a test's `tmp_path` config.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


class RunConfigError(Exception):
    """Raised for anything that makes the active/named run unusable."""


# ── Repo-root-anchored paths ─────────────────────────────────────────────────

def repo_root() -> Path:
    return Path(__file__).parent.parent


def active_run_pointer_path() -> Path:
    return repo_root() / "active_run.yaml"


def get_active_run_name() -> str:
    """Read `active_run.yaml`'s `name` field. Raises `RunConfigError` with an
    actionable message if the pointer file is missing or malformed."""
    pointer = active_run_pointer_path()
    if not pointer.exists():
        raise RunConfigError(
            f"{pointer} not found — create a run first, e.g. "
            f"core.run_scope.create_run(\"my_dataset\")."
        )
    raw = yaml.safe_load(pointer.read_text()) or {}
    name = raw.get("name")
    if not name:
        raise RunConfigError(f"{pointer} is missing its 'name' field.")
    return name


def default_config_path() -> Path:
    """`project_config.yaml` for the currently active run — the only function
    here that consults `active_run.yaml`."""
    return repo_root() / get_active_run_name() / "config" / "project_config.yaml"


# ── Paths derived from an explicit config_path (no global state) ────────────

def run_root(config_path: Path) -> Path:
    """`<name>/`, given `<name>/config/project_config.yaml` (or any sibling
    config file in the same directory)."""
    return config_path.parent.parent


def reports_dir(config_path: Path) -> Path:
    return run_root(config_path) / "reports"


def sessions_dir(config_path: Path) -> Path:
    return reports_dir(config_path) / "sessions"


def drafts_dir(config_path: Path) -> Path:
    return reports_dir(config_path) / "drafts"


def decisions_log_path(config_path: Path) -> Path:
    return reports_dir(config_path) / "actuary_decisions.csv"


def gbm_model_path(config_path: Path, configured_name: str = "gbm_model.txt") -> Path:
    """Resolve a configured `gbm.model_path` value under the run's own
    reports/ dir — only the filename component of `configured_name` is used,
    so a stray directory prefix left over in project_config.yaml (e.g. the
    pre-run-scoping `reports/gbm_model.txt`) can never resolve against the
    process's CWD instead of the active run."""
    return reports_dir(config_path) / Path(configured_name).name


# ── Run management ───────────────────────────────────────────────────────────

_TEMPLATE_NAME = "project_config.example.yaml"


def create_run(
    label: str,
    template: Path | None = None,
    *,
    dataset_path: str | None = None,
    sep: str | None = None,
    target_col: str | None = None,
    exposure_col: str | None = None,
    objective: str | None = None,
) -> str:
    """Scaffold a brand-new `<label>_<timestamp>/` run, copy `template` in as
    its starting `project_config.yaml`, and make it the active run.

    Refuses to overwrite: the timestamp suffix makes a collision with an
    existing folder practically impossible, but this still asserts rather
    than silently reusing one, as a defense-in-depth backstop. Returns the
    new run's full name (label + timestamp) — use it with `open_run` later
    to switch back.

    `dataset_path`/`sep`/`target_col`/`exposure_col`/`objective` are optional
    overrides for the copied template's `data:` block — each one given
    replaces that key; each one omitted (None) is left as whatever the
    template already has (its own placeholder, or a real default if the
    template was hand-edited to carry one). None of these are required —
    a bare `create_run(label)` still works exactly as before, still expecting
    the actuary to hand-edit `data:` afterward.
    """
    template = template or (repo_root() / "config" / _TEMPLATE_NAME)
    if not template.exists():
        raise RunConfigError(f"Template not found: {template}")

    name = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = repo_root() / name
    if run_dir.exists():
        raise RunConfigError(f"{run_dir} already exists — refusing to overwrite.")

    (run_dir / "config").mkdir(parents=True)
    (run_dir / "reports" / "sessions").mkdir(parents=True)
    for kind in ("initial", "modified", "finalized"):
        (run_dir / "reports" / "drafts" / kind).mkdir(parents=True)

    config_path = run_dir / "config" / "project_config.yaml"
    shutil.copy(template, config_path)

    # dataset_path -> data.path: "dataset_path" is the clearer name at this
    # call site, but the config key it fills in is still just data.path —
    # same semantics, never assumed to live under a fixed data/ directory.
    overrides = {
        "path": dataset_path, "sep": sep, "target_col": target_col,
        "exposure_col": exposure_col, "objective": objective,
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if overrides:
        config = yaml.safe_load(config_path.read_text()) or {}
        config.setdefault("data", {}).update(overrides)
        config_path.write_text(yaml.dump(config, allow_unicode=True, sort_keys=False))

    _write_active_run(name)
    return name


def open_run(name: str) -> None:
    """Point `active_run.yaml` at an existing `<name>/`. Raises `RunConfigError`
    naming the missing piece if the run or its `project_config.yaml` isn't there."""
    config_path = repo_root() / name / "config" / "project_config.yaml"
    if not config_path.exists():
        raise RunConfigError(
            f"{config_path} not found — did you mean to call create_run({name!r}) instead?"
        )
    _write_active_run(name)


def _write_active_run(name: str) -> None:
    active_run_pointer_path().write_text(yaml.dump({"name": name}, sort_keys=False))


# ── Startup validation ───────────────────────────────────────────────────────

_REQUIRED_DATA_KEYS = ("path", "target_col", "exposure_col", "objective")
_SEED_SECTIONS = ("numeric", "categorical")


def validate_config(config_path: Path | None = None) -> list[str]:
    """Fail fast on anything that makes the run unusable; return a list of
    non-fatal warnings for anything recoverable.

    Raises `RunConfigError` if: the run folder is missing, `project_config.yaml`
    is missing/unparseable, a required `data.*` key is absent, or the dataset
    file it names doesn't exist. Returns (doesn't raise for) one kind of
    warning today: a `feature_seed.yaml`/`distillation_seed.yaml` entry naming
    a column that isn't actually in the dataset — reads only the CSV header
    (`nrows=0`), not the full file, to stay cheap even on a large dataset.
    """
    if config_path is None:
        config_path = default_config_path()

    run_dir = run_root(config_path)
    if not run_dir.exists():
        raise RunConfigError(f"Run folder '{run_dir}' does not exist.")
    if not config_path.exists():
        raise RunConfigError(
            f"{config_path} not found — did you mean to call create_run() or open_run()?"
        )

    config = yaml.safe_load(config_path.read_text()) or {}
    data_cfg = config.get("data", {})
    for key in _REQUIRED_DATA_KEYS:
        if key not in data_cfg:
            raise RunConfigError(f"{config_path}: missing required 'data.{key}'.")

    dataset_path = Path(data_cfg["path"])
    if not dataset_path.exists():
        raise RunConfigError(
            f"{config_path}: data.path '{dataset_path}' does not exist — "
            f"edit data.path (and sep/target_col/exposure_col/objective as needed) "
            f"to point at your dataset."
        )

    columns = set(
        pd.read_csv(dataset_path, sep=data_cfg.get("sep", ","), nrows=0).columns
    )

    # target_col/exposure_col are only checked for *presence as a key* above —
    # a leftover template placeholder (or a typo) would otherwise sail through
    # silently instead of failing here, exactly the mistake a fresh create_run
    # without --target-col/--exposure-col overrides is prone to.
    for key in ("target_col", "exposure_col"):
        if data_cfg[key] not in columns:
            raise RunConfigError(
                f"{config_path}: data.{key} '{data_cfg[key]}' is not a column in {dataset_path.name}."
            )

    warnings: list[str] = []
    for filename in ("feature_seed.yaml", "distillation_seed.yaml"):
        seed_path = config_path.parent / filename
        if not seed_path.exists():
            continue
        seed_raw = yaml.safe_load(seed_path.read_text()) or {}
        for section in _SEED_SECTIONS:
            for entry in seed_raw.get(section, []) or []:
                col = entry.get("name")
                if col and col not in columns:
                    warnings.append(f"{filename}: '{col}' is not a column in {dataset_path.name}")
    return warnings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage run-scoped config/reports folders.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Scaffold a new <label>_<timestamp>/ run.")
    p_create.add_argument("label")
    p_create.add_argument("--dataset-path", dest="dataset_path", help="data.path — CSV file to model")
    p_create.add_argument("--sep", help="data.sep — CSV field delimiter")
    p_create.add_argument("--target-col", dest="target_col", help="data.target_col")
    p_create.add_argument("--exposure-col", dest="exposure_col", help="data.exposure_col")
    p_create.add_argument("--objective", help="data.objective, e.g. gamma")

    p_open = sub.add_parser("open", help="Switch the active run to an existing <name>/.")
    p_open.add_argument("name")

    sub.add_parser("validate", help="Validate the currently active run.")

    args = parser.parse_args()
    if args.command == "create":
        new_name = create_run(
            args.label, dataset_path=args.dataset_path, sep=args.sep,
            target_col=args.target_col, exposure_col=args.exposure_col, objective=args.objective,
        )
        print(f"Created and activated run: {new_name}")
        remaining = [
            f"data.{k}" for k, v in (
                ("path", args.dataset_path), ("sep", args.sep), ("target_col", args.target_col),
                ("exposure_col", args.exposure_col), ("objective", args.objective),
            ) if v is None
        ]
        if remaining:
            print(f"Next: edit {new_name}/config/project_config.yaml's {', '.join(remaining)}, then run the app.")
        else:
            print("All data fields were provided — run `python -m core.run_scope validate` to confirm, then run the app.")
    elif args.command == "open":
        open_run(args.name)
        print(f"Activated run: {args.name}")
    elif args.command == "validate":
        issues = validate_config()
        if issues:
            print("Warnings:")
            for w in issues:
                print(f"  - {w}")
        else:
            print("OK — no issues found.")
