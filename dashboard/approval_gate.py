import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from core.schemas import (
    CategoricalFeatureConfig,
    CategoryCluster,
    FeatureProposal,
    GLMProposal,
    GLMTerm,
    GroupingResponse,
    NumericFeatureConfig,
    ValidationResult,
)

console = Console()


# ── Feature selection gate ────────────────────────────────────────────────────

def run_feature_gate(
    proposal: FeatureProposal,
    agent,                       # FeatureSelectionAgent (avoid circular import)
    objective: str,
    target_col: str,
    exposure_col: str,
) -> FeatureProposal:
    """Feature-by-feature review gate with LLM refinement loop.

    The actuary can approve, reject, or leave a note per feature.
    Any notes are sent back to the LLM for a revised proposal.
    The loop repeats until the actuary finalises with no outstanding remarks.
    """
    session_id = datetime.now().strftime("%Y-%m-%d %H:%M")

    while True:
        remarks: dict[str, str] = {}
        all_features: list[NumericFeatureConfig | CategoricalFeatureConfig] = (
            list(proposal.numeric) + list(proposal.categorical)
        )

        console.rule(f"[bold blue]FEATURE SELECTION GATE – {session_id}[/bold blue]")
        console.print(
            f"{len(proposal.numeric)} numeric  |  "
            f"{len(proposal.categorical)} categorical  |  "
            f"{len(proposal.excluded)} excluded by agent\n"
        )

        if proposal.excluded:
            console.print("[dim]Agent-excluded (not shown for review):[/dim]")
            for col in proposal.excluded:
                reason = proposal.exclusion_rationale.get(col, "")
                console.print(f"  [dim]• {col}: {reason}[/dim]")
            console.print()

        for feat in all_features:
            _display_feature(feat)
            console.print("[bold]\\[A]pprove  \\[R]eject  \\[N]ote  \\[S]kip  \\[Q]uit[/bold]")

            while True:
                choice = input("Decision: ").strip().lower()
                if choice in ("a", "r", "n", "s", "q"):
                    break
                console.print("[red]Enter A, R, N, S, or Q.[/red]")

            if choice == "q":
                console.print("[yellow]Session ended by user.[/yellow]")
                _save_feature_decisions(proposal, session_id)
                return proposal
            elif choice == "a":
                feat.approved = True
                console.print(f"[green]✓ {feat.name}[/green]")
            elif choice == "r":
                feat.approved = False
                console.print(f"[red]✗ {feat.name}[/red]")
            elif choice == "n":
                note = input("Note for agent: ").strip()
                feat.actuary_note = note
                remarks[feat.name] = note
                console.print(f"[yellow]⚑ Note recorded for {feat.name}[/yellow]")
            # "s" → leave status unchanged

        if not remarks:
            console.print("\n[green]No outstanding remarks — feature selection finalised.[/green]")
            break

        console.print(
            f"\n[yellow]Sending {len(remarks)} remark(s) to agent for refinement...[/yellow]"
        )
        proposal = agent.refine(
            previous_proposal=proposal,
            actuary_remarks=remarks,
            objective=objective,
            target_col=target_col,
            exposure_col=exposure_col,
        )
        console.print("[green]Revised proposal ready. Restarting review.[/green]\n")

    _save_feature_decisions(proposal, session_id)
    console.print("[dim]Feature decisions saved to reports/actuary_decisions.csv[/dim]")
    return proposal


def _display_feature(feat: NumericFeatureConfig | CategoricalFeatureConfig) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan bold", width=22)
    table.add_column("Value")

    kind = "Numeric" if isinstance(feat, NumericFeatureConfig) else "Categorical"
    table.add_row("Feature", f"[bold]{feat.name}[/bold]  [{kind}]")
    table.add_row("Description", feat.description)

    if isinstance(feat, CategoricalFeatureConfig):
        if feat.ordinal and feat.order:
            table.add_row("Ordinal order", " → ".join(feat.order))
        table.add_row("Suggested clusters", str(feat.n_clusters))

    if feat.data_quality_note:
        table.add_row("[yellow]Data quality[/yellow]", feat.data_quality_note)
    if feat.actuary_note:
        table.add_row("[yellow]Previous note[/yellow]", feat.actuary_note)

    status = (
        "[green]Approved[/green]" if feat.approved is True
        else "[red]Rejected[/red]" if feat.approved is False
        else "[dim]Pending[/dim]"
    )
    table.add_row("Status", status)

    console.print()
    console.rule(style="dim")
    console.print(table)


def _save_feature_decisions(proposal: FeatureProposal, session_id: str) -> None:
    path = Path("reports/actuary_decisions.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["session", "stage", "name", "type", "decision", "actuary_note"])
        for feat in list(proposal.numeric) + list(proposal.categorical):
            if feat.approved is not None:
                writer.writerow([
                    session_id, "feature_selection", feat.name,
                    "numeric" if isinstance(feat, NumericFeatureConfig) else "categorical",
                    "approved" if feat.approved else "rejected",
                    feat.actuary_note or "",
                ])


# ── GLM distillation gate ─────────────────────────────────────────────────────

def run_glm_gate(
    proposal: GLMProposal,
    agent,                       # DistillationAgent (avoid circular import)
    objective: str,
    target_col: str,
    exposure_col: str,
) -> GLMProposal:
    """Term-by-term review gate for the GLM distillation phase."""
    session_id = datetime.now().strftime("%Y-%m-%d %H:%M")

    while True:
        remarks: dict[str, str] = {}

        console.rule(f"[bold blue]GLM DISTILLATION GATE – {session_id}[/bold blue]")
        console.print(f"{len(proposal.terms)} terms to review\n")

        for term in proposal.terms:
            _display_glm_term(term)
            console.print("[bold]\\[A]pprove  \\[R]eject  \\[N]ote  \\[S]kip  \\[Q]uit[/bold]")

            while True:
                choice = input("Decision: ").strip().lower()
                if choice in ("a", "r", "n", "s", "q"):
                    break
                console.print("[red]Enter A, R, N, S, or Q.[/red]")

            if choice == "q":
                console.print("[yellow]Session ended by user.[/yellow]")
                _save_glm_decisions(proposal, session_id)
                return proposal
            elif choice == "a":
                term.approved = True
                console.print(f"[green]✓ {term.name}[/green]")
            elif choice == "r":
                term.approved = False
                console.print(f"[red]✗ {term.name}[/red]")
            elif choice == "n":
                note = input("Note for agent: ").strip()
                term.actuary_note = note
                remarks[term.name] = note
                console.print(f"[yellow]⚑ Note recorded for {term.name}[/yellow]")

        if not remarks:
            console.print("\n[green]No outstanding remarks — GLM terms finalised.[/green]")
            break

        console.print(
            f"\n[yellow]Sending {len(remarks)} remark(s) to agent for refinement...[/yellow]"
        )
        proposal = agent.refine(
            previous_proposal=proposal,
            actuary_remarks=remarks,
            objective=objective,
            target_col=target_col,
            exposure_col=exposure_col,
        )
        console.print("[green]Revised proposal ready. Restarting review.[/green]\n")

    _save_glm_decisions(proposal, session_id)
    console.print("[dim]GLM decisions saved to reports/actuary_decisions.csv[/dim]")
    return proposal


def _display_glm_term(term: GLMTerm) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan bold", width=22)
    table.add_column("Value")

    table.add_row("Term", f"[bold]{term.name}[/bold]  [{term.term_type}]")
    if term.h_statistic is not None:
        table.add_row("H-statistic", f"{term.h_statistic:.4f}")
    table.add_row("Rationale", term.rationale)
    if term.actuary_note:
        table.add_row("[yellow]Previous note[/yellow]", term.actuary_note)

    status = (
        "[green]Approved[/green]" if term.approved is True
        else "[red]Rejected[/red]" if term.approved is False
        else "[dim]Pending[/dim]"
    )
    table.add_row("Status", status)

    console.print()
    console.rule(style="dim")
    console.print(table)


def _save_glm_decisions(proposal: GLMProposal, session_id: str) -> None:
    path = Path("reports/actuary_decisions.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["session", "stage", "name", "type", "decision", "actuary_note"])
        for term in proposal.terms:
            if term.approved is not None:
                writer.writerow([
                    session_id, "glm_distillation", term.name, term.term_type,
                    "approved" if term.approved else "rejected",
                    term.actuary_note or "",
                ])


# ── Grouping gate ─────────────────────────────────────────────────────────────

def run_grouping_gate(
    col_name: str,
    response: GroupingResponse,
    agent,                       # GroupingAgent (avoid circular import)
    df: pd.DataFrame,
    exposure_col: str,
    n_clusters: int,
    claim_freq_col: str | None = None,
) -> GroupingResponse:
    """Cluster-by-cluster review gate for one categorical variable.

    The actuary can approve or note each cluster; notes loop back to the LLM
    for a revised proposal until no outstanding remarks remain.
    """
    session_id = datetime.now().strftime("%Y-%m-%d %H:%M")

    while True:
        remarks: dict[str, str] = {}

        console.rule(f"[bold blue]GROUPING GATE: {col_name} – {session_id}[/bold blue]")
        console.print(
            f"{len(response.clusters)} clusters proposed  |  {n_clusters} target\n"
        )

        for cluster in response.clusters:
            _display_cluster(cluster)
            console.print("[bold]\\[A]pprove  \\[N]ote  \\[S]kip  \\[Q]uit[/bold]")

            while True:
                choice = input("Decision: ").strip().lower()
                if choice in ("a", "n", "s", "q"):
                    break
                console.print("[red]Enter A, N, S, or Q.[/red]")

            if choice == "q":
                console.print("[yellow]Session ended by user.[/yellow]")
                _save_grouping_decisions(col_name, response, session_id)
                return response
            elif choice == "a":
                console.print(f"[green]✓ {cluster.cluster_name}[/green]")
            elif choice == "n":
                note = input("Note for agent: ").strip()
                remarks[cluster.cluster_name] = note
                console.print(f"[yellow]⚑ Note recorded for {cluster.cluster_name}[/yellow]")
            # "s" → no opinion, leave as-is

        if not remarks:
            console.print(f"\n[green]Grouping for {col_name} finalised.[/green]")
            break

        console.print(
            f"\n[yellow]Sending {len(remarks)} remark(s) to agent for refinement...[/yellow]"
        )
        response = agent.refine(
            df=df,
            col_name=col_name,
            exposure_col=exposure_col,
            n_clusters=n_clusters,
            previous_response=response,
            actuary_remarks=remarks,
            claim_freq_col=claim_freq_col,
        )
        console.print("[green]Revised grouping ready. Restarting review.[/green]\n")

    _save_grouping_decisions(col_name, response, session_id)
    console.print("[dim]Grouping decisions saved to reports/actuary_decisions.csv[/dim]")
    return response


def _display_cluster(cluster: CategoryCluster) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan bold", width=22)
    table.add_column("Value")

    table.add_row("Cluster", f"[bold]{cluster.cluster_name}[/bold]")
    table.add_row("Elements", ", ".join(cluster.elements))
    table.add_row("Rationale", cluster.rationale)

    console.print()
    console.rule(style="dim")
    console.print(table)


def _save_grouping_decisions(
    col_name: str, response: GroupingResponse, session_id: str
) -> None:
    path = Path("reports/actuary_decisions.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["session", "stage", "name", "type", "decision", "actuary_note"])
        for cluster in response.clusters:
            writer.writerow([
                session_id, "grouping", col_name,
                cluster.cluster_name,
                "approved",
                ", ".join(cluster.elements),
            ])


# ── Hypothesis validation gate (Phase 1, kept for reference) ──────────────────

def run_approval_gate(results: list[ValidationResult]) -> list[ValidationResult]:
    """Interactive CLI approval gate for hypothesis validation results."""
    if not results:
        return []

    session_id = datetime.now().strftime("%Y-%m-%d %H:%M")
    console.rule(f"[bold blue]ACTUARY APPROVAL GATE – Session {session_id}[/bold blue]")

    approved: list[ValidationResult] = []

    for result in results:
        h = result.hypothesis
        _display_result(result)
        console.print("\n[bold]\\[A]pprove  \\[R]eject  \\[S]kip  \\[Q]uit[/bold]\n")

        while True:
            choice = input("Decision: ").strip().lower()
            if choice in ("a", "r", "s", "q"):
                break
            console.print("[red]Invalid. Enter A, R, S, or Q.[/red]")

        if choice == "q":
            console.print("[yellow]Session ended by user.[/yellow]")
            break
        elif choice == "a":
            result.approved = True
            approved.append(result)
            console.print(f"[green]✓ Approved: {h.new_feature_name}[/green]")
        elif choice == "r":
            result.approved = False
            console.print(f"[red]✗ Rejected: {h.new_feature_name}[/red]")

    _save_decisions(results, session_id)
    return approved


def _display_result(result: ValidationResult) -> None:
    h = result.hypothesis
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="cyan bold", width=18)
    table.add_column("Value")

    table.add_row("Feature", f"[bold]{h.new_feature_name}[/bold]")
    table.add_row("Formula", f"{h.feature_a}  {h.operation}  {h.feature_b}")
    table.add_row("Rationale", h.rationale)
    delta_color = "green" if result.deviance_delta_pct < 0 else "red"
    table.add_row("Deviance Δ", f"[{delta_color}]{result.deviance_delta_pct:+.3f}%[/{delta_color}]")
    table.add_row("Gain Rank", f"#{result.gain_rank}")

    console.print("\n")
    console.rule(style="dim")
    console.print(table)


def _save_decisions(results: list[ValidationResult], session_id: str) -> None:
    path = Path("reports/actuary_decisions.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["session", "stage", "name", "type", "decision", "actuary_note"]
            )
        for r in results:
            if r.approved is not None:
                h = r.hypothesis
                writer.writerow([
                    session_id, "hypothesis_validation",
                    h.new_feature_name,
                    f"{h.feature_a} {h.operation} {h.feature_b}",
                    "approved" if r.approved else "rejected",
                    "",
                ])
