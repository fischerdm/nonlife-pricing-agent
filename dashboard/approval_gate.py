import csv
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from core.schemas import ValidationResult

console = Console()


def run_approval_gate(results: list[ValidationResult]) -> list[ValidationResult]:
    """Interactive CLI approval gate. Returns the list of actuary-approved results."""
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
    console.print(f"\n[dim]Decisions saved to reports/actuary_decisions.csv[/dim]")
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
                ["session", "feature", "formula", "rationale", "deviance_delta_pct",
                 "gain_rank", "decision"]
            )
        for r in results:
            if r.approved is not None:
                h = r.hypothesis
                writer.writerow([
                    session_id,
                    h.new_feature_name,
                    f"{h.feature_a} {h.operation} {h.feature_b}",
                    h.rationale,
                    f"{r.deviance_delta_pct:+.3f}",
                    r.gain_rank,
                    "approved" if r.approved else "rejected",
                ])
