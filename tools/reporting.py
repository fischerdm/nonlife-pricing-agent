"""Phase 4 — Markdown/HTML report generator for the actuary dashboard."""

from __future__ import annotations

from pathlib import Path

from core.schemas import ValidationResult


def generate_markdown_report(
    results: list[ValidationResult], output_path: str = "reports/session_report.md"
) -> Path:
    raise NotImplementedError("Phase 4 — implement Markdown report generator")


def generate_html_report(
    results: list[ValidationResult], output_path: str = "reports/session_report.html"
) -> Path:
    raise NotImplementedError("Phase 4 — implement HTML report generator")
