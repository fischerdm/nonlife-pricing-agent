"""Shared snapshot/run timestamp formatting, used everywhere a stage's history
gets displayed or compared: feature and GLM Distillation snapshot filenames
(`reports/drafts/<kind>/<prefix>_<timestamp>_<micro>.yaml`) and GBM run
timestamps (a session-log event's own `ts` field, ISO-formatted). Both end up
in the same plain "YYYY-MM-DD HH:MM:SS" display format so a lineage reference
can be looked up by position against either kind of history list.
"""
from datetime import datetime
from pathlib import Path


def snapshot_ts(path: Path, prefix: str) -> str:
    """A snapshot file's own timestamp, parsed from its filename."""
    stem = path.stem.removeprefix(prefix)
    try:
        return datetime.strptime(stem[:15], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return stem


def format_ts(ts: str | None) -> str:
    """Format a raw ISO timestamp (e.g. a session-log event's `ts`) the same
    way `snapshot_ts` formats a filename's — directly comparable to it."""
    return ts[:19].replace("T", " ") if ts else "—"
