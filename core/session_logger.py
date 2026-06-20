import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SessionLogger:
    """Append-safe JSONL session log for dashboard consumption.

    Each call to ``log()`` writes one JSON record immediately so partial
    runs are preserved on crash or early exit.

    Log files land in ``reports/sessions/session_<run_id>.jsonl``.
    """

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("reports/sessions")
        path.mkdir(parents=True, exist_ok=True)
        self._path = path / f"session_{self.run_id}.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")

    # ------------------------------------------------------------------
    def log(self, event: str, **data: Any) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "SessionLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
