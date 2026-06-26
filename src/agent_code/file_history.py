from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def backup(cwd: Path, path: Path, old_content: str) -> Path | None:
    """Before any write, snapshot the file's old content to .agent/history/<rel>/<ts>.
    Backup is not a tool — the model can't see it. It's a harness-wide safety net.
    A backup failure does not block the edit; returns None."""
    try:
        rel = path.resolve().relative_to(cwd.resolve())
    except ValueError:
        return None  # Path is outside cwd; don't back it up.

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"
    backup_dir = cwd / ".agent" / "history" / rel
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / ts

    try:
        backup_path.write_text(old_content, encoding="utf-8")
    except OSError:
        return None
    return backup_path