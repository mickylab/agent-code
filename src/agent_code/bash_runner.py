from __future__ import annotations
import os
import subprocess
from pathlib import Path

from .fs_safety import truncate_output

# Minimal env for bash — don't leak host secrets into the child process
_MINIMAL_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "HOME": os.environ.get("HOME", ""),
    "USER": os.environ.get("USER", ""),
    "SHELL": os.environ.get("SHELL", "/bin/bash"),
}


def run_sync(command: str, cwd: Path, timeout: int = 30) -> str:
    """Run a shell command synchronously. cwd is pinned; the process is killed on timeout."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            env=_MINIMAL_ENV,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    output = proc.stdout.decode("utf-8", errors="replace")
    if proc.stderr:
        stderr_text = proc.stderr.decode("utf-8", errors="replace")
        if stderr_text.strip():
            output += "\n[stderr]\n" + stderr_text
    truncated = truncate_output(output.strip(), max_chars=12000)
    if proc.returncode != 0:
        return f"exit code {proc.returncode}\n{truncated}"
    return truncated if truncated else "(no output)"