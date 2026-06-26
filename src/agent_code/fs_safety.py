from __future__ import annotations

import pathspec
from dataclasses import dataclass, field
from pathlib import Path

# Text-file suffix allowlist: pass through, no need to peek headers.
TEXT_SUFFIXES = {
    ".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json",
    ".cfg", ".ini", ".env", ".sh", ".bash", ".zsh", ".js", ".ts", ".tsx",
    ".jsx", ".html", ".css", ".sql", ".lock", ".gitignore",
}
MAX_READ_BYTES = 256 * 1024     # single file read limit, to prevent OOM.
DEFAULT_MAX_CHARS = 8000        # single observation char limit, to prevent flooding the model with too much info.
DEFAULT_SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "dist", "build",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
})

@dataclass
class SkipPolicy:
    skip_dirs: frozenset[str] = DEFAULT_SKIP_DIRS
    gitignore: pathspec.PathSpec | None = None

    @classmethod
    def default(cls, gitignore: pathspec.PathSpec | None = None) -> SkipPolicy:
        return cls(gitignore = gitignore)

@dataclass
class ReadFileState:
    # avoid overwriting the latest file with stale content
    entries: dict[Path, tuple[int, int]] = field(default_factory = dict)

    def record(self, path: Path, content: str) -> None:
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return
        self.entries[path] = (mtime_ns, len(content))

def resolve_in_cwd(cwd: Path, user_path: str) -> Path:
    # resolve the user path against the cwd, preventing path traversal outside of cwd
    resolved_path = (cwd / user_path).resolve()
    resolved_cwd = cwd.resolve()
    if not resolved_path.is_relative_to(resolved_cwd):
        raise ValueError("Access to paths outside of the CWD is not allowed.")
    return resolved_path

def ensure_text_file(path: Path) -> None:
    # Allowlisted suffix passes; otherwise peek 1 KB — a NUL byte means binary.
    if path.suffix.lower() in TEXT_SUFFIXES:
        return
    with path.open("rb") as f:
        if b"\x00" in f.read(1024):
            raise ValueError(f"binary file: {path.name}")
        
def ensure_within_size(path: Path, max_bytes: int = MAX_READ_BYTES) -> None:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"file too large: {path.name} ({path.stat().st_size} bytes > {max_bytes} bytes)")
        
def should_skip(rel_path: Path, policy: SkipPolicy) -> bool:
    # Check if any part of the relative path is in the skip_dirs, or if it matches gitignore patterns.
    if any(part in policy.skip_dirs for part in rel_path.parts):
        return True
    if policy.gitignore and policy.gitignore.match_file(str(rel_path)):
        return True
    return False
        
def truncate_output(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[truncated {len(text) - max_chars} chars]"

def load_gitignore(cwd: Path) -> pathspec.PathSpec | None:
    gitignore_path = cwd / ".gitignore"
    if not gitignore_path.exists():
        return None
    lines = gitignore_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)

def ensure_read_before_edit(state: ReadFileState, path: Path) -> str | None:
    # Has the file been read this session? Return an error string if not.
    if path not in state.entries:
        return f"Error: File {path} has not been read before edit."
    return None

def check_mtime_conflict(state: ReadFileState, path: Path) -> str | None:
    # Check if the file has been modified since it was last read. mtime changed = conflict.
    # No content-equals fallback in this version - re-read to refresh
    entry = state.entries.get(path)
    if entry is None:
        return None # never-read case first
    old_mtime_ns, _ = entry
    try:
        cur_mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return None
    if cur_mtime_ns > old_mtime_ns:
        return f"Error: File {path} has been modified since it was last read."
    return None

def apply_single_replace(content: str, old: str, new: str, replace_all: bool) -> tuple[str | None, str | None]:
    # find old content and replace with new content
    # Returns (new_content, error): success -> error is None; failure -> new_content is None.
    if old == "":
        return None, "Error: Old content must not be empty."
    if old == new:
        return None, "Error: Old content and new content are exactly the same."
    
    count = content.count(old)
    if count == 0:
        return None, f"Error: content to replace not found in file."
    if not replace_all and count > 1:
        return None, f"Error: Multiple occurrences found, {count} matches but replace_all is False."
    new_content = content.replace(old, new) if replace_all else content.replace(old, new, 1)
    return new_content, None