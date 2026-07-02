from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Iterable

from .errors import PermissionDenied, ToolError


IGNORED_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".next",
    "target",
}

BINARY_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".class",
    ".jar",
    ".wasm",
    ".pyc",
}


@dataclass(slots=True)
class Workspace:
    """Safe file operations rooted inside one repository."""

    root: Path | str

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        if not self.root.exists():
            raise ToolError(f"workspace root does not exist: {self.root}")
        if not self.root.is_dir():
            raise ToolError(f"workspace root is not a directory: {self.root}")

    def resolve(self, rel_path: str | Path) -> Path:
        p = Path(rel_path)
        if p.is_absolute():
            candidate = p.resolve()
        else:
            candidate = (self.root / p).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as e:
            raise PermissionDenied(f"path escapes workspace: {rel_path}") from e
        return candidate

    def rel(self, path: str | Path) -> str:
        return str(self.resolve(path).relative_to(self.root)).replace(os.sep, "/")

    def exists(self, rel_path: str | Path) -> bool:
        return self.resolve(rel_path).exists()

    def is_probably_text(self, path: Path) -> bool:
        if path.suffix.lower() in BINARY_EXTS:
            return False
        try:
            sample = path.read_bytes()[:2048]
        except OSError:
            return False
        return b"\x00" not in sample

    def iter_files(self) -> Iterable[Path]:
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
            for name in filenames:
                p = Path(dirpath) / name
                if p.suffix.lower() in BINARY_EXTS:
                    continue
                yield p

    def ls(self, rel_path: str | None = None, default_dir: str | None = None) -> list[dict[str, str]]:
        target = rel_path or default_dir or "."
        path = self.resolve(target)
        if path.is_file():
            path = path.parent
        if not path.exists():
            raise ToolError(f"path does not exist: {target}")
        if not path.is_dir():
            raise ToolError(f"path is not a directory: {target}")
        items: list[dict[str, str]] = []
        for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name in IGNORED_DIRS:
                continue
            kind = "dir" if child.is_dir() else "file"
            items.append({"path": self.rel(child), "type": kind})
        return items

    def search(self, content: str, max_results: int = 20) -> list[dict[str, object]]:
        if not content:
            raise ToolError("search content must not be empty")
        results: list[dict[str, object]] = []
        needle = content
        for p in self.iter_files():
            if not self.is_probably_text(p):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            count = text.count(needle)
            if count:
                results.append({"path": self.rel(p), "count": count})
        results.sort(key=lambda x: (-int(x["count"]), str(x["path"])))
        return results[:max_results]

    def create_file(self, rel_path: str) -> str:
        path = self.resolve(rel_path)
        if path.exists():
            raise ToolError(f"file already exists: {rel_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return self.rel(path)

    def delete_file(self, rel_path: str) -> str:
        path = self.resolve(rel_path)
        if not path.exists():
            raise ToolError(f"file does not exist: {rel_path}")
        if not path.is_file():
            raise ToolError(f"not a file: {rel_path}")
        path.unlink()
        return self.rel(path)

    def read_lines(self, rel_path: str, start_line: int, offset: int) -> str:
        if start_line < 1:
            raise ToolError("start_line must be >= 1")
        if offset < 1:
            raise ToolError("offset must be >= 1")
        path = self.resolve(rel_path)
        if not path.exists() or not path.is_file():
            raise ToolError(f"file does not exist: {rel_path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start_idx = start_line - 1
        end_idx = min(len(lines), start_idx + offset)
        selected = lines[start_idx:end_idx]
        if not selected:
            return f"<no lines in requested range; file has {len(lines)} lines>"
        width = max(4, len(str(end_idx)))
        return "\n".join(
            f"{i:>{width}}: {line}"
            for i, line in zip(range(start_line, end_idx + 1), selected)
        )

    def write_lines(self, rel_path: str, start_line: int, end_line: int, content: str) -> str:
        """Replace inclusive [start_line, end_line]. Empty range allowed.

        For an empty file or insertion before line N, use end_line = start_line - 1.
        To append to a file with L lines, use start_line = L + 1 and end_line = L.
        """
        path = self.resolve(rel_path)
        if not path.exists() or not path.is_file():
            raise ToolError(f"file does not exist: {rel_path}")
        if start_line < 1:
            raise ToolError("start_line must be >= 1")
        if end_line < start_line - 1:
            raise ToolError("end_line must be >= start_line - 1")

        old_text = path.read_text(encoding="utf-8", errors="replace")
        had_trailing_newline = old_text.endswith("\n")
        old_lines = old_text.splitlines()
        line_count = len(old_lines)
        if start_line > line_count + 1:
            raise ToolError(
                f"start_line {start_line} is past EOF; file has {line_count} lines"
            )
        if end_line > line_count:
            raise ToolError(f"end_line {end_line} is past EOF; file has {line_count} lines")

        new_lines = content.splitlines()
        start_idx = start_line - 1
        end_idx_exclusive = end_line
        merged = old_lines[:start_idx] + new_lines + old_lines[end_idx_exclusive:]

        new_text = "\n".join(merged)
        if merged and (content.endswith("\n") or had_trailing_newline):
            new_text += "\n"
        path.write_text(new_text, encoding="utf-8")
        return (
            f"wrote {len(new_lines)} lines to {self.rel(path)}; "
            f"replaced lines {start_line}-{end_line}"
        )
