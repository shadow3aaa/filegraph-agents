from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import FGAConfig
from .observer import NullObserver
from .runtime import FGARuntime


def _load_dotenv(repo: str) -> None:
    """Load .env from the current directory and the target repo.

    Real environment variables always win over .env values, so an explicit
    `export FGA_MODEL=...` still overrides the file.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path.cwd() / ".env", override=False)
    load_dotenv(Path(repo) / ".env", override=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run FileGraph Agents v0 on a repository.")
    parser.add_argument("repo", help="Path to repository/workspace root")
    parser.add_argument("instruction", nargs="*", help="Task instruction. If omitted, read stdin.")
    parser.add_argument("--model", help="Model name override")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL override")
    args = parser.parse_args(argv)

    instruction = " ".join(args.instruction).strip() or sys.stdin.read().strip()
    if not instruction:
        print("No instruction provided.", file=sys.stderr)
        return 2

    _load_dotenv(args.repo)

    config = FGAConfig.from_env()
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url

    # Use the rich TUI on a real terminal; fall back to plain output otherwise
    # (redirected/piped stdout, or if rich is unavailable).
    observer = NullObserver()
    ui = None
    if sys.stdout.isatty():
        try:
            from .ui import RichObserver

            ui = RichObserver()
            observer = ui
        except Exception:
            ui = None

    observer.on_start(model=config.model, repo=str(Path(args.repo).resolve()), instruction=instruction)

    runtime = FGARuntime(Path(args.repo), config=config, observer=observer)
    try:
        result = runtime.run(instruction)
    except Exception as e:
        message = f"{type(e).__name__}: {e}"
        observer.on_error(message)
        if ui is None:
            print(f"\nError: {message}", file=sys.stderr)
        return 1

    if ui is not None:
        ui.print_final(result)
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
