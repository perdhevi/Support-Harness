#!/usr/bin/env python3
"""
reset_memory.py — Wipe ChromaDB persistence directories created by the stages.

Stages 2-4 each use their own .chroma_stageN/ directory so memory from one
stage never contaminates another while you're A/B-ing them. Over time these
accumulate cases from your demos. Run this script to start fresh.

Usage:
    python reset_memory.py             # reset all stages (with confirmation)
    python reset_memory.py 2 4         # reset only stages 2 and 4
    python reset_memory.py -y          # skip the y/N prompt

Stage 1 has no memory, so it's not affected.

Each stage also exposes a `flush` REPL command that wipes its own directory
without leaving the running session. Use this script when you want to
reset multiple stages at once, or when no stage script is running.
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
CHROMA_GLOB = ".chroma_stage*"


def find_chroma_dirs(stages: list[int] | None) -> list[Path]:
    """Return matching chroma directories, sorted by stage number."""
    dirs = sorted(p for p in REPO_ROOT.glob(CHROMA_GLOB) if p.is_dir())
    if stages is None:
        return dirs
    keep = []
    for d in dirs:
        suffix = d.name.removeprefix(".chroma_stage")
        try:
            n = int(suffix)
        except ValueError:
            continue
        if n in stages:
            keep.append(d)
    return keep


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wipe ChromaDB memory directories for one or more stages."
    )
    parser.add_argument(
        "stages",
        nargs="*",
        type=int,
        help="Stage numbers to reset (e.g. '2 4'). Default: all stages.",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip the y/N confirmation prompt.",
    )
    args = parser.parse_args()

    stages = args.stages or None
    targets = find_chroma_dirs(stages)

    if not targets:
        if stages:
            print(f"No .chroma_stage* directories found for stages: {stages}.")
        else:
            print("No .chroma_stage* directories found — nothing to reset.")
        return 0

    print("Found:")
    total = 0
    for d in targets:
        size = dir_size_bytes(d)
        total += size
        print(f"  {d.name:<22} {fmt_size(size):>10}")
    print(f"  {'TOTAL':<22} {fmt_size(total):>10}\n")

    if not args.yes:
        try:
            answer = input("Delete these directories? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
        if answer not in {"y", "yes"}:
            print("Aborted.")
            return 1

    failures: list[tuple[Path, OSError]] = []
    for d in targets:
        try:
            shutil.rmtree(d)
            print(f"  removed {d.name}")
        except OSError as e:
            failures.append((d, e))
            print(f"  FAILED to remove {d.name}: {e}")

    if failures:
        print(
            "\nSome directories could not be removed. Make sure no stage\n"
            "script is currently running (ChromaDB locks its files while open)."
        )
        return 1

    print("\nDone — memory reset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
