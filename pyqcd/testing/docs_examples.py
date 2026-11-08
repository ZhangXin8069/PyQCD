"""Execute the Python code blocks of the user documentation.

The Read the Docs pages under ``docs/en`` and ``docs/zh`` contain runnable
examples. This module extracts every fenced ``python`` block and executes it in
one namespace, in page order, so that a documentation edit that silently breaks
an example fails here instead of in front of a user.

Blocks that cannot run in a plain install (they need the source checkout, or they
are deliberately abbreviated) must be marked:

* ``# pseudo-code`` anywhere inside the block, or
* ``# docs-skip`` anywhere inside the block.

Usage::

    python -m pyqcd.testing.docs_examples            # all English pages
    python -m pyqcd.testing.docs_examples --lang zh  # Chinese pages
    python -m pyqcd.testing.docs_examples --keep-going
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import traceback
from pathlib import Path

PAGES = ("overview", "usage", "performance", "install")
_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
_SKIP_MARKERS = ("# pseudo-code", "# docs-skip")


def extract(path: Path) -> list[str]:
    """Return the runnable Python blocks of one Markdown page."""
    text = path.read_text(encoding="utf-8")
    blocks = _BLOCK.findall(text)
    return [block for block in blocks
            if not any(marker in block for marker in _SKIP_MARKERS)]


def run_page(path: Path, namespace: dict) -> tuple[int, int, list[str]]:
    """Execute one page; return (ok, skipped, failures)."""
    text = path.read_text(encoding="utf-8")
    blocks = _BLOCK.findall(text)
    failures: list[str] = []
    ok = 0
    for index, block in enumerate(blocks):
        if any(marker in block for marker in _SKIP_MARKERS):
            continue
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(compile(block, f"{path.name}#{index}", "exec"), namespace)
            ok += 1
        except Exception:
            failures.append(
                f"{path.name}#{index}\n{traceback.format_exc(limit=3)}")
    skipped = len(blocks) - ok - len(failures)
    return ok, skipped, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", default="en", choices=("en", "zh"),
                        help="which documentation tree to check (default: en)")
    parser.add_argument("--root", default=None,
                        help="repository root (default: two levels above this file)")
    parser.add_argument("--keep-going", action="store_true",
                        help="keep going after a failing page")
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else Path(__file__).resolve().parents[2]
    docs = root / "docs" / args.lang
    namespace: dict = {"__name__": "__docs__"}
    totals = [0, 0]
    failures: list[str] = []
    for page in PAGES:
        path = docs / f"{page}.md"
        if not path.is_file():
            continue
        ok, skipped, page_failures = run_page(path, namespace)
        totals[0] += ok
        totals[1] += skipped
        status = "OK" if not page_failures else f"FAIL ({len(page_failures)})"
        print(f"{args.lang}/{page:12s} blocks ok={ok} skipped={skipped} -> {status}")
        failures.extend(page_failures)
        if page_failures and not args.keep_going:
            break
    for failure in failures:
        print("\n" + failure)
    print(f"\ntotal: ok={totals[0]} skipped={totals[1]} failed={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
