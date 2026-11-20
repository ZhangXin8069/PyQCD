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
import os
import re
import sys
import traceback
from pathlib import Path

#: 进程内关掉的两个开关（与 ``conftest.main()`` 同口径）：文档示例不该被用户家目录
#: 里的配置表或自动 tune 改变选路。
_PROFILE_ENV = "PYQCD_EINSUM_PROFILE"
_AUTOTUNE_ENV = "PYQCD_EINSUM_AUTOTUNE"

PAGES = ("overview", "usage", "performance", "install",
         "acknowledgements", "appendix")
_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
_SKIP_MARKERS = ("# pseudo-code", "# docs-skip")


def extract(path: Path) -> list[str]:
    """Return the runnable Python blocks of one Markdown page."""
    text = path.read_text(encoding="utf-8")
    blocks = _BLOCK.findall(text)
    return [block for block in blocks
            if not any(marker in block for marker in _SKIP_MARKERS)]


def run_page(path: Path, namespace: dict) -> tuple[int, int, list[str]]:
    """Execute one page; return (ok, skipped, failures).

    A block that runs without asserting anything verifies nothing, so every block
    that is not explicitly skipped must contain an ``assert``: the documentation is
    then a correctness suite, not a set of printouts.
    """
    text = path.read_text(encoding="utf-8")
    blocks = _BLOCK.findall(text)
    failures: list[str] = []
    ok = 0
    for index, block in enumerate(blocks):
        if any(marker in block for marker in _SKIP_MARKERS):
            continue
        if "assert " not in block:
            failures.append(
                f"{path.name}#{index}: no assertion — every example must verify its "
                f"own result (add `assert`, or mark the block `# pseudo-code`)")
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
    parser = argparse.ArgumentParser(
        description="Execute the Python code blocks of the user documentation.")
    parser.add_argument("--lang", default="en", choices=("en", "zh"),
                        help="which documentation tree to check (default: en)")
    parser.add_argument("--root", default=None,
                        help="repository root (default: two levels above this file)")
    parser.add_argument("--keep-going", action="store_true",
                        help="keep going after a failing page")
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else Path(__file__).resolve().parents[2]
    docs = root / "docs" / args.lang
    # 文档示例必须**只依赖文档里写的东西**：用户家目录里的配置表与自动 tune 都会
    # 改变选路（实测 ``kernel=backend`` 命中时整式交给后端、一个中间体都不产生，
    # 于是 ``shared_intermediates`` 的断言拿到空 Counter）。与 ``conftest.main()``
    # 同口径，进程内关掉两者，退出时原样还回去。
    saved_env = {name: os.environ.get(name)
                 for name in (_PROFILE_ENV, _AUTOTUNE_ENV)}
    os.environ[_PROFILE_ENV] = "0"
    os.environ[_AUTOTUNE_ENV] = "0"
    try:
        return _run_pages(args, docs)
    finally:
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_pages(args: argparse.Namespace, docs: Path) -> int:
    """逐页跑代码块并汇总（``main`` 的实体；环境隔离在调用方）。"""
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
