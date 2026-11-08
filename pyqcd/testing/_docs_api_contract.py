"""Every API the documentation mentions must exist in the installed package.

The Read the Docs pages are written against the published distribution, so a
renamed or removed function silently turns a tutorial into a lie. This gate
parses every fenced ``python`` block, resolves the imports that reference
``pyqcd``, and asserts that each attribute access on such an alias exists.

``pyqcd.testing`` is intentionally absent from the published package (it holds
the test suites and benchmarks), so it is allow-listed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGES = ("overview", "usage", "performance", "install")
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
ALLOW_MISSING = {"testing"}


def _blocks(page: str, lang: str = "en") -> list[str]:
    path = ROOT / "docs" / lang / f"{page}.md"
    if not path.is_file():
        return []
    return BLOCK.findall(path.read_text(encoding="utf-8"))


def _module_aliases(tree: ast.AST) -> dict[str, str]:
    """Map local alias -> pyqcd subpackage for ``import pyqcd.x as y``."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                parts = item.name.split(".")
                if len(parts) >= 2 and parts[0] == "pyqcd" and item.asname:
                    aliases[item.asname] = parts[1]
    return aliases


def _from_imports(tree: ast.AST) -> list[tuple[str, str]]:
    """Return ``(module, name)`` pairs for ``from pyqcd.x import name``."""
    pairs: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if len(parts) >= 2 and parts[0] == "pyqcd":
                module = parts[1]
                for item in node.names:
                    pairs.append((module, item.name))
    return pairs


def _attributes(tree: ast.AST, alias: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == alias):
            names.add(node.attr)
    return names


def collect() -> tuple[int, list[str]]:
    """Return (number of checked references, list of failures)."""
    import importlib

    checked = 0
    failures: list[str] = []
    for page in PAGES:
        for index, block in enumerate(_blocks(page)):
            try:
                tree = ast.parse(block)
            except SyntaxError:
                continue  # pseudo-code blocks are allowed to be incomplete
            for alias, module in _module_aliases(tree).items():
                if module in ALLOW_MISSING:
                    continue
                target = importlib.import_module(f"pyqcd.{module}")
                for name in sorted(_attributes(tree, alias)):
                    checked += 1
                    if not hasattr(target, name):
                        failures.append(
                            f"{page}.md#{index}: {alias}.{name} "
                            f"(pyqcd.{module}.{name}) does not exist")
            for module, name in _from_imports(tree):
                if module in ALLOW_MISSING:
                    continue
                target = importlib.import_module(f"pyqcd.{module}")
                checked += 1
                if not hasattr(target, name):
                    failures.append(
                        f"{page}.md#{index}: from pyqcd.{module} import {name} "
                        "does not exist")
    return checked, failures


def test_documented_api_exists() -> None:
    checked, failures = collect()
    assert checked >= 20, f"only {checked} API references found; parser regression?"
    assert not failures, "documented API does not exist:\n  " + "\n  ".join(failures)


def test_pages_are_present() -> None:
    for page in PAGES:
        assert _blocks(page), f"docs/en/{page}.md has no runnable python block"
