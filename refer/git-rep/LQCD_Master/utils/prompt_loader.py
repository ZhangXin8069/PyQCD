from __future__ import annotations

from pathlib import Path


class PromptLoader:
    def __init__(self, root: Path):
        self.root = root

    def load(self, relative_path: str) -> str:
        path = self.root / relative_path
        if not path.exists():
            raise FileNotFoundError(f"prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()
