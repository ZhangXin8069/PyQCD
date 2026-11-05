"""KUPL (CPU) binding for the native einsum library.

Mirrors ``pyqcu/kupl``: a ctypes bridge that needs no compiler at import time
and degrades to the pure Python planner when ``cpp/kupl/einsum`` has not been
built.
"""

from __future__ import annotations

from . import einsum

__all__ = ["einsum"]
