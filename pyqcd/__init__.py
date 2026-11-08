"""PyQCD — lattice-QCD toolkit with a native, `opt_einsum`-compatible engine.

The top-level package is intentionally almost empty: subpackages are imported on
demand so that a plain ``import pyqcd`` stays cheap and never pulls a heavy
dependency (torch, matplotlib, cupy, h5py ...) unless the module that needs it is
actually used.

Subpackages
-----------
``pyqcd.lattice``      constants, DeGrand–Rossi gamma matrices, SU(2) algebra
``pyqcd.gauge``        Wilson/Polyakov loops, Clover field strength, topology
``pyqcd.smear``        HYP and Stout smearing, SU(3) projection
``pyqcd.operator``     Clover / dual field strength, flowed gluon OPE operators
``pyqcd.vertex``       VdV/VVV vertices, momentum smearing, eigenvector compression
``pyqcd.contraction``  Wick contraction, baryon operators, distillation contractions
``pyqcd.renorm``       gradient flow, Z_R, hybrid scheme, NLO matching
``pyqcd.tools``        backend switching, I/O, slices, environment capture
``pyqcd.analysis``     resampling, fits, ratio / effective-mass / energy helpers
``pyqcd.pipeline``     configuration and the multi-step pipeline
``pyqcd.parallel``     resource detection and MPI meta-task planning
``pyqcd.einsum``       contraction engine (``opt_einsum``-compatible API)

Examples
--------
>>> import numpy as np
>>> import pyqcd.einsum as pe
>>> a = np.arange(6.0).reshape(2, 3)
>>> b = np.arange(12.0).reshape(3, 4)
>>> bool(np.allclose(pe.contract("ij,jk->ik", a, b), a @ b))
True

``python -m pyqcd`` prints the version, the active native backend, the optional
dependencies and the subpackage list; ``python -m pyqcd --version`` prints the
version only.
"""

__all__ = [
    "analysis",
    "contraction",
    "einsum",
    "gauge",
    "lattice",
    "operator",
    "parallel",
    "pipeline",
    "renorm",
    "smear",
    "tools",
    "vertex",
]

__version__ = "0.2.0"

import importlib as _importlib


def __getattr__(name: str):
    """Import a subpackage on first attribute access (PEP 562).

    ``import pyqcd`` stays cheap; ``pyqcd.einsum`` and friends resolve lazily so
    that ``dir(pyqcd)`` and attribute access both work as users expect.
    """
    if name in __all__:
        module = _importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
