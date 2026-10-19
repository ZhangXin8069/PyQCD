#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Sun et al. (2018) ``A1^{-+}`` spatial-Wilson-loop spectrum.

Dependencies: numpy, scipy; lqcddb supplies LIME I/O, Stout smearing, Jackknife,
and GEVP helpers when available.  The default ``sun2018`` basis implements the
four spatially solid prototypes at the end of Fig. 1 of Sun et al. and the six
single-link/double-link smoothing schemes of Chen et al. (2006).
Input links: (4, Nt, Nz, Ny, Nx, 3, 3), directions X,Y,Z,T, periodic BC.

Examples (fit windows are starting choices, not physics results):
  python glueball_a1mp_spectrum.py selftest
  python glueball_a1mp_spectrum.py measure --gauge cfg_1000.lime \
      --cfg-id 1000 --ensemble E32P29 --out op_1000.npz
  python glueball_a1mp_spectrum.py analyze --manifest measurements.txt \
      --t0 0 --td 1 --rank 24 --states 2 \
      --fit-ranges 2:6 2:5 --at-fm 0.0897 --out spectrum

The ``table1-ps`` measurement basis implements the Type-I $0^{-+}$ operator
from Chen et al. (2006), Eq. (12), using the complete Table-I path catalog;
its pseudoscalar combination contains the C10, C11, and C12 terms.  It is a
separate operator definition from the spatial Sun-2018 $A_1^{-+}$ basis.

The manifest lists measurement NPZ paths IN MONTE CARLO ORDER, one per line;
relative paths are resolved relative to the manifest. Distinct configurations
must have distinct cfg-id values. Every configuration is retained as one input
to the delete-one Jackknife; no binning is performed.

Physics scope:
* The default `sun2018` basis has 24 operators: four spatially solid,
  eight-link prototypes (the last four shapes in Sun et al. Fig. 1, also
  reproduced in Chen et al. Fig. 3)
  combined with the six schemes $s_{\lambda_s}^{2,4,6}$ and
  $f_{\lambda_f}\!\circ s_{\lambda_s}^{2,4,6}$, using
  $\lambda_s=0.1$ and $\lambda_f=0.5$.
* Exact geometric A1, P=-, C=+ projection and zero spatial momentum.
* No temporal smearing, gauge fixing, per-configuration mean subtraction,
  or use of the E.B operator. No renormalization is needed to extract energies.
* A fixed (t0,td) GEVP defines optimized projected correlators. Delete-one
  Jackknife repeats the GEVP in the central-data retained subspace and matches modes.
  The subspace rank and fit windows stay fixed within a resampling analysis;
  their choices must be studied separately.
* Sun et al.'s plotted effective mass is the logarithmic estimator
  $\log[C(t)/C(t+1)]$; correlated periodic fits remain available for the
  optional spectrum summary and their chi2/dof is only diagnostic.
* A1 also contains higher continuum J. In full QCD the measured finite-volume
  levels can mix with flavor-singlet mesons and multiparticle states. A
  gluonic-only basis cannot establish glueball purity or a resonance pole.

Method reference: Sun et al., arXiv:1702.08174, Sections III-IV.
This is a correctness-first, slice-by-slice CPU reference, not MPI/GPU software.
The ``analyze`` output contains the two GEVP projected correlators and the
logarithmic effective-mass arrays needed for the $A_1^{-+}$ panel of Sun et al.
Fig. 3.
"""

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize_scalar

try:
    from lqcddb import set_backend as _lq_set_backend
    from lqcddb import get_backend as _lq_get_backend
    
except ImportError:  # Keep .npy/selftest usable without the optional package.
    _lq_set_backend = None

try:
    from lqcddb.io import read_gauge as _lq_read_gauge
except ImportError:
    _lq_read_gauge = None

try:
    from lqcddb import stout_smear_ndarray as _lq_stout_smear_ndarray
except ImportError:
    _lq_stout_smear_ndarray = None

try:
    from lqcddb.analyse import Jackknife as _lq_jackknife
    from lqcddb.analyse import meff as _lq_meff
    from lqcddb.analyse import solve_gevp as _lq_solve_gevp
except ImportError:
    _lq_jackknife = _lq_meff = _lq_solve_gevp = None

HBARC_GEV_FM = 0.1973269804
VERSION = "sun2018-a1mp-v3-lqcddb"
# Signed directions: +1,+2,+3 = +X,+Y,+Z; negatives reverse a link.
#
# These are directed representatives of the four spatially solid loops at the
# end of Fig. 1 in Sun et al. (2018), read left-to-right on the second row and
# then left-to-right on the last row.  A cyclic shift, reversal, or proper cubic
# rotation gives the same spatially summed ReTr A1^{-+} operator.  The first
# two are the two corner-connected rectangle products in the middle row; the
# last two are the bottom-row solid loops.  They are all closed eight-link
# paths, and each is used with all 24 proper cubic rotations.
PROTOTYPES = (
    (1, 1, 2, -1, -2, 3, -1, -3),
    (1, 2, 1, 3, -1, -3, -1, -2),
    (1, 1, 2, -1, 3, -2, -1, -3),
    (1, 1, 2, 3, -1, -2, -1, -3),
)
PROTOTYPE_LABELS = (
    "fig3-solid-middle-left",
    "fig3-solid-middle-right",
    "fig3-solid-bottom-left",
    "fig3-solid-bottom-right",
)
SUN_LAMBDA_S = 0.1
SUN_LAMBDA_F = 0.5
SUN_SINGLE_LEVELS = (2, 4, 6)
SUN2018_SCHEMES = (
    ("s2", 2, False), ("s4", 4, False), ("s6", 6, False),
    ("fs2", 2, True), ("fs4", 4, True), ("fs6", 6, True),
)
# Prefer the GPU backend in production, but make the reference program and its
# self-tests usable on a CPU-only login node as well.  A CuPy installation can
# import successfully while still failing at the first CUDA allocation.
if _lq_set_backend is None or _lq_get_backend is None:
    backend = np
else:
    try:
        _lq_set_backend('numpy')
        backend = _lq_get_backend()
        backend.asarray(np.zeros(1, dtype=np.float64))
    except Exception:
        _lq_set_backend('numpy')
        backend = _lq_get_backend()

# Chen et al. (2006), Table I.  Directions are +X,+Y,+Z,+T = +1,+2,+3,+4.
# The first three fields are the paper's index, name, and ordered path; the
# link counts are retained for an explicit audit of the tadpole powers.
TABLE_I_PATHS = (
    (1, "S-Plaquette", (1, 2, -1, -2), (4, 0)),
    (2, "S-Rectangle", (1, 1, 2, -1, -1, -2), (6, 0)),
    (3, "T-Plaquette", (1, 4, -1, -4), (2, 2)),
    (4, "T-Rectangle", (1, 1, 4, -1, -1, -4), (4, 2)),
    (5, "S-Chair", (1, 2, -1, 3, -2, -3), (6, 0)),
    (6, "S-Butterfly", (1, 2, -1, -2, 3, -2, -3, 2), (8, 0)),
    (7, "S-Sunbed", (1, 1, 2, -1, -1, 3, -2, -3), (8, 0)),
    (8, "T-Chair", (1, 4, -1, 3, -4, -3), (4, 2)),
    (9, "T-Sunbed", (1, 1, 4, -1, -1, 3, -4, -3), (6, 2)),
    (10, "Knot", (1, 4, -1, -4, 3, -2, -3, 2), (6, 2)),
    (11, "LS-Knot", (1, 4, -1, -4, 3, 3, -2, -3, -3, 2), (8, 2)),
    (12, "LT-Knot", (1, 1, 4, -1, -1, -4, 3, -2, -3, 2), (8, 2)),
)
TABLE_I_PATH_BY_INDEX = {item[0]: item for item in TABLE_I_PATHS}


def dagger(a):
    return a.conj().swapaxes(-1, -2)


def cubic_rotations():
    """Images of e_X,e_Y,e_Z under the 24 proper cubic rotations."""
    rotations = []
    for p in itertools.permutations((1, 2, 3)):
        parity = (-1) ** sum(p[i] > p[j] for i in range(3) for j in range(i+1, 3))
        for s in itertools.product((-1, 1), repeat=3):
            if parity * np.prod(s) == 1:
                rotations.append(tuple(p[i] * s[i] for i in range(3)))
    return rotations


def rotate_path(path, rotation):
    return tuple(
        d if abs(d) == 4 else rotation[abs(d)-1] * (1 if d > 0 else -1)
        for d in path
    )


def canonical_trace_path(path):
    """Valid for SPATIALLY SUMMED ReTr loops with periodic boundary conditions.

    Cyclic permutations change the base point, hence use translation invariance
    of the spatial sum; reversed paths have conjugate traces. Do not use this
    optimization for a local untraced clover matrix or a nonzero-momentum loop.
    """
    inverse = tuple(-d for d in path[::-1])
    return min(q[i:]+q[:i] for q in (path, inverse) for i in range(len(q)))


def projected_paths(path):
    """Integer coefficients of sum_R [ReTr W(Rp)-ReTr W(-Rp)]."""
    displacement = [sum((1 if d > 0 else -1) for d in path if abs(d) == j)
                    for j in (1, 2, 3)]
    if any(displacement):
        raise ValueError("Prototype is not closed")
    weights = Counter()
    for rotation in cubic_rotations():
        q = rotate_path(path, rotation)
        weights[canonical_trace_path(q)] += 1
        weights[canonical_trace_path(tuple(-d for d in q))] -= 1
    terms = tuple((p, w) for p, w in sorted(weights.items()) if w)
    if not terms:
        raise ValueError("Prototype has identically zero A1^{-+} projection")
    return terms


PROJECTED_PATHS = tuple(projected_paths(p) for p in PROTOTYPES)


def shift(field, offset):
    """field[x+offset], with offset ordered (z,y,x)."""
    if not any(offset):
        return field
    return backend.roll(field, tuple(-int(n) for n in offset), axis=(0, 1, 2))


def shift4(field, offset):
    """field[t+dt,z+dz,y+dy,x+dx], with periodic boundaries."""
    if not any(offset):
        return field
    return backend.roll(field, tuple(-int(n) for n in offset),
                   axis=(0, 1, 2, 3))


SPATIAL_AXIS_OFFSETS = ((0, 0, 1), (0, 1, 0), (1, 0, 0))


def _offset_add(*offsets):
    return tuple(sum(values) for values in zip(*offsets))


def loop_matrix(links, path, link_length=1):
    """All loops start/end at x; ``link_length`` is 1 or 2 lattice spacings.

    ``link_length=1`` consumes ordinary smeared links.  ``link_length=2``
    consumes the fuzzy superlinks produced by :func:`sun_double_link_step`.
    The latter are still indexed by their starting site, so only the
    coordinate displacement changes while traversing a path.
    """
    if link_length not in (1, 2):
        raise ValueError("link_length must be 1 or 2")
    offset = [0, 0, 0]
    product = None
    for d in path:
        mu, axis = abs(d)-1, 3-abs(d)
        if d < 0:
            offset[axis] -= link_length
        link = shift(links[mu], offset)
        if d < 0:
            link = dagger(link)
        product = link if product is None else product @ link
        if d > 0:
            offset[axis] += link_length
    if any(offset):
        raise ValueError("Non-closed path")
    return product


def loop_matrix_4d(gauge, path):
    """All-base-point Wilson loops for a path using X,Y,Z,T links."""
    # gauge layout is (direction, time, z, y, x, color, color); the coordinate
    # offset is ordered (time,z,y,x).
    axes = (3, 2, 1, 0)
    offset = [0, 0, 0, 0]
    product = None
    for direction in path:
        mu = abs(direction)-1
        axis = axes[mu]
        if direction < 0:
            offset[axis] -= 1
        link = shift4(gauge[mu], offset)
        if direction < 0:
            link = dagger(link)
        product = link if product is None else product @ link
        if direction > 0:
            offset[axis] += 1
    if any(offset):
        raise ValueError(f"Non-closed four-dimensional path: {path}")
    return product


def loop_matrix_4d_at_time(gauge, path, time):
    """Wilson loop on one spatial slice, retaining temporal-link wrapping."""
    axes = (3, 2, 1, 0)
    nt = gauge.shape[1]
    offset = [0, 0, 0, 0]  # (dt,dz,dy,dx)
    product = None
    for direction in path:
        mu = abs(direction)-1
        axis = axes[mu]
        if direction < 0:
            offset[axis] -= 1
        link = gauge[mu, (time + offset[0]) % nt]
        link = shift(link, offset[1:])
        if direction < 0:
            link = dagger(link)
        product = link if product is None else product @ link
        if direction > 0:
            offset[axis] += 1
    if any(offset):
        raise ValueError(f"Non-closed four-dimensional path: {path}")
    return product


def table1_path_trace(gauge, path, time):
    """Spatially averaged real trace of a Table-I loop at one time slice."""
    loop = loop_matrix_4d(gauge, path)
    return backend.trace(loop[time], axis1=-2, axis2=-1).real.mean()


def table1_path_trace_at_time(gauge, path, time):
    loop = loop_matrix_4d_at_time(gauge, path, time)
    return backend.trace(loop, axis1=-2, axis2=-1).real.mean()


def table1_tadpoles(gauge):
    """Estimate spatial and temporal tadpole factors from plaquettes."""
    spatial = []
    temporal = []
    for mu in range(3):
        for nu in range(mu+1, 3):
            path = (mu+1, nu+1, -(mu+1), -(nu+1))
            loop = loop_matrix_4d(gauge, path)
            spatial.append(backend.trace(loop, axis1=-2, axis2=-1).real.mean().get()/3)
        path = (mu+1, 4, -(mu+1), -4)
        loop = loop_matrix_4d(gauge, path)
        temporal.append(backend.trace(loop, axis1=-2, axis2=-1).real.mean().get()/3)
    us = float(np.mean(spatial)**0.25)
    ut = float(np.mean(temporal)**0.25)
    if not backend.isfinite(us) or not backend.isfinite(ut) or us <= 0 or ut <= 0:
        raise ValueError(f"Invalid tadpole factors us={us}, ut={ut}")
    return us, ut


def table1_pseudoscalar_timeseries(gauge, xi=1.0, us=1.0, ut=1.0):
    """Chen-2006 Table-I Type-I pseudoscalar operator at every time, Eq. (12).

    The 24 cubic rotations have alpha_r^(A1)=1.  C10--C12 are the three
    temporal-containing paths in Table I; they are deliberately kept as the
    paper's E.B/topological-type operator and are not mixed with the spatial
    BDB basis used by the default measurement mode.
    """
    if xi <= 0 or us <= 0 or ut <= 0:
        raise ValueError("xi, us and ut must be positive")
    coefficients = {
        10: 6/(96*us**6*ut**2),
        11: -1/(96*us**8*ut**2),
        12: -1/(192*us**8*ut**2),
    }
    total = backend.zeros(gauge.shape[1], dtype=float)
    for time in range(gauge.shape[1]):
        value = 0.0
        for rotation in cubic_rotations():
            for index, coefficient in coefficients.items():
                path = TABLE_I_PATH_BY_INDEX[index][2]
                value += coefficient * table1_path_trace_at_time(
                    gauge, rotate_path(path, rotation), time)
        total[time] = value
    # The paper has a spatial-volume sum.  Use a rotation average and spatial
    # average here; a common overall normalization does not change energies.
    return xi * total / len(cubic_rotations())


def table1_pseudoscalar(gauge, time, xi=1.0, us=1.0, ut=1.0):
    """One time slice of the cached-timeseries Table-I pseudoscalar operator."""
    return float(table1_pseudoscalar_timeseries(gauge, xi=xi, us=us, ut=ut)[time])


def operators(links, link_length=1):
    r"""Sun's spatially summed $A_1^{-+}$ projection for one prototype set.

    The four values are

    $$O_a = \frac{1}{48\,3L^3}\sum_{x,R}
       \operatorname{ReTr}\left[W_x(Rp_a)-W_x(-Rp_a)\right].$$

    ``link_length=2`` is used for the $f_{\lambda_f}\circ s^n$ schemes,
    whose path edges are fuzzy two-link superlinks.
    """
    values = []
    for terms in PROJECTED_PATHS:
        total = 0.0
        for path, coefficient in terms:
            w = loop_matrix(links, path, link_length=link_length)
            total += coefficient * backend.trace(w, axis1=-2, axis2=-1).real.mean()
        values.append(total / (48.0 * 3.0))
    return backend.asarray(values)


def project_su3(matrix):
    """Polar reunitarization followed by a common determinant phase removal."""
    u, singular, vh = backend.linalg.svd(matrix)
    if backend.min(singular) < 1e-12:
        raise ValueError("Nearly singular APE matrix; reduce alpha")
    unitary = u @ vh
    return unitary * backend.exp(-1j * backend.angle(backend.linalg.det(unitary))/3)[..., None, None]


def ape_step(links, alpha):
    """Simultaneous spatial APE update, four staples per link, no time links."""
    result = backend.empty_like(links)
    for mu in range(3):
        m = backend.eye(3, dtype=int)[2-mu]
        staples = backend.zeros_like(links[mu])
        for nu in range(3):
            if nu == mu:
                continue
            n = backend.eye(3, dtype=int)[2-nu]
            staples += links[nu] @ shift(links[mu], n) @ dagger(shift(links[nu], m))
            staples += (dagger(shift(links[nu], -n)) @ shift(links[mu], -n)
                        @ shift(links[nu], m-n))
        result[mu] = project_su3((1-alpha)*links[mu] + (alpha/4)*staples)
    return result


def _validate_spatial_links(links):
    if links.ndim != 6 or links.shape[0] != 3 or links.shape[-2:] != (3, 3):
        raise ValueError("Expected spatial links (3,Nz,Ny,Nx,3,3)")
    nz, ny, nx = links.shape[1:4]
    if not nz == ny == nx or nx < 4:
        raise ValueError("Require a periodic cubic spatial lattice L >= 4")


def sun_single_link_step(links, lambda_s=SUN_LAMBDA_S):
    r"""Morningstar--Peardon single-link map $s_{\lambda_s}$.

    This is the map used in the six schemes quoted by Chen et al. and Sun et
    al., not the normalized APE parameterization used by ``ape_step``:

    $$U_j^s(x)=\mathcal P_{SU(3)}\left[U_j(x)+\lambda_s
      \sum_{\pm k\ne j}U_{\pm k}(x)U_j(x\!\pm\!\hat k)
      U_{\pm k}^{\dagger}(x\!\pm\!\hat j)\right].$$
    """
    _validate_spatial_links(links)
    if not np.isfinite(lambda_s) or lambda_s < 0:
        raise ValueError("lambda_s must be finite and nonnegative")
    result = backend.empty_like(links)
    for mu, e_mu in enumerate(SPATIAL_AXIS_OFFSETS):
        staples = backend.zeros_like(links[mu])
        for nu, e_nu in enumerate(SPATIAL_AXIS_OFFSETS):
            if nu == mu:
                continue
            plus = (links[nu] @ shift(links[mu], e_nu)
                    @ dagger(shift(links[nu], e_mu)))
            minus = (dagger(shift(links[nu], tuple(-v for v in e_nu)))
                     @ shift(links[mu], tuple(-v for v in e_nu))
                     @ shift(links[nu], _offset_add(e_mu,
                                                    tuple(-v for v in e_nu))))
            staples += plus + minus
        result[mu] = project_su3(links[mu] + lambda_s*staples)
    return result


def sun_double_link_step(links, lambda_f=SUN_LAMBDA_F):
    r"""Morningstar--Peardon double-link map $f_{\lambda_f}$.

    The returned array contains gauge-covariant superlinks from $x$ to
    $x+2\hat\jmath$.  Its last three spatial dimensions remain the starting
    site, so :func:`loop_matrix` can consume it with ``link_length=2``.
    """
    _validate_spatial_links(links)
    if not np.isfinite(lambda_f) or lambda_f < 0:
        raise ValueError("lambda_f must be finite and nonnegative")
    result = backend.empty_like(links)
    for mu, e_mu in enumerate(SPATIAL_AXIS_OFFSETS):
        two_e_mu = tuple(2*v for v in e_mu)
        straight = links[mu] @ shift(links[mu], e_mu)
        staples = backend.zeros_like(links[mu])
        for nu, e_nu in enumerate(SPATIAL_AXIS_OFFSETS):
            if nu == mu:
                continue
            # Positive k staple in Eq. (21) of Morningstar--Peardon.
            plus = (links[nu] @ shift(links[mu], e_nu)
                    @ shift(links[mu], _offset_add(e_mu, e_nu))
                    @ dagger(shift(links[nu], two_e_mu)))
            # The negative-k staple uses U_k^dagger(x-k) at its start and
            # U_k(x+2j-k) at its end.
            minus = (dagger(shift(links[nu], tuple(-v for v in e_nu)))
                     @ shift(links[mu], tuple(-v for v in e_nu))
                     @ shift(links[mu], _offset_add(e_mu,
                                                    tuple(-v for v in e_nu)))
                     @ shift(links[nu], _offset_add(two_e_mu,
                                                    tuple(-v for v in e_nu))))
            staples += plus + minus
        result[mu] = project_su3(straight + lambda_f*staples)
    return result


def sun2018_scheme_links(links, lambda_s=SUN_LAMBDA_S,
                         lambda_f=SUN_LAMBDA_F):
    r"""Return the six Sun/Chen smoothing schemes.

    The tuple entries are ``(name, link_array, link_length)``.  The first
    three are $s^2,s^4,s^6$ ordinary links; the last three are
    $f\circ s^2,f\circ s^4,f\circ s^6$ two-link superlinks.
    """
    _validate_spatial_links(links)
    current = backend.array(links, dtype=backend.complex128, copy=True)
    singles = {}
    for iteration in range(1, max(SUN_SINGLE_LEVELS)+1):
        current = sun_single_link_step(current, lambda_s)
        if iteration in SUN_SINGLE_LEVELS:
            singles[iteration] = backend.array(current, copy=True)
    schemes = []
    for name, nstep, fuzzy in SUN2018_SCHEMES:
        source = singles[nstep]
        if fuzzy:
            schemes.append((name, sun_double_link_step(source, lambda_f), 2))
        else:
            schemes.append((name, source, 1))
    return tuple(schemes)


def sun2018_operator_matrix(links, lambda_s=SUN_LAMBDA_S,
                            lambda_f=SUN_LAMBDA_F):
    """Measure the 24 Sun et al. $A_1^{-+}$ operators on one time slice."""
    values = [operators(smeared, link_length=link_length)
              for _, smeared, link_length
              in sun2018_scheme_links(links, lambda_s, lambda_f)]
    return backend.concatenate(values)


def read_gauge(path):
    if Path(path).suffix.lower() == ".npy":
        return backend.load(path, mmap_mode="r", allow_pickle=False)
    if _lq_read_gauge is None:
        raise ImportError("LIME gauge input requires the lqcddb package")
    # lqcddb validates the ILDG/LIME metadata, payload size, and optional CRC.
    return _lq_read_gauge(str(path), backend="numpy")


def _stout_slice(links):
    """Build the full layout expected by lqcddb's spatial Stout routine."""
    if _lq_stout_smear_ndarray is None or _lq_set_backend is None:
        raise ImportError("Stout smearing requires lqcddb")
    nz, ny, nx = links.shape[1:4]
    gauge = backend.empty((4, 1, nz, ny, nx, 3, 3), dtype=links.dtype)
    gauge[:3, 0] = links
    gauge[3, 0] = backend.broadcast_to(backend.eye(3, dtype=links.dtype), (nz, ny, nx, 3, 3))
    return gauge


def _stout_step(gauge, rho):
    """Apply one lqcddb spatial Stout step in the explicit NumPy backend."""
    _lq_stout_smear_ndarray(gauge, nstep=1, rho=rho)


def measure_table1_ps(args):
    """Measure the complete Chen-2006 Table-I pseudoscalar combination."""
    steps = sorted(set(args.smear_steps or [0, 2, 4, 6, 8, 10]))
    if not steps or steps[0] < 0:
        raise ValueError("Require nonnegative smearing levels")
    if args.smear_method == "ape" and not 0 <= args.alpha < 1:
        raise ValueError("APE requires 0 <= alpha < 1")
    if args.smear_method == "stout":
        if not backend.isfinite(args.stout_rho) or args.stout_rho < 0:
            raise ValueError("Stout requires a finite nonnegative rho")
        if _lq_stout_smear_ndarray is None:
            raise ImportError("--smear-method stout requires lqcddb")
    gauge = backend.asarray(read_gauge(args.gauge), dtype=backend.complex128)
    if gauge.ndim != 7 or gauge.shape[0] != 4 or gauge.shape[-2:] != (3, 3):
        raise ValueError("Expected gauge shape (4,Nt,Nz,Ny,Nx,3,3)")
    nt, nz, ny, nx = gauge.shape[1:5]
    if not nz == ny == nx or nx < 4:
        raise ValueError("This reference requires a periodic cubic spatial lattice L >= 4")
    if args.us is None:
        us, measured_ut = table1_tadpoles(gauge)
    else:
        us, measured_ut = float(args.us), None
    ut = measured_ut if args.ut is None else float(args.ut)
    if args.xi <= 0 or us <= 0 or ut <= 0:
        raise ValueError("xi, us and ut must be positive")

    working = backend.array(gauge, copy=True)
    obs = backend.empty((nt, len(steps)), dtype=float)
    labels = [f"table1_ps_{args.smear_method}{s}" for s in steps]
    next_level = 0
    for iteration in range(steps[-1]+1):
        if iteration == steps[next_level]:
            obs[:, next_level] = table1_pseudoscalar_timeseries(
                working, xi=args.xi, us=us, ut=ut)
            next_level += 1
            if next_level == len(steps):
                break
        if args.smear_method == "stout":
            _stout_step(working, args.stout_rho)
        else:
            for time in range(nt):
                working[:3, time] = ape_step(working[:3, time], args.alpha)

    meta = dict(
        version=VERSION, cfg_id=args.cfg_id, ensemble=args.ensemble,
        gauge=str(Path(args.gauge).resolve()), shape=list(gauge.shape),
        operator_basis="table1-ps", table1_paths=TABLE_I_PATHS,
        table1_ps_paths=[10, 11, 12],
        table1_ps_coefficients={
            "C10": 6/(96*us**6*ut**2),
            "C11": -1/(96*us**8*ut**2),
            "C12": -1/(192*us**8*ut**2),
        },
        smear_steps=steps, smear_method=args.smear_method,
        alpha=args.alpha, stout_rho=args.stout_rho,
        xi=args.xi, us=us, ut=ut,
        normalization="rotation average and spatial-volume average",
        boundary="periodic", direction_order="X,Y,Z,T",
    )
    out = Path(args.out)
    if out.suffix != ".npz":
        raise ValueError("Measurement output must end in .npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, O=obs.get(), labels=np.asarray(labels),
                        metadata=json.dumps(meta))
    print(f"Saved {out}: Table-I PS O shape={obs.shape}; "
          f"paths C10,C11,C12; us={us:.8f}, ut={ut:.8f}, xi={args.xi:.8f}")


def measure_sun2018(args):
    r"""Measure Sun et al.'s 24-operator spatial $A_1^{-+}$ basis.

    Each configuration contributes four Fig. 1 spatial-solid prototypes for
    each of $s^2,s^4,s^6,f\circ s^2,f\circ s^4,f\circ s^6$.  The ``f``
    schemes use two-link superlinks, as in the cited construction.
    """
    if args.smear_steps is not None:
        raise ValueError("sun2018 uses fixed schemes s2,s4,s6,fs2,fs4,fs6; "
                         "do not pass --smear-steps")
    if not np.isfinite(args.sun_lambda_s) or args.sun_lambda_s < 0:
        raise ValueError("--sun-lambda-s must be finite and nonnegative")
    if not np.isfinite(args.sun_lambda_f) or args.sun_lambda_f < 0:
        raise ValueError("--sun-lambda-f must be finite and nonnegative")
    gauge = read_gauge(args.gauge)
    if gauge.ndim != 7 or gauge.shape[0] != 4 or gauge.shape[-2:] != (3, 3):
        raise ValueError("Expected gauge shape (4,Nt,Nz,Ny,Nx,3,3)")
    nt, nz, ny, nx = gauge.shape[1:5]
    if not nz == ny == nx or nx < 4:
        raise ValueError("This reference requires a periodic cubic spatial lattice L >= 4")
    obs = backend.empty((nt, len(SUN2018_SCHEMES)*len(PROTOTYPES)), dtype=float)
    labels = [f"{scheme}_{prototype}"
              for scheme, _, _ in SUN2018_SCHEMES
              for prototype in PROTOTYPE_LABELS]
    for t in range(nt):
        links = backend.array(gauge[:3, t], dtype=backend.complex128, copy=True)
        if not bool(backend.all(backend.isfinite(links))):
            raise ValueError(f"Nonfinite links on time slice {t}")
        obs[t] = sun2018_operator_matrix(
            links, lambda_s=args.sun_lambda_s, lambda_f=args.sun_lambda_f)
        print(f"time slice {t+1}/{nt}", flush=True)
    meta = dict(
        version=VERSION, cfg_id=args.cfg_id, ensemble=args.ensemble,
        gauge=str(Path(args.gauge).resolve()), shape=list(gauge.shape),
        operator_basis="sun2018-a1mp", irrep="A1^{-+}",
        prototypes=PROTOTYPES, prototype_labels=PROTOTYPE_LABELS,
        prototype_catalog="Sun et al. 2018 Fig. 1 last four spatially solid shapes",
        smear_method="morningstar-single-double-link",
        smear_schemes=[name for name, _, _ in SUN2018_SCHEMES],
        smear_scheme_definitions={
            "s2": "s_lambda_s^2", "s4": "s_lambda_s^4", "s6": "s_lambda_s^6",
            "fs2": "f_lambda_f o s_lambda_s^2",
            "fs4": "f_lambda_f o s_lambda_s^4",
            "fs6": "f_lambda_f o s_lambda_s^6",
        },
        lambda_s=args.sun_lambda_s, lambda_f=args.sun_lambda_f,
        normalization="1/(48*3*L^3)", parity="minus", charge_conjugation="plus",
        boundary="periodic", direction_order="X,Y,Z,T",
    )
    out = Path(args.out)
    if out.suffix != ".npz":
        raise ValueError("Measurement output must end in .npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    # CuPy cannot represent Unicode string arrays; keep labels on the host
    # while allowing the numeric observable to be written from the GPU.
    backend.savez_compressed(out, O=obs, labels=np.asarray(labels),
                             metadata=json.dumps(meta))
    print(f"Saved {out}: Sun-2018 A1^{{-+}} O shape={obs.shape}; "
          "schemes=s2,s4,s6,fs2,fs4,fs6")


def measure(args):
    if args.operator_basis == "table1-ps":
        measure_table1_ps(args)
        return
    if args.operator_basis in ("sun2018", "bdb"):
        measure_sun2018(args)
        return
    steps = sorted(set(args.smear_steps or [0, 6, 12, 20]))
    if not steps or steps[0] < 0:
        raise ValueError("Require nonnegative smearing levels")
    if args.smear_method == "ape":
        if not 0 <= args.alpha < 1:
            raise ValueError("APE requires 0 <= alpha < 1")
        if len(steps) > 1 and args.alpha == 0:
            raise ValueError("alpha=0 would duplicate all APE smearing levels")
    else:
        if not backend.isfinite(args.stout_rho) or args.stout_rho < 0:
            raise ValueError("Stout requires a finite nonnegative rho")
        if len(steps) > 1 and args.stout_rho == 0:
            raise ValueError("rho=0 would duplicate all Stout smearing levels")
        if _lq_stout_smear_ndarray is None:
            raise ImportError("--smear-method stout requires lqcddb")
    gauge = read_gauge(args.gauge)
    if gauge.ndim != 7 or gauge.shape[0] != 4 or gauge.shape[-2:] != (3, 3):
        raise ValueError("Expected gauge shape (4,Nt,Nz,Ny,Nx,3,3)")
    nt, nz, ny, nx = gauge.shape[1:5]
    if not nz == ny == nx or nx < 4:
        raise ValueError("This reference requires a periodic cubic spatial lattice L >= 4")
    obs = backend.empty((nt, 4*len(steps)))
    labels = [f"loop{a+1}_{args.smear_method}{s}" for s in steps for a in range(4)]
    for t in range(nt):
        links = backend.array(gauge[:3, t], dtype=backend.complex128, copy=True)
        if not backend.all(backend.isfinite(links)):
            raise ValueError(f"Nonfinite links on time slice {t}")

        stout_gauge = _stout_slice(links) if args.smear_method == "stout" else None
        next_level = 0
        for iteration in range(steps[-1]+1):
            if iteration == steps[next_level]:
                current_links = (backend.asarray(stout_gauge[:3, 0])
                                 if stout_gauge is not None else links)
                obs[t, 4*next_level:4*(next_level+1)] = operators(current_links)
                next_level += 1
                if next_level == len(steps):
                    break
            if stout_gauge is not None:
                _stout_step(stout_gauge, args.stout_rho)
            else:
                links = ape_step(links, args.alpha)
        print(f"time slice {t+1}/{nt}", flush=True)
    meta = dict(version=VERSION, cfg_id=args.cfg_id, ensemble=args.ensemble,
                gauge=str(Path(args.gauge).resolve()),
                shape=list(gauge.shape), smear_steps=steps,
                operator_basis="bdb", 
                smear_method=args.smear_method, alpha=args.alpha,
                stout_rho=args.stout_rho,
                prototypes=PROTOTYPES, normalization="1/(48*3*L^3)",
                boundary="periodic", direction_order="X,Y,Z,T")
    out = Path(args.out)
    if out.suffix != ".npz":
        raise ValueError("Measurement output must end in .npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    backend.savez_compressed(out, O=obs, labels=backend.asarray(labels), metadata=json.dumps(meta))
    print(f"Saved {out}: O shape={obs.shape}; no configuration-wise mean subtraction")


def correlation(obs):
    """C_ab(tau)=mean_t O_a(t+tau)O_b(t), with time-reflection symmetrization."""
    nt = len(obs)
    c = np.array([np.roll(obs, -t, axis=0).T @ obs / nt for t in range(nt)])
    return (c + c.swapaxes(1, 2))/2


def read_correlations(manifest):
    manifest = Path(manifest).resolve()
    paths = [line.strip() for line in manifest.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(paths) < 2:
        raise ValueError("Delete-one Jackknife requires at least two configurations")
    correlations, ids, labels, signature = [], set(), None, None
    for item in paths:
        p = Path(item)
        if not p.is_absolute():
            p = manifest.parent / p
        with np.load(p, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            obs = np.asarray(data["O"], dtype=float)
            current_labels = data["labels"].tolist()
        cid = meta["cfg_id"]
        if cid in ids:
            raise ValueError(f"Duplicate configuration id {cid}")
        ids.add(cid)
        sig = {k: v for k, v in meta.items() if k not in ("cfg_id", "gauge")}
        if signature is None:
            signature, labels = sig, current_labels
        elif sig != signature or current_labels != labels:
            raise ValueError(f"Incompatible operator basis/ensemble metadata: {p}")
        if obs.shape != (meta["shape"][1], len(labels)) or not np.all(np.isfinite(obs)):
            raise ValueError(f"Invalid O array: {p}")
        correlations.append(correlation(obs))
    return np.asarray(correlations), labels, signature, len(paths)


def choose_subspace(c0, rcond, rank):
    diagonal = np.diag(c0)
    if np.any(diagonal <= 0):
        raise ValueError("C(t0) has a nonpositive diagonal; check t0 and signal")
    scale = 1/np.sqrt(diagonal)
    eigenvalues, vectors = np.linalg.eigh(scale[:, None]*c0*scale[None, :])
    keep = np.flatnonzero(eigenvalues > rcond*eigenvalues[-1])
    if rank is not None:
        if not 1 <= rank <= len(keep):
            raise ValueError(f"Requested rank {rank}, but only {len(keep)} modes pass rcond")
        keep = keep[-rank:]
    if not len(keep):
        raise ValueError("No usable C(t0) subspace")
    return scale[:, None]*vectors[:, keep], eigenvalues


def gevp(c, t0, td, subspace, reference=None, reference_metric=None):
    if _lq_solve_gevp is not None and _lq_set_backend is not None:
        # lqcddb.solve_gevp uses C(t) v = lambda C(t0) v with layout
        # (operator, operator, time).  The fixed subspace is retained here so
        # rank conditioning and the Jackknife protocol remain unchanged.
        _lq_set_backend("numpy")
        reduced = np.einsum("ai,tab,bj->ijt", subspace, c, subspace, optimize=True)
        reduced_values, reduced_vectors = _lq_solve_gevp(
            reduced, t0, track_states=True, hermitize=True, rcond=1e-12)
        eigenvalues = np.asarray(reduced_values[:, td])
        v = subspace @ np.asarray(reduced_vectors[:, :, td])
    else:
        metric = subspace.T @ c[t0] @ subspace
        ev, u = np.linalg.eigh((metric+metric.T)/2)
        if ev[0] <= 1e-12*ev[-1]:
            raise ValueError("Retained C(t0) is not positive definite in this sample")
        w = subspace @ (u/np.sqrt(ev)[None, :])
        h = w.T @ c[td] @ w
        eigenvalues, v = np.linalg.eigh((h+h.T)/2)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, v = eigenvalues[order], w @ v[:, order]
    overlaps = np.ones(len(eigenvalues))
    if reference is not None:
        cross = reference.T @ reference_metric @ v
        norms = np.sqrt(np.diag(v.T @ reference_metric @ v))
        score = np.abs(cross)/norms[None, :]
        rows, cols = linear_sum_assignment(-score)
        permutation = cols[np.argsort(rows)]
        v, eigenvalues = v[:, permutation], eigenvalues[permutation]
        overlaps = score[np.arange(len(permutation)), permutation]
        signs = np.sign(np.diag(reference.T @ reference_metric @ v))
        v *= np.where(signs == 0, 1, signs)[None, :]
    g = np.einsum("an,tab,bn->tn", v, c, v, optimize=True)
    return eigenvalues, v, g, overlaps


def jackknife_samples(data):
    """Delete one unbinned configuration using lqcddb's Jackknife API."""
    if data.ndim < 1 or len(data) < 2:
        raise ValueError("Delete-one Jackknife requires at least two configurations")
    if _lq_jackknife is not None and _lq_set_backend is not None:
        _lq_set_backend("numpy")
        result = _lq_jackknife(data, Nconf_axes=0, only_sample=True)
        return np.asarray(result["data_sample"])
    return (data.sum(axis=0, keepdims=True) - data)/(len(data)-1)


def effective_mass(g):
    r"""Sun et al.'s logarithmic effective mass $\log[G(t)/G(t+1)]$.

    The final time slice is undefined and is returned as NaN.  This is the
    estimator used in Sun et al. Fig. 3 (their Eq. (13)), rather than the
    symmetric cosh estimator used by the optional fit model below.
    """
    result = np.full_like(g, np.nan, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = g[:-1]/g[1:]
        valid = (ratio > 0) & np.isfinite(ratio)
        result[:-1][valid] = np.log(ratio[valid])
    return result


def jackknife_effective_mass(samples, at_fm):
    """Cosh effective masses and Jackknife errors, preserving invalid NaNs."""
    conversion = HBARC_GEV_FM/at_fm
    if _lq_meff is not None and _lq_set_backend is not None:
        _lq_set_backend("numpy")
        result = _lq_meff(samples, at_fm, Nconf_axes=0, Nt_axes=1,
                          meff_type="log", Atype="JK")
        return (np.asarray(result["data_sample"])/conversion,
                np.asarray(result["data_err"])/conversion,
                np.asarray(result["valid_mask"]))
    values = np.asarray([effective_mass(sample) for sample in samples])
    valid = np.isfinite(values)
    error = np.full(values.shape[1:], np.nan, dtype=float)
    if np.all(valid, axis=0).any():
        full_valid = np.all(valid, axis=0)
        error[full_valid] = (np.sqrt(len(samples)-1)
                             * np.std(values[:, full_valid], axis=0))
    return values, error, valid


def jackknife_covariance(samples):
    """Covariance of delete-one estimators, with the Jackknife prefactor."""
    if samples.ndim != 2 or len(samples) < 2:
        raise ValueError("Jackknife covariance requires at least two samples")
    residual = samples-samples.mean(axis=0, keepdims=True)
    return np.einsum("ni,nj->ij", residual, residual, optimize=True) * (
        (len(samples)-1)/len(samples))


def jackknife_error(samples):
    """Standard error of finite delete-one Jackknife samples."""
    if samples.ndim < 1 or len(samples) < 2 or not np.all(np.isfinite(samples)):
        raise ValueError("Jackknife error requires finite delete-one samples")
    return np.sqrt(len(samples)-1)*np.std(samples, axis=0)


def cosh_fit(y, times, nt, inverse_covariance, max_mass):
    """Analytically profile positive amplitude; fit energy in lattice units."""
    def evaluate(m):
        f = np.exp(-m*(times-times[0])) + np.exp(-m*(nt-times-times[0]))
        amplitude = (f @ inverse_covariance @ y)/(f @ inverse_covariance @ f)
        if amplitude <= 0:
            return 1e100, amplitude
        residual = y-amplitude*f
        return float(residual @ inverse_covariance @ residual), float(amplitude)
    fit = minimize_scalar(lambda m: evaluate(m)[0], bounds=(1e-6, max_mass),
                          method="bounded", options={"xatol": 1e-10})
    chi2, amplitude = evaluate(fit.x)
    if not fit.success or amplitude <= 0 or fit.x < 1e-5 or fit.x > max_mass*(1-1e-5):
        raise ValueError("Cosh fit failed or reached an energy bound; inspect the fit window")
    return float(fit.x), amplitude, chi2


def analyze_jackknife(correlations, t0, td, rank, rcond, states,
                      fit_ranges, at_fm, shrinkage, max_mass):
    if correlations.ndim != 4 or not np.all(np.isfinite(correlations)):
        raise ValueError("Invalid per-configuration correlation matrices")
    ncfg, nt, nop, nop2 = correlations.shape
    if nop != nop2 or ncfg < 2:
        raise ValueError("Require square correlation matrices from at least two configurations")
    if not (0 <= t0 < td < nt//2) or states < 1:
        raise ValueError("Require 0 <= t0 < td < Nt/2 and states >= 1")
    if not 0 <= shrinkage <= 1 or at_fm <= 0 or not 0 < rcond < 1 or max_mass <= 0:
        raise ValueError("Invalid covariance, subspace, mass bound or lattice spacing parameter")
    if fit_ranges is not None and len(fit_ranges) != states:
        raise ValueError("Provide one fit range per requested state")
    for lo, hi in fit_ranges or []:
        if not 1 <= lo <= hi < nt//2 or hi-lo < 2:
            raise ValueError("Fit ranges need >=3 points, excluding t=0 and the thermal midpoint")
        if lo <= td:
            raise ValueError("Use a fit window after td; inspect early-time contamination separately")
    mean = correlations.mean(axis=0)
    basis, metric_spectrum = choose_subspace(mean[t0], rcond, rank)
    if states > basis.shape[1]:
        raise ValueError("Requested more states than retained operator modes")
    lam, vectors, g, _ = gevp(mean, t0, td, basis)
    if np.any(lam[:states] <= 0):
        raise ValueError("Target GEVP eigenvalues are nonpositive")
    projected_jackknife, overlap_samples = [], []
    correlation_jackknife = jackknife_samples(correlations)
    # Fail explicitly instead of silently discarding non-SPD or unstable samples.
    for sample, cb in enumerate(correlation_jackknife):
        try:
            lb, _, gb, ov = gevp(cb, t0, td, basis, vectors, mean[t0])
            if np.any(lb[:states] <= 0):
                raise ValueError("Nonpositive target GEVP eigenvalue")
        except ValueError as error:
            raise ValueError(f"Jackknife sample {sample} failed: {error}. Reassess rank/t0/statistics.") from error
        projected_jackknife.append(gb[:, :states])
        overlap_samples.append(ov[:states])
    projected_jackknife = np.asarray(projected_jackknife)
    meff_samples, meff_error, meff_valid = jackknife_effective_mass(
        projected_jackknife, at_fm)
    energy_jackknife = np.empty((ncfg, states if fit_ranges else 0))
    fits = []
    conversion = HBARC_GEV_FM/at_fm
    for n, (lo, hi) in enumerate(fit_ranges or []):
        times = np.arange(lo, hi+1)
        cov = jackknife_covariance(projected_jackknife[:, lo:hi+1, n])
        variance = np.diag(cov)
        if np.any(variance <= 0):
            raise ValueError("Zero fit variance; check independent configurations")
        sd = np.sqrt(variance)
        corr = cov/sd[:, None]/sd[None, :]
        corr = (1-shrinkage)*corr + shrinkage*np.eye(len(times))
        ev, u = np.linalg.eigh(corr)
        if ev[0] <= 1e-10*ev[-1]:
            raise ValueError("Fit covariance is singular; more configurations or a different window are needed")
        inverse = ((u/ev[None, :]) @ u.T)/sd[:, None]/sd[None, :]
        mass, amplitude, chi2 = cosh_fit(g[lo:hi+1, n], times, nt, inverse, max_mass)
        for sample in range(ncfg):
            try:
                energy_jackknife[sample, n] = cosh_fit(
                    projected_jackknife[sample, lo:hi+1, n], times, nt,
                    inverse, max_mass)[0]
            except ValueError as error:
                raise ValueError(f"Jackknife fit {sample}, state {n} failed: {error}") from error
        error = float(jackknife_error(energy_jackknife[:, n]))
        fits.append(dict(state=n, fit_range=[lo, hi], a_t_E=mass, a_t_E_error=error,
                         E_GeV=mass*conversion, E_GeV_error=error*conversion,
                         shifted_amplitude=amplitude, chi2=chi2, dof=len(times)-2,
                         regularized_chi2_dof=chi2/(len(times)-2)))
    report = dict(version=VERSION, interpretation="finite-volume A1^{-+} levels in a gluonic basis",
                  t0=t0, td=td, rank=basis.shape[1], rcond=rcond, states=states,
                  effective_mass_type="log (Sun et al. Eq. 13)",
                  resampling="delete-one jackknife", jackknife_samples=ncfg, at_fm=at_fm,
                  covariance_shrinkage=shrinkage, fits=fits,
                  mode_overlap_min=np.min(overlap_samples, axis=0).tolist(),
                  mode_overlap_median=np.median(overlap_samples, axis=0).tolist(),
                  note="Errors exclude scale uncertainty, rank/window choices and finite-volume/continuum systematics")
    arrays = dict(C_mean=mean, projected=g[:, :states],
                  projected_jackknife=projected_jackknife,
                  projected_error=jackknife_error(projected_jackknife),
                  effective_mass=effective_mass(g[:, :states]),
                  effective_mass_jackknife=meff_samples,
                  effective_mass_error=meff_error,
                  effective_mass_valid_mask=meff_valid,
                  energy_jackknife=energy_jackknife, coefficients=vectors[:, :states],
                  metric_eigenvalues=metric_spectrum, gevp_eigenvalues=lam,
                  mode_overlaps=np.asarray(overlap_samples))
    return report, arrays


def analyze(args):
    correlations, labels, meta, ncfg = read_correlations(args.manifest)
    windows = ([tuple(map(int, item.split(":"))) for item in args.fit_ranges]
               if args.fit_ranges else None)
    if any(len(w) != 2 for w in windows or []):
        raise ValueError("Fit ranges have inclusive form lo:hi")
    report, arrays = analyze_jackknife(
        correlations, args.t0, args.td, args.rank, args.rcond, args.states,
        windows, args.at_fm, args.cov_shrinkage, args.max_mass)
    report.update(configurations=ncfg, measurement_metadata=meta,
                  manifest=str(Path(args.manifest).resolve()))
    arrays["labels"] = np.asarray(labels)
    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(prefix)+".json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    np.savez_compressed(str(prefix)+".npz", **arrays)
    # A plain table allows inspection before any fit window is selected.
    rows = []
    for n in range(args.states):
        for t in range(len(arrays["projected"])//2+1):
            mb = arrays["effective_mass_jackknife"][:, t, n]
            finite = np.isfinite(mb)
            fraction = float(finite.mean())
            err = arrays["effective_mass_error"][t, n]
            rows.append([n, t, arrays["projected"][t, n],
                         arrays["projected_error"][t, n],
                         arrays["effective_mass"][t, n], err, fraction])
    np.savetxt(str(prefix)+".csv", np.asarray(rows), delimiter=",",
               header="state,t,G,G_error,a_t_meff,a_t_meff_error,valid_jackknife_fraction", comments="")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if min(report["mode_overlap_min"]) < 0.7:
        print("CAUTION: weak Jackknife mode matching; individual excited-level labels may be unstable.")
    if windows is None:
        print("No energy fit requested. Inspect projected correlators/effective masses, then supply --fit-ranges.")
    print("Vary rank, (t0,td), smearing basis, and fit windows before quoting energies.")


def selftest():
    rng = np.random.default_rng(314159)
    def random_su3(shape):
        z = rng.normal(size=shape+(3, 3)) + 1j*rng.normal(size=shape+(3, 3))
        return project_su3(backend.asarray(z))
    def assert_close(a, b):
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-10)
    links, gauge = random_su3((3, 4, 4, 4)), random_su3((4, 4, 4))
    def transform(v, link_length=1):
        return np.array([
            gauge @ v[d] @ dagger(shift(
                gauge, tuple(link_length*n for n in SPATIAL_AXIS_OFFSETS[d])))
            for d in range(3)
        ])
    def rotate_links(v, rotation):
        r = np.zeros((3, 3), int)
        for i, d in enumerate(rotation):
            r[abs(d)-1, i] = 1 if d > 0 else -1
        xyz = np.indices(v.shape[1:4])[::-1]
        old = np.einsum("ij,jzyx->izyx", r.T, xyz)
        output = []
        for d in range(3):
            i = int(np.flatnonzero(r[d])[0])
            pos = old.copy()
            if r[d, i] < 0:
                pos[i] -= 1
            a = v[i, pos[2] % 4, pos[1] % 4, pos[0] % 4]
            output.append(a if r[d, i] > 0 else dagger(a))
        return np.array(output)
    o = operators(links)
    assert np.linalg.norm(o) > 1e-6
    assert_close(operators(transform(links)), o)
    assert_close(operators(links.conj()), o)
    assert_close(operators(rotate_links(links, (-1, -2, -3))), -o)
    for rotation in cubic_rotations():
        assert_close(operators(rotate_links(links, rotation)), o)
    for a, path in enumerate(PROTOTYPES):
        direct = 0.0
        for rotation in cubic_rotations():
            p = rotate_path(path, rotation)
            for q, sign in [(p, 1), (tuple(-d for d in p), -1)]:
                direct += sign*np.trace(loop_matrix(links, q), axis1=-2, axis2=-1).real.mean()
        assert_close(direct/(48*3), o[a])
    smeared = ape_step(links, 0.4)
    assert_close(ape_step(transform(links), 0.4), transform(smeared))
    assert_close(ape_step(rotate_links(links, (-1, -2, -3)), 0.4),
                 rotate_links(smeared, (-1, -2, -3)))
    assert_close(smeared @ dagger(smeared), np.broadcast_to(np.eye(3), smeared.shape))
    assert_close(np.linalg.det(smeared), np.ones(smeared.shape[:-2]))
    # Sun/Chen six-scheme smearing: gauge covariance, unitarity, and 24
    # independent operator columns (four prototypes x six schemes).
    single = sun_single_link_step(links, SUN_LAMBDA_S)
    fuzzy = sun_double_link_step(single, SUN_LAMBDA_F)
    assert_close(sun_single_link_step(transform(links), SUN_LAMBDA_S),
                 transform(single))
    assert_close(sun_double_link_step(transform(single), SUN_LAMBDA_F),
                 transform(fuzzy, link_length=2))
    schemes = sun2018_scheme_links(links)
    assert tuple(name for name, _, _ in schemes) == tuple(
        name for name, _, _ in SUN2018_SCHEMES)
    assert tuple(length for _, _, length in schemes) == (1, 1, 1, 2, 2, 2)
    for (_, scheme_links, link_length), (_, transformed_links, _) in zip(
            schemes,
            sun2018_scheme_links(transform(links))):
        assert_close(transformed_links,
                     transform(scheme_links, link_length=link_length))
        assert_close(scheme_links @ dagger(scheme_links),
                     np.broadcast_to(np.eye(3), scheme_links.shape))
    full_observable = sun2018_operator_matrix(links)
    assert full_observable.shape == (24,)
    assert np.all(np.isfinite(np.asarray(full_observable)))
    samples = np.array([operators(random_su3((3, 4, 4, 4))) for _ in range(12)])
    assert np.linalg.matrix_rank(samples) == 4
    # Correlation convention, time reversal and the identity at t=0.
    obs = rng.normal(size=(12, 4))
    c = correlation(obs)
    assert_close(c[0], obs.T @ obs/len(obs))
    assert_close(c[1], c[-1])
    exponential = np.exp(-0.37*np.arange(8.0))
    np.testing.assert_allclose(effective_mass(exponential)[:-1], 0.37,
                               atol=2e-12, rtol=2e-10)
    assert np.isnan(effective_mass(exponential)[-1])
    # Chen-2006 Table-I path catalog and Type-I pseudoscalar combination.
    gauge4 = random_su3((4, 4, 4, 4, 4))
    assert len(TABLE_I_PATHS) == 12
    for index, _, path, _ in TABLE_I_PATHS:
        for rotation in cubic_rotations()[:3]:
            loop = loop_matrix_4d(gauge4, rotate_path(path, rotation))
            assert loop.shape == gauge4.shape[1:]
        assert index in TABLE_I_PATH_BY_INDEX
    table1_values = table1_pseudoscalar_timeseries(
        gauge4, xi=1., us=1., ut=1.)
    assert np.all(np.isfinite(table1_values))
    # Exact mixed three-state periodic spectrum, with independent smooth
    # amplitude/energy fluctuations per synthetic configuration, validates
    # GEVP, delete-one Jackknife samples and correlated periodic-spectrum fits.
    nt = 32
    true_energies = np.array([0.32, 0.61, 0.95])
    z = np.array([[1., .25, -.15], [.2, .9, .3], [-.1, .2, 1.1]])
    configurations = []
    for _ in range(60):
        energies = true_energies+rng.normal(scale=.002, size=3)
        amplitudes = np.exp(rng.normal(scale=.01, size=3))
        t = np.arange(nt)[:, None]
        f = amplitudes*(np.exp(-t*energies)+np.exp(-(nt-t)*energies))
        configurations.append(np.einsum("an,tn,bn->tab", z, f, z))
    configuration_data = np.array(configurations)
    expected_jackknife = ((configuration_data.sum(axis=0, keepdims=True)
                           - configuration_data)/(len(configuration_data)-1))
    np.testing.assert_allclose(jackknife_samples(configuration_data),
                               expected_jackknife)
    report, arrays = analyze_jackknife(configuration_data, 0, 1, 3, 1e-8, 3,
                                       [(2, 7)]*3, .0897, .1, 3.)
    recovered = np.array([f["a_t_E"] for f in report["fits"]])
    np.testing.assert_allclose(recovered, true_energies, atol=.002, rtol=0)
    assert arrays["energy_jackknife"].shape == (60, 3)
    print("PASS: Sun-2018 solid prototypes; gauge, P, C and all 24 cubic rotations;")
    print("      Morningstar single/double-link six-scheme smearing and covariance;")
    print("      Chen-2006 Table-I paths and Type-I pseudoscalar combination;")
    print("      direct 48-term projection; APE covariance/unitarity; correlator;")
    print("      GEVP, Jackknife mode matching and correlated periodic-spectrum fits.")
    print("Synthetic target a_t E:", true_energies, "recovered:", recovered)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="Measure one configuration")
    m.add_argument("--gauge", required=True)
    m.add_argument("--cfg-id", required=True)
    m.add_argument("--ensemble", required=True, help="Ensemble label, checked when combining files")
    m.add_argument("--out", required=True)
    m.add_argument("--operator-basis", choices=("sun2018", "bdb", "table1-ps"),
                   default="sun2018",
                   help="Sun-2018 24-operator A1^{-+} basis (bdb is a compatibility alias), "
                        "or Chen-2006 Table-I Type-I pseudoscalar")
    m.add_argument("--smear-steps", nargs="+", type=int, default=None,
                   help="Smearing levels; table1-ps defaults to six levels")
    m.add_argument("--smear-method", choices=("ape", "stout"), default="stout",
                   help="Only for table1-ps/legacy paths; sun2018 uses its six fixed "
                        "single/double-link schemes")
    m.add_argument("--alpha", type=float, default=.4)
    m.add_argument("--stout-rho", type=float, default=.12)
    m.add_argument("--sun-lambda-s", type=float, default=SUN_LAMBDA_S,
                   help="Sun/Chen single-link smearing parameter (default 0.1)")
    m.add_argument("--sun-lambda-f", type=float, default=SUN_LAMBDA_F,
                   help="Sun/Chen double-link fuzzing parameter (default 0.5)")
    m.add_argument("--xi", type=float, default=1.0,
                   help="Spatial/temporal anisotropy for table1-ps")
    m.add_argument("--us", type=float, default=None,
                   help="Spatial tadpole factor; table1-ps estimates it when omitted")
    m.add_argument("--ut", type=float, default=1.0,
                   help="Temporal tadpole factor; 2006 convention defaults to 1")
    a = sub.add_parser("analyze", help="Analyze a chronological manifest of measurements")
    a.add_argument("--manifest", required=True)
    a.add_argument("--t0", type=int, default=0)
    a.add_argument("--td", type=int, default=1)
    a.add_argument("--rank", type=int)
    a.add_argument("--rcond", type=float, default=1e-4)
    a.add_argument("--states", type=int, default=2)
    a.add_argument("--fit-ranges", nargs="+", help="Optional inclusive ranges lo:hi, one per state")
    a.add_argument("--at-fm", type=float, required=True)
    a.add_argument("--cov-shrinkage", type=float, default=.1)
    a.add_argument("--max-mass", type=float, default=5.)
    a.add_argument("--out", required=True)
    sub.add_parser("selftest", help="Small-lattice symmetry and synthetic-spectrum checks")
    args = parser.parse_args()
    if args.command == "measure":
        measure(args)
    elif args.command == "analyze":
        analyze(args)
    else:
        selftest()


if __name__ == "__main__":
    main()
