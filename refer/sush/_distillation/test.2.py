import itertools
import numpy as np
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys

from lqcddb import set_backend, get_backend
from lqcddb.io import read_gauge
from lqcddb.base import stout_smear_ndarray
from lqcddb.constant import Nd, Nc

set_backend('numpy')
backend = get_backend()
Lz, Ly, Lx, Lt = [16, 16, 16, 128]

conf_id = sys.argv[1]
conf_name = sys.argv[2]

gauge_path = f'/public/group/lqcd/configurations/aniso_ihep/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05766_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/clqcd_nf2_clov_L16_T128_b2.0_xi5_ml-0.05766_cfg_{conf_id}.lime'
gauge = read_gauge(gauge_path)
gauge = backend.asarray(gauge)

PROTOTYPES = (
    (1, 1, 2, -1, -2, 3, -1, -3),
    (1, 2, 1, 3, -1, -3, -1, -2),
    (1, 1, 2, -1, 3, -2, -1, -3),
    (1, 1, 2, 3, -1, -2, -1, -3),
)

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

def projected_paths(path):
    """Integer coefficients of sum_R [ReTr W(Rp)-ReTr W(-Rp)]."""
    displacement = [sum((1 if d > 0 else -1) for d in path if abs(d) == j) for j in (1, 2, 3)]
    if any(displacement):
        raise ValueError("Prototype is not closed")
    
    weights = Counter()
    for rotation in cubic_rotations():
        q = rotate_path(path, rotation)
        weights[q] += 1
        weights[tuple(-d for d in q)] -= 1
        
    terms = tuple((p, w) for p, w in sorted(weights.items()) if w)
    
    if not terms:
        raise ValueError("Prototype has identically zero A1^{-+} projection")
    return terms

PROJECTED_PATHS = tuple(projected_paths(p) for p in PROTOTYPES)

def shift(field, offset):
    """field[x+offset], with offset ordered (z,y,x)."""
    if not any(offset):
        return field
    return backend.roll(field, tuple(-int(n) for n in offset), axis=(-5, -4, -3))

def dagger(a):
    return a.conj().swapaxes(-1, -2)

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

def project_su3(matrix):
    """Polar reunitarization followed by a common determinant phase removal."""
    u, singular, vh = backend.linalg.svd(matrix)
    if backend.min(singular) < 1e-12:
        raise ValueError("Nearly singular APE matrix; reduce alpha")
    unitary = u @ vh
    return unitary * backend.exp(-1j * backend.angle(backend.linalg.det(unitary))/3)[..., None, None]

def _offset_add(*offsets):
    return tuple(sum(values) for values in zip(*offsets))

SPATIAL_AXIS_OFFSETS = ((0, 0, 1), (0, 1, 0), (1, 0, 0))

def single_link_step(links, lambda_s=0.1):
    r"""Morningstar--Peardon single-link map $s_{\lambda_s}$.

    This is the map used in the six schemes quoted by Chen et al. and Sun et
    al., not the normalized APE parameterization used by ``ape_step``:

    $$U_j^s(x)=\mathcal P_{SU(3)}\left[U_j(x)+\lambda_s
      \sum_{\pm k\ne j}U_{\pm k}(x)U_j(x\!\pm\!\hat k)
      U_{\pm k}^{\dagger}(x\!\pm\!\hat j)\right].$$
    """

    result = backend.empty_like(links)
    for mu, e_mu in enumerate(SPATIAL_AXIS_OFFSETS):
        staples = backend.zeros_like(links[mu])
        for nu, e_nu in enumerate(SPATIAL_AXIS_OFFSETS):
            if nu == mu:
                continue
            plus = (links[nu] @ shift(links[mu], e_nu) @ dagger(shift(links[nu], e_mu)))
            minus = (dagger(shift(links[nu], tuple(-v for v in e_nu)))
                     @ shift(links[mu], tuple(-v for v in e_nu))
                     @ shift(links[nu], _offset_add(e_mu, tuple(-v for v in e_nu)))
                     )
            staples += plus + minus
        result[mu] = project_su3(links[mu] + lambda_s*staples)
    return result


def double_link_step(links, lambda_f=0.5):
    r"""Morningstar--Peardon double-link map $f_{\lambda_f}$.

    The returned array contains gauge-covariant superlinks from $x$ to
    $x+2\hat\jmath$.  Its last three spatial dimensions remain the starting
    site, so :func:`loop_matrix` can consume it with ``link_length=2``.
    """
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
            total += coefficient * backend.trace(w, axis1=-2, axis2=-1).real.sum()
        values.append(total / (48.0 * 3.0))
    return backend.asarray(values)

def correlation(obs):
    """C_ab(tau)=mean_t O_a(t+tau)O_b(t), with time-reflection symmetrization."""
    nt = len(obs)
    c = backend.array([backend.roll(obs, -t, axis=0).T @ obs / nt for t in range(nt)])
    return (c + c.swapaxes(1, 2))/2

A = backend.zeros((Lt, 6 * 4))
# gauge_stout = stout_smear_ndarray(gauge, nstep = 10, rho = 0.12)
for step in range(3):
    for _ in range(2):
        gauge = single_link_step(gauge)
        
    gauge_double = double_link_step(gauge)
    
    for t in range(Lt):
        A[t, step*4:(step+1)*4] = operators(gauge[:, t])
    
    for t in range(Lt):
        A[t, (step + 3)*4:((step + 3)+1)*4] = operators(gauge_double[:, t], link_length=2)

    print(f'step: {step}')
    

backend.save(f'/public/home/sush/distillation/glueball/result/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05766_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/0-+_{conf_id}.npy', correlation(A))
backend.save(f'/public/home/sush/distillation/glueball/result/clqcd_nf2_clov_L16_T128_b2.0_ml-0.05766_sn2_srho0.12_gg5.65_gf5.2_usg0.780268_usf0.949104/0-+_ope_{conf_id}.npy', A)
