import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# Run: mpirun -n 4 python3 main.py <resource_path> <n_cfg>

# ── Parameters ────────────────────────────────────────────
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_c = 0.4159
tol = 1.0e-10
maxiter = 5000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

t_seq = 8
x_src = [0, 0, 0, 0]

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_path = "d0_to_pi_3pt_result.txt"

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

# ── Init ──────────────────────────────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Load gauge ────────────────────────────────────────────
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Dirac operators ───────────────────────────────────────
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_c = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)

# ── Point source ──────────────────────────────────────────
pt_src = source.source12(latt_info, "point", x_src)

# ── Forward light propagator (u spectator) ────────────────
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# ── Forward charm propagator (c -> current) ───────────────
with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# ── Gamma matrices (FROM generate_einsum meson_3pt) ───────
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_snk = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_src = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)

# G5-conjugate gammas:  Γ̄ = γ₅ · Γ · γ₅
Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

# ── Sink block B(x) = Γ̄_snk · S_u · Γ̄_src ────────────────
# spectator = u (prop_l)
B = core.LatticePropagator(latt_info)
B.data = contract(
    'AB, wtzyxBCab, CD -> wtzyxADab',
    Gamma_snk_bar, prop_l.data, Gamma_src_bar)

# ── Sequential source (two-dagger convention) ─────────────
src_seq = source.sequential12(B, t_seq)

# Sequential inversion for d-quark (same Dirac op as light)
with dirac_l.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_l, src_seq)

# ── Final contraction ─────────────────────────────────────
# tmp = γ₅ · S_seq† · γ₅
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_seq.data.conj(), G5)

# Tr[ tmp · Γ_cur · S_c ]  →  time dimension only
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    tmp_prop.data, Gamma_cur, prop_c.data)

# MPI gather (time dim 0, reduce spatial dims)
C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"), [0, -1, -1, -1])

# ── Save ──────────────────────────────────────────────────
if core.getMPIRank() == 0:
    # tau = 0 .. t_seq (inclusive)
    C3_window = np.asarray(C3_t[:t_seq + 1], dtype=np.complex128)
    np.savetxt(out_path,
               np.column_stack((C3_window.real, C3_window.imag)),
               fmt="%.16e %.16e")
