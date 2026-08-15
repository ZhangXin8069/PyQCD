# Run: mpirun -np 4 python main.py <resource_path> <cfg_number>
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract

from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ============================================================
# 1. Parameter definitions (hard-coded, per the approved plan)
# ============================================================
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

x_src = [0, 0, 0, 0]          # point source position
t_seq = 8                      # sink time separation

xi_0 = 1.0                     # gauge anisotropy
csw = 1.160920226              # clover coefficient (isotropic)

# Quark masses (Wilson-clover)
m_l = -0.277                   # light (u/d)
m_s = -0.2356                  # strange
m_c = 0.4159                   # charm

tol = 1.0e-12                  # solver tolerance
maxiter = 20000                # max solver iterations
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Stout link smearing for Dirac inversions
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# ============================================================
# 2. Initialize PyQUDA and read gauge configuration
# ============================================================
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Smeared copy for Dirac inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ============================================================
# 3. Construct Dirac operators (Clover fermion)
# ============================================================
dirac_l = core.getDirac(latt_info, m_l, tol, maxiter,
                        xi_0, csw, csw, multigrid)
dirac_c = core.getDirac(latt_info, m_c, tol, maxiter,
                        xi_0, csw, csw, multigrid)
dirac_s = core.getDirac(latt_info, m_s, tol, maxiter,
                        xi_0, csw, csw, multigrid)

# ============================================================
# 4. Compute forward propagators from point source [0,0,0,0]
# ============================================================
pt_src = source.source12(latt_info, "point", x_src)

# Light (u) propagator -- spectator
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# Charm (c) propagator -- forward quark line through the current
with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# ============================================================
# 5. D0 -> K- three-point function via sequential source
#    (code from generate_einsum, type=meson_3pt)
# ============================================================

# FROM generate_einsum (meson_3pt)
#   spectator  = u  (prop_l)
#   forward    = c  (prop_c)
#   sequential = s  (prop_seq)
#   sink Gamma = G5,  src Gamma = G5,  cur Gamma = g1 (gamma_x)

# --- Gamma matrices on GPU (DeGrand-Rossi basis) ---
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)          # gamma_5
Gamma_snk = cp.asarray(gamma.gamma(15), dtype=cp.complex128)   # gamma_5
Gamma_src = cp.asarray(gamma.gamma(15), dtype=cp.complex128)   # gamma_5
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)    # gamma_x (gamma_1)

# G5-conjugate gammas:  Gamma_bar = gamma_5 . Gamma . gamma_5
Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

# --- Step 5a: Sink block B(x) = Gamma_snk_bar . S_u . Gamma_src_bar ---
#     The spectator u-quark line connects source directly to sink.
B = core.LatticePropagator(latt_info)
B.data = contract(
    'AB, wtzyxBCab, CD -> wtzyxADab',
    Gamma_snk_bar, prop_l.data, Gamma_src_bar)

# --- Step 5b: Sequential source at t=t_seq and solve ---
#     Construct full-timeslice source from B at t=8.
src_seq = source.sequential12(B, t_seq)

# Solve for the strange sequential propagator
with dirac_s.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_s, src_seq)

# --- Step 5c: Final contraction ---
#     Two-dagger convention: apply gamma_5 dagger to sequential propagator.
#     tmp = gamma_5 . S_seq^dagger . gamma_5  (second dagger)
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_seq.data.conj(), G5)

#     C_3(t) = Tr[ tmp(t) . Gamma_cur . S_c(t) ]  contracted over spin/color,
#     summed over spatial sites (q=0, no momentum phase).
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    tmp_prop.data, Gamma_cur, prop_c.data)

# --- Step 5d: MPI gather ---
C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"),
    [0, -1, -1, -1])

# ============================================================
# 6. Save result (rank 0 only, plain text, no headers)
# ============================================================
if core.getMPIRank() == 0:
    # C3_t covers all time slices; save the window [0, t_seq] inclusive
    C3_window = np.asarray(C3_t[:t_seq + 1], dtype=np.complex128).reshape(-1)
    t_ins = np.arange(C3_window.shape[0], dtype=np.int32)
    tseq_col = np.full(t_ins.shape, t_seq, dtype=np.int32)
    out = np.column_stack((tseq_col, t_ins, C3_window.real, C3_window.imag))
    out_path = os.path.join(
        os.getcwd(),
        f"d0_to_K-_3pt_vx_cfg{n_cfg:05d}.txt")
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])