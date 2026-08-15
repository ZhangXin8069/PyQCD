# Run: mpirun -n 4 python main.py ~/.cache 10000
#
# Bc- -> J/psi three-point function via b->c vector current
# Source: Bc- (anti-c b) with gamma_5, point source [0,0,0,0]
# Sink:   J/psi (anti-c c) with gamma_x, zero-momentum via coherent spatial sum
# Current: c-bar gamma_x b, inserted at all tau in [0, t_seq]
# t_seq = 8

import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# ═══════════════════════════════════════════════════════════════
# 1. Parameters
# ═══════════════════════════════════════════════════════════════
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_dir = ("/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
           "/Configurations/Original")
cfg_path = f"{cfg_dir}/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

x_src = [0, 0, 0, 0]          # point source position
t_seq = 8                     # source-sink separation

# Clover action parameters
xi_0 = 1.0
csw  = 1.160920226

# Quark masses (bare)
m_c = 0.4159                  # charm
m_b = 1.5                     # bottom

# Forward charm solver: multigrid
tol_c     = 1.0e-12
maxiter_c = 2000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Forward bottom solver: direct BiCGStab (no multigrid, heavy quark)
tol_b     = 1.0e-8
maxiter_b = 5000

# Sequential charm solver: direct BiCGStab (relaxed tolerance)
tol_seq     = 1.0e-10
maxiter_seq = 3000

# Stout link smearing
stout_nstep = 1
stout_rho   = 0.125
stout_ndim  = 4

out_dir = "."

# ═══════════════════════════════════════════════════════════════
# 2. Initialize PyQUDA
# ═══════════════════════════════════════════════════════════════
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ═══════════════════════════════════════════════════════════════
# 3. Read gauge configuration
# ═══════════════════════════════════════════════════════════════
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Stout-smeared copy for Dirac inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ═══════════════════════════════════════════════════════════════
# 4. Dirac operators
# ═══════════════════════════════════════════════════════════════
# Charm: multigrid for forward solve
dirac_c = core.getClover(
    latt_info, m_c, tol_c, maxiter_c, xi_0, csw, csw, multigrid)

# Bottom: direct BiCGStab (multigrid=None), heavy quark
dirac_b = core.getClover(
    latt_info, m_b, tol_b, maxiter_b, xi_0, csw, csw, None)

# Sequential charm: direct BiCGStab, relaxed tolerance
dirac_c_seq = core.getClover(
    latt_info, m_c, tol_seq, maxiter_seq, xi_0, csw, csw, None)

# ═══════════════════════════════════════════════════════════════
# 5. Forward propagators
# ═══════════════════════════════════════════════════════════════
pt_src = source.source12(latt_info, "point", x_src)

# Forward charm propagator — spectator anti-c line (source -> sink)
with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# Forward bottom propagator — b quark (source -> current insertion)
with dirac_b.useGauge(gauge_stout):
    prop_b = core.invertPropagator(dirac_b, pt_src)

# ═══════════════════════════════════════════════════════════════
# 6. Three-point contraction
#    FROM generate_einsum (meson_3pt)
#    spectator = c (prop_c), forward = b (prop_b), sequential = c
#    sink Gamma = g1, src Gamma = G5, cur Gamma = g1
# ═══════════════════════════════════════════════════════════════

# --- Gamma matrices (GPU) ---
I4        = cp.eye(4, dtype=cp.complex128)
G5        = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_snk = cp.asarray(gamma.gamma(1),  dtype=cp.complex128)   # gamma_x
Gamma_src = cp.asarray(gamma.gamma(15), dtype=cp.complex128)   # gamma_5
Gamma_cur = cp.asarray(gamma.gamma(1),  dtype=cp.complex128)   # gamma_x

# G5-conjugate gammas:  Gamma_bar = gamma_5 @ Gamma @ gamma_5
Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

# --- Sink block:  B(x) = Gamma_snk_bar @ prop_c @ Gamma_src_bar ---
B = core.LatticePropagator(latt_info)
B.data = contract(
    'AB, wtzyxBCab, CD -> wtzyxADab',
    Gamma_snk_bar, prop_c.data, Gamma_src_bar)

# --- Sequential source at t_sink ---
src_seq = source.sequential12(B, t_seq)

# --- Sequential inversion ---
with dirac_c_seq.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_c_seq, src_seq)

# --- Second dagger:  G_seq_dag = gamma_5 @ S_seq^dag @ gamma_5 ---
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    'AB, wtzyxCBji, CD -> wtzyxADij',
    G5, prop_seq.data.conj(), G5)

# --- Final contraction:  Tr[ G_seq_dag @ Gamma_cur @ S_fwd ] ---
three_pt_local = contract(
    'wtzyxijba, jk, wtzyxkiab -> t',
    tmp_prop.data, Gamma_cur, prop_b.data)

# --- MPI gather ---
C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"), [0, -1, -1, -1])

# ═══════════════════════════════════════════════════════════════
# 7. Save result (rank 0 only)
#    O^dag_Bc = -b-bar gamma_5 c  → factor -1 in correlator
# ═══════════════════════════════════════════════════════════════
if core.getMPIRank() == 0:
    C3_with_sign = -np.asarray(C3_t, dtype=np.complex128)

    out_path = os.path.join(
        out_dir, f"bc_jpsi_3pt_cfg{n_cfg:05d}_tseq{t_seq}.txt")

    tau_vals   = np.arange(t_seq + 1, dtype=np.int32)
    tseq_col   = np.full(tau_vals.shape, t_seq, dtype=np.int32)
    out = np.column_stack((
        tseq_col, tau_vals,
        C3_with_sign[tau_vals].real,
        C3_with_sign[tau_vals].imag))
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])