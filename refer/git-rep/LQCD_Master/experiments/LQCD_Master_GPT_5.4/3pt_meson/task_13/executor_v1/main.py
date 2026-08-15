import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# Run with: mpirun -n 4 python3 main.py <resource_path> <cfg_number>

resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
xi_0 = 1.0
csw = 1.160920226

mass_c = 0.4159
tol_c = 1.0e-10
maxiter_c = 10000

mass_b = 1.5
tol_b = 1.0e-12
maxiter_b = 10000

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

x_src = [0, 0, 0, 0]
tseq = 8

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
cfg_path = cfg_path_template.format(n_cfg=n_cfg)

out_path = f"bc_to_jpsi_c3_cfg{n_cfg}_src0000_gsrcg5_gsnkgx_gcurgx_tseq8.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

if core.getMPIRank() == 0:
    print(f"task=bc_to_jpsi_vector_3pt cfg={n_cfg} cfg_path={cfg_path}")
    print("source=anti-c b gamma_src=gamma5 sink=anti-c c gamma_snk=gamma_x current=cbar gamma_x b tseq=8 source_pos=[0,0,0,0]")
    print("action=clover stout=(1,0.125,4) masses: charm=0.4159 bottom=1.5")

# read gauge and prepare stout links for inversions
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# heavy-clover solves on stout-smeared gauge
dirac_c = core.getClover(latt_info, mass_c, tol_c, maxiter_c, xi_0, csw, csw, None)
dirac_b = core.getClover(latt_info, mass_b, tol_b, maxiter_b, xi_0, csw, csw, None)

pt_src = source.source12(latt_info, "point", x_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

with dirac_b.useGauge(gauge_stout):
    prop_b = core.invertPropagator(dirac_b, pt_src)

# FROM generate_einsum (meson_3pt)
# ═══════════════════════════════════════════════════════
# Meson 3pt:
#   spectator  = c  (prop_c)
#   forward    = b  (prop_b)
#   sequential = c (prop_seq)
#   sink Gamma = g1
#   src Gamma  = G5
#   cur Gamma  = g1
# ═══════════════════════════════════════════════════════

I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_snk = cp.asarray(gamma.gamma(1), dtype=cp.complex128)
Gamma_src = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)

Gamma_snk_bar = G5 @ Gamma_snk @ G5
Gamma_src_bar = G5 @ Gamma_src @ G5

B = core.LatticePropagator(latt_info)
B.data = contract(
    "AB, wtzyxBCab, CD -> wtzyxADab",
    Gamma_snk_bar, prop_c.data, Gamma_src_bar
)

src_seq = source.sequential12(B, tseq)

with dirac_c.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_c, src_seq)

tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij",
    G5, prop_seq.data.conj(), G5
)

three_pt_local = contract(
    "wtzyxijba, jk, wtzyxkiab -> t",
    tmp_prop.data, Gamma_cur, prop_b.data
)

C3_t = core.gatherLattice(
    array.arrayAsNumpy(three_pt_local, backend="cupy"), [0, -1, -1, -1]
)

if core.getMPIRank() == 0:
    t_ins = np.arange(1, tseq, dtype=np.int32)
    c3_window = np.asarray(C3_t[t_ins], dtype=np.complex128).reshape(-1)
    tseq_col = np.full(t_ins.shape, tseq, dtype=np.int32)
    out = np.column_stack((tseq_col, t_ins, c3_window.real, c3_window.imag))
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
