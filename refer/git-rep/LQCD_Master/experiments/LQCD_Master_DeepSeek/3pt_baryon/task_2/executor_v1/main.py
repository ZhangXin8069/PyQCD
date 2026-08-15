# Run: mpirun -np 4 python main.py ~/.cache 10000
# Proton axial 3pt: connected-only, u->u axial current (\gamma_1 \gamma_5)
# Point source at [0,0,0,0], zero momentum, tseq=8, P_plus projector
import sys, os
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_utils import core, io, gamma, source

# ── Parameters ──
resource_path = os.path.expanduser(sys.argv[1])
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
x_src = [0, 0, 0, 0]
t_seq = 8

xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# ── Initialize PyQUDA ──
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Load gauge configuration ──
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Stout-smeared copy for Dirac inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Dirac operator (Clover, multigrid CG) ──
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)

# ── Forward light propagator (point source, zero momentum) ──
pt_src = source.source12(latt_info, "point", x_src)
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# ── Gamma matrices and tensors on GPU ──
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)          # \gamma_5
Gamma_cur = cp.asarray(
    gamma.gamma(1) @ gamma.gamma(15), dtype=cp.complex128      # \gamma_1 \gamma_5  (axial, x-dir)
)

# Levi-Civita epsilon_{abc}
eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = 1.0
eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0

Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5                                                # C \gamma_5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_+ = (1+\gamma_4)/2

# ══════════════════════════════════════════════════════════════
# Sink block B: 4 Wick topologies for proton -> proton (u->u)
# Both source u-quarks (u^d in diquark, u^f spectator) and
# both sink u-quarks (u^a diquark, u^c spectator) participate.
# Topologies: (sink u^a vs u^c) x (source u^d vs u^f)
# ══════════════════════════════════════════════════════════════
B = core.LatticePropagator(latt_info)
B.data = (
    + contract(
        "AJ,aic,KH,jge,CF,wtzyxFCec,wtzyxHAga->wtzyxJKij",
        Cg5, eps, Cg5, eps, Tmat, prop_l.data, prop_l.data,
    )
    - contract(
        "AJ,aic,GH,fgj,CK,wtzyxHAga,wtzyxGCfc->wtzyxJKij",
        Cg5, eps, Cg5, eps, Tmat, prop_l.data, prop_l.data,
    )
    - contract(
        "AB,abi,KH,jge,JF,wtzyxFBeb,wtzyxHAga->wtzyxJKij",
        Cg5, eps, Cg5, eps, Tmat, prop_l.data, prop_l.data,
    )
    + contract(
        "AB,abi,GH,fgj,JK,wtzyxHAga,wtzyxGBfb->wtzyxJKij",
        Cg5, eps, Cg5, eps, Tmat, prop_l.data, prop_l.data,
    )
)

# ── First dagger: \tilde{B} = \gamma_5 B^\dagger \gamma_5 ──
B.data = contract("AB, wtzyxCBji, CD -> wtzyxADij", G5, B.data.conj(), G5)

# ── Sequential source at tseq=8 ──
src_seq = source.sequential12(B, t_seq)

# ── Sequential propagator solve ──
with dirac_l.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_l, src_seq)

# ── Second dagger on sequential propagator ──
tmp_prop = core.LatticePropagator(latt_info)
tmp_prop.data = contract(
    "AB, wtzyxCBji, CD -> wtzyxADij", G5, prop_seq.data.conj(), G5
)

# ── Final contraction: Tr[ G_seq^dag  (\gamma_1\gamma_5)  S_l ] ──
three_pt_site = contract(
    "wtzyxijba, jk, wtzyxkiab -> wtzyx",
    tmp_prop.data, Gamma_cur, prop_l.data,
)
C3_t_local = contract("wtzyx -> t", three_pt_site)

# ── MPI gather to rank 0 ──
C3_t = core.gatherLattice(C3_t_local.get(), [0, -1, -1, -1])

# ── Save (rank 0 only): tau = 1 .. 7, one row, no header ──
if core.getMPIRank() == 0:
    C3_window = np.asarray(C3_t[1:8], dtype=np.complex128)  # tau=1..7
    out_path = f"proton_axial_3pt_result_{n_cfg}.txt"
    # One row: Re(tau=1) Im(tau=1) ... Re(tau=7) Im(tau=7)
    out_data = np.column_stack((C3_window.real, C3_window.imag)).reshape(1, -1)
    np.savetxt(out_path, out_data, fmt="%.16e")
