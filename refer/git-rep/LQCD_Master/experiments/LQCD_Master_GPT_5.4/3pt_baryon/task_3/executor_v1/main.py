import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_dir = "."
out_name = f"c3_lambda_lambda_ss_gamma_x_tseq8_cfg{int(n_cfg):05d}.txt"
out_path = os.path.join(out_dir, out_name)

x_src = [0, 0, 0, 0]
tseq = 8
t_list = list(range(1, tseq))

anisotropy = 1.0
xi_0 = 1.0
csw = 1.160920226
mass_l = -0.277
mass_s = -0.2356
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

n_noise = 4
backend = "cupy"

contribution_connected = 0
contribution_disconnected = 1
contribution_full = 2

cfg_path = cfg_path_template.format(n_cfg=n_cfg)

core.init(grid_size, latt_size, backend=backend, resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

# Read gauge configuration
#gauge on host first, then move to device and build the stout-smeared copy used by all inversions
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# Construct the clover Dirac operators for light and strange quarks
dirac_l = core.getClover(latt_info, mass_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, mass_s, tol, maxiter, xi_0, csw, csw, multigrid)

# Compute forward point-source propagators from x_src = [0,0,0,0]
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# Extract the connected 3pt correlator with the current on the strange line
# FROM generate_einsum (baryon_3pt)
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)

eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)

# Zero-momentum Lambda sink block built from the two spectator light propagators
B = core.LatticePropagator(latt_info)
B.data = contract(
    "AB,abi,GH,fgj,JK,wtzyxHAga,wtzyxGBfb->wtzyxJKij",
    Cg5,
    eps,
    Cg5,
    eps,
    Tmat,
    prop_l.data,
    prop_l.data,
)

# First dagger to form the strange sequential source at tseq
B.data = contract("AB,wtzyxCBji,CD->wtzyxADij", G5, B.data.conj(), G5)
src_seq = source.sequential12(B, tseq)

with dirac_s.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_s, src_seq)

G_s_seq_dag = core.LatticePropagator(latt_info)
G_s_seq_dag.data = contract("AB,wtzyxCBji,CD->wtzyxADij", G5, prop_seq.data.conj(), G5)

three_pt_connected_local = contract(
    "wtzyxijba,jk,wtzyxkiab->t",
    G_s_seq_dag.data,
    Gamma_cur,
    prop_s.data,
)
three_pt_connected = core.gatherLattice(
    array.arrayAsNumpy(three_pt_connected_local, backend=backend),
    [0, -1, -1, -1],
)

# Build the internal Lambda baryon block for the disconnected contribution
# FROM generate_einsum (baryon_2pt)
lambda_block_local = contract(
    "AB,abc,EF,efd,CD,wtzyxDCdc,wtzyxFAfa,wtzyxEBeb->t",
    Cg5,
    eps,
    Cg5,
    eps,
    Tmat,
    prop_s.data,
    prop_l.data,
    prop_l.data,
)
lambda_block = core.gatherLattice(
    array.arrayAsNumpy(lambda_block_local, backend=backend),
    [0, -1, -1, -1],
)

# Estimate the disconnected strange loop Tr[gamma_x S_s(x,x)] with stochastic volume-source inversions
loop_t_local = cp.zeros(latt_info.Lt, dtype=cp.complex128)

for _ in range(n_noise):
    eta = source.propagator(latt_info, "volume", None)
    z4 = cp.random.randint(0, 4, size=eta.data.shape, dtype=cp.int32)
    eta.data[...] = cp.take(
        cp.asarray([1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 1.0j, 0.0 - 1.0j], dtype=cp.complex128),
        z4,
    )

    with dirac_s.useGauge(gauge_stout):
        phi = core.invertPropagator(dirac_s, eta)

    loop_site = contract("wtzyxijab,jk,wtzyxkiab->wtzyx", eta.data.conj(), Gamma_cur, phi.data)
    loop_t_local += contract("wtzyx->t", loop_site)

loop_t_local /= n_noise
loop_t = core.gatherLattice(
    array.arrayAsNumpy(loop_t_local, backend=backend),
    [0, -1, -1, -1],
)

# Combine connected and disconnected pieces and save t_ins = 1..7 only
if core.getMPIRank() == 0:
    connected_window = np.asarray(three_pt_connected[t_list], dtype=np.complex128).reshape(-1)
    disconnected_window = np.asarray(lambda_block[tseq] * loop_t[t_list], dtype=np.complex128).reshape(-1)
    full_window = connected_window + disconnected_window

    t_ins = np.asarray(t_list, dtype=np.int32)
    tseq_col = np.full(t_ins.shape, tseq, dtype=np.int32)

    out_connected = np.column_stack(
        (
            np.full(t_ins.shape, contribution_connected, dtype=np.int32),
            tseq_col,
            t_ins,
            connected_window.real,
            connected_window.imag,
        )
    )
    out_disconnected = np.column_stack(
        (
            np.full(t_ins.shape, contribution_disconnected, dtype=np.int32),
            tseq_col,
            t_ins,
            disconnected_window.real,
            disconnected_window.imag,
        )
    )
    out_full = np.column_stack(
        (
            np.full(t_ins.shape, contribution_full, dtype=np.int32),
            tseq_col,
            t_ins,
            full_window.real,
            full_window.imag,
        )
    )

    out = np.vstack((out_connected, out_disconnected, out_full))
    np.savetxt(out_path, out, fmt=["%d", "%d", "%d", "%.16e", "%.16e"])
