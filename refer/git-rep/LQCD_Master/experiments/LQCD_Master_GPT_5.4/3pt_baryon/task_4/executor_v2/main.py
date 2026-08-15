import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

resource_path = os.path.expanduser(sys.argv[1])
n_cfg = sys.argv[2]
cfg_int = int(n_cfg)

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_path = f"c3_lambda_lambda_axial_ss_cfg{n_cfg}_tseq8.txt"

x_src = [0, 0, 0, 0]
tseq = 8

anisotropy = 1.0
xi_0 = 1.0
mass_l = -0.277
mass_s = -0.2356
csw = 1.160920226
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

n_noise = 4

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

dirac_l = core.getClover(latt_info, mass_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, mass_s, tol, maxiter, xi_0, csw, csw, multigrid)

src_point = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, src_point)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, src_point)

I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1) @ gamma.gamma(15), dtype=cp.complex128)

# Epsilon tensor and baryon projector for the Lambda sink block.
eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0

Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)

# Connected Lambda -> Lambda sink block with open strange index.
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

B.data = contract("AB, wtzyxCBji, CD -> wtzyxADij", G5, B.data.conj(), G5)
src_seq = source.sequential12(B, tseq)

with dirac_s.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_s, src_seq)

prop_seq_dag = core.LatticePropagator(latt_info)
prop_seq_dag.data = contract("AB, wtzyxCBji, CD -> wtzyxADij", G5, prop_seq.data.conj(), G5)

three_pt_site_conn = contract(
    "wtzyxijba, jk, wtzyxkiab -> wtzyx",
    prop_seq_dag.data,
    Gamma_cur,
    prop_s.data,
)
C3_conn_local = contract("wtzyx -> t", three_pt_site_conn)
C3_conn = core.gatherLattice(array.arrayAsNumpy(C3_conn_local, backend="cupy"), [0, -1, -1, -1])

# Internal Lambda source-sink block for the disconnected sector.
baryon_block_local = contract(
    "AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFAfa, wtzyxEBeb -> t",
    Cg5,
    eps,
    Cg5,
    eps,
    Tmat,
    prop_s.data,
    prop_l.data,
    prop_l.data,
)
baryon_block = core.gatherLattice(array.arrayAsNumpy(baryon_block_local, backend="cupy"), [0, -1, -1, -1])

# The stochastic loop lives on the local time extent before MPI gather.
lt_local = prop_s.data.shape[1]
loop_t_local = cp.zeros((lt_local,), dtype=cp.complex128)

for noise_idx in range(n_noise):
    eta_noise = source.source12(latt_info, "volume", None)
    cp.random.seed(cfg_int * 1000 + 17 * noise_idx + core.getMPIRank())
    z4_r = 2 * cp.random.randint(0, 2, size=eta_noise.data.shape, dtype=cp.int32) - 1
    z4_i = 2 * cp.random.randint(0, 2, size=eta_noise.data.shape, dtype=cp.int32) - 1
    eta_noise.data[...] = ((z4_r + 1j * z4_i) / np.sqrt(2.0)).astype(cp.complex128)

    with dirac_s.useGauge(gauge_stout):
        prop_noise = core.invertPropagator(dirac_s, eta_noise)

    loop_site = contract(
        "wtzyxijab,jk,wtzyxkiab->wtzyx",
        eta_noise.data.conj(),
        Gamma_cur,
        prop_noise.data,
    )
    loop_t_local += contract("wtzyx -> t", loop_site)

loop_t_local /= n_noise
loop_t = core.gatherLattice(array.arrayAsNumpy(loop_t_local, backend="cupy"), [0, -1, -1, -1])

if core.getMPIRank() == 0:
    C3_conn_root = np.asarray(C3_conn, dtype=np.complex128).reshape(-1)
    baryon_block_root = np.asarray(baryon_block, dtype=np.complex128).reshape(-1)
    loop_root = np.asarray(loop_t, dtype=np.complex128).reshape(-1)

    C3_disc_root = baryon_block_root[tseq] * loop_root
    C3_full_root = C3_conn_root + C3_disc_root

    t_ins = np.arange(tseq + 1, dtype=np.int32)
    tseq_col = np.full(t_ins.shape, tseq, dtype=np.int32)
    out = np.column_stack((
        tseq_col,
        t_ins,
        C3_full_root[t_ins].real,
        C3_full_root[t_ins].imag,
    ))
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
