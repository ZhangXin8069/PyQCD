import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_path = f"proton_proton_vector_u_gammax_full_cfg{n_cfg:05d}_tseq8.txt"

x_src = [0, 0, 0, 0]
tseq = 8

t_boundary = -1
anisotropy = 1.0
xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

n_noise = 8
t_list = list(range(1, tseq))

cfg_path = cfg_path_template.format(n_cfg=n_cfg)

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
rank = core.getMPIRank()
rng_seed = 13579 + 1000 * rank + n_cfg
latt_info = core.LatticeInfo(latt_size, t_boundary=t_boundary, anisotropy=anisotropy)

gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
pt_src = source.source12(latt_info, "point", x_src)

I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gamma_cur = cp.asarray(gamma.gamma(1), dtype=cp.complex128)
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)

eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
eps[0, 2, 1] = eps[2, 1, 0] = eps[1, 0, 2] = -1.0
epsilon = eps

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

    B = core.LatticePropagator(latt_info)
    B.data = (
        + contract(
            "AJ,aic,KH,jge,CF,wtzyxFCec,wtzyxHAga->wtzyxJKij",
            Cg5,
            eps,
            Cg5,
            eps,
            Tmat,
            prop_l.data,
            prop_l.data,
        )
        - contract(
            "AJ,aic,GH,fgj,CK,wtzyxHAga,wtzyxGCfc->wtzyxJKij",
            Cg5,
            eps,
            Cg5,
            eps,
            Tmat,
            prop_l.data,
            prop_l.data,
        )
        - contract(
            "AB,abi,KH,jge,JF,wtzyxFBeb,wtzyxHAga->wtzyxJKij",
            Cg5,
            eps,
            Cg5,
            eps,
            Tmat,
            prop_l.data,
            prop_l.data,
        )
        + contract(
            "AB,abi,GH,fgj,JK,wtzyxHAga,wtzyxGBfb->wtzyxJKij",
            Cg5,
            eps,
            Cg5,
            eps,
            Tmat,
            prop_l.data,
            prop_l.data,
        )
    )
    B.data = contract("AB, wtzyxCBji, CD -> wtzyxADij", G5, B.data.conj(), G5)
    src_seq = source.sequential12(B, tseq)
    prop_seq = core.invertPropagator(dirac_l, src_seq)

    prop_l_seq_dag = core.LatticePropagator(latt_info)
    prop_l_seq_dag.data = contract("AB, wtzyxCBji, CD -> wtzyxADij", G5, prop_seq.data.conj(), G5)
    three_pt_connected_local = contract(
        "wtzyxijba, jk, wtzyxkiab -> t",
        prop_l_seq_dag.data,
        Gamma_cur,
        prop_l.data,
    )

    two_pt_local = (
        -contract(
            "AB, abc, EF, efd, CD, wtzyxDBdb, wtzyxFAfa, wtzyxECec -> t",
            Cg5,
            epsilon,
            Cg5,
            epsilon,
            Tmat,
            prop_l.data,
            prop_l.data,
            prop_l.data,
        )
        + contract(
            "AB, abc, EF, efd, CD, wtzyxDCdc, wtzyxFAfa, wtzyxEBeb -> t",
            Cg5,
            epsilon,
            Cg5,
            epsilon,
            Tmat,
            prop_l.data,
            prop_l.data,
            prop_l.data,
        )
    )

    loop_local = cp.zeros(prop_l.data.shape[1], dtype=cp.complex128)
    rng = np.random.default_rng(rng_seed)
    noise_shape = tuple(prop_l.data.shape)
    z4_values = np.array([1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 1.0j, 0.0 - 1.0j], dtype=np.complex128)

    for _ in range(n_noise):
        noise = core.LatticePropagator(latt_info)
        noise_host = z4_values[rng.integers(0, 4, size=noise_shape)]
        noise.data = cp.asarray(noise_host, dtype=cp.complex128)
        prop_noise = core.invertPropagator(dirac_l, noise)
        loop_local += contract(
            "wtzyxijba, jk, wtzyxkiab -> t",
            noise.data.conj(),
            Gamma_cur,
            prop_noise.data,
        )

    loop_local /= (n_noise * 12.0)

connected_t = core.gatherLattice(array.arrayAsNumpy(three_pt_connected_local, backend="cupy"), [0, -1, -1, -1])
two_pt_t = core.gatherLattice(array.arrayAsNumpy(two_pt_local, backend="cupy"), [0, -1, -1, -1])
loop_t = core.gatherLattice(array.arrayAsNumpy(loop_local, backend="cupy"), [0, -1, -1, -1])

if rank == 0:
    connected_window = np.asarray(connected_t[t_list], dtype=np.complex128).reshape(-1)
    disconnected_window = np.asarray(two_pt_t[tseq], dtype=np.complex128) * np.asarray(loop_t[t_list], dtype=np.complex128).reshape(-1)
    full_window = connected_window + disconnected_window
    t_ins = np.asarray(t_list, dtype=np.int32)
    tseq_col = np.full(t_ins.shape, tseq, dtype=np.int32)
    out = np.column_stack((tseq_col, t_ins, full_window.real, full_window.imag))
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
