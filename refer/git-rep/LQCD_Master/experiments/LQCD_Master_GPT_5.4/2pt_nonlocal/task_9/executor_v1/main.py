import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source
from pyquda_utils.core import Z

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

# 1. parameter definitions (hard-coded)
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

x_src = [0, 0, 0, 0]
zmax = 10

anisotropy = 1.0
xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_c = 0.4159
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

t_boundary = -1
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_path = f"dstar_plus_nonlocal_vector_2pt_cfg{n_cfg}.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=t_boundary, anisotropy=anisotropy)

# 2. read gauge configuration
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# 3. construct the Dirac operator
dirac_l = core.getDirac(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_c = core.getDirac(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)

# 4. compute forward propagators
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# 5. extract observable / compute contraction
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
g1 = cp.asarray(gamma.gamma(1), dtype=cp.complex128)
g2 = cp.asarray(gamma.gamma(2), dtype=cp.complex128)
g3 = cp.asarray(gamma.gamma(4), dtype=cp.complex128)

lt_local = prop_l.data.shape[1]
Cx_loc = cp.zeros((zmax + 1, lt_local), dtype=cp.complex128)
Cy_loc = cp.zeros((zmax + 1, lt_local), dtype=cp.complex128)
Cz_loc = cp.zeros((zmax + 1, lt_local), dtype=cp.complex128)

with gauge_raw.use() as dirac_shift:
    for zsep in range(zmax + 1):
        prop_c_shift = prop_c.copy()
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_c_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_c_shift.setFermion(tmp, spin, color)

        # FROM generate_einsum (meson_2pt)
        Cx_loc[zsep] = contract('wtzyxCBba, CD, wtzyxDAba, AB -> t', prop_l.data.conj(), G5 @ g1, prop_c_shift.data, g1 @ G5)
        Cy_loc[zsep] = contract('wtzyxCBba, CD, wtzyxDAba, AB -> t', prop_l.data.conj(), G5 @ g2, prop_c_shift.data, g2 @ G5)
        Cz_loc[zsep] = contract('wtzyxCBba, CD, wtzyxDAba, AB -> t', prop_l.data.conj(), G5 @ g3, prop_c_shift.data, g3 @ G5)

Cx = np.zeros((zmax + 1, latt_size[3]), dtype=np.complex128)
Cy = np.zeros((zmax + 1, latt_size[3]), dtype=np.complex128)
Cz = np.zeros((zmax + 1, latt_size[3]), dtype=np.complex128)

for zsep in range(zmax + 1):
    cx_t = core.gatherLattice(array.arrayAsNumpy(Cx_loc[zsep], backend="cupy"), [0, -1, -1, -1])
    cy_t = core.gatherLattice(array.arrayAsNumpy(Cy_loc[zsep], backend="cupy"), [0, -1, -1, -1])
    cz_t = core.gatherLattice(array.arrayAsNumpy(Cz_loc[zsep], backend="cupy"), [0, -1, -1, -1])
    if core.getMPIRank() == 0:
        Cx[zsep, :] = np.asarray(cx_t, dtype=np.complex128).reshape(-1)
        Cy[zsep, :] = np.asarray(cy_t, dtype=np.complex128).reshape(-1)
        Cz[zsep, :] = np.asarray(cz_t, dtype=np.complex128).reshape(-1)

# 6. save the result
if core.getMPIRank() == 0:
    CT = 0.5 * (Cx + Cy)
    CL = Cz
    Cavg = (Cx + Cy + Cz) / 3.0

    rows = []
    for zsep in range(zmax + 1):
        for t in range(latt_size[3]):
            rows.append([
                n_cfg,
                zsep,
                t,
                Cx[zsep, t].real,
                Cx[zsep, t].imag,
                Cy[zsep, t].real,
                Cy[zsep, t].imag,
                Cz[zsep, t].real,
                Cz[zsep, t].imag,
                CT[zsep, t].real,
                CT[zsep, t].imag,
                CL[zsep, t].real,
                CL[zsep, t].imag,
                Cavg[zsep, t].real,
                Cavg[zsep, t].imag,
            ])

    out = np.asarray(rows, dtype=np.float64)
    np.savetxt(out_path, out, fmt=["%d", "%d", "%d", "%.16e", "%.16e", "%.16e", "%.16e", "%.16e", "%.16e", "%.16e", "%.16e", "%.16e", "%.16e", "%.16e", "%.16e"])
