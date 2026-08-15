import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source
from pyquda_utils.core import Z

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

x_src = [0, 0, 0, 0]
z_max = 10

anisotropy = 1.0
xi_0 = 1.0
csw = 1.160920226
m_s = -0.2356
m_c = 0.4159
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_path = f"dsstar_nonlocal_vec_2pt_cfg{n_cfg}.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
gamma_list = [
    cp.asarray(gamma.gamma(1), dtype=cp.complex128),
    cp.asarray(gamma.gamma(2), dtype=cp.complex128),
    cp.asarray(gamma.gamma(4), dtype=cp.complex128),
]

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_c = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)

pt_src = source.source12(latt_info, "point", x_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

lt_local = prop_s.data.shape[1]
cz_local = cp.zeros((z_max + 1, lt_local), dtype=cp.complex128)

with gauge_raw.use() as dirac_shift:
    for zsep in range(z_max + 1):
        prop_c_shift = prop_c.copy()
        for spin in range(4):
            for color in range(3):
                tmp = prop_c.getFermion(spin, color)
                for _ in range(zsep):
                    tmp = dirac_shift.covDev(tmp, Z)
                prop_c_shift.setFermion(tmp, spin, color)

        c_pol_local = 0
        for g1 in gamma_list:
            # FROM generate_einsum (meson_2pt)
            c_pol_local += contract('wtzyxCBba, CD, wtzyxDAba, AB -> t', prop_s.data.conj(), G5 @ g1, prop_c_shift.data, g1 @ G5)
        cz_local[zsep] = c_pol_local / 3.0

cz_full = np.zeros((z_max + 1, latt_size[3]), dtype=np.complex128)
for zsep in range(z_max + 1):
    ct = core.gatherLattice(array.arrayAsNumpy(cz_local[zsep], backend="cupy"), [0, -1, -1, -1])
    if core.getMPIRank() == 0:
        cz_full[zsep] = np.asarray(ct, dtype=np.complex128).reshape(-1)

if core.getMPIRank() == 0:
    rows = []
    for zsep in range(z_max + 1):
        for t in range(latt_size[3]):
            rows.append([zsep, t, cz_full[zsep, t].real, cz_full[zsep, t].imag])
    out = np.asarray(rows, dtype=np.float64)
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
