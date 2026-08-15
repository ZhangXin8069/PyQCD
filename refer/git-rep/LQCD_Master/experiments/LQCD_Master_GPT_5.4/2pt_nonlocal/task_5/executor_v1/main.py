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
z_max = 10

mass_l = -0.277
tol = 1.0e-10
maxiter = 10000
xi_0 = 1.0
csw = 1.160920226
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_path = f"rho_nonlocal_2pt_zavg_cfg{n_cfg:05d}.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# 2. read gauge configuration
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# 3. construct the Dirac operator
dirac_l = core.getClover(latt_info, mass_l, tol, maxiter, xi_0, csw, csw, multigrid)

# 4. compute forward propagators
pt_src = source.source12(latt_info, "point", x_src)
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# 5. extract observable / compute contraction
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
g_x = cp.asarray(gamma.gamma(1), dtype=cp.complex128)
g_y = cp.asarray(gamma.gamma(2), dtype=cp.complex128)
g_z = cp.asarray(gamma.gamma(4), dtype=cp.complex128)

gammas = [g_x, g_y, g_z]

C_local = cp.zeros((z_max + 1, latt_info.Lt), dtype=cp.complex128)
C_x_local = cp.zeros((z_max + 1, latt_info.Lt), dtype=cp.complex128)
C_y_local = cp.zeros((z_max + 1, latt_info.Lt), dtype=cp.complex128)
C_z_local = cp.zeros((z_max + 1, latt_info.Lt), dtype=cp.complex128)

with gauge_raw.use() as dirac_shift:
    for zsep in range(z_max + 1):
        prop_shift = prop_l.copy()
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_shift.setFermion(tmp, spin, color)

        # FROM generate_einsum (meson_2pt)
        cx = contract('wtzyxCBba, CD, wtzyxDAba, AB -> t', prop_l.data.conj(), G5 @ g_x, prop_shift.data, g_x @ G5)
        cy = contract('wtzyxCBba, CD, wtzyxDAba, AB -> t', prop_l.data.conj(), G5 @ g_y, prop_shift.data, g_y @ G5)
        cz = contract('wtzyxCBba, CD, wtzyxDAba, AB -> t', prop_l.data.conj(), G5 @ g_z, prop_shift.data, g_z @ G5)

        C_x_local[zsep] = cx
        C_y_local[zsep] = cy
        C_z_local[zsep] = cz
        C_local[zsep] = (cx + cy + cz) / 3.0

# 6. save the result
C_full = np.zeros((z_max + 1, latt_size[3]), dtype=np.complex128)
for zsep in range(z_max + 1):
    gathered = core.gatherLattice(array.arrayAsNumpy(C_local[zsep], backend="cupy"), [0, -1, -1, -1])
    if core.getMPIRank() == 0:
        C_full[zsep, :] = np.asarray(gathered, dtype=np.complex128).reshape(-1)

if core.getMPIRank() == 0:
    rows = []
    for zsep in range(z_max + 1):
        for t in range(latt_size[3]):
            rows.append([zsep, t, C_full[zsep, t].real, C_full[zsep, t].imag])
    out = np.asarray(rows)
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])