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
mass_c = 0.4159
csw = 1.160920226
tol = 1.0e-12
maxiter = 10000
multigrid = None

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_dir = "."
out_path = os.path.join(out_dir, f"jpsi_nonlocal_2pt_cfg{n_cfg}.txt")

# 2. read gauge configuration
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# 3. construct the Dirac operator
dirac_c = core.getClover(latt_info, mass_c, tol, maxiter, xi_0, csw, csw, multigrid)

# 4. compute forward propagators
pt_src = source.source12(latt_info, "point", x_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# 5. extract observable / compute contraction
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
g_x = cp.asarray(gamma.gamma(1), dtype=cp.complex128)
g_y = cp.asarray(gamma.gamma(2), dtype=cp.complex128)
g_z = cp.asarray(gamma.gamma(4), dtype=cp.complex128)
gammas = [g_x, g_y, g_z]

lt_local = prop_c.data.shape[1]
C_loc = cp.zeros((zmax + 1, lt_local), dtype=cp.complex128)

with gauge_raw.use() as dirac_shift:
    for zsep in range(zmax + 1):
        prop_shift = prop_c.copy()
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_shift.setFermion(tmp, spin, color)

        c_pol = 0
        for g1 in gammas:
            # FROM generate_einsum (meson_2pt)
            c_pol += contract('wtzyxCBba, CD, wtzyxDAba, AB -> t', prop_c.data.conj(), G5 @ g1, prop_shift.data, g1 @ G5)
        C_loc[zsep] = c_pol / 3.0

C_full = np.zeros((zmax + 1, latt_size[3]), dtype=np.complex128)
for zsep in range(zmax + 1):
    c_t = core.gatherLattice(array.arrayAsNumpy(C_loc[zsep], backend="cupy"), [0, -1, -1, -1])
    if core.getMPIRank() == 0:
        C_full[zsep, :] = np.asarray(c_t, dtype=np.complex128).reshape(-1)

# 6. save the result
if core.getMPIRank() == 0:
    rows = []
    for zsep in range(zmax + 1):
        for t in range(latt_size[3]):
            rows.append([zsep, t, C_full[zsep, t].real, C_full[zsep, t].imag])
    out = np.asarray(rows)
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
