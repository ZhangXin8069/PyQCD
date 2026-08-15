import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, source
from pyquda_utils.core import Z

# Run: mpirun -n 4 python3 main.py <resource_path> <cfg>

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

x_src = [0, 0, 0, 0]
zmax = 10

mass_c = 0.4159
tol = 1.0e-10
maxiter = 10000
xi_0 = 1.0
csw = 1.160920226
multigrid = None

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_name = f"etac_nonlocal_z_2pt_cfg{n_cfg:05d}.txt"
out_path = os.path.join(os.getcwd(), out_name)

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

dirac_c = core.getClover(latt_info, mass_c, tol, maxiter, xi_0, csw, csw, multigrid)

pt_src = source.source12(latt_info, "point", x_src)
with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

lt_local = prop_c.data.shape[1]
cz_t_local = cp.zeros((zmax + 1, lt_local), dtype=cp.complex128)

with gauge_raw.use() as dirac_shift:
    for zsep in range(zmax + 1):
        prop_shift = prop_c.copy()
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_shift.setFermion(tmp, spin, color)

        cz_t_local[zsep] = contract('wtzyxCBba, wtzyxCBba -> t', prop_c.data.conj(), prop_shift.data)

czt = []
for zsep in range(zmax + 1):
    c_t = core.gatherLattice(array.arrayAsNumpy(cz_t_local[zsep], backend="cupy"), [0, -1, -1, -1])
    if core.getMPIRank() == 0:
        c_t = np.asarray(c_t, dtype=np.complex128).reshape(-1)
        czt.append(c_t)

if core.getMPIRank() == 0:
    rows = []
    for zsep in range(zmax + 1):
        c_t = czt[zsep]
        for t in range(c_t.shape[0]):
            rows.append([t, zsep, c_t[t].real, c_t[t].imag])
    np.savetxt(out_path, np.asarray(rows), fmt=["%d", "%d", "%.16e", "%.16e"])
