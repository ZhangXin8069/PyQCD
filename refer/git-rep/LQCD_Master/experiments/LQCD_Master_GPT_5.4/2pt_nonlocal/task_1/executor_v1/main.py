import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, source
from pyquda_utils.core import Z

# Run: mpirun -n 4 python3 main.py ~/.cache 10000

# 1. parameter definitions (hard-coded)
resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

x_src = [0, 0, 0, 0]
zmax = 10

anisotropy = 1.0
xi_0 = 1.0
mass_l = -0.277
csw = 1.160920226
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

t_boundary = -1
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_path = f"pion_nonlocal_2pt_cfg_{n_cfg}.txt"

# 2. read gauge configuration
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=t_boundary, anisotropy=anisotropy)

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
# This is a custom bare hybrid correlator:
# - propagator inversion uses stout-smeared links
# - Wilson-line transport uses the original unsmeared links

backend = "cupy"
lt_local = prop_l.data.shape[1]
cz_local = cp.zeros((zmax + 1, lt_local), dtype=cp.complex128)

with gauge_raw.use() as dirac_shift:
    for zsep in range(zmax + 1):
        prop_shift = prop_l.copy()
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_shift.setFermion(tmp, spin, color)

        # FROM generate_einsum (meson_2pt)
        cz_local[zsep] = -contract('wtzyxCBba, wtzyxCBba -> t', prop_l.data.conj(), prop_shift.data)

# 6. save the result
if core.getMPIRank() == 0:
    rows = []
    for zsep in range(zmax + 1):
        ct = core.gatherLattice(array.arrayAsNumpy(cz_local[zsep], backend=backend), [0, -1, -1, -1])
        ct = np.asarray(ct, dtype=np.complex128).reshape(-1)
        for t in range(ct.shape[0]):
            rows.append([zsep, t, ct[t].real, ct[t].imag])
    out = np.asarray(rows, dtype=np.float64)
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
else:
    for zsep in range(zmax + 1):
        core.gatherLattice(array.arrayAsNumpy(cz_local[zsep], backend=backend), [0, -1, -1, -1])
