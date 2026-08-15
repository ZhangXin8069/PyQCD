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
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

x_src = [0, 0, 0, 0]
zmax = 10

anisotropy = 1.0
xi_0 = 1.0
mass_s = -0.2356
csw = 1.160920226
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

t_boundary = -1
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_dir = "."
out_path = os.path.join(out_dir, f"eta_s_nonlocal_2pt_zmax10_cfg{int(n_cfg):05d}.txt")

# 2. read gauge configuration
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=t_boundary, anisotropy=anisotropy)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# 3. construct the Dirac operator
# Strange propagator inversion uses stout-smeared links; Wilson-line transport uses original links.
dirac_s = core.getClover(latt_info, mass_s, tol, maxiter, xi_0, csw, csw, multigrid)

# 4. compute forward propagators
pt_src = source.source12(latt_info, "point", x_src)
with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# 5. extract observable / compute contraction
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)

C_loc = cp.zeros((zmax + 1, latt_info.Lt), dtype=cp.complex128)

with gauge_raw.use() as dirac_shift:
    for zsep in range(zmax + 1):
        prop_shift = prop_s.copy()
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_shift.setFermion(tmp, spin, color)

        # FROM generate_einsum (meson_2pt)
        C_loc[zsep] = contract('wtzyxCBba, wtzyxCBba -> t', prop_s.data.conj(), prop_shift.data)

C_full = np.zeros((zmax + 1, latt_size[3]), dtype=np.complex128)
for zsep in range(zmax + 1):
    C_t = core.gatherLattice(array.arrayAsNumpy(C_loc[zsep], backend="cupy"), [0, -1, -1, -1])
    if core.getMPIRank() == 0:
        C_full[zsep, :] = np.asarray(C_t, dtype=np.complex128).reshape(-1)

# 6. save the result
if core.getMPIRank() == 0:
    out = np.zeros((zmax + 1, 2 * latt_size[3]), dtype=np.float64)
    out[:, 0::2] = C_full.real
    out[:, 1::2] = C_full.imag
    np.savetxt(out_path, out, fmt="%.16e")
