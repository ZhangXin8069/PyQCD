import os
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, source
from pyquda_utils.core import Z

# 1. parameter definitions (hard-coded)
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
m_light = -0.277
m_charm = 0.4159
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_path = f"dplus_nonlocal_2pt_cfg{n_cfg}.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)

# 2. read gauge configuration
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# 3. construct the Dirac operator
dirac_light = core.getDirac(latt_info, m_light, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_charm = core.getDirac(latt_info, m_charm, tol, maxiter, xi_0, csw, csw, multigrid)

# 4. compute forward propagators
pt_src = source.source12(latt_info, "point", x_src)

with dirac_light.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_light, pt_src)

with dirac_charm.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_charm, pt_src)

# 5. extract observable / compute contraction
# Nonlocal D+(dbar c) correlator with the +z Wilson-line shift applied only to the charm line.
# The light line is used through gamma5-hermiticity, giving Tr[S_l^dagger S_c^shift].
# FROM generate_einsum (meson_2pt)
# contract('wtzyxCBba, wtzyxCBba -> t', prop_l.conj(), prop_c)

C_loc = cp.zeros((z_max + 1, latt_info.Lt), dtype=cp.complex128)

with gauge_raw.use() as dirac_shift:
    for zsep in range(z_max + 1):
        prop_shift = prop_c.copy()
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_shift.setFermion(tmp, spin, color)

        C_loc[zsep] = contract(
            'wtzyxCBba, wtzyxCBba -> t',
            prop_l.data.conj(),
            prop_shift.data,
        )

C_full = np.zeros((z_max + 1, latt_size[3]), dtype=np.complex128)
for zsep in range(z_max + 1):
    C_t = core.gatherLattice(array.arrayAsNumpy(C_loc[zsep], backend="cupy"), [0, -1, -1, -1])
    if core.getMPIRank() == 0:
        C_full[zsep, :] = np.asarray(C_t, dtype=np.complex128).reshape(-1)

# 6. save the result
if core.getMPIRank() == 0:
    rows = []
    for zsep in range(z_max + 1):
        for t in range(latt_size[3]):
            rows.append([zsep, t, C_full[zsep, t].real, C_full[zsep, t].imag])
    out = np.asarray(rows, dtype=np.float64)
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
