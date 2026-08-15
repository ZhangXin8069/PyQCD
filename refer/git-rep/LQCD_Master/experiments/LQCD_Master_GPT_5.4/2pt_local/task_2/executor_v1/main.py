import os
import sys
import numpy as np
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, source

# Run: mpirun -n 4 python3 main.py <resource_path> <n_cfg>

# 1. parameter definitions (hard-coded)
resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
source_position = [0, 0, 0, 0]

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

anisotropy = 1.0
xi_0 = 1.0
clover_coeff = 1.160920226
m_light = -0.277
m_strange = -0.2356
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_n_steps = 1
stout_rho = 0.125
stout_ndim = 4

out_filename = f"kaon_2pt_cfg{n_cfg}.txt"
out_path = os.path.join(os.getcwd(), out_filename)

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

# 2. read gauge configuration
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_n_steps, stout_rho, stout_ndim)

# 3. construct the Dirac operator
if hasattr(core, "getClover"):
    dirac_light = core.getClover(latt_info, m_light, tol, maxiter, xi_0, clover_coeff, clover_coeff, multigrid)
    dirac_strange = core.getClover(latt_info, m_strange, tol, maxiter, xi_0, clover_coeff, clover_coeff, multigrid)
else:
    dirac_light = core.getDirac(latt_info, m_light, tol, maxiter, xi_0, clover_coeff, clover_coeff, multigrid)
    dirac_strange = core.getDirac(latt_info, m_strange, tol, maxiter, xi_0, clover_coeff, clover_coeff, multigrid)

# 4. compute forward propagators
pt_src = source.source12(latt_info, "point", source_position)

with dirac_light.useGauge(gauge_stout):
    prop_light = core.invertPropagator(dirac_light, pt_src)

with dirac_strange.useGauge(gauge_stout):
    prop_strange = core.invertPropagator(dirac_strange, pt_src)

# 5. extract observable / compute contraction
prop_light = prop_light.data
prop_strange = prop_strange.data

# FROM generate_einsum (meson_2pt)
kaon_2pt_local = contract('wtzyxCBba, wtzyxCBba -> t', prop_light.conj(), prop_strange)

kaon_2pt = core.gatherLattice(array.arrayAsNumpy(kaon_2pt_local, backend="cupy"), [0, -1, -1, -1])

# 6. save the result
if core.getMPIRank() == 0:
    kaon_2pt = np.asarray(kaon_2pt, dtype=np.complex128).reshape(-1)
    t = np.arange(kaon_2pt.shape[0], dtype=np.int32)
    out = np.column_stack((t, kaon_2pt.real, kaon_2pt.imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
