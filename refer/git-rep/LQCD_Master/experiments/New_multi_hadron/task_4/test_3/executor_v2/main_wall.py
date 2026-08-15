import sys
import numpy as np
from pyquda_utils import core, io, source

# Run: mpirun -n 8 python3 main.py ~/.cache 10000

resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
preferred_grid_size = [1, 1, 2, 4]
test_grid_size = [1, 1, 1, 4]
# cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
cfg_path_template = f"/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/CoulombGaugeFixed/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}_hyp0_gfixed3.scidac"
x_src = [0, 0, 0, 0]

xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_s = -0.2356
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

t_boundary = -1
anisotropy = 1.0

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

sink_file = "sink_three_baryon_local9q.py"
out_path = f"./output/three_baryon_local9q_cfg{n_cfg}_wall.txt"

mpi_size = core.getMPISize()
if mpi_size == 8:
    grid_size = preferred_grid_size
elif mpi_size == 4:
    grid_size = test_grid_size
else:
    grid_size = preferred_grid_size

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=t_boundary, anisotropy=anisotropy)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# pt_src = source.source12(latt_info, "point", x_src)
wall_source = source.source12(latt_info, source_type="wall", t_srce=x_src[3])

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, wall_source)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, wall_source)

# Full combined 9-quark multi-hadron contraction for ordered baryon blocks udd, uud, uds.
two_pt_result = None
exec(open(sink_file).read())

if core.getMPIRank() == 0:
    corr_t = np.asarray(two_pt_result, dtype=np.complex128).reshape(-1)
    t = np.arange(corr_t.shape[0], dtype=np.int32)
    out = np.column_stack((t, corr_t.real, corr_t.imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
