import os
import sys
import shutil
import numpy as np

# Run: mpirun -n 8 python3 main.py ~/.cache 10000
# Test fallback: mpirun -n 4 python3 main.py ~/.cache 10000

resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
# grid_size = [1, 1, 2, 4]
grid_size = [1, 2, 2, 4]
# cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
cfg_path_template = f"/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/CoulombGaugeFixed/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}_hyp0_gfixed3.scidac"
out_path = f"./output/three_baryon_local9q_multi_hadron_2pt_{n_cfg}_wall.txt"

x_src = [0, 0, 0, 36]
t_src = x_src[3] % latt_size[3]

anisotropy = 1.0
xi_0 = 1.0
mass_l = -0.277
clover_coeff = 1.160920226
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

sink_file = "sink_three_baryon_local9q_multi_hadron_2pt.py"
sink_source_path = "/public/home/tangmen/work_pyquda/LQCD_Master/sink_three_baryon_local9q_multi_hadron_2pt.py"

from pyquda_utils import core, io, source

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)

gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

dirac_l = core.getClover(
    latt_info,
    mass_l,
    tol,
    maxiter,
    xi_0,
    clover_coeff,
    clover_coeff,
    multigrid,
)

# point_source = source.source12(latt_info, "point", x_src)
wall_source = source.source12(latt_info, source_type="wall", t_srce=x_src[3])
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, wall_source)

if core.getMPIRank() == 0 and sink_source_path != sink_file and not os.path.exists(sink_file):
    shutil.copyfile(sink_source_path, sink_file)

# Combined local 9-quark operator:
#   O = [(u^T Cg5 d)d][(u^T Cg5 d)u][(u^T Cg5 d)d]
# One combined multi_hadron_2pt contraction object with per-baryon P_plus projection.
exec(open(sink_file).read())

if core.getMPIRank() == 0:
    if "two_pt_result" in locals():
        two_pt_root = np.asarray(two_pt_result, dtype=np.complex128).reshape(-1)
    elif "twopt" in locals():
        two_pt_root = np.asarray(twopt, dtype=np.complex128).reshape(-1)
    elif "C_t" in locals():
        two_pt_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    else:
        raise NameError("No correlator result found after executing sink_file")
    # Shift correlator to put source time slice at output t=0.
    two_pt_root = np.roll(two_pt_root, -t_src)
    t = np.arange(two_pt_root.shape[0], dtype=np.int32)
    cfg_col = np.full(two_pt_root.shape[0], int(n_cfg), dtype=np.int32)
    out = np.column_stack((cfg_col, t, two_pt_root.real, two_pt_root.imag))
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
