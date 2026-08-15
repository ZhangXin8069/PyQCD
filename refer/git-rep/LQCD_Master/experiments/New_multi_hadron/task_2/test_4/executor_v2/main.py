import os
import sys
import numpy as np
import cupy as cp
from pyquda_comm import array
from pyquda_utils import core, io, source

# Run: mpirun -n 8 python3 main.py ~/.cache 10000

resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 2, 2, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
sink_file = "sink_three_baryon_local_9q.py"

x_src = [0, 0, 0, 0]
xi_0 = 1.0
clover_coeff = 1.160920226
mass_l = -0.277
tol = 1.0e-10
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
out_path = f"./three_baryon_local_9q_cfg{n_cfg}.txt"

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# read gauge and build stout links for inversion
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# common light propagator for all u/d lines
point_src = source.source12(latt_info, "point", x_src)
dirac_l = core.getClover(latt_info, mass_l, tol, maxiter, xi_0, clover_coeff, clover_coeff, multigrid)
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, point_src)

# genuine multi-hadron 9-quark contraction generated in executor_generate stage
exec(open(sink_file).read())

corr_local = cp.asarray(two_pt_result, dtype=cp.complex128)
corr_root = core.gatherLattice(array.arrayAsNumpy(corr_local, backend="cupy"), [0, -1, -1, -1])

if core.getMPIRank() == 0:
    metadata = [
        "# observable: local three-baryon nine-quark two-point function",
        f"# cfg: {n_cfg}",
        f"# cfg_path: {cfg_path}",
        f"# lattice: {' '.join(str(x) for x in latt_size)}",
        f"# grid_size: {' '.join(str(x) for x in grid_size)}",
        f"# source_position: {' '.join(str(x) for x in x_src)}",
        "# quark_flavor: light",
        f"# mass_l: {mass_l}",
        f"# clover_coeff: {clover_coeff}",
        f"# tol: {tol}",
        f"# maxiter: {maxiter}",
        f"# multigrid: {multigrid}",
        f"# stout_smear: n_steps={stout_nstep} rho={stout_rho} ndim={stout_ndim}",
        "# operator: (P_plus[(u^T Cgamma5 d) d]) * (P_plus[(u^T Cgamma5 d) u]) * (P_plus[(u^T Cgamma5 d) d])",
        "# contraction: one genuine multi_hadron_2pt nine-quark correlator",
        f"# sink_file: {sink_file}",
        "# columns: t real imag",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        for line in metadata:
            f.write(line + "\n")
        corr_out = np.asarray(corr_root, dtype=np.complex128).reshape(-1)
        for t in range(corr_out.shape[0]):
            f.write(f"{t} {corr_out[t].real:.16e} {corr_out[t].imag:.16e}\n")
