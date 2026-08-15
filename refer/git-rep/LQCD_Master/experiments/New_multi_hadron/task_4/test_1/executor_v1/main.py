import sys
import numpy as np

from pyquda_utils import core, io, source

# Run: mpirun -n 8 python3 main.py ~/.cache 10000

resource_path = sys.argv[1]
n_cfg = sys.argv[2]

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 2, 4]
backend = "cupy"

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
out_path = "./three_baryon_local_9q_multihadron_2pt.txt"
sink_file = "sink_three_baryon_local_9q_multihadron_2pt.py"

x_src = [0, 0, 0, 0]

anisotropy = 1.0
xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_s = -0.2356
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

core.init(grid_size, latt_size, backend=backend, resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=anisotropy)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

dirac_l = core.getDirac(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getDirac(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

exec(open(sink_file).read())

if core.getMPIRank() == 0:
    two_pt_root = np.asarray(two_pt_result, dtype=np.complex128).reshape(-1)
    t = np.arange(two_pt_root.shape[0], dtype=np.int32)
    out = np.column_stack((t, two_pt_root.real, two_pt_root.imag))
    header = (
        "three_baryon_local_9q_multihadron_2pt\n"
        f"cfg={n_cfg}\n"
        f"cfg_path={cfg_path}\n"
        "operator=((u^T Cgamma5 d) d)*((u^T Cgamma5 d) u)*((u^T Cgamma5 d) s)\n"
        "hadron_blocks=udd:P_plus,uud:P_plus,uds:P_plus\n"
        "source_type=point source_position=[0,0,0,0]\n"
        "gauge_smear=stout nstep=1 rho=0.125 ndim=4\n"
        "grid_size=[1,1,2,4]\n"
        "sink_file=sink_three_baryon_local_9q_multihadron_2pt.py\n"
        "columns: t re im"
    )
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"], header=header)
