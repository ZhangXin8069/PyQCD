# Run: mpirun -np 4 python main.py ~/.cache 10000
import sys
import numpy as np
from opt_einsum import contract
from pyquda_utils import core, io, gamma, source
from pyquda_comm import array

# =============================================================================
# 1. Parameter definitions
# =============================================================================
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

cfg_path = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/"
    "Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
).format(n_cfg=n_cfg)

# Strange quark mass
m_s = -0.2356
tol = 1e-12
maxiter = 10000

# Clover parameters (isotropic lattice: csw_t = csw_r = csw)
xi_0 = 1.0
csw = 1.160920226

# Multigrid blocking: ensemble stores as [T, Z, Y, X]
# QUDA / PyQUDA expects [X, Y, Z, T] — transpose accordingly
# Level 1: [6,6,6,3] (T,Z,Y,X) -> [3,6,6,6] (X,Y,Z,T)
# Level 2: [4,4,4,6] (T,Z,Y,X) -> [6,4,4,4] (X,Y,Z,T)
multigrid = [[3, 6, 6, 6], [6, 4, 4, 4]]

# Stout link smearing (applied to a copy of the gauge field)
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# Wuppertal Gaussian source smearing
gauss_n_steps = 60
gauss_rho = 2.0

# Source position (point source at the origin)
x_src = [0, 0, 0, 0]

# =============================================================================
# 2. Read gauge configuration
# =============================================================================
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# =============================================================================
# 3. Construct the Dirac operator (stout-smeared gauge copy)
# =============================================================================
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# =============================================================================
# 4. Compute forward propagator (Gaussian-smeared point source)
# =============================================================================
with dirac_s.useGauge(gauge_stout):
    # Create bare point source at [0,0,0,0]
    src_pt = source.propagator(latt_info, "point", x_src)
    # Apply Wuppertal Gaussian smearing to improve ground-state overlap
    src_smear = source.gaussianSmear(src_pt, gauge_stout, gauss_rho, gauss_n_steps)
    # Solve D_s(m_s) * prop_s = src_smear
    prop_s = core.invertPropagator(dirac_s, src_smear)

# =============================================================================
# 5. Extract observable: connected s-bar-s pseudoscalar two-point correlator
#    C_conn(t) = sum_x Tr[S_s^dag(x,t) S_s(x,t)]   (zero-momentum projection)
#
#    FROM generate_einsum (meson_2pt): antiquark=s, quark=s, gamma_snk=gamma5, gamma_src=gamma5
#    Connected diagram: Tr[S_s^dag * S_s] summed over parity, spatial, spin, color
#    Disconnected diagram (equal-order for eta_s!) is NOT computed here.
# =============================================================================
C_t_local = contract(
    "wtzyxCBba, wtzyxCBba -> t",
    prop_s.data.conj(),   # S_s^dag
    prop_s.data,          # S_s
)

# MPI gather: concatenate time slices from all ranks, reduce spatial dims
C_t = core.gatherLattice(array.arrayAsNumpy(C_t_local, backend="cupy"), [0, -1, -1, -1])

# =============================================================================
# 6. Save the result — plain text, no header, no metadata
# =============================================================================
if core.getMPIRank() == 0:
    C_t_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    t_arr = np.arange(C_t_root.shape[0], dtype=np.int32)
    out = np.column_stack((t_arr, C_t_root.real, C_t_root.imag))
    out_path = "eta_s_connected_corr_cfg{n_cfg}.txt".format(n_cfg=n_cfg)
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
