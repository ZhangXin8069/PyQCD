# Run: mpirun -np 4 python main.py ~/.cache 10000
import sys
import os
import numpy as np
import cupy as cp
from opt_einsum import contract

from pyquda_utils import core, io, gamma, source
from pyquda_comm import array

# ============================================================================
# 1. Parameter definitions (hard-coded)
# ============================================================================
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

# Clover fermion parameters
m_l = -0.277
csw = 1.160920226
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]
xi_0 = 1.0

# Stout link smearing
stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

# Point source position
x_src = [0, 0, 0, 0]

# Gauge configuration path template
cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# CLI arguments: resource_path and configuration number
resource_path = sys.argv[1]
n_cfg = sys.argv[2]

# ============================================================================
# 2. Read gauge configuration
# ============================================================================
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Apply stout smearing to the gauge links for the Dirac inversion
gauge.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ============================================================================
# 3. Construct the Dirac operator (Clover fermion with multigrid)
# ============================================================================
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)

# ============================================================================
# 4. Compute forward propagator (point source, light quark)
# ============================================================================
pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# ============================================================================
# 5. Extract observable: rho+ 2pt correlator (average over 3 polarizations)
#    Physics: C_rho(t) = (1/3) Σ_i Tr[S† · (γ5·γi) · S · (γi·γ5)]
#    with γi ∈ {γ1, γ2, γ3} for the three spatial polarizations.
#    In the DeGrand-Rossi basis: gamma(1)=γ₁, gamma(2)=γ₂, gamma(4)=γ₃.
# ============================================================================
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)  # γ5

C_t_local = cp.zeros(latt_info.Lt, dtype=cp.complex128)

for gi_bit in [1, 2, 4]:
    gi = cp.asarray(gamma.gamma(gi_bit), dtype=cp.complex128)
    gamma_left = G5 @ gi    # γ5·γi
    gamma_right = gi @ G5   # γi·γ5

    # FROM generate_einsum (meson_2pt):
    # contract S†(prop_l.data.conj()) with left gamma (γ5·γi) and right gamma (γi·γ5)
    C_t_local += contract(
        'wtzyxCBba, CD, wtzyxDAba, AB -> t',
        prop_l.data.conj(),
        gamma_left,
        prop_l.data,
        gamma_right,
    )

C_t_local /= 3.0

# ============================================================================
# 6. Save the result
#    MPI gather: reduce over spatial dimensions, gather time slices
# ============================================================================
C_t = core.gatherLattice(C_t_local.get(), [0, -1, -1, -1])

if core.getMPIRank() == 0:
    C_t_real = np.asarray(C_t, dtype=np.complex128).reshape(-1).real
    out_path = f"rho_2pt_cfg{n_cfg}.txt"
    np.savetxt(out_path, C_t_real, fmt="%.16e")
