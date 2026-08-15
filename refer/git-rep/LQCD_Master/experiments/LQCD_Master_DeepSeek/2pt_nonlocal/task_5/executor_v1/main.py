# Run: mpirun -np 4 python3 main.py <resource_path> <cfg_number>
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_utils import core, io, gamma, source
from pyquda_utils.core import X, Y, Z, T

# ============================================================
# 1. Parameter definitions
# ============================================================

# MPI / lattice
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

# Quark action (clover-improved Wilson)
mass    = -0.277
csw     = 1.160920226
xi_0    = 1.0
tol     = 1.0e-12
maxiter = 1000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

# Gauge link smearing (for Dirac inversion)
stout_nstep = 1
stout_rho   = 0.125
stout_ndim  = 4

# Source
x_src = [0, 0, 0, 0]   # point source at origin

# Nonlocal shift
zmax = 10              # maximum separation in +z direction

# Gamma polarizations for rho (gamma_x, gamma_y, gamma_z)
gamma_bits = [1, 2, 4]   # gamma(1)=γ₁, gamma(2)=γ₂, gamma(4)=γ₃

# CLI arguments
resource_path = sys.argv[1]
n_cfg         = int(sys.argv[2])

# Config path
cfg_path = (
    "/public/share/weiwang/clqcd/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/"
    f"beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# Output
out_path = "rho_nonlocal_2pt.txt"

# ============================================================
# 2. Read gauge configuration
# ============================================================

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Keep a copy of the original (unsmeared) gauge field for the Wilson line
gauge_raw = gauge.copy()

# Smear a second copy for the Dirac operator
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ============================================================
# 3. Construct the Dirac operator
# ============================================================

dirac_l = core.getDirac(
    latt_info, mass, tol, maxiter, xi_0, csw, csw, multigrid
)

# ============================================================
# 4. Compute forward propagator (point source)
# ============================================================

pt_src = source.source12(latt_info, "point", x_src)

with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# ============================================================
# 5. Nonlocal contraction: Wilson line + rho correlator
# ============================================================

# Build gamma matrices on GPU
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)  # γ₅

Gi_list = []
for b in gamma_bits:
    Gi_list.append(cp.asarray(gamma.gamma(b), dtype=cp.complex128))

Lt_local = latt_info.Lt
C_loc = cp.zeros((zmax + 1, Lt_local), dtype=cp.complex128)

# Covariant displacement on raw (unsmeared) gauge links
with gauge_raw.use() as dirac_shift:
    for zsep in range(zmax + 1):
        # Build shifted propagator: apply zsep covDev steps in +z
        if zsep == 0:
            prop_shift = prop_l
        else:
            prop_shift = prop_l.copy()
            for _ in range(zsep):
                for spin in range(4):
                    for color in range(3):
                        tmp = prop_shift.getFermion(spin, color)
                        tmp = dirac_shift.covDev(tmp, Z)
                        prop_shift.setFermion(tmp, spin, color)

        # Contract for each polarization and average
        # C_i(z,t) = Tr[ S_l^dag * (γ₅γ_i) * W·S_l * (γ_iγ₅) ]
        C_t_pol = cp.zeros(Lt_local, dtype=cp.complex128)
        for Gi in Gi_list:
            C_t_pol += contract(
                "wtzyxjiba, jk, wtzyxklba, li -> t",
                prop_l.data.conj(),      # S_l^dag
                G5 @ Gi,                 # γ₅ γ_i
                prop_shift.data,         # W(x,x+ẑ) · S_l(x+ẑ)
                Gi @ G5,                 # γ_i γ₅
            )
        C_t_pol /= 3.0
        C_loc[zsep] = C_t_pol

# ============================================================
# 6. MPI gather and save result
# ============================================================

# Gather t-sliced correlator from all MPI ranks to rank 0
C_full = np.zeros((zmax + 1, latt_size[3]), dtype=np.complex128)
for zsep in range(zmax + 1):
    t_field_global = core.gatherLattice(C_loc[zsep].get(), [0, -1, -1, -1])
    if core.getMPIRank() == 0:
        C_full[zsep, :] = t_field_global

if core.getMPIRank() == 0:
    with open(out_path, "w") as f:
        for zsep in range(zmax + 1):
            for t in range(latt_size[3]):
                f.write(f"{zsep} {t} {C_full[zsep, t].real:.16e}\n")
