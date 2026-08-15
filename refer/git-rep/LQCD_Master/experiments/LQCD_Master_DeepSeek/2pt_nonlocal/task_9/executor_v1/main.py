# Run: mpirun -np 4 python3 main.py ~/.cache 10000
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source
from pyquda_utils.core import X, Y, Z, T

# ===========================================================================
# 1. Parameter definitions (hard-coded)
# ===========================================================================
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72"
    "/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_c = 0.4159
tol = 1.0e-12
maxiter = 20000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

zmax = 10
global_Lt = latt_size[3]         # 72
local_Lt = global_Lt // grid_size[3]  # 18

out_path = f"./Dstar_nonlocal_2pt_cfg{n_cfg}.txt"

# ===========================================================================
# 2. Read gauge configuration
# ===========================================================================
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

# Copy before smearing: gauge_stout for Dirac inversions, gauge_raw for Wilson line
gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ===========================================================================
# 3. Construct the Dirac operators
# ===========================================================================
# Light quark: multigrid solver for critical slowing down
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
# Charm quark: heavy, standard Krylov solver (BiCGStab) without multigrid.
# Note: plan specifies CG, but PyQUDA's getClover with multigrid=None defaults
# to BiCGStab, which is more robust for non-Hermitian Clover operators.
dirac_c = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, None)

# ===========================================================================
# 4. Compute forward propagators (point source at origin, zero momentum)
# ===========================================================================
x_src = [0, 0, 0, 0]
pt_src = source.source12(latt_info, "point", x_src)

# Light propagator on stout-smeared gauge
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# Charm propagator on stout-smeared gauge
with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# ===========================================================================
# 5. Nonlocal shift + contraction to correlator
# ===========================================================================
# Gamma matrices on GPU (DeGrand-Rossi basis)
# gamma(1)=gamma_x, gamma(2)=gamma_y, gamma(4)=gamma_z, gamma(15)=gamma_5
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
g1 = cp.asarray(gamma.gamma(1), dtype=cp.complex128)   # gamma_x
g2 = cp.asarray(gamma.gamma(2), dtype=cp.complex128)   # gamma_y
g4 = cp.asarray(gamma.gamma(4), dtype=cp.complex128)   # gamma_z
gammas = [g1, g2, g4]

# Pre-compute the gamma combinations for the three polarizations
# Physics: Tr[S_l^dag * (G5*gi) * S_c * (gi*G5)]  for gi in {g1, g2, g4}
G5_gi_list = [G5 @ gi for gi in gammas]
gi_G5_list = [gi @ G5 for gi in gammas]

# Storage for correlator C(t, Delta): shape (zmax+1, local_Lt)
C_loc = cp.zeros((zmax + 1, local_Lt), dtype=cp.complex128)

# Covariant displacement on raw (unsmeared) gauge links for the Wilson line.
# The shift is applied to the CHARM propagator (quark, not antiquark).
# For each separation Delta, construct:
#   S_c_shifted(x,t) = W(x, x+zhat*Delta; t) * S_c(x+zhat*Delta, t; 0,0)
# by applying covDev in +Z direction Delta times.
with gauge_raw.use() as dirac_shift:
    for zsep in range(zmax + 1):
        # Fresh copy of charm propagator for each separation
        prop_c_shift = prop_c.copy()
        # Apply covDev zsep times in +Z direction.
        # Periodic spatial BCs are handled automatically by QUDA.
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_c_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_c_shift.setFermion(tmp, spin, color)

        # Contraction: average over three vector polarizations
        # FROM generate_einsum (meson_2pt, with manual gamma insertion):
        #   contract('wtzyxCBba, CD, wtzyxDAba, AB -> t',
        #           prop_l.data.conj(), G5@gi, prop_c_shift.data, gi@G5)
        # The prop_l.data.conj() provides S_l^dag; prop_c_shift is the
        # nonlocally-shifted charm propagator.
        C_z_t = 0
        for G5_gi, gi_G5 in zip(G5_gi_list, gi_G5_list):
            C_z_t += contract(
                "wtzyxCBba, CD, wtzyxDAba, AB -> t",
                prop_l.data.conj(), G5_gi, prop_c_shift.data, gi_G5
            )
        C_z_t /= 3.0
        C_loc[zsep] = C_z_t

# ===========================================================================
# 6. MPI gather and save
# ===========================================================================
C_full = np.zeros((zmax + 1, global_Lt), dtype=np.complex128)
for zsep in range(zmax + 1):
    t_field_global = core.gatherLattice(
        array.arrayAsNumpy(C_loc[zsep], backend="cupy"), [0, -1, -1, -1]
    )
    if core.getMPIRank() == 0:
        C_full[zsep, :] = t_field_global

# Save real part only, no header, format: zsep t Re(C)
if core.getMPIRank() == 0:
    with open(out_path, "w") as f:
        for zsep in range(zmax + 1):
            for t in range(global_Lt):
                f.write(f"{zsep} {t} {C_full[zsep, t].real:.16e}\n")
