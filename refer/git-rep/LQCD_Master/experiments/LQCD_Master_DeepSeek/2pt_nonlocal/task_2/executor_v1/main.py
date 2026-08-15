# Run: mpirun -n 4 python3 main.py <resource_path> <n_cfg>
import os, sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, source
from pyquda_utils.core import X, Y, Z, T

# ── Parameters ──
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

m_l = -0.277
m_s = -0.2356
csw = 1.160920226
xi_0 = 1.0
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

zmax = 10

# ── Initialize PyQUDA ──
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Load gauge configuration ──
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

# Copy raw gauge before smearing (retain original links for Wilson line)
gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Dirac operators (clover fermions) ──
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# ── Point source at [0,0,0,0] ──
pt_src = source.source12(latt_info, "point", [0, 0, 0, 0])

# ── Forward propagator inversions on stout-smeared gauge ──
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# ── Nonlocal shift + contraction ──
# C_loc[zsep, t] holds the correlator for separation zsep at local timeslice t
C_loc = cp.zeros((zmax + 1, latt_info.Lt), dtype=cp.complex128)

# FROM generate_einsum (meson_2pt, gamma_snk=I4, gamma_src=I4):
# Contract Tr[S_s^dag · S_u_shifted], summed over spatial volume
# Indices: w=parity, t=time, z=Z, y=Y, x=X//2,
#          C=color_snk(3), B=color_src(3), b=spin_snk(4), a=spin_src(4)
einsum_str = "wtzyxCBba, wtzyxCBba -> t"

# covDev on raw (unsmeared) gauge — Wilson line construction
with gauge_raw.use() as dirac_shift:
    for zsep in range(zmax + 1):
        # Shift light-quark (u) propagator by zsep steps in +Z direction
        # \tilde{S}_u(x) = W(x, x+z*\hat{z}) · S_u(x+z*\hat{z}, 0)
        prop_shift = prop_l.copy()
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_shift.setFermion(tmp, spin, color)
        # Contraction: C(z,t) = + sum_x Tr[S_s^dag(x,t;0) · \tilde{S}_u(x,t)]
        C_loc[zsep] = contract(
            einsum_str,
            prop_s.data.conj(),
            prop_shift.data,
        )

# MPI gather (MUST be outside gauge_raw.use() context to avoid deadlock)
C_full = np.zeros((zmax + 1, latt_size[3]), dtype=np.complex128)
for zsep in range(zmax + 1):
    t_field_global = core.gatherLattice(
        array.arrayAsNumpy(C_loc[zsep], backend="cupy"),
        [0, -1, -1, -1]
    )
    if core.getMPIRank() == 0:
        C_full[zsep, :] = t_field_global

# ── Save output ──
# Format: 11 rows (z=0..10), each 72 entries "(re,im)" space-separated
# No header, no trailing newline after last row
if core.getMPIRank() == 0:
    out_path = "kaon_nonlocal_2pt.txt"
    with open(out_path, "w") as f:
        for zsep in range(zmax + 1):
            parts = []
            for t in range(latt_size[3]):
                parts.append(f"({C_full[zsep, t].real:.16e},{C_full[zsep, t].imag:.16e})")
            f.write(" ".join(parts))
            if zsep < zmax:
                f.write("\n")
