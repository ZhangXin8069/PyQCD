# Run: mpirun -np 4 python3 main.py <resource_path> <cfg_number>
import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_utils import core, io, gamma, source
from pyquda_utils.core import Z

# ── Parameters ────────────────────────────────────────────
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/"
    "Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

x_src = [0, 0, 0, 0]

xi_0 = 1.0
csw = 1.160920226
m_s = -0.2356
m_c = 0.4159
tol = 1.0e-12
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

n_z_max = 10
backend_name = "cupy"

# ── Initialize PyQUDA ─────────────────────────────────────
core.init(grid_size, latt_size, backend=backend_name, resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Load gauge configuration ──────────────────────────────
cfg_path = cfg_path_template.format(n_cfg=n_cfg)
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

# Copy BEFORE smearing: gauge_stout for Dirac inversions, gauge_raw for Wilson line
gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Dirac operators (clover fermions with multigrid) ──────
dirac_s = core.getDirac(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)
dirac_c = core.getDirac(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)

# ── Point source at the origin ────────────────────────────
pt_src = source.source12(latt_info, "point", x_src)

# ── Forward propagators (inverted with stout-smeared gauge) ──
with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

# ── Gamma matrices on GPU (DeGrand–Rossi basis) ───────────
# γ₅ = gamma(15);  γ₁=gamma(1), γ₂=gamma(2), γ₃=gamma(4)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
gamma_vec = [
    cp.asarray(gamma.gamma(1), dtype=cp.complex128),   # γ_x
    cp.asarray(gamma.gamma(2), dtype=cp.complex128),   # γ_y
    cp.asarray(gamma.gamma(4), dtype=cp.complex128),   # γ_z
]

# Pre-compute gamma_snk = γ₅γ_i  and  gamma_src = γ_iγ₅
# C(n_z,t) = –(1/3) Σ_i Tr[ S_s† · (γ₅γ_i) · W·S_c · (γ_iγ₅) ]
gamma_snk_list = [G5 @ gi for gi in gamma_vec]
gamma_src_list = [gi @ G5 for gi in gamma_vec]

# ── Nonlocal shift and contraction ────────────────────────
# Wilson line along +z built from ORIGINAL (unsmeared) gauge links.
# Each n_z: shift charm propagator by n_z covDev(Z) steps, then contract.
lt_local = latt_info.Lt
C_loc = cp.zeros((n_z_max + 1, lt_local), dtype=cp.complex128)

with gauge_raw.use() as dirac_shift:
    for n_z in range(n_z_max + 1):
        # n_z = 0  →  no shift, W = 1, local Ds* correlator
        prop_c_shift = prop_c.copy()
        for _step in range(n_z):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_c_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_c_shift.setFermion(tmp, spin, color)

        # Average over three vector polarisations
        C_pol = cp.zeros(lt_local, dtype=cp.complex128)
        for gamma_snk, gamma_src in zip(gamma_snk_list, gamma_src_list):
            # Einsum layout (from generate_einsum tool, meson_2pt):
            #   prop.data.shape = [w, t, z, y, x//2, spin_snk, spin_src, col_snk, col_src]
            #   "wtzyxjiba"  →  j=spin_snk, i=spin_src, b=col_snk, a=col_src
            #   gamma_snk[j,k] connects S_s†_snk → S_c_snk
            #   gamma_src[l,i] connects S_c_src  → S_s†_src
            C_pol += contract(
                "wtzyxjiba, jk, wtzyxklba, li -> t",
                prop_s.data.conj(),
                gamma_snk,
                prop_c_shift.data,
                gamma_src,
            )
        # Overall minus sign from Wick contraction, 1/3 for polarisation average
        C_loc[n_z] = -C_pol / 3.0

# ── MPI gather (OUTSIDE gauge_raw.use() context to avoid deadlock) ──
C_full = np.zeros((n_z_max + 1, latt_size[3]), dtype=np.complex128)
for n_z in range(n_z_max + 1):
    t_global = core.gatherLattice(C_loc[n_z].get(), [0, -1, -1, -1])
    if core.getMPIRank() == 0:
        C_full[n_z, :] = t_global

# ── Save output (rank 0 only) ─────────────────────────────
# Columns: n_z  t  Re[C(n_z,t)]  Im[C(n_z,t)]
# No header line, plain text in the run directory.
if core.getMPIRank() == 0:
    out_path = f"ds_star_nonlocal_2pt_cfg{n_cfg}.txt"
    with open(out_path, "w") as f:
        for n_z in range(n_z_max + 1):
            for t in range(latt_size[3]):
                val = C_full[n_z, t]
                f.write(f"{n_z} {t} {val.real:.16e} {val.imag:.16e}\n")
