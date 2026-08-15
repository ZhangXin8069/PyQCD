# Nonlocal D_s two-point correlation function
# D_s+ (c s̄), J^P = 0^-, with spatial separation z = 0..10 along +z
# Point source at [0,0,0,0]
# Wilson line on original gauge links; inversions on stout-smeared links
# Run: mpirun -n 4 python3 main.py <resource_path> <n_cfg>

import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, source
from pyquda_utils.core import Z

# ── Runtime parameters ──────────────────────────────────────
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

# ── Lattice geometry ────────────────────────────────────────
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

# ── Gauge configuration ─────────────────────────────────────
cfg_path = (
    "/public/share/weiwang/clqcd/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/"
    f"beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

# ── Physics parameters ──────────────────────────────────────
x_src = [0, 0, 0, 0]         # source position [x, y, z, t]
z_max = 10                    # maximum z separation for Wilson line

xi_0 = 1.0                    # gauge anisotropy
csw = 1.160920226             # clover coefficient
m_s = -0.2356                 # strange quark mass (kappa)
m_c = 0.4159                  # charm quark mass (kappa)
tol = 1.0e-8
maxiter = 10000

multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

out_path = "ds_nonlocal_2pt.txt"

# ── Initialize PyQUDA ───────────────────────────────────────
core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

# ── Load gauge configuration ────────────────────────────────
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()

# Keep original gauge for Wilson line, create smeared copy for inversions
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Dirac operators ─────────────────────────────────────────
# Charm: BiCGStab (no multigrid — heavy quark, multigrid nullspace vectors
# tuned for light quarks are ineffective and may destabilize convergence)
dirac_c = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, None)

# Strange: multigrid (light enough for effective coarse-grid correction)
dirac_s = core.getClover(latt_info, m_s, tol, maxiter, xi_0, csw, csw, multigrid)

# ── Point source ────────────────────────────────────────────
pt_src = source.source12(latt_info, "point", x_src)

# ── Forward propagators on stout-smeared gauge ──────────────
with dirac_c.useGauge(gauge_stout):
    prop_c = core.invertPropagator(dirac_c, pt_src)

with dirac_s.useGauge(gauge_stout):
    prop_s = core.invertPropagator(dirac_s, pt_src)

# ═══════════════════════════════════════════════════════════════
# Nonlocal shift: Wilson line on ORIGINAL gauge links
# Apply covDev to charm (quark) propagator; strange stays local
# ═══════════════════════════════════════════════════════════════

C_loc = cp.zeros((z_max + 1, latt_info.Lt), dtype=cp.complex128)

with gauge.use() as dirac_shift:
    for zsep in range(z_max + 1):
        # Fresh copy of charm propagator for each z
        prop_shift = prop_c.copy()
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_shift.setFermion(tmp, spin, color)

        # Contract: C(z,t) = Tr[ S_s†(x,t) · S_c_shifted(x,t;z) ]
        # einsum from generate_einsum(type="meson_2pt", antiquark="s", quark="c",
        #                              gamma_snk="gamma5", gamma_src="gamma5")
        C_loc[zsep] = contract(
            "wtzyxCBba, wtzyxCBba -> t",
            prop_s.data.conj(),
            prop_shift.data,
        )

# ── MPI gather (AFTER closing gauge context) ────────────────
C_full = np.zeros((z_max + 1, latt_size[3]), dtype=np.complex128)
for zsep in range(z_max + 1):
    t_field_global = core.gatherLattice(
        array.arrayAsNumpy(C_loc[zsep], backend="cupy"),
        [0, -1, -1, -1],
    )
    if core.getMPIRank() == 0:
        C_full[zsep, :] = t_field_global

# ── Save output (rank 0 only) ───────────────────────────────
if core.getMPIRank() == 0:
    with open(out_path, "w") as f:
        for zsep in range(z_max + 1):
            for t in range(latt_size[3]):
                c_val = C_full[zsep, t]
                f.write(f"{zsep} {t} {c_val.real:.16e} {c_val.imag:.16e}\n")
    print(f"Saved nonlocal D_s correlator to {out_path}")
