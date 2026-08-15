# Run: mpirun -n 4 python3 main.py ~/.cache 10000

import sys
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, source
from pyquda_utils.core import Z

# ── Parameters ──
resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]

cfg_path_template = (
    "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/"
    "Configurations/Original/"
    "beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

x_src = [0, 0, 0, 0]

xi_0 = 1.0
csw = 1.160920226
m_l = -0.277
m_c = 0.4159
tol = 1.0e-08
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

# Preserve original gauge for Wilson line; stout-smear a copy for inversions
gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

# ── Dirac operators (clover fermions + multigrid) ──
dirac_l = core.getClover(latt_info, m_l, tol, maxiter, xi_0, csw, csw, multigrid)

# ── Point source ──
pt_src = source.source12(latt_info, "point", x_src)

# ── Forward light propagator (stout-smeared gauge, multigrid) ──
with dirac_l.useGauge(gauge_stout):
    prop_l = core.invertPropagator(dirac_l, pt_src)

# ── Forward charm propagator: try multigrid, fall back to plain CG ──
dirac_c_mg = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, multigrid)
try:
    with dirac_c_mg.useGauge(gauge_stout):
        prop_c = core.invertPropagator(dirac_c_mg, pt_src)
except Exception:
    if core.getMPIRank() == 0:
        print("Charm multigrid failed, falling back to plain CG")
    dirac_c_cg = core.getClover(latt_info, m_c, tol, maxiter, xi_0, csw, csw, None)
    with dirac_c_cg.useGauge(gauge_stout):
        prop_c = core.invertPropagator(dirac_c_cg, pt_src)

# ── Nonlocal correlator ──
# C(z,t) = Σ_x Tr[ S_l^†(x,t) · W(x,x+z·e_z,t) · S_c(x+z·e_z,t) ]
# covDev on raw gauge builds the Wilson line W step-by-step.
# After zsep covDev steps:  prop_c_shifted(x) = W(x, x+z·e_z) · S_c(x+z·e_z)

Lt_local = latt_info.Lt
C_loc = cp.zeros((zmax + 1, Lt_local), dtype=cp.complex128)

with gauge_raw.use() as dirac_shift:
    for zsep in range(zmax + 1):
        prop_c_shifted = prop_c.copy()
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_c_shifted.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_c_shifted.setFermion(tmp, spin, color)

        # FROM generate_einsum (meson_2pt)
        C_loc[zsep] = contract(
            "wtzyxCBba, wtzyxCBba -> t",
            prop_l.data.conj(),
            prop_c_shifted.data,
        )

# ── MPI gather (outside gauge context to avoid deadlock) ──
C_full = np.zeros((zmax + 1, latt_size[3]), dtype=np.complex128)
for zsep in range(zmax + 1):
    t_field = core.gatherLattice(
        array.arrayAsNumpy(C_loc[zsep], backend="cupy"),
        [0, -1, -1, -1]
    )
    if core.getMPIRank() == 0:
        C_full[zsep, :] = t_field

# ── Save result (rank 0 only) ──
if core.getMPIRank() == 0:
    out_path = f"dplus_nonlocal_2pt_cfg{n_cfg}.txt"
    with open(out_path, "w") as f:
        for zsep in range(zmax + 1):
            for t in range(latt_size[3]):
                val = C_full[zsep, t]
                f.write(f"{zsep} {t} {val.real:.16e} {val.imag:.16e}\n")
    print(f"Saved to {out_path}")
