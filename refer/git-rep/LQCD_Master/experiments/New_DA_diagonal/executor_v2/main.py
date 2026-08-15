import os
import sys
import gc
import itertools
import numpy as np
import cupy as cp
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, io, gamma, phase_v2
from pyquda_utils.core import X, Y, Z

# Run with: mpirun -n 4 python3 main.py <resource_path> <cfg_number>

resource_path = sys.argv[1]
n_cfg = int(sys.argv[2])

latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
backend = "cupy"

cfg_path_template_coulomb = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/CoulombGaugeFixed/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{cfg}_hyp0_gfixed3.scidac"
out_dir = "output"

source_position = [0, 0, 0, 31]
source_t = 31

mass_l = -0.2770
xi_0 = 1.0
csw = 1.0 / 0.951479**3
tol = 1.0e-6
maxiter = 1000
multigrid = [[4, 4, 4, 4]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

z_max = 12
source_momenta = {
    "p0": [0, 0, 0],
    "pplus1": [1, 1, 1],
    "pplus2": [2, 2, 2],
    "pminus1": [-1, -1, -1],
    "pminus2": [-2, -2, -2],
}

channel_defs = [
    {"name": "P0", "shift_prop": "p0", "anti_prop": "p0", "sink_phase_momentum": [0, 0, 0]},
    {"name": "P2", "shift_prop": "pplus1", "anti_prop": "pminus1", "sink_phase_momentum": [-2, -2, -2]},
    {"name": "P3_a", "shift_prop": "pplus1", "anti_prop": "pminus2", "sink_phase_momentum": [-3, -3, -3]},
    {"name": "P3_b", "shift_prop": "pplus2", "anti_prop": "pminus1", "sink_phase_momentum": [-3, -3, -3]},
]

shift_directions = [
    {"name": "plus_diag", "dirs": [X, Y, Z]},
    {"name": "minus_diag", "dirs": [-X, -Y, -Z]},
]

perm_orders = list(itertools.permutations([0, 1, 2]))

os.makedirs(out_dir, exist_ok=True)

core.init(grid_size, latt_size, backend=backend, resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template_coulomb.format(cfg=n_cfg)
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

gauge_stout = gauge_raw.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

dirac_l = core.getClover(latt_info, mass_l, tol, maxiter, xi_0, csw, csw, multigrid)
momentum_phase_builder = phase_v2.MomentumPhase(latt_info)

propagators = {}
with dirac_l.useGauge(gauge_stout):
    for mom_name, mom in source_momenta.items():
        phase = momentum_phase_builder.getPhase(mom, source_position[:3])
        propagators[mom_name] = core.invert(dirac_l, "wall", source_t, phase.data)

Gamma_src = cp.asarray(gamma.gamma(7) @ gamma.gamma(15), dtype=cp.complex128)
Gamma_snk = cp.asarray(gamma.gamma(15) @ gamma.gamma(7), dtype=cp.complex128)
Gamma_src_bar = cp.asarray(Gamma_src.conj().T @ gamma.gamma(15), dtype=cp.complex128)
Gamma_snk_bar = cp.asarray(gamma.gamma(15) @ Gamma_snk.conj().T, dtype=cp.complex128)

result_data = np.zeros((4, 2, z_max + 1, 1, 1, latt_size[3]), dtype=np.complex128)

phase_fields = {}
for channel in channel_defs:
    phase_fields[channel["name"]] = cp.asarray(
        momentum_phase_builder.getPhase(channel["sink_phase_momentum"], [0, 0, 0]).data,
        dtype=cp.complex128,
    )

# FROM generate_einsum (meson_2pt)
# base local structure: contract('wtzyxCBba, wtzyxCBba -> t', prop_l.data.conj(), prop_l.data)
# for the requested nonlocal operator, keep the same meson 2pt line pairing but insert
# the specified sink/source gamma structures on the shifted and anti lines.

def diagonal_step_average_inplace(prop_out, prop_in, dirs3, dirac_shift):
    prop_out.data[:] = 0
    for perm in perm_orders:
        prop_tmp = prop_in.copy()
        for mu in perm:
            for spin in range(4):
                for color in range(3):
                    tmp = prop_tmp.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, dirs3[mu])
                    prop_tmp.setFermion(tmp, spin, color)
        prop_out.data += prop_tmp.data
        del prop_tmp
    prop_out.data /= 6.0

with gauge_raw.use() as dirac_shift:
    for idir, dir_info in enumerate(shift_directions):
        dirs3 = dir_info["dirs"]
        shifted_cache = {}
        for mom_name in source_momenta:
            shifted_cache[mom_name] = propagators[mom_name].copy()

        for z_sep in range(z_max + 1):
            for ich, channel in enumerate(channel_defs):
                shift_prop = shifted_cache[channel["shift_prop"]]
                anti_prop = propagators[channel["anti_prop"]]
                sink_phase = phase_fields[channel["name"]]

                corr_local = contract(
                                    "wtzyx,wtzyxjiba,jk,wtzyxklba,li->t",
                                    sink_phase,
                                    anti_prop.data.conj(),
                                    Gamma_snk,
                                    shift_prop.data,
                                    Gamma_src,
                                )
                corr_t = core.gatherLattice(array.arrayAsNumpy(corr_local, backend=backend), [0, -1, -1, -1])
                if core.getMPIRank() == 0:
                    result_data[ich, idir, z_sep, 0, 0, :] = np.asarray(corr_t, dtype=np.complex128).reshape(-1)
                del corr_local

            if z_sep < z_max:
                for mom_name in source_momenta:
                    prop_next = core.LatticePropagator(latt_info)
                    prop_next.data = cp.empty_like(shifted_cache[mom_name].data)
                    diagonal_step_average_inplace(prop_next, shifted_cache[mom_name], dirs3, dirac_shift)
                    del shifted_cache[mom_name]
                    shifted_cache[mom_name] = prop_next
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()

        for mom_name in source_momenta:
            del shifted_cache[mom_name]
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

if core.getMPIRank() == 0:
    metadata = {
        "task_name": "pion_nonlocal_2pt_diagonal_wilson_line",
        "observable": "nonlocal charged-pion two-point function with recursive averaged diagonal Wilson-line displacement",
        "cfg_number": int(n_cfg),
        "runtime_cfg_path": cfg_path,
        "ensemble_preset": "C24P29",
        "latt_size": latt_size,
        "grid_size": grid_size,
        "boundary_conditions": {"spatial": "periodic", "temporal": "anti-periodic"},
        "source_position": source_position,
        "source_type": "wall",
        "source_momenta": source_momenta,
        "channel_definitions": channel_defs,
        "shift_directions": [d["name"] for d in shift_directions],
        "z_max": z_max,
        "recursive_diagonal_rule": "z=0 unshifted; z>0 apply one averaged diagonal covDev step recursively to z-1 using 6 axis-order permutations",
        "diagonal_permutations": [list(p) for p in perm_orders],
        "gamma_source_definition": "gamma.gamma(7) @ gamma.gamma(15)",
        "gamma_sink_definition": "gamma.gamma(15) @ gamma.gamma(7)",
        "quark_mass_light": mass_l,
        "clover_coefficient": csw,
        "xi_0": xi_0,
        "solver_tolerance": tol,
        "solver_maxiter": maxiter,
        "multigrid": multigrid,
        "stout_smear": {"nstep": stout_nstep, "rho": stout_rho, "ndim": stout_ndim},
        "transport_links": "same Coulomb-gauge-fixed links before stout smearing",
        "data_shape": list(result_data.shape),
        "axis_labels": ["momentum_channel", "shift_direction", "z_sep", "gamma_sink", "gamma_source", "t"],
        "channel_names": [c["name"] for c in channel_defs],
        "contraction_structure": "nonlocal meson 2pt with generated meson pairing and explicit sink/source gamma insertions on shifted and anti lines, reduced to time",
        "output_format": "npy dictionary with complex data and metadata",
    }
    out_dict = {
        "data": result_data,
        "metadata": metadata,
    }
    out_path = os.path.join(out_dir, f"pion_nonlocal_diag_wl_2pt_cfg{n_cfg}.npy")
    np.save(out_path, out_dict, allow_pickle=True)

gc.collect()
cp.get_default_memory_pool().free_all_blocks()
