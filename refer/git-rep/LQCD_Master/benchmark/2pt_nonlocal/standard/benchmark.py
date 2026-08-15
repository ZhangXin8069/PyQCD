# Run: mpirun -np 4 python nonlocal_2pt_LQCD_master_benchmark.py <resource_path> <cfg_num>
import gc
import os
import sys

import cupy as cp
import numpy as np
from opt_einsum import contract
from pyquda_utils import core, gamma, io, phase_v2, source

script_dir = os.path.dirname(os.path.abspath(__file__))
default_resource_path = os.path.join(script_dir, ".cache")

if len(sys.argv) >= 3:
    resource_path = sys.argv[1]
    cfg_num = str(sys.argv[2])
elif len(sys.argv) == 2:
    resource_path = default_resource_path
    cfg_num = str(sys.argv[1])
else:
    resource_path = default_resource_path
    cfg_num = "10000"

os.makedirs(resource_path, exist_ok=True)

# Ensemble / solver parameters
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

m_l = -0.2770
m_s = -0.2356
m_c = 0.4159
xi_0 = 1.0
csw = 1.160920226
tol_l = 1.0e-10
tol_s = 1.0e-10
tol_c = 1.0e-14
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

source_position = [0, 0, 0, 0]
z_max = 10
shift_directions = [2]

out_dir = os.path.join(script_dir, "benchmark")
os.makedirs(out_dir, exist_ok=True)

tasks = [
    {
        "id": 1,
        "label": "pion",
        "particle": "pion",
        "source": source_position,
        "flavors": ["l"],
        "operator": "pi_plus_nonlocal_ps",
        "compute": "pion",
        "task_note": "task_1 nonlocal shift on quark propagator with original gauge links",
    },
    {
        "id": 2,
        "label": "kaon",
        "particle": "kaon",
        "source": source_position,
        "flavors": ["l", "s"],
        "operator": "kaon_nonlocal_ps",
        "compute": "kaon",
        "task_note": "task_2 nonlocal shift on light quark propagator with original gauge links",
    },
    {
        "id": 3,
        "label": "eta_s",
        "particle": "eta_s",
        "source": source_position,
        "flavors": ["s"],
        "operator": "eta_s_nonlocal_ps",
        "compute": "eta_s",
        "task_note": "task_3 nonlocal shift on strange quark propagator with original gauge links",
    },
    {
        "id": 4,
        "label": "eta_c",
        "particle": "eta_c",
        "source": source_position,
        "flavors": ["c"],
        "operator": "eta_c_nonlocal_ps",
        "compute": "eta_c",
        "task_note": "task_4 nonlocal shift on charm quark propagator with original gauge links",
    },
    {
        "id": 5,
        "label": "rho",
        "particle": "rho",
        "source": source_position,
        "flavors": ["l"],
        "operator": "rho_nonlocal_vector_avg_spatial",
        "compute": "rho",
        "task_note": "task_5 nonlocal shift on quark propagator with spatial gamma average",
    },
    {
        "id": 6,
        "label": "D",
        "particle": "D_plus",
        "source": source_position,
        "flavors": ["l", "c"],
        "operator": "D_plus_nonlocal_ps",
        "compute": "D",
        "task_note": "task_6 nonlocal shift on charm quark propagator with original gauge links",
    },
    {
        "id": 7,
        "label": "D_s",
        "particle": "D_s",
        "source": source_position,
        "flavors": ["s", "c"],
        "operator": "D_s_nonlocal_ps",
        "compute": "D_s",
        "task_note": "task_7 nonlocal shift on charm quark propagator with original gauge links",
    },
    {
        "id": 8,
        "label": "Jpsi",
        "particle": "Jpsi",
        "source": source_position,
        "flavors": ["c"],
        "operator": "Jpsi_nonlocal_vector_avg_spatial",
        "compute": "Jpsi",
        "task_note": "task_8 nonlocal shift on quark propagator with spatial gamma average",
    },
    {
        "id": 9,
        "label": "Dstar",
        "particle": "D_star_plus",
        "source": source_position,
        "flavors": ["l", "c"],
        "operator": "D_star_plus_nonlocal_vector_avg_spatial",
        "compute": "D_star",
        "task_note": "task_9 nonlocal shift on charm quark propagator with spatial gamma average",
    },
    {
        "id": 10,
        "label": "Dsstar",
        "particle": "D_s_star",
        "source": source_position,
        "flavors": ["s", "c"],
        "operator": "D_s_star_nonlocal_vector_avg_spatial",
        "compute": "D_s_star",
        "task_note": "task_10 nonlocal shift on charm quark propagator with spatial gamma average",
    },
]

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=cfg_num)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)
gauge_shift = gauge.copy()

G5 = cp.asarray(gamma.gamma(15))
Gx = cp.asarray(gamma.gamma(1))
Gy = cp.asarray(gamma.gamma(2))
Gz = cp.asarray(gamma.gamma(4))
vector_gamma_list = [Gx, Gy, Gz]
mom_phase = phase_v2.MomentumPhase(latt_info).getPhase([0, 0, 0], [0, 0, 0])

dirac_l = core.getDirac(latt_info, m_l, tol_l, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getDirac(latt_info, m_s, tol_s, maxiter, xi_0, csw, csw, multigrid)
dirac_c = core.getDirac(latt_info, m_c, tol_c, maxiter, xi_0, csw, csw, multigrid)

pt_src = source.source12(latt_info, "point", source_position)
with dirac_l.useGauge(gauge_stout):
    propagator_l = core.invertPropagator(dirac_l, pt_src)
with dirac_s.useGauge(gauge_stout):
    propagator_s = core.invertPropagator(dirac_s, pt_src)
with dirac_c.useGauge(gauge_stout):
    propagator_c = core.invertPropagator(dirac_c, pt_src)

for task in tasks:
    mode = task["compute"]
    c2_task = cp.zeros((len(shift_directions), z_max + 1, latt_info.Lt)) + 0j

    if mode == "pion":
        propagator_conj = propagator_l
        propagator_shift_seed = propagator_l
    elif mode == "kaon":
        propagator_conj = propagator_s
        propagator_shift_seed = propagator_l
    elif mode == "eta_s":
        propagator_conj = propagator_s
        propagator_shift_seed = propagator_s
    elif mode == "eta_c":
        propagator_conj = propagator_c
        propagator_shift_seed = propagator_c
    elif mode == "rho":
        propagator_conj = propagator_l
        propagator_shift_seed = propagator_l
    elif mode == "D":
        propagator_conj = propagator_l
        propagator_shift_seed = propagator_c
    elif mode == "D_s":
        propagator_conj = propagator_s
        propagator_shift_seed = propagator_c
    elif mode == "Jpsi":
        propagator_conj = propagator_c
        propagator_shift_seed = propagator_c
    elif mode == "D_star":
        propagator_conj = propagator_l
        propagator_shift_seed = propagator_c
    elif mode == "D_s_star":
        propagator_conj = propagator_s
        propagator_shift_seed = propagator_c
    else:
        raise ValueError(f"Unsupported task mode: {mode}")

    for direction_index, shift_mu in enumerate(shift_directions):
        propagator_shift = propagator_shift_seed.copy()
        gauge_shift.ensurePureGauge()
        gauge_shift.pure_gauge.loadGauge(gauge_shift)

        for z in range(z_max + 1):
            if z > 0:
                for spin in range(4):
                    for color in range(3):
                        fermion = propagator_shift.getFermion(spin, color)
                        fermion_shift = gauge_shift.pure_gauge.covDev(fermion, shift_mu)
                        propagator_shift.setFermion(fermion_shift, spin, color)

            if mode in ["rho", "Jpsi", "D_star", "D_s_star"]:
                c2_local = 0
                for gamma_i in vector_gamma_list:
                    c2_local += contract(
                        "wtzyx,wtzyxjiba,jk,wtzyxklba,li->t",
                        mom_phase.data,
                        propagator_conj.data.conj(),
                        G5 @ gamma_i,
                        propagator_shift.data,
                        gamma_i.conj().T @ G5,
                    )
                c2_task[direction_index, z] = c2_local / len(vector_gamma_list)
            else:
                c2_task[direction_index, z] = contract(
                    "wtzyx,wtzyxjiba,jk,wtzyxklba,li->t",
                    mom_phase.data,
                    propagator_conj.data.conj(),
                    G5 @ G5,
                    propagator_shift.data,
                    G5.conj().T @ G5,
                )

        del propagator_shift

    # Gather each local-time vector separately so MPI combines all t-ranks.
    # if core.getMPIRank() == 0:
        # c2_t_root = np.zeros((len(shift_directions), z_max + 1, latt_size[3]), dtype=np.complex128)

    # for direction_index in range(len(shift_directions)):
    #     for z in range(z_max + 1):
    #         c_t = core.gatherLattice(c2_task[direction_index, z].get(), [0, -1, -1, -1])
    #         if core.getMPIRank() == 0:
    #             c2_t_root[direction_index, z] = np.asarray(c_t, dtype=np.complex128).reshape(-1)
    c2_t_root = core.gatherLattice(c2_task.get(), [2,-1,-1,-1])



    if core.getMPIRank() == 0:
        rows = []
        for z in range(z_max + 1):
            t = np.arange(c2_t_root.shape[2], dtype=np.int32)
            rows.append(
                np.column_stack(
                    (
                        np.full(c2_t_root.shape[2], z, dtype=np.int32),
                        t,
                        c2_t_root[0, z].real,
                        c2_t_root[0, z].imag,
                    )
                )
            )

        out = np.vstack(rows)
        out_path = os.path.join(out_dir, f"task_{task['id']:02d}_{task['label']}_cfg{cfg_num}.txt")
        np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])

    del c2_task

del pt_src
del propagator_l
del propagator_s
del propagator_c
gc.collect()
cp.get_default_memory_pool().free_all_blocks()