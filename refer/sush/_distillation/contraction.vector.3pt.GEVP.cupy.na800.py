import sys

# Phase-aware equivalence requires the current lqcddb implementation.
sys.path.insert(0, '/public/home/sush/distillation/lqcddb_update/src')
from lqcddb import *

import time
import numpy as np

# M0 Runtime: retain the reference's top-level CuPy/MPI program.
st_total = time.perf_counter()
set_backend('cupy')
backend = get_backend()

# M1 Lattice and MPI
lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 2]

conf_id = sys.argv[1]

mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()

Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[x]//grid_size[x] for x in range(len(lattice_size))]

fun_eigen = vertex_creator(Nx = Nx)

# M2 Momentum and phases: lists use [pz, py, px], as in the reference.
Mom_sink_VDV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))
Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))[::-1]

Mom_len = len(Mom_sink_VDV)
source_mom_chunk = 9
# Conservative thresholds for all 27 source momenta: static scratch including
# copies is 61.52/41.01/20.51 GiB for N=3/2/1; source chunking reduces this.
# Allow 7.2 GiB for resident arrays and at least 8 GiB for CUDA/workspace.
gpu_free_bytes, _ = backend.cuda.runtime.memGetInfo()
# sink_mom_chunk = 3 if gpu_free_bytes >= 77*1024**3 else (2 if gpu_free_bytes >= 57*1024**3 else 1)
# sink_mom_chunk = min(comm.allgather(sink_mom_chunk))
sink_mom_chunk = 7

phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, Nc), dtype = complex)
phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype = complex)

for Mom_indx in range(Mom_len):
    # These are the current lqcddb names of the reference's phase routines.
    phase_exp_2pt[Mom_indx] = fun_eigen.phase_exp_VdV(Mom = Mom_sink_VDV[Mom_indx])
    phase_exp_3pt[Mom_indx] = fun_eigen.phase_exp_VVV(Mom = Mom_sink_VVV[Mom_indx])

Nev_src = 100
Nev_max = 400
tsep = int(sys.argv[2]) if len(sys.argv) > 2 else 18
if not 8 < tsep < Nt:
    raise ValueError(f'tsep must satisfy 8 < tsep < Nt={Nt}: the retained insertion window is 5 <= tJ-tsrc <= tsep-4.')
source_times = [int(sys.argv[3])] if len(sys.argv) > 3 else list(range(Nt))
if any(t < 0 or t >= Nt for t in source_times):
    raise ValueError(f'source time must satisfy 0 <= tsrc < Nt={Nt}.')
current_start = int(sys.argv[4]) if len(sys.argv) > 4 else 5
current_stop = int(sys.argv[5]) if len(sys.argv) > 5 else tsep - 3
if not 5 <= current_start < current_stop <= tsep - 3:
    raise ValueError('current window must be a nonempty subset of [5, tsep-3).')
peram_sources = sorted(set(source_times) | {(t + tsep)%Nt for t in source_times})
peram_source_index = {t: i for i, t in enumerate(peram_sources)}
vertex_times = set(peram_sources) | {(t + dt)%Nt for t in source_times for dt in range(current_start, current_stop)}

trank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'TScatter')

# M3 Dynamic manifest. O_pi^(1,0) = -pi0_E in Eqs. (S8),(S11).
# Common source charge basis: A = p*pi0_E, B = n*pi+_E, each with Iz=+1/2.
# O_(1/2,+1/2) = A/sqrt(3) + sqrt(2/3)*B;
# O_(3/2,+1/2) = -sqrt(2/3)*A + B/sqrt(3).
# The final N and both N-side admixtures have fixed I=1/2, Iz=+1/2.
proton = ['|', 'u', 'u', 'gamma_7', 'd', '|']
neutron = ['|', 'd', 'u', 'gamma_7', 'd', '|']
pion_u = ['|', 'u^d', 'gamma_5', 'u', '|']
pion_d = ['|', 'd^d', 'gamma_5', 'd', '|']
pion_plus = ['|', 'd^d', 'gamma_5', 'u', '|']

sink_operator = [
    [proton],
    [[np.sqrt(1/6), *proton, *pion_u], [-np.sqrt(1/6), *proton, *pion_d], [np.sqrt(2/3), *neutron, *pion_plus]],
    [[np.sqrt(1/6), *proton, *pion_u], [-np.sqrt(1/6), *proton, *pion_d], [np.sqrt(2/3), *neutron, *pion_plus]],
    ]

initial_operator = [
    [[np.sqrt(1/2), *proton, *pion_u], [-np.sqrt(1/2), *proton, *pion_d]],
    [[*neutron, *pion_plus]],
    ]
source_operator = [[conjugate_operator(t) for t in group] for group in initial_operator]
current_operator = [
    ['|', 'u^d', 'gamma_1', 'u', '|'],
    ['|', 'd^d', 'gamma_1', 'd', '|'],
    ]

Cpt = '3pt'
use_equivalence = True
ignore_dis = True
Projection = True
optimize = ['greedy', 'dp', 'auto']
dynamic_plan = []
for i in range(len(sink_operator)):
    for k in range(len(source_operator)):
        for f in range(len(current_operator)):
            operator_groups = [[s, t, current_operator[f]] for s in sink_operator[i] for t in source_operator[k]]
            # Wick vertex order is sink, CURRENT, source (not sink, source, current).
            Vindex = ['N', 'N', 'M', 'M'] if i == 0 else ['N', 'N', 'N', 'M', 'M']
            Gindex = ['', 'G', '', ''] if i == 0 else ['', '', 'G', '', '']
            dynamic_plan += run_wick_analysis(
                operator_groups, Cpt = Cpt, Vindex = Vindex, Gindex = Gindex,
                # Discover all keys; M7 uses the concrete gamma registry for merging.
                use_equivalence = False, ignore_dis = ignore_dis, verbose = False)
required_peram = sorted({(p[2], tuple(p[4])) for entry in dynamic_plan for p in entry[2]['peram'][entry[3]]})
# There is at most one VVV and one VDV per time region; registry numbering
# restarts independently for each kind and region, unlike raw Wick V names.
required_v = sorted({(v[1].split('_')[0] + '_0', v[3]) for entry in dynamic_plan for v in entry[2]['V']})
required_gamma = sorted({g[1] for entry in dynamic_plan for g in entry[2]['gamma_pos']})
# required peram: light, eight pairs of tsrc/tsink/tcur0 excluding current self;
# required V: VVV_0 at tsrc/tsink, VDV_0 at tsrc/tsink/tcur0;
# required gamma: gamma_7, gamma_5, gamma_1, Projector.
# peram axes (spin_q,spin_aq,eigen_q,eigen_aq): endpoints 100x100,
# current->endpoint 400x100, endpoint->current 100x400; no current self loop.
# VDV axes (momentum,eigen_aq,eigen_q); VVV axes (momentum,eigen1,eigen2,eigen3).
# Output axes: (N_basis,source_charge,current_flavor,tsrc,tJ,mu,sN,sNpi,p,r).
# N_basis = [N(p), sigma.p_hat*(N(p)pi(0))_I=1/2,
#                       sigma.p_hat*(N(0)pi(p))_I=1/2], Eq. (19).
# Npi source has total momentum zero: N(-r)pi(r). Average r over shells
# r^2=0,1,2,3 for Eq. (17); retain directions here for later analysis.
# Current self loops are omitted; remaining diagrams have no factorization or
# c_m*d_n truncation. GEVP eigenvectors come from separate two-point matrices.
# With a shared light peram and these CG phases, exact SU(2) gives
# C_Au=(C_Bu+C_Bd)/sqrt(2)-C_Ad for each N basis and all external indices.

# M4 Vertex data: keep the reference's smearing/blending weights.
VdV_link = backend.zeros((Lt, Mom_len, Nev_max, Nev_max), dtype = complex)
sink_VVV = {}  # Cache endpoint VVV on the host; copy only the active endpoints.

st_eigen = time.perf_counter()
eigvecs = None
for tsrc_indx, tsrc in enumerate(trank):
    if tsrc not in vertex_times:
        continue
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')

    if tsrc in peram_source_index:
        sink_VVV[tsrc] = np.zeros((Mom_len, Nev_src, Nev_src, Nev_src), dtype = complex)
    for Mom_indx in range(Mom_len):
        VdV_link[tsrc_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs)
        if tsrc in peram_source_index:
            sink_VVV[tsrc][Mom_indx] = fun_eigen.Mom_VVV_sink_t(phase_exp = phase_exp_3pt[Mom_indx], eigvecs = eigvecs[:Nev_src]).get()

VdV_link *= fun_eigen.create_omega_accelerate(
    exact = 100,
    N_eigen = [100, 200, 400, 700, 1100],
    N_sum = [20, 20, 20, 20, 20],
    N_extract = [10, 10, 10, 10, 10],
    noise = 200,
)

del phase_exp_2pt, phase_exp_3pt, eigvecs
backend.cuda.get_current_stream().synchronize()
comm.Barrier()
if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')

# M5 Peram data: host time cache avoids ~19.6 GiB/rank of resident GPU data.
corr_NJN_GEVP = backend.zeros((len(sink_operator), len(source_operator), len(current_operator), len(source_times), Lt, 4, Ns//2, Ns//2, sink_mom_chunk, Mom_len), dtype = complex)
# Eq. (19) defines the admixtures only at nonzero total N momentum.
corr_NJN_GEVP[1:, ..., 0, :] = complex(np.nan, np.nan)

st_load = time.perf_counter()
if rank == 0:
    peram_u = np.zeros((len(peram_sources), Nt, Ns, Ns, Nev_max, Nev_src), dtype = complex)
    for tsrc in peram_sources:
        peram_all = np.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
            )
        peram_u[peram_source_index[tsrc]] = peram_all[..., :Nev_max, :Nev_src]
    del peram_all
else:
    peram_u = None

peram_u = get_mpi_data(data = peram_u, mdtype = 'TScatter', root = 0, axis = 1)
backend.get_default_memory_pool().free_all_blocks()
backend.cuda.get_current_stream().synchronize()
comm.Barrier()
if rank == 0:
    print(f'load peram use time {time.perf_counter() - st_load :.3f} s')

# M6 Registries: retain the reference's projected-spin normalization.
GR = GammaRegistry()
VR = VRegistry()
PR = PeramRegistry()

GR.register('gamma_7', gamma(7))
GR.register('gamma_5', gamma(5))
GR.register('gamma_1', backend.stack([gamma(j + 1) for j in range(4)]))
GR.register('Projector', [((gamma(4) + gamma(0))/2.0)[:, :2], ((gamma(4) + gamma(0))/2.0)[:, :2]])

# M7 Dynamic loop
if rank == 0:
    print(f'contraction settings: ranks={size} sources={len(source_times)} ignore_dis={ignore_dis} use_equivalence={use_equivalence} mu_batch=4 sink_mom_chunk={sink_mom_chunk} source_mom_chunk={source_mom_chunk}', flush = True)
for source_indx, tsrc in enumerate(source_times):
    comm.Barrier()
    st_cal = time.perf_counter()
    tsink = (tsrc + tsep)%Nt
    src_rank, src_indx = get_mpi_tlist(Nt = Nt, t = tsrc, gtype = 'find')
    sink_rank, sink_indx = get_mpi_tlist(Nt = Nt, t = tsink, gtype = 'find')
    src_peram_indx = peram_source_index[tsrc]
    sink_peram_indx = peram_source_index[tsink]

    VVV_src = backend.asarray(get_mpi_data(data = sink_VVV[tsrc] if rank == src_rank else None, mdtype = 'Bcast', root = src_rank))
    VDV_src = backend.asarray(get_mpi_data(data = VdV_link[src_indx, :, :Nev_src, :Nev_src], mdtype = 'Bcast', root = src_rank))
    VVV_src = VVV_src.conj()
    VDV_src = VDV_src.conj().transpose(0, 2, 1)

    VVV_sink = backend.asarray(get_mpi_data(data = sink_VVV[tsink] if rank == sink_rank else None, mdtype = 'Bcast', root = sink_rank))
    VDV_sink = backend.asarray(get_mpi_data(data = VdV_link[sink_indx, :, :Nev_src, :Nev_src], mdtype = 'Bcast', root = sink_rank))
    VDV_sink = VDV_sink.conj().transpose(0, 2, 1)

    _, _, tsink_indx = get_mpi_tlist(Nt = Nt, t = range(tsrc + current_start, tsrc + current_stop), gtype = 'TScatter')

    peram_u_src_src = get_mpi_data(data = peram_u[src_peram_indx, src_indx, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = src_rank)
    peram_u_sink_src = get_mpi_data(data = peram_u[src_peram_indx, sink_indx, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = sink_rank)
    peram_u_sink_sink = get_mpi_data(data = peram_u[sink_peram_indx, sink_indx, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = sink_rank)

    PR.register('light', ('tsrc', 'tsrc'), backend.asarray(peram_u_src_src))
    PR.register('light', ('tsink', 'tsrc'), backend.asarray(peram_u_sink_src))
    PR.register('light', ('tsrc', 'tsink'), seq_peram(backend.asarray(peram_u_sink_src)))
    PR.register('light', ('tsink', 'tsink'), backend.asarray(peram_u_sink_sink))

    for tcur0 in tsink_indx:
        peram_u_tcur0_tsrc = backend.asarray(peram_u[src_peram_indx, tcur0])
        peram_u_tcur0_tsink = backend.asarray(peram_u[sink_peram_indx, tcur0])

        PR.register('light', ('tcur0', 'tsrc'), peram_u_tcur0_tsrc)
        PR.register('light', ('tsrc', 'tcur0'), seq_peram(peram_u_tcur0_tsrc))
        PR.register('light', ('tcur0', 'tsink'), peram_u_tcur0_tsink)
        PR.register('light', ('tsink', 'tcur0'), seq_peram(peram_u_tcur0_tsink))

        for i in range(len(sink_operator)):
            # A common N label pairs equal sink/current momenta; M labels
            # pair source N(-r) and pi(r). Neither label is summed or crossed.
            # Start at 1 for the two admixtures, where p_hat is undefined at 0.
            for Mom_indx in range(1): #range(0 if i == 0 else 1, sink_mom_chunk, sink_mom_chunk):
                Mom_stop = min(Mom_indx + sink_mom_chunk, Mom_len)
                VR.register('VDV_0', 'tcur0', VdV_link[tcur0, Mom_indx:Mom_stop])
                if i == 1:
                    VR.register('VDV_0', 'tsink', VDV_sink[0:1])
                    VR.register('VVV_0', 'tsink', VVV_sink[Mom_indx:Mom_stop])
                elif i == 2:
                    VR.register('VDV_0', 'tsink', VDV_sink[Mom_indx:Mom_stop])
                    VR.register('VVV_0', 'tsink', VVV_sink[0:1])
                else:
                    VR.register('VVV_0', 'tsink', VVV_sink[Mom_indx:Mom_stop])
                if i > 0:
                    sigma_p = backend.stack([Mom_times_sigma(Mom = p) / np.sqrt(sum(x*x for x in p)) for p in Mom_sink_VVV[Mom_indx:Mom_stop]])

                for source_mom_start in range(0, Mom_len, source_mom_chunk):
                    source_mom_stop = min(source_mom_start + source_mom_chunk, Mom_len)
                    # The common M label pairs N(-r) and pi(r) within this block.
                    VR.register('VDV_0', 'tsrc', VDV_src[source_mom_start:source_mom_stop])
                    VR.register('VVV_0', 'tsrc', VVV_src[source_mom_start:source_mom_stop])
                    for k in range(len(source_operator)):
                        for f in range(len(current_operator)):
                            if k == 0 and f == 0:
                                continue  # Recover A,u from the three independent channels.
                            operator_groups = [[s, t, current_operator[f]] for s in sink_operator[i] for t in source_operator[k]]
                            Vindex = ['N', 'N', 'M', 'M'] if i == 0 else ['N', 'N', 'N', 'M', 'M']
                            Gindex = ['', 'G', '', ''] if i == 0 else ['', '', 'G', '', '']
                            dycn_NN = dynamic_contraction(
                                operator_groups,
                                peram_registry = PR,
                                v_registry = VR,
                                gamma_registry = GR,
                                Cpt = Cpt,
                                Vindex = Vindex,
                                Gindex = Gindex,
                                use_equivalence = use_equivalence,
                                ignore_dis = ignore_dis,
                                Projection = Projection,
                                optimize = optimize,
                                verbose = False,
                                )

                            corr = dycn_NN.calculate_all()  # (sN,sNpi,mu,p,r_block)
                            if i > 0:
                                # Current has phase exp(+i*p.x); sink has exp(-i*p.x).
                                corr = cached_contract('Nab,bcGNM->acGNM', sigma_p, corr)
                            corr_NJN_GEVP[i, k, f, source_indx, tcur0, ..., Mom_indx:Mom_stop, source_mom_start:source_mom_stop] = corr.transpose(2, 0, 1, 3, 4)

                    corr_NJN_GEVP[i, 0, 0, source_indx, tcur0, ..., Mom_indx:Mom_stop, source_mom_start:source_mom_stop] = (
                        (corr_NJN_GEVP[i, 1, 0, source_indx, tcur0, ..., Mom_indx:Mom_stop, source_mom_start:source_mom_stop]
                         + corr_NJN_GEVP[i, 1, 1, source_indx, tcur0, ..., Mom_indx:Mom_stop, source_mom_start:source_mom_stop]) / np.sqrt(2)
                        - corr_NJN_GEVP[i, 0, 1, source_indx, tcur0, ..., Mom_indx:Mom_stop, source_mom_start:source_mom_stop])

    backend.cuda.get_current_stream().synchronize()
    comm.Barrier()
    if rank == 0:
        print(f'calculate 3pt of tsrc {tsrc} use time {(time.perf_counter() - st_cal):.3f} s')

# M8 Gather and save. All operators carry odd fermion number (3 or 5).
# One temporal wrap gives -1; meson pairs are periodic. Apply this once here.
backend.get_default_memory_pool().free_all_blocks()
corr_NJN_GEVP = get_mpi_data(corr_NJN_GEVP, mdtype = 'TGather', root = 0, axis = 4)

if rank == 0:
    corr_save_path = f'/public/home/sush/distillation/0v2b/result_nex/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'

    import pathlib
    path = pathlib.Path(corr_save_path)

    if path.exists():
        print('save_path:',corr_save_path)
    else:
        path.mkdir(parents = True, exist_ok = True)
        print('mkdir_save_path:',corr_save_path)

    # Source times are SUMMED; divide by source_time_count for a mean.
    if len(source_times) == Nt:
        corr_sum = loop_tsrc(corr_NJN_GEVP, indx = [3, 4], Boundary_Conditions = 'Antiperiodic', Ctype = '3pt', t_sep = tsep)
        source_tag = '_noloop_equiv'
    else:
        corr_sum = backend.zeros_like(corr_NJN_GEVP[:, :, :, :1])
        for source_indx, tsrc in enumerate(source_times):
            phase = -1 if tsrc + tsep >= Nt else 1
            corr_sum += phase * backend.roll(corr_NJN_GEVP[:, :, :, source_indx:(source_indx + 1)], -tsrc, axis = 4)
        source_tag = f'_tsrc{source_times[0]}_noloop_equiv'
    if current_start != 5 or current_stop != tsep - 3:
        source_tag += f'_tau{current_start}-{current_stop - 1}'
    backend.save(f'{corr_save_path}/corr_3pt_vector_GEVP_tsep{tsep}{source_tag}.npy', corr_sum)
    # Store only common correlators. These coefficients specify later combinations:
    # V^(1,0)=(C_d-C_u)/sqrt(2); V^(0,0)=(C_u+C_d)/sqrt(2), Eq. (S9).
    # In this reversed orientation, C_opt = sum_a,m d_a * conj(c_m) * C_a,m.
    # For the paper's linear truncation retain a=0 OR m=0, after shell averaging.
    
    np.savez(f'{corr_save_path}/corr_3pt_vector_GEVP_tsep{tsep}{source_tag}_basis.npz',
            axes = ['N_basis', 'source_charge', 'current_flavor', 'tsrc_sum', 'tJ_minus_tsrc', 'mu', 'spin_N', 'spin_Npi', 'p', 'r'],
            N_basis = ['N(p)', 'sigma_p*N(p)pi(0)', 'sigma_p*N(0)pi(p)'],
            source_charge = ['p*pi0_E', 'n*pi+_E'], current_flavor = ['u', 'd'],
            mu = [1, 2, 3, 4], p = Mom_sink_VVV, r = Mom_sink_VDV,
            source_shell_q2 = np.sum(np.asarray(Mom_sink_VDV)**2, axis = 1),
            source_isospin = [0.5, 1.5], source_CG = [[np.sqrt(1/3), np.sqrt(2/3)], [-np.sqrt(2/3), np.sqrt(1/3)]],
            current_isospin = [1, 0], current_coefficients = [[-np.sqrt(1/2), np.sqrt(1/2)], [np.sqrt(1/2), np.sqrt(1/2)]],
            N_admixture_valid = np.any(np.asarray(Mom_sink_VVV) != 0, axis = 1),
            insertion_time_offsets = np.arange(current_start, current_stop),
            current_window = [current_start, current_stop],
            momentum_component_order = ['pz', 'py', 'px'], current_convention = 'Euclidean; mu=4 is time',
            source_time_count = len(source_times), source_times = source_times, tsep = tsep,
            current_renormalized = False, source_time_reduction = 'sum',
            full_five_point = False, five_point_factorized = False, gevp_cross_terms = True,
            ignore_current_loops = ignore_dis, use_equivalence = use_equivalence,
            batched_lorentz = True, batched_N_admixtures = False, shared_current_insertion = False,
            sink_mom_chunk = sink_mom_chunk, source_mom_chunk = source_mom_chunk,
            perambulator_storage = 'host', endpoint_VVV_storage = 'host',
            independent_channels = [[0, 1], [1, 0], [1, 1]], reconstructed_channel = [0, 0],
            isospin_reconstruction = 'C_Au=(C_Bu+C_Bd)/sqrt(2)-C_Ad'
            )
    
    print(f'total elapsed time {time.perf_counter() - st_total:.3f} s')
