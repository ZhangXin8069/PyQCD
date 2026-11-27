import sys
import os
from pathlib import Path

sys.path.insert(0, '/public/home/sush/distillation/0v2b/3pt_acc')
from threept import Workspace, contract, workspace_bytes

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

if not 2 <= len(sys.argv) <= 6:
    raise SystemExit('Usage: e32p29.py CONF [TSEP [TSRC [TAU_START [TAU_STOP]]]]')
conf_id = sys.argv[1]

# Slurm can expose both allocated cards to both ranks, or a single card per rank.
device = 0 if backend.cuda.runtime.getDeviceCount() == 1 else -1
mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__, device = device)

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()
gpu = backend.cuda.Device()
bindings = comm.allgather((os.uname().nodename, gpu.id, gpu.pci_bus_id))
if len({host for host, _, _ in bindings}) != 1 or len({pci for _, _, pci in bindings}) != size:
    raise RuntimeError(f'E32P29 needs two different GPUs on one node. MPI bindings: {bindings}')
if rank == 0:
    for owner_rank, (host, ordinal, pci) in enumerate(bindings):
        print(f'conf={conf_id} rank={owner_rank} host={host} CUDA device={ordinal} PCI={pci}', flush=True)

Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[x]//grid_size[x] for x in range(len(lattice_size))]

fun_eigen = vertex_creator(Nx = Nx)

# M2 Momentum and phases: lists use [pz, py, px], as in the reference.
Mom_sink_VDV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))
Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))[::-1]

Mom_len = len(Mom_sink_VDV)
source_mom_chunk = 27
contract_dtype = np.dtype(os.environ.get('THREEPT_DTYPE', 'complex128'))
if contract_dtype not in (np.complex64, np.complex128):
    raise ValueError('THREEPT_DTYPE must be complex64 or complex128')
sink_mom_chunk = 1
sink_mom_len = 7

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

ensemble = 'beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64'
file_tag = 'e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
data_root = Path('/nexdata/project/lqcd/sush')
eigen_dir = data_root / 'eigensystem' / ensemble / conf_id
peram_dir = data_root / 'perambulators' / ensemble / 'light' / conf_id
eigen_files = {t: eigen_dir / f'{conf_id}_t{t:03d}_{file_tag}' for t in sorted(vertex_times)}
peram_files = {t: peram_dir / f't{t:03d}_{file_tag}' for t in peram_sources}
missing = None
if rank == 0:
    missing = next((p for p in (*eigen_files.values(), *peram_files.values()) if not p.is_file()), None)
missing = comm.bcast(missing, root=0)
if missing is not None:
    raise FileNotFoundError(f'E32P29 conf={conf_id}, missing input: {missing}')

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
# The signed Wick expansion and endpoint projections are compiled into libthreept.so.
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

if rank == 0:
    print(f'construct vertices for conf={conf_id}', flush=True)
st_eigen = time.perf_counter()
eigvecs = None
for tsrc_indx, tsrc in enumerate(trank):
    if tsrc not in vertex_times:
        continue
    eigvecs = backend.load(eigen_files[tsrc])

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
corr_NJN_GEVP = backend.zeros((len(sink_operator), len(source_operator), len(current_operator), len(source_times), Lt, 4, Ns//2, Ns//2, sink_mom_len, Mom_len), dtype = complex)
# Eq. (19) defines the admixtures only at nonzero total N momentum.
corr_NJN_GEVP[1:, ..., 0, :] = complex(np.nan, np.nan)

if rank == 0:
    print(f'load perambulators for conf={conf_id}', flush=True)
# st_load = time.perf_counter()
# # Scatter the 64 sink times of each source file directly into the rank-local cache.
# peram_u = np.empty((len(peram_sources), Lt, Ns, Ns, Nev_max, Nev_src), dtype=complex)
# for source_index, tsrc in enumerate(peram_sources):
#     slab = None
#     if rank == 0:
#         # Read one NPY file sequentially before MPI packs even and odd times.
#         slab = np.load(peram_files[tsrc], allow_pickle=False)[..., :Nev_max, :Nev_src]
#         if slab.shape != (Nt, Ns, Ns, Nev_max, Nev_src):
#             raise ValueError(f'Unexpected perambulator slice {slab.shape} in {peram_files[tsrc]}')
#         slab = np.asarray(slab, dtype=peram_u.dtype)
#     get_mpi_data(data=slab, mdtype='TScatter', root=0, axis=0, recv_buff=peram_u[source_index])
#     del slab
#     if rank == 0 and (source_index + 1) % 8 == 0:
#         print(f'conf={conf_id} distributed peram sources {source_index+1}/{len(peram_sources)}', flush=True)
# backend.get_default_memory_pool().free_all_blocks()
# backend.cuda.get_current_stream().synchronize()
# comm.Barrier()
# if rank == 0:
#     print(f'load peram use time {time.perf_counter() - st_load :.3f} s')
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
# M6 The native schedule fixes the four Euclidean gammas, gamma_5, gamma_7
# and the two projector columns. Inputs below have the registry orientations.

# M7 Specialized CUDA contraction
scratch = Workspace(max(workspace_bytes(i, Nev_src, Nev_max, source_mom_chunk, contract_dtype) for i in (0, 1)))
st_main = time.perf_counter()
if rank == 0:
    print(f'contraction settings: ranks={size} sources={len(source_times)} ignore_dis={ignore_dis} use_equivalence={use_equivalence} mu_batch=4 sink_mom_chunk={sink_mom_chunk} source_mom_chunk={source_mom_chunk}', flush = True)
for source_indx, tsrc in enumerate(source_times):
    comm.Barrier()
    if rank == 0:
        print(f'begin tsrc={tsrc} ({source_indx+1}/{len(source_times)})', flush=True)
    st_cal = time.perf_counter()
    tsink = (tsrc + tsep)%Nt
    src_rank, src_indx = get_mpi_tlist(Nt = Nt, t = tsrc, gtype = 'find')
    sink_rank, sink_indx = get_mpi_tlist(Nt = Nt, t = tsink, gtype = 'find')
    src_peram_indx = peram_source_index[tsrc]
    sink_peram_indx = peram_source_index[tsink]

    VVV_src = backend.asarray(get_mpi_data(data = sink_VVV[tsrc] if rank == src_rank else None, mdtype = 'Bcast', root = src_rank))
    VDV_src = backend.asarray(get_mpi_data(data = VdV_link[src_indx, :, :Nev_src, :Nev_src], mdtype = 'Bcast', root = src_rank))
    VVV_src = backend.ascontiguousarray(VVV_src.conj(), dtype=contract_dtype)
    VDV_src = backend.ascontiguousarray(VDV_src.conj().transpose(0, 2, 1), dtype=contract_dtype)

    VVV_sink = backend.asarray(get_mpi_data(data = sink_VVV[tsink] if rank == sink_rank else None, mdtype = 'Bcast', root = sink_rank))
    VDV_sink = backend.asarray(get_mpi_data(data = VdV_link[sink_indx, :, :Nev_src, :Nev_src], mdtype = 'Bcast', root = sink_rank))
    VVV_sink = backend.ascontiguousarray(VVV_sink, dtype=contract_dtype)
    VDV_sink = backend.ascontiguousarray(VDV_sink.conj().transpose(0, 2, 1), dtype=contract_dtype)

    _, _, tsink_indx = get_mpi_tlist(Nt = Nt, t = range(tsrc + current_start, tsrc + current_stop), gtype = 'TScatter')

    peram_u_src_src = get_mpi_data(data = peram_u[src_peram_indx, src_indx, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = src_rank)
    peram_u_sink_src = get_mpi_data(data = peram_u[src_peram_indx, sink_indx, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = sink_rank)
    peram_u_sink_sink = get_mpi_data(data = peram_u[sink_peram_indx, sink_indx, ..., :Nev_src, :Nev_src], mdtype = 'Bcast', root = sink_rank)

    ds = backend.asarray(peram_u_sink_src, dtype=contract_dtype, order='C')
    endpoints = (backend.asarray(peram_u_src_src, dtype=contract_dtype, order='C'),
                 ds, backend.ascontiguousarray(seq_peram(ds), dtype=contract_dtype),
                 backend.asarray(peram_u_sink_sink, dtype=contract_dtype, order='C'))

    for tcur0 in tsink_indx:
        st_current = time.perf_counter()
        js = backend.asarray(peram_u[src_peram_indx, tcur0], dtype=contract_dtype, order='C')
        jd = backend.asarray(peram_u[sink_peram_indx, tcur0], dtype=contract_dtype, order='C')
        currents = (js, backend.ascontiguousarray(seq_peram(js), dtype=contract_dtype),
                    jd, backend.ascontiguousarray(seq_peram(jd), dtype=contract_dtype))
        for i in range(len(sink_operator)):
            for Mom_indx in range(1, sink_mom_len):
                baryon = VVV_sink[0 if i == 2 else Mom_indx]
                pion = VDV_sink[0 if i == 1 else Mom_indx] if i else None
                vertex = backend.ascontiguousarray(VdV_link[tcur0, Mom_indx], dtype=contract_dtype)
                for source_mom_start in range(0, Mom_len, source_mom_chunk):
                    block = slice(source_mom_start, min(source_mom_start + source_mom_chunk, Mom_len))
                    corr = contract(*endpoints, *currents, baryon, pion, VVV_src[block],
                                    VDV_src[block], vertex, basis=i, momentum=Mom_sink_VVV[Mom_indx], workspace=scratch)
                    corr_NJN_GEVP[i, :, :, source_indx, tcur0, ..., Mom_indx, block] = corr
        backend.cuda.get_current_stream().synchronize()
        print(f'rank={rank} local tsrc={tsrc} tJ={trank[tcur0]} contraction seconds={time.perf_counter()-st_current:.3f}', flush=True)

    backend.cuda.get_current_stream().synchronize()
    comm.Barrier()
    if rank == 0:
        print(f'calculate 3pt of tsrc {tsrc} use time {(time.perf_counter() - st_cal):.3f} s')
    
del scratch
if rank == 0:
    print(f'main loop seconds={time.perf_counter()-st_main:.3f}, gathering', flush=True)
st_gather = time.perf_counter()
# M8 Gather and save. All operators carry odd fermion number (3 or 5).
# One temporal wrap gives -1; meson pairs are periodic. Apply this once here.
backend.get_default_memory_pool().free_all_blocks()
corr_NJN_GEVP = get_mpi_data(corr_NJN_GEVP, mdtype = 'TGather', root = 0, axis = 4)

if rank == 0:
    print(f'gather seconds={time.perf_counter()-st_gather:.3f}, saving', flush=True)
    st_save = time.perf_counter()
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
    if contract_dtype == np.complex64:
        source_tag += '_complex64'
    backend.save(f'{corr_save_path}/corr_3pt_vector_GEVP_tsep{tsep}{source_tag}.npy', corr_sum)
    # Store only common correlators. These coefficients specify later combinations:
    # V^(1,0)=(C_d-C_u)/sqrt(2); V^(0,0)=(C_u+C_d)/sqrt(2), Eq. (S9).
    # In this reversed orientation, C_opt = sum_a,m d_a * conj(c_m) * C_a,m.
    # For the paper's linear truncation retain a=0 OR m=0, after shell averaging.
    
    np.savez(f'{corr_save_path}/corr_3pt_vector_GEVP_tsep{tsep}{source_tag}_basis.npz',
            axes = ['N_basis', 'source_charge', 'current_flavor', 'tsrc_sum', 'tJ_minus_tsrc', 'mu', 'spin_N', 'spin_Npi', 'p', 'r'],
            N_basis = ['N(p)', 'sigma_p*N(p)pi(0)', 'sigma_p*N(0)pi(p)'],
            source_charge = ['p*pi0_E', 'n*pi+_E'], current_flavor = ['u', 'd'],
            mu = [1, 2, 3, 4], p = Mom_sink_VVV[:sink_mom_len], r = Mom_sink_VDV,
            source_shell_q2 = np.sum(np.asarray(Mom_sink_VDV)**2, axis = 1),
            source_isospin = [0.5, 1.5], source_CG = [[np.sqrt(1/3), np.sqrt(2/3)], [-np.sqrt(2/3), np.sqrt(1/3)]],
            current_isospin = [1, 0], current_coefficients = [[-np.sqrt(1/2), np.sqrt(1/2)], [np.sqrt(1/2), np.sqrt(1/2)]],
            N_admixture_valid = np.any(np.asarray(Mom_sink_VVV[:sink_mom_len]) != 0, axis = 1),
            sink_momentum_computed = np.arange(sink_mom_len) > 0, contraction_dtype = contract_dtype.name,
            insertion_time_offsets = np.arange(current_start, current_stop),
            current_window = [current_start, current_stop],
            momentum_component_order = ['pz', 'py', 'px'], current_convention = 'Euclidean; mu=4 is time',
            source_time_count = len(source_times), source_times = source_times, tsep = tsep,
            current_renormalized = False, source_time_reduction = 'sum',
            full_five_point = False, five_point_factorized = False, gevp_cross_terms = True,
            ignore_current_loops = ignore_dis, use_equivalence = use_equivalence,
            batched_lorentz = True, batched_N_admixtures = False, shared_current_insertion = True,
            sink_mom_chunk = sink_mom_chunk, source_mom_chunk = source_mom_chunk,
            perambulator_storage = 'host', endpoint_VVV_storage = 'host',
            independent_channels = [[0, 1], [1, 0], [1, 1]], reconstructed_channel = [0, 0],
            isospin_reconstruction = 'C_Au=(C_Bu+C_Bd)/sqrt(2)-C_Ad'
            )
    
    print(f'saved {corr_save_path}/corr_3pt_vector_GEVP_tsep{tsep}{source_tag}.npy and basis metadata in {time.perf_counter()-st_save:.3f} s', flush=True)
    print(f'total elapsed time {time.perf_counter() - st_total:.3f} s')
