import os
import sys

# Set BLAS/OpenMP limits before NumPy or lqcddb imports.
# Slurm: 2 ranks x 18 threads = 36 CPU cores; shared-array contraction workers.
threads_per_rank = int(os.environ.get('SLURM_CPUS_PER_TASK', os.environ.get('OMP_NUM_THREADS', '12')))
if threads_per_rank < 1:
    raise ValueError('threads per MPI rank must be positive')
for thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[thread_variable] = str(threads_per_rank)
os.environ['OMP_DYNAMIC'] = 'FALSE'
os.environ['MKL_DYNAMIC'] = 'FALSE'

# Phase-aware equivalence requires the current lqcddb implementation.
sys.path.insert(0, '/public/home/sush/distillation/lqcddb_update/src')
from lqcddb import *

import time
import numpy as np
import resource
from concurrent.futures import ThreadPoolExecutor, as_completed
import mkl


def report_memory(stage, **labels):
    # Linux ru_maxrss is KiB and is a lifetime high-water mark per MPI rank.
    rss_kib = 0
    with open('/proc/self/status') as status:
        for line in status:
            if line.startswith('VmRSS:'):
                rss_kib = int(line.split()[1])
                break
    fields = ' '.join(f'{key}={value}' for key, value in labels.items())
    print(f'MEMORY stage={stage} rank={rank} {fields} rss_gib={rss_kib/2**20:.6f} '
          f'lifetime_peak_rss_gib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.6f}', flush=True)

# M0 Runtime: NumPy CPU backend; retain the original M0-M8 structure.
st_total = time.perf_counter()
set_backend('numpy')
backend = get_backend()

# M1 Lattice and MPI
lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 2]

conf_id = sys.argv[1]

mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()
if os.environ.get('CPU_BENCHMARK', '0') == '1':
    # Independent of MPI: one sample/second for simultaneous rank RSS totals.
    import threading
    from pathlib import Path
    memory_stop = threading.Event()
    memory_phase = 'setup'
    memory_tcurr = -1
    def sample_memory():
        sample_path = Path(os.environ['CPU_BENCHMARK_DIR']) / f'memory.rank{rank}.tsv'
        with sample_path.open('w', buffering=1) as samples:
            samples.write('unix_time\trank\tphase\ttcurr\trss_kib\n')
            while not memory_stop.is_set():
                with open('/proc/self/status') as status:
                    rss = next(int(line.split()[1]) for line in status if line.startswith('VmRSS:'))
                samples.write(f'{time.time():.6f}\t{rank}\t{memory_phase}\t{memory_tcurr}\t{rss}\n')
                memory_stop.wait(1.0)
    memory_thread = threading.Thread(target=sample_memory, daemon=True)
    memory_thread.start()
    print(f'BENCHMARK rank={rank} pid={os.getpid()} affinity={sorted(os.sched_getaffinity(0))} threads={threads_per_rank}', flush=True)

Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[x]//grid_size[x] for x in range(len(lattice_size))]

fun_eigen = vertex_creator(Nx = Nx)

# M2 Momentum and phases: lists use [pz, py, px], as in the reference.
Mom_sink_VDV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True)) + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))
Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [0, 1, 1], fix_Q2 = True))[::-1] + sorted(creat_mom_list(Mom = [1, 1, 1], fix_Q2 = True))[::-1]

Mom_len = len(Mom_sink_VDV)
source_mom_chunk = int(os.environ.get('CPU_SOURCE_MOM_CHUNK', '1'))
if not 1 <= source_mom_chunk <= Mom_len:
    raise ValueError('CPU_SOURCE_MOM_CHUNK must be in [1, Mom_len]')
# Retain the original seven N-side momenta, but contract one at a time.
sink_mom_count = 7
sink_mom_chunk = 1

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
# Endpoint VVV is expensive: split its momentum rows across MPI ranks, then
# assemble at the existing time owner. Keep the original cyclic time mapping.
for endpoint in peram_sources:
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{endpoint:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy', mmap_mode='r')
    endpoint_owner, _ = get_mpi_tlist(Nt=Nt, t=endpoint, gtype='find')
    if rank == endpoint_owner:
        sink_VVV[endpoint] = np.empty((Mom_len, Nev_src, Nev_src, Nev_src), dtype=complex)
    for Mom_indx in range(Mom_len):
        worker = Mom_indx % size
        if rank == worker:
            vertex = np.ascontiguousarray(fun_eigen.Mom_VVV_sink_t(
                phase_exp=phase_exp_3pt[Mom_indx], eigvecs=eigvecs[:Nev_src])[0])
            if worker == endpoint_owner:
                sink_VVV[endpoint][Mom_indx] = vertex
            else:
                comm.Send(vertex, dest=endpoint_owner, tag=101)
            del vertex
        elif rank == endpoint_owner:
            comm.Recv(sink_VVV[endpoint][Mom_indx], source=worker, tag=101)
    print(f'TIMING_VERTEX rank={rank} endpoint={endpoint} seconds={time.perf_counter()-st_eigen:.6f}', flush=True)
for tsrc_indx, tsrc in enumerate(trank):
    if tsrc not in vertex_times:
        continue
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')

    for Mom_indx in range(Mom_len):
        VdV_link[tsrc_indx, (Mom_indx):((Mom_indx + 1))] = fun_eigen.Mom_VdV_sink_t(phase_exp = phase_exp_2pt[(Mom_indx):((Mom_indx + 1))], eigvecs = eigvecs)

VdV_link *= fun_eigen.create_omega_accelerate(
    exact = 100,
    N_eigen = [100, 200, 400, 700, 1100],
    N_sum = [20, 20, 20, 20, 20],
    N_extract = [10, 10, 10, 10, 10],
    noise = 200,
)

del phase_exp_2pt, phase_exp_3pt, eigvecs
comm.Barrier()
if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')
report_memory('vertices')

# M5 Peram data: retain the original host cache and TScatter time mapping.
corr_NJN_GEVP = backend.zeros((len(sink_operator), len(source_operator), len(current_operator), len(source_times), Lt, 4, Ns//2, Ns//2, sink_mom_count, Mom_len), dtype = complex)
# Eq. (19) defines the admixtures only at nonzero total N momentum.
corr_NJN_GEVP[1:, ..., 0, :] = complex(np.nan, np.nan)

st_load = time.perf_counter()
if rank == 0:
    peram_u = np.zeros((len(peram_sources), Nt, Ns, Ns, Nev_max, Nev_src), dtype = complex)
    for tsrc in peram_sources:
        peram_all = np.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy'
            , mmap_mode='r')
        peram_u[peram_source_index[tsrc]] = peram_all[..., :Nev_max, :Nev_src]
    del peram_all
else:
    peram_u = None

peram_u = get_mpi_data(data = peram_u, mdtype = 'TScatter', root = 0, axis = 1)
comm.Barrier()
if rank == 0:
    print(f'load peram use time {time.perf_counter() - st_load :.3f} s')
report_memory('perambulators')

# M6 Registries: retain the reference's projected-spin normalization.
GR = GammaRegistry()
VR = VRegistry()
PR = PeramRegistry()

GR.register('gamma_7', gamma(7))
GR.register('gamma_5', gamma(5))
GR.register('gamma_1', backend.stack([gamma(j + 1) for j in range(4)]))
GR.register('Projector', [((gamma(4) + gamma(0))/2.0)[:, :2], ((gamma(4) + gamma(0))/2.0)[:, :2]])

# Each task owns its V registry; PR and GR are read-only until all tasks finish.
# Only free momentum axes are partitioned. Each diagram retains its original sum.
def contract_momentum_block(task):
    i, Mom_indx, source_mom_start, tcur0 = task
    task_VR = VRegistry()
    task_corr = backend.zeros((2, 2, 4, Ns//2, Ns//2,
                               min(sink_mom_chunk, sink_mom_count-Mom_indx),
                               min(source_mom_chunk, Mom_len-source_mom_start)), dtype=complex)
    Mom_stop = min(Mom_indx + sink_mom_chunk, sink_mom_count)
    task_VR.register('VDV_0', 'tcur0', VdV_link[tcur0, Mom_indx:Mom_stop])
    if i == 1:
        task_VR.register('VDV_0', 'tsink', VDV_sink[0:1])
        task_VR.register('VVV_0', 'tsink', VVV_sink[Mom_indx:Mom_stop])
    elif i == 2:
        task_VR.register('VDV_0', 'tsink', VDV_sink[Mom_indx:Mom_stop])
        task_VR.register('VVV_0', 'tsink', VVV_sink[0:1])
    else:
        task_VR.register('VVV_0', 'tsink', VVV_sink[Mom_indx:Mom_stop])
    if i > 0:
        sigma_p = backend.stack([Mom_times_sigma(Mom = p) / np.sqrt(sum(x*x for x in p)) for p in Mom_sink_VVV[Mom_indx:Mom_stop]])

    source_mom_stop = min(source_mom_start + source_mom_chunk, Mom_len)
    # The common M label pairs N(-r) and pi(r) within this block.
    task_VR.register('VDV_0', 'tsrc', VDV_src[source_mom_start:source_mom_stop])
    task_VR.register('VVV_0', 'tsrc', VVV_src[source_mom_start:source_mom_stop])
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
                v_registry = task_VR,
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
            task_corr[k, f] = corr.transpose(2, 0, 1, 3, 4)

    task_corr[0, 0] = (
        (task_corr[1, 0]
         + task_corr[1, 1]) / np.sqrt(2)
        - task_corr[0, 1])
    return task, task_corr


def initialize_contraction_thread():
    mkl.set_num_threads_local(1)


# M7 Dynamic loop
if rank == 0:
    print(f'contraction settings: ranks={size} sources={len(source_times)} ignore_dis={ignore_dis} use_equivalence={use_equivalence} backend=numpy threads_per_rank={threads_per_rank} mu_batch=4 sink_mom_count={sink_mom_count} sink_mom_chunk={sink_mom_chunk} source_mom_chunk={source_mom_chunk}', flush = True)
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
        st_tcurr = time.perf_counter()
        memory_phase = 'tcurr'
        memory_tcurr = int(trank[tcur0])
        print(f'TIMING_TCURR_START rank={rank} tsrc={tsrc} tcurr={trank[tcur0]} tau={(trank[tcur0]-tsrc)%Nt}', flush=True)
        peram_u_tcur0_tsrc = backend.asarray(peram_u[src_peram_indx, tcur0])
        peram_u_tcur0_tsink = backend.asarray(peram_u[sink_peram_indx, tcur0])

        PR.register('light', ('tcur0', 'tsrc'), peram_u_tcur0_tsrc)
        PR.register('light', ('tsrc', 'tcur0'), seq_peram(peram_u_tcur0_tsrc))
        PR.register('light', ('tcur0', 'tsink'), peram_u_tcur0_tsink)
        PR.register('light', ('tsink', 'tcur0'), seq_peram(peram_u_tcur0_tsink))

        # Reused arrays are shared by threads. Limit concurrent workspaces.
        requested_workers = int(os.environ.get('CPU_CONTRACTION_WORKERS', str(threads_per_rank)))
        if not 1 <= requested_workers <= threads_per_rank:
            raise ValueError('CPU_CONTRACTION_WORKERS must be in [1, threads_per_rank]')
        with open('/proc/self/status') as status:
            rss_gib = next(int(line.split()[1])/2**20 for line in status if line.startswith('VmRSS:'))
        node_budget_gib = float(os.environ.get('CPU_MEMORY_BUDGET_GIB', '144'))
        rank_budget_gib = node_budget_gib/size
        # Estimate rather than allocator-enforced cap; monitor measured RSS.
        workspace_gib = float(os.environ.get('CPU_WORKSPACE_GIB', '3')) * source_mom_chunk
        if workspace_gib <= 0 or rank_budget_gib <= rss_gib+8:
            raise MemoryError('Memory budget leaves insufficient contraction workspace')
        worker_count = min(requested_workers, max(1, int((rank_budget_gib-rss_gib-8)/workspace_gib)))
        print(f'CONCURRENCY rank={rank} workers={worker_count} blas_threads_per_worker=1 '
              f'rank_budget_gib={rank_budget_gib:.3f} workspace_estimate_gib={workspace_gib:.3f}', flush=True)
        tasks = [(i, p, r, tcur0) for i in range(len(sink_operator))
                 for p in range(0 if i == 0 else 1, sink_mom_count, sink_mom_chunk)
                 for r in range(0, Mom_len, source_mom_chunk)]
        # Warm immutable Wick plans before concurrent cache access.
        for i in range(len(sink_operator)):
            for k, f in ((0, 1), (1, 0), (1, 1)):
                groups = [[s, t, current_operator[f]] for s in sink_operator[i] for t in source_operator[k]]
                run_wick_analysis(groups, Cpt=Cpt,
                    Vindex=['N', 'N', 'M', 'M'] if i == 0 else ['N', 'N', 'N', 'M', 'M'],
                    Gindex=['', 'G', '', ''] if i == 0 else ['', '', 'G', '', ''],
                    use_equivalence=use_equivalence, ignore_dis=ignore_dis,
                    Projection=Projection, gamma_registry=GR, verbose=False)
        st_tasks = time.perf_counter()
        with ThreadPoolExecutor(max_workers=worker_count, initializer=initialize_contraction_thread) as executor:
            futures = [executor.submit(contract_momentum_block, task) for task in tasks]
            for completed, future in enumerate(as_completed(futures), 1):
                (i, p, r, _), task_corr = future.result()
                corr_NJN_GEVP[i, :, :, source_indx, tcur0, ..., p:min(p+sink_mom_chunk, sink_mom_count), r:min(r+source_mom_chunk, Mom_len)] = task_corr
                if completed == 1 or completed % worker_count == 0 or completed == len(tasks):
                    print(f'TIMING_BLOCKS rank={rank} tsrc={tsrc} tcurr={trank[tcur0]} '
                          f'completed={completed}/{len(tasks)} seconds={time.perf_counter()-st_tasks:.6f}', flush=True)
                del task_corr
        elapsed_tcurr = time.perf_counter() - st_tcurr
        memory_phase = 'between_currents'
        print(f'TIMING_TCURR rank={rank} tsrc={tsrc} tcurr={trank[tcur0]} tau={(trank[tcur0]-tsrc)%Nt} seconds={elapsed_tcurr:.6f}', flush=True)
        report_memory('tcurr', tsrc=tsrc, tcurr=trank[tcur0])

    comm.Barrier()
    if rank == 0:
        print(f'calculate 3pt of tsrc {tsrc} use time {(time.perf_counter() - st_cal):.3f} s')
    report_memory('tsrc', tsrc=tsrc)

# M8 Gather and save. All operators carry odd fermion number (3 or 5).
# One temporal wrap gives -1; meson pairs are periodic. Apply this once here.
corr_NJN_GEVP = get_mpi_data(corr_NJN_GEVP, mdtype = 'TGather', root = 0, axis = 4)

if rank == 0:
    corr_save_path = os.environ.get('CPU_BENCHMARK_RESULT_DIR', f'/public/home/sush/distillation/0v2b/result_nex/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}')

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
    backend.save(f'{corr_save_path}/corr_3pt_vector_GEVP_tsep{tsep}{source_tag}_cpu.npy', corr_sum)
    # Store only common correlators. These coefficients specify later combinations:
    # V^(1,0)=(C_d-C_u)/sqrt(2); V^(0,0)=(C_u+C_d)/sqrt(2), Eq. (S9).
    # In this reversed orientation, C_opt = sum_a,m d_a * conj(c_m) * C_a,m.
    # For the paper's linear truncation retain a=0 OR m=0, after shell averaging.
    
    np.savez(f'{corr_save_path}/corr_3pt_vector_GEVP_tsep{tsep}{source_tag}_cpu_basis.npz',
            axes = ['N_basis', 'source_charge', 'current_flavor', 'tsrc_sum', 'tJ_minus_tsrc', 'mu', 'spin_N', 'spin_Npi', 'p', 'r'],
            N_basis = ['N(p)', 'sigma_p*N(p)pi(0)', 'sigma_p*N(0)pi(p)'],
            source_charge = ['p*pi0_E', 'n*pi+_E'], current_flavor = ['u', 'd'],
            mu = [1, 2, 3, 4], p = Mom_sink_VVV[:sink_mom_count], r = Mom_sink_VDV,
            source_shell_q2 = np.sum(np.asarray(Mom_sink_VDV)**2, axis = 1),
            source_isospin = [0.5, 1.5], source_CG = [[np.sqrt(1/3), np.sqrt(2/3)], [-np.sqrt(2/3), np.sqrt(1/3)]],
            current_isospin = [1, 0], current_coefficients = [[-np.sqrt(1/2), np.sqrt(1/2)], [np.sqrt(1/2), np.sqrt(1/2)]],
            N_admixture_valid = np.any(np.asarray(Mom_sink_VVV[:sink_mom_count]) != 0, axis = 1),
            insertion_time_offsets = np.arange(current_start, current_stop),
            current_window = [current_start, current_stop],
            momentum_component_order = ['pz', 'py', 'px'], current_convention = 'Euclidean; mu=4 is time',
            source_time_count = len(source_times), source_times = source_times, tsep = tsep,
            current_renormalized = False, source_time_reduction = 'sum',
            full_five_point = False, five_point_factorized = False, gevp_cross_terms = True,
            ignore_current_loops = ignore_dis, use_equivalence = use_equivalence,
            batched_lorentz = True, batched_N_admixtures = False, shared_current_insertion = False,
            sink_mom_count = sink_mom_count, sink_mom_chunk = sink_mom_chunk, source_mom_chunk = source_mom_chunk,
            compute_backend = backend.__name__, mpi_ranks = size, threads_per_rank = threads_per_rank,
            contraction_workers_requested = requested_workers, source_momentum_parallel = True,
            contraction_blas_threads_per_worker = 1, memory_budget_gib = node_budget_gib,
            perambulator_storage = 'host', endpoint_VVV_storage = 'host',
            independent_channels = [[0, 1], [1, 0], [1, 1]], reconstructed_channel = [0, 0],
            isospin_reconstruction = 'C_Au=(C_Bu+C_Bd)/sqrt(2)-C_Ad'
            )
    
    print(f'total elapsed time {time.perf_counter() - st_total:.3f} s')
report_memory('final')
if os.environ.get('CPU_BENCHMARK', '0') == '1':
    memory_stop.set()
    memory_thread.join()
