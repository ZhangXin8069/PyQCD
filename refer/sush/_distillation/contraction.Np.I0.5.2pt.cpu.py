import os
import sys
import time
import pathlib
import resource
from concurrent.futures import ThreadPoolExecutor, as_completed

# M0 Runtime: CPU backend; keep library numerical kernels unchanged.
threads_per_rank = int(os.environ.get('SLURM_CPUS_PER_TASK', '3'))
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = str(threads_per_rank)
os.environ['OMP_DYNAMIC'] = 'FALSE'
os.environ['MKL_DYNAMIC'] = 'FALSE'
sys.path.insert(0, '/public/home/sush/distillation/lqcddb_update/src')
import numpy as np
import mkl
from mpi4py import MPI
from lqcddb import (set_backend, get_backend, mpinit, getMPIRank, getMPISize,
                    getMPIComm, get_mpi_tlist, get_mpi_data, creat_mom_list,
                    vertex_creator, conjugate_operator, run_wick_analysis,
                    PeramRegistry, VRegistry, GammaRegistry, dynamic_contraction,
                    gamma, seq_peram, loop_tsrc)

st_total = time.perf_counter()
set_backend('numpy')
backend = get_backend()
conf_id = sys.argv[1]

# M1 Lattice and MPI: arbitrary cyclic time owners, including 24 ranks.
lattice_size = [32, 32, 32, 64]
mpi_size = MPI.COMM_WORLD.Get_size()
grid_size = [1, 1, 1, mpi_size]
mpinit(grid_size=grid_size, latt_size=lattice_size, backend=backend.__name__)
rank, size, comm = getMPIRank(), getMPISize(), getMPIComm()
Nx, Ny, Nz, Nt = lattice_size
trank, _, _ = get_mpi_tlist(Nt=Nt, t=range(Nt), gtype='TScatter')
Lt = len(trank)
source_times = [int(sys.argv[2])] if len(sys.argv) > 2 else list(range(Nt))
dt_start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
dt_stop = int(sys.argv[4]) if len(sys.argv) > 4 else Nt
if any(t < 0 or t >= Nt for t in source_times) or not 0 <= dt_start < dt_stop <= Nt:
    raise ValueError('Require 0 <= source < 64 and 0 <= dt_start < dt_stop <= 64')
vertex_times = sorted(set(source_times) | {(t+d) % Nt for t in source_times for d in range(dt_start, dt_stop)})
print(f'RUNTIME rank={rank} pid={os.getpid()} affinity={sorted(os.sched_getaffinity(0))} ranks={size} threads={threads_per_rank}', flush=True)

# M2 Momenta: p labels pion momentum; nucleon momentum is -p on BOTH sides.
fun_eigen = vertex_creator(Nx=Nx)
Mom_VDV = [[0, 0, 0]] + sorted(creat_mom_list(Mom=[0, 0, 1], fix_Q2=True)) + sorted(creat_mom_list(Mom=[0, 1, 1], fix_Q2=True)) + sorted(creat_mom_list(Mom=[1, 1, 1], fix_Q2=True))
Mom_VVV = [[-x for x in p] for p in Mom_VDV]
Mom_len = len(Mom_VDV)
if Mom_len != 27 or len({tuple(p) for p in Mom_VDV}) != 27:
    raise ValueError('Expected 27 distinct momentum modes')
phase_VDV = np.stack([fun_eigen.phase_exp_VdV(Mom=p) for p in Mom_VDV])
phase_VVV = np.stack([fun_eigen.phase_exp_VVV(Mom=p) for p in Mom_VVV])
Nev_src = 100
mom_chunk = int(os.environ.get('NPI_MOM_CHUNK', '1'))
sample_pairs = os.environ.get('NPI_BENCHMARK_SAMPLE', '0') == '1'
if sample_pairs and mom_chunk != 1:
    raise ValueError('Benchmark sample requires mom_chunk=1')
if not 1 <= mom_chunk <= Mom_len:
    raise ValueError('NPI_MOM_CHUNK must be in 1..27')

# M3 Dynamic manifest: pi0_E=(ubar gamma5 u-dbar gamma5 d)/sqrt(2).
# Quark-field isospin phases give O_1/2,+1/2 = p*pi0_E/sqrt(3)
# + sqrt(2/3)*n*pi+_E (global sign immaterial when used at both ends).
# Source is the exact conjugate, including coefficient and hadron reversal.
proton = ['|', 'u', 'u', 'gamma_7', 'd', '|']
neutron = ['|', 'd', 'u', 'gamma_7', 'd', '|']
pion_u = ['|', 'u^d', 'gamma_5', 'u', '|']
pion_d = ['|', 'd^d', 'gamma_5', 'd', '|']
pion_plus = ['|', 'd^d', 'gamma_5', 'u', '|']
sink_operator = [[np.sqrt(1/6), *proton, *pion_u],
                 [-np.sqrt(1/6), *proton, *pion_d],
                 [np.sqrt(2/3), *neutron, *pion_plus]]
source_operator = [conjugate_operator(term) for term in sink_operator]
operator_groups = [[s, t] for s in sink_operator for t in source_operator]
Cpt, Projection, ignore_dis = '2pt', True, False
# Exact only with shared u/d perams, antisymmetric VVV, and actual gamma signs.
use_equivalence = True
optimize = ['greedy', 'dp', 'auto']
Vindex = ['M', 'M', 'N', 'N']  # source hadrons reversed: pi then N; both label N.
dynamic_plan = run_wick_analysis(operator_groups, Cpt=Cpt, Vindex=Vindex,
    Oindex='MN', use_equivalence=False, ignore_dis=ignore_dis, verbose=False)
required_peram = sorted({(p[2], tuple(p[4])) for entry in dynamic_plan for p in entry[2]['peram'][entry[3]]})
required_v = sorted({(v[1].split('_')[0]+'_0', v[3]) for entry in dynamic_plan for v in entry[2]['V']})
required_gamma = sorted({g[1] for entry in dynamic_plan for g in entry[2]['gamma_pos']})
# required peram: shared light, all four (tsink/tsrc,tsink/tsrc) directions;
# required V: VVV_0 and VDV_0 at tsink/tsrc; gamma_7,gamma_5,Projector.
# peram axes (spin_q,spin_aq,eigen_q,eigen_aq)=(4,4,100,100).
# VVV axes (momentum,e1,e2,e3); VDV axes (momentum,e_aq,e_q).
# Output MN=(pion_p_sink,pion_p_source), no momentum trace/sum/shell average.
# Estimator Tr[P_+ <O(t_sink) Obar(t_src)>], P_+=(1+gamma4)/2, complex128.
if rank == 0:
    print(f'MANIFEST raw_diagrams={len(dynamic_plan)} peram={required_peram} V={required_v} gamma={required_gamma}', flush=True)

# M4 Vertices: only requested times; persistent rank-local cache in RAM.
eigen_base = '/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64'
peram_base = '/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light'
eigen_tag = 'e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy'
VDV_cache, VVV_cache = {}, {}
comm.Barrier()
st_eigen = time.perf_counter()
for t in trank:
    if t not in vertex_times:
        continue
    st_vertex = time.perf_counter()
    eigvecs = np.ascontiguousarray(np.load(f'{eigen_base}/{conf_id}/{conf_id}_t{t:03d}_{eigen_tag}.npy', mmap_mode='r')[:Nev_src])
    VDV_cache[t] = fun_eigen.Mom_VdV_sink_t(phase_exp=phase_VDV, eigvecs=eigvecs)
    VVV_cache[t] = np.empty((Mom_len, Nev_src, Nev_src, Nev_src), dtype=complex)
    for p in range(Mom_len):
        VVV_cache[t][p] = fun_eigen.Mom_VVV_sink_t(phase_exp=phase_VVV[p], eigvecs=eigvecs)[0]
    print(f'TIMING_VERTEX rank={rank} t={t} seconds={time.perf_counter()-st_vertex:.6f}', flush=True)
    del eigvecs
comm.Barrier()
if rank == 0:
    print(f'TIMING vertices_seconds={time.perf_counter()-st_eigen:.6f}', flush=True)
del phase_VDV, phase_VVV

# M5 Perams: cache forward data and required local diagonals, not all 64x64
# unless the requested sources require it. mmap avoids reading unused eigen rows.
comm.Barrier()
st_load = time.perf_counter()
if rank == 0:
    peram_u = np.empty((len(source_times), Nt, 4, 4, Nev_src, Nev_src), dtype=complex)
    for si, t in enumerate(source_times):
        file_data = np.load(f'{peram_base}/{conf_id}/t{t:03d}_{eigen_tag}.npy', mmap_mode='r')
        peram_u[si] = file_data[..., :Nev_src, :Nev_src]
    del file_data
else:
    peram_u = None
peram_u = get_mpi_data(peram_u, mdtype='TScatter', root=0, axis=1)
diagonal = {}
for t in trank:
    if t in vertex_times:
        file_data = np.load(f'{peram_base}/{conf_id}/t{t:03d}_{eigen_tag}.npy', mmap_mode='r')
        diagonal[t] = np.ascontiguousarray(file_data[t, ..., :Nev_src, :Nev_src])
        del file_data
comm.Barrier()
if rank == 0:
    print(f'TIMING peram_seconds={time.perf_counter()-st_load:.6f}', flush=True)

# M6 Registries: exact full positive-parity trace, with no extra spin average.
GR, PR, VR = GammaRegistry(), PeramRegistry(), VRegistry()
GR.register('gamma_7', gamma(7))
GR.register('gamma_5', gamma(5))
projection = (gamma(0)+gamma(4))/2
GR.register('Projector', {'sink': projection, 'source': projection, 'contraction': ['0-0']})
plan = run_wick_analysis(operator_groups, Cpt=Cpt, Vindex=Vindex, Oindex='MN',
    use_equivalence=use_equivalence, ignore_dis=ignore_dis, Projection=Projection,
    gamma_registry=GR, verbose=False)
if rank == 0:
    print(f'PLAN classes={len(plan)} nonzero={sum(abs(sum(c for _,_,c in e[0]))>1e-12 for e in plan)}', flush=True)

# M7 Dynamic loop. Physically meaningful task: one independent momentum block.
# Inputs PR/GR are immutable until all blocks of this time pair are complete.
def contract_momentum_block(momentum_pair):
    m, n = momentum_pair
    local_VR = VRegistry()
    local_VR.register('VVV_0', 'tsink', VVV_cache[t_sink][m:m+mom_chunk])
    local_VR.register('VDV_0', 'tsink', VDV_cache[t_sink][m:m+mom_chunk])
    local_VR.register('VVV_0', 'tsrc', VVV_source[n:n+mom_chunk])
    local_VR.register('VDV_0', 'tsrc', VDV_source[n:n+mom_chunk])
    dc = dynamic_contraction(operator_groups,
        peram_registry=PR, v_registry=local_VR, gamma_registry=GR,
        Cpt=Cpt, Vindex=Vindex, Oindex='MN', use_equivalence=use_equivalence,
        ignore_dis=ignore_dis, Projection=Projection, optimize=optimize, verbose=False)
    return m, n, dc.calculate_all()

job_id = os.environ.get('SLURM_JOB_ID', 'local')
sample_tag = '_sample27' if sample_pairs else ''
result_dir = pathlib.Path(os.environ.get('NPI_RESULT_DIR', f'/public/home/sush/distillation/0v2b/result_nex/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}/Npi_I0.5_cpu_{job_id}{sample_tag}'))
result_dir.mkdir(parents=True, exist_ok=True)
corr = np.full((len(source_times), Lt, Mom_len, Mom_len), np.nan+1j*np.nan, dtype=complex)
workers = min(threads_per_rank, int(os.environ.get('NPI_WORKERS', str(threads_per_rank))))
if workers < 1:
    raise ValueError('NPI_WORKERS must be positive')
if rank == 0:
    print(f'BENCHMARK sample_pairs={sample_pairs} workers={workers} momenta_per_pair={Mom_len if sample_pairs else Mom_len**2}', flush=True)
# MKL remains threaded during vertices, then use outer momentum parallelism.
mkl.set_num_threads(1)
with ThreadPoolExecutor(max_workers=workers) as executor:
    for source_index, t_source in enumerate(source_times):
        comm.Barrier()
        st_cal = time.perf_counter()
        owner, local_source = get_mpi_tlist(Nt=Nt, t=t_source, gtype='find')
        VVV_source = get_mpi_data(VVV_cache[t_source] if rank == owner else None, mdtype='Bcast', root=owner).conj()
        VDV_source = get_mpi_data(VDV_cache[t_source] if rank == owner else None, mdtype='Bcast', root=owner).conj().transpose(0, 2, 1)
        source_diagonal = get_mpi_data(diagonal[t_source] if rank == owner else None, mdtype='Bcast', root=owner)
        PR.register('light', ('tsrc', 'tsrc'), source_diagonal)
        sink_times, _, sink_indices = get_mpi_tlist(Nt=Nt, t=range(t_source+dt_start, t_source+dt_stop), gtype='TScatter')
        for t_sink, sink_index in zip(sink_times, sink_indices):
            st_pair = time.perf_counter()
            PR.register('light', ('tsink', 'tsrc'), peram_u[source_index, sink_index])
            PR.register('light', ('tsrc', 'tsink'), seq_peram(peram_u[source_index, sink_index]))
            PR.register('light', ('tsink', 'tsink'), diagonal[t_sink])
            momentum_pairs = ([(m, (5*m) % Mom_len) for m in range(Mom_len)] if sample_pairs else
                [(m,n) for m in range(0,Mom_len,mom_chunk) for n in range(0,Mom_len,mom_chunk)])
            futures = [executor.submit(contract_momentum_block, pair) for pair in momentum_pairs]
            for completed, future in enumerate(as_completed(futures), 1):
                m, n, block = future.result()
                corr[source_index, sink_index, m:m+mom_chunk, n:n+mom_chunk] = block
                if completed == 1 or completed % 27 == 0 or completed == len(futures):
                    print(f'TIMING_BLOCKS rank={rank} tsrc={t_source} tsink={t_sink} completed={completed}/{len(futures)} seconds={time.perf_counter()-st_pair:.6f}', flush=True)
            np.save(result_dir/f'pair_tsrc{t_source}_tsink{t_sink}.npy', corr[source_index, sink_index])
            print(f'TIMING_PAIR rank={rank} tsrc={t_source} tsink={t_sink} seconds={time.perf_counter()-st_pair:.6f} peak_rss_gib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.6f}', flush=True)
        comm.Barrier()
        if rank == 0:
            print(f'TIMING_SOURCE tsrc={t_source} seconds={time.perf_counter()-st_cal:.6f}', flush=True)

# M8 Gather and source sum. Npi has odd fermion parity: one wrap gives -1.
if Nt % size == 0:
    corr = get_mpi_data(corr, mdtype='TGather', root=0, axis=1)
else:
    # TGather requires equal local lengths. Gather actual time labels so that
    # 24 ranks with 2/3 local times retain the physical Nt=64 and exact order.
    parts = comm.gather((trank, corr), root=0)
    if rank == 0:
        corr = np.full((len(source_times), Nt, Mom_len, Mom_len), np.nan+1j*np.nan, dtype=complex)
        for times, values in parts:
            corr[:, times] = values
        del parts
if rank == 0:
    np.save(result_dir/'corr_raw.npy', corr)
    if source_times == list(range(Nt)):
        corr_sum = loop_tsrc(corr, indx=[0,1], Boundary_Conditions='Antiperiodic', Ctype='2pt')
    else:
        corr_sum = np.full((1,Nt,Mom_len,Mom_len), np.nan+1j*np.nan, dtype=complex)
        for si,t_source in enumerate(source_times):
            for dt in range(dt_start,dt_stop):
                t_sink = (t_source+dt)%Nt
                phase = -1 if t_sink < t_source else 1
                corr_sum[0,dt] = phase*corr[si,t_sink]
    np.save(result_dir/'corr_tsrc_sum.npy', corr_sum)
    np.savez(result_dir/'basis.npz', axes=['tsrc','tsink','p_sink','p_source'],
        summed_axes=['tsrc_sum','tau','p_sink','p_source'], pion_momenta=Mom_VDV,
        nucleon_momenta=Mom_VVV, momentum_component_order=['pz','py','px'],
        isospin=0.5, Iz=0.5, source_times=source_times, source_time_count=len(source_times),
        source_time_reduction='sum', time_offsets=np.arange(dt_start,dt_stop),
        coefficients=[np.sqrt(1/6),-np.sqrt(1/6),np.sqrt(2/3)],
        spin_projection='Tr[(1+gamma4)/2 C], no spin average', total_momentum=[0,0,0],
        ignore_dis=ignore_dis, use_equivalence=use_equivalence, mpi_ranks=size,
        workers_per_rank=workers, mom_chunk=mom_chunk, eigenvectors=Nev_src,
        benchmark_sample=sample_pairs, momentum_pairs=momentum_pairs,
        complete_momentum_matrix=not sample_pairs)
    print(f'SAVED {result_dir} raw_shape={corr.shape} summed_shape={corr_sum.shape}', flush=True)
    print(f'TIMING total_seconds={time.perf_counter()-st_total:.6f}', flush=True)
