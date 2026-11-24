import os
import sys
import time
import pathlib
import resource

# M0 Runtime: one MPI process on one GPU, complex128 throughout.
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
sys.path.insert(0, '/public/home/sush/distillation/lqcddb_update/src')
import numpy as np
from mpi4py import MPI
from lqcddb import (set_backend, get_backend, mpinit, getMPIRank, getMPISize,
                    getMPIComm, get_mpi_tlist,
                    creat_mom_list, vertex_creator, conjugate_operator,
                    run_wick_analysis, PeramRegistry, VRegistry, GammaRegistry,
                    dynamic_contraction, gamma, seq_peram, loop_tsrc)

st_total = time.perf_counter()
set_backend('cupy')
backend = get_backend()
conf_id = sys.argv[1]

# M1 Lattice and MPI: one rank owns all absolute times.
lattice_size = [32, 32, 32, 64]
mpi_size = MPI.COMM_WORLD.Get_size()
if mpi_size != 1:
    raise ValueError('Require exactly one MPI process for the NA800 version')
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
print(f'RUNTIME rank={rank} gpu={backend.cuda.runtime.getDevice()} ranks={size} CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES", "unset")}', flush=True)

# M2 Momenta: pion p, nucleon -p, independently at source and sink.
fun_eigen = vertex_creator(Nx=Nx)
Mom_VDV = [[0, 0, 0]] + sorted(creat_mom_list(Mom=[0, 0, 1], fix_Q2=True)) + sorted(creat_mom_list(Mom=[0, 1, 1], fix_Q2=True)) + sorted(creat_mom_list(Mom=[1, 1, 1], fix_Q2=True))
Mom_VVV = [[-x for x in p] for p in Mom_VDV]
Mom_len = len(Mom_VDV)
Nev_src = 100
mom_chunk = int(os.environ.get('NPI_MOM_CHUNK', '9'))
sample_pairs = os.environ.get('NPI_BENCHMARK_SAMPLE', '0') == '1'
if not 1 <= mom_chunk <= Mom_len or (sample_pairs and mom_chunk != 1):
    raise ValueError('Require 1 <= NPI_MOM_CHUNK <= 27; sample requires 1')
phase_VDV = backend.stack([fun_eigen.phase_exp_VdV(Mom=p) for p in Mom_VDV])
phase_VVV = backend.stack([fun_eigen.phase_exp_VVV(Mom=p) for p in Mom_VVV])

# M3 Dynamic manifest. pi0_E=(ubar gamma5 u-dbar gamma5 d)/sqrt(2).
# O_1/2,+1/2 = p*pi0_E/sqrt(3) + sqrt(2/3)*n*pi+_E (CPU convention).
# O_3/2,+1/2 = sqrt(2/3)*p*pi0_E - n*pi+_E/sqrt(3).
# The relative minus follows I_- pi+_E = -sqrt(2)*pi0_E.
# Unweighted basis A=(p*ubar5u, p*dbar5d, n*dbar5u); source=A^dagger.
proton = ['|', 'u', 'u', 'gamma_7', 'd', '|']
neutron = ['|', 'd', 'u', 'gamma_7', 'd', '|']
pion_u = ['|', 'u^d', 'gamma_5', 'u', '|']
pion_d = ['|', 'd^d', 'gamma_5', 'd', '|']
pion_plus = ['|', 'd^d', 'gamma_5', 'u', '|']
sink_operator = [[*proton, *pion_u], [*proton, *pion_d], [*neutron, *pion_plus]]
source_operator = [conjugate_operator(term) for term in sink_operator]
operator_groups = [[s, t] for s in sink_operator for t in source_operator]
cg = np.array([[np.sqrt(1/6), -np.sqrt(1/6), np.sqrt(2/3)],
               [np.sqrt(1/3), -np.sqrt(1/3), -np.sqrt(1/3)]])
Cpt, Projection, ignore_dis = '2pt', True, False
optimize = ['greedy', 'dp', 'auto']
Vindex = ['M', 'M', 'N', 'N']
dynamic_plan = run_wick_analysis(operator_groups, Cpt=Cpt, Vindex=Vindex,
    Oindex='MN', use_equivalence=False, ignore_dis=ignore_dis, verbose=False)
required_peram = sorted({(p[2], tuple(p[4])) for entry in dynamic_plan for p in entry[2]['peram'][entry[3]]})
required_v = sorted({(v[1].split('_')[0]+'_0', v[3]) for entry in dynamic_plan for v in entry[2]['V']})
required_gamma = sorted({g[1] for entry in dynamic_plan for g in entry[2]['gamma_pos']})
# required peram: light at all four (tsink/tsrc,tsink/tsrc), axes (4,4,100,100);
# required V: VVV_0 (mom,e1,e2,e3), VDV_0 (mom,e_aq,e_q) at tsink/tsrc;
# required gamma: gamma_7,gamma_5,Projector; output axes: I,tsrc,tsink,M,N.
# Estimator: Tr[P_+ <O_I Obar_I>], P_+=(1+gamma4)/2; no spin average.
if rank == 0:
    print(f'MANIFEST raw_diagrams={len(dynamic_plan)} peram={required_peram} V={required_v} gamma={required_gamma}', flush=True)

# M4 Vertices: construct and retain all required vertices on GPU.
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
    eigvecs = backend.asarray(np.load(f'{eigen_base}/{conf_id}/{conf_id}_t{t:03d}_{eigen_tag}.npy')[:Nev_src])
    VDV_cache[t] = fun_eigen.Mom_VdV_sink_t(phase_exp=phase_VDV, eigvecs=eigvecs)
    VVV_cache[t] = backend.empty((Mom_len, Nev_src, Nev_src, Nev_src), dtype=complex)
    for p in range(Mom_len):
        VVV_cache[t][p] = fun_eigen.Mom_VVV_sink_t(phase_exp=phase_VVV[p], eigvecs=eigvecs)[0]
    backend.cuda.get_current_stream().synchronize()
    print(f'TIMING_VERTEX rank={rank} t={t} seconds={time.perf_counter()-st_vertex:.6f}', flush=True)
    del eigvecs
del phase_VDV, phase_VVV
backend.cuda.get_current_stream().synchronize()
backend.get_default_memory_pool().free_all_blocks()
comm.Barrier()
if rank == 0:
    print(f'TIMING vertices_seconds={time.perf_counter()-st_eigen:.6f}', flush=True)

# M5 Perams: file t_source stores tau[t_sink,sq,saq,eq,eaq], ABSOLUTE times.
# Load each source row directly into its slot in the GPU cache.
# peram_u[si,li] = tau(t_sink=trank[li], t_source=source_times[si]).
comm.Barrier()
st_load = time.perf_counter()
peram_u = backend.empty((len(source_times), Lt, 4, 4, Nev_src, Nev_src), dtype=complex)
for si, t in enumerate(source_times):
    file_data = np.load(f'{peram_base}/{conf_id}/t{t:03d}_{eigen_tag}.npy')
    peram_row = np.ascontiguousarray(file_data[..., :Nev_src, :Nev_src], dtype=np.complex128)
    del file_data
    peram_u[si].set(peram_row)
    del peram_row
    if rank == 0:
        print(f'TIMING_PERAM_SOURCE tsrc={t} loaded={si+1}/{len(source_times)} seconds={time.perf_counter()-st_load:.6f}', flush=True)
diagonal = {}
for local_index, t in enumerate(trank):
    if t in vertex_times:
        if t in source_times:
            diagonal[t] = peram_u[source_times.index(t), local_index]
        else:
            file_data = np.load(f'{peram_base}/{conf_id}/t{t:03d}_{eigen_tag}.npy')
            diagonal[t] = backend.asarray(file_data[t, ..., :Nev_src, :Nev_src])
            del file_data
backend.cuda.get_current_stream().synchronize()
comm.Barrier()
if rank == 0:
    print(f'TIMING peram_seconds={time.perf_counter()-st_load:.6f}', flush=True)

# M6 Registries and shared representatives. Equivalence is valid because u/d
# resolve to the SAME light tensor, VVV is alternating in its last three axes,
# gamma_7.T=-gamma_7, gamma_5.T=gamma_5, and topology/time/output stay fixed.
# Only numerical CG weights differ; do not sum an unweighted class first:
# its unweighted sum may vanish while a CG-weighted sum does not.
GR, PR, VR = GammaRegistry(), PeramRegistry(), VRegistry()
GR.register('gamma_7', gamma(7))
GR.register('gamma_5', gamma(5))
projection = (gamma(0)+gamma(4))/2
GR.register('Projector', {'sink': projection, 'source': projection, 'contraction': ['0-0']})
use_equivalence = True
plan = run_wick_analysis(operator_groups, Cpt=Cpt, Vindex=Vindex, Oindex='MN',
    use_equivalence=use_equivalence, ignore_dis=ignore_dis, Projection=Projection,
    gamma_registry=GR, verbose=False)
shared_plan, channel_weights = [], []
for entry in plan:
    weights = np.array([sum(coeff*c[g//3]*c[g%3].conjugate() for g, _, coeff in entry[0]) for c in cg])
    if np.any(np.abs(weights) > 1e-12):
        shared_plan.append([[(entry[1], entry[3], 1.0)], *entry[1:]])
        channel_weights.append(weights)
channel_weights = np.array(channel_weights).T
weights_gpu = backend.asarray(channel_weights)
if rank == 0:
    print(f'PLAN classes={len(plan)} shared_contractions={len(shared_plan)} per_channel={np.sum(abs(channel_weights)>1e-12, axis=1).tolist()}', flush=True)

# M7 Dynamic loop: calculate(i) exposes one representative for BOTH channels.
# The library's cached plan is preserved; dc.plan is a separate unit-weight list.
job_id = os.environ.get('SLURM_JOB_ID', 'local')
sample_tag = '_sample27' if sample_pairs else ''
result_dir = pathlib.Path(os.environ.get('NPI_RESULT_DIR', f'/public/home/sush/distillation/0v2b/result_nex/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}'))
result_dir.mkdir(parents=True, exist_ok=True)
corr = backend.full((2, len(source_times), Lt, Mom_len, Mom_len), np.nan+1j*np.nan, dtype=complex)
momentum_pairs = ([(m, (5*m) % Mom_len) for m in range(Mom_len)] if sample_pairs else
    [(m,n) for m in range(0,Mom_len,mom_chunk) for n in range(0,Mom_len,mom_chunk)])
dc = None
for source_index, t_source in enumerate(source_times):
    comm.Barrier()
    st_cal = time.perf_counter()
    VVV_source = VVV_cache[t_source].conj()
    VDV_source = VDV_cache[t_source].conj().transpose(0, 2, 1)
    PR.register('light', ('tsrc', 'tsrc'), diagonal[t_source])
    sink_times, _, sink_indices = get_mpi_tlist(Nt=Nt, t=range(t_source+dt_start, t_source+dt_stop), gtype='TScatter')
    for t_sink, sink_index in zip(sink_times, sink_indices):
        st_pair = time.perf_counter()
        forward = peram_u[source_index, sink_index]
        PR.register('light', ('tsink', 'tsrc'), forward)
        # Reverse BOTH spin/eigenvector ends with gamma5 Hermitian conjugation.
        # No additional temporal phase: the saved tau already contains the BC.
        PR.register('light', ('tsrc', 'tsink'), seq_peram(forward))
        PR.register('light', ('tsink', 'tsink'), diagonal[t_sink])
        VVV_sink = VVV_cache[t_sink]
        VDV_sink = VDV_cache[t_sink]
        for completed, (m, n) in enumerate(momentum_pairs, 1):
            VR.register('VVV_0', 'tsink', VVV_sink[m:m+mom_chunk])
            VR.register('VDV_0', 'tsink', VDV_sink[m:m+mom_chunk])
            VR.register('VVV_0', 'tsrc', VVV_source[n:n+mom_chunk])
            VR.register('VDV_0', 'tsrc', VDV_source[n:n+mom_chunk])
            if dc is None:
                dc = dynamic_contraction(operator_groups,
                    peram_registry=PR, v_registry=VR, gamma_registry=GR,
                    Cpt=Cpt, Vindex=Vindex, Oindex='MN', use_equivalence=use_equivalence,
                    ignore_dis=ignore_dis, Projection=Projection, optimize=optimize, verbose=False)
                dc.plan = shared_plan
            block = backend.zeros((2, min(mom_chunk, Mom_len-m), min(mom_chunk, Mom_len-n)), dtype=complex)
            for diagram_index in range(len(shared_plan)):
                value = dc.calculate(diagram_index)
                block += weights_gpu[:, diagram_index, None, None]*value[None]
            corr[:, source_index, sink_index, m:m+mom_chunk, n:n+mom_chunk] = block
            if completed == 1 or completed % 9 == 0 or completed == len(momentum_pairs):
                backend.cuda.get_current_stream().synchronize()
                print(f'TIMING_BLOCKS rank={rank} tsrc={t_source} tsink={t_sink} completed={completed}/{len(momentum_pairs)} seconds={time.perf_counter()-st_pair:.6f}', flush=True)
        # np.save(result_dir/f'pair_tsrc{t_source}_tsink{t_sink}.npy', backend.asnumpy(corr[:, source_index, sink_index]))
        print(f'TIMING_PAIR rank={rank} tsrc={t_source} tsink={t_sink} seconds={time.perf_counter()-st_pair:.6f} peak_rss_gib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.6f} gpu_pool_gib={backend.get_default_memory_pool().total_bytes()/2**30:.6f}', flush=True)
    backend.cuda.get_current_stream().synchronize()
    comm.Barrier()
    if rank == 0:
        print(f'TIMING_SOURCE tsrc={t_source} seconds={time.perf_counter()-st_cal:.6f}', flush=True)

# M8 Source sum in absolute-time order. Five cross-time fermion lines or three plus
# equal-time loops give odd wrap parity; apply the correlator wrap sign ONCE.
if rank == 0:
    # np.save(result_dir/'corr_2pt_Iz_0.5_raw.npy', backend.asnumpy(corr))
    if source_times == list(range(Nt)):
        corr_sum = loop_tsrc(corr, indx=[1,2], Boundary_Conditions='Antiperiodic', Ctype='2pt')
    else:
        corr_sum = backend.full((2,1,Nt,Mom_len,Mom_len), np.nan+1j*np.nan, dtype=complex)
        for si,t_source in enumerate(source_times):
            for dt in range(dt_start,dt_stop):
                t_sink = (t_source+dt)%Nt
                phase = -1 if t_sink < t_source else 1
                corr_sum[:,0,dt] = phase*corr[:,si,t_sink]
    np.save(result_dir/'corr_2pt_Iz_0.5_tsrc_sum.npy', backend.asnumpy(corr_sum))
    np.savez(result_dir/'corr_2pt_Iz_0.5_basis.npz', axes=['isospin','tsrc','tsink','p_sink','p_source'],
        summed_axes=['isospin','tsrc_sum','tau','p_sink','p_source'], pion_momenta=Mom_VDV,
        nucleon_momenta=Mom_VVV, momentum_component_order=['pz','py','px'],
        isospin=[0.5,1.5], Iz=0.5, source_times=source_times, source_time_count=len(source_times),
        source_time_reduction='sum', time_offsets=np.arange(dt_start,dt_stop), coefficients=cg,
        shared_weights=channel_weights, basis=['p*ubar5u','p*dbar5d','n*dbar5u'],
        tensor_backend='cupy', vertex_cache='device', peram_cache='device',
        reverse_peram='gamma5 Hermitian conjugation of forward tau',
        spin_projection='Tr[(1+gamma4)/2 C], no spin average', total_momentum=[0,0,0],
        ignore_dis=ignore_dis, use_equivalence=use_equivalence, mpi_ranks=size,
        mom_chunk=mom_chunk, eigenvectors=Nev_src, benchmark_sample=sample_pairs,
        momentum_pairs=momentum_pairs, complete_momentum_matrix=not sample_pairs)
    print(f'SAVED {result_dir} raw_shape={corr.shape} summed_shape={corr_sum.shape}', flush=True)
    print(f'TIMING total_seconds={time.perf_counter()-st_total:.6f}', flush=True)
