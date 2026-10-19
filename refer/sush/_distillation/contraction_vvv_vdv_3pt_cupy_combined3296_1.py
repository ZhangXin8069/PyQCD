import os
import sys
import atexit
from lqcddb import *
from opt_einsum import contract
try:
    from opt_einsum import contract_expression, shared_intermediates
except ImportError:
    contract_expression = None
    shared_intermediates = None
import time
import gc
from collections import OrderedDict
from contextlib import nullcontext
from itertools import combinations
import numpy as np
import cupy as cp

set_backend('cupy')
backend = get_backend()

lattice_size = [32, 32, 32, 96]
grid_size = [1, 1, 1, int(os.environ.get('MPI_TGRID', '1'))]

conf_id = sys.argv[1]

mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)
comm = getMPIComm()
rank = getMPIRank()
size = getMPISize()
cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()

def env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def create_omega(
      Nx:int,
      exact:int,
      N_eigen:list,
      N_sum:list,
      N_extract:list,
      noise:int,
      conserved:str = False,
      Nbin:int = 1,
      dim:int = 2,
      ):

  if conserved:
      dim = 2

  Nc = 3
  Nv = Nx**3 * Nc

  if N_eigen != [0]:
      tran_N_eigen = []
      tran_N_sum = []

      for i in range(len(N_sum)):
          for j in range(int(N_sum[i] / N_extract[i])):
              if (N_sum[i] / N_extract[i])%1 != 0:
                  raise print(f"extract: {N_extract[i]} can't to be divisible by sum vector :{N_sum[i]}, please check the block is right")

              tran_N_sum += [N_extract[i]]

              if (N_eigen[i]/(N_sum[i] / N_extract[i]))%1 != 0:
                  raise print(f"sum vectors / extract vectors:{N_sum[i] / N_extract[i]} can't to be divisible by block vectors:{N_eigen[i]}, please check the block is right")

              tran_N_eigen += [int(N_eigen[i]/(N_sum[i] / N_extract[i]))]

  else:
      tran_N_sum = N_sum
      tran_N_eigen = N_eigen

  ev_space = [x for x in ([exact] + tran_N_eigen + [(Nv / Nbin - sum([exact] + tran_N_eigen))]) if x != 0]
  ev_sum = [x for x in ([exact] + tran_N_sum + [noise]) if x != 0]

  len_of_space = len(ev_sum)
  Nev = sum(ev_sum)

  coordinate_position = []
  for i in range(len_of_space):
      coordinate_position += [range(sum(ev_sum[:i]), sum(ev_sum[:(i+1)]))]

  weights = np.zeros((len_of_space, dim), dtype = float)
  for i in range(len_of_space):
      for j in range(dim):
          weights[i, j] = (ev_space[i] - j) / (ev_sum[i] - j)

  omega = np.ones([Nev]*dim, dtype=float)
  all_position = np.unique(np.asarray(list(combinations(range(dim * len_of_space), dim)))%len_of_space, axis = 0)

  for indx_of_position in all_position:
      sum_of_position = [0] * len_of_space
      for j in indx_of_position:
          sum_of_position[j] += 1

      _position = []
      for dim_indx in range(dim):
          _position += [coordinate_position[indx_of_position[dim_indx]]]

      _weights = 1.0
      for space_indx in range(len_of_space):
          _weights *= np.prod(weights[space_indx, :sum_of_position[space_indx]])
      omega[np.ix_(*_position)] = _weights

      indices = np.ogrid[_position]

      for extra_dim in range(1, len_of_space):
          for i in range(sum_of_position[extra_dim] - 1):
              for j in list(combinations((np.argwhere(indx_of_position == extra_dim).reshape(-1)).tolist(), 2 + i)):
                  _W = omega[np.ix_(*_position)]
                  _W_bool = np.ones(tuple([len(_position[x]) for x in range(dim)]), dtype = bool)
                  _weights_diag = _weights.copy()

                  for k in range(len(j) - 1):
                      _W_bool &= indices[j[k]] == indices[j[k + 1]]
                      _weights_diag = _weights_diag / weights[extra_dim, sum_of_position[extra_dim] - k - 1]

                  _W[_W_bool] = _weights_diag
                  omega[np.ix_(*_position)] = _W

                  if len(j) == 2 and dim == 4:
                      m, n = [x for x in range(dim) if x not in j]

                      if indx_of_position[m] == indx_of_position[n] and indx_of_position[m] != extra_dim:
                          _W_bool &= indices[m] == indices[n]
                          _weights_diag = _weights_diag / weights[indx_of_position[m], 1]
                          _W[_W_bool] = _weights_diag
                          omega[np.ix_(*_position)] = _W

                  if sum_of_position[extra_dim] - (2 + i) == 2:
                      m, n = [x for x in range(dim) if x not in j]
                      _W_bool &= indices[m] == indices[n]

                      _weights_diag = _weights_diag / weights[extra_dim, sum_of_position[extra_dim] - k - 2]
                      _W[_W_bool] = _weights_diag
                      omega[np.ix_(*_position)] = _W

  return omega * Nbin


def eigvec_path(conf_id, t):
    return (
        '/nexdata/project/lqcd/sush/eigensystem/'
        'beta6.41_mu-0.2295_ms-0.2050_L32x96/'
        f'{conf_id}/{conf_id}_t{t:03d}_e100_s[100, 200, 400, 600, 800]_'
        '[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n400_V1_stout_smear_20_0.12.npy'
    )


def combine_diagrams(diagrams):
    combined = OrderedDict()
    for _, prop_str, coeff in diagrams:
        combined[prop_str] = combined.get(prop_str, 0) + coeff
    return [
        (i, prop_str, coeff)
        for i, (prop_str, coeff) in enumerate(combined.items())
        if coeff != 0
    ]


def compile_contracts(contract_specs, operand_shapes, optimize, label):
    compiled = []
    for n, einsum_str, coeff in contract_specs:
        expr = None
        if contract_expression is not None:
            try:
                expr = contract_expression(einsum_str, *operand_shapes, optimize=optimize)
            except Exception as err:
                if rank == 0:
                    print(
                        f"{label} diagram {n}: contract_expression failed, "
                        f"falling back to contract(..., optimize='greedy'): {err}",
                        flush=True,
                    )
        compiled.append((n, einsum_str, expr, coeff))
    if rank == 0:
        print(f"compiled {label} contractions with optimize='{optimize}'", flush=True)
    return compiled


def run_contract(einsum_str, expr, tensors):
    if expr is None:
        return contract(einsum_str, *tensors, optimize='greedy')
    return expr(*tensors)


def shared_context():
    if shared_intermediates is None or not globals().get("USE_SHARED_INTERMEDIATES", 1):
        return nullcontext()
    return shared_intermediates()


class GeneratedTimeSliceReader:
    def __init__(self, label):
        self.label = label
        self.kind = "VdV" if "VdV" in label else "VVV"

    def _mom_bounds(self, mom_slice):
        if mom_slice is None:
            return 0, n_mom
        p_start = 0 if mom_slice.start is None else int(mom_slice.start)
        p_stop = n_mom if mom_slice.stop is None else int(mom_slice.stop)
        if p_start < 0 or p_stop > n_mom or p_start > p_stop:
            raise ValueError("invalid momentum slice %s for %s" % (mom_slice, self.label))
        return p_start, p_stop

    def _generate_local(self, phys_t, mom_slice=None):
        p_start, p_stop = self._mom_bounds(mom_slice)
        if self.kind == "VVV":
            out = np.empty((p_stop - p_start, Nev_src, Nev_src, Nev_src), dtype=np.complex128)
            nev = Nev_src
            phase_list = phase_exp_proton_list
        else:
            out = np.empty((p_stop - p_start, Nev_link, Nev_link), dtype=np.complex128)
            nev = Nev_link
            phase_list = phase_exp_Q2_list

        eigvecs = cp.load(eigvec_path(conf_id, int(phys_t) % Nt), allow_pickle=True)
        eigvecs_current = eigvecs[:nev]
        offset = 0
        for p_batch_start in range(p_start, p_stop, GEN_MOM_BATCH):
            p_batch_stop = min(p_batch_start + GEN_MOM_BATCH, p_stop)
            phase = cp.asarray(phase_list[p_batch_start:p_batch_stop])
            if self.kind == "VVV":
                tmp = fun_eigen.Mom_VVV_sink_t(phase_exp=phase, eigvecs=eigvecs_current)
                batch = cp.asnumpy(tmp)
            else:
                tmp = fun_eigen.Mom_VdV_sink_t(phase_exp=phase, eigvecs=eigvecs_current)
                batch = cp.asnumpy(tmp)
                batch *= omega
            batch_size = p_batch_stop - p_batch_start
            out[offset:offset + batch_size] = batch
            offset += batch_size
            del tmp, batch, phase

        del eigvecs_current, eigvecs
        cp.get_default_memory_pool().free_all_blocks()
        return out

    def _read_time(self, phys_t, mom_slice=None):
        t = int(phys_t) % Nt
        owner = t % size
        if rank == owner:
            data = np.ascontiguousarray(self._generate_local(t, mom_slice))
        else:
            data = None
        return get_mpi_data(data=data, mdtype="Bcast", root=owner)

    def read_time(self, phys_t):
        return self._read_time(phys_t)

    def read_time_mom_batch(self, phys_t, p_start, p_stop):
        return self._read_time(phys_t, slice(p_start, p_stop))

    def close(self):
        pass


Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[x]//grid_size[x] for x in range(len(lattice_size))]
fun_eigen = vertex_creator(Nx = Nx)

Nev_src = 100
Nev_link = 400
lambdaPz, lambdaPy, lambdaPx = [0, 0, 0]
Mom_lambda = np.array([lambdaPz, lambdaPy, lambdaPx])
Mom_proton = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [-1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [-1, 0, 1],
    [1, 0, -1],
    [-1, 0, -1],
    [0, -1, 1],
    [0, 1, -1],
    [0, -1, -1],
    [-1, 1, 1],
    [1, -1, 1],
    [1, 1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1],
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 2],
    [-2, 0, 0],
    [0, -2, 0],
    [0, 0, -2],
])
n_mom = len(Mom_proton)
MOM_Q2 = Mom_lambda - Mom_proton
n_tsep = 14
G7Gx_sink_gpu = cp.asarray(gamma(7))
GxG7_source_gpu = cp.asarray(gamma(7))
p0_gpu = cp.asarray((gamma(0) + gamma(4)) / 2)

t0 = time.time()
t_rank, _, t_rank_indx = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = "Scatter")
LOW_MEMORY_MODE = env_int("LOW_MEMORY_MODE", 1)
MAX_MOM_BATCH = env_int("MAX_MOM_BATCH", 6)
MAX_2PT_MOM_BATCH = env_int("MAX_2PT_MOM_BATCH", 6)
CACHE_3PT_VVV_SINK_GPU = env_int("CACHE_3PT_VVV_SINK_GPU", 0 if LOW_MEMORY_MODE else 1)
FREE_CUPY_POOL_EACH_TSRC = env_int("FREE_CUPY_POOL_EACH_TSRC", 1 if LOW_MEMORY_MODE else 0)
FREE_CUPY_POOL_EACH_TSEP = env_int("FREE_CUPY_POOL_EACH_TSEP", 1 if LOW_MEMORY_MODE else 0)
FREE_CUPY_POOL_EACH_TSINK = env_int("FREE_CUPY_POOL_EACH_TSINK", 1 if LOW_MEMORY_MODE else 0)
FREE_CUPY_POOL_EACH_TCUT = env_int("FREE_CUPY_POOL_EACH_TCUT", 1 if LOW_MEMORY_MODE else 0)
USE_SHARED_INTERMEDIATES = env_int("USE_SHARED_INTERMEDIATES", 0 if LOW_MEMORY_MODE else 1)
GPU_VDV_CACHE_ITEMS = env_int("GPU_VDV_CACHE_ITEMS", 0 if LOW_MEMORY_MODE else 16)
GPU_3PT_VVV_CACHE_ITEMS = env_int("GPU_3PT_VVV_CACHE_ITEMS", 0 if LOW_MEMORY_MODE else 4)
GPU_PERAM_SRC_CACHE = env_int("GPU_PERAM_SRC_CACHE", 0 if LOW_MEMORY_MODE else 1)
GEN_MOM_BATCH = env_int("GEN_MOM_BATCH", 1 if LOW_MEMORY_MODE else 6)
if LOW_MEMORY_MODE:
    MAX_MOM_BATCH = min(MAX_MOM_BATCH, env_int("LOW_MEMORY_MAX_MOM_BATCH", 6))
    MAX_2PT_MOM_BATCH = min(MAX_2PT_MOM_BATCH, env_int("LOW_MEMORY_MAX_2PT_MOM_BATCH", 6))
    GEN_MOM_BATCH = min(GEN_MOM_BATCH, env_int("LOW_MEMORY_GEN_MOM_BATCH", 1))
MAX_MOM_BATCH = max(1, MAX_MOM_BATCH)
MAX_2PT_MOM_BATCH = max(1, MAX_2PT_MOM_BATCH)
GEN_MOM_BATCH = max(1, GEN_MOM_BATCH)
omega = create_omega(
    Nx = Nx,
    exact = Nev_src,
    N_eigen = [100, 200, 400, 600, 800],
    N_sum = [20, 20, 20, 20, 20],
    N_extract = [10, 10, 10, 10, 10],
    noise = 200,
    Nbin = 1,
)
phase_exp_proton_list = [fun_eigen.phase_exp_VVV(Mom = Mom_proton[i]) for i in range(n_mom)]
phase_exp_Q2_list = [fun_eigen.phase_exp_VdV(Mom = MOM_Q2[i]) for i in range(n_mom)]

VVV_reader = GeneratedTimeSliceReader("VVV/on-demand")
VdV_reader = GeneratedTimeSliceReader("VdV/on-demand")
if rank == 0:
    print(
        "conf %s: VVV/VdV will be generated on demand; only C3 will be saved" % conf_id,
        flush=True,
    )
    print("all complete of VVV/VdV on-demand setup, time used: %.3f s" % (time.time() - t0), flush=True)
    print(
        "runtime params: LOW_MEMORY_MODE=%d, MAX_MOM_BATCH=%d, MAX_2PT_MOM_BATCH=%d, "
        "CACHE_3PT_VVV_SINK_GPU=%d, GPU_VDV_CACHE_ITEMS=%d, "
        "GPU_3PT_VVV_CACHE_ITEMS=%d, GPU_PERAM_SRC_CACHE=%d, "
        "FREE_CUPY_POOL_EACH_TSRC=%d, FREE_CUPY_POOL_EACH_TSEP=%d, "
        "FREE_CUPY_POOL_EACH_TSINK=%d, FREE_CUPY_POOL_EACH_TCUT=%d, "
        "GEN_MOM_BATCH=%d"
        % (
            LOW_MEMORY_MODE,
            MAX_MOM_BATCH,
            MAX_2PT_MOM_BATCH,
            CACHE_3PT_VVV_SINK_GPU,
            GPU_VDV_CACHE_ITEMS,
            GPU_3PT_VVV_CACHE_ITEMS,
            GPU_PERAM_SRC_CACHE,
            FREE_CUPY_POOL_EACH_TSRC,
            FREE_CUPY_POOL_EACH_TSEP,
            FREE_CUPY_POOL_EACH_TSINK,
            FREE_CUPY_POOL_EACH_TCUT,
            GEN_MOM_BATCH,
        ),
        flush=True,
    )

# 费曼图收缩形式
diagrams_3pt = [
    (0,  "nodg,ikea,jlfb,pmhc", -2),
    (1,  "ioeg,nkda,jlfb,pmhc",  2),
    (2,  "nodg,ikea,jmfc,plhb",  1),
    (3,  "ioeg,nkda,jmfc,plhb", -1),
    (4,  "nodg,imec,jkfa,plhb",  1),
    (5,  "ioeg,nmdc,jkfa,plhb", -1)
]

einsum_3pt = "abc,Pdef,Pgh,ij,kl,"
einsum_suffix  = "->Pponm"

diagrams_3pt_compiled = [(n, f"{einsum_3pt}{prop_str}{einsum_suffix}", coeff) for n, prop_str, coeff in diagrams_3pt]

optimize_3pt = os.environ.get("CUPY_OE_OPT_3PT", "greedy" if LOW_MEMORY_MODE else "optimal")
if rank == 0:
    print(
        "contraction planner params: optimize_3pt=%s, USE_SHARED_INTERMEDIATES=%d"
        % (optimize_3pt, USE_SHARED_INTERMEDIATES),
        flush=True,
    )
diagrams_3pt_exprs_by_batch = {}


def get_3pt_exprs(batch_size):
    exprs = diagrams_3pt_exprs_by_batch.get(batch_size)
    if exprs is not None:
        return exprs
    shapes_3pt = [
        (Nev_src, Nev_src, Nev_src),
        (batch_size, Nev_src, Nev_src, Nev_src),
        (batch_size, Nev_link, Nev_link),
        (4, 4),
        (4, 4),
        (4, 4, Nev_src, Nev_link),
        (4, 4, Nev_src, Nev_src),
        (4, 4, Nev_src, Nev_src),
        (4, 4, Nev_link, Nev_src),
    ]
    exprs = compile_contracts(
        diagrams_3pt_compiled,
        shapes_3pt,
        optimize_3pt,
        f'3pt batch {batch_size}',
    )
    diagrams_3pt_exprs_by_batch[batch_size] = exprs
    return exprs


gpu_vdv_cache = OrderedDict()
gpu_vvv_3pt_cache = OrderedDict()


def gpu_lru_get(cache, max_items, key, loader):
    key = int(key) % Nt
    if max_items <= 0:
        return loader(key)

    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
        return value

    value = loader(key)
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_items:
        _, old_value = cache.popitem(last=False)
        del old_value
    return value


def get_vdv_gpu(phys_t):
    return gpu_lru_get(
        gpu_vdv_cache,
        GPU_VDV_CACHE_ITEMS,
        phys_t,
        lambda t: cp.asarray(VdV_reader.read_time(t)),
    )


def get_vvv_3pt_gpu(phys_t):
    return gpu_lru_get(
        gpu_vvv_3pt_cache,
        GPU_3PT_VVV_CACHE_ITEMS,
        phys_t,
        lambda t: cp.asarray(np.ascontiguousarray(VVV_reader.read_time(t))),
    )

# 计算主程序
output_root = os.environ.get('OUTPUT_ROOT', '/public/home/sush/distillation/guoj')
output_suffix = os.environ.get('OUTPUT_SUFFIX', '')
corr_dir_3pt    = f'{output_root}/C3_tsep7_20_conf{conf_id}{output_suffix}.npy'
if rank == 0:
    os.makedirs(output_root, exist_ok=True)
    corr_3pt_con = np.lib.format.open_memmap(corr_dir_3pt, mode='w+', dtype=np.complex128, shape=(n_tsep, Nt, Nt, n_mom, 4, 4, 4, 4))
    corr_3pt_con.flush()
else:
    corr_3pt_con = None
comm.Barrier()
if rank != 0:
    corr_3pt_con = np.load(corr_dir_3pt, mmap_mode='r+')

for t_src in range(Nt):
    
    t0 = time.time()
    
    VVV_tsrc_0 = VVV_reader.read_time_mom_batch(t_src, 0, 1)[0]
    VVV_tsrc_0_gpu = cp.asarray(np.ascontiguousarray(VVV_tsrc_0).conj())
    del VVV_tsrc_0

    peram_u_file = np.load(
        f'/nexdata/project/lqcd/sush/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light/{conf_id}/t{t_src:03d}_e100_s[100, 200, 400, 600, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n400_V1_stout_smear_20_0.12.npy',
        allow_pickle=True
        )
    peram_s_file = np.load(
        f'/nexdata/project/lqcd/sush/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/strange/{conf_id}/t{t_src:03d}_e100_s[100, 200, 400, 600, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n400_V1_stout_smear_20_0.12.npy',
        allow_pickle=True
        )
    peram_u_src = np.ascontiguousarray(peram_u_file[t_rank, :, :, :Nev_src, :Nev_src])
    peram_s_src = np.ascontiguousarray(peram_s_file[t_rank, :, :, :Nev_link, :Nev_src])
    del peram_u_file, peram_s_file
    use_gpu_peram_src_cache = bool(GPU_PERAM_SRC_CACHE and size == 1)
    if use_gpu_peram_src_cache:
        peram_u_src_gpu = cp.asarray(peram_u_src)
        peram_s_src_gpu = cp.asarray(peram_s_src)
    else:
        peram_u_src_gpu = None
        peram_s_src_gpu = None
    
    t1 = time.time()
    print("read peram_u_src for t_source %d, time used: %.3f s" %(t_src, t1-t0))
    
    t2 = time.time()
    if FREE_CUPY_POOL_EACH_TSRC:
        cp.get_default_memory_pool().free_all_blocks()
    
    for itsep, t_sep in enumerate(range(7, 21, 1)):
        
        t3 = time.time()
        t_cut_phys_list, _, t_cut_list_indx = get_mpi_tlist(Nt = Nt, t = range(t_src, t_src + t_sep + 1, 1), gtype = 'Scatter')
        
        peram_u_sep = np.load(
            f'/nexdata/project/lqcd/sush/perambulators/beta6.41_mu-0.2295_ms-0.2050_L32x96/light/{conf_id}/t{(t_src + t_sep)%Nt:03d}_e100_s[100, 200, 400, 600, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n400_V1_stout_smear_20_0.12.npy',
            allow_pickle=True
            )
        if use_gpu_peram_src_cache:
            peram_u_2pt = None
            peram_u_2pt_gpu = peram_u_src_gpu[(t_src + t_sep) % Nt]
        else:
            peram_u_2pt = get_mpi_data(data = peram_u_src[((t_src + t_sep)%Nt)//size], mdtype = 'Bcast', root = (t_src + t_sep)%size)
            peram_u_2pt_gpu = cp.asarray(peram_u_2pt)
        t_sink_3pt_phys = (t_src + t_sep) % Nt
        VVV_sink_3pt_all = None
        VVV_sink_3pt_all_gpu = None
        VVV_sink_3pt_batches_gpu = {}
        if GPU_3PT_VVV_CACHE_ITEMS > 0:
            VVV_sink_3pt_all_gpu = get_vvv_3pt_gpu(t_sink_3pt_phys)
        else:
            VVV_sink_3pt_all = VVV_reader.read_time(t_sink_3pt_phys)
            if CACHE_3PT_VVV_SINK_GPU:
                for p_start in range(0, n_mom, MAX_MOM_BATCH):
                    p_stop = min(p_start + MAX_MOM_BATCH, n_mom)
                    VVV_sink_3pt_batches_gpu[(p_start, p_stop)] = cp.asarray(
                        np.ascontiguousarray(VVV_sink_3pt_all[p_start:p_stop])
                    )
    
        for tcut_phys, tcut in zip(t_cut_phys_list, t_cut_list_indx):
            if GPU_VDV_CACHE_ITEMS > 0:
                VdV_current_all = None
                VdV_current_all_gpu = get_vdv_gpu(tcut_phys)
            else:
                VdV_current_all = VdV_reader.read_time(tcut_phys)
                VdV_current_all_gpu = cp.asarray(VdV_current_all)
            if use_gpu_peram_src_cache:
                peram_s_current_gpu = peram_s_src_gpu[tcut]
            else:
                peram_s_current = peram_s_src[tcut, :, :, :Nev_link, :Nev_src]
                peram_s_current_gpu = cp.asarray(peram_s_current)
            peram_l_raw_gpu = cp.asarray(np.ascontiguousarray(peram_u_sep[tcut_phys, :, :, :Nev_link, :Nev_src]))
            peram_l_conj_current_gpu = seq_peram(peram_l_raw_gpu)
            del peram_l_raw_gpu

            result_3pt_tcut_gpu = cp.zeros((n_mom, 4, 4, 4, 4), dtype=cp.complex128)
            for p_start in range(0, n_mom, MAX_MOM_BATCH):
                p_stop = min(p_start + MAX_MOM_BATCH, n_mom)
                batch_size = p_stop - p_start
                if GPU_3PT_VVV_CACHE_ITEMS > 0:
                    VVV_sink_3pt_gpu = VVV_sink_3pt_all_gpu[p_start:p_stop]
                elif CACHE_3PT_VVV_SINK_GPU:
                    VVV_sink_3pt_gpu = VVV_sink_3pt_batches_gpu[(p_start, p_stop)]
                else:
                    VVV_sink_3pt_gpu = cp.asarray(
                        np.ascontiguousarray(VVV_sink_3pt_all[p_start:p_stop])
                    )
                VdV_current_gpu = cp.ascontiguousarray(VdV_current_all_gpu[p_start:p_stop])

                tensors_3pt = [
                    VVV_tsrc_0_gpu, VVV_sink_3pt_gpu, VdV_current_gpu,
                    G7Gx_sink_gpu, GxG7_source_gpu,
                    peram_l_conj_current_gpu,
                    peram_u_2pt_gpu,
                    peram_u_2pt_gpu,
                    peram_s_current_gpu
                ]

                result_3pt = cp.zeros((p_stop - p_start, 4, 4, 4, 4), dtype=cp.complex128)
                with shared_context():
                    for _, einsum_str, expr, coeff in get_3pt_exprs(batch_size):
                       result_3pt += run_contract(einsum_str, expr, tensors_3pt) * coeff
                result_3pt_tcut_gpu[p_start:p_stop] = result_3pt
                del VdV_current_gpu, tensors_3pt, result_3pt
                if GPU_3PT_VVV_CACHE_ITEMS <= 0 and not CACHE_3PT_VVV_SINK_GPU:
                    del VVV_sink_3pt_gpu
            corr_3pt_con[itsep, t_src, tcut_phys] += cp.asnumpy(result_3pt_tcut_gpu)
            if GPU_VDV_CACHE_ITEMS <= 0:
                del VdV_current_all, VdV_current_all_gpu
            else:
                del VdV_current_all_gpu
            del peram_s_current_gpu, peram_l_conj_current_gpu, result_3pt_tcut_gpu
            if FREE_CUPY_POOL_EACH_TCUT:
                cp.get_default_memory_pool().free_all_blocks()
        del peram_u_sep, peram_u_2pt_gpu
        if peram_u_2pt is not None:
            del peram_u_2pt
        if VVV_sink_3pt_all is not None:
            del VVV_sink_3pt_all
        if GPU_3PT_VVV_CACHE_ITEMS > 0:
            del VVV_sink_3pt_all_gpu
        del VVV_sink_3pt_batches_gpu
        if FREE_CUPY_POOL_EACH_TSEP:
            cp.get_default_memory_pool().free_all_blocks()
        t4 = time.time()
        print("3pt for t_source %d tsep %d, time used: %.3f s" %(t_src, t_sep,t4-t3)) 
    t5 = time.time()        
    print("3pt for t_source %d, time used: %.3f s" %(t_src, t5-t2))
    if use_gpu_peram_src_cache:
        del peram_u_src_gpu, peram_s_src_gpu
    del peram_u_src, peram_s_src, VVV_tsrc_0_gpu
    if FREE_CUPY_POOL_EACH_TSRC:
        cp.get_default_memory_pool().free_all_blocks()
    
    
if corr_3pt_con is not None:
    corr_3pt_con.flush()
VVV_reader.close()
VdV_reader.close()
comm.Barrier()
    
        
