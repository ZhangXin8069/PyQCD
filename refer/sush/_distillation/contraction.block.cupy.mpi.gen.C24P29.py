import glob
import os
import sys
from time import perf_counter

total_start = perf_counter()

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/public/home/sush/EasyDistillation/")

from lattice import set_backend, get_backend, check_QUDA
grid_size = [1, 1, 1, 1]

# check and init PyQuda & MPI
if not check_QUDA(grid_size=grid_size, backend="cupy", resource_path=os.environ.get("C24P29_QUDA_RESOURCE_PATH", '/public/home/sush/EasyDistillation/work/resource/beta6.20_mu-0.2770_ms-0.2400_L24x72/V100')):
    raise ImportError("Please install PyQuda")

from lattice import GaugeFieldIldg, EigenvectorNpy, Nc, Nd

Nx = int(sys.argv[1])
Ny = int(sys.argv[2])
Nz = int(sys.argv[3])
Nt = int(sys.argv[4])

# Half-open global source-time interval; the default preserves the first-half run.
# tsrc_start = int(os.environ.get("C24P29_TSRC_START", "0"))
# tsrc_stop = int(os.environ.get("C24P29_TSRC_STOP", str(Nt // 2)))

tsrc_start = 0
tsrc_stop = Nt

if not 0 <= tsrc_start < tsrc_stop <= Nt:
    raise ValueError(f"Invalid source-time interval [{tsrc_start}, {tsrc_stop}) for Nt={Nt}")
print(f"Source times: [{tsrc_start}, {tsrc_stop}), sink times: [0, {Nt})", flush=True)

latt_size = [Nx, Ny, Nz, Nt]
Gx, Gy, Gz, Gt = grid_size

Ne = int(sys.argv[5])
Ns = 4
conf_id = sys.argv[6]
file_path = F"{sys.argv[7]}"
eigen_path = F"{sys.argv[8]}"
out_prefix = sys.argv[9]
source_shape = int(sys.argv[10])
from typing import List, Literal

from opt_einsum import contract

from lattice.constant import Nc, Ns, Nd
from lattice.backend import set_backend, get_backend, check_QUDA
from lattice.preset import GaugeField, Eigenvector

import numpy as np

class PerambulatorGenerator:
    """
     Generate perambulators in distillation,
        based on PyQUDA + QUDA for GPU-accelerated computations.

    Parameters:
    -----------
    latt_size : List[int]
        dimensions of the lattice, order as [Lx, Ly, Lz, Lt].
    gauge_field : GaugeField.
    eigenvector : Eigenvector.
    mass : float
        The mass parameter for the Dirac operator.
    tol : float
        The Dirac operator tol.
    maxiter : int
        The maximum number of iterations of solver.
    xi_0 : float, optional
        The anisotropy, defaults to 1.0.
    nu : float, optional.
    clover_coeff_t : float, optional
        The temporal clover coefficient, defaults to 0.0.
    clover_coeff_r : float, optional
        The spatial clover coefficient, defaults to 1.0.
    t_boundary : Literal[1, -1], optional
        The temporal boundary condition, defaults to 1 (periodic).
    multigrid : List[List[int]], optional
        The multigrid levels for the solver, defaults to None.
    contract_prec : str, optional
        The precision for the contraction operations, defaults to '<c16'.
    usedNe : int, optianal
        The used eigenvectors number, defaults to None, usedNe is eigenvector.Ne .
    MRHS : bool, optional:
        Use MRHS methods to solve perambulators, defaults to False.
        This option requires more device memory.

    Notes:
    ------
    - This class requires PyQUDA + QUDA for GPU-accelerated computations.
    """

    def __init__(
        self,
        latt_size: List[int],
        gauge_field: GaugeField,
        eigenvector: Eigenvector,
        mass: float,
        tol: float,
        maxiter: int,
        Ne: int = 0,
        xi_0: float = 1.0,
        nu: float = 1.0,
        clover_coeff_t: float = 0.0,
        clover_coeff_r: float = 1.0,
        t_boundary: Literal[1, -1] = 1,
        multigrid: List[List[int]] = None,
        contract_prec: str = "<c16",
        MRHS: bool = False,
        source_range: range = range(0, 1, 1),

    ) -> None:
        if not check_QUDA():
            raise ImportError("Please install PyQuda to generate the perambulator or check MPI_init again.")
        from pyquda_utils import core

        self.latt_info = core.LatticeInfo(latt_size=latt_size, t_boundary=t_boundary, anisotropy=xi_0 / nu)
        self.contract_prec = contract_prec
        self.Nt = latt_size[3]
        backend = get_backend()
        assert backend.__name__ == "cupy", "PyQuda only support cupy as the ndarray implementation"
        Lx, Ly, Lz, Lt = self.latt_info.size

        if Ne == 0:
            Ne =  eigenvector[0].Ne
        
        self.usedNe = Ne
        
        self.source_range = source_range
        self.gauge_field = gauge_field
        self.gauge_field_smear = None
        self.gauge_field_new = None
        self.eigenvector = eigenvector
        self.MRHS = MRHS
        self.dirac = core.getDirac(
            self.latt_info,
            mass,
            tol,
            maxiter,
            xi_0,
            clover_coeff_t,
            clover_coeff_r,
            multigrid,
        )
        import numpy
        self._SV = backend.zeros((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc), self.contract_prec)
        # self._VSV = numpy.zeros((Lt, Ns, Ns, Ne, len(source_range)), self.contract_prec)

    def _stout_smear_quda(self, nstep: int, rho: float, dir_ignore: int):
        gauge = self.gauge_field_smear
        if self.gauge_field_smear is None:
            raise ValueError("Gauge not loaded, please use .load() before .stout_smear().")

        gauge.smearSTOUT(nstep, rho, dir_ignore)

        self.gauge_field_smear = gauge

    def stout_smear(self, nstep: int, rho: float, dir_ignore: int = 3):
        backend = get_backend()
        if backend.__name__ == "numpy":
            raise NotImplementedError("Ndarray stout smear not implement in PerambulatorGenerator.")
        elif backend.__name__ == "cupy":
            # __init__() has check_QUDA() before !
            self._stout_smear_quda(nstep, rho, dir_ignore)

    def calc(self, t: int, splitNe:bool = False, source_range=None):
        import cupy as cp

        backend = get_backend()
        from pyquda_utils.core import LatticeFermion, MultiLatticeFermion

        if self.gauge_field_new:
            self.dirac.loadGauge(self.gauge_field_smear)  # loadGauge after
            self.gauge_field_new = False

        latt_info = self.latt_info
        Lx, Ly, Lz, Lt = latt_info.size
        Vol = Lx * Ly * Lz * Lt
        Ne = self.usedNe
        
        eigenvector_dagger = self._eigenvector_data_dagger
        # eigenvector_B_dagger = self._eigenvector_data_B_dagger

        dirac = self.dirac
        gx, gy, gz, gt = self.latt_info.grid_coord
        if source_range is None:
            source_range = self.source_range
        # Keep propagators on the GPU in QUDA's (eo,t,z,y,x/2,spin,spin,color,source) layout.
        gen_SV = backend.empty((2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, len(source_range)), self.contract_prec)
        
        SV = self._SV
        # VSV = self._VSV.copy()
        
        from time import perf_counter

        for eigen_indx, eigen in enumerate(source_range):
            cp.cuda.runtime.deviceSynchronize()
            s = perf_counter()
            if self.MRHS:
                print("Warning: use MRHS.")
                V_MRHS = MultiLatticeFermion(latt_info, Ns)
                for spin in range(Ns):
                    data = V_MRHS[spin].data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
                    if gt * Lt <= t and (gt + 1) * Lt > t:
                        data[:, t % Lt, :, :, :, spin, :] = backend.asarray(
                            eigenvector_dagger[eigen, :, t % Lt, :, :, :, :].conj()
                        )
                SV_MRHS = dirac.invertMultiSrc(V_MRHS)
                for spin in range(Ns):
                    SV.reshape(Vol, Ns, Ns, Nc)[:, :, spin, :] = SV_MRHS[spin].data.reshape(Vol, Ns, Nc)
            else:
                for spin in range(Ns):
                    V = LatticeFermion(latt_info)  # V.data is double prec.
                    data = V.data.reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)
                    if gt * Lt <= t and (gt + 1) * Lt > t:
                        data[:, t % Lt, :, :, :, spin, :] = backend.asarray(
                            eigenvector_dagger[eigen, :, t % Lt, :, :, :, :].conj()
                        )  # [Ne, etzyx, Nc]
                    SV.reshape(Vol, Ns, Ns, Nc)[:, :, spin, :] = dirac.invert(V).data.reshape(Vol, Ns, Nc)  # .get()
                    
            cp.cuda.runtime.deviceSynchronize()
            invert_time = perf_counter() - s
            
            cp.cuda.runtime.deviceSynchronize()
            s = perf_counter()

            # for t_sink in range(2):
            # if splitNe:
            #     for i in range(10):
            #         VSV[:, :, :, (Ne//10)*i:(Ne//10)*(i+1), eigen_indx] = contract(
            #             "ketzyxa,etzyxija->tijk", 
            #             backend.asarray(eigenvector_dagger[(Ne//10)*i:(Ne//10)*(i+1)]), 
            #             backend.asarray(SV), 
            #             optimize=True
            #         ).get()
                
            # else:
            #     VSV[:, :, :, :, eigen_indx] = contract(
            #         "ketzyxa,etzyxija->tijk", 
            #         backend.asarray(eigenvector_dagger), 
            #         backend.asarray(SV), 
            #         optimize=True
            #     ).get()
            #     # VSV_B[t_sink * half_T:(t_sink + 1) * half_T, :, :, :, eigen] = contract(
            #     #     "ketzyxa,etzyxija->tijk", 
            #     #     backend.asarray(eigenvector_B_dagger[:, :, t_sink * half_T:(t_sink + 1) * half_T]), 
            #     #     backend.asarray(SV[:, t_sink * half_T:(t_sink + 1) * half_T]), 
            #     #     optimize=True
            #     # ).get()
            
            gen_SV[..., eigen_indx] = SV
            
            cp.cuda.runtime.deviceSynchronize()
            contraction_time = perf_counter() - s

            # print for check device mem
            free, total = cp.cuda.runtime.memGetInfo()
            print(
                f"Ne = {eigen}:  inv t = {invert_time:.4f} sec, EO store t = {contraction_time:.4f} sec, device mem: {(total - free) / 1024**3} GB, free:{free / 1024**3} GB."
            )
        # return VSV
        
        return gen_SV


def load_compress(
        self, 
        key: str, 
        dtype: str = 'timeslice', 
        ttype: str = '', 
        bin: int = -1, 
        used_vector:list = [0],
        eigen_save_path:list = [''],
        eigen_name:list = ['']
        ):
    import numpy as np
    from pyquda_utils import core
    from pyquda_utils import io

    backend = get_backend()
    Lx, Ly, Lz, Lt = self.latt_info.size
    gx, gy, gz, gt = self.latt_info.grid_coord
    
    Ne = self.usedNe
    
    self.gauge_field_smear = io.readQIOGauge(self.gauge_field.load(key).file)
    self.gauge_field_new = True

    if dtype == 'timeslice':
        eigenvector_data = np.zeros((Lt, sum(used_vector), Lz, Ly, Lx, Nc), dtype=np.complex128)
        # eigenvector_data_B = np.zeros((self.Nt, Ne, Lz, Ly, Lx, Nc), dtype=np.complex128)

        for i in range(len(eigen_save_path)):
            for t_indx, t in enumerate([x for x in range(Lt * gt, (gt+1) * Lt, 1)]):
                if ttype == ':03d':
                    A = np.load(f'{eigen_save_path[i]}{t:03d}{eigen_name[i]}')

                else:
                    A = np.load(f'{eigen_save_path[i]}{t}{eigen_name[i]}')

                if bin == 1:
                    eigenvector_data[t_indx] = A[:sum(used_vector), gz * Lz : (gz + 1) * Lz, gy * Ly : (gy + 1) * Ly, gx * Lx : (gx + 1) * Lx, :]
                
        del A

        
    elif dtype == 'allslice':
        eigenvector_data = self.eigenvector.load(key)

    self._eigenvector_data_dagger = backend.asarray(core.evenodd(eigenvector_data.transpose(1, 0, 2, 3, 4, 5).conj(), [1, 2, 3, 4]))
    
from math import prod


def lexico(data, axes: List[int], dtype=None):
    assert len(axes) == 5
    shape = data.shape
    Np, Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    assert Np == 2
    Lx *= 2
    Npre = int(np.prod(shape[: axes[0]]))
    Nsuf = int(np.prod(shape[axes[-1] + 1 :]))
    dtype = data.dtype if dtype is None else dtype
    backend = get_backend()

    # Reshape to separate even/odd and half x-dimension
    data_evenodd = data.reshape(Npre, 2, Lt, Lz, Ly, Lx // 2, Nsuf)

    # Create parity array for (t, z, y) using broadcasting
    t = backend.arange(Lt)
    z = backend.arange(Lz)
    y = backend.arange(Ly)
    eo = (t[:, None, None] + z[None, :, None] + y[None, None, :]) % 2  # (Lt, Lz, Ly)
    eo_expanded = eo.reshape(Lt, Lz, Ly, 1, 1)  # (Lt, Lz, Ly, 1, 1)

    # Split even and odd halves
    even_half = data_evenodd[:, 0, ...]  # (Npre, Lt, Lz, Ly, Lx//2, Nsuf)
    odd_half = data_evenodd[:, 1, ...]   # (Npre, Lt, Lz, Ly, Lx//2, Nsuf)

    # Allocate output array
    data_lexico = backend.zeros((Npre, Lt, Lz, Ly, Lx, Nsuf), dtype=dtype)

    # Vectorized assignment based on parity – corrected slicing on Lx dimension
    data_lexico[..., 0::2, :] = backend.where(eo_expanded == 0, even_half, odd_half)
    data_lexico[..., 1::2, :] = backend.where(eo_expanded == 0, odd_half, even_half)

    # Restore original shape with modified axes
    return data_lexico.reshape(*shape[: axes[0]], Lt, Lz, Ly, Lx, *shape[axes[-1] + 1 :])

def creat_mom_list(Mom:list = [0, 0, 0], fix_Q2:bool = False, only_g0:bool = False):
    """Generate a sorted list of unique lattice momenta.

    ``Mom`` is either one three-component reference momentum or a list of
    reference momenta.  For each reference, the routine enumerates the integer
    cube between its minimum and maximum component and applies all sign
    combinations.  ``fix_Q2=True`` retains only momenta with the reference
    squared norm, and ``only_g0=True`` retains only non-negative components.

    Duplicate momenta generated by overlapping sign/range enumerations are
    removed before returning.
    """
    import itertools
    def add_negative_signs(lst):
    # 找到所有非零元素的索引
        nonzero_indices = [i for i, val in enumerate(lst) if val != 0]

        result = []
        # 为每个非零元素生成所有可能的正负号组合
        for signs in itertools.product([1, -1], repeat=len(nonzero_indices)):
            new_list = lst.copy()
            for idx, sign in zip(nonzero_indices, signs):
                new_list[idx] = lst[idx] * sign
            result.append(new_list)
    
        return result

    if 'array' in str(type(Mom)):
        Mom = Mom.tolist()

    if type(Mom[0]) == list:
        _num = len(Mom)

    else:
        _num = 1
        Mom = [Mom]
        
    Mom_list_all = []

    for i in range(_num):
        _Mom = Mom[i]

        min_Mom = min(_Mom)
        max_Mom = max(_Mom)

        len_Mom = max_Mom - min_Mom + 1

        Q2 = sum([x**2 for x in _Mom])

        
        for i in range((len_Mom) ** 3):
            Mom_list = [(i//((len_Mom)**2))%len_Mom + min_Mom, (i//((len_Mom)**1))%len_Mom + min_Mom, (i//((len_Mom)**0))%len_Mom + min_Mom]
            Mom_list = add_negative_signs(Mom_list)

            if fix_Q2:
                for Mom_list_indx, Mom_list_one in enumerate(Mom_list):
                    indx = []

                    if Q2 == sum([x**2 for x in Mom_list_one]):
                        indx += [Mom_list_indx]

                    Mom_list_all += [Mom_list[x] for x in indx]

            else:
                Mom_list_all += Mom_list
                
    # Sign expansion and overlapping reference ranges can generate the same
    # momentum repeatedly.  A momentum shell is a set of lattice momenta, so
    # deduplicate before returning while preserving deterministic sort order.
    unique_momenta = sorted({tuple(momentum) for momentum in Mom_list_all})
    if only_g0:
        unique_momenta = [
            momentum for momentum in unique_momenta
            if all(component >= 0 for component in momentum)
        ]

    return [list(momentum) for momentum in unique_momenta]

def phase_exp_VdV(Nx, Mom: list = [0, 0, 0]):
    backend = get_backend()
    
    if all(x == 0 for x in Mom):
        return backend.ones((Nx, Nx, Nx, Nc), dtype=complex)
    
    # 更高效的方法：使用单独的一维数组和广播
    mom_array = backend.asarray(Mom, dtype=complex)
    factor = -2j * backend.pi / Nx
    
    # 创建一维坐标数组
    z = backend.arange(Nx, dtype=complex)
    y = backend.arange(Nx, dtype=complex)
    x = backend.arange(Nx, dtype=complex)
    
    # 分别计算三个方向的贡献
    # 使用reshape和广播避免meshgrid的内存开销
    z_phase = backend.exp(factor * mom_array[0] * z[:, None, None])
    y_phase = backend.exp(factor * mom_array[1] * y[None, :, None])
    x_phase = backend.exp(factor * mom_array[2] * x[None, None, :])
    
    # 组合相位因子（乘法对应相位相加）
    phase_3d = z_phase * y_phase * x_phase
    
    # 扩展到颜色维度

    phase_exp = backend.stack([phase_3d, phase_3d, phase_3d], axis=-1)
    

    return phase_exp

gauge_field = GaugeFieldIldg(f"{file_path}", ".lime", [Nt, Nz, Ny, Nx, Nd, Nc, Nc])

Lx, Ly, Lz, Lt = [latt_size[0] // Gx, latt_size[1] // Gy, latt_size[2] // Gz, latt_size[3] // Gt]

from pyquda import enum_quda
from pyquda_utils import core
from pyquda_utils.core import getMPIRank, gatherLattice

PerambulatorGenerator.load_compress = load_compress

set_backend("cupy")
backend = get_backend()
 
exact_eigen = 50
N_eigen = [100, 200, 300, 500, 800]
N_sum = [20, 20, 20, 20, 20]
N_extract = [10, 10, 10, 10, 10]
noise_v = 500

N_bin = 1

Ne = (exact_eigen + sum(N_sum) + noise_v)
Ne = 650
#gamma1
g1=backend.zeros((4,4),dtype=complex)
g1[0,3]=0.0+1.0*1j
g1[1,2]=0.0+1.0*1j
g1[2,1]=0.0-1.0*1j
g1[3,0]=0.0-1.0*1j

#gamma2
g2=backend.zeros((4,4),dtype=complex)
g2[0,3]=-1.0+0.0*1j
g2[1,2]=1.0+0.0*1j
g2[2,1]=1.0+0.0*1j
g2[3,0]=-1.0+0.0*1j

#gamma3
g3=backend.zeros((4,4),dtype=complex)
g3[0,2]=0.0+1.0*1j
g3[1,3]=0.0-1.0*1j
g3[2,0]=0.0-1.0*1j
g3[3,1]=0.0+1.0*1j

#gamma4
g4=backend.zeros((4,4),dtype=complex)
g4[0,2]=1.0+0.0*1j
g4[1,3]=1.0+0.0*1j
g4[2,0]=1.0+0.0*1j
g4[3,1]=1.0+0.0*1j

#gamma5
g5=backend.zeros((4,4),dtype=complex)
g5[0,0]=1.0+0.0*1j
g5[1,1]=1.0+0.0*1j
g5[2,2]=-1.0+0.0*1j
g5[3,3]=-1.0+0.0*1j

gamma_mu = backend.asarray([g1, g2, g3, g4])

phase_mom = creat_mom_list([1, 1, 1], fix_Q2 = True) + creat_mom_list([0, 0, 2], fix_Q2 = True)

phase_link = backend.zeros((len(phase_mom), Nx, Nx, Nx, Nc), dtype = complex)

for mom_indx, mom in enumerate(phase_mom):
    phase_link[mom_indx] = phase_exp_VdV(Nx = Nx, Mom = mom)

phase_link_lexico = phase_link.reshape(len(phase_mom), -1, Nc)
# QUDA parity is (t+z+y+x)%2. Two time slices provide both temporal parities;
# flatten (eo,z,y,x/2) in exactly the same order as gen_peramb[:, tsink].
phase_link = backend.asarray(core.evenodd(
    np.broadcast_to(backend.asnumpy(phase_link)[:, None],
                    (len(phase_mom), 2, Nz, Ny, Nx, Nc)),
    [1, 2, 3, 4],
)).transpose(2, 0, 1, 3, 4, 5, 6).reshape(2, len(phase_mom), -1, Nc)

source_block_size = 8
validate_evenodd = os.environ.get("C24P29_VALIDATE_EO", "0") == "1"
validation_count = 0
validation_max_abs = 0.0
validation_max_rel_l2 = 0.0

from opt_einsum import contract


def drop_source_time_parts(corr_base, full_path, expected):
    """Delete the per-interval ``_tsrc###-###.npy`` files of one result.

    Those files only carry a source-time interval between runs (the first half
    is the input of the merging run). Once ``{corr_base}.npy`` contains the
    complete source-time range and reproduces the parts exactly, the parts are
    removed, so that only the final file is left on disk.
    """
    stored = np.load(full_path, allow_pickle=False)
    if (stored.shape != expected.shape or stored.dtype != np.dtype("<c16") or
            not np.array_equal(stored, expected)):
        raise ValueError(f"{full_path} does not reproduce the source-time parts; keeping them")
    removed = []
    for part_path in sorted(glob.glob(f'{corr_base}_tsrc*.npy')):
        if os.path.abspath(part_path) == os.path.abspath(full_path):
            continue
        os.remove(part_path)
        removed.append(os.path.basename(part_path))
    print(f"Removed source-time part file(s) of {os.path.basename(corr_base)}: "
          f"{', '.join(removed)}", flush=True)


for N_stout in [20]:
    name = [
        f'e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_v}_V1_stout_smear_{N_stout}_0.12',
        ]

    eigenvector = EigenvectorNpy(f"{eigen_path}", f"_{name[0]}.npy", [Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nc], Ne)

    perambulator = PerambulatorGenerator(
        latt_size=latt_size,
        gauge_field=gauge_field,
        eigenvector = [eigenvector],
        mass=-0.2770,
        tol=1e-9,
        maxiter=10000,
        xi_0=1.0,
        nu=1.0,
        clover_coeff_t=1.160920226,
        clover_coeff_r=1.160920226,
        Ne = 650,
        t_boundary=-1,  # for this test lattice, use t_boundary=-1
        # multigrid=False, # set List[int] to enable multigrid. multigrid layer set.
        # multigrid=[[3, 3, 3, 2], [2, 2, 2, 2], [2, 2, 2, 9]],
        multigrid=[[6, 6, 6, 2], [2, 2, 2, 9]],
        contract_prec="<c16",
        # MRHS=False, # set True to enable quda mrhs support.
        MRHS=True, # set True to enable quda mrhs support.
        source_range = range(0, 50, 1),
    )  # arbitrary dirac parameters
    
    # perambulator.dirac.setVerbosity(enum_quda.QudaVerbosity.QUDA_SUMMARIZE)
    perambulator.dirac.invert_param.verbosity = enum_quda.QudaVerbosity.QUDA_SUMMARIZE
    perambulator.dirac.multigrid.param.setup_use_mma = [0]*5
    perambulator.dirac.multigrid.param.dslash_use_mma = [0]*5 
    
    # The source index n is summed independently in the original contraction;
    # adding source-block results preserves it without storing all 50 sources.
    corr_2pt = backend.zeros((tsrc_stop - tsrc_start, Nt, len(phase_mom), len(gamma_mu)), dtype="<c16")
    
    for cfg in [conf_id]:
        print(cfg)

        perambulator.load_compress(
            key = cfg, 
            dtype = 'timeslice', 
            ttype = ':03d', 
            bin = 1, 
            # used_vector = [50, 100, 500],
            used_vector = [50],
            eigen_save_path=[f'{eigen_path}/{cfg}_t'],
            eigen_name = [f'_{name[0]}.npy']
            )
        
        print('load complete')
        perambulator.stout_smear(1, 0.125, 4)
        
        for t in range(tsrc_start, tsrc_stop, 1):
            source_start = perf_counter()
            invert_store_time = 0.0
            contraction_time = 0.0
            validation_time = 0.0
            for block_start in range(0, len(perambulator.source_range), source_block_size):
                source_block = perambulator.source_range[block_start:block_start + source_block_size]
                stage_start = perf_counter()
                gen_peramb = perambulator.calc(t, splitNe=False, source_range=source_block)
                invert_store_time += perf_counter() - stage_start
                for tsink in range(Nt):
                    stage_start = perf_counter()
                    propa = gen_peramb[:, tsink].reshape(-1, Ns, Ns, Nc, len(source_block))
                    prop_sep_tsink = contract('ab,...bcef,cd->...daef', g5, propa.conj(), g5)
                    block_corr = contract('Vijcn,Vklcn,MVc,Gli,jk->MG', propa, prop_sep_tsink, phase_link[tsink % 2], gamma_mu, g5, optimize='greedy')
                    corr_2pt[t - tsrc_start, tsink] += block_corr
                    backend.cuda.runtime.deviceSynchronize()
                    contraction_time += perf_counter() - stage_start

                    # # Optional real-propagator comparison against the original lexico/get path.
                    # # Check every sink, momentum and gamma at the first/last source times,
                    # # including both a full source block and the final partial source block.
                    # if (validate_evenodd and t in (0, Nt // 2 - 1) and
                    #         block_start in (0, ((len(perambulator.source_range) - 1) // source_block_size) * source_block_size)):
                    #     stage_start = perf_counter()
                    #     reference_eo = gen_peramb[:, tsink:tsink + 1]
                    #     # lexico sees local time zero for this slice: restore its true time parity.
                    #     if tsink % 2:
                    #         reference_eo = reference_eo[::-1].copy()
                    #     reference_propa = backend.asarray(lexico(reference_eo, [0, 1, 2, 3, 4]).get())
                    #     reference_propa = reference_propa.reshape(-1, Ns, Ns, Nc, len(source_block))
                    #     reference_sep = contract('ab,...bcef,cd->...daef', g5, reference_propa.conj(), g5)
                    #     reference_corr = contract('Vijcn,Vklcn,MVc,Gli,jk->MG', reference_propa, reference_sep,
                    #                               phase_link_lexico, gamma_mu, g5, optimize='greedy').get()
                    #     actual_corr = block_corr.get()
                    #     np.testing.assert_allclose(actual_corr, reference_corr, rtol=1e-11, atol=1e-12)
                    #     error = actual_corr - reference_corr
                    #     validation_count += 1
                    #     validation_max_abs = max(validation_max_abs, float(np.max(np.abs(error))))
                    #     validation_max_rel_l2 = max(validation_max_rel_l2,
                    #         float(np.linalg.norm(error) / max(np.linalg.norm(reference_corr), np.finfo(float).tiny)))
                    #     validation_time += perf_counter() - stage_start
                    #     del reference_eo, reference_propa, reference_sep, reference_corr, actual_corr
                    
                    del propa, prop_sep_tsink, block_corr

                del gen_peramb

            print(f"TIMING tsrc={t} invert_store={invert_store_time:.6f} contraction={contraction_time:.6f} "
                  f"validation={validation_time:.6f} total={perf_counter() - source_start:.6f} sec", flush=True)

        for Mom_indx, Mom_list in enumerate(phase_mom[:]):
            corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{Nx}x{Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_650/conf{conf_id}'
            import pathlib

            path = pathlib.Path(corr_save_path)

            if path.exists():
                print('save_path:',corr_save_path)
            
            else:
                path.mkdir(parents = True, exist_ok = True)
                print('mkdir_save_path:',corr_save_path)
            
            # The 100-vector sink is exact and needs no fourth-order compensation.
            for mu_indx_1, mu in enumerate([1, 2, 3, 4]):
                corr_base = f'{corr_save_path}/corr_2pt_gamma0{mu}_gen_stout_smear_20_0.12_src50'
                part_path = f'{corr_base}_tsrc{tsrc_start:03d}-{tsrc_stop - 1:03d}.npy'
                backend.save(part_path, corr_2pt[..., Mom_indx, mu_indx_1])
                full_path = f'{corr_base}.npy'
                if tsrc_start == Nt // 2 and tsrc_stop == Nt:
                    first_path = f'{corr_base}_tsrc000-{Nt // 2 - 1:03d}.npy'
                    first_half = np.load(first_path, allow_pickle=False)
                    second_half = np.load(part_path, allow_pickle=False)
                    if (first_half.shape != (Nt // 2, Nt) or
                            second_half.shape != (Nt - Nt // 2, Nt) or
                            first_half.dtype != np.dtype("<c16") or second_half.dtype != first_half.dtype):
                        raise ValueError(f"Incompatible source-time parts: {first_path}, {part_path}")
                    full_corr = np.concatenate((first_half, second_half), axis=0)
                    np.save(full_path, full_corr)
                elif tsrc_start == 0 and tsrc_stop == Nt:
                    backend.save(full_path, corr_2pt[..., Mom_indx, mu_indx_1])
                    full_corr = np.load(part_path, allow_pickle=False)
                else:
                    # A partial interval only: the part file is the input of the
                    # run that computes the remaining source times, so it stays.
                    continue
                drop_source_time_parts(corr_base, full_path, full_corr)

        
        print(f"Saved {corr_save_path}, axes=(tsrc,tsink), shape={corr_2pt.shape}", flush=True)
        if tsrc_start == Nt // 2 and tsrc_stop == Nt:
            print(f"Merged both source-time halves: axes=(tsrc,tsink), shape=({Nt},{Nt})", flush=True)
        
        if validate_evenodd:
            print(f"VALIDATION PASS comparisons={validation_count} max_abs_error={validation_max_abs:.16e} "
                  f"max_rel_l2_error={validation_max_rel_l2:.16e} rtol=1e-11 atol=1e-12", flush=True)

    perambulator.dirac.destroy()

print(f"TIMING total={perf_counter() - total_start:.6f} sec", flush=True)
