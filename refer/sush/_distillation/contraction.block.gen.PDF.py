import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/public/home/sush/distillation/')

from function_contraction import *

from opt_einsum import contract, contract_path

import time

from math import prod
from typing import List
import numpy as np

set_backend('numpy')
backend = get_backend()

lattice_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 12]

param = creat_param(
    sink_operators = [['d^d', 'gamma_sink', 'u', 1]],
    source_operators = [['u^d', 'gamma_source', 'd', -1]]
)
import pyquda 

param_data = param.data()
# param_data['type1_uu'] = []

def Readin_gauge(conf_file, lattice_size):
    
    Nz, Ny, Nx, Nt = lattice_size
    
    f = open("%s" % conf_file, "rb")
    gauge = np.fromfile(f, dtype=">f8")
    gauge = np.array(gauge)

    gauge = gauge.reshape(Nt, Nx, Nx, Nx, 4, 3, 3, 2)
    gauge = gauge[..., 0] + gauge[..., 1] * 1j
    f.close()

    return gauge

from typing import Literal

def gauge_link_transpose(gauge, link_max, lattice_size, axis: Literal[0, 1, 2] = 0):
    Nz, Ny, Nx, Nt = lattice_size
    from math import prod
    gauge_shape = list(gauge.shape)    
    V = prod(gauge_shape[:3])
    
    if axis >= 0 and axis <= 2:
        gauge_link_rolled = backend.empty([2 * link_max + 1] + gauge_shape, dtype = complex)
                
        for link_indx in range(-link_max, link_max+1, 1):
            
            _gauge_link_rolled = backend.zeros((V, 3, 3), dtype=backend.complex128)
            _gauge_link_rolled[:] = backend.identity(3,dtype=backend.complex128)
            
            if link_indx < 0:
                for link_indx_2 in range(abs(link_indx)):
                    _gauge_link_rolled = _gauge_link_rolled @ backend.roll(gauge, abs(link_indx) - link_indx_2, axis = axis).reshape(V, 3, 3)

                _gauge_link_rolled = _gauge_link_rolled.transpose(0, 2, 1).conj()
                
            if link_indx > 0:
                for link_indx_2 in range(abs(link_indx)):
                    _gauge_link_rolled = _gauge_link_rolled @ backend.roll(gauge, -1 * link_indx_2, axis = axis).reshape(V, 3, 3)
                    
            gauge_link_rolled[link_indx] = _gauge_link_rolled.reshape(gauge_shape)

        return gauge_link_rolled

    elif axis == 3:
        return None
    
def prop_link_transpose(prop, link_max, lattice_size, axis: Literal[0, 1, 2] = 0):
    Nz, Ny, Nx, Nt = lattice_size
    
    _prop_link_shape = [2 * link_max + 1] + list(prop.shape)
    
    if axis >= 0 and axis <= 2:
        prop_link_rolled = backend.empty([2 * link_max + 1] + list(prop.shape), dtype = complex)
                
        for link_indx in range(-link_max, link_max+1, 1):
            prop_link_rolled[link_indx] = backend.roll(prop, -1 * link_indx, axis = axis)

        return prop_link_rolled.reshape(_prop_link_shape)

    elif axis == 3:
        return None
    
conf_id = sys.argv[1]
exact_eigen = 50
N_eigen = [100, 200, 300, 500, 800]
N_sum = [20, 20, 20, 20, 20]
N_extract = [10, 10, 10, 10, 10]
noise_eigen = 500

exact_eigen_name = 50
N_eigen_name = [100, 200, 300, 500, 800]
N_sum_name = [20, 20, 20, 20, 20]
N_extract_name = [10, 10, 10, 10, 10]
noise_eigen_name = 500

Nev_max = 650
t_sep = 12
link_max = 10

corr = distillation_func(
    conf_id = conf_id,
    param = param_data,
    
    lattice_size = lattice_size, # Nx, Ny, Nz, Nt 
    grid_size = grid_size, 
    
    backend = backend.__name__
)

comm = getMPIComm()
rank = getMPIRank()
size = getMPISize()

random.sample()

if rank == 0:
    random_list = []
    A = len(random_list)
    random_list = np.asarray(random_list)
    
else:
    random_list = None
    
random_list_rank = get_mpi_data(random_list, mdtype = 'Scatter', root = 0)

Mom_link = [[0, 0, 0]]

import time

g7 = backend.asarray(gamma(7))

omega1 = corr.eigen.create_omega_accelerate(
    exact = exact_eigen,
    N_eigen = N_eigen,
    N_sum = N_sum,
    N_extract = N_extract,
    noise = noise_eigen,
    dim = 1
)

# peram_sep = backend.empty((corr.Nt, Ns, Ns, Nev_max, Nev_max), dtype = complex)

# bubble_light_matrix = backend.zeros((len(Mom_link), 2, corr.Nt), dtype = complex)
import random 

# disconnect Z
bubble_strange_matrix = backend.zeros((2 * link_max + 1, corr.Nt), dtype = complex)
for t_src in range(corr.Nt):

    if rank == 0:
        # prop_bubble_light = backend.empty((1, corr.Nx, corr.Nx, corr.Nx, Ns, Ns, Nc, Nev_max), dtype = complex)
        vectors = backend.empty((Nev_max, corr.Nx, corr.Nx, corr.Nx, corr.Nc), dtype = complex)
        prop_bubble_strange = backend.empty((1, corr.Nx, corr.Nx, corr.Nx, Ns, Ns, Nc, Nev_max), dtype = complex)
        gauge_link = backend.empty((corr.Nx, corr.Nx, corr.Nx, 4, Nc, Nc), dtype = complex)
                
        st_load_time = time.perf_counter()
        
        try:
            # prop_bubble_light[:] = np.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_src:03d}_tsink{t_src:03d}_gen_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')
            prop_bubble_strange[:] = np.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/strange/{conf_id}/t{t_src:03d}_tsink{t_src:03d}_gen_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')
            
            vectors[:] = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/{conf_id}_t{t_src:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')

            gauge_link[:] = Readin_gauge(
                f'/nexdata/project/lqcd/sush/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{conf_id}.lime.contents/msg02.rec04.ildg-binary-data',
                lattice_size = lattice_size
                )[t_src]
            
        except Exception as e:
            print(f"{e}")
            comm.Abort(1)
            
    else:
        # prop_bubble_light = None
        
        prop_bubble_strange = None
        vectors = None
        gauge_link = None
        
    if rank == 0:
        print(f'load prop use time {(time.perf_counter() - st_load_time):.3f} s of tsrc {t_src}')
        
    # prop_bubble_light = get_mpi_data(data = prop_bubble_light, mdtype = 'TScatter', axis = 1)
    
    random_x = np.sort(backend.asarray(random.sample([x for x in range(corr.Nx)], k = corr.Nx)))
    random_y = np.sort(backend.asarray(random.sample([x for x in range(corr.Nx//size)], k = corr.Nx//size)))
    random_z = np.sort(backend.asarray(random.sample([x for x in range(corr.Nx)], k = corr.Nx)))
    
    prop_bubble_strange = get_mpi_data(data = prop_bubble_strange, mdtype = 'TScatter', axis = 2)
    prop_bubble_strange = contract('tzyxijCn,ji->tzyxCn', prop_bubble_strange, gamma(4))
    prop_bubble_strange = prop_link_transpose(prop = prop_bubble_strange[0], link_max = link_max, lattice_size = lattice_size, axis = 0)
    
    gauge_link = get_mpi_data(data = gauge_link, mdtype = 'TScatter', axis = 1)
    gauge_link = gauge_link_transpose(gauge = gauge_link[:, :, :, 2, :, :], link_max = link_max, lattice_size = lattice_size, axis = 0) # Nd = (X Y Z T)
    
    vectors = get_mpi_data(data = vectors, mdtype = 'TScatter', axis = 2)
    
    st_cal_time = time.perf_counter()
    
    vectors_slicer = ArraySlicer(vectors.conj())
    gauge_link_slicer = ArraySlicer(gauge_link)
    prop_bubble_strange_slicer = ArraySlicer(prop_bubble_strange)
    
    # bubble_light_matrix[:, :, t_src] = contract('zyxijcn,zyxcC,nzyxC,ji,n->', prop_bubble_light[0], vectors.conj(), gauge_link, backend.asarray(gamma(4)), omega1, optimize = 'optimal')
    # bubble_strange_matrix[:, t_src] = contract('nzyxc,LzyxcC,LzyxCn,n->L', vectors.conj(), gauge_link, prop_bubble_strange, omega1, optimize = 'optimal')
    
    bubble_strange_matrix[:, t_src] = contract(
        'nzyxc,LzyxcC,Mzyxc,LzyxCn,n->ML', 
        vectors_slicer.slice(dims = [2, 3], indices = [random_y, random_x]), 
        gauge_link_slicer.slice(dims = [2, 3], indices = [random_y, random_x]), 
        phase,
        prop_bubble_strange_slicer.slice(dims = [2, 3], indices = [random_y, random_x]), 
        omega1, 
        optimize = 'optimal'
        )
    
    
    if rank == 0:
        print(f'cal bubble use time {(time.perf_counter() - st_cal_time):.3f} s of tsrc {t_src}')

# bubble_light_matrix = get_mpi_data(data = bubble_light_matrix, mdtype = 'Gather', root = 0)
bubble_strange_matrix = get_mpi_data(data = bubble_strange_matrix, mdtype = 'Gather', root = 0)

if rank == 0:
    # bubble_light_matrix = bubble_light_matrix.sum(0)
    bubble_strange_matrix = bubble_strange_matrix.sum(0)
    
    for Mom_indx, Mom_list in enumerate(Mom_link[:]):
        corr_save_path = f'/public/home/sush/distillation/proton/result/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_50/conf{conf_id}'
        import pathlib

        path = pathlib.Path(corr_save_path)

        if path.exists():
            print('save_path:',corr_save_path)
        
        else:
            path.mkdir(parents = True, exist_ok = True)
            print('mkdir_save_path:',corr_save_path)

        for link_indx in range(-link_max, link_max + 1, 1):
            # backend.save(f'{corr_save_path}/corr_bubble_light_gamma{(mu):02d}_gen_stout_smear_20_0.12_src50.npy', bubble_light_matrix[Mom_indx, mu_indx_1])
            backend.save(f'{corr_save_path}/corr_bubble_strange_gamma04_link_Z_indx{link_indx}_gen_stout_smear_20_0.12_src50.npy', bubble_strange_matrix[link_indx])
            # backend.save(f'{corr_save_path}/corr_bubble_strange_gamma04_link_Z_random_YX_indx{link_indx}_gen_stout_smear_20_0.12_src50.npy', bubble_strange_matrix[link_indx])
            # backend.save(f'{corr_save_path}/corr_bubble_u-s_gamma{(mu):02d}_gen_stout_smear_20_0.12_src50.npy', bubble_light_matrix[Mom_indx, mu_indx_1] - bubble_strange_matrix[Mom_indx, mu_indx_1])

# disconnect Y
bubble_strange_matrix = backend.zeros((2 * link_max + 1, corr.Nt), dtype = complex)

for t_src in range(corr.Nt):

    if rank == 0:
        
        # prop_bubble_light = backend.empty((1, corr.Nx, corr.Nx, corr.Nx, Ns, Ns, Nc, Nev_max), dtype = complex)
        vectors = backend.empty((Nev_max, corr.Nx, corr.Nx, corr.Nx, corr.Nc), dtype = complex)
        prop_bubble_strange = backend.empty((1, corr.Nx, corr.Nx, corr.Nx, Ns, Ns, Nc, Nev_max), dtype = complex)
        gauge_link = backend.empty((corr.Nx, corr.Nx, corr.Nx, 4, Nc, Nc), dtype = complex)
                
        st_load_time = time.perf_counter()
        
        try:
            # prop_bubble_light[:] = np.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_src:03d}_tsink{t_src:03d}_gen_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')
            prop_bubble_strange[:] = np.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/strange/{conf_id}/t{t_src:03d}_tsink{t_src:03d}_gen_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')
            
            vectors[:] = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/{conf_id}_t{t_src:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')

            gauge_link[:] = Readin_gauge(
                f'/nexdata/project/lqcd/sush/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{conf_id}.lime.contents/msg02.rec04.ildg-binary-data',
                lattice_size = lattice_size
                )[t_src]
            
        except Exception as e:
            print(f"{e}")
            comm.Abort(1)
            
    else:
        
        # prop_bubble_light = None
        prop_bubble_strange = None
        vectors = None
        gauge_link = None
        
    if rank == 0:
        print(f'load prop use time {(time.perf_counter() - st_load_time):.3f} s of tsrc {t_src}')
        
    random_x = np.sort(backend.asarray(random.sample([x for x in range(corr.Nx)], k = corr.Nx)))
    random_y = np.sort(backend.asarray(random.sample([x for x in range(corr.Nx)], k = corr.Nx)))
    random_z = np.sort(backend.asarray(random.sample([x for x in range(corr.Nx//size)], k = corr.Nx//size)))

    # prop_bubble_light = get_mpi_data(data = prop_bubble_light, mdtype = 'TScatter', axis = 1)
    prop_bubble_strange = get_mpi_data(data = prop_bubble_strange, mdtype = 'TScatter', axis = 1)
    prop_bubble_strange = contract('tzyxijCn,ji->tzyxCn', prop_bubble_strange, gamma(4))
    prop_bubble_strange = prop_link_transpose(prop = prop_bubble_strange[0], link_max = link_max, lattice_size = lattice_size, axis = 1)
    
    gauge_link = get_mpi_data(data = gauge_link, mdtype = 'TScatter', axis = 0)
    gauge_link = gauge_link_transpose(gauge = gauge_link[:, :, :, 1, :, :], link_max = link_max, lattice_size = lattice_size, axis = 1) # Nd = (X Y Z T)
    
    vectors = get_mpi_data(data = vectors, mdtype = 'TScatter', axis = 1)
    
    st_cal_time = time.perf_counter()
    
    # bubble_light_matrix[:, :, t_src] = contract('zyxijcn,zyxcC,nzyxC,ji,n->', prop_bubble_light[0], vectors.conj(), gauge_link, backend.asarray(gamma(4)), omega1, optimize = 'optimal')
    # bubble_strange_matrix[:, t_src] = contract('nzyxc,LzyxcC,LzyxCn,n->L', vectors.conj(), gauge_link, prop_bubble_strange, omega1, optimize = 'optimal')
    bubble_strange_matrix[:, t_src] = contract(
        'nzyxc,LzyxcC,LzyxCn,n->L', 
        vectors_slicer.slice(dims = [1, 3], indices = [random_z, random_x]), 
        gauge_link_slicer.slice(dims = [1, 3], indices = [random_z, random_x]), 
        prop_bubble_strange_slicer.slice(dims = [1, 3], indices = [random_z, random_x]), 
        omega1, 
        optimize = 'optimal'
        )
    
    if rank == 0:
        print(f'cal bubble use time {(time.perf_counter() - st_cal_time):.3f} s of tsrc {t_src}')

# bubble_light_matrix = get_mpi_data(data = bubble_light_matrix, mdtype = 'Gather', root = 0)
bubble_strange_matrix = get_mpi_data(data = bubble_strange_matrix, mdtype = 'Gather', root = 0)

if rank == 0:
    # bubble_light_matrix = bubble_light_matrix.sum(0)
    bubble_strange_matrix = bubble_strange_matrix.sum(0)
    
    for Mom_indx, Mom_list in enumerate(Mom_link[:]):
        corr_save_path = f'/public/home/sush/distillation/proton/result/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_50/conf{conf_id}'
        import pathlib

        path = pathlib.Path(corr_save_path)

        if path.exists():
            print('save_path:',corr_save_path)
        
        else:
            path.mkdir(parents = True, exist_ok = True)
            print('mkdir_save_path:',corr_save_path)

        for link_indx in range(-link_max, link_max + 1, 1):
            # backend.save(f'{corr_save_path}/corr_bubble_light_gamma{(mu):02d}_gen_stout_smear_20_0.12_src50.npy', bubble_light_matrix[Mom_indx, mu_indx_1])
            backend.save(f'{corr_save_path}/corr_bubble_strange_gamma04_link_Y_indx{link_indx}_gen_stout_smear_20_0.12_src50.npy', bubble_strange_matrix[link_indx])
            # backend.save(f'{corr_save_path}/corr_bubble_strange_gamma04_link_Y_random_ZX_indx{link_indx}_gen_stout_smear_20_0.12_src50.npy', bubble_strange_matrix[link_indx])
            # backend.save(f'{corr_save_path}/corr_bubble_u-s_gamma{(mu):02d}_gen_stout_smear_20_0.12_src50.npy', bubble_light_matrix[Mom_indx, mu_indx_1] - bubble_strange_matrix[Mom_indx, mu_indx_1])            
            
# disconnect X
bubble_strange_matrix = backend.zeros((2 * link_max + 1, corr.Nt), dtype = complex)

for t_src in range(corr.Nt):

    if rank == 0:
        # prop_bubble_light = backend.empty((1, corr.Nx, corr.Nx, corr.Nx, Ns, Ns, Nc, Nev_max), dtype = complex)
        vectors = backend.empty((Nev_max, corr.Nx, corr.Nx, corr.Nx, corr.Nc), dtype = complex)
        prop_bubble_strange = backend.empty((1, corr.Nx, corr.Nx, corr.Nx, Ns, Ns, Nc, Nev_max), dtype = complex)
        gauge_link = backend.empty((corr.Nx, corr.Nx, corr.Nx, 4, Nc, Nc), dtype = complex)
                
        st_load_time = time.perf_counter()
        
        try:
            # prop_bubble_light[:] = np.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/light/{conf_id}/t{t_src:03d}_tsink{t_src:03d}_gen_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')
            prop_bubble_strange[:] = np.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2770_ms-0.2400_L24x72/strange/{conf_id}/t{t_src:03d}_tsink{t_src:03d}_gen_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')
            
            vectors[:] = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2770_ms-0.2400_L24x72/{conf_id}/{conf_id}_t{t_src:03d}_e50_s[100, 200, 300, 500, 800]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n500_V1_stout_smear_20_0.12.npy')

            gauge_link[:] = Readin_gauge(
                f'/nexdata/project/lqcd/sush/configurations/CLOVER/beta6.20_mu-0.2770_ms-0.2400_L24x72/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{conf_id}.lime.contents/msg02.rec04.ildg-binary-data',
                lattice_size = lattice_size
                )[t_src]
            
        except Exception as e:
            print(f"{e}")
            comm.Abort(1)
            
    else:
        # prop_bubble_light = None
        prop_bubble_strange = None
        vectors = None
        gauge_link = None
        
    if rank == 0:
        print(f'load prop use time {(time.perf_counter() - st_load_time):.3f} s of tsrc {t_src}')
    
    random_x = np.sort(backend.asarray(random.sample([x for x in range(corr.Nx)], k = corr.Nx)))
    random_y = np.sort(backend.asarray(random.sample([x for x in range(corr.Nx//size)], k = corr.Nx//size)))
    random_z = np.sort(backend.asarray(random.sample([x for x in range(corr.Nx)], k = corr.Nx)))
    
    # prop_bubble_light = get_mpi_data(data = prop_bubble_light, mdtype = 'TScatter', axis = 1)
    prop_bubble_strange = get_mpi_data(data = prop_bubble_strange, mdtype = 'TScatter', axis = 2)
    prop_bubble_strange = contract('tzyxijCn,ji->tzyxCn', prop_bubble_strange, gamma(4))
    prop_bubble_strange = prop_link_transpose(prop = prop_bubble_strange[0], link_max = link_max, lattice_size = lattice_size, axis = 2)
    
    gauge_link = get_mpi_data(data = gauge_link, mdtype = 'TScatter', axis = 1)
    gauge_link = gauge_link_transpose(gauge = gauge_link[:, :, :, 0, :, :], link_max = link_max, lattice_size = lattice_size, axis = 2) # Nd = (X Y Z T)
    
    vectors = get_mpi_data(data = vectors, mdtype = 'TScatter', axis = 2)
    
    st_cal_time = time.perf_counter()
    
    # bubble_light_matrix[:, :, t_src] = contract('zyxijcn,zyxcC,nzyxC,ji,n->', prop_bubble_light[0], vectors.conj(), gauge_link, backend.asarray(gamma(4)), omega1, optimize = 'optimal')
    # bubble_strange_matrix[:, t_src] = contract('nzyxc,LzyxcC,LzyxCn,n->L', vectors.conj(), gauge_link, prop_bubble_strange, omega1, optimize = 'optimal')
    
    bubble_strange_matrix[:, t_src] = contract(
        'nzyxc,LzyxcC,LzyxCn,n->L', 
        vectors_slicer.slice(dims = [1, 2], indices = [random_z, random_y]), 
        gauge_link_slicer.slice(dims = [1, 2], indices = [random_z, random_y]), 
        prop_bubble_strange_slicer.slice(dims = [1, 2], indices = [random_z, random_y]), 
        omega1, 
        optimize = 'optimal'
        )
    
    if rank == 0:
        print(f'cal bubble use time {(time.perf_counter() - st_cal_time):.3f} s of tsrc {t_src}')

# bubble_light_matrix = get_mpi_data(data = bubble_light_matrix, mdtype = 'Gather', root = 0)
bubble_strange_matrix = get_mpi_data(data = bubble_strange_matrix, mdtype = 'Gather', root = 0)

if rank == 0:
    # bubble_light_matrix = bubble_light_matrix.sum(0)
    bubble_strange_matrix = bubble_strange_matrix.sum(0)
    
    for Mom_indx, Mom_list in enumerate(Mom_link[:]):
        corr_save_path = f'/public/home/sush/distillation/proton/result/{corr.Nx}x{corr.Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_50/conf{conf_id}'
        import pathlib

        path = pathlib.Path(corr_save_path)

        if path.exists():
            print('save_path:',corr_save_path)
        
        else:
            path.mkdir(parents = True, exist_ok = True)
            print('mkdir_save_path:',corr_save_path)

        for link_indx in range(-link_max, link_max + 1, 1):
            # backend.save(f'{corr_save_path}/corr_bubble_light_gamma{(mu):02d}_gen_stout_smear_20_0.12_src50.npy', bubble_light_matrix[Mom_indx, mu_indx_1])
            backend.save(f'{corr_save_path}/corr_bubble_strange_gamma04_link_X_indx{link_indx}_gen_stout_smear_20_0.12_src50.npy', bubble_strange_matrix[link_indx])
            # backend.save(f'{corr_save_path}/corr_bubble_strange_gamma04_link_X_random_ZY_indx{link_indx}_gen_stout_smear_20_0.12_src50.npy', bubble_strange_matrix[link_indx])
            # backend.save(f'{corr_save_path}/corr_bubble_u-s_gamma{(mu):02d}_gen_stout_smear_20_0.12_src50.npy', bubble_light_matrix[Mom_indx, mu_indx_1] - bubble_strange_matrix[Mom_indx, mu_indx_1])            
            