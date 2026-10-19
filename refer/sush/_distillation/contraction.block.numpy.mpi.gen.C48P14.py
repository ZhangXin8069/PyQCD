import sys
from lqcddb import *
import time

set_backend('numpy')
backend = get_backend()

lattice_size = [48, 48, 48, 96]
grid_size = [1, 1, 1, 48]


conf_id = sys.argv[1]
exact_eigen = 100
N_eigen = [50, 100, 150, 200, 300]
N_sum = [10, 10, 10, 10, 10]
N_extract = [5, 5, 5, 5, 5]
noise_eigen = 250

exact_eigen_name = 100
N_eigen_name = [50, 100, 150, 200, 300]
N_sum_name = [10, 10, 10, 10, 10]
N_extract_name = [5, 5, 5, 5, 5]
noise_eigen_name = 250

Nev_src = 100
Nev_max = 400
t_sep = 20

mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

comm = getMPIComm()
rank = getMPIRank()
size = getMPISize()

Nx, Ny, Nz, Nt = lattice_size
Gt = Nt//size

V_list = [x for x in range(Nx**3)][rank*(Nx**3//size):(rank+1)*(Nx**3//size)]
V_max = Nx**3

eigen = vertex_creator(Nx = Nx)
Mom_link = creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)
mu_nu_indx, mu_nu_gamma, mu_indx, mu_gamma =  PFF_Mom_to_gamma_new(Mom_link[:], allow_t = False)

mu_nu_gamma = backend.asarray(mu_nu_gamma)
mu_gamma = backend.asarray(mu_gamma)
phase_link = backend.empty((len(Mom_link), Nx, Nx, Nx, Nc), dtype = complex)

corr_3pt_matrix = backend.zeros((len(Mom_link), len(mu_nu_indx[0]), Nt, Nt), dtype = complex)
corr_2pt_mu_matrix = backend.zeros((len(Mom_link), len(mu_indx[0]), Nt, Nt), dtype = complex)

bubble_light_matrix = backend.zeros((Nev_max, len(Mom_link), len(mu_indx[0]), Nt), dtype = complex)
bubble_strange_matrix = backend.zeros((Nev_max, len(Mom_link), len(mu_indx[0]), Nt), dtype = complex)

for i in range(len(Mom_link)):
    phase_link[i] = backend.asarray(eigen.phase_exp_VdV(Mom = Mom_link[i]))

phase_link = phase_link.reshape(len(Mom_link), Nx * Nx * Nx, Nc)[:, V_list]

import time

g5 = backend.asarray(gamma(5))

omega1 = eigen.create_omega_accelerate(
    exact = exact_eigen,
    N_eigen = N_eigen,
    N_sum = N_sum,
    N_extract = N_extract,
    noise = noise_eigen,
    dim = 1
)

omega2 = eigen.create_omega_accelerate(
    exact = exact_eigen,
    N_eigen = N_eigen,
    N_sum = N_sum,
    N_extract = N_extract,
    noise = noise_eigen,
    dim = 2
)

# omega3 = eigen.create_omega_accelerate(
#     exact = exact_eigen,
#     N_eigen = N_eigen,
#     N_sum = N_sum,
#     N_extract = N_extract,
#     noise = noise_eigen,
#     dim = 3
# )

V = Nx ** 3
# vectors = backend.zeros((Gt, Nev_max, Nx, Nx, Nx, Nc), dtype = complex)
vectors_tmp = backend.zeros((Nt, Nev_max, len(V_list), Nc), dtype = complex)

for tsrc in range(Gt):
    vectors = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2825_ms-0.2310_L48x96/{conf_id}/{conf_id}_t{(tsrc + rank*Gt):03d}_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_20_0.12.npy')

    for rank_indx in range(size):
        vectors_tmp[rank_indx * Gt + tsrc] = get_mpi_data(data = vectors.reshape(Nev_max, V_max, Nc), mdtype = 'Scatter', axis = 1, root = rank_indx)

del vectors

VdV_link = backend.zeros((Gt, len(Mom_link), 1, Nev_max, Nev_max), dtype = complex)
sink = backend.zeros((Gt, 1, Nev_src, Nev_src), dtype = complex)

for tsrc in range(Nt):
    if rank == 0:
        st_load_time = time.perf_counter()
        
        prop_bubble_light = backend.empty((1, Nx, Nx, Nx, Ns, Ns, Nc, Nev_max), dtype = complex)
        prop_bubble_strange = backend.empty((1, Nx, Nx, Nx, Ns, Ns, Nc, Nev_max), dtype = complex)
        
        try:
            prop_bubble_light[:] = backend.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2825_ms-0.2310_L48x96/light/{conf_id}/t{tsrc:03d}_tsink{tsrc:03d}_gen_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_20_0.12.npy')
            prop_bubble_strange[:] = backend.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2825_ms-0.2310_L48x96/strange/{conf_id}/t{tsrc:03d}_tsink{tsrc:03d}_gen_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_20_0.12.npy')
            
            prop_bubble_light = prop_bubble_light.reshape(1, V_max, Ns, Ns, Nc, Nev_max)
            prop_bubble_strange = prop_bubble_strange.reshape(1, V_max, Ns, Ns, Nc, Nev_max)
            
        except:
            comm.Abort(1)

    else:
        prop_bubble_light = None
        prop_bubble_strange = None
        
    if rank == 0:
        print(f'load prop use time {(time.perf_counter() - st_load_time):.3f} s of tsrc {tsrc}')
        
    prop_bubble_light = get_mpi_data(data = prop_bubble_light, mdtype = 'Scatter', axis = 1)
    prop_bubble_strange = get_mpi_data(data = prop_bubble_strange, mdtype = 'Scatter', axis = 1)

    st_cal_time = time.perf_counter()
    
    bubble_light_matrix[..., tsrc] = cached_contract('Vijcn,MVc,MGji,nVc->nMG', prop_bubble_light[0], phase_link, mu_gamma, vectors_tmp[tsrc].conj(), optimize = True)
    bubble_strange_matrix[..., tsrc] = cached_contract('Vijcn,MVc,MGji,nVc->nMG', prop_bubble_strange[0], phase_link, mu_gamma, vectors_tmp[tsrc].conj(), optimize = True)

    if rank == 0:
        print(f'cal bubble use time {(time.perf_counter() - st_cal_time):.3f} s of tsrc {tsrc}')

    del prop_bubble_light, prop_bubble_strange

bubble_light_matrix = get_mpi_data(data = bubble_light_matrix, mdtype = 'Gather', root = 0)
bubble_strange_matrix = get_mpi_data(data = bubble_strange_matrix, mdtype = 'Gather', root = 0)

if rank == 0:
    bubble_light_matrix = bubble_light_matrix.sum(0)
    bubble_strange_matrix = bubble_strange_matrix.sum(0)
    
    print(bubble_light_matrix.shape)
    
    for Mom_indx, Mom_list in enumerate(Mom_link[:]):
        corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_gen/{Nx}x{Nt}/Px{Mom_list[2]}Py{Mom_list[1]}Pz{Mom_list[0]}/ENV_400/conf{conf_id}'
        import pathlib

        path = pathlib.Path(corr_save_path)

        if path.exists():
            print('save_path:',corr_save_path)
        
        else:
            path.mkdir(parents = True, exist_ok = True)
            print('mkdir_save_path:',corr_save_path)

        for mu_indx_1, mu in enumerate(mu_indx[Mom_indx]):
            # backend.save(f'{corr_save_path}/corr_2pt_gamma{(mu):02d}_gen_stout_20_0.12_src50.npy', corr_2pt_mu_matrix[Mom_indx, mu_indx_1])
            backend.save(f'{corr_save_path}/corr_bubble_light_gamma{(mu):02d}_gen_stout_20_0.12_src400.npy', bubble_light_matrix[:, Mom_indx, mu_indx_1])
            backend.save(f'{corr_save_path}/corr_bubble_strange_gamma{(mu):02d}_gen_stout_20_0.12_src400.npy', bubble_strange_matrix[:, Mom_indx, mu_indx_1])
            backend.save(f'{corr_save_path}/corr_bubble_u-s_gamma{(mu):02d}_gen_stout_20_0.12_src400.npy', bubble_light_matrix[:, Mom_indx, mu_indx_1] - bubble_strange_matrix[:, Mom_indx, mu_indx_1])
            
        # for gamma_indx, gamma_list in enumerate(mu_nu_indx[Mom_indx]):
        #     backend.save(f'{corr_save_path}/corr_3pt_gamma0{gamma_list[1]}0{gamma_list[0]}05_tsep{t_sep}_gen_stout_20_0.12_src50_sink{Nev_max}_curr{Nev_max}_compensation.npy', corr_3pt_matrix[Mom_indx, gamma_indx])