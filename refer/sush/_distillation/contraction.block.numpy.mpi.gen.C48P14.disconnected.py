import sys
from lqcddb import *
import time

set_backend('numpy')
backend = get_backend()

lattice_size = [48, 48, 48, 96]
grid_size = [1, 1, 1, 24]

def find_opposite_momentum(A, mom_list):
    """在 creat_mom_list 返回的列表中查找 B = -A，返回 (B, index)。

    A、B 的分量顺序均为 [pz, py, px]；index 从 0 开始。
    如果重复出现 B，返回第一次出现的位置；找不到时抛出 ValueError。
    """
    A = list(A)
    if len(A) != 3:
        raise ValueError("A 必须包含三个动量分量 [pz, py, px]")

    target = [-x for x in A]
    for index, B in enumerate(mom_list):
        if list(B) == target:
            return list(B), index

    raise ValueError(f"动量列表中未找到 B = {target}（A = {A}）")

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
tsep = 20

mpinit(grid_size = grid_size, latt_size = lattice_size, backend = backend.__name__)

comm = getMPIComm()
rank = getMPIRank()
size = getMPISize()

Nx, Ny, Nz, Nt = lattice_size
Gt = Nt//size

eigen = vertex_creator(Nx = Nx)
Mom_link = creat_mom_list(Mom = [0, 0, 1], fix_Q2 = True)

Mom_link_dis = []
for A in Mom_link:
    B, _ = find_opposite_momentum(A, Mom_link)
    Mom_link_dis += [B]
    
mu_nu_indx, mu_nu_gamma, mu_indx, mu_gamma =  PFF_Mom_to_gamma_new(Mom_link[:], allow_t = False)

mu_nu_gamma = backend.asarray(mu_nu_gamma)
mu_gamma = backend.asarray(mu_gamma)
phase_link = backend.empty((len(Mom_link), Nx, Nx, Nx, Nc), dtype = complex)

corr_3pt_matrix = backend.zeros((len(Mom_link), len(mu_nu_indx[0]), Nt, Nt), dtype = complex)
corr_2pt_mu_matrix = backend.zeros((len(Mom_link), len(mu_indx[0]), Nt, Nt), dtype = complex)
corr_2pt_matrix = backend.zeros((Nt, Gt), dtype = complex)

for i in range(len(Mom_link)):
    phase_link[i] = backend.asarray(eigen.phase_exp_VdV(Mom = Mom_link[i]))

phase_link = phase_link.reshape(len(Mom_link), Nx * Nx * Nx, Nc)

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

omega3 = eigen.create_omega_accelerate(
    exact = exact_eigen,
    N_eigen = N_eigen,
    N_sum = N_sum,
    N_extract = N_extract,
    noise = noise_eigen,
    dim = 3
)

V = Nx ** 3

t_rank, _, _ = get_mpi_tlist(Nt = Nt, t = range(Nt), gtype = 'TScatter')

VdV_link = backend.zeros((Gt, len(Mom_link), 1, Nev_max, Nev_max), dtype = complex)
sink = backend.zeros((Gt, 1, Nev_src, Nev_src), dtype = complex)

st_eigen = time.perf_counter()
# for tsrc_indx, tsrc in enumerate(t_rank):
#     eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.20_mu-0.2825_ms-0.2310_L48x96/{conf_id}/{conf_id}_t{tsrc:03d}_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_20_0.12.npy')
    
#     for Mom_indx in range(len(Mom_link)):
#         VdV_link[tsrc_indx, (Mom_indx):((Mom_indx + 1))] = eigen.VdV_sink_t_link(eigvecs = eigvecs[:], link_dir = 'Z', link_max = 0, phase_exp = phase_link[(Mom_indx):((Mom_indx + 1))], gauge_link = False)
        
if rank == 0:
    print(f'load eigen and cal VVV VDV use time {(time.perf_counter() - st_eigen):.3f} s')

if rank == 0 :
    bubble_light_matrix = backend.zeros((Nev_max, len(Mom_link), len(mu_indx[0]), Nt), dtype = complex)
    bubble_strange_matrix = backend.zeros((Nev_max, len(Mom_link), len(mu_indx[0]), Nt), dtype = complex)
    
    # for Mom_indx, Mom in enumerate(Mom_link_dis):
    #     corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_gen/{Nx}x{Nt}/Px{Mom[2]}Py{Mom[1]}Pz{Mom[0]}/ENV_100/conf{conf_id}'
    #     for mu_indx, mu in enumerate(mu_indx[Mom_indx]):
    #         bubble_light_matrix[:, Mom_indx, mu_indx] = backend.load(f'{corr_save_path}/corr_bubble_u-s_gamma{(mu):02d}_gen_stout_20_0.12_src400.npy')

else:
    bubble_light_matrix = None
    bubble_strange_matrix = None
    
    
bubble_light_matrix = get_mpi_data(data = bubble_light_matrix, mdtype = 'TScatter', root = 0, axis = -1)
bubble_strange_matrix = get_mpi_data(data = bubble_strange_matrix, mdtype = 'TScatter', root = 0, axis = -1)

time_src = time.perf_counter()
for tsrc in range(Nt):
    try:
        if rank == 0:
            peram_u = backend.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2825_ms-0.2310_L48x96/light/{conf_id}/t{tsrc:03d}_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_20_0.12.npy')

        else:
            peram_u = None
            
    except:
        comm.Abort(1)
            
    peram_u = get_mpi_data(data = peram_u, mdtype = 'TScatter', root = 0, axis = 0)
    peram_d = seq_peram(peram_u[..., :Nev_src])

    t_curr_0_list, _, t_curr_0_list_indx = get_mpi_tlist(Nt = Nt, t = [x%Nt for x in range(tsrc, tsrc + Nt, 1)], gtype = 'TScatter')

    if rank == 0:
        peram_u_sep = backend.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.20_mu-0.2825_ms-0.2310_L48x96/light/{conf_id}/t{((tsep + tsrc)%Nt):03d}_e{exact_eigen}_s{N_eigen}_{N_sum}_{N_extract}_BI_n{noise_eigen_name}_V1_stout_20_0.12.npy')
    else:
        peram_u_sep = None
    
    peram_u_sep = get_mpi_data(data = peram_u_sep, mdtype = 'TScatter', root = 0, axis = 0)
    peram_u_sep = seq_peram(peram_u_sep)
    
    peram_d_2pt = backend.asarray(get_mpi_data(peram_d[((tsep + tsrc)%Nt)//size], mdtype = 'Bcast', root = ((tsep + tsrc)%Nt)%size))
    sink_3pt = backend.asarray(get_mpi_data(VdV_link[((tsep + tsrc)%Nt)//size, :, 0].conj().transpose(0, 2, 1), mdtype = 'Bcast', root = ((tsep + tsrc)%Nt)%size))
        
    time_2pt_mu = time.perf_counter()
    for t_sink_list_indx, t_sink_indx in enumerate(t_curr_0_list_indx):
        corr_2pt_matrix[tsrc, t_sink_list_indx] = cached_contract(
            'manb,cebn,ac,em->', 
            backend.asarray(peram_u[t_sink_indx, ..., :Nev_src, :Nev_src]), backend.asarray(peram_d[t_sink_indx, ..., :Nev_src, :Nev_src]), 
            backend.asarray(gamma(5)), backend.asarray(gamma(5)), 
            optimize = ['auto', 'greedy', 'dp']
            )
        
        # corr_2pt_mu_matrix_mid = cached_contract(
        #     'manb,cebf,Mfn,ac,MFem->MFfn', 
        #     backend.asarray(peram_u[t_sink_indx, ..., :Nev_max, :Nev_src]), backend.asarray(peram_d[t_sink_indx, ..., :Nev_src, :Nev_max]), 
        #     backend.asarray(VdV_link[t_sink_indx, :, 0, :Nev_max, :Nev_max]), 
        #     backend.asarray(gamma(5)), backend.asarray(mu_gamma), 
        #     optimize = ['auto', 'greedy', 'dp']
        #     )
        
        # corr_2pt_bubble_3pt = cached_contract(
        #     'MFij,kMf,ijk->MFf',
        #     corr_2pt_mu_matrix_mid, bubble_light_matrix[..., t_sink_list_indx], omega3, optimize = ['auto', 'greedy', 'dp']
        # )

        if rank == 0:
            print(f"complete cal 2pt_mu and bubble of tsrc {tsrc} tsink {t_sink_indx} use time {(time.perf_counter() - time_2pt_mu):.3f} s")
            
    comm.Barrier()
    if rank == 0:
        print(f"complete cal 2pt_mu and bubble of tsrc {tsrc} use time {(time.perf_counter() - time_2pt_mu):.3f} s")
    

corr_2pt_matrix = get_mpi_data(corr_2pt_matrix, mdtype = 'TGather', root = 0, axis = -1)
if rank == 0: 

    corr_save_path = f'/public/home/sush/distillation/pion_form_factor/result_compensation/{Nx}x{Nt}/Px0Py0Pz0/ENV_100/conf{conf_id}'
    import pathlib

    path = pathlib.Path(corr_save_path)

    if path.exists():
        print('save_path:',corr_save_path)
    
    else:
        path.mkdir(parents = True, exist_ok = True)
        print('mkdir_save_path:',corr_save_path)

    backend.save(f"{corr_save_path}/corr_ud_2pt_src100_sink100.npy", loop_tsrc(corr_2pt_matrix, indx = [0, 1]))

if rank == 0:
    print('complete')