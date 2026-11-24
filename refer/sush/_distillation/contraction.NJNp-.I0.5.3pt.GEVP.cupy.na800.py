import sys
import numpy as np

sys.path.insert(0, '/public/home/sush/distillation/lqcddb_update/src')
from lqcddb import *
import time

# M0 Runtime: retain the NA800 top-level CuPy/MPI structure.
st_total = time.perf_counter()
set_backend('cupy')
backend = get_backend()

lattice_size = [32, 32, 32, 64]
grid_size = [1, 1, 1, 2]
conf_id = sys.argv[1]

# M1 Lattice and MPI: two ranks, one NA800 GPU per rank.
mpinit(grid_size=grid_size, latt_size=lattice_size, backend=backend.__name__)
rank = getMPIRank()
size = getMPISize()
comm = getMPIComm()
Nx, Ny, Nz, Nt = lattice_size
Lx, Ly, Lz, Lt = [lattice_size[x]//grid_size[x] for x in range(len(lattice_size))]
fun_eigen = vertex_creator(Nx=Nx)

# M2 Momentum and phases: [pz,py,px], as in the reference.
# Sink N carries p; the current carries -p. Source N(-r)pi(r) has total p=0.
Mom_sink_VDV = [[0, 0, 0]] + sorted(creat_mom_list(Mom=[0, 0, 1], fix_Q2=True)) + sorted(creat_mom_list(Mom=[0, 1, 1], fix_Q2=True)) + sorted(creat_mom_list(Mom=[1, 1, 1], fix_Q2=True))
Mom_sink_VVV = [[0, 0, 0]] + sorted(creat_mom_list(Mom=[0, 0, 1], fix_Q2=True))[::-1] + sorted(creat_mom_list(Mom=[0, 1, 1], fix_Q2=True))[::-1] + sorted(creat_mom_list(Mom=[1, 1, 1], fix_Q2=True))[::-1]
Mom_len = len(Mom_sink_VDV)
phase_exp_2pt = backend.zeros((Mom_len, Nx, Nx, Nx, Nc), dtype=complex)
phase_exp_3pt = backend.zeros((Mom_len, Nx, Nx, Nx), dtype=complex)
for Mom_indx in range(Mom_len):
    phase_exp_2pt[Mom_indx] = fun_eigen.phase_exp_VdV(Mom=Mom_sink_VDV[Mom_indx])
    phase_exp_3pt[Mom_indx] = fun_eigen.phase_exp_VVV(Mom=Mom_sink_VVV[Mom_indx])
Nev_src = 100
Nev_max = 400
tsep = int(sys.argv[2]) if len(sys.argv) > 2 else 20
if not 8 < tsep < Nt:
    raise ValueError('Require 8 < tsep < 64; measured offsets are 5,...,tsep-4.')
trank, _, _ = get_mpi_tlist(Nt=Nt, t=range(Nt), gtype='TScatter')

# M3 Dynamic manifest: I(Npi)=1/2 -> I(N)=1/2, charged axial current.
# pi0_E=(ubar gamma5 u-dbar gamma5 d)/sqrt(2), pi+_E=dbar gamma5 u.
# Initial Iz=+1/2: O=p*pi0_E/sqrt(3)+sqrt(2/3)*n*pi+_E.
# Sink Iz=-1/2: neutron. dbar gamma_mu gamma5 u conserves flavor here;
# the old pure n*pi- source instead required the opposite charged current.
proton = ['|', 'u', 'u', 'gamma_7', 'd', '|']
neutron = ['|', 'd', 'u', 'gamma_7', 'd', '|']
pion_u = ['|', 'u^d', 'gamma_5', 'u', '|']
pion_d = ['|', 'd^d', 'gamma_5', 'd', '|']
pion_plus = ['|', 'd^d', 'gamma_5', 'u', '|']
pion_minus = ['|', 'u^d', 'gamma_5', 'd', '|']
sink_operator = [
    [neutron],
    # OPTIONAL_N_GEVP_BEGIN
    # [[-np.sqrt(1/6), *neutron, *pion_u], [np.sqrt(1/6), *neutron, *pion_d], [np.sqrt(2/3), *proton, *pion_minus]],
    # [[-np.sqrt(1/6), *neutron, *pion_u], [np.sqrt(1/6), *neutron, *pion_d], [np.sqrt(2/3), *proton, *pion_minus]],
    # OPTIONAL_N_GEVP_END
]
initial_operator = [
    [np.sqrt(1/6), *proton, *pion_u],
    [-np.sqrt(1/6), *proton, *pion_d],
    [np.sqrt(2/3), *neutron, *pion_plus],
]
source_operator = [[conjugate_operator(term) for term in initial_operator]]
current_operator = ['|', 'd^d', 'gamma_1', 'u', '|']
Cpt, Projection = '3pt', True
use_equivalence, ignore_dis = False, False
optimize = ['greedy', 'dp', 'auto']
# To enable N-side GEVP data, uncomment CODE in every OPTIONAL_N_GEVP block.
# Basis order is [N, Sigma.p_hat*N(p)pi(0), Sigma.p_hat*N(0)pi(p)].
# The commented mixed groups have the same I=1/2, Iz=-1/2 CG as N C2.
dynamic_operator_groups = [[s, t, current_operator]
                           for sink_group in sink_operator
                           for s in sink_group for t in source_operator[0]]
dynamic_plan = []
for i in range(len(sink_operator)):
    Vindex = ['N', 'N', 'M', 'M']
    # OPTIONAL_N_GEVP_BEGIN
    # if i > 0:
    #     Vindex = ['N', 'N', 'N', 'M', 'M']
    # OPTIONAL_N_GEVP_END
    operator_groups = [[s, t, current_operator] for s in sink_operator[i] for t in source_operator[0]]
    dynamic_plan += run_wick_analysis(operator_groups, Cpt=Cpt, Vindex=Vindex,
        use_equivalence=use_equivalence, ignore_dis=ignore_dis, verbose=False)
required_peram = sorted({(p[2], tuple(p[4])) for entry in dynamic_plan for p in entry[2]['peram'][entry[3]]})
required_v = sorted({(v[1].split('_')[0]+'_0', v[3]) for entry in dynamic_plan for v in entry[2]['V']})
required_gamma = sorted({g[1] for entry in dynamic_plan for g in entry[2]['gamma_pos']})
# required peram: light at (tsrc,tsrc),(tsink,tsrc),(tcur0,tsrc),
# (tsrc,tcur0),(tsink,tcur0); endpoint=(4,4,100,100), current ends
# (4,4,400,100)/(4,4,100,400). No (tcur0,tcur0): off-diagonal flavor.
# required V: VVV_0@tsrc/tsink; VDV_0@tsrc/tcur0; gamma_7,gamma_5,
# gamma_1=gamma_mu@gamma5, Projector=[Q,Q], Q=P+[:,:2].
# output: (N_basis,tsrc_sum,tau,mu,sN,sNpi,p,r), complex128.
# Optional mixed groups additionally require VDV_0@tsink and the other
# endpoint/current peram directions; their registrations remain commented.

# M4 Vertex data: preserve smearing/noise/omega and Nev=100/400.
VdV_link = backend.zeros((Lt, Mom_len, Nev_max, Nev_max), dtype=complex)
sink_VVV = backend.zeros((Lt, Mom_len, Nev_src, Nev_src, Nev_src), dtype=complex)
comm.Barrier()
st_eigen = time.perf_counter()
for tsrc_indx, tsrc in enumerate(trank):
    eigvecs = backend.load(f'/nexdata/project/lqcd/sush/eigensystem/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/{conf_id}/{conf_id}_t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')
    for Mom_indx in range(Mom_len):
        VdV_link[tsrc_indx, Mom_indx:Mom_indx+1] = fun_eigen.Mom_VdV_sink_t(phase_exp=phase_exp_2pt[Mom_indx:Mom_indx+1], eigvecs=eigvecs)
        sink_VVV[tsrc_indx, Mom_indx] = fun_eigen.Mom_VVV_sink_t(phase_exp=phase_exp_3pt[Mom_indx], eigvecs=eigvecs[:Nev_src])[0]
VdV_link *= fun_eigen.create_omega_accelerate(
    exact=100, N_eigen=[100, 200, 400, 700, 1100],
    N_sum=[20, 20, 20, 20, 20], N_extract=[10, 10, 10, 10, 10], noise=200)
del phase_exp_2pt, phase_exp_3pt, eigvecs
backend.cuda.get_current_stream().synchronize()
comm.Barrier()
if rank == 0:
    print(f'TIMING vertices_seconds={time.perf_counter()-st_eigen:.6f}', flush=True)

# M5 Peram data: disk row is absolute sink time, never a time offset.
corr_NJN_GEVP = backend.zeros((len(sink_operator), Nt, Lt, 4, Ns//2, Ns//2, Mom_len, Mom_len), dtype=complex)
# OPTIONAL_N_GEVP_BEGIN
# corr_NJN_GEVP[1:, ..., 0, :] = complex(np.nan, np.nan)
# OPTIONAL_N_GEVP_END
comm.Barrier()
st_load = time.perf_counter()
if rank == 0:
    peram_u = np.zeros((Nt, Nt, Ns, Ns, Nev_max, Nev_src), dtype=complex)
    for tsrc in range(Nt):
        peram_u[tsrc] = np.load(f'/nexdata/project/lqcd/sush/perambulators/beta6.308_mu-0.2510_ms-0.2170_u00.859727_u0s0.954467_L32x64/light/{conf_id}/t{tsrc:03d}_e100_s[100, 200, 400, 700, 1100]_[20, 20, 20, 20, 20]_[10, 10, 10, 10, 10]_BI_n200_V1_hpy.npy')[..., :Nev_max, :Nev_src]
else:
    peram_u = None
peram_u = backend.asarray(get_mpi_data(data=peram_u, mdtype='TScatter', root=0, axis=1))
backend.cuda.get_current_stream().synchronize()
comm.Barrier()
if rank == 0:
    print(f'TIMING peram_seconds={time.perf_counter()-st_load:.6f}', flush=True)

# M6 Registries: retain reference Q normalization; no Z_A or weak coupling.
GR, VR, PR = GammaRegistry(), VRegistry(), PeramRegistry()
GR.register('gamma_7', gamma(7))
GR.register('gamma_5', gamma(5))
GR.register('Projector', [((gamma(4)+gamma(0))/2)[:, :2], ((gamma(4)+gamma(0))/2)[:, :2]])
# OPTIONAL_N_GEVP_BEGIN
# U = backend.sqrt(2.)*((gamma(4)+gamma(0))/2)[:, :2]
# Sigma_DR = backend.stack([-1j*U.conj().T@gamma(j)@gamma(k)@U for j,k in ((2,3),(3,1),(1,2))])
# OPTIONAL_N_GEVP_END

# M7 Dynamic loop: variables use the library owner/local-index mapping.
for tsrc in range(Nt):
    comm.Barrier()
    st_cal = time.perf_counter()
    tsink = (tsrc+tsep) % Nt
    src_rank, src_indx = get_mpi_tlist(Nt=Nt, t=tsrc, gtype='find')
    sink_rank, sink_indx = get_mpi_tlist(Nt=Nt, t=tsink, gtype='find')
    VVV_src = backend.asarray(get_mpi_data(data=sink_VVV[src_indx], mdtype='Bcast', root=src_rank)).conj()
    VDV_src = backend.asarray(get_mpi_data(data=VdV_link[src_indx, :, :Nev_src, :Nev_src], mdtype='Bcast', root=src_rank)).conj().transpose(0, 2, 1)
    VVV_sink = backend.asarray(get_mpi_data(data=sink_VVV[sink_indx], mdtype='Bcast', root=sink_rank))
    # OPTIONAL_N_GEVP_BEGIN
    # VDV_sink = backend.asarray(get_mpi_data(data=VdV_link[sink_indx, :, :Nev_src, :Nev_src], mdtype='Bcast', root=sink_rank)).conj().transpose(0,2,1)
    # OPTIONAL_N_GEVP_END
    peram_u_src_src = get_mpi_data(data=peram_u[tsrc, src_indx, ..., :Nev_src, :Nev_src], mdtype='Bcast', root=src_rank)
    peram_u_sink_src = get_mpi_data(data=peram_u[tsrc, sink_indx, ..., :Nev_src, :Nev_src], mdtype='Bcast', root=sink_rank)
    PR.register('light', ('tsrc', 'tsrc'), backend.asarray(peram_u_src_src))
    PR.register('light', ('tsink', 'tsrc'), backend.asarray(peram_u_sink_src))
    # OPTIONAL_N_GEVP_BEGIN
    # peram_u_sink_sink = get_mpi_data(data=peram_u[tsink, sink_indx, ..., :Nev_src, :Nev_src], mdtype='Bcast', root=sink_rank)
    # PR.register('light', ('tsink', 'tsink'), backend.asarray(peram_u_sink_sink))
    # PR.register('light', ('tsrc', 'tsink'), seq_peram(backend.asarray(peram_u_sink_src)))
    # OPTIONAL_N_GEVP_END
    VR.register('VDV_0', 'tsrc', VDV_src)
    VR.register('VVV_0', 'tsrc', VVV_src)
    _, _, tsink_indx = get_mpi_tlist(Nt=Nt, t=range(tsrc+5, tsrc+tsep-3), gtype='TScatter')
    for tcur0 in tsink_indx:
        peram_u_tcur0_tsrc = backend.asarray(peram_u[tsrc, tcur0])
        peram_u_tcur0_tsink = backend.asarray(peram_u[tsink, tcur0])
        PR.register('light', ('tcur0', 'tsrc'), peram_u_tcur0_tsrc)
        PR.register('light', ('tsrc', 'tcur0'), seq_peram(peram_u_tcur0_tsrc))
        PR.register('light', ('tsink', 'tcur0'), seq_peram(peram_u_tcur0_tsink))
        # OPTIONAL_N_GEVP_BEGIN
        # PR.register('light', ('tcur0', 'tsink'), peram_u_tcur0_tsink)
        # OPTIONAL_N_GEVP_END
        for Mom_indx in range(Mom_len):
            VR.register('VDV_0', 'tcur0', VdV_link[tcur0, Mom_indx:Mom_indx+1])
            for i in range(len(sink_operator)):
                # OPTIONAL_N_GEVP_BEGIN
                # if i > 0 and Mom_indx == 0:
                #     continue  # p_hat is undefined for the two rest admixtures.
                # OPTIONAL_N_GEVP_END
                VR.register('VVV_0', 'tsink', VVV_sink[Mom_indx:Mom_indx+1])
                Vindex = ['N', 'N', 'M', 'M']
                # OPTIONAL_N_GEVP_BEGIN
                # if i == 1:
                #     VR.register('VDV_0', 'tsink', VDV_sink[0:1])
                # elif i == 2:
                #     VR.register('VDV_0', 'tsink', VDV_sink[Mom_indx:Mom_indx+1])
                #     VR.register('VVV_0', 'tsink', VVV_sink[0:1])
                # if i > 0:
                #     Vindex = ['N', 'N', 'N', 'M', 'M']
                # OPTIONAL_N_GEVP_END
                operator_groups = [[s, t, current_operator] for s in sink_operator[i] for t in source_operator[0]]
                for j in range(4):
                    GR.register('gamma_1', gamma(j+1)@gamma(5))
                    dc = dynamic_contraction(
                        operator_groups,
                        peram_registry=PR, v_registry=VR, gamma_registry=GR,
                        Cpt=Cpt, Vindex=Vindex,
                        use_equivalence=use_equivalence, ignore_dis=ignore_dis,
                        Projection=Projection, optimize=optimize, verbose=False)
                    corr = dc.calculate_all()  # (sN,sNpi,p_block=1,r=27)
                    # OPTIONAL_N_GEVP_BEGIN
                    # if i > 0:
                    #     p = Mom_sink_VVV[Mom_indx]
                    #     sigma_p = sum(p[2-a]*Sigma_DR[a] for a in range(3))/np.linalg.norm(p)
                    #     corr = cached_contract('ab,bcNM->acNM', sigma_p, corr)
                    # OPTIONAL_N_GEVP_END
                    corr_NJN_GEVP[i, tsrc, tcur0, j, ..., Mom_indx:Mom_indx+1, :] = corr
    backend.cuda.get_current_stream().synchronize()
    comm.Barrier()
    if rank == 0:
        print(f'TIMING contraction tsrc={tsrc} seconds={time.perf_counter()-st_cal:.6f}', flush=True)

# M8 Final-only output: no per-source/per-insertion checkpoints or raw dumps.
# Both endpoint operators contain odd fermion number; same-time pairs change
# crossing count by two. A complete temporal wrap therefore gives -1 once.
comm.Barrier()
corr_NJN_GEVP = get_mpi_data(corr_NJN_GEVP, mdtype='TGather', root=0, axis=2)
if rank == 0:
    import pathlib
    corr_save_path = pathlib.Path(f'/public/home/sush/distillation/0v2b/result_nex/E32P29/Px0Py0Pz0/ENV_{Nev_src}/conf{conf_id}')
    corr_save_path.mkdir(parents=True, exist_ok=True)
    corr_sum = loop_tsrc(corr_NJN_GEVP, indx=[1,2], Boundary_Conditions='Antiperiodic', Ctype='3pt', t_sep=tsep)
    n_basis_tag = 'Ndiag' if len(sink_operator) == 1 else 'NGEVP'
    output_stem = f'corr_3pt_NJN_I0.5_{n_basis_tag}_tsep{tsep}'
    backend.save(corr_save_path/f'{output_stem}.npy', corr_sum)
    np.savez(corr_save_path/f'{output_stem}_basis.npz',
        axes=['N_basis','tsrc_sum','tau','mu','spin_N','spin_Npi','p','r'],
        initial_isospin=0.5, initial_Iz=0.5, final_isospin=0.5, final_Iz=-0.5,
        initial_basis=['p*ubar5u','p*dbar5d','n*dbar5u'],
        initial_coefficients=[np.sqrt(1/6),-np.sqrt(1/6),np.sqrt(2/3)],
        current_flavor='dbar u', current_convention='Euclidean gamma_mu @ gamma5; charged isovector',
        mu=[1,2,3,4], p=Mom_sink_VVV, r=Mom_sink_VDV,
        momentum_component_order=['pz','py','px'],
        N_basis=['N(p)','Sigma_DR.p_hat*N(p)pi(0)','Sigma_DR.p_hat*N(0)pi(p)'][:len(sink_operator)],
        N_side_GEVP_data=(len(sink_operator)>1), N_admixture_spin_convention='Dirac-rotation',
        Q_normalization='Q=P_plus[:,:2]; QdagQ=I_2/2; no spin trace yet',
        source_times=np.arange(Nt), source_time_count=Nt, source_time_reduction='sum',
        insertion_time_offsets=np.arange(5,tsep-3), tsep=tsep,
        baryon_boundary_condition='Antiperiodic; source-wrap phase applied once',
        current_renormalized=False, weak_coupling_included=False,
        ignore_dis=ignore_dis, use_equivalence=use_equivalence, eigenvectors=Nev_src,
        current_eigenvectors=Nev_max, mpi_ranks=size)
    print(f'SAVED {corr_save_path/output_stem}.npy and _basis.npz', flush=True)
    print(f'TIMING total_seconds={time.perf_counter()-st_total:.6f}', flush=True)
