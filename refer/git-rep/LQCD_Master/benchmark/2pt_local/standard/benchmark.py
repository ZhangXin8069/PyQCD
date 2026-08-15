# Run: mpirun -np 4 python benchmark.py <resource_path> <cfg_num>
import gc
import os
import sys

import cupy as cp
import numpy as np
from opt_einsum import contract
from pyquda_comm import array
from pyquda_utils import core, gamma, io, source

resource_path = sys.argv[1]
cfg_num = sys.argv[2]

# Ensemble / solver parameters
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = '/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime'

m_l = -0.2770
m_s = -0.2356
m_c = 0.4159
xi_0 = 1.0
csw = 1.160920226
tol_l = 1.0e-10
tol_s = 1.0e-10
tol_c = 1.0e-14
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, 'benchmark')
os.makedirs(out_dir, exist_ok=True)

tasks = [
    {
        'id': 1,
        'label': 'pion',
        'particle': 'pion',
        'source': [0, 0, 0, 0],
        'flavors': ['l'],
        'operator': 'pi_plus_local_ps',
        'compute': 'pion',
        'task_note': 'task_1 point source',
    },
    {
        'id': 2,
        'label': 'kaon',
        'particle': 'kaon',
        'source': [0, 0, 0, 0],
        'flavors': ['l', 's'],
        'operator': 'kaon_local_ps',
        'compute': 'kaon',
        'task_note': 'task_2 point source',
    },
    {
        'id': 3,
        'label': 'eta_s',
        'particle': 'eta_s',
        'source': [0, 0, 0, 0],
        'flavors': ['s'],
        'operator': 'eta_s_local_ps',
        'compute': 'eta_s',
        'task_note': 'task_3 point source',
    },
    {
        'id': 4,
        'label': 'eta_c',
        'particle': 'eta_c',
        'source': [0, 0, 0, 0],
        'flavors': ['c'],
        'operator': 'eta_c_local_ps',
        'compute': 'eta_c',
        'task_note': 'task_4 point source',
    },
    {
        'id': 5,
        'label': 'rho',
        'particle': 'rho',
        'source': [0, 0, 0, 0],
        'flavors': ['l'],
        'operator': 'rho_local_vector_avg_spatial',
        'compute': 'rho',
        'task_note': 'task_5 spatial gamma average',
    },
    {
        'id': 6,
        'label': 'D',
        'particle': 'D_plus',
        'source': [0, 0, 0, 0],
        'flavors': ['l', 'c'],
        'operator': 'D_plus_local_ps',
        'compute': 'D',
        'task_note': 'task_6 point source',
    },
    {
        'id': 7,
        'label': 'D_s',
        'particle': 'D_s',
        'source': [0, 0, 0, 0],
        'flavors': ['s', 'c'],
        'operator': 'D_s_local_ps',
        'compute': 'D_s',
        'task_note': 'task_7 point source',
    },
    {
        'id': 8,
        'label': 'Jpsi',
        'particle': 'Jpsi',
        'source': [0, 0, 0, 0],
        'flavors': ['c'],
        'operator': 'Jpsi_local_vector_avg_spatial',
        'compute': 'Jpsi',
        'task_note': 'task_8 spatial gamma average',
    },
    {
        'id': 9,
        'label': 'proton',
        'particle': 'proton',
        'source': [0, 0, 0, 0],
        'flavors': ['l'],
        'operator': 'proton_local_Pplus',
        'compute': 'proton',
        'task_note': 'task_9 Gaussian smearing disabled in benchmark',
    },
    {
        'id': 10,
        'label': 'Lambda',
        'particle': 'Lambda',
        'source': [0, 0, 0, 0],
        'flavors': ['l', 's'],
        'operator': 'Lambda_local_Pplus',
        'compute': 'lambda',
        'task_note': 'task_10 Gaussian smearing disabled in benchmark',
    },
    {
        'id': 11,
        'label': 'Xi_minus',
        'particle': 'Xi_minus',
        'source': [0, 0, 0, 0],
        'flavors': ['l', 's'],
        'operator': 'Xi_minus_local_Pplus',
        'compute': 'xi_minus',
        'task_note': 'task_11 point source',
    },
    {
        'id': 12,
        'label': 'Sigma_plus',
        'particle': 'Sigma_plus',
        'source': [0, 0, 0, 0],
        'flavors': ['l', 's'],
        'operator': 'Sigma_plus_local_Pplus',
        'compute': 'sigma_plus',
        'task_note': 'task_12 Cg5 interpolator note kept from task text',
    },
    {
        'id': 13,
        'label': 'Sigma_minus',
        'particle': 'Sigma_minus',
        'source': [0, 0, 0, 0],
        'flavors': ['l', 's'],
        'operator': 'Sigma_minus_local_Pplus',
        'compute': 'sigma_minus',
        'task_note': 'task_13 Cg5 interpolator note kept from task text',
    },
    {
        'id': 14,
        'label': 'Lambda_c',
        'particle': 'Lambda_c',
        'source': [0, 0, 0, 0],
        'flavors': ['l', 'c'],
        'operator': 'Lambda_c_local_Pplus',
        'compute': 'lambda_c',
        'task_note': 'task_14 Gaussian smearing disabled in benchmark',
    },
    {
        'id': 15,
        'label': 'Xi_c_plus',
        'particle': 'Xi_c_plus',
        'source': [0, 0, 0, 0],
        'flavors': ['l', 's', 'c'],
        'operator': 'Xi_c_plus_local_Pplus',
        'compute': 'xi_c_plus',
        'task_note': 'task_15 point source',
    },
    {
        'id': 16,
        'label': 'Xi_c_zero',
        'particle': 'Xi_c_zero',
        'source': [0, 0, 0, 0],
        'flavors': ['l', 's', 'c'],
        'operator': 'Xi_c_zero_local_Pplus',
        'compute': 'xi_c_zero',
        'task_note': 'task_16 point source',
    },
    {
        'id': 17,
        'label': 'Omega_c_zero',
        'particle': 'Omega_c_zero',
        'source': [0, 0, 0, 0],
        'flavors': ['s', 'c'],
        'operator': 'Omega_c_zero_local_Pplus_symmetric',
        'compute': 'omega_c_zero',
        'task_note': 'task_17 uses comparison.py symmetric charmed-baryon contraction',
    },
    {
        'id': 18,
        'label': 'Sigma_c_plus',
        'particle': 'Sigma_c_plus',
        'source': [0, 0, 0, 0],
        'flavors': ['l', 'c'],
        'operator': 'Sigma_c_plus_local_Pplus_symmetric',
        'compute': 'sigma_c_plus',
        'task_note': 'task_18 uses comparison.py symmetric charmed-baryon contraction',
    },
    {
        'id': 19,
        'label': 'Xi_cc',
        'particle': 'Xi_cc',
        'source': [0, 0, 0, 0],
        'flavors': ['l', 'c'],
        'operator': 'Xi_cc_local_Pplus_symmetric',
        'compute': 'Xi_cc',
        'task_note': 'task_19 uses comparison.py symmetric charmed-baryon contraction',
    },
        {
        'id': 20,
        'label': 'Omega_cc',
        'particle': 'Omega_cc',
        'source': [0, 0, 0, 0],
        'flavors': ['s', 'c'],
        'operator': 'Omega_cc_local_Pplus_symmetric',
        'compute': 'Omega_cc',
        'task_note': 'task_20 uses comparison.py symmetric charmed-baryon contraction',
    },
]


core.init(grid_size, latt_size, backend='cupy', resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=cfg_num)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

epsilon = cp.zeros((3, 3, 3))
epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
epsilon[2, 1, 0] = epsilon[1, 0, 2] = epsilon[0, 2, 1] = -1

C = cp.asarray(gamma.gamma(2)) @ cp.asarray(gamma.gamma(8))
G0 = cp.asarray(gamma.gamma(0))
Gx = cp.asarray(gamma.gamma(1))
Gy = cp.asarray(gamma.gamma(2))
Gz = cp.asarray(gamma.gamma(4))
Gt = cp.asarray(gamma.gamma(8))
G5 = cp.asarray(gamma.gamma(15))
Tmat = (G0 + Gt) / 2
vector_gamma_list = [Gx, Gy, Gz]
# symmetric_gamma_list = [Gx, Gy, Gz, Gt]
symmetric_gamma_list = [Gx]

dirac_l = core.getDirac(latt_info, m_l, tol_l, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getDirac(latt_info, m_s, tol_s, maxiter, xi_0, csw, csw, multigrid)
dirac_c = core.getDirac(latt_info, m_c, tol_c, maxiter, xi_0, csw, csw, multigrid)

pt_src = source.source12(latt_info, "point", [0,0,0,0])
with dirac_l.useGauge(gauge_stout):
    propagator_l = core.invertPropagator(dirac_l, pt_src)
with dirac_s.useGauge(gauge_stout):
    propagator_s = core.invertPropagator(dirac_s, pt_src)
with dirac_c.useGauge(gauge_stout):
    propagator_c = core.invertPropagator(dirac_c, pt_src)

for task in tasks:
    mode = task['compute']

    if mode == 'pion':
        c2_local = contract('wtzyxabij,wtzyxabij->t', propagator_l.data.conj(), propagator_l.data)
    elif mode == 'kaon':
        c2_local = contract('wtzyxabij,wtzyxabij->t', propagator_s.data.conj(), propagator_l.data)
    elif mode == 'eta_s':
        c2_local = contract('wtzyxabij,wtzyxabij->t', propagator_s.data.conj(), propagator_s.data)
    elif mode == 'eta_c':
        c2_local = contract('wtzyxabij,wtzyxabij->t', propagator_c.data.conj(), propagator_c.data)
    elif mode == 'rho':
        c2_local = 0
        for gamma_i in vector_gamma_list:
            c2_local += contract(
                'ab,cd,wtzyxadij,wtzyxbcij->t',
                G5 @ gamma_i,
                gamma_i @ G5,
                propagator_l.data.conj(),
                propagator_l.data,
            )
        c2_local /= len(vector_gamma_list)
    elif mode == 'D':
        c2_local = contract('wtzyxabij,wtzyxabij->t', propagator_l.data.conj(), propagator_c.data)
    elif mode == 'D_s':
        c2_local = contract('wtzyxabij,wtzyxabij->t', propagator_s.data.conj(), propagator_c.data)
    elif mode == 'Jpsi':
        c2_local = 0
        for gamma_i in vector_gamma_list:
            c2_local += contract(
                'ab,cd,wtzyxadij,wtzyxbcij->t',
                G5 @ gamma_i,
                gamma_i @ G5,
                propagator_c.data.conj(),
                propagator_c.data,
            )
        c2_local /= len(vector_gamma_list)
    elif mode == 'proton':
        c2_local = +contract(
            'abc,def,AB,EF,DC,wtzyxBEbe,wtzyxCDcd,wtzyxAFaf->t',
            epsilon,
            epsilon,
            C @ G5,
            C @ G5,
            Tmat,
            propagator_l.data,
            propagator_l.data,
            propagator_l.data,
        ) - contract(
            'abc,def,AB,EF,DC,wtzyxBEbe,wtzyxCFcf,wtzyxADad->t',
            epsilon,
            epsilon,
            C @ G5,
            C @ G5,
            Tmat,
            propagator_l.data,
            propagator_l.data,
            propagator_l.data,
        )
    elif mode == 'lambda':
        c2_local = +contract(
            'abc,def,AB,EF,DC,wtzyxBEbe,wtzyxCDcd,wtzyxAFaf->t',
            epsilon,
            epsilon,
            C @ G5,
            C @ G5,
            Tmat,
            propagator_l.data,
            propagator_s.data,
            propagator_l.data,
        )
    elif mode == 'xi_minus':
        c2_local = +contract(
            'abc,def,AB,EF,DC,wtzyxBEbe,wtzyxCDcd,wtzyxAFaf->t',
            epsilon,
            epsilon,
            C @ G5,
            C @ G5,
            Tmat,
            propagator_s.data,
            propagator_s.data,
            propagator_l.data,
        ) - contract(
            'abc,def,AB,EF,DC,wtzyxBDbd,wtzyxCEce,wtzyxAFaf->t',
            epsilon,
            epsilon,
            C @ G5,
            C @ G5,
            Tmat,
            propagator_s.data,
            propagator_s.data,
            propagator_l.data,
        )
    elif mode == 'sigma_plus' or mode == 'sigma_minus':
        c2_local = +contract(
            'abc,def,AB,EF,DC,wtzyxBEbe,wtzyxCDcd,wtzyxAFaf->t',
            epsilon,
            epsilon,
            C @ Gx,
            C @ Gx,
            Tmat,
            propagator_l.data,
            propagator_s.data,
            propagator_l.data,
        ) - contract(
            'abc,def,AB,EF,DC,wtzyxBFbf,wtzyxCDcd,wtzyxAEae->t',
            epsilon,
            epsilon,
            C @ Gx,
            C @ Gx,
            Tmat,
            propagator_l.data,
            propagator_s.data,
            propagator_l.data,
        )
    elif mode == 'lambda_c':
        c2_local = +contract(
            'abc,def,AB,EF,DC,wtzyxBEbe,wtzyxCDcd,wtzyxAFaf->t',
            epsilon,
            epsilon,
            C @ G5,
            C @ G5,
            Tmat,
            propagator_l.data,
            propagator_c.data,
            propagator_l.data,
        )
    elif mode == 'xi_c_plus' or mode == 'xi_c_zero':
        c2_local = +contract(
            'abc,def,AB,EF,DC,wtzyxBEbe,wtzyxCDcd,wtzyxAFaf->t',
            epsilon,
            epsilon,
            C @ G5,
            C @ G5,
            Tmat,
            propagator_s.data,
            propagator_c.data,
            propagator_l.data,
        )
    elif mode == 'sigma_c_plus':
        c2_local = 0
        for gamma_mu in symmetric_gamma_list:
            for gamma_nu in symmetric_gamma_list:
                c2_local += +contract(
                            'abc,def,AB,EF,DC,wtzyxBEbe,wtzyxCDcd,wtzyxAFaf->t',
                            epsilon,
                            epsilon,
                            C @ gamma_mu,
                            gamma_nu @ C,
                            # G5 @ gamma_nu @ Tmat @ gamma_mu @ G5,
                            Tmat,
                            propagator_l.data,
                            propagator_c.data,
                            propagator_l.data,)
    elif mode == 'omega_c_zero':
        c2_local = 0
        for gamma_mu in symmetric_gamma_list:
            for gamma_nu in symmetric_gamma_list:
                c2_local += +contract(
                            'abc,def,AB,EF,DC,wtzyxBEbe,wtzyxCDcd,wtzyxAFaf->t',
                            epsilon,
                            epsilon,
                            C @ gamma_mu,
                            gamma_nu @ C,
                            # G5 @ gamma_nu @ Tmat @ gamma_mu @ G5,
                            Tmat,
                            propagator_s.data,
                            propagator_c.data,
                            propagator_s.data,
                        ) - contract(
                            'abc,def,AB,EF,DC,wtzyxBFbf,wtzyxCDcd,wtzyxAEae->t',
                            epsilon,
                            epsilon,
                            C @ gamma_mu,
                            gamma_nu @ C,
                            # G5 @ gamma_nu @ Tmat @ gamma_mu @ G5,
                            Tmat,
                            propagator_s.data,
                            propagator_c.data,
                            propagator_s.data,
                        )
    elif mode == 'Xi_cc':
        c2_local = +contract(
            'abc,def,AB,EF,DC,wtzyxBEbe,wtzyxCDcd,wtzyxAFaf->t',
            epsilon,
            epsilon,
            C @ G5,
            C @ G5,
            Tmat,
            propagator_c.data,
            propagator_c.data,
            propagator_l.data,
        ) - contract(
            'abc,def,AB,EF,DC,wtzyxBDbd,wtzyxCEce,wtzyxAFaf->t',
            epsilon,
            epsilon,
            C @ G5,
            C @ G5,
            Tmat,
            propagator_c.data,
            propagator_c.data,
            propagator_l.data,
        )
    elif mode == 'Omega_cc':
        c2_local = +contract(
            'abc,def,AB,EF,DC,wtzyxBEbe,wtzyxCDcd,wtzyxAFaf->t',
            epsilon,
            epsilon,
            C @ G5,
            C @ G5,
            Tmat,
            propagator_c.data,
            propagator_c.data,
            propagator_s.data,
        ) - contract(
            'abc,def,AB,EF,DC,wtzyxBDbd,wtzyxCEce,wtzyxAFaf->t',
            epsilon,
            epsilon,
            C @ G5,
            C @ G5,
            Tmat,
            propagator_c.data,
            propagator_c.data,
            propagator_s.data,
        )
    else:
        raise ValueError(f"Unsupported task mode: {mode}")

    c2_t = core.gatherLattice(array.arrayAsNumpy(c2_local, backend='cupy'), [0, -1, -1, -1])

    if core.getMPIRank() == 0:
        c2_t = np.asarray(c2_t, dtype=np.complex128).reshape(-1)
        t = np.arange(c2_t.shape[0], dtype=np.int32)
        out = np.column_stack((t, c2_t.real, c2_t.imag))
        out_path = os.path.join(out_dir, f"task_{task['id']:02d}_{task['label']}_cfg{cfg_num}.txt")
        source_text = '[' + ','.join(str(value) for value in task['source']) + ']'
        # header = '\n'.join([
        #     f"cfg_num={cfg_num}",
        #     f"task_id={task['id']}",
        #     f"particle={task['particle']}",
        #     f"source_position={source_text}",
        #     f"operator={task['operator']}",
        #     'source_type=point',
        #     'gaussian_smearing=disabled',
        #     'sink_projection=spatial_sum_zero_momentum',
        #     f"stout_smear=n_steps:{stout_nstep},rho:{stout_rho},ndim:{stout_ndim}",
        #     f"task_note={task['task_note']}",
        #     'columns=t ReC ImC',
        # ])
        # np.savetxt(out_path, out, fmt=['%d', '%.16e', '%.16e'], header=header, comments='# ')
        np.savetxt(out_path, out, fmt=['%d', '%.16e', '%.16e'])

    del c2_local

del pt_src
del propagator_l
del propagator_s
del propagator_c
gc.collect()
cp.get_default_memory_pool().free_all_blocks()
