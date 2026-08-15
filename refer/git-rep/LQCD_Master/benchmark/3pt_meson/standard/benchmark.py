# Run: mpirun -np 4 python benchmark.py <resource_path> <cfg_num>
import gc
import os
import sys

import cupy as cp
import numpy as np
from opt_einsum import contract
from pyquda_utils import core, gamma, io, source

resource_path = sys.argv[1]
cfg_num = sys.argv[2]

# Ensemble / solver parameters
latt_size = [24, 24, 24, 72]
grid_size = [1, 1, 1, 4]
cfg_path_template = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"

m_l = -0.2770
m_s = -0.2356
m_c = 0.4159
m_b = 1.5
xi_0 = 1.0
csw = 1.160920226
tol_l = 1.0e-10
tol_s = 1.0e-10
tol_c = 1.0e-14
tol_b = 1.0e-14
maxiter = 10000
multigrid = [[6, 6, 6, 3], [4, 4, 4, 6]]

stout_nstep = 1
stout_rho = 0.125
stout_ndim = 4

source_position = [0, 0, 0, 0]
t_sink = 8

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "benchmark")
os.makedirs(out_dir, exist_ok=True)

tasks = [
	{
		"id": 1,
		"label": "D0_to_Kminus",
		"process": "D0 -> K-",
		"source_particle": "D0",
		"sink_particle": "K-",
		"spectator": "l",
		"source_active": "c",
		"sink_active": "s",
		"gamma_src": "gamma5",
		"gamma_snk": "gamma5",
		"current": "c_to_s",
	},
	{
		"id": 2,
		"label": "D0_to_piminus",
		"process": "D0 -> pi-",
		"source_particle": "D0",
		"sink_particle": "pi-",
		"spectator": "l",
		"source_active": "c",
		"sink_active": "l",
		"gamma_src": "gamma5",
		"gamma_snk": "gamma5",
		"current": "c_to_l",
	},
	{
		"id": 3,
		"label": "Bminus_to_D0",
		"process": "B- -> D0",
		"source_particle": "B-",
		"sink_particle": "D0",
		"spectator": "l",
		"source_active": "b",
		"sink_active": "c",
		"gamma_src": "gamma5",
		"gamma_snk": "gamma5",
		"current": "b_to_c",
	},
	{
		"id": 4,
		"label": "Bminus_to_Kminus",
		"process": "B- -> K-",
		"source_particle": "B-",
		"sink_particle": "K-",
		"spectator": "l",
		"source_active": "b",
		"sink_active": "s",
		"gamma_src": "gamma5",
		"gamma_snk": "gamma5",
		"current": "b_to_s",
	},
	{
		"id": 5,
		"label": "Bminus_to_piminus",
		"process": "B- -> pi-",
		"source_particle": "B-",
		"sink_particle": "pi-",
		"spectator": "l",
		"source_active": "b",
		"sink_active": "l",
		"gamma_src": "gamma5",
		"gamma_snk": "gamma5",
		"current": "b_to_l",
	},
	{
		"id": 6,
		"label": "D0_to_Kstarminus",
		"process": "D0 -> K*-",
		"source_particle": "D0",
		"sink_particle": "K*-",
		"spectator": "l",
		"source_active": "c",
		"sink_active": "s",
		"gamma_src": "gamma5",
		"gamma_snk": "gammax",
		"current": "c_to_s",
	},
	{
		"id": 7,
		"label": "Bminus_to_Kstarminus",
		"process": "B- -> K*-",
		"source_particle": "B-",
		"sink_particle": "K*-",
		"spectator": "l",
		"source_active": "b",
		"sink_active": "s",
		"gamma_src": "gamma5",
		"gamma_snk": "gammax",
		"current": "b_to_s",
	},
	{
		"id": 8,
		"label": "Dsplus_to_phi",
		"process": "Ds+ -> phi",
		"source_particle": "Ds+",
		"sink_particle": "phi",
		"spectator": "s",
		"source_active": "c",
		"sink_active": "s",
		"gamma_src": "gamma5",
		"gamma_snk": "gammax",
		"current": "c_to_s",
	},
	{
		"id": 9,
		"label": "Dsplus_to_Dsplus_em",
		"process": "Ds+ -> Ds+",
		"source_particle": "Ds+",
		"sink_particle": "Ds+",
		"spectator": "s",
		"source_active": "c",
		"sink_active": "c",
		"gamma_src": "gamma5",
		"gamma_snk": "gamma5",
		"current": "c_to_c",
	},
	{
		"id": 10,
		"label": "antiBs0_to_Dsplus",
		"process": "anti-Bs0 -> Ds+",
		"source_particle": "anti-Bs0",
		"sink_particle": "Ds+",
		"spectator": "s",
		"source_active": "b",
		"sink_active": "c",
		"gamma_src": "gamma5",
		"gamma_snk": "gamma5",
		"current": "b_to_c",
	},
	{
		"id": 11,
		"label": "Kminus_to_piminus",
		"process": "K- -> pi-",
		"source_particle": "K-",
		"sink_particle": "pi-",
		"spectator": "l",
		"source_active": "s",
		"sink_active": "l",
		"gamma_src": "gamma5",
		"gamma_snk": "gamma5",
		"current": "s_to_l",
	},
	{
		"id": 12,
		"label": "D0_to_etau",
		"process": "D0 -> eta_u",
		"source_particle": "D0",
		"sink_particle": "eta_u",
		"spectator": "l",
		"source_active": "c",
		"sink_active": "l",
		"gamma_src": "gamma5",
		"gamma_snk": "gamma5",
		"current": "c_to_l",
	},
	{
		"id": 13,
		"label": "Bcminus_to_Jpsi",
		"process": "Bc- -> Jpsi",
		"source_particle": "Bc-",
		"sink_particle": "Jpsi",
		"spectator": "c",
		"source_active": "b",
		"sink_active": "c",
		"gamma_src": "gamma5",
		"gamma_snk": "gammax",
		"current": "b_to_c",
	},
]

core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=cfg_num)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gx = cp.asarray(gamma.gamma(1), dtype=cp.complex128)

dirac_l = core.getDirac(latt_info, m_l, tol_l, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getDirac(latt_info, m_s, tol_s, maxiter, xi_0, csw, csw, multigrid)
dirac_c = core.getDirac(latt_info, m_c, tol_c, maxiter, xi_0, csw, csw, multigrid)
dirac_b = core.getDirac(latt_info, m_b, tol_b, maxiter, xi_0, csw, csw, multigrid)

pt_src = source.source12(latt_info, "point", source_position)

with dirac_l.useGauge(gauge_stout):
	propagator_l = core.invertPropagator(dirac_l, pt_src)

with dirac_s.useGauge(gauge_stout):
	propagator_s = core.invertPropagator(dirac_s, pt_src)

with dirac_c.useGauge(gauge_stout):
		propagator_c = core.invertPropagator(dirac_c, pt_src)

with dirac_b.useGauge(gauge_stout):
		propagator_b = core.invertPropagator(dirac_b, pt_src)

propagator_map = {
	"l": propagator_l,
	"u": propagator_l,
	"d": propagator_l,
	"s": propagator_s,
		"c": propagator_c,
		"b": propagator_b,
}

dirac_map = {
	"l": dirac_l,
	"u": dirac_l,
	"d": dirac_l,
	"s": dirac_s,
		"c": dirac_c,
		"b": dirac_b,
}

gamma_map = {
	"gamma5": G5,
	"gammax": Gx,
}

# current_gamma_list = [
# 	("Gx", Gx),
# 	("G5", G5),
# ]
current_gamma_list = [
	("Gx", gamma.gamma(1)),
	("GxG5", gamma.gamma(14)),
]

for task in tasks:
	spectator_prop = propagator_map[task["spectator"]]
	source_prop = propagator_map[task["source_active"]]
	sink_dirac = dirac_map[task["sink_active"]]
	gamma_src = gamma_map[task["gamma_src"]]
	gamma_snk = gamma_map[task["gamma_snk"]]

	gamma_src_bar = contract("ab,cb,cd -> ad", G5, gamma_src.conj(), G5)
	gamma_snk_bar = contract("ab,cb,cd -> ad", G5, gamma_snk.conj(), G5)

	sink_block = core.LatticePropagator(latt_info)
	sink_block.data = contract(
		"AB, wtzyxBCba, CD -> wtzyxADba",
		gamma_snk_bar,
		spectator_prop.data,
		gamma_src_bar,
	)

	src_seq = source.sequential12(sink_block, t_sink)

	with sink_dirac.useGauge(gauge_stout):
		prop_seq = core.invertPropagator(sink_dirac, src_seq)

	prop_seq_dag = core.LatticePropagator(latt_info)
	prop_seq_dag.data = contract(
		"AB, wtzyxCBab, CD -> wtzyxADba",
		G5,
		prop_seq.data.conj(),
		G5,
	)

	for current_label, gamma_cur in current_gamma_list:
		three_pt_site = contract(
			"wtzyxABba, BC, wtzyxCAab -> wtzyx",
			prop_seq_dag.data,
			gamma_cur,
			source_prop.data,
		)

		three_pt_local = contract("wtzyx -> t", three_pt_site)
		c3_t = core.gatherLattice(three_pt_local.get(), [0, -1, -1, -1])

		if core.getMPIRank() == 0:
			c3_t = np.asarray(c3_t, dtype=np.complex128).reshape(-1)
			t = np.arange(c3_t.shape[0], dtype=np.int32)
			out = np.column_stack((t, c3_t.real, c3_t.imag))
			out_path = os.path.join(
				out_dir,
				f"task_{task['id']:02d}_{task['label']}_current_{current_label}_cfg{cfg_num}.txt",
			)
			np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])

		del three_pt_site
		del three_pt_local

	del sink_block
	del src_seq
	del prop_seq
	del prop_seq_dag

del pt_src
del propagator_l
del propagator_s
del propagator_c
del propagator_b
gc.collect()
cp.get_default_memory_pool().free_all_blocks()
