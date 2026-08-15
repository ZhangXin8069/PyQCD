# pyright: reportAttributeAccessIssue=false
"""Run: mpirun -np 4 python benchmark.py <resource_path> <cfg_num>"""

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
cfg_path_template = (
	"/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/"
	"beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_{n_cfg}.lime"
)

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

# template:
# - proton_sink: p -> p
# - lambda_sink: sink epsilon(u^T Cg5 d) q type
# - lambda_to_proton_sink: Lambda-like -> proton-like
# - xi_to_lambda_sink: Xi-like -> Lambda-like
# - xic_to_xi_sink: Xi_c -> Xi (two sink topologies)
# - xicc_to_xic_sink: Xi_cc++ -> Xi_c+ (special topology from validated reference)
tasks = [
	{
		"id": 1,
		"label": "proton_to_proton_u_to_u_vector",
		"template": "proton_sink",
		"sink_active": "l",
		"source_active": "l",
		"current": "vector",
	},
	{
		"id": 2,
		"label": "proton_to_proton_u_to_u_axial",
		"template": "proton_sink",
		"sink_active": "l",
		"source_active": "l",
		"current": "axial",
	},
	{
		"id": 3,
		"label": "lambda_to_lambda_s_to_s_vector",
		"template": "lambda_sink",
		"sink_active": "s",
		"source_active": "s",
		"current": "vector",
	},
	{
		"id": 4,
		"label": "lambda_to_lambda_s_to_s_axial",
		"template": "lambda_sink",
		"sink_active": "s",
		"source_active": "s",
		"current": "axial",
	},
	{
		"id": 5,
		"label": "lambda_to_proton_s_to_u_vector",
		"template": "lambda_to_proton_sink",
		"sink_active": "l",
		"source_active": "s",
		"current": "vector",
	},
	{
		"id": 6,
		"label": "lambdac_to_lambda_c_to_s_vector",
		"template": "lambda_sink",
		"sink_active": "s",
		"source_active": "c",
		"current": "vector",
	},
	{
		"id": 7,
		"label": "xic_to_xi_c_to_s_vector",
		"template": "xic_to_xi_sink",
		"sink_active": "s",
		"source_active": "c",
		"current": "vector",
	},
	{
		"id": 8,
		"label": "lambdab_to_lambdac_b_to_c_vector",
		"template": "lambda_sink",
		"sink_active": "c",
		"source_active": "b",
		"current": "vector",
	},
	{
		"id": 9,
		"label": "lambdab_to_lambda_b_to_s_vector",
		"template": "lambda_sink",
		"sink_active": "s",
		"source_active": "b",
		"current": "vector",
	},
	{
		"id": 10,
		"label": "lambdab_to_proton_b_to_u_vector",
		"template": "lambda_to_proton_sink",
		"sink_active": "l",
		"source_active": "b",
		"current": "vector",
	},
	{
		"id": 11,
		"label": "xi_to_lambda_s_to_u_vector",
		"template": "xi_to_lambda_sink",
		"sink_active": "l",
		"source_active": "s",
		"current": "vector",
	},
	{
		"id": 12,
		"label": "lambda_to_proton_s_to_u_axial",
		"template": "lambda_to_proton_sink",
		"sink_active": "l",
		"source_active": "s",
		"current": "axial",
	},
	{
		"id": 13,
		"label": "xicc_to_xic_c_to_d_vector",
		"template": "xicc_to_xic_sink",
		"sink_active": "s",
		"source_active": "c",
		"current": "vector",
	},
	{
		"id": 14,
		"label": "lambdab_to_lambdac_b_to_c_axial",
		"template": "lambda_sink",
		"sink_active": "c",
		"source_active": "b",
		"current": "axial",
	},
	{
		"id": 15,
		"label": "xi_to_lambda_s_to_u_axial",
		"template": "xi_to_lambda_sink",
		"sink_active": "l",
		"source_active": "s",
		"current": "axial",
	},
]


core.init(grid_size, latt_size, backend="cupy", resource_path=resource_path)
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)

cfg_path = cfg_path_template.format(n_cfg=cfg_num)
gauge = io.readChromaQIOGauge(cfg_path)
gauge.toDevice()
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(stout_nstep, stout_rho, stout_ndim)

I4 = cp.eye(4).astype(cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Gt = cp.asarray(gamma.gamma(8), dtype=cp.complex128)
Gx = cp.asarray(gamma.gamma(1), dtype=cp.complex128)
GxG5 = Gx @ G5
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = 0.5 * (I4 + Gt)

eps = cp.zeros((3, 3, 3)).astype(cp.complex128)
eps[0, 1, 2] = 1.0
eps[1, 2, 0] = 1.0
eps[2, 0, 1] = 1.0
eps[0, 2, 1] = -1.0
eps[2, 1, 0] = -1.0
eps[1, 0, 2] = -1.0

dirac_l = core.getDirac(latt_info, m_l, tol_l, maxiter, xi_0, csw, csw, multigrid)
dirac_s = core.getDirac(latt_info, m_s, tol_s, maxiter, xi_0, csw, csw, multigrid)
dirac_c = core.getDirac(latt_info, m_c, tol_c, maxiter, xi_0, csw, csw, multigrid)
dirac_b = core.getDirac(latt_info, m_b, tol_b, maxiter, xi_0, csw, csw, multigrid)

pt_src = source.source12(latt_info, "point", source_position)

with dirac_l.useGauge(gauge_stout):
	prop_l = core.invertPropagator(dirac_l, pt_src)
with dirac_s.useGauge(gauge_stout):
	prop_s = core.invertPropagator(dirac_s, pt_src)
with dirac_c.useGauge(gauge_stout):
	prop_c = core.invertPropagator(dirac_c, pt_src)
with dirac_b.useGauge(gauge_stout):
	prop_b = core.invertPropagator(dirac_b, pt_src)

prop_map = {
	"l": prop_l,
	"u": prop_l,
	"d": prop_l,
	"s": prop_s,
	"c": prop_c,
	"b": prop_b,
}
dirac_map = {
	"l": dirac_l,
	"u": dirac_l,
	"d": dirac_l,
	"s": dirac_s,
	"c": dirac_c,
	"b": dirac_b,
}

ones_phase = cp.ones(prop_l.data.shape[:5]).astype(cp.complex128)

for task in tasks:
	source_prop = prop_map[task["source_active"]]
	sink_dirac = dirac_map[task["sink_active"]]
	gamma_cur = Gx if task["current"] == "vector" else GxG5

	B = core.LatticePropagator(latt_info)

	if task["template"] == "proton_sink":
		B.data = (
			-contract(
				"abc,def,AB,EF,DC,wtzyxBEbe,wtzyxAFaf->wtzyxDCdc",
                eps,
                eps,
				Cg5,
				Cg5,
                Tmat,
				prop_l.data,
				prop_l.data,
			)
			+contract(
				"abc,def,AB,EF,DC,wtzyxBEbe,wtzyxCFcf->wtzyxDAda",
                eps,
                eps,
				Cg5,
				Cg5,
                Tmat,
				prop_l.data,
				prop_l.data,
			)
			+contract(
				"abc,def,AB,EF,DC,wtzyxBEbe,wtzyxADad->wtzyxFCfc",
                eps,
                eps,
				Cg5,
				Cg5,
                Tmat,
				prop_l.data,
				prop_l.data,
			)
			-contract(
				"abc,def,AB,EF,DC,wtzyxBEbe,wtzyxCDcd->wtzyxFAfa",
                eps,
                eps,
				Cg5,
				Cg5,
                Tmat,
				prop_l.data,
				prop_l.data,
			)
		)

	elif task["template"] == "lambda_sink":
		B.data = -contract(
				"abc,def,AB,EF,DC,wtzyxBEbe,wtzyxAFaf->wtzyxDCdc",
                eps,
                eps,
				Cg5,
				Cg5,
                Tmat,
				prop_l.data,
				prop_l.data,
			)

	elif task["template"] == "lambda_to_proton_sink":
		B.data = (
			-contract(
				"abc,def,AB,EF,DC,wtzyxBEbe,wtzyxAFaf->wtzyxDCdc",
                eps,
                eps,
				Cg5,
				Cg5,
                Tmat,
				prop_l.data,
				prop_l.data,
			)
			+contract(
				"abc,def,AB,EF,DC,wtzyxBEbe,wtzyxCFcf->wtzyxDAda",
                eps,
                eps,
				Cg5,
				Cg5,
                Tmat,
				prop_l.data,
				prop_l.data,
			)
		)

	elif task["template"] == "xi_to_lambda_sink":
		B.data = (
			-contract(
				"abc,def,AB,EF,DC,wtzyxCEce,wtzyxBFbf->wtzyxDAda",
                eps,
                eps,
				Cg5,
				Cg5,
                Tmat,
				prop_s.data,
				prop_l.data,
			)
			+contract(
				"abc,def,AB,EF,DC,wtzyxCDcd,wtzyxBFbf->wtzyxEAea",
                eps,
                eps,
				Cg5,
				Cg5,
                Tmat,
				prop_s.data,
				prop_l.data,
			)
		)

	elif task["template"] == "xic_to_xi_sink":
		B.data = (
			-contract(
				"abc,def,AB,EF,DC,wtzyxBEbe,wtzyxAFaf->wtzyxDCdc",
                eps,
                eps,
				Cg5,
				Cg5,
                Tmat,
				prop_s.data,
				prop_l.data,
			)
			+contract(
				"abc,def,AB,EF,DC,wtzyxCEce,wtzyxAFaf->wtzyxDBdb",
                eps,
                eps,
				Cg5,
				Cg5,
                Tmat,
				prop_s.data,
				prop_l.data,
			)
		)

	elif task["template"] == "xicc_to_xic_sink":
		B.data = (
			-contract(
				"abc,def,AB,EF,DC,wtzyxAFaf,wtzyxCDcd->wtzyxEBeb",
                eps,
                eps,
				Cg5,
				Cg5,
                Tmat,
				prop_l.data,
				prop_c.data,
			)
			+contract(
				"abc,def,AB,EF,DC,wtzyxAFaf,wtzyxCEce->wtzyxDBdb",
                eps,
                eps,
				Cg5,
				Cg5,
                Tmat,
				prop_l.data,
				prop_c.data,
			)
		)

	else:
		raise ValueError(f"Unsupported template: {task['template']}")

	# Two-dagger sequential-source convention used in existing baryon scripts.
	B.data = -contract("AB, wtzyxCBji, CD -> wtzyxADij", G5, B.data.conj(), G5)
    #Negative sign is due to sign convention
	src_seq = source.sequential12(B, t_sink)

	with sink_dirac.useGauge(gauge_stout):
		prop_seq = core.invertPropagator(sink_dirac, src_seq)

	prop_seq_dag = core.LatticePropagator(latt_info)
	prop_seq_dag.data = contract("AB, wtzyxCBji, CD -> wtzyxADij", G5, prop_seq.data.conj(), G5)

	three_pt_local = contract(
		"wtzyxijba, jk, wtzyxkiab -> t",
		prop_seq_dag.data,
		gamma_cur,
		source_prop.data,
	)

	c3_t = core.gatherLattice(three_pt_local.get(), [0, -1, -1, -1])

	if core.getMPIRank() == 0:
		c3_t = np.asarray(c3_t, dtype=np.complex128).reshape(-1)
		t = np.arange(c3_t.shape[0], dtype=np.int32)
		out = np.column_stack((t, c3_t.real, c3_t.imag))
		out_path = os.path.join(out_dir, f"task_{task['id']:02d}_{task['label']}_cfg{cfg_num}.txt")
		np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])


del pt_src
del prop_l
del prop_s
del prop_c
del prop_b
gc.collect()
cp.get_default_memory_pool().free_all_blocks()
