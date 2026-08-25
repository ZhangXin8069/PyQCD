"""对照用例第三批：阶段4 补充功能（gamma_index/PFF 投影/叉积 σ/perm_comb/
缓存键/切片增强/Peram_truncated/绘图常量/unpol F·F 选项）。"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from harness import Case
from ref_bridge import load_lqcddb, load_donghx
import datalib

SEED = 20260825


def build():
    L = load_lqcddb()
    from lqcddb.constant.gamma_matrix import gamma_index as r_gidx
    from lqcddb.constant.gamma_matrix import PFF_Mom_to_gamma_new as r_pff
    from lqcddb.constant.sigma_matrix import Mom_cross_sigma as _rmcs
    from lqcddb.eigvectors.vertex import vertex_creator as RVC
    from lqcddb.base.base_functions import get_cache_keys as r_gck
    from lqcddb.base.base_functions import ArraySlicer as RAS
    dh_io = load_donghx('input_output_4_cupy.py', ['Peram_truncated'])

    from pyqcd.lattice._gamma import gamma_index as p_gidx
    from pyqcd.lattice._gamma import PFF_Mom_to_gamma_new as p_pffg
    from pyqcd.lattice._sigma import Mom_cross_sigma as _pmcs
    from pyqcd.vertex._vertex import perm_comb as _ppc
    from pyqcd.tools._base import get_cache_keys as p_gck
    from pyqcd.tools._base import cached_contract as p_cc
    from pyqcd.tools._base import ArraySlicer as PAS
    from pyqcd.tools._io import peram_truncated as _ppt
    from pyqcd.analysis import _analyse as AN
    from pyqcd.operator._gluon_ope import gluon_ope_operator_z0 as pq_ope

    cases = []

    def add(cid, desc, rf, pf, **kw):
        cases.append(Case(cid, 'suppl', desc, rf, pf, **kw))

    def r_gamma_index():
        return [r_gidx(L.gamma(i)) for i in range(16)]

    def p_gamma_index():
        from pyqcd.lattice._gamma import gamma as pg
        return [p_gidx(np.asarray(pg(i))) for i in range(16)]

    add('S01', '补充 gamma_index 稀疏分解 i=0..15（P± 越界为双方共同契约边界）',
        r_gamma_index, p_gamma_index, tol=0.0)

    moms = [[0, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 2], [0, 0, 0]]

    moms_nz = [m for m in moms if any(m)]

    def r_pffm():
        return [r_pff(moms_nz, allow_t=False),
                r_pff([[0, 0, 1]], allow_t=True)]

    def p_pffm():
        return [p_pffg([list(m) for m in moms_nz], allow_t=False),
                p_pffg([[0, 0, 1]], allow_t=True)]

    add('S02', '补充 PFF_Mom_to_gamma_new 投影表（±t）', r_pffm, p_pffm)

    mm = [[0, 0, 1], [1, -1, 2]]

    def r_mcs():
        return [_rmcs(list(m), upto4dim=u4) for m in mm
                for u4 in (False, True)]

    def p_mcs():
        return [_pmcs(list(m), upto4dim=u4) for m in mm
                for u4 in (False, True)]

    add('S03', '补充 Mom_cross_sigma p×σ 叉积', r_mcs, p_mcs, tol=1e-13)

    rvc = RVC(24)

    def r_pc():
        return [rvc.perm_comb(N=10, M=3, dtype='perm', renormal=False),
                rvc.perm_comb(N=10, M=3, dtype='comb', renormal=False),
                rvc.perm_comb(N=10, M=0, dtype='perm', renormal=True),
                rvc.perm_comb(N=5, M=2, dtype='comb', renormal=True)]

    def p_pc():
        return [_ppc(10, 3, 'perm', False),
                _ppc(10, 3, 'comb', False),
                _ppc(10, 0, 'perm', True),
                _ppc(5, 2, 'comb', True)]

    add('S04', '补充 perm_comb 排列组合数', r_pc, p_pc, tol=1e-12)

    A = np.random.default_rng(SEED).normal(size=(5, 5)) \
        + 1j * np.random.default_rng(SEED).normal(size=(5, 5))

    def r_keys():
        r_gck_clear = getattr(__import__('lqcddb'), 'clear_cache')
        r_gck_clear()
        __import__('lqcddb').cached_contract('ij,jk->ik', A, A)
        __import__('lqcddb').cached_contract('ab,bc->ac', A, A)
        keys = r_gck()
        return sorted(k[0] for k in keys), len(keys)

    def p_keys():
        from pyqcd.tools._base import clear_cache
        clear_cache()
        p_cc('ij,jk->ik', A, A)
        p_cc('ab,bc->ac', A, A)
        keys = p_gck()
        return sorted(k[0] for k in keys), len(keys)

    add('S05', '补充 get_cache_keys 缓存内省', r_keys, p_keys)

    base = np.random.default_rng(SEED + 1).normal(size=(4, 5, 6))

    def r_sl():
        sl = RAS(base.copy())
        gs = sl.get_slices(dims=[-1], indices=[[1, 3]])
        out = [base[gs].shape,
               sl.get_slice_shape(dims=[0], indices=[[2]]),
               dict(sl.get_info())]
        return base[gs].shape, out

    def p_sl():
        sl = PAS(base.copy())
        gs = sl.get_slices(dims=[-1], indices=[[1, 3]])
        out = [base[gs].shape,
               sl.get_slice_shape(dims=[0], indices=[[2]]),
               dict(sl.get_info())]
        return base[gs].shape, out

    def _sl_cmp(a, b):
        return 0.0 if tuple(a) == tuple(b) else float('inf')

    add('S06', '补充 ArraySlicer get_slices/get_slice_shape/get_info',
        r_sl, p_sl, compare=_sl_cmp)

    peram8 = datalib.peram(nev1=8)

    def r_pt():
        return dh_io['Peram_truncated'](peram8.copy())

    def p_pt():
        return _ppt(peram8.copy())

    add('S07', '补充 Peram_truncated 截断（真实 peram）', r_pt, p_pt,
        tol=0.0)

    def r_const():
        import lqcddb.analyse.analyse as RA
        return list(RA.plot_analyse_marker), list(RA.plot_analyse_color)

    def p_const():
        return list(AN.plot_analyse_marker), list(AN.plot_analyse_color)

    add('S08', '补充 plot_analyse_marker/color 常量', r_const, p_const,
        compare='none')

    g = np.ascontiguousarray(datalib.gauge()[:1])
    Ntz, Nx = g.shape[0], datalib.NX

    def r_unpol_ff():
        ops = load_donghx('Operator.py', ['operators_new_z0_mu2'])
        from pyqcd.operator._gluon_ope import plaquette_clover as pp
        pla = {(mu, nu): pp(g, mu, nu)
               for mu in range(4) for nu in range(4)}
        out = []
        for dz in (1, 2):
            out.append(ops['operators_new_z0_mu2'](
                g, 2, pla, pla, dz, 3, 1, 3, 1, Ntz))
        return out

    def p_unpol_ff():
        out = []
        for dz in (1, 2):
            out.append(np.asarray(pq_ope(g, 3, 1, 2, dz, Ntz, Nx,
                                          second_insert='F'))[-1])
        return out

    def _unpol_cmp(a, b):
        worst = 0.0
        for x, y in zip(a, b):
            xa = np.asarray(x).reshape(-1)
            ya = np.asarray(y).reshape(-1)
            worst = max(worst, float(np.linalg.norm(xa - ya)
                                     / max(np.linalg.norm(ya), 1e-300)))
        return worst

    add('S09', '补充 unpol 第二插入=F 选项（对照 donghx pla,pla 通道）',
        r_unpol_ff, p_unpol_ff, tol=1e-9, compare=_unpol_cmp, timeout=900)

    return cases
