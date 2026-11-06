"""差分测试：`pyqcd.einsum` vs oracle（`refer/git-rep/opt_einsum`）。

三个面：
- API  面：contract / contract_path / contract_expression / PathInfo 字段齐全。
- 路径 面：≥200 个随机网络 × {greedy, optimal, dp, branch-2}，逐项比较
  `path` / `opt_cost` / `largest_intermediate`，并分别统计
  「逐位一致」「代价等价」「不一致」。
- 数值 面：≥50 条方程（含 ellipsis、hadamard、trace、outer、批量广播），
  complex128，`rel <= 1e-12`。

`pyqcd.einsum` 未就绪时整组 SKIP 并打印原因（exit 0）；oracle 自检始终运行，
用于证明差分夹具本身可信。每条断言独立计数，末尾打印 `PASS x/y`。
"""
from __future__ import annotations

import argparse
import zlib
from decimal import Decimal

import numpy as np

from . import _oracle, _target

OPTIMIZERS = ('greedy', 'optimal', 'dp', 'branch-2')
TOL_REL = 1e-12

# (标签, 名称, 方程, 各操作数形状)：覆盖 ellipsis / hadamard / trace / outer /
# 批量广播 / 隐式输出 / 蒸馏实战方程（小尺寸）
CURATED_CASES = (
    ('matmul', 'matmul', 'ij,jk->ik', [(5, 4), (4, 6)]),
    ('matmul', 'vecmat', 'ij,j->i', [(5, 4), (4,)]),
    ('trace', 'diag', 'ii->i', [(4, 4)]),
    ('trace', 'trace', 'ii->', [(5, 5)]),
    ('trace', 'triple_repeat', 'iji->j', [(3, 4, 3)]),
    ('trace', 'dot', 'i,i->', [(7,), (7,)]),
    ('outer', 'outer', 'i,j->ij', [(5,), (4,)]),
    ('outer', 'triple_outer', 'i,j,k->ijk', [(3,), (4,), (5,)]),
    ('hadamard', 'hadamard', 'ij,ij->ij', [(4, 5), (4, 5)]),
    ('hadamard', 'frobenius', 'ij,ij->', [(4, 5), (4, 5)]),
    ('hadamard', 'hadamard_chain', 'ab,ab,ab,ab->ab', [(3, 4)] * 4),
    ('hadamard', 'diag_times_matrix', 'ii,ij->j', [(4, 4), (4, 6)]),
    ('ellipsis', 'ellipsis_batch_matmul', '...ij,...jk->...ik',
     [(2, 3, 5, 4), (2, 3, 4, 6)]),
    ('ellipsis', 'ellipsis_reduce', '...i->...', [(2, 3, 4)]),
    ('ellipsis', 'ellipsis_transpose', '...ij->...ji', [(2, 3, 4)]),
    ('ellipsis', 'ellipsis_mixed', '...ij,jk->...ik', [(2, 4, 5), (5, 6)]),
    ('ellipsis', 'gamma_sandwich', 'ab,...bce,cd->...dae',
     [(4, 4), (2, 3, 4, 4, 7), (4, 4)]),
    ('batch', 'broadcast_vec_dot', 'i,...i->...', [(6,), (2, 3, 6)]),
    ('batch', 'broadcast_size_one', '...i,...i->...', [(2, 1, 3), (2, 4, 3)]),
    ('batch', 'batch_matrix_norm', '...ij,...ij->...', [(2, 3, 4), (2, 3, 4)]),
    ('batch', 'batch_hadamard_bcast', 'ij,ij,ij->ij',
     [(4, 1), (4, 5), (1, 5)]),
    ('implicit', 'implicit_matmul', 'ij,jk', [(4, 5), (5, 3)]),
    ('implicit', 'implicit_dot', 'i,i', [(6,), (6,)]),
    ('implicit', 'implicit_chain', 'ij,jk,kl', [(3, 4), (4, 5), (5, 6)]),
    ('chain', 'chain4', 'ab,bc,cd,de->ae',
     [(4, 4), (4, 5), (5, 6), (6, 3)]),
    ('chain', 'select', 'ijk,jk->i', [(3, 4, 5), (4, 5)]),
    ('chain', 'three_four', 'ijk,jkl,klm->im',
     [(3, 4, 5), (4, 5, 6), (5, 6, 7)]),
    ('distill', 'wick_2pt', 'gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M',
     [(2, 2, 3, 3), (2, 2, 3, 3), (2, 2, 3, 3), (2, 2), (2, 2),
      (1, 3, 3, 3), (1, 3, 3, 3), (2, 2)]),
    ('distill', 'wick_3pt', 'Geifj,ckdl,agbh,ce,gi,Mbdf,Mhjl,Gka->GM',
     [(1, 2, 2, 3, 3), (2, 2, 3, 3), (2, 2, 3, 3), (2, 2), (2, 2),
      (1, 3, 3, 3), (1, 3, 3, 3), (1, 2, 2)]),
    ('distill', 'peram_gauge_gamma',
     'zyxcTabn,zyxgTdef,Gae,Tzyxgc->TGdbfn',
     [(2, 2, 2, 3, 2, 2, 2, 4), (2, 2, 2, 3, 2, 2, 2, 4), (1, 2, 2),
      (2, 2, 2, 2, 3, 3)]),
)


class Tally:
    """逐项断言记录器：ok=True 通过 / False 失败 / None 跳过。"""

    def __init__(self):
        self.rows = []

    def check(self, group, name, ok, detail=''):
        ok = bool(ok)
        self.rows.append({'group': group, 'name': name, 'ok': ok,
                          'detail': str(detail)})
        return ok

    def skip(self, group, name, reason):
        self.rows.append({'group': group, 'name': name, 'ok': None,
                          'detail': str(reason)})

    def group_counts(self, group):
        rows = [r for r in self.rows if r['group'] == group]
        passed = sum(1 for r in rows if r['ok'] is True)
        failed = sum(1 for r in rows if r['ok'] is False)
        skipped = sum(1 for r in rows if r['ok'] is None)
        return passed, failed, skipped

    def counts(self):
        passed = sum(1 for r in self.rows if r['ok'] is True)
        failed = sum(1 for r in self.rows if r['ok'] is False)
        skipped = sum(1 for r in self.rows if r['ok'] is None)
        return passed, failed, skipped

    def failures(self, limit=5):
        return [r for r in self.rows if r['ok'] is False][:limit]


def _exc_text(exc):
    return f'{type(exc).__name__}: {exc}'


def _seed_of(name):
    """稳定的整数种子（不用 hash()，避免 PYTHONHASHSEED 破坏可复现性）。"""
    return zlib.crc32(name.encode('utf-8')) % 9973


def _diff_stats(got, ref):
    """返回 (相对差, 最大绝对差)；形状不一致返回 (None, None)。"""
    first = np.asarray(got)
    second = np.asarray(ref)
    if first.shape != second.shape:
        return None, None
    delta = np.abs(first - second)
    max_abs = float(delta.max()) if delta.size else 0.0
    denom = float(np.linalg.norm(second))
    if denom == 0.0:
        return (0.0 if max_abs == 0.0 else float('inf')), max_abs
    return float(np.linalg.norm(first - second) / denom), max_abs


def build_cases(n_random, seed):
    """人工方程 + 随机网络方程（尺寸压到 2..3，保证数值面秒级完成）。"""
    cases = [{'tag': tag, 'name': name, 'eq': eq,
              'shapes': [tuple(shape) for shape in shapes]}
             for tag, name, eq, shapes in CURATED_CASES]
    if n_random > 0:
        for index, network in enumerate(_oracle.rand_networks(
                n_random, seed=seed, regularity=4, n_out=2,
                d_min=2, d_max=3)):
            cases.append({'tag': 'random', 'name': f'random_{index:03d}',
                          'eq': network['eq'],
                          'shapes': network['shapes']})
    return cases


def _section_selfcheck(tally, cases):
    """oracle 与 numpy.einsum 交叉验证（证明差分夹具本身可用）。"""
    group = 'oracle 自检'
    module = _oracle.oracle()
    for case in cases:
        try:
            arrays = _oracle.rand_arrays(case, seed=_seed_of(case['name']))
            ref = module.contract(case['eq'], *arrays, optimize='auto')
            native = np.einsum(case['eq'], *arrays, optimize=True)
            rel, max_abs = _diff_stats(native, ref)
            tally.check(group, f"{case['name']}[{case['tag']}] oracle==numpy",
                        rel is not None and rel <= TOL_REL,
                        f'rel={rel}, max|d|={max_abs}')
        except Exception as exc:  # noqa: BLE001 - 单项失败不影响其余断言
            tally.check(group, f"{case['name']}[{case['tag']}] oracle==numpy",
                        False, _exc_text(exc))


def _section_api(tally, target):
    group = 'API 面'
    rng = np.random.default_rng(11)
    left = rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))
    right = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
    expected = np.einsum('ij,jk->ik', left, right)

    for name in _target.REQUIRED_CALLABLES:
        tally.check(group, f'{name} 存在且可调用',
                    callable(getattr(target, name, None)))
    info_class = getattr(target, 'PathInfo', None)
    tally.check(group, 'PathInfo 存在', info_class is not None)

    try:
        got = target.contract('ij,jk->ik', left, right)
        rel, max_abs = _diff_stats(got, expected)
        tally.check(group, 'contract 基本收缩', rel is not None
                    and rel <= TOL_REL, f'rel={rel}, max|d|={max_abs}')
    except Exception as exc:  # noqa: BLE001
        tally.check(group, 'contract 基本收缩', False, _exc_text(exc))

    try:
        out = np.zeros((3, 2), dtype=complex)
        target.contract('ij,jk->ik', left, right, out=out)
        rel, max_abs = _diff_stats(out, expected)
        tally.check(group, 'contract out= 写回', rel is not None
                    and rel <= TOL_REL, f'rel={rel}, max|d|={max_abs}')
    except Exception as exc:  # noqa: BLE001
        tally.check(group, 'contract out= 写回', False, _exc_text(exc))

    info = None
    try:
        result = target.contract_path('ij,jk->ik', left, right)
        structural = (isinstance(result, (tuple, list)) and len(result) == 2
                      and isinstance(result[0], list))
        tally.check(group, 'contract_path 返回 (list, PathInfo)', structural)
        if structural:
            info = result[1]
            tally.check(group, 'contract_path 第二项为 PathInfo',
                        info_class is not None
                        and isinstance(info, info_class),
                        type(info).__name__)
    except Exception as exc:  # noqa: BLE001
        tally.check(group, 'contract_path 返回 (list, PathInfo)', False,
                    _exc_text(exc))

    if info is None:
        for field in _target.PATHINFO_FIELDS:
            tally.check(group, f'PathInfo.{field} 存在', False,
                        'contract_path 未返回 PathInfo')
    else:
        for field in _target.PATHINFO_FIELDS:
            tally.check(group, f'PathInfo.{field} 存在', hasattr(info, field))
        for field in _target.DECIMAL_FIELDS:
            value = getattr(info, field, None)
            tally.check(group, f'PathInfo.{field} 为 Decimal',
                        isinstance(value, Decimal), type(value).__name__)
        type_checks = (('contraction_list', list), ('input_subscripts', str),
                       ('output_subscript', str), ('path', list),
                       ('indices', frozenset), ('scale_list', (list, tuple)),
                       ('size_list', (list, tuple)), ('size_dict', dict),
                       ('shapes', list), ('eq', str))
        for field, kind in type_checks:
            value = getattr(info, field, None)
            tally.check(group, f'PathInfo.{field} 类型',
                        isinstance(value, kind),
                        f'{field}={type(value).__name__}')
        try:
            steps = list(getattr(info, 'path', []) or [])
            contraction_list = list(getattr(info, 'contraction_list', []) or [])
            well_formed = all(isinstance(step, tuple) and len(step) == 2
                              and all(isinstance(i, int) for i in step)
                              for step in steps)
            tally.check(group, 'path 为 (int, int) 步序列且与 contraction_list 等长',
                        well_formed and len(steps) == len(contraction_list),
                        f'n_steps={len(steps)}, '
                        f'n_contraction={len(contraction_list)}')
        except Exception as exc:  # noqa: BLE001
            tally.check(group, 'path 为 (int, int) 步序列且与 contraction_list 等长',
                        False, _exc_text(exc))

    try:
        expr = target.contract_expression('ij,jk->ik', left.shape, right.shape)
        got = expr(left, right)
        rel, max_abs = _diff_stats(got, expected)
        tally.check(group, 'contract_expression 构造并求值',
                    rel is not None and rel <= TOL_REL,
                    f'rel={rel}, max|d|={max_abs}')
    except Exception as exc:  # noqa: BLE001
        tally.check(group, 'contract_expression 构造并求值', False,
                    _exc_text(exc))


def _section_paths(tally, target, networks):
    """路径面：path / opt_cost / largest_intermediate 三件套对比。"""
    group = '路径面'
    module = _oracle.oracle()
    exact = cost_only = mismatch = 0
    for index, network in enumerate(networks):
        eq = network['eq']
        shapes = [tuple(shape) for shape in network['shapes']]
        for optimizer in OPTIMIZERS:
            name = f'net{index:03d}|{optimizer}|{eq}'
            try:
                ref_path, ref_info = module.contract_path(
                    eq, *shapes, shapes=True, optimize=optimizer)
                got_path, got_info = target.contract_path(
                    eq, *shapes, shapes=True, optimize=optimizer)
            except Exception as exc:  # noqa: BLE001
                mismatch += 1
                tally.check(group, name, False, _exc_text(exc))
                continue
            path_equal = [tuple(step) for step in got_path] == \
                [tuple(step) for step in ref_path]
            cost_equal = (Decimal(getattr(got_info, 'opt_cost', -1))
                          == Decimal(ref_info.opt_cost))
            largest_equal = (
                Decimal(getattr(got_info, 'largest_intermediate', -1))
                == Decimal(ref_info.largest_intermediate))
            if path_equal and cost_equal and largest_equal:
                exact += 1
            elif cost_equal and largest_equal:
                cost_only += 1
            else:
                mismatch += 1
            detail = (f'path_eq={path_equal} cost_eq={cost_equal} '
                      f'largest_eq={largest_equal} '
                      f'got_cost={getattr(got_info, "opt_cost", None)} '
                      f'ref_cost={ref_info.opt_cost}')
            tally.check(group, name, cost_equal and largest_equal, detail)
    if len(networks) >= 200:
        tally.check(group, f'随机网络数 >= 200（实际 {len(networks)}）', True)
    else:
        print(f'  INFO 随机网络数 {len(networks)} < 200：非完整验收')
    tally.check(group, f'路径面不一致 == 0（实际 {mismatch}）', mismatch == 0)
    print(f'  路径面统计: 逐位一致 {exact} / 代价等价 {cost_only} / '
          f'不一致 {mismatch}（共 {exact + cost_only + mismatch} 项）')


def _section_numeric(tally, target, cases):
    group = '数值面'
    for case in cases:
        eq = case['eq']
        name = f"{case['name']}[{case['tag']}]"
        try:
            arrays = _oracle.rand_arrays(case, seed=_seed_of(case['name']))
            ref = _oracle.oracle().contract(eq, *arrays, optimize='auto')
            got = target.contract(eq, *arrays)
            rel, max_abs = _diff_stats(got, ref)
            tally.check(group, name, rel is not None and rel <= TOL_REL,
                        f'rel={rel}, max|d|={max_abs}, eq={eq}')
        except Exception as exc:  # noqa: BLE001
            tally.check(group, name, False, f'{_exc_text(exc)}, eq={eq}')


def run(networks=200, equations=40, seed=20260830, verbose=True):
    """执行全部差分断言，返回 (tally, skipped, reason)。"""
    tally = Tally()
    module, missing = _target.probe()
    oracle_version, oracle_path = _oracle.oracle_version()

    cases = build_cases(equations, seed)
    print(f'oracle      : opt_einsum {oracle_version} @ {oracle_path}')
    print(f'数值面方程   : {len(cases)} 条（人工 {len(CURATED_CASES)} 条 + '
          f'随机 {equations} 条）')
    print(f'路径面网络   : {networks} 个 × {len(OPTIMIZERS)} 优化器 '
          f'= {networks * len(OPTIMIZERS)} 项')

    _section_selfcheck(tally, cases)

    skipped = False
    reason = ''
    if module is None or missing:
        skipped = True
        reason = _target.skip_reason()
        planned_api = (len(_target.REQUIRED_CALLABLES) + 1 + 4
                       + len(_target.PATHINFO_FIELDS)
                       + len(_target.DECIMAL_FIELDS) + 10)
        print(f'\nSKIP 差异对比：{reason}')
        print(f'  计划断言：API 面 {planned_api} 项，'
              f'路径面 {networks * len(OPTIMIZERS)} 项，'
              f'数值面 {len(cases)} 项（全部 SKIP）')
        for group, planned in (('API 面', planned_api),
                               ('路径面', networks * len(OPTIMIZERS)),
                               ('数值面', len(cases))):
            tally.skip(group, f'{group}全部 {planned} 项断言', reason)
    else:
        networks_data = _oracle.rand_networks(networks, seed=seed)
        _section_api(tally, module)
        _section_paths(tally, module, networks_data)
        _section_numeric(tally, module, cases)

    passed, failed, skip_count = tally.counts()
    total = passed + failed
    print('\n──────────── verify_compat 汇总 ────────────')
    for group in ('oracle 自检', 'API 面', '路径面', '数值面'):
        group_passed, group_failed, group_skipped = tally.group_counts(group)
        if group_passed + group_failed + group_skipped == 0:
            continue
        if group_skipped and not (group_passed + group_failed):
            print(f'{group:<10s} SKIP   {reason[:80]}')
        else:
            print(f'{group:<10s} {group_passed}/{group_passed + group_failed}'
                  + (' PASS' if group_failed == 0 else f' FAIL({group_failed})'))
    if verbose and failed:
        print('\n前 5 条失败：')
        for row in tally.failures():
            print(f"  - {row['group']}/{row['name']}: {row['detail']}")
    print()
    print(f'PASS {passed}/{total}' + (f'  SKIP {skip_count} 项' if skipped else ''))
    return tally, skipped, reason


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='pyqcd.einsum 与 opt_einsum 的差分测试')
    parser.add_argument('--networks', type=int, default=200,
                        help='路径面随机网络数（默认 200）')
    parser.add_argument('--equations', type=int, default=40,
                        help='数值面追加的随机方程数（默认 40）')
    parser.add_argument('--seed', type=int, default=20260830)
    parser.add_argument('--smoke', action='store_true',
                        help='小样本冒烟（8 网络 / 6 随机方程）')
    args = parser.parse_args(argv)
    networks = 8 if args.smoke else args.networks
    equations = 6 if args.smoke else args.equations

    print('════════ verify_compat：pyqcd.einsum vs opt_einsum ════════')
    tally, skipped, _reason = run(networks=networks, equations=equations,
                                 seed=args.seed)
    passed, failed, _skipped = tally.counts()
    if failed:
        print('FAIL verify_compat')
        return 1
    if skipped:
        print('NON_ACCEPTANCE verify_compat：目标不可用，必需断言未执行')
        return 2
    print('PASS verify_compat')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
