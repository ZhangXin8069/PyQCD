"""MatX `einsum` 文档契约验证入口（语料 + 接口层 + pyqcd 参考对比）。

分组断言（每条独立计数，末尾打印 `PASS x/y`）：

- `doc-case`   ：文档 a–h 全量用例，numpy 参考 vs `pyqcd.einsum.contract`，
                 complex128 `rel <= 1e-12`，同时断言 shape 与 dtype，
                 并验证 `einsum_into` 写回输出张量。
- `implicit`   ：隐式输出推断（`implicit_output`）与 numpy 的自身推断一致，
                 显式/省略号输入被拒绝，隐式写法在 numpy 与 pyqcd 上一致。
- `ellipsis`   ：`...` 批维尺寸完全一致时可用；尺寸不一致、维数不一致、
                 size-1 批维在 MatX 语义下报错（numpy 侧行为一并记录）。
- `limitation` ：LIMITATIONS 完整性 + size-1 广播/对角输出/输出张量校验等
                 行为断言——numpy 与 pyqcd 的结果只记录，不当作 MatX 通过。
- `contradiction`：`...`/`->` 文档矛盾登记完整性（正文为规范、API 段待核实）。
- `availability`：原生 MatX 探针返回不可用（SKIP）且不抛异常，含注入式分支。

`pyqcd.einsum` 未就绪时，依赖它的断言逐条 SKIP 并打印原因，结构性断言
照常执行，退出码 0；有失败断言时退出码 1。不做任何联网或写入操作。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import warnings
import zlib

import numpy as np

from . import _bridge
from . import _contract as contract
from ..einsum import _target

TOL_REL = 1e-12
GROUPS = ('doc-case', 'implicit', 'ellipsis', 'limitation',
          'contradiction', 'availability', 'native')

REQUIRED_LIMITATION_KEYS = (
    'cpu-executor-unsupported',
    'ellipsis-shape-must-match',
    'no-size-1-broadcast',
    'optional-feature-and-versions',
    'first-call-autotuning',
    'tested-scope-contraction-inner-gemm',
)

REQUIRED_REASON_TOKENS = ('cuTENSOR', 'cuTensorNet', 'DMATX_EN_CUTENSOR=ON',
                          'GPU executor')


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


def _seed_of(name, base=0):
    """稳定的整数种子：`crc32(name)` 与 `base` 混合。

    不用 `hash()`，避免 PYTHONHASHSEED 破坏可复现性；`--seed` 只平移取值，
    同一 seed 下每条用例的种子固定。
    """
    return (zlib.crc32(name.encode('utf-8')) + int(base)) % 4294967296


def _rand_operands(shapes, seed):
    """固定 seed 的 complex128 操作数（实数与虚部分别取标准正态）。"""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(tuple(shape))
            + 1j * rng.standard_normal(tuple(shape)) for shape in shapes]


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


def _close(got, ref):
    rel, max_abs = _diff_stats(got, ref)
    detail = f'rel={rel}, max|d|={max_abs}'
    return (rel is not None and rel <= TOL_REL), detail


def _check_raises(tally, group, name, expected, call):
    """断言 call() 抛出 expected；通过与否都记录 detail。"""
    try:
        call()
    except expected as exc:
        tally.check(group, name, True, _exc_text(exc))
        return exc
    except Exception as exc:  # noqa: BLE001 - 类型不符按失败处理
        tally.check(group, name, False, f'异常类型不符: {_exc_text(exc)}')
        return exc
    tally.check(group, name, False, '未抛出异常')
    return None


def _call_pyqcd(module, subscripts, *operands):
    """调用被测实现并收集 warning（pyqcd 在 size-1 广播上会降级并告警）。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = module.contract(subscripts, *operands)
    return result, [str(item.message) for item in caught]


def _section_doc_case(tally, target, skip_reason='', seed=0):
    """文档 a–h 全量：numpy 参考 vs pyqcd.einsum，逐条断言 shape/dtype/值。"""
    group = 'doc-case'
    missing = [f'{case.get("name", index)}.{field}'
               for index, case in enumerate(contract.DOC_CASES)
               for field in contract.CASE_FIELDS if field not in case]
    tally.check(group, 'DOC_CASES 字段齐全', not missing, f'缺 {missing[:5]}')
    statuses = {contract.STATUS_TESTED, contract.STATUS_UNTESTED,
                contract.STATUS_UNSUPPORTED}
    bad_status = [case['name'] for case in contract.DOC_CASES
                  if case.get('matx_status') not in statuses]
    tally.check(group, 'DOC_CASES 的 matx_status 取值合法', not bad_status,
                f'{bad_status[:5]}')
    bad_mode = [case['name'] for case in contract.DOC_CASES
                if case.get('mode') not in (contract.MODE_EXPLICIT,
                                            contract.MODE_IMPLICIT)]
    tally.check(group, 'DOC_CASES 的 mode 取值合法', not bad_mode,
                f'{bad_mode[:5]}')
    no_source = [case['name'] for case in contract.DOC_CASES
                 if len(str(case.get('source_note', '')).strip()) < 8]
    tally.check(group, 'DOC_CASES 每条都有可追溯的 source_note', not no_source,
                f'{no_source[:5]}')
    groups = sorted({case['group'] for case in contract.DOC_CASES})
    tally.check(group, f'DOC_CASES 覆盖文档 a–h 共 8 条（实际 '
                       f'{len(contract.DOC_CASES)} 条，组 {groups}）',
                len(contract.DOC_CASES) == 8)

    for case in contract.DOC_CASES:
        name = case['name']
        subscripts = case['subscripts']
        shapes = [tuple(shape) for shape in case['shapes']]
        expected_shape = tuple(case['out_shape'])
        operands = _rand_operands(shapes, _seed_of(name, seed))
        try:
            inferred = contract.infer_output_shape(subscripts, shapes)
        except Exception as exc:  # noqa: BLE001 - 单项失败不影响其余断言
            inferred = None
            tally.check(group, f'{name}|推断输出形状', False, _exc_text(exc))
        else:
            tally.check(group, f'{name}|推断输出形状', inferred == expected_shape,
                        f'inferred={inferred}, doc={expected_shape}')
        try:
            ref = np.einsum(contract.normalize_subscripts(subscripts),
                            *operands)
        except Exception as exc:  # noqa: BLE001
            ref = None
            tally.check(group, f'{name}|numpy 参考可实现', False,
                        _exc_text(exc))
        else:
            tally.check(group, f'{name}|numpy 参考形状与文档一致',
                        tuple(ref.shape) == expected_shape,
                        f'numpy={tuple(ref.shape)}, doc={expected_shape}')

        if target is None:
            for label in ('shape', 'dtype', '值', 'einsum_into 写回'):
                tally.skip(group, f'{name}|{label}', skip_reason)
            continue
        if ref is None:
            for label in ('shape', 'dtype', '值', 'einsum_into 写回'):
                tally.check(group, f'{name}|{label}', False, 'numpy 参考不可用')
            continue

        try:
            got = target.contract(subscripts, *operands)
        except Exception as exc:  # noqa: BLE001
            for label in ('shape', 'dtype', '值'):
                tally.check(group, f'{name}|{label}', False, _exc_text(exc))
        else:
            got_shape = tuple(np.shape(got))
            tally.check(group, f'{name}|shape', got_shape == expected_shape,
                        f'got={got_shape}, doc={expected_shape}')
            tally.check(group, f'{name}|dtype',
                        np.asarray(got).dtype == np.complex128,
                        f'got={np.asarray(got).dtype}')
            ok, detail = _close(got, ref)
            tally.check(group, f'{name}|值', ok, detail)

        out = np.zeros(expected_shape, dtype=np.complex128)
        try:
            returned = contract.einsum_into(out, subscripts, *operands,
                                            backend='pyqcd')
        except Exception as exc:  # noqa: BLE001
            tally.check(group, f'{name}|einsum_into 写回', False,
                        _exc_text(exc))
        else:
            ok, detail = _close(out, ref)
            tally.check(group, f'{name}|einsum_into 写回',
                        ok and returned is out,
                        f'returned_is_out={returned is out}, {detail}')


def _section_implicit(tally, target, skip_reason='', seed=0):
    """隐式输出推断：纯字母规则、拒绝显式/省略号、与 numpy 推断一致。"""
    group = 'implicit'
    for subscripts, expected in contract.IMPLICIT_EXAMPLES:
        try:
            got = contract.implicit_output(subscripts)
        except Exception as exc:  # noqa: BLE001
            tally.check(group, f'{subscripts!r} -> {expected!r}', False,
                        _exc_text(exc))
            continue
        tally.check(group, f'{subscripts!r} -> {expected!r}', got == expected,
                    f'got={got!r}')

    example = contract.DOC_IMPLICIT_EXAMPLE
    try:
        got = contract.implicit_output(example['subscripts'])
    except Exception as exc:  # noqa: BLE001
        tally.check(group, '文档隐式示例可推断', False, _exc_text(exc))
    else:
        tally.check(group, '文档隐式示例可推断',
                    got == example['inferred_output'],
                    f"got={got!r}, doc={example['inferred_output']!r}")

    _check_raises(tally, group, '显式下标被 implicit_output 拒绝',
                  contract.MatXInterfaceError,
                  lambda: contract.implicit_output('ij,jk->ik'))
    _check_raises(tally, group, '隐式省略号被 implicit_output 拒绝',
                  contract.MatXUnsupportedError,
                  lambda: contract.implicit_output('...ij,...jk'))
    _check_raises(tally, group, '非法字符被拒绝',
                  contract.MatXInterfaceError,
                  lambda: contract.implicit_output('ij,j+'))

    for case in contract.DOC_CASES:
        name = case['name']
        subscripts = case['subscripts']
        if '...' in subscripts:
            continue
        implicit = subscripts.split('->')[0]
        operands = _rand_operands([tuple(s) for s in case['shapes']],
                                  _seed_of(name, seed))
        try:
            normalized = contract.normalize_subscripts(implicit)
            ref = np.einsum(implicit, *operands)
            alt = np.einsum(normalized, *operands)
        except Exception as exc:  # noqa: BLE001
            tally.check(group, f'{name}|归一化下标与 numpy 隐式一致', False,
                        _exc_text(exc))
        else:
            ok, detail = _close(alt, ref)
            tally.check(group, f'{name}|归一化下标与 numpy 隐式一致', ok,
                        f'{normalized!r}, {detail}')
        if target is None:
            tally.skip(group, f'{name}|pyqcd 隐式', skip_reason)
            continue
        try:
            got = target.contract(implicit, *operands)
        except Exception as exc:  # noqa: BLE001
            tally.check(group, f'{name}|pyqcd 隐式', False, _exc_text(exc))
            continue
        ok, detail = _close(got, np.einsum(implicit, *operands))
        tally.check(group, f'{name}|pyqcd 隐式', ok, detail)


def _section_ellipsis(tally, target, skip_reason='', seed=0):
    """`...` 批维：尺寸完全一致可用；不一致/维数不同/size-1 在 MatX 下报错。"""
    group = 'ellipsis'
    _check_raises(tally, group, '隐式模式下 ... 无法推断输出',
                  contract.MatXUnsupportedError,
                  lambda: contract.normalize_subscripts('...ij,...jk'))

    case = next(item for item in contract.DOC_CASES
                if item['name'] == 'ellipsis_batch_gemm')
    operands = _rand_operands([tuple(s) for s in case['shapes']],
                              _seed_of('ellipsis-positive', seed))
    ref = np.einsum(case['subscripts'], *operands)
    try:
        got = contract.infer_output_shape(case['subscripts'], case['shapes'])
    except Exception as exc:  # noqa: BLE001
        got = None
        tally.check(group, '... 尺寸一致时形状推断通过', False, _exc_text(exc))
    else:
        tally.check(group, '... 尺寸一致时形状推断通过',
                    tuple(got) == tuple(case['out_shape']),
                    f'got={tuple(got)}, doc={tuple(case["out_shape"])}')
    if target is None:
        tally.skip(group, '... 尺寸一致时数值一致', skip_reason)
    else:
        try:
            value = target.contract(case['subscripts'], *operands)
        except Exception as exc:  # noqa: BLE001
            tally.check(group, '... 尺寸一致时数值一致', False, _exc_text(exc))
        else:
            ok, detail = _close(value, ref)
            tally.check(group, '... 尺寸一致时数值一致',
                        ok and tuple(value.shape) == tuple(ref.shape),
                        f'{detail}, got={tuple(value.shape)}')

    mixed = _rand_operands([(5, 2, 7), (7,)],
                           _seed_of('ellipsis-vector', seed))
    try:
        got = contract.infer_output_shape('...i,i->...', [(5, 2, 7), (7,)])
    except Exception as exc:  # noqa: BLE001
        tally.check(group, '... 与普通标签混用时形状推断通过', False,
                    _exc_text(exc))
    else:
        tally.check(group, '... 与普通标签混用时形状推断通过',
                    tuple(got) == (5, 2), f'got={tuple(got)}')
    if target is None:
        tally.skip(group, '... 与普通标签混用时数值一致', skip_reason)
    else:
        try:
            value = target.contract('...i,i->...', *mixed)
            ok, detail = _close(value, np.einsum('...i,i->...', *mixed))
        except Exception as exc:  # noqa: BLE001
            tally.check(group, '... 与普通标签混用时数值一致', False,
                        _exc_text(exc))
        else:
            tally.check(group, '... 与普通标签混用时数值一致', ok, detail)

    left = _rand_operands([(5, 2, 3, 4)],
                          _seed_of('ellipsis-batch-left', seed))[0]
    right = _rand_operands([(5, 3, 4, 6)],
                           _seed_of('ellipsis-batch-right', seed))[0]
    out = np.zeros((5, 3, 3, 6), dtype=np.complex128)
    _check_raises(
        tally, group, '批维尺寸不一致时 MatX 语义报错',
        contract.MatXShapeError,
        lambda: contract.einsum_into(out, '...ij,...jk->...ik', left, right))
    _check_raises(
        tally, group, '批维尺寸不一致时 numpy 也报错（对照记录）',
        ValueError,
        lambda: np.einsum('...ij,...jk->...ik', left, right))

    wide = _rand_operands([(5, 1, 3, 4)], _seed_of('size1-batch', seed))[0]
    out = np.zeros((5, 3, 3, 6), dtype=np.complex128)
    _check_raises(
        tally, group, 'size-1 批维在 MatX 语义下 unsupported',
        contract.MatXUnsupportedError,
        lambda: contract.einsum_into(out, '...ij,...jk->...ik', wide, right))
    try:
        broadcast = np.einsum('...ij,...jk->...ik', wide, right)
    except Exception as exc:  # noqa: BLE001
        tally.check(group, 'size-1 批维 numpy 行为（仅记录）', False,
                    _exc_text(exc))
    else:
        tally.check(group, 'size-1 批维 numpy 行为（仅记录）',
                    tuple(broadcast.shape) == (5, 3, 3, 6),
                    f'numpy 把 size-1 批维广播成 '
                    f'{tuple(broadcast.shape)}（MatX 不接受，故不作为通过）')

    flat = _rand_operands([(3, 4)],
                          _seed_of('ellipsis-ndim-left', seed))[0]
    batched = _rand_operands([(2, 4, 6)],
                             _seed_of('ellipsis-ndim-right', seed))[0]
    out = np.zeros((2, 3, 6), dtype=np.complex128)
    _check_raises(
        tally, group, '... 批维维数不一致时 MatX 语义报错',
        contract.MatXShapeError,
        lambda: contract.einsum_into(out, '...ij,...jk->...ik', flat,
                                     batched))
    try:
        ragged = np.einsum('...ij,...jk->...ik', flat, batched)
    except Exception as exc:  # noqa: BLE001
        tally.check(group, '... 批维维数不一致时 numpy 行为（仅记录）',
                    False, _exc_text(exc))
    else:
        tally.check(group, '... 批维维数不一致时 numpy 行为（仅记录）',
                    tuple(ragged.shape) == (2, 3, 6),
                    f'numpy 返回 {tuple(ragged.shape)}（MatX 不接受，'
                    '故不作为通过）')


def _section_limitation(tally, target, skip_reason='', seed=0):
    """LIMITATIONS 完整性 + 语义差异行为断言（差异只记录，不算 MatX 通过）。"""
    group = 'limitation'
    keys = [entry.get('key') for entry in contract.LIMITATIONS]
    tally.check(group, f'LIMITATIONS 条数 >= 6（实际 {len(keys)}）',
                len(keys) >= 6)
    tally.check(group, 'LIMITATIONS key 唯一', len(set(keys)) == len(keys),
                f'{keys}')
    empty = [f'{entry.get("key")}.{field}'
             for entry in contract.LIMITATIONS
             for field in ('key', 'statement', 'impact', 'source',
                           'enforced_by')
             if not str(entry.get(field, '')).strip()]
    tally.check(group, '每条 LIMITATIONS 字段非空', not empty,
                f'空字段 {empty[:5]}')
    for key in REQUIRED_LIMITATION_KEYS:
        tally.check(group, f'登记限制 {key}', key in keys)

    shapes = [(2, 1, 3), (2, 4, 3)]
    subscripts = '...i,...i->...'
    operands = _rand_operands(shapes, _seed_of('size1-limitation', seed))
    out = np.zeros((2, 4), dtype=np.complex128)
    _check_raises(
        tally, group, 'size-1 广播在 MatX 语义下 unsupported',
        contract.MatXUnsupportedError,
        lambda: contract.einsum_into(out, subscripts, *operands,
                                     backend='numpy'))
    try:
        reference = np.einsum(subscripts, *operands)
    except Exception as exc:  # noqa: BLE001
        tally.check(group, 'size-1 广播 numpy 行为（仅记录）', False,
                    _exc_text(exc))
    else:
        tally.check(group, 'size-1 广播 numpy 行为（仅记录）',
                    tuple(reference.shape) == (2, 4),
                    f'numpy 按 size-1 广播返回 {tuple(reference.shape)}；'
                    'MatX 不接受，故不作为 MatX 通过')
    if target is None:
        tally.skip(group, 'size-1 广播 pyqcd 行为（仅记录）', skip_reason)
        tally.skip(group, '对角输出 pyqcd 行为（仅记录）', skip_reason)
    else:
        try:
            value, caught = _call_pyqcd(target, subscripts, *operands)
        except Exception as exc:  # noqa: BLE001
            consistent = True
            note = f'pyqcd 报错 {_exc_text(exc)}（拒绝该调用，与 MatX 一致）'
        else:
            consistent = bool(np.allclose(value,
                                          np.einsum(subscripts, *operands),
                                          rtol=TOL_REL, atol=TOL_REL))
            note = (f'pyqcd 按 numpy 语义返回 {tuple(np.shape(value))}'
                    f'，warning={caught[:1]}')
        tally.check(group, 'size-1 广播 pyqcd 行为（仅记录）', consistent,
                    note + '；该结果只作行为记录，不作为 MatX 通过')

    diagonal = _rand_operands([(4, 4)], _seed_of('diagonal', seed))[0]
    _check_raises(
        tally, group, '对角输出 ii->i 在 MatX 语义下 unsupported',
        contract.MatXUnsupportedError,
        lambda: contract.einsum_into(np.zeros((4,), dtype=np.complex128),
                                     'ii->i', diagonal))
    try:
        diagonal_ref = np.einsum('ii->i', diagonal)
    except Exception as exc:  # noqa: BLE001
        tally.check(group, '对角输出 numpy 行为（仅记录）', False,
                    _exc_text(exc))
    else:
        tally.check(group, '对角输出 numpy 行为（仅记录）',
                    tuple(diagonal_ref.shape) == (4,),
                    'numpy 返回对角线 (4,)；文档只给 `ii->` 迹，'
                    '对角输出未列入用例')
    trace = np.trace(diagonal)
    try:
        got = contract.infer_output_shape('ii->', [(4, 4)])
    except Exception as exc:  # noqa: BLE001
        tally.check(group, 'ii-> 迹在 MatX 语义下可推断', False,
                    _exc_text(exc))
    else:
        tally.check(group, 'ii-> 迹在 MatX 语义下可推断', got == (),
                    f'got={got}, trace={trace}')
    if target is not None:
        try:
            value, caught = _call_pyqcd(target, 'ii->i', diagonal)
        except Exception as exc:  # noqa: BLE001
            ok, note = True, f'pyqcd 报错 {_exc_text(exc)}'
        else:
            ok = bool(np.allclose(value, np.einsum('ii->i', diagonal),
                                  rtol=TOL_REL, atol=TOL_REL))
            note = f'pyqcd 返回 {tuple(np.shape(value))}, warning={caught[:1]}'
        tally.check(group, '对角输出 pyqcd 行为（仅记录）', ok,
                    note + '；只作行为记录，不作为 MatX 通过')

    wrong = np.zeros((5, 3), dtype=np.complex128)
    left = _rand_operands([(3, 4)], _seed_of('out-shape', seed))[0]
    right = _rand_operands([(4, 5)], _seed_of('out-shape-right', seed))[0]
    _check_raises(
        tally, group, '输出张量尺寸不符时接口报错', contract.MatXShapeError,
        lambda: contract.einsum_into(wrong, 'ij,jk->ik', left, right))
    read_only = np.zeros((3, 5), dtype=np.complex128)
    read_only.flags.writeable = False
    _check_raises(
        tally, group, '输出张量不可写时接口报错', contract.MatXInterfaceError,
        lambda: contract.einsum_into(read_only, 'ij,jk->ik', left, right))
    _check_raises(
        tally, group, '输出 dtype 无法容纳计算 dtype 时接口报错',
        contract.MatXInterfaceError,
        lambda: contract.einsum_into(np.zeros((3, 5), dtype=np.float64),
                                     'ij,jk->ik', left, right))
    _check_raises(
        tally, group, '未知 backend 被拒绝', contract.MatXInterfaceError,
        lambda: contract.einsum_into(np.zeros((3, 5), dtype=np.complex128),
                                     'ij,jk->ik', left, right,
                                     backend='matx'))
    escaped = np.zeros((2, 4), dtype=np.complex128)
    try:
        value = contract.einsum_into(escaped, subscripts, *operands,
                                     backend='numpy', matx_semantics=False)
        ok, detail = _close(value, np.einsum(subscripts, *operands))
    except Exception as exc:  # noqa: BLE001
        tally.check(group, 'matx_semantics=False 时退回 numpy 语义（仅记录）',
                    False, _exc_text(exc))
    else:
        tally.check(group, 'matx_semantics=False 时退回 numpy 语义（仅记录）',
                    ok, detail + '；该路径只用于对照，不代表 MatX 行为')


def _section_contradiction(tally):
    """`...`/`->` 文档矛盾登记完整性：正文为规范、API 段待核实。"""
    group = 'contradiction'
    entries = contract.CONTRADICTIONS
    tally.check(group, f'登记矛盾条数 >= 1（实际 {len(entries)}）',
                len(entries) >= 1)
    case_names = {case['name'] for case in contract.DOC_CASES}
    case_status = {case['name']: case['matx_status']
                   for case in contract.DOC_CASES}
    for entry in entries:
        key = entry.get('key', '?')
        for field in ('key', 'topic', 'doc_body_claim', 'doc_api_claim',
                      'resolution', 'api_claim_status'):
            value = str(entry.get(field, '')).strip()
            tally.check(group, f'{key}|{field} 非空', bool(value),
                        f'got={entry.get(field)!r}')
        api_claim = str(entry.get('doc_api_claim', ''))
        tally.check(group, f'{key}|API 段原文含省略号未支持声明',
                    'Ellipses are not supported' in api_claim,
                    api_claim[:70])
        tally.check(group, f'{key}|API 段原文含 -> 强制必需声明',
                    "'->' is required" in api_claim, api_claim[:70])
        body_claim = str(entry.get('doc_body_claim', ''))
        tally.check(group, f'{key}|正文声明含 ... 广播与显式输出',
                    '...' in body_claim and '->' in body_claim,
                    body_claim[:70])
        resolution = str(entry.get('resolution', ''))
        tally.check(group, f'{key}|规范化取正文、API 段待核实',
                    '正文' in resolution and 'API' in resolution,
                    resolution[:70])
        tally.check(group, f'{key}|API 段状态为 untested',
                    entry.get('api_claim_status') == contract.STATUS_UNTESTED,
                    f'got={entry.get("api_claim_status")!r}')
        affected = tuple(entry.get('affected_cases', ()))
        tally.check(group, f'{key}|受影响用例存在', bool(affected)
                    and set(affected) <= case_names, f'{affected}')
        tally.check(group, f'{key}|省略号用例在受影响列表内',
                    'ellipsis_batch_gemm' in affected, f'{affected}')
        tally.check(group, f'{key}|受影响用例未被标为 unsupported',
                    all(case_status.get(name)
                        != contract.STATUS_UNSUPPORTED for name in affected),
                    f'{[case_status.get(n) for n in affected]}')


def _section_availability(tally):
    """原生 MatX 探针：必须返回 SKIP 语义且不抛异常。"""
    group = 'availability'
    tally.check(group, '文档来源常量指向 matx 1.1.0 的 einsum 页',
                contract.DOC_VERSION == 'matx 1.1.0'
                and contract.DOC_URL.endswith(
                    '/MatX/api/linalg/matvec/einsum.html')
                and contract.DOC_NAMESPACE == 'matx::cutensor',
                f'{contract.DOC_VERSION} {contract.DOC_URL}')
    constants = (contract.FEATURE_MACRO, contract.REQUIRED_CUTENSOR,
                 contract.REQUIRED_CUTENSORNET, contract.MATX_ROOT_ENV,
                 contract.CALL_CONVENTION)
    tally.check(group, '可选特性/依赖版本/调用约定常量非空',
                all(str(value).strip() for value in constants),
                f'{constants}')
    try:
        result = contract.availability()
    except Exception as exc:  # noqa: BLE001
        tally.check(group, 'availability() 不抛异常', False, _exc_text(exc))
        return
    shaped = isinstance(result, tuple) and len(result) == 2
    tally.check(group, 'availability() 返回 (bool, reason)', shaped,
                f'got={result!r}')
    if not shaped:
        return
    available, reason = result
    tally.check(group, '可用性为 bool', isinstance(available, bool),
                f'got={type(available).__name__}')
    tally.check(group, '原因非空', bool(str(reason).strip()), str(reason)[:90])
    text = str(reason)
    if not available:
        for token in REQUIRED_REASON_TOKENS:
            tally.check(group, f'不可用原因含 {token}', token in text,
                        text[:90])
    else:
        tally.check(group, '可用时原因给出检测依据', bool(text.strip()),
                    text[:90])

    root = os.path.dirname(os.path.abspath(__file__))

    def _present(name):
        return f'/usr/lib/lib{name}.so.1'

    simulated = contract.availability(env={contract.MATX_ROOT_ENV: root},
                                      find_library=_present)
    tally.check(group, '注入式探针：环境齐全时返回可用',
                simulated[0] is True and bool(simulated[1]),
                f'got={simulated!r}')
    missing = contract.availability(env={}, find_library=lambda name: None)
    required = all(token in missing[1] for token in REQUIRED_REASON_TOKENS)
    tally.check(group, '注入式探针：缺失时返回不可用并给出要求',
                missing[0] is False and required, f'got={missing!r}')

    def _boom(name):
        raise OSError(f'模拟 find_library({name}) 失败')

    broken = contract.availability(env={}, find_library=_boom)
    tally.check(group, '注入式探针：find_library 抛错被吞掉',
                broken[0] is False and 'OSError' in broken[1],
                f'got={broken!r}')
    tally.check(group, '探针结果稳定（重复调用相同，无副作用）',
                contract.availability() == result,
                'availability() 只读环境变量与 find_library')


def _section_native(tally):
    """`cpp/cuda/matx` C ABI 合约：可加载、版本、fail-closed 与桩行为。"""
    group = 'native'
    info = _bridge.probe()
    if not info['available']:
        tally.skip(group, '桥归档与 ABI 可加载', info['reason'])
        return
    tally.check(group, 'ABI 版本不低于 Python 侧要求',
                info['abi_version'] >= _bridge.ABI_VERSION,
                f"abi={info['abi_version']} 要求>={_bridge.ABI_VERSION}")
    tally.check(group, '归档路径指向 libpyqcmatx_cuda.so',
                str(info['library']).endswith('libpyqcmatx_cuda.so'),
                info['library'])
    tally.check(group, 'cuTENSOR 开关可查询且为布尔',
                isinstance(info['cutensor'], bool),
                f"cutensor={info['cutensor']}")

    float64 = _bridge.DTYPE_IDS[np.dtype('float64')]
    status, message = _bridge.native_status(
        'ij,jk->ik', 0, [0x2000, 0x3000], [(4, 4), (4, 4)], (4, 4), float64)
    tally.check(group, '空输出指针被拒绝为 INVALID',
                status == _bridge.STATUS_INVALID,
                f'status={status} {message}')
    status, message = _bridge.native_status(
        'ij,jk->ik', 0x1000, [], [], (4, 4), float64)
    tally.check(group, '操作数个数越界被拒绝为 INVALID',
                status == _bridge.STATUS_INVALID,
                f'status={status} {message}')
    status, message = _bridge.native_status(
        'ij,jk,kl->il', 0x1000, [0x2000, 0x3000, 0x4000],
        [(2, 2), (2, 2), (2, 2)], (2, 2), float64)
    tally.check(group, '三个操作数超出桥接上限被拒绝',
                status == _bridge.STATUS_INVALID,
                f'status={status} {message}')
    status, message = _bridge.native_status(
        'ijklm->m', 0x1000, [0x2000], [(2, 2, 2, 2, 2)], (2,), float64)
    tally.check(group, 'rank>4 被拒绝为 INVALID',
                status == _bridge.STATUS_INVALID,
                f'status={status} {message}')
    status, message = _bridge.native_status(
        'ij->i', 0x1000, [0x2000], [(-1, 4)], (4,), float64)
    tally.check(group, '负尺寸被拒绝为 INVALID',
                status == _bridge.STATUS_INVALID,
                f'status={status} {message}')
    status, message = _bridge.native_status(
        'ij->j', 0x1000, [0x2000], [(2, 3)], (3,), 99)
    tally.check(group, '未知 dtype_id 被拒绝为 INVALID',
                status == _bridge.STATUS_INVALID,
                f'status={status} {message}')

    if info['cutensor']:
        tally.skip(group, '未编 cuTENSOR 的桩返回 UNAVAILABLE',
                   '本库已编入 cuTENSOR，跳过桩断言（需真实 GPU 张量才能执行）')
    else:
        status, message = _bridge.native_status(
            'i,i->', 0x1000, [0x2000, 0x3000], [(4,), (4,)], (), float64)
        tally.check(group, '桩库对合法调用返回 UNAVAILABLE(4)',
                    status == _bridge.STATUS_UNAVAILABLE,
                    f'status={status} {message}')
        tally.check(group, 'UNAVAILABLE 原因可操作',
                    'MatX' in message
                    and ('PYQCD_MATX_ROOT' in message
                         or 'MATX_EN_CUTENSOR' in message),
                    message)

    try:
        _bridge.einsum_into(np.zeros((4, 4)), 'ij,jk->ik',
                            np.zeros((4, 4)), np.zeros((4, 4)))
        raised = False
        detail = '主机数组被放行'
    except _bridge.NativeUnavailable as exc:
        raised = True
        detail = str(exc)
    tally.check(group, '主机 numpy 数组在进入 C ABI 前被拒绝',
                raised and ('GPU' in detail or '主机' in detail), detail)
    tally.check(group, '探针重复调用结果稳定',
                _bridge.probe()['abi_version'] == info['abi_version'],
                'probe() 只读归档头')


def _print_header(available, reason):
    print('════════ verify_matx：MatX einsum 文档契约 vs numpy/pyqcd '
          '════════')
    print(f'文档依据   : {contract.DOC_URL}（{contract.DOC_VERSION}）')
    print(f'原生 MatX  : ' + ('可用' if available else '不可用')
          + f'（{reason[:96]}）')
    bridge = _bridge.probe()
    print(f'桥归档     : {bridge["library"]}（'
          + ('可加载' if bridge['available'] else '不可用')
          + f'，abi={bridge["abi_version"]}'
          + f'，cutensor={bridge["cutensor"]}）')
    print(f'语料       : DOC_CASES {len(contract.DOC_CASES)} 条 / '
          f'LIMITATIONS {len(contract.LIMITATIONS)} 条 / '
          f'CONTRADICTIONS {len(contract.CONTRADICTIONS)} 条')


def run(seed=20260831, verbose=False):
    """执行全部分组断言，返回 (tally, skipped, reason)。"""
    tally = Tally()
    module, missing = _target.probe()
    ready = module is not None and not missing
    reason = '' if ready else _target.skip_reason()

    available, availability_reason = contract.availability()
    _print_header(available, availability_reason)
    print('被测实现   : ' + ('pyqcd.einsum 就绪' if ready else f'SKIP {reason}'))
    print(f'随机种子   : seed={seed}（每条用例由 crc32(name) 派生）')
    if not ready:
        print(f'\nSKIP 依赖 pyqcd.einsum 的断言：{reason}')

    target = module if ready else None
    _section_doc_case(tally, target, reason, seed)
    _section_implicit(tally, target, reason, seed)
    _section_ellipsis(tally, target, reason, seed)
    _section_limitation(tally, target, reason, seed)
    _section_contradiction(tally)
    _section_availability(tally)
    _section_native(tally)

    passed, failed, skip_count = tally.counts()
    total = passed + failed
    print('\n──────────── verify_matx 汇总 ────────────')
    for group in GROUPS:
        group_passed, group_failed, group_skipped = tally.group_counts(group)
        executed = group_passed + group_failed
        if executed + group_skipped == 0:
            continue
        if executed == 0:
            print(f'{group:<14s} SKIP   {reason[:70]}')
            continue
        state = 'PASS' if group_failed == 0 else f'FAIL({group_failed})'
        suffix = f'  SKIP {group_skipped}' if group_skipped else ''
        print(f'{group:<14s} {group_passed}/{executed} {state}{suffix}')
    if verbose:
        print('\n全部断言：')
        for row in tally.rows:
            state = {True: 'PASS', False: 'FAIL', None: 'SKIP'}[row['ok']]
            print(f"  {state} {row['group']}/{row['name']}: {row['detail']}")
    elif failed:
        print('\n前 5 条失败：')
        for row in tally.failures():
            print(f"  - {row['group']}/{row['name']}: {row['detail']}")
    print()
    print(f'PASS {passed}/{total}'
          + (f'  SKIP {skip_count} 项' if skip_count else ''))
    return tally, not ready, reason


def build_payload(tally, skipped, reason, seed):
    """机器可读摘要（`--json`）。"""
    passed, failed, skip_count = tally.counts()
    groups = {}
    for group in GROUPS:
        group_passed, group_failed, group_skipped = tally.group_counts(group)
        if group_passed + group_failed + group_skipped == 0:
            continue
        groups[group] = {'passed': group_passed, 'failed': group_failed,
                         'skipped': group_skipped}
    available, availability_reason = contract.availability()
    return {
        'ok': failed == 0,
        'verdict': 'FAIL' if failed else ('SKIP' if skipped else 'PASS'),
        'passed': passed,
        'total': passed + failed,
        'failed': failed,
        'skipped': skip_count,
        'target_ready': not skipped,
        'skip_reason': reason,
        'seed': seed,
        'doc': {'url': contract.DOC_URL, 'version': contract.DOC_VERSION,
                'namespace': contract.DOC_NAMESPACE},
        'corpus': {'doc_cases': len(contract.DOC_CASES),
                   'limitations': len(contract.LIMITATIONS),
                   'contradictions': len(contract.CONTRADICTIONS)},
        'native_matx': {'available': available,
                        'reason': availability_reason},
        'native_bridge': _bridge.probe(),
        'groups': groups,
        'failures': [{'group': row['group'], 'name': row['name'],
                      'detail': row['detail']}
                     for row in tally.rows if row['ok'] is False],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='MatX einsum 文档契约验证（语料 + 接口层 + pyqcd 对比）')
    parser.add_argument('--seed', type=int, default=20260831,
                        help='随机种子（默认 20260831）')
    parser.add_argument('--verbose', action='store_true',
                        help='打印每条断言的 detail')
    parser.add_argument('--json', action='store_true', dest='as_json',
                        help='只输出机器可读 JSON 摘要（优先于 --verbose 的'
                             '人类可读输出）')
    args = parser.parse_args(argv)

    if args.as_json:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            tally, skipped, reason = run(seed=args.seed, verbose=args.verbose)
        payload = build_payload(tally, skipped, reason, args.seed)
        print(json.dumps(payload, ensure_ascii=False, indent=2,
                         sort_keys=True))
        return 1 if payload['failed'] else 0

    tally, skipped, _reason = run(seed=args.seed, verbose=args.verbose)
    _passed, failed, _skipped = tally.counts()
    if failed:
        print('FAIL verify_matx')
        return 1
    if skipped:
        print('SKIP verify_matx：pyqcd.einsum 不可用，依赖它的断言未执行')
        return 0
    print('PASS verify_matx')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
