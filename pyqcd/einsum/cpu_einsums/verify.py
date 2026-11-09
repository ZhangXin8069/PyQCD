"""Einsums 字符串 ``einsum`` 契约验证入口（语料 + spec 层 + ABI + 原生桥）。

分组断言（每条独立计数，末尾打印 ``PASS x/y``）：

- ``doc-case``：``DOC_CASES`` 语料；numpy 参考 vs spec 层推断，complex128
  ``rel <= 1e-12``，同时断言 shape 与 dtype；原生库可用时逐条与原生结果对拍。
- ``spec``：两种写法互转、隐式输出规则、非法输入（空段/重复输出/``...`` 等）。
- ``contract``：文档常量、``LIMITATIONS`` 完整性、冻结 ABI 常量、桥上限与
  ``einsum_into`` 在进入 C ABI **之前**的 fail-closed 拒绝。
- ``native``：ABI 探针 + 真实调用 ``native_status``（指针表/形状表编组）+
  C 侧 fail-closed 负例 + 桩库返回 ``UNAVAILABLE(4)``。库缺失时整组 SKIP
  并给出构建命令。

退出码：有失败断言 → 1；只有 SKIP → 0。不做任何联网或写入操作。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import zlib
from pathlib import Path

import numpy as np

from . import _bridge
from . import _contract

TOL_REL = 1e-12
GROUPS = ('doc-case', 'spec', 'contract', 'native')

REQUIRED_LIMITATION_KEYS = (
    'cxx20-required',
    'host-cpu-eager-api',
    'deps-blas-lapack-hdf5-openmp',
    'first-configure-network',
    'python-package-opt-in-pybind11',
    'bridge-bounded-nops-rank',
    'prefactors-alpha-beta',
    'multi-operand-nary',
    'compile-time-format-validation',
    'python-boundary-copies',
)

REQUIRED_LIMITATION_FIELDS = ('key', 'statement', 'impact', 'source',
                              'enforced_by')

REQUIRED_SYNTAX_TOKENS = ('<-', '->', ';', '...')

EXPECTED_DTYPE_IDS = (
    (np.dtype('complex128'), 0),
    (np.dtype('complex64'), 1),
    (np.dtype('float64'), 2),
    (np.dtype('float32'), 3),
)

#: 导入时刻的状态快照：``_bridge`` 导入不应该加载 .so。
_LIBRARY_AT_IMPORT = _bridge._LIBRARY

_NATIVE_CHECK_NAMES = (
    'ABI 版本满足 Python 侧要求',
    '库路径指向 libpyqceinsums.so',
    'backend_enabled 可查询且为布尔',
    'probe 重复调用稳定',
    '真实调用 native_status（指针表/形状表编组）',
    '合法调用状态码符合后端状态',
    '非 OK 状态带原因串',
    'nops=3 fail-closed 返回 INVALID',
    'rank=5 fail-closed 返回 INVALID',
    'out_ndim=5 fail-closed 返回 INVALID',
    '空输出指针 fail-closed 返回 INVALID',
    '未知 dtype_id fail-closed 返回 INVALID',
    '异 rank 输入在 Python 侧即返回 INVALID',
    'einsum_into 的数值路径（真跑或 UNAVAILABLE 桩）',
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
        rows = [row for row in self.rows if row['group'] == group]
        passed = sum(1 for row in rows if row['ok'] is True)
        failed = sum(1 for row in rows if row['ok'] is False)
        skipped = sum(1 for row in rows if row['ok'] is None)
        return passed, failed, skipped

    def counts(self):
        passed = sum(1 for row in self.rows if row['ok'] is True)
        failed = sum(1 for row in self.rows if row['ok'] is False)
        skipped = sum(1 for row in self.rows if row['ok'] is None)
        return passed, failed, skipped

    def failures(self, limit=5):
        return [row for row in self.rows if row['ok'] is False][:limit]


def _exc_text(exc):
    return f'{type(exc).__name__}: {exc}'


def _seed_of(name, base=0):
    """稳定的整数种子：``crc32(name)`` 与 ``base`` 混合。

    不用 ``hash()``，避免 PYTHONHASHSEED 破坏可复现性；``--seed`` 只平移取值。
    """
    return (zlib.crc32(name.encode('utf-8')) + int(base)) % 4294967296


def _rand_operands(shapes, seed):
    """固定 seed 的 complex128 操作数（实部与虚部分别取标准正态）。"""
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
    """断言 ``call()`` 抛出 ``expected``；通过与否都记录 detail。"""
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


def _section_contract(tally, info):
    """文档常量、语料元数据、冻结 ABI 常量与 Python 侧 fail-closed。"""
    group = 'contract'

    tally.check(group, 'DOC_URL 指向 einsums.github.io latest',
                _contract.DOC_URL
                == 'https://einsums.github.io/Einsums/latest/',
                _contract.DOC_URL)
    tally.check(group, 'DOC_VERSION 标记为 docs latest',
                _contract.DOC_VERSION == 'Einsums main (docs latest)',
                _contract.DOC_VERSION)
    tally.check(group, 'DOC_SOURCE_URL 指向 Einsums GitHub 仓库',
                _contract.DOC_SOURCE_URL
                == 'https://github.com/Einsums/Einsums',
                _contract.DOC_SOURCE_URL)
    tally.check(group, 'string_einsum 文档页 URL 完整',
                _contract.DOC_EINSUM_URL.endswith('string_einsum.html')
                and 'ComputeGraph' in _contract.DOC_EINSUM_URL,
                _contract.DOC_EINSUM_URL)
    missing_tokens = [token for token in REQUIRED_SYNTAX_TOKENS
                      if token not in _contract.SPEC_SYNTAX]
    tally.check(group, 'SPEC_SYNTAX 覆盖箭头/分号/省略号说明',
                not missing_tokens, f'缺 {missing_tokens}')
    tally.check(group, 'SPEC_SYNTAX 声明只支持字母标签',
                '字母标签' in _contract.SPEC_SYNTAX
                and '多字符' in _contract.SPEC_SYNTAX,
                _contract.SPEC_SYNTAX[:60])

    keys = _contract.limitation_keys()
    tally.check(group, f'LIMITATIONS 条数 >= 6（实际 {len(keys)}）',
                len(keys) >= 6)
    tally.check(group, 'LIMITATIONS key 唯一', len(set(keys)) == len(keys),
                f'{keys}')
    empty = [f'{entry.get("key")}.{field}'
             for entry in _contract.LIMITATIONS
             for field in REQUIRED_LIMITATION_FIELDS
             if not str(entry.get(field, '')).strip()]
    tally.check(group, '每条 LIMITATIONS 字段非空', not empty,
                f'空字段 {empty[:5]}')
    for key in REQUIRED_LIMITATION_KEYS:
        tally.check(group, f'登记限制 {key}', key in keys)

    tally.check(group, 'ABI_VERSION 与冻结 ABI 一致',
                _bridge.ABI_VERSION == 1, f'ab={_bridge.ABI_VERSION}')
    statuses = (_bridge.STATUS_OK, _bridge.STATUS_INVALID,
                _bridge.STATUS_MEMORY, _bridge.STATUS_INTERNAL,
                _bridge.STATUS_UNAVAILABLE)
    tally.check(group, '状态码 OK/INVALID/MEMORY/INTERNAL/UNAVAILABLE = 0..4',
                statuses == (0, 1, 2, 3, 4), f'{statuses}')
    dtype_ok = all(_bridge.DTYPE_IDS.get(dtype) == expected
                   for dtype, expected in EXPECTED_DTYPE_IDS)
    tally.check(group, 'dtype id 与冻结 ABI 一致（complex128=0…float32=3）',
                dtype_ok and len(_bridge.DTYPE_IDS) == 4,
                f'{_bridge.DTYPE_IDS}')
    tally.check(group, '桥上限 nops<=2 / rank<=4 / out_ndim<=4',
                (_bridge.MAX_OPERANDS, _bridge.MAX_RANK,
                 _bridge.MAX_OUT_NDIM) == (2, 4, 4),
                f'{_bridge.MAX_OPERANDS}/{_bridge.MAX_RANK}/'
                f'{_bridge.MAX_OUT_NDIM}')
    tally.check(group, 'PYQCD_EINSUMS_LIB / libpyqceinsums.so 常量',
                _bridge.LIBRARY_ENV == 'PYQCD_EINSUMS_LIB'
                and _bridge.LIBRARY_NAME == 'libpyqceinsums.so',
                f'{_bridge.LIBRARY_ENV}/{_bridge.LIBRARY_NAME}')

    path = _bridge.library_path()
    env_value = os.environ.get(_bridge.LIBRARY_ENV)
    expected_path = env_value or str(_bridge._REPO_LIB)
    tally.check(group, 'library_path 默认落在 cpp/cpu/cpu-einsums/'
                       '（可由 PYQCD_EINSUMS_LIB 覆盖）',
                str(_bridge._REPO_LIB).endswith(
                    'cpp/cpu/cpu-einsums/libpyqceinsums.so')
                and str(path) == str(Path(expected_path).expanduser()),
                f'path={path}, {_bridge.LIBRARY_ENV}={env_value!r}')
    tally.check(group, 'probe 返回 available/library/abi_version/'
                       'backend_enabled/reason',
                set(info) == {'available', 'library', 'abi_version',
                              'backend_enabled', 'reason'}
                and isinstance(info['available'], bool),
                f'{sorted(info)}')
    tally.check(group, 'probe 的 library 与 library_path 一致',
                info['library'] == str(path), info['library'])
    tally.check(group, 'probe 的 reason 非空（可用或不可用都给依据）',
                bool(str(info['reason']).strip()), str(info['reason'])[:90])
    tally.check(group, '_bridge 导入时不加载 .so',
                _LIBRARY_AT_IMPORT is None, f'got={_LIBRARY_AT_IMPORT!r}')

    bounds = (
        ('合法 gemm', (2, [2, 2], 2), True),
        ('nops=3 越界', (3, [2, 2, 2], 2), False),
        ('rank=5 越界', (1, [5], 1), False),
        ('异 rank 无法表达', (2, [2, 1], 2), False),
        ('out_ndim=5 越界', (1, [2], 5), False),
    )
    for label, (nops, ranks, out_ndim), expected in bounds:
        ok, reason = _bridge.check_abi_bounds(nops, ranks, out_ndim)
        tally.check(group, f'check_abi_bounds: {label}', ok is expected,
                    f'ok={ok}, reason={reason[:70]}')

    left = _rand_operands([(7, 7)], _seed_of('contract-left'))[0]
    right = _rand_operands([(7, 7)], _seed_of('contract-right'))[0]
    gemm = 'ij <- ik ; kj'
    out = np.zeros((7, 7), dtype=np.complex128)

    def _expect_bridge_error(name, call):
        _check_raises(tally, group, name, _bridge.EinsumsBridgeError, call)

    _expect_bridge_error(
        'einsum_into 拒绝操作数个数超出桥上限',
        lambda: _bridge.einsum_into(np.zeros((7, 7), np.complex128),
                                    'il <- ij ; jk ; kl', left, right, left))
    _expect_bridge_error(
        'einsum_into 拒绝操作数个数与 spec 不符',
        lambda: _bridge.einsum_into(out, gemm, left))
    _expect_bridge_error(
        'einsum_into 拒绝 rank>4',
        lambda: _bridge.einsum_into(
            np.zeros((2,), np.complex128), 'a <- abcde',
            np.zeros((2, 2, 2, 2, 2), np.complex128)))
    _expect_bridge_error(
        'einsum_into 拒绝异 rank（GEMV，冻结 ABI 只有单个 INPUT_RANK）',
        lambda: _bridge.einsum_into(
            np.zeros((4,), np.complex128), 'i <- ik ; k',
            np.zeros((4, 5), np.complex128), np.zeros((5,), np.complex128)))
    _expect_bridge_error(
        'einsum_into 拒绝 dtype 混用（混合精度 fail closed）',
        lambda: _bridge.einsum_into(out, gemm, left.astype(np.complex64),
                                    right.astype(np.complex64)))
    _expect_bridge_error(
        'einsum_into 拒绝非 C 连续输出',
        lambda: _bridge.einsum_into(np.zeros((7, 14), np.complex128)[:, ::2],
                                    gemm, left, right))
    read_only = np.zeros((7, 7), np.complex128)
    read_only.flags.writeable = False
    _expect_bridge_error(
        'einsum_into 拒绝不可写输出',
        lambda: _bridge.einsum_into(read_only, gemm, left, right))
    _check_raises(
        tally, group, 'einsum_into 拒绝输出形状不符',
        _contract.EinsumsShapeError,
        lambda: _bridge.einsum_into(np.zeros((7, 8), np.complex128), gemm,
                                    left, right))


def _section_spec(tally):
    """spec 层：两种写法互转、隐式输出规则、非法输入拒绝。"""
    group = 'spec'

    def _normalize(text):
        return _contract.normalize_spec(text)

    cases = (
        ('箭头写法规范化幂等', 'ij <- ik ; kj', 'ij <- ik ; kj'),
        ('numpy 显式写法转 Einsums', 'ik,kj->ij', 'ij <- ik ; kj'),
        ('numpy 分号写法转 Einsums', 'ik;kj -> ij', 'ij <- ik ; kj'),
        ('numpy 隐式写法转 Einsums', 'ik,kj', 'ij <- ik ; kj'),
        ('Einsums 原样保留空白规范化', '  ij<-ik ;kj ', 'ij <- ik ; kj'),
        ('标量输出保留空输出段', ' <- i ; i', ' <- i ; i'),
        ('numpy 标量输出转空输出段', 'i,i->', ' <- i ; i'),
        ('重复输入标签（迹）转空输出段', 'ii->', ' <- ii'),
        ('隐式输出按字母序', 'ba,ac', 'bc <- ba ; ac'),
    )
    for label, text, expected in cases:
        try:
            got = _normalize(text)
        except Exception as exc:  # noqa: BLE001
            tally.check(group, label, False, _exc_text(exc))
            continue
        tally.check(group, label, got == expected, f'{text!r} -> {got!r}')

    try:
        parsed = _contract.parse_einsums_spec('ij <- ik ; kj')
    except Exception as exc:  # noqa: BLE001
        tally.check(group, 'parse 返回 (out, [in...])', False, _exc_text(exc))
    else:
        tally.check(group, 'parse 返回 (out, [in...])',
                    parsed == (('i', 'j'), [('i', 'k'), ('k', 'j')]),
                    f'{parsed!r}')

    try:
        parsed = _contract.parse_einsums_spec(' <- i ; i')
    except Exception as exc:  # noqa: BLE001
        tally.check(group, '标量输出解析为空标签元组', False, _exc_text(exc))
    else:
        tally.check(group, '标量输出解析为空标签元组',
                    parsed == ((), [('i',), ('i',)]), f'{parsed!r}')

    try:
        parsed = _contract.parse_einsums_spec('mu,nu <- mu,rho ; rho,nu')
    except Exception as exc:  # noqa: BLE001
        tally.check(group, '多字符模式可解析（未验证桥支不支持）', False,
                    _exc_text(exc))
    else:
        tally.check(group, '多字符模式可解析（未验证桥支不支持）',
                    parsed == (('mu', 'nu'), [('mu', 'rho'), ('rho', 'nu')]),
                    f'{parsed!r}')

    implicit = (
        ('ij,jk', [(3, 4), (4, 5)]),
        ('ba,ac', [(2, 3), (3, 4)]),
        ('ij,jk,ki', [(2, 3), (3, 4), (4, 2)]),
        ('ijk,jil', [(3, 4, 5), (4, 3, 2)]),
    )
    for subscripts, shapes in implicit:
        try:
            converted = _contract.numpy_to_einsums(subscripts)
            out_labels, in_labels = _contract.parse_einsums_spec(converted)
            sizes = _contract.check_shapes(out_labels, in_labels, shapes)
            got = tuple(sizes[label] for label in out_labels)
            ref = np.einsum(subscripts,
                            *[np.zeros(shape, np.complex128)
                              for shape in shapes])
        except Exception as exc:  # noqa: BLE001
            tally.check(group, f'隐式输出与 numpy 一致: {subscripts!r}', False,
                        _exc_text(exc))
            continue
        tally.check(group, f'隐式输出与 numpy 一致: {subscripts!r}',
                    got == ref.shape,
                    f'{converted!r} -> {got}, numpy={ref.shape}')

    illegal = (
        ('... 省略号被 normalize_spec 拒绝', _contract.EinsumsSpecError,
         lambda: _normalize('...ij,...jk->...ik')),
        ('... 省略号被 numpy_to_einsums 拒绝', _contract.EinsumsSpecError,
         lambda: _contract.numpy_to_einsums('...ij,...jk->...ik')),
        ('空 spec 被拒绝', _contract.EinsumsSpecError,
         lambda: _normalize('   ')),
        ('numpy 写法被 parse_einsums_spec 拒绝（提示先转换）',
         _contract.EinsumsSpecError,
         lambda: _contract.parse_einsums_spec('ik,kj->ij')),
        ('无箭头被拒绝', _contract.EinsumsSpecError,
         lambda: _contract.parse_einsums_spec('ij ik kj')),
        ('两种箭头并存被拒绝', _contract.EinsumsSpecError,
         lambda: _contract.parse_einsums_spec('ij <- ik -> kj')),
        ('箭头重复被拒绝', _contract.EinsumsSpecError,
         lambda: _contract.parse_einsums_spec('ij <- ik <- kj')),
        ('空输入段被拒绝（尾部 ;）', _contract.EinsumsSpecError,
         lambda: _contract.parse_einsums_spec('ij <- ik ;')),
        ('空输入段被拒绝（连续 ;;）', _contract.EinsumsSpecError,
         lambda: _contract.parse_einsums_spec('ij <- ik ;; kj')),
        ('缺少输入操作数被拒绝', _contract.EinsumsSpecError,
         lambda: _contract.parse_einsums_spec('ij <-')),
        ('空输出不是错误但空输入是', _contract.EinsumsSpecError,
         lambda: _contract.parse_einsums_spec(' <- ')),
        ('输出标签重复被拒绝', _contract.EinsumsSpecError,
         lambda: _contract.parse_einsums_spec('ii <- ij')),
        ('numpy 输出标签重复被拒绝', _contract.EinsumsSpecError,
         lambda: _contract.numpy_to_einsums('ij->ii')),
        ('输出标签不在输入里被拒绝', _contract.EinsumsSpecError,
         lambda: _contract.parse_einsums_spec('ik <- ij')),
        ('numpy 输出标签不在输入里被拒绝', _contract.EinsumsSpecError,
         lambda: _contract.numpy_to_einsums('ij->ik')),
        ('非字母标签被拒绝（Einsums 写法）', _contract.EinsumsSpecError,
         lambda: _contract.parse_einsums_spec('ij <- i+ ; kj')),
        ('非字母标签被拒绝（numpy 写法）', _contract.EinsumsSpecError,
         lambda: _contract.numpy_to_einsums('ij,j+->ik')),
        ('P(...) 置换算子被拒绝', _contract.EinsumsSpecError,
         lambda: _contract.parse_einsums_spec(
             'i,j,a,b <- P(ij) P(ab) i,k,a,c ; k,c,j,b')),
        ('多字符标签无法映射到 numpy 下标', _contract.EinsumsSpecError,
         lambda: _contract.einsums_to_numpy('mu,nu <- mu,rho ; rho,nu')),
    )
    for label, expected, call in illegal:
        _check_raises(tally, group, label, expected, call)

    shape_errors = (
        ('rank 与标签数不符', lambda: _contract.check_shapes(
            ('j',), [('i', 'j')], [(3, 4, 5)])),
        ('同名标签尺寸冲突', lambda: _contract.check_shapes(
            ('k',), [('i', 'j'), ('j', 'k')], [(3, 4), (5, 2)])),
        ('负维被拒绝', lambda: _contract.check_shapes(
            ('j',), [('i', 'j')], [(3, -4)])),
        ('非整数维被拒绝', lambda: _contract.check_shapes(
            ('j',), [('i', 'j')], [(3, 4.5)])),
        ('标签组数与形状个数不符', lambda: _contract.check_shapes(
            ('j',), [('i', 'j')], [(3, 4), (4,)])),
        ('输出标签无已知尺寸', lambda: _contract.check_shapes(
            ('k',), [('i', 'j')], [(3, 4)])),
    )
    for label, call in shape_errors:
        _check_raises(tally, group, label, _contract.EinsumsShapeError, call)

    try:
        sizes = _contract.check_shapes(('i', 'j'), [('i', 'k'), ('k', 'j')],
                                       [(7, 9), (9, 5)])
        inferred = _contract.infer_output_shape(
            ('i', 'j'), [(7, 9), (9, 5)], sizes, [('i', 'k'), ('k', 'j')])
    except Exception as exc:  # noqa: BLE001
        tally.check(group, 'check_shapes/infer_output_shape 协同工作', False,
                    _exc_text(exc))
    else:
        tally.check(group, 'check_shapes/infer_output_shape 协同工作',
                    sizes == {'i': 7, 'k': 9, 'j': 5}
                    and inferred == (7, 5),
                    f'sizes={sizes}, out={inferred}')


def _section_doc_case(tally, info, seed=0):
    """文档语料：numpy 参考 vs spec 层推断；原生可用时逐条对拍。"""
    group = 'doc-case'
    cases = _contract.DOC_CASES

    missing = [f'{case.get("name", index)}.{field}'
               for index, case in enumerate(cases)
               for field in _contract.CASE_FIELDS if field not in case]
    tally.check(group, 'DOC_CASES 字段齐全', not missing, f'缺 {missing[:5]}')
    sources = (_contract.SOURCE_DOCUMENTED, _contract.SOURCE_INFERRED)
    bad_source = [case['name'] for case in cases
                  if case.get('source') not in sources]
    tally.check(group, 'DOC_CASES 的 source 取值合法', not bad_source,
                f'{bad_source[:5]}')
    bad_status = [case['name'] for case in cases
                  if case.get('status') != _contract.STATUS_UNTESTED]
    tally.check(group, 'DOC_CASES 全部标 untested（不冒充官方已测）',
                not bad_status, f'{bad_status[:5]}')
    thin = [case['name'] for case in cases
            if len(str(case.get('source_note', '')).strip()) < 8]
    tally.check(group, '每条 DOC_CASES 都有可追溯的 source_note', not thin,
                f'{thin[:5]}')
    documented = _contract.documented_cases()
    inferred = _contract.inferred_cases()
    tally.check(group, f'documented 用例 >= 9（实际 {len(documented)}）',
                len(documented) >= 9)
    tally.check(group, f'inferred 用例 >= 4（实际 {len(inferred)}）',
                len(inferred) >= 4)
    gemm = next((case for case in cases if case['name'] == 'gemm_7x7'), None)
    tally.check(group, '文档首页 GEMM "ij <- ik ; kj"（7×7）在语料里',
                gemm is not None
                and gemm['spec'] == 'ij <- ik ; kj'
                and tuple(gemm['shapes']) == ((7, 7), (7, 7))
                and gemm['source'] == _contract.SOURCE_DOCUMENTED,
                f'{gemm and gemm.get("spec")}')
    bare = [case['name'] for case in cases
            if case.get('native_check') is False
            and len(str(case.get('native_note', '')).strip()) < 8]
    tally.check(group, '不向原生桥断言的用例给出理由', not bare, f'{bare[:5]}')

    for case in cases:
        name = case['name']
        shapes = [tuple(shape) for shape in case['shapes']]
        expected = tuple(case['out_shape'])
        operands = _rand_operands(shapes, _seed_of(name, seed))
        try:
            normalized = _contract.normalize_spec(case['spec'])
            out_labels, in_labels = _contract.parse_einsums_spec(normalized)
        except Exception as exc:  # noqa: BLE001
            for label in ('spec 解析', 'spec 互转', '形状推断',
                          'numpy 参考形状/dtype', '原生数值'):
                tally.check(group, f'{name}|{label}', False, _exc_text(exc))
            continue

        ranks = [len(labels) for labels in in_labels]
        tally.check(group, f'{name}|spec 解析',
                    ranks == [len(shape) for shape in shapes],
                    f'{normalized!r}, ranks={ranks}')

        try:
            back_to_numpy = _contract.einsums_to_numpy(normalized)
            forward = _contract.numpy_to_einsums(case['numpy_spec'])
        except Exception as exc:  # noqa: BLE001
            tally.check(group, f'{name}|spec 互转', False, _exc_text(exc))
        else:
            tally.check(group, f'{name}|spec 互转',
                        back_to_numpy == case['numpy_spec']
                        and forward == normalized,
                        f'{back_to_numpy!r} vs {case["numpy_spec"]!r}；'
                        f'{forward!r} vs {normalized!r}')

        try:
            sizes = _contract.check_shapes(out_labels, in_labels, shapes)
            inferred_shape = _contract.infer_output_shape(
                out_labels, shapes, sizes, in_labels)
        except Exception as exc:  # noqa: BLE001
            inferred_shape = None
            tally.check(group, f'{name}|形状推断', False, _exc_text(exc))
        else:
            tally.check(group, f'{name}|形状推断', inferred_shape == expected,
                        f'inferred={inferred_shape}, doc={expected}')

        try:
            ref = np.einsum(case['numpy_spec'], *operands)
        except Exception as exc:  # noqa: BLE001
            ref = None
            tally.check(group, f'{name}|numpy 参考形状/dtype', False,
                        _exc_text(exc))
        else:
            tally.check(group, f'{name}|numpy 参考形状/dtype',
                        tuple(ref.shape) == expected
                        and ref.dtype == np.complex128,
                        f'numpy={tuple(ref.shape)}, dtype={ref.dtype}')

        native_name = f'{name}|原生结果（数值或 fail-closed）'
        if not info['available']:
            tally.skip(group, native_name, info['reason'])
            continue
        if not case['native_check']:
            tally.skip(group, native_name, case['native_note'])
            continue

        out = np.zeros(expected, dtype=np.complex128)
        if not info['backend_enabled']:
            # 桩库：合法调用必须 fail-closed 返回 UNAVAILABLE，不能伪造结果。
            try:
                _bridge.einsum_into(out, case['spec'], *operands)
            except _bridge.EinsumsNativeError as exc:
                tally.check(group, native_name,
                            exc.status == _bridge.STATUS_UNAVAILABLE,
                            f'桩库 fail-closed: status={exc.status}, '
                            f'msg={exc.message[:70]}')
            except Exception as exc:  # noqa: BLE001
                tally.check(group, native_name, False, _exc_text(exc))
            else:
                tally.check(group, native_name, False,
                            '桩库未报错：可能伪造了原生成功')
            continue

        if ref is None:
            tally.check(group, native_name, False, 'numpy 参考不可用')
            continue
        try:
            returned = _bridge.einsum_into(out, case['spec'], *operands)
        except Exception as exc:  # noqa: BLE001
            tally.check(group, native_name, False, _exc_text(exc))
            continue
        ok, detail = _close(out, ref)
        tally.check(group, native_name,
                    ok and returned is out
                    and out.dtype == np.complex128,
                    f'returned_is_out={returned is out}, {detail}')


def _section_native(tally, info):
    """ABI 探针 + 真实调用编组 + C 侧 fail-closed + 桩返回 UNAVAILABLE。"""
    group = 'native'
    if not info['available']:
        for name in _NATIVE_CHECK_NAMES:
            tally.skip(group, name, info['reason'])
        return

    tally.check(group, 'ABI 版本满足 Python 侧要求',
                int(info['abi_version']) >= _bridge.ABI_VERSION,
                f"abi={info['abi_version']} 要求>={_bridge.ABI_VERSION}")
    tally.check(group, '库路径指向 libpyqceinsums.so',
                str(info['library']).endswith('libpyqceinsums.so'),
                info['library'])
    tally.check(group, 'backend_enabled 可查询且为布尔',
                isinstance(info['backend_enabled'], bool),
                f"backend_enabled={info['backend_enabled']}")
    again = _bridge.probe()
    tally.check(group, 'probe 重复调用稳定',
                again['abi_version'] == info['abi_version']
                and again['backend_enabled'] == info['backend_enabled'],
                f'{again}')

    dtype = np.dtype('complex128')
    left = _rand_operands([(7, 7)], _seed_of('native-left'))[0]
    right = _rand_operands([(7, 7)], _seed_of('native-right'))[0]
    out = np.zeros((7, 7), dtype=dtype)

    status, message = _bridge.native_status('ij <- ik ; kj', out,
                                            [left, right], dtype)
    expected_status = (_bridge.STATUS_OK if info['backend_enabled']
                       else _bridge.STATUS_UNAVAILABLE)
    tally.check(group, '真实调用 native_status（指针表/形状表编组）',
                status == expected_status,
                f'status={status}（{_bridge.STATUS_NAMES.get(status)}），'
                f'期望 {expected_status}；message={message[:80]}')
    tally.check(group, '合法调用状态码符合后端状态',
                status in (_bridge.STATUS_OK, _bridge.STATUS_UNAVAILABLE,
                           _bridge.STATUS_INVALID),
                f'status={status}, backend_enabled={info["backend_enabled"]}')
    tally.check(group, '非 OK 状态带原因串',
                status == _bridge.STATUS_OK or bool(message.strip()),
                f'status={status}, message={message[:120]!r}')

    vector_x = _rand_operands([(4,)], _seed_of('native-dot-x'))[0]
    vector_y = _rand_operands([(4,)], _seed_of('native-dot-y'))[0]
    negatives = (
        ('nops=3 fail-closed 返回 INVALID',
         'il <- ij ; jk ; kl', np.zeros((7, 7), dtype),
         [left, right, left]),
        ('rank=5 fail-closed 返回 INVALID',
         'a <- abcde', np.zeros((2,), dtype),
         [np.zeros((2, 2, 2, 2, 2), dtype)]),
        ('out_ndim=5 fail-closed 返回 INVALID',
         'abcde <- a', np.zeros((2, 2, 2, 2, 2), dtype),
         [np.zeros((2,), dtype)]),
        ('空输出指针 fail-closed 返回 INVALID',
         ' <- i ; i', 0, [vector_x, vector_y]),
    )
    for name, spec, target, operands in negatives:
        status, message = _bridge.native_status(spec, target, operands, dtype)
        tally.check(group, name, status == _bridge.STATUS_INVALID,
                    f'status={status}, message={message[:80]}')

    status, message = _bridge.native_status('ij <- ik ; kj', out,
                                            [left, right], 99)
    tally.check(group, '未知 dtype_id fail-closed 返回 INVALID',
                status == _bridge.STATUS_INVALID,
                f'status={status}, message={message[:80]}')

    status, message = _bridge.native_status(
        'i <- ik ; k', np.zeros((4,), dtype),
        [np.zeros((4, 5), dtype), np.zeros((5,), dtype)], dtype)
    tally.check(group, '异 rank 输入在 Python 侧即返回 INVALID',
                status == _bridge.STATUS_INVALID
                and 'INPUT_RANK' in message,
                f'status={status}, message={message[:80]}')

    reference = np.einsum('ik,kj->ij', left, right)
    if info['backend_enabled']:
        target = np.zeros((7, 7), dtype=dtype)
        try:
            returned = _bridge.einsum_into(target, 'ij <- ik ; kj', left,
                                           right)
        except Exception as exc:  # noqa: BLE001
            tally.check(group, 'einsum_into 的数值路径（真跑或 UNAVAILABLE 桩）',
                        False, _exc_text(exc))
        else:
            ok, detail = _close(target, reference)
            tally.check(group, 'einsum_into 的数值路径（真跑或 UNAVAILABLE 桩）',
                        ok and returned is target,
                        f'returned_is_out={returned is target}, {detail}')
    else:
        target = np.zeros((7, 7), dtype=dtype)
        try:
            _bridge.einsum_into(target, 'ij <- ik ; kj', left, right)
        except _bridge.EinsumsNativeError as exc:
            tally.check(
                group, 'einsum_into 的数值路径（真跑或 UNAVAILABLE 桩）',
                exc.status == _bridge.STATUS_UNAVAILABLE
                and exc.name == 'UNAVAILABLE',
                f'status={exc.status}, name={exc.name}, '
                f'msg={exc.message[:80]}')
        except Exception as exc:  # noqa: BLE001
            tally.check(group, 'einsum_into 的数值路径（真跑或 UNAVAILABLE 桩）',
                        False, f'异常类型不符: {_exc_text(exc)}')
        else:
            tally.check(group, 'einsum_into 的数值路径（真跑或 UNAVAILABLE 桩）',
                        False, '桩库未报错：可能伪造了原生成功')

    tally.check(group, 'UNAVAILABLE 原因可操作（给出构建/环境线索）',
                info['backend_enabled']
                or ('Einsums' in info['reason']
                    or 'einsums' in info['reason']),
                str(info['reason'])[:110])


def _print_header(info):
    print('════════ verify_cpu_einsums：Einsums 字符串 einsum 契约 vs numpy '
          '════════')
    print(f'文档依据   : {_contract.DOC_URL}（{_contract.DOC_VERSION}）')
    print(f'            : {_contract.DOC_EINSUM_URL}')
    print(f'上游仓库   : {_contract.DOC_SOURCE_URL}')
    print('原生桥     : ' + ('可加载' if info['available'] else '不可用')
          + f'（{info["library"]}，abi={info["abi_version"]}'
          + f'，backend_enabled={info["backend_enabled"]}）')
    if not info['available']:
        print(f'             SKIP 原因: {info["reason"][:160]}')
    print(f'语料       : DOC_CASES {len(_contract.DOC_CASES)} 条'
          f'（documented {len(_contract.documented_cases())} / '
          f'inferred {len(_contract.inferred_cases())}）'
          f' / LIMITATIONS {len(_contract.LIMITATIONS)} 条')


def run(seed=20260831, verbose=False):
    """执行全部分组断言，返回 ``(tally, native_skipped, reason)``。"""
    tally = Tally()
    info = _bridge.probe()
    _print_header(info)
    print(f'随机种子   : seed={seed}（每条用例由 crc32(name) 派生）')

    _section_contract(tally, info)
    _section_spec(tally)
    _section_doc_case(tally, info, seed)
    _section_native(tally, info)

    passed, failed, skip_count = tally.counts()
    total = passed + failed
    print('\n──────────── verify_cpu_einsums 汇总 ────────────')
    for group in GROUPS:
        group_passed, group_failed, group_skipped = tally.group_counts(group)
        executed = group_passed + group_failed
        if executed + group_skipped == 0:
            continue
        if executed == 0:
            print(f'{group:<12s} SKIP   {info["reason"][:70]}')
            continue
        state = 'PASS' if group_failed == 0 else f'FAIL({group_failed})'
        suffix = f'  SKIP {group_skipped}' if group_skipped else ''
        print(f'{group:<12s} {group_passed}/{executed} {state}{suffix}')
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
    return tally, not info['available'], info['reason']


def build_payload(tally, native_skipped, reason, seed):
    """机器可读摘要（``--json``）。"""
    passed, failed, skip_count = tally.counts()
    info = _bridge.probe()
    groups = {}
    for group in GROUPS:
        group_passed, group_failed, group_skipped = tally.group_counts(group)
        if group_passed + group_failed + group_skipped == 0:
            continue
        groups[group] = {'passed': group_passed, 'failed': group_failed,
                         'skipped': group_skipped}
    return {
        'ok': failed == 0,
        'verdict': ('FAIL' if failed
                    else 'SKIP' if native_skipped else 'PASS'),
        'passed': passed,
        'total': passed + failed,
        'failed': failed,
        'skipped': skip_count,
        'native_skipped': bool(native_skipped),
        'skip_reason': reason,
        'seed': seed,
        'doc': {'url': _contract.DOC_URL,
                'einsum_url': _contract.DOC_EINSUM_URL,
                'source': _contract.DOC_SOURCE_URL,
                'version': _contract.DOC_VERSION},
        'corpus': {'doc_cases': len(_contract.DOC_CASES),
                   'documented': len(_contract.documented_cases()),
                   'inferred': len(_contract.inferred_cases()),
                   'limitations': len(_contract.LIMITATIONS)},
        'abi': {'version': _bridge.ABI_VERSION,
                'max_operands': _bridge.MAX_OPERANDS,
                'max_rank': _bridge.MAX_RANK,
                'max_out_ndim': _bridge.MAX_OUT_NDIM,
                'library_env': _bridge.LIBRARY_ENV},
        'bridge': info,
        'groups': groups,
        'failures': [{'group': row['group'], 'name': row['name'],
                      'detail': row['detail']}
                     for row in tally.rows if row['ok'] is False],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Einsums 字符串 einsum 契约验证（语料 + spec 层 + C ABI）')
    parser.add_argument('--seed', type=int, default=20260831,
                        help='随机种子（默认 20260831）')
    parser.add_argument('--verbose', action='store_true',
                        help='打印每条断言的 detail')
    parser.add_argument('--json', action='store_true', dest='as_json',
                        help='只输出机器可读 JSON 摘要（优先于人类可读输出）')
    args = parser.parse_args(argv)

    if args.as_json:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            tally, native_skipped, reason = run(seed=args.seed,
                                                verbose=args.verbose)
        payload = build_payload(tally, native_skipped, reason, args.seed)
        print(json.dumps(payload, ensure_ascii=False, indent=2,
                         sort_keys=True))
        return 1 if payload['failed'] else 0

    tally, native_skipped, reason = run(seed=args.seed, verbose=args.verbose)
    _passed, failed, _skipped = tally.counts()
    if failed:
        print('FAIL verify_cpu_einsums')
        return 1
    if native_skipped:
        print('SKIP verify_cpu_einsums：原生库不可用，native 组未执行'
              f'（{reason[:120]}）')
        return 0
    print('PASS verify_cpu_einsums')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
