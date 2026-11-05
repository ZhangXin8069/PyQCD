"""CUDA 路径验证：Cython 桥接 / 原生规划器 / 数值 / 降级 / 计时（独立可跑）。

五个面：
- 桥接面：`pyqcd.einsum.cuda.einsum` 可导入，`backend_info()` / 四个入口齐全，
  非法地址 fail-closed。不可导入时打印原因并 SKIP（exit 0），其余面照跑。
- 规划面：>= 50 个随机表达式，Cython 入口与 ctypes（KUPL）入口给出逐位相同的
  `(path, opt_cost, largest_intermediate)`。
- 数值面：>= 20 条方程，numpy 与 cupy 结果 `rel <= 1e-12`；无 GPU 则 SKIP。
- 降级面：`PYQCD_EINSUM_NATIVE=kupl|cuda` 与 `PYQCD_EINSUM_LIB=/nonexistent`
  各跑一次 `contract`，结果与 `numpy.einsum` 逐位一致且不抛异常（子进程隔离，
  因为原生加载失败在进程内是粘性的）。
- 计时面：同后端同 dtype，`median(>= 5)`，打印 `speedup_vs_opt`。未达标项如实打印，
  默认只记录、不改变退出码（性能不是正确性门槛）；`--strict-timing` 可强制失败。

用法：
    python3 -m pyqcd.testing.einsum.verify_cuda
    python3 -m pyqcd.testing.einsum.verify_cuda --smoke
    python3 -m pyqcd.testing.einsum.verify_cuda --module /tmp/einsum.so   # 旁路 .so

`--module` 用于树内产物暂时不可导入（例如原生库尚未导出 `applyContractQcd`）时，
直接加载一份带相同模块名的编译产物做桥接面/规划面验证，不影响其余面。
"""
from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import sysconfig
import time
import zlib

import numpy as np

from . import _oracle

COMPILED_NAME = 'pyqcd.einsum.cuda.einsum'
TOL_REL = 1e-12

TIMING_CASES = (
    ('chain4', 'ab,bc,cd,de->ae',
     [(24, 24), (24, 24), (24, 32), (32, 16)]),
    ('three_hop', 'abc,cd,de->abe',
     [(8, 16, 24), (24, 24), (24, 12)]),
    ('blas_chain', 'ij,jk,kl->il',
     [(48, 40), (40, 40), (40, 24)]),
    ('hadamard_reduce', 'ij,ij,ij->i',
     [(64, 64), (64, 64), (64, 64)]),
    ('chain4_big', 'ab,bc,cd,de->ae', [(96, 96)] * 4),
    ('matmul_big', 'ij,jk,kl->il',
     [(256, 192), (192, 192), (192, 128)]),
)

CURATED_EQUATIONS = (
    ('matmul', 'ij,jk->ik', [(5, 4), (4, 6)]),
    ('chain', 'ab,bc,cd->ad', [(4, 5), (5, 6), (6, 3)]),
    ('hadamard', 'ij,ij->ij', [(4, 5), (4, 5)]),
    ('trace', 'ii->', [(6, 6)]),
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
        print(f'  SKIP [{group}] {name}: {reason}')

    def counts(self):
        passed = sum(1 for row in self.rows if row['ok'] is True)
        failed = sum(1 for row in self.rows if row['ok'] is False)
        skipped = sum(1 for row in self.rows if row['ok'] is None)
        return passed, failed, skipped

    def group_counts(self, group):
        rows = [row for row in self.rows if row['group'] == group]
        passed = sum(1 for row in rows if row['ok'] is True)
        failed = sum(1 for row in rows if row['ok'] is False)
        skipped = sum(1 for row in rows if row['ok'] is None)
        return passed, failed, skipped

    def failures(self, limit=5):
        return [row for row in self.rows if row['ok'] is False][:limit]

    def report(self):
        for group in dict.fromkeys(row['group'] for row in self.rows):
            passed, failed, skipped = self.group_counts(group)
            print(f'  [{group}] PASS {passed} / FAIL {failed} / SKIP {skipped}')
        for row in self.failures():
            print(f'  FAIL [{row["group"]}] {row["name"]}: {row["detail"]}')


def _exc_text(exc):
    return f'{type(exc).__name__}: {exc}'


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


def _seed_of(name):
    return zlib.crc32(name.encode('utf-8')) % 9973


##### 桥接面


def load_bridge(module_path=None):
    """返回 `(模块 | None, 原因)`；导入失败不抛给调用方。"""
    if module_path:
        try:
            spec = importlib.util.spec_from_file_location(
                COMPILED_NAME, module_path)
            if spec is None or spec.loader is None:
                return None, f'无法为 {module_path} 构造 import loader'
            module = importlib.util.module_from_spec(spec)
            sys.modules[COMPILED_NAME] = module
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - 桥接缺失只报告
            sys.modules.pop(COMPILED_NAME, None)
            return None, f'{module_path} 加载失败: {_exc_text(exc)}'
        return module, ''

    try:
        from pyqcd.einsum.cuda import einsum as module
    except Exception as exc:  # noqa: BLE001
        return None, f'{COMPILED_NAME} 不可导入: {_exc_text(exc)}'
    return module, ''


def _control_arrays():
    set_ptrs = np.zeros(16, dtype=np.int64)
    params = np.zeros(32, dtype=np.int32)
    return set_ptrs, params


def _section_bridge(tally, bridge, reason):
    group = '桥接面'
    if bridge is None:
        tally.skip(group, '编译桥接可导入', reason)
        return
    tally.check(group, '编译桥接可导入', True, getattr(bridge, '__file__', ''))

    try:
        info = bridge.backend_info()
    except Exception as exc:  # noqa: BLE001
        info = _exc_text(exc)
    tally.check(group, 'backend_info() == {"backend": "cuda", "compiled": True}',
                info == {'backend': 'cuda', 'compiled': True}, info)

    for name in ('applyPlanQcd', 'applyContractQcd',
                 'applyInitQcd', 'applyEndQcd'):
        tally.check(group, f'{name} 存在且可调用',
                    callable(getattr(bridge, name, None)))

    rejects = (
        ('applyPlanQcd', (0, 0), ValueError, '空地址'),
        ('applyPlanQcd', (-8, 8), ValueError, '负地址'),
        ('applyPlanQcd', (8, 1.5), TypeError, '非整数地址'),
        ('applyContractQcd', (8, 8, 8, 0), ValueError, '空 params'),
    )
    for name, arguments, expected, label in rejects:
        entry = getattr(bridge, name, None)
        if not callable(entry):
            tally.check(group, f'{name} 拒绝{label}', False, '入口缺失')
            continue
        try:
            entry(*arguments)
        except expected as exc:
            tally.check(group, f'{name} 拒绝{label}', True, _exc_text(exc))
        except Exception as exc:  # noqa: BLE001
            tally.check(group, f'{name} 拒绝{label}', False,
                        f'期望 {expected.__name__}，实际 {_exc_text(exc)}')
        else:
            tally.check(group, f'{name} 拒绝{label}', False, '未抛异常')

    for name in ('applyInitQcd', 'applyEndQcd'):
        entry = getattr(bridge, name, None)
        if not callable(entry):
            tally.check(group, f'{name} 无会话时可调用', False, '入口缺失')
            continue
        set_ptrs, params = _control_arrays()
        try:
            result = entry(int(set_ptrs.ctypes.data), int(params.ctypes.data))
        except Exception as exc:  # noqa: BLE001
            tally.check(group, f'{name} 无会话时可调用', False, _exc_text(exc))
        else:
            tally.check(group, f'{name} 无会话时可调用', result is None, result)


##### 规划面


def _ctypes_entry():
    """返回 `(kupl 模块, applyPlanQcd)`；KUPL 库不可用时返回 (模块, None)。"""
    from pyqcd.einsum.kupl import einsum as kupl

    try:
        library = kupl._load_library()
    except Exception:  # noqa: BLE001 - 库缺失即无 ctypes 参照
        return kupl, None
    return kupl, getattr(library, 'applyPlanQcd', None)


def _network_request(network):
    left, _, right = network['eq'].partition('->')
    return left.split(','), right, network['size_dict']


def _section_plan(tally, bridge, networks, minimum=50):
    group = '规划面'
    if bridge is None:
        tally.skip(group, 'applyPlanQcd：Cython vs ctypes', '编译桥接不可用')
        return
    kupl, ctypes_entry = _ctypes_entry()
    if ctypes_entry is None:
        tally.skip(group, 'applyPlanQcd：Cython vs ctypes',
                   f'KUPL 原生库不可用: {kupl.library_path()}')
        return

    compared = 0
    for index, network in enumerate(networks):
        inputs, output, sizes = _network_request(network)
        label = f'net{index:03d}[{len(inputs)}ops] {network["eq"]}'
        try:
            got = kupl._plan_with(bridge.applyPlanQcd, inputs, output, sizes)
            ref = kupl._plan_with(ctypes_entry, inputs, output, sizes)
        except Exception as exc:  # noqa: BLE001 - 单项失败不影响其余断言
            tally.check(group, label, False, _exc_text(exc))
            continue
        if ref is None:
            tally.check(group, label, False, 'ctypes 参照未给出方案')
            continue
        compared += 1
        tally.check(group, label, got == ref, f'cython={got} ctypes={ref}')
    if minimum:
        tally.check(group, f'比较样本数 >= {minimum}（实际 {compared}）',
                    compared >= minimum)


##### 数值面


def _require_cupy():
    try:
        import cupy
    except Exception as exc:  # noqa: BLE001
        return None, f'cupy 不可导入: {_exc_text(exc)}'
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:  # noqa: BLE001
        return None, f'cupy 设备探测失败: {_exc_text(exc)}'
    if count <= 0:
        return None, '无可见 CUDA 设备'
    return cupy, ''


def build_equations(n_random, seed):
    cases = [{'name': f'curated_{name}', 'eq': eq,
              'shapes': [tuple(shape) for shape in shapes]}
             for name, eq, shapes in CURATED_EQUATIONS]
    if n_random > 0:
        for index, network in enumerate(_oracle.rand_networks(
                n_random, seed=seed, regularity=3, n_out=2,
                d_min=2, d_max=3)):
            cases.append({'name': f'random_{index:03d}', 'eq': network['eq'],
                          'shapes': network['shapes']})
    return cases


def _section_numeric(tally, cases, minimum=20):
    group = '数值面'
    cupy, reason = _require_cupy()
    if cupy is None:
        tally.skip(group, 'pyqcd.einsum.contract：numpy vs cupy', reason)
        return

    import pyqcd.einsum as target

    for case in cases:
        seed = _seed_of(case['name'])
        arrays = _oracle.rand_arrays(case, seed=seed)
        try:
            got_np = target.contract(case['eq'], *arrays, backend='numpy')
        except Exception as exc:  # noqa: BLE001
            tally.check(group, f'{case["name"]} numpy', False, _exc_text(exc))
            continue
        rel_np, delta_np = _diff_stats(
            got_np, np.einsum(case['eq'], *arrays, optimize=True))
        tally.check(group, f'{case["name"]}[{case["eq"]}] numpy vs np.einsum',
                    rel_np is not None and rel_np <= TOL_REL,
                    f'rel={rel_np}, max|d|={delta_np}')

        gpu_arrays = [cupy.asarray(item) for item in arrays]
        try:
            got_gpu = target.contract(case['eq'], *gpu_arrays, backend='cupy')
            cupy.cuda.Stream.null.synchronize()
            host = cupy.asnumpy(got_gpu)
        except Exception as exc:  # noqa: BLE001
            tally.check(group, f'{case["name"]}[{case["eq"]}] cupy vs numpy',
                        False, _exc_text(exc))
            continue
        rel, delta = _diff_stats(host, got_np)
        tally.check(group, f'{case["name"]}[{case["eq"]}] cupy vs numpy',
                    rel is not None and rel <= TOL_REL,
                    f'rel={rel}, max|d|={delta}')
    if minimum:
        tally.check(group, f'方程数 >= {minimum}（实际 {len(cases)}）',
                    len(cases) >= minimum)


##### 降级面

_FALLBACK_CODE = r'''
import json
import numpy as np
import pyqcd.einsum as target
from pyqcd.einsum import _native

rng = np.random.default_rng(7)
left = rng.integers(-4, 5, size=(6, 5)).astype(np.float64) + 0.0j
right = rng.integers(-4, 5, size=(5, 4)).astype(np.float64) + 0.0j
got = np.asarray(target.contract('ij,jk->ik', left, right))
ref = np.einsum('ij,jk->ik', left, right)
print(json.dumps({
    'shape': list(got.shape),
    'bitwise': bool(np.array_equal(got, ref)),
    'max_abs': float(np.max(np.abs(got - ref))),
    'backend': _native.backend_name(),
    'requested': _native.requested_backend(),
    'library': str(_native.library_path()),
}))
'''


def _repo_root():
    import pathlib
    return str(pathlib.Path(__file__).resolve().parents[3])


def _run_fallback_case(extra_env):
    """在子进程里跑一次 contract，返回 `(结果 dict | None, 原因)`。"""
    import json
    import os
    import subprocess

    env = dict(os.environ)
    env.pop('PYQCD_EINSUM_LIB', None)
    root = _repo_root()
    env['PYTHONPATH'] = root + (
        os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.update(extra_env)
    completed = subprocess.run([sys.executable, '-c', _FALLBACK_CODE],
                               capture_output=True, text=True, env=env,
                               timeout=600)
    if completed.returncode != 0:
        lines = completed.stderr.strip().splitlines()
        return None, (lines or [''])[-1]
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1]), ''
    except (IndexError, ValueError) as exc:
        return None, _exc_text(exc)


def _section_fallback(tally):
    """无原生库时回退纯 Python 规划器：数值逐位不变、不抛异常。"""
    group = '降级面'
    missing = '/nonexistent/libpyqcdeinsum.so'
    cases = (
        ('PYQCD_EINSUM_NATIVE=kupl', {'PYQCD_EINSUM_NATIVE': 'kupl'}, False),
        ('PYQCD_EINSUM_NATIVE=cuda', {'PYQCD_EINSUM_NATIVE': 'cuda'}, False),
        ('PYQCD_EINSUM_NATIVE=auto + 无 .so',
         {'PYQCD_EINSUM_NATIVE': 'auto', 'PYQCD_EINSUM_LIB': missing}, True),
        ('PYQCD_EINSUM_NATIVE=kupl + 无 .so',
         {'PYQCD_EINSUM_NATIVE': 'kupl', 'PYQCD_EINSUM_LIB': missing}, True),
    )
    for label, env, expect_python in cases:
        result, reason = _run_fallback_case(env)
        if result is None:
            tally.check(group, label, False, reason)
            continue
        ok = result['bitwise'] and result['shape'] == [6, 4]
        if expect_python:
            ok = ok and result['backend'] == 'python'
        tally.check(group, label, ok,
                    f'backend={result["backend"]} '
                    f'library={result["library"]} '
                    f'bitwise={result["bitwise"]} max|d|={result["max_abs"]}')


##### 计时面


def _median_seconds(call, runs, synchronize=None):
    """返回 `(median, minimum)`；min 一并保留用于暴露同机并发构建带来的抖动。"""
    call()
    if synchronize is not None:
        synchronize()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        call()
        if synchronize is not None:
            synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), min(samples)


def _format_ms(seconds):
    return f'{seconds * 1e3:.3f} ms'


def _section_timing(tally, runs, min_speedup):
    """同后端、同 dtype 与 opt_einsum 对比；不达标只记录不篡改。"""
    group = '计时面'
    import pyqcd.einsum as target

    try:
        opt_einsum = _oracle.oracle()
    except Exception as exc:  # noqa: BLE001
        tally.skip(group, 'opt_einsum 参照不可用', _exc_text(exc))
        return

    cupy, reason = _require_cupy()
    backends = [('numpy', 'complex128', None)]
    if cupy is not None:
        backends.append(('cupy', 'complex128', cupy.cuda.Stream.null.synchronize))
    else:
        print(f'  SKIP [计时面] cupy 后端: {reason}')

    measurements = []
    for backend, dtype, synchronize in backends:
        for name, eq, shapes in TIMING_CASES:
            arrays = _oracle.rand_arrays({'shapes': shapes}, seed=_seed_of(name),
                                         dtype=dtype)
            if backend == 'cupy':
                arrays = [cupy.asarray(item) for item in arrays]
            try:
                ours, ours_best = _median_seconds(
                    lambda: target.contract(eq, *arrays, backend=backend),
                    runs, synchronize)
                reference, reference_best = _median_seconds(
                    lambda: opt_einsum.contract(eq, *arrays, optimize='auto'),
                    runs, synchronize)
            except Exception as exc:  # noqa: BLE001
                tally.check(group, f'{backend}:{name} 计时', False, _exc_text(exc))
                continue
            speedup = reference / ours if ours else float('inf')
            measurements.append((backend, name, ours, reference, speedup))
            print(f'  [{backend}/{dtype}] {name}: pyqcd={_format_ms(ours)} '
                  f'(min {_format_ms(ours_best)}) '
                  f'opt_einsum={_format_ms(reference)} '
                  f'(min {_format_ms(reference_best)}) '
                  f'speedup_vs_opt={speedup:.3f}x')
            tally.check(group, f'{backend}:{name} speedup_vs_opt >= {min_speedup}',
                        speedup >= min_speedup, f'{speedup:.3f}x')

    if measurements:
        median_speedup = statistics.median(item[4] for item in measurements)
        below = [item for item in measurements if item[4] < min_speedup]
        print(f'  median speedup_vs_opt = {median_speedup:.3f}x '
              f'（{len(measurements)} 组，达标线 {min_speedup:.2f}x，'
              f'未达标 {len(below)} 组）')
        tally.check(group, f'median speedup_vs_opt >= {min_speedup}',
                    median_speedup >= min_speedup, f'{median_speedup:.3f}x')


##### 入口


def run(networks=60, equations=24, median_runs=5, min_speedup=1.0,
        module_path=None, seed=20260830, strict_counts=True):
    tally = Tally()
    bridge, reason = load_bridge(module_path)
    if bridge is None:
        print(f'桥接: SKIP（{reason}）')
    else:
        print(f'桥接: {getattr(bridge, "__file__", COMPILED_NAME)}')
        print(f'backend_info: {bridge.backend_info()}')

    _section_bridge(tally, bridge, reason)
    _section_plan(tally, bridge, _oracle.rand_networks(
        networks, seed=seed, regularity=3, n_out=2, d_min=2, d_max=4),
        minimum=50 if strict_counts else 0)
    _section_numeric(tally, build_equations(equations, seed),
                     minimum=20 if strict_counts else 0)
    _section_fallback(tally)
    _section_timing(tally, median_runs, min_speedup)
    return tally


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--networks', type=int, default=60,
                        help='规划面随机表达式数（默认 60，须 >= 50）')
    parser.add_argument('--equations', type=int, default=24,
                        help='数值面随机方程数（默认 24，+4 条人工方程）')
    parser.add_argument('--median', type=int, default=5,
                        help='计时面 median 次数（默认 5）')
    parser.add_argument('--min-speedup', type=float, default=1.0,
                        help='speedup_vs_opt 达标线（默认 1.0）')
    parser.add_argument('--strict-timing', action='store_true',
                        help='计时面的未达标项也计入退出码（默认只如实打印）')
    parser.add_argument('--module', default=None,
                        help='显式加载编译产物 .so（树内产物不可导入时使用）')
    parser.add_argument('--seed', type=int, default=20260830)
    parser.add_argument('--smoke', action='store_true',
                        help='小样本冒烟（6 网络 / 4 随机方程 / median 3）')
    args = parser.parse_args(argv)
    networks = 6 if args.smoke else args.networks
    equations = 4 if args.smoke else args.equations
    median_runs = 3 if args.smoke else args.median

    print('════════ verify_cuda：Cython 桥接与 CUDA 路径 ════════')
    print(f'python={sys.version.split()[0]} '
          f'platform={sysconfig.get_platform()}')
    tally = run(networks=networks, equations=equations,
                median_runs=median_runs, min_speedup=args.min_speedup,
                module_path=args.module, seed=args.seed,
                strict_counts=not args.smoke)
    passed, failed, skipped = tally.counts()
    print('────────────────────────────────────────────')
    tally.report()
    print(f'PASS {passed} / FAIL {failed} / SKIP {skipped}')
    correctness_failed = sum(1 for row in tally.rows
                             if row['ok'] is False and row['group'] != '计时面')
    timing_failed = failed - correctness_failed
    if timing_failed:
        print(f'计时面未达标 {timing_failed} 项（performance，不改变退出码；'
              f'--strict-timing 可强制失败）')
    if correctness_failed or (args.strict_timing and timing_failed):
        print('FAIL verify_cuda')
        return 1
    print('PASS verify_cuda' + ('（含 SKIP）' if skipped else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
