"""蒸馏实战基准：`numpy.einsum` 朴素 / `opt_einsum.contract` / `pyqcd.einsum.contract`。

真实数据（只读，不复制）：`/public/group/lqcd/perambulators/
beta6.20_mu-0.2770_ms-0.2400_L24x72/light/4150/perams.4150.{0,1,2,3}.{t_src}`
每个文件 46,080,000 字节（`<f8` 5,760,000 个数 = Nt×Nvec×Nvec 的实/虚部对）。
四个 spin 分片拼成 `reshape(4, 72, 100, 4, 100, 2)` → 复化 →
`transpose(1, 3, 0, 4, 2)` 得 `[Nt, 4, 4, Nvec, Nvec]`。

基准方程取自 `refer/sush/_distillation/contraction.block.cupy.gen.py`
（行号已核对：E1:165、E2:234、E4:211、E5:200）。

口径：同 backend、同 dtype（complex128）、同线程数；预热 2 次后取 ≥10 次中位数；
GPU 只与 GPU 比、CPU 只与 CPU 比；`speedup_vs_opt = median(opt_einsum) /
median(本实现)`。不足 3.0 时如实打印，不做任何调参美化。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime

# 线程数必须在 numpy 导入前钉住；main.py 入口保证本模块先于 numpy 被导入。
THREADS = int(os.environ.get('PYQCD_BENCH_THREADS', '8') or 8)
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, str(THREADS))

import numpy as np  # noqa: E402 - 必须在钉住线程环境变量之后导入

from . import _oracle, _target  # noqa: E402

DTYPE = 'complex128'
SPEEDUP_TARGET = 3.0

DATA_ROOT = ('/public/group/lqcd/perambulators/'
             'beta6.20_mu-0.2770_ms-0.2400_L24x72/light/4150')
DATA_CONF = 4150
NT_FULL = 72
NVEC_FULL = 100
SPIN_FILES = 4
FILE_BYTES = 46_080_000
VALUES_PER_FILE = FILE_BYTES // 8          # 5_760_000

EQUATIONS = (
    {'name': 'E1', 'line': 165, 'desc': '8 算子 2pt',
     'eq': 'gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M'},
    {'name': 'E2', 'line': 234, 'desc': '8 算子 3pt',
     'eq': 'Geifj,ckdl,agbh,ce,gi,Mbdf,Mhjl,Gka->GM'},
    {'name': 'E4', 'line': 211, 'desc': '4 算子 peram×gauge×γ',
     'eq': 'zyxcTabn,zyxgTdef,Gae,Tzyxgc->TGdbfn'},
    {'name': 'E5', 'line': 200, 'desc': 'γ 夹心',
     'eq': 'ab,...bce,cd->...dae'},
)
EQUATION_BY_NAME = {item['name']: item for item in EQUATIONS}

PROFILES = {
    'T0n': {'nvec': 6, 'spatial': 1, 'n_t': 2, 't_src': (0,), 't_sink': 4,
            'note': '朴素 numpy 可行探针（最小档，用于给出朴素基线实数）'},
    'T0': {'nvec': 20, 'spatial': 1, 'n_t': 2, 't_src': (0,), 't_sink': 4,
           'note': 'CI 规模 Nvec=20'},
    'T1': {'nvec': 100, 'spatial': 1, 'n_t': 2, 't_src': (0,), 't_sink': 8,
           'note': '单 t_src，全 Nvec=100（真实数据）'},
    'T2': {'nvec': 100, 'spatial': 1, 'n_t': 2, 't_src': (0, 1), 't_sink': 8,
           'note': '多 t_src 持续吞吐（只跑 opt_einsum / pyqcd）',
           'impls': ('opt_einsum_contract', 'pyqcd_einsum_contract')},
}
DEFAULT_PROFILES = ('T0n', 'T0', 'T1', 'T2')
DEFAULT_BACKENDS = ('cpu', 'gpu')
BACKEND_NAMES = {'cpu': 'numpy', 'gpu': 'cupy'}


# ═══════════════════════════════════════════════════════════════════
# 真实数据（只读）
# ═══════════════════════════════════════════════════════════════════

def data_path(t_src, spin):
    return os.path.join(DATA_ROOT, f'perams.{DATA_CONF}.{spin}.{t_src}')


def data_status(t_src_list):
    """返回缺失/尺寸不符的文件清单（空列表表示数据齐全）。"""
    problems = []
    for t_src in t_src_list:
        for spin in range(SPIN_FILES):
            path = data_path(t_src, spin)
            if not os.path.isfile(path):
                problems.append(f'缺失 {path}')
            elif os.path.getsize(path) != FILE_BYTES:
                problems.append(f'{path} 大小 {os.path.getsize(path)} '
                                f'!= {FILE_BYTES}')
    return problems


def load_peram(t_src, nvec):
    """读取 4 个 spin 分片，返回 `[Nt, 4, 4, Nvec, Nvec]` complex128。"""
    slices = []
    for spin in range(SPIN_FILES):
        buffer = np.fromfile(data_path(t_src, spin), dtype='<f8')
        if buffer.size != VALUES_PER_FILE:
            raise ValueError(f'{data_path(t_src, spin)} 元素数 '
                             f'{buffer.size} != {VALUES_PER_FILE}')
        slices.append(buffer)
    cube = np.stack(slices).reshape(
        SPIN_FILES, NT_FULL, NVEC_FULL, SPIN_FILES, NVEC_FULL, 2)
    cube = cube[..., 0] + 1j * cube[..., 1]
    peram = cube.transpose(1, 3, 0, 4, 2)
    peram = peram[:, :, :, :nvec, :nvec]
    return np.ascontiguousarray(peram, dtype=np.complex128)


# ═══════════════════════════════════════════════════════════════════
# 操作数构造（peram 真实，VVV / 传播子 / 规范场为可复现合成数据）
# ═══════════════════════════════════════════════════════════════════

def _gamma(index):
    """DeGrand-Rossi γ 矩阵（来自被测仓库自身的 lattice 层，只读使用）。"""
    try:
        from pyqcd.lattice import gamma
        return np.asarray(gamma(index), dtype=np.complex128)
    except Exception:  # noqa: BLE001 - γ 只是基准操作数，缺失时退化为随机矩阵
        rng = np.random.default_rng(index)
        base = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        return base / np.linalg.norm(base)


def _projector():
    return np.asarray(
        (_gamma(0) + _gamma(4)) / 2.0, dtype=np.complex128)


def _rand(shape, rng):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)).astype(np.complex128)


def build_operands(name, peram, cfg, seed):
    """按参考脚本的算符形状构造一次基准的全部操作数。"""
    rng = np.random.default_rng(seed)
    nvec = cfg['nvec']
    n_t = cfg['n_t']
    side = cfg['spatial']
    t_sink = min(cfg['t_sink'], peram.shape[0] - 1)
    if name == 'E1':
        sl = peram[t_sink]
        g7 = _gamma(7)
        return [sl, sl, sl, g7, g7,
                _rand((1, nvec, nvec, nvec), rng),
                _rand((1, nvec, nvec, nvec), rng),
                _projector()]
    if name == 'E2':
        current = peram[t_sink][None]                  # 'Geifj'，G 取单切片
        src_a = peram[0]
        src_b = peram[1 % peram.shape[0]]
        g7 = _gamma(7)
        return [current, src_a, src_b, g7, g7,
                _rand((1, nvec, nvec, nvec), rng),
                _rand((1, nvec, nvec, nvec), rng),
                _projector()[None]]
    prop_shape = (side, side, side, 3, n_t, 4, 4, nvec)
    if name == 'E4':
        prop_src = _rand(prop_shape, rng)
        prop_sep = _rand(prop_shape, rng)
        gauge = _rand((n_t, side, side, side, 3, 3), rng)
        return [prop_src, prop_sep, _gamma(3)[None], gauge]
    if name == 'E5':
        prop_sep = _rand(prop_shape, rng)
        g5 = _gamma(5)
        return [g5, np.conj(prop_sep), g5]
    raise KeyError(f'未知方程 {name}')


def operand_shapes(name, cfg, peram):
    """只取形状（用于朴素代价估计与报告）。"""
    key = ('peram', cfg['n_t'])
    del key
    return [tuple(array.shape)
            for array in build_operands(name, peram, cfg, seed=0)]


# ═══════════════════════════════════════════════════════════════════
# 代价估计
# ═══════════════════════════════════════════════════════════════════

def size_dict_of(eq, shapes):
    lhs = eq.split('->')[0]
    terms = lhs.split(',')
    sizes = {}
    for term, shape in zip(terms, shapes):
        if '...' in term:
            head, tail = term.split('...')
            if head:
                raise ValueError(f'暂不估计前缀 ellipsis: {eq}')
            n_ellipsis = len(shape) - len(tail)
            pairs = [(f'@{i}', d) for i, d in enumerate(shape[:n_ellipsis])]
            pairs += list(zip(tail, shape[n_ellipsis:]))
        else:
            pairs = list(zip(term, shape))
        for label, dim in pairs:
            dim = int(dim)
            sizes[label] = max(sizes.get(label, 1), dim)
    return sizes


def naive_op_count(eq, shapes):
    """朴素 einsum 的标量乘加次数估计 = 所有唯一索引尺寸之积。"""
    total = 1
    for dim in size_dict_of(eq, shapes).values():
        total *= dim
    return total


def optimized_cost(eq, shapes):
    """oracle 的 dp 规划代价（Decimal），用于报告对照。"""
    module = _oracle.oracle()
    _path, info = module.contract_path(eq, *shapes, shapes=True, optimize='dp')
    return info.opt_cost


# ═══════════════════════════════════════════════════════════════════
# 计时与显存
# ═══════════════════════════════════════════════════════════════════

def _sync(backend):
    if backend == 'gpu':
        import cupy

        cupy.cuda.Stream.null.synchronize()


def _pool_used_bytes():
    try:
        import cupy

        return int(cupy.get_default_memory_pool().used_bytes())
    except Exception:  # noqa: BLE001
        return 0


def _cpu_peak_bytes(fn):
    """tracemalloc 峰值（单次调用，不计时；numpy 数据分配会被追踪）。"""
    import tracemalloc

    tracemalloc.start()
    try:
        fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return int(peak)


def measure(fn, backend, iterations, warmup=2):
    """预热后取 iterations 次中位数，同时记录峰值显存/内存。"""
    for _ in range(warmup):
        fn()
    _sync(backend)
    base = _pool_used_bytes() if backend == 'gpu' else 0
    peak = 0
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        _sync(backend)
        times.append(time.perf_counter() - start)
        if backend == 'gpu':
            peak = max(peak, _pool_used_bytes() - base)
    stats = {'median_s': statistics.median(times),
             'min_s': min(times),
             'max_s': max(times),
             'stdev_s': statistics.pstdev(times) if len(times) > 1 else 0.0,
             'iterations': iterations}
    if backend == 'gpu':
        stats['peak_bytes'] = int(peak)
    else:
        stats['peak_bytes'] = _cpu_peak_bytes(fn)
    return stats


# ═══════════════════════════════════════════════════════════════════
# 三个实现的登记
# ═══════════════════════════════════════════════════════════════════

def implementations(backend, optimize, target_module):
    """返回 [(名称, callable(eq, operands))]，顺序即打印顺序。"""
    backend_name = BACKEND_NAMES[backend]
    entries = []
    if backend == 'cpu':
        entries.append(
            ('numpy_einsum_naive',
             lambda eq, ops: np.einsum(eq, *ops, optimize=False)))
        entries.append(
            ('numpy_einsum_path',
             lambda eq, ops: np.einsum(eq, *ops, optimize=optimize)))
    else:
        import cupy

        entries.append(
            ('cupy_einsum_naive',
             lambda eq, ops: cupy.einsum(eq, *ops, optimize=False)))
        # cupy.einsum 只认 True/False/'greedy'/'optimal'，没有 'dp'
        entries.append(
            ('cupy_einsum_path',
             lambda eq, ops: cupy.einsum(eq, *ops, optimize='optimal')))
    oracle_module = _oracle.oracle()
    entries.append(
        ('opt_einsum_contract',
         lambda eq, ops: oracle_module.contract(
             eq, *ops, optimize=optimize, backend=backend_name)))
    if target_module is not None:
        entries.append(
            ('pyqcd_einsum_contract',
             lambda eq, ops: target_module.contract(
                 eq, *ops, optimize=optimize, backend=backend_name)))
    return entries


def _to_backend(arrays, backend):
    if backend == 'gpu':
        import cupy

        return [cupy.asarray(array) for array in arrays]
    return arrays


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def run(profiles=DEFAULT_PROFILES, backends=DEFAULT_BACKENDS,
        equations=tuple(item['name'] for item in EQUATIONS),
        iterations=10, warmup=2, naive_budget=2e9, optimize='dp',
        log=print):
    """执行基准，返回 (rows, env, skipped, reasons)。"""
    target_module, missing = _target.probe()
    if missing:
        target_module = None
    oracle_version, oracle_path = _oracle.oracle_version()

    env = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hostname': platform.node(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'opt_einsum': oracle_version,
        'opt_einsum_path': oracle_path,
        'pyqcd_einsum': ('ready' if target_module is not None
                         else f'not ready: {_target.skip_reason()}'),
        'dtype': DTYPE,
        'threads': THREADS,
        'cpu_count': os.cpu_count(),
        'optimize': optimize,
        'iterations': iterations,
        'warmup': warmup,
        'profiles': list(profiles),
        'backends': list(backends),
        'equations': list(equations),
        'naive_budget_ops': naive_budget,
    }
    try:
        import cupy

        env['cupy'] = cupy.__version__
        free, total = cupy.cuda.Device(0).mem_info
        env['gpu'] = {'name': cupy.cuda.runtime.getDeviceProperties(0)['name']
                      .decode() if isinstance(
                          cupy.cuda.runtime.getDeviceProperties(0)['name'],
                          bytes) else str(
                              cupy.cuda.runtime.getDeviceProperties(0)['name']),
                      'free_mb': round(free / 2 ** 20, 1),
                      'total_mb': round(total / 2 ** 20, 1)}
    except Exception as exc:  # noqa: BLE001
        env['cupy'] = f'不可用: {type(exc).__name__}: {exc}'
        env['gpu'] = None

    log(f"oracle      : opt_einsum {oracle_version}")
    log(f"pyqcd.einsum: "
        + ('就绪' if target_module is not None else
           f"未就绪（{_target.skip_reason()}），pyqcd 列整列 SKIP"))
    log(f"线程/精度   : {THREADS} 线程, {DTYPE}, optimize={optimize!r}")

    problems = data_status(sorted({t for name in profiles
                                  for t in PROFILES[name]['t_src']}))
    if problems:
        for problem in problems[:5]:
            log(f'SKIP 原因：{problem}')
        return [], env, True, problems

    rows = []
    for profile_name in profiles:
        cfg = dict(PROFILES[profile_name])
        note = cfg.get('note', '')
        allowed = cfg.get('impls')
        for backend in backends:
            if backend == 'gpu' and env.get('gpu') is None:
                log(f'  SKIP {profile_name}/{backend}: cupy 不可用')
                continue
            for t_src in cfg['t_src']:
                load_start = time.perf_counter()
                peram = load_peram(t_src, cfg['nvec'])
                load_s = time.perf_counter() - load_start
                log(f"\n══ {profile_name} · Nvec={cfg['nvec']} · t_src={t_src}"
                    f" · backend={backend} · 读盘 {load_s:.2f}s "
                    f"[{cfg['nvec']}×{cfg['nvec']} 切片]  ({note})")
                for eq_name in equations:
                    meta = EQUATION_BY_NAME[eq_name]
                    arrays = build_operands(eq_name, peram, cfg,
                                            seed=1000 + 17 * t_src)
                    shapes = [tuple(array.shape) for array in arrays]
                    naive_ops = naive_op_count(meta['eq'], shapes)
                    try:
                        opt_cost = optimized_cost(meta['eq'], shapes)
                    except Exception as exc:  # noqa: BLE001
                        opt_cost = f'规划失败: {type(exc).__name__}'
                    backend_arrays = _to_backend(arrays, backend)
                    log(f"  {eq_name} {meta['desc']}  naive≈{naive_ops:.2e} "
                        f"dp_cost={opt_cost}")
                    group = []
                    for impl_name, fn in implementations(
                            backend, optimize, target_module):
                        if allowed is not None and impl_name not in allowed:
                            continue
                        row = {'profile': profile_name, 'eq': eq_name,
                               'desc': meta['desc'], 'subscripts': meta['eq'],
                               'line': meta['line'], 'impl': impl_name,
                               'backend': backend, 'dtype': DTYPE,
                               'nvec': cfg['nvec'], 't_src': t_src,
                               'spatial': cfg['spatial'], 'n_t': cfg['n_t'],
                               'iterations': iterations,
                               'naive_est_ops': naive_ops,
                               'optimize': optimize,
                               'opt_cost': (str(opt_cost)
                                            if hasattr(opt_cost, 'as_tuple')
                                            else str(opt_cost)),
                               'load_s': round(load_s, 4),
                               'median_s': None, 'speedup_vs_opt': None,
                               'peak_bytes': None, 'status': 'ok',
                               'note': ''}
                        if 'naive' in impl_name and naive_ops > naive_budget:
                            row['status'] = 'skip'
                            row['note'] = (f'朴素代价 {naive_ops:.2e} 次标量乘加 '
                                           f'> 预算 {naive_budget:.1e}，不运行')
                            rows.append(row)
                            group.append(row)
                            log(f"    {impl_name:<22s} SKIP  {row['note']}")
                            continue
                        try:
                            stats = measure(lambda: fn(
                                meta['eq'], backend_arrays), backend,
                                iterations, warmup)
                            row.update({key: round(value, 6) if isinstance(
                                value, float) else value
                                for key, value in stats.items()})
                        except Exception as exc:  # noqa: BLE001
                            row['status'] = 'error'
                            row['note'] = f'{type(exc).__name__}: {exc}'
                        rows.append(row)
                        group.append(row)

                    opt_row = next((item for item in group
                                    if item['impl'] == 'opt_einsum_contract'
                                    and item['median_s']), None)
                    if opt_row:
                        for item in group:
                            if item['median_s']:
                                item['speedup_vs_opt'] = round(
                                    opt_row['median_s'] / item['median_s'], 4)
                    for item in group:
                        _print_row(log, item)
                del peram

    return rows, env, False, []


def _print_row(log, row):
    median = row['median_s']
    median_text = f"{median:.4f}s" if median else '   —   '
    speedup = row['speedup_vs_opt']
    speedup_text = f'{speedup:.2f}x' if speedup is not None else '  —  '
    if median and row['status'] == 'ok':
        peak = row['peak_bytes']
        peak_text = (f"{peak / 2 ** 20:.1f}MB" if peak else '  —  ')
        warn = ''
        if row['impl'] == 'pyqcd_einsum_contract' and speedup is not None \
                and speedup < SPEEDUP_TARGET:
            warn = f'  [< {SPEEDUP_TARGET:.1f} 未达标]'
        log(f"    {row['impl']:<22s} {median_text}  "
            f"speedup_vs_opt={speedup_text}  peak={peak_text}{warn}")
    else:
        log(f"    {row['impl']:<22s} {row['status'].upper():<6s}  "
            f"{row['note'][:96]}")


# ═══════════════════════════════════════════════════════════════════
# 产物
# ═══════════════════════════════════════════════════════════════════

def artifact_dir(outdir=None):
    if outdir:
        path = os.path.abspath(outdir)
    else:
        stamp = datetime.now().strftime('%Y%m%d%H%M%S')
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'artifacts', f'bench_{stamp}')
    os.makedirs(path, exist_ok=True)
    return path


def write_artifacts(rows, env, directory):
    payload = {'env': env,
               'speedup_target': SPEEDUP_TARGET,
               'rows': rows,
               'checks': _checks(rows)}
    with open(os.path.join(directory, 'bench.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(directory, 'env.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(env, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(directory, 'bench.md'), 'w',
              encoding='utf-8') as handle:
        handle.write(render_markdown(rows, env))
    return directory


def _checks(rows):
    """pyqcd 相对 opt_einsum 的达标判定（目标 3.0x）。"""
    checks = []
    for row in rows:
        if row['impl'] != 'pyqcd_einsum_contract' or not row['median_s']:
            continue
        speedup = row['speedup_vs_opt']
        checks.append({'profile': row['profile'], 'eq': row['eq'],
                       'backend': row['backend'], 'nvec': row['nvec'],
                       't_src': row['t_src'],
                       'speedup_vs_opt': speedup,
                       'target': SPEEDUP_TARGET,
                       'pass': bool(speedup is not None
                                    and speedup >= SPEEDUP_TARGET)})
    return checks


def render_markdown(rows, env):
    lines = ['# bench_distillation 结果', '']
    lines.append(f"- 时间：{env['time']}  主机：{env['hostname']}")
    lines.append(f"- oracle：opt_einsum {env['opt_einsum']}")
    lines.append(f"- pyqcd.einsum：{env['pyqcd_einsum']}")
    lines.append(f"- dtype={env['dtype']} threads={env['threads']} "
                 f"optimize={env['optimize']} iterations={env['iterations']}")
    lines.append('')
    lines.append('| profile | eq | backend | impl | nvec | median_s | '
                 'speedup_vs_opt | peak_bytes | status |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for row in rows:
        median = (f"{row['median_s']:.6f}" if row['median_s'] else '')
        speedup = (f"{row['speedup_vs_opt']:.3f}"
                   if row['speedup_vs_opt'] is not None else '')
        peak = row['peak_bytes'] if row['peak_bytes'] else ''
        lines.append(f"| {row['profile']} | {row['eq']} | {row['backend']} | "
                     f"{row['impl']} | {row['nvec']} | {median} | {speedup} | "
                     f"{peak} | {row['status']} |")
    checks = _checks(rows)
    if checks:
        lines += ['', '## pyqcd vs opt_einsum（目标 3.0x）', '']
        for item in checks:
            verdict = 'PASS' if item['pass'] else 'FAIL(<3.0)'
            lines.append(f"- {item['profile']}/{item['eq']}/"
                         f"{item['backend']}: {item['speedup_vs_opt']}x "
                         f"{verdict}")
    else:
        lines += ['', '## pyqcd vs opt_einsum', '',
                  f"未测量：{env['pyqcd_einsum']}"]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='蒸馏实战基准：numpy naive / opt_einsum / pyqcd.einsum')
    parser.add_argument('--profiles', default=','.join(DEFAULT_PROFILES),
                        help='逗号列表：T0n,T0,T1,T2')
    parser.add_argument('--backends', default=','.join(DEFAULT_BACKENDS),
                        help='逗号列表：cpu,gpu')
    parser.add_argument('--equations',
                        default=','.join(item['name'] for item in EQUATIONS))
    parser.add_argument('--iters', type=int, default=10,
                        help='计时次数（>=10，默认 10）')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--naive-budget', type=float, default=2e9,
                        help='朴素 numpy 的标量乘加次数预算，超出则 SKIP')
    parser.add_argument('--outdir', default=None,
                        help='产物目录（默认 artifacts/bench_<时间戳>/）')
    parser.add_argument('--smoke', action='store_true',
                        help='快速冒烟：T0n+T0、仅 CPU、3 次计时')
    args = parser.parse_args(argv)

    if args.smoke:
        profiles = ('T0n', 'T0')
        backends = ('cpu',)
        iterations, warmup = 3, 1
    else:
        profiles = tuple(item for item in args.profiles.split(',') if item)
        backends = tuple(item for item in args.backends.split(',') if item)
        iterations, warmup = max(1, args.iters), max(0, args.warmup)
    equations = tuple(item for item in args.equations.split(',') if item)

    print('════════ bench_distillation：蒸馏实战基准 ════════')
    rows, env, skipped, reasons = run(
        profiles=profiles, backends=backends, equations=equations,
        iterations=iterations, warmup=warmup,
        naive_budget=args.naive_budget)
    if skipped:
        print('\nSKIP bench_distillation：真实数据不可用')
        for reason in reasons[:5]:
            print(f'  - {reason}')
        return 0

    directory = artifact_dir(args.outdir)
    write_artifacts(rows, env, directory)

    print('\n──────────── bench 汇总（speedup_vs_opt）────────────')
    checks = _checks(rows)
    errors = [row for row in rows if row['status'] == 'error']
    if checks:
        for item in checks:
            verdict = ('未达标(<3.0)' if not item['pass'] else '达标')
            print(f"  {item['profile']:<4s} {item['eq']:<3s} "
                  f"{item['backend']:<4s} Nvec={item['nvec']:<4d} "
                  f"t_src={item['t_src']}  "
                  f"speedup_vs_opt={item['speedup_vs_opt']:.2f}x  {verdict}")
    else:
        print(f"  pyqcd 列未测量：{env['pyqcd_einsum']}")
    for row in errors:
        print(f"  ERROR {row['profile']}/{row['eq']}/{row['backend']}/"
              f"{row['impl']}: {row['note']}")
    print(f"\n产物: {directory}")
    print(f"       bench.json / bench.md / env.json（{len(rows)} 行）")
    if errors:
        print('FAIL bench_distillation（存在实现报错的对照行）')
        return 1
    print('PASS bench_distillation')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
