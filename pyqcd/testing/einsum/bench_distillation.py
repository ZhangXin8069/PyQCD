"""严格、可复现的 qcd-einsum vs opt_einsum 蒸馏实战基准。

真实数据（只读）来自 conf6250 的四个 perambulator spin 分片。目标
``L24x64/ms0.2310`` 数据不存在，因此当前报告明确标记为 provisional。

公平性约束：

* 两边使用同一组操作数、同一 complex128 dtype、同一显式收缩 path；
* CPU 只与 CPU 比，GPU 只与 GPU 比；
* 默认让两边分别在全新解释器中预热并计时，可用 ``--isolation abba``
  切回进程内 ABBA 交错计时；
* GPU 每次计时前后均同步，计时外校验逐元素 max relative error；
* qcd-einsum 低于 3x、数值失败或任何对照错误均返回非零。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime

DEFAULT_THREADS = 4
THREAD_ENV_VARS = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
)


def _configured_threads():
    raw = os.environ.get('PYQCD_BENCH_THREADS', str(DEFAULT_THREADS))
    try:
        threads = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f'PYQCD_BENCH_THREADS 必须是正整数，收到 {raw!r}') from exc
    if threads < 1:
        raise RuntimeError(
            f'PYQCD_BENCH_THREADS 必须是正整数，收到 {raw!r}')
    return threads


# 必须在 import numpy 前覆盖而不是 setdefault，避免调用者环境造成两边线程不同。
THREADS = _configured_threads()
for _thread_var in THREAD_ENV_VARS:
    os.environ[_thread_var] = str(THREADS)

NUMPY_WAS_IMPORTED = 'numpy' in sys.modules
THREAD_PIN_BEFORE_NUMPY_IMPORT = (
    not NUMPY_WAS_IMPORTED
    or os.environ.get('PYQCD_BENCH_THREAD_PINNED') == '1'
)

import numpy as np  # noqa: E402 - 线程变量必须在 numpy 导入前钉住

from . import _target  # noqa: E402

DTYPE = 'complex128'
RELATIVE_ERROR_TOLERANCE = 1e-12
SPEEDUP_TARGET = 3.0
MIN_ACCEPTANCE_ITERATIONS = 10
MIN_MEANINGFUL_SECONDS = 0.1
# 实测：CUDA 路径前 3-4 次调用会因显存池增长慢 3-4 倍（L24/E4: 4.16/2.95/3.57 -> 1.095 s）。
# 预热不足会把中位数拖到非稳态，故默认预热 4 轮。
MIN_WARMUP = 4

DATA_ROOT = (
    '/public/group/lqcd/perambulators/'
    'beta6.20_mu-0.2770_ms-0.2400_L24x72/light/6250'
)
DATA_CONF = 6250
NT_FULL = 72
NVEC_FULL = 100
SPIN_FILES = 4
FILE_BYTES = 46_080_000
VALUES_PER_FILE = FILE_BYTES // 8

TARGET_DATA = {
    'lattice': 'L24x64',
    'mass': 'ms0.2310',
    'available': False,
}
DATA_SCOPE = {
    'provisional': True,
    'requested_target': TARGET_DATA,
    'used_data': {
        'ensemble': 'beta6.20_mu-0.2770_ms-0.2400_L24x72',
        'lattice': 'L24x72',
        'mass': 'ms0.2400',
        'conf': DATA_CONF,
    },
    'note': ('目标 L24x64/ms0.2310 数据集不存在；'
             '本次使用 conf6250/L24x72 真实 perambulator，结论标记 provisional'),
}

DATA_MODES = ('real', 'synthetic')
SYNTHETIC_DATA_SCOPE = {
    'provisional': True,
    'requested_target': TARGET_DATA,
    'used_data': {
        'ensemble': 'synthetic',
        'lattice': TARGET_DATA['lattice'],
        'mass': TARGET_DATA['mass'],
        'conf': None,
    },
    'note': ('目标数据集与本机 perambulator 均不存在；使用固定种子合成 '
             'complex128 操作数，形状与真实蒸馏一致，正确性以 opt_einsum/numpy '
             '逐元素对照（rel<=1e-12）为准，性能结论标记 provisional'),
}

FROZEN_ORACLE_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'refer', 'git-rep', 'opt_einsum'))
FROZEN_ORACLE_PACKAGE = os.path.join(FROZEN_ORACLE_ROOT, 'opt_einsum')

EQUATIONS = (
    {'name': 'E1', 'line': 165, 'desc': '8 算子 2pt',
     'eq': 'gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M'},
    {'name': 'E2', 'line': 234, 'desc': '8 算子 3pt',
     'eq': 'Geifj,ckdl,agbh,ce,gi,Mbdf,Mhjl,Gka->GM'},
    {'name': 'E4', 'line': 211, 'desc': '4 算子 peram×gauge×γ',
     'eq': 'zyxcTabn,zyxgTdef,Gae,Tzyxgc->TGdbfn'},
    {'name': 'E5', 'line': 200, 'desc': 'γ 夹心',
     'eq': 'ab,...bce,cd->...dae'},
    {'name': 'S1', 'line': 119, 'desc': 'sush NN 2pt（8 算子，真实生产）',
     'eq': 'eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN'},
    {'name': 'S2', 'line': 402, 'desc': 'sush 3pt（9 算子，含 μν 插入，真实生产）',
     'eq': 'gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG'},
    {'name': 'S3', 'line': 301, 'desc': 'sush 2pt（6 算子，含 μ 插入，真实生产）',
     'eq': 'manb,cedf,bd,Mfn,ac,MFem->MF'},
    {'name': 'P1', 'line': 168, 'desc': 'sush 3pt 主项（15 算子，含 γ_curr 与投影，真实生产）',
     'eq': 'agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy'},
    {'name': 'P2', 'line': 176, 'desc': 'sush 3pt 主项（15 算子，置换型，真实生产）',
     'eq': 'agbh,eqfr,ospt,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy'},
)
EQUATION_BY_NAME = {item['name']: item for item in EQUATIONS}
SUSH_EQUATIONS = frozenset(('S1', 'S2', 'S3'))

PROFILES = {
    'T0n': {'nvec': 6, 'spatial': 1, 'n_t': 2, 't_src': (0,), 't_sink': 4,
            'note': '最小真实数据探针'},
    'T0': {'nvec': 20, 'spatial': 1, 'n_t': 2, 't_src': (0,), 't_sink': 4,
           'note': 'CI 规模 Nvec=20'},
    'T1': {'nvec': 100, 'spatial': 1, 'n_t': 2, 't_src': (0,), 't_sink': 8,
           'note': '单 t_src，全 Nvec=100'},
    'T2': {'nvec': 100, 'spatial': 1, 'n_t': 2, 't_src': (0, 1),
           't_sink': 8, 'note': '多 t_src 持续吞吐'},
    'L24': {'nvec': 50, 'spatial': 24, 'n_t': 2, 't_src': (0,), 't_sink': 8,
            'note': 'L=24 真实空间格点；prop 单算子 1.06 GB，峰值约 3.2 GB'},
    'L24n': {'nvec': 24, 'spatial': 24, 'n_t': 2, 't_src': (0,), 't_sink': 8,
             'note': 'L=24 显存保守档；prop 单算子 0.51 GB'},
}
DEFAULT_PROFILES = ('T0n', 'T0', 'T1', 'T2')
ACCEPTANCE_PROFILES = frozenset(('T1', 'T2'))
DIAGNOSTIC_PROFILES = frozenset(('T0n', 'T0'))
DEFAULT_BACKENDS = ('cpu', 'gpu')
BACKEND_NAMES = {'cpu': 'numpy', 'gpu': 'cupy'}


def data_path(t_src, spin):
    return os.path.join(DATA_ROOT, f'perams.{DATA_CONF}.{spin}.{t_src}')


def data_status(t_src_list, data_mode='real'):
    """返回缺失或尺寸不符的真实数据文件清单（synthetic 模式无文件依赖）。"""
    if data_mode == 'synthetic':
        return []
    problems = []
    for t_src in t_src_list:
        for spin in range(SPIN_FILES):
            path = data_path(t_src, spin)
            if not os.path.isfile(path):
                problems.append(f'缺失 {path}')
            elif os.path.getsize(path) != FILE_BYTES:
                problems.append(
                    f'{path} 大小 {os.path.getsize(path)} != {FILE_BYTES}')
    return problems


def synthetic_peram(t_src, nvec):
    """以固定种子构造 ``[Nt,4,4,Nvec,Nvec]`` complex128 perambulator。

    形状与 ``load_peram`` 一致；数值只用于正确性对照与计时，不代表任何物理结果。
    """
    rng = np.random.default_rng(20261107 + 7919 * int(t_src))
    shape = (NT_FULL, 4, 4, nvec, nvec)
    data = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    return np.ascontiguousarray(data, dtype=np.complex128)


def load_peram(t_src, nvec, data_mode='real'):
    """读取四个 spin 分片，返回 ``[Nt, 4, 4, Nvec, Nvec]`` complex128。"""
    if data_mode == 'synthetic':
        return synthetic_peram(t_src, nvec)
    slices = []
    for spin in range(SPIN_FILES):
        path = data_path(t_src, spin)
        buffer = np.fromfile(path, dtype='<f8')
        if buffer.size != VALUES_PER_FILE:
            raise ValueError(
                f'{path} 元素数 {buffer.size} != {VALUES_PER_FILE}')
        slices.append(buffer)
    cube = np.stack(slices).reshape(
        SPIN_FILES, NT_FULL, NVEC_FULL, SPIN_FILES, NVEC_FULL, 2)
    cube = cube[..., 0] + 1j * cube[..., 1]
    peram = cube.transpose(1, 3, 0, 4, 2)
    peram = peram[:, :, :, :nvec, :nvec]
    return np.ascontiguousarray(peram, dtype=np.complex128)


def _gamma(index):
    from pyqcd.lattice import gamma

    return np.asarray(gamma(index), dtype=np.complex128)


def _projector():
    return np.asarray((_gamma(0) + _gamma(4)) / 2.0, dtype=np.complex128)


def _rand(shape, rng):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)).astype(np.complex128)


def build_operands(name, peram, cfg, seed):
    """按参考脚本形状，用固定 seed 构造一例的全部操作数。"""
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
        current = peram[t_sink][None]
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
    if name == 'S1':
        sl = peram[t_sink]
        g7 = _gamma(7)
        return [sl, sl, sl, g7, g7,
                _rand((1, nvec, nvec, nvec), rng),
                _rand((1, nvec, nvec, nvec), rng),
                _projector()]
    if name == 'S2':
        return [peram[t_sink], peram[0], peram[1 % peram.shape[0]],
                _rand((nvec, nvec), rng),
                _rand((1, nvec, nvec), rng),
                _rand((1, nvec, nvec), rng),
                _gamma(5),
                _rand((1, 1, 4, 4), rng),
                _rand((1, 1, 4, 4), rng)]
    if name == 'S3':
        return [peram[t_sink], peram[0],
                _rand((nvec, nvec), rng),
                _rand((1, nvec, nvec), rng),
                _gamma(5),
                _rand((1, 1, 4, 4), rng)]
    if name in ('P1', 'P2'):
        mom, link = 1, 2
        projection = _projector()[:, :2]
        return [peram[t_sink], peram[t_sink], peram[-1], peram[0],
                peram[1 % peram.shape[0]],
                _gamma(7), _rand((4, 4, 4), rng), _gamma(5), _gamma(7),
                _rand((mom, nvec, nvec, nvec), rng),
                _rand((mom, link, nvec, nvec), rng),
                _rand((mom, nvec, nvec), rng),
                _rand((mom, nvec, nvec, nvec), rng),
                projection, projection]
    raise KeyError(f'未知方程 {name}')


def _purge_opt_einsum():
    for name in [n for n in list(sys.modules)
                 if n == 'opt_einsum' or n.startswith('opt_einsum.')]:
        del sys.modules[name]


def _package_dir(module):
    return os.path.abspath(os.path.dirname(module.__file__))


def resolve_oracle():
    """按显式规则加载冻结 oracle 或站点包。"""
    mode = os.environ.get('PYQCD_EINSUM_ORACLE', 'frozen').strip().lower()
    _purge_opt_einsum()
    if mode == 'frozen':
        if not os.path.isfile(os.path.join(
                FROZEN_ORACLE_PACKAGE, '__init__.py')):
            raise RuntimeError(
                '固定 opt_einsum 子模块不可用；如仅需非冻结诊断，'
                '显式设置 PYQCD_EINSUM_ORACLE=installed')
        head = subprocess.run(
            ['git', '-C', FROZEN_ORACLE_ROOT, 'rev-parse', 'HEAD'],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        index = subprocess.run(
            ['git', '-C', os.path.abspath(os.path.join(
                FROZEN_ORACLE_ROOT, '..', '..', '..')),
             'ls-files', '-s', '--', 'refer/git-rep/opt_einsum'],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        fields = index.split()
        if len(fields) < 2 or fields[0] != '160000' or fields[1] != head:
            raise RuntimeError(
                'frozen opt_einsum HEAD does not match the parent gitlink')
        dirty = subprocess.run(
            ['git', '-C', FROZEN_ORACLE_ROOT, 'status', '--porcelain'],
            check=True, capture_output=True, text=True,
        ).stdout
        if dirty:
            raise RuntimeError('frozen opt_einsum worktree is not clean')
        if FROZEN_ORACLE_ROOT not in sys.path:
            sys.path.insert(0, FROZEN_ORACLE_ROOT)
        module = importlib.import_module('opt_einsum')
        resolved = _package_dir(module)
        if resolved != os.path.abspath(FROZEN_ORACLE_PACKAGE):
            raise RuntimeError(
                f'frozen oracle 解析到 {resolved}，应为 '
                f'{os.path.abspath(FROZEN_ORACLE_PACKAGE)}')
        return module, {
            'kind': 'frozen',
            'label': 'frozen（冻结子模块）',
            'frozen': True,
            'provisional': False,
            'version': getattr(module, '__version__', 'unknown'),
            'path': resolved,
            'commit': head,
        }

    if mode == 'installed':
        while FROZEN_ORACLE_ROOT in sys.path:
            sys.path.remove(FROZEN_ORACLE_ROOT)
        module = importlib.import_module('opt_einsum')
        resolved = _package_dir(module)
        if resolved == os.path.abspath(FROZEN_ORACLE_PACKAGE):
            raise RuntimeError('installed 模式仍解析到冻结 oracle')
        return module, {
            'kind': 'installed',
            'label': 'installed（非冻结站点包）',
            'frozen': False,
            'provisional': True,
            'version': getattr(module, '__version__', 'unknown'),
            'path': resolved,
        }

    raise RuntimeError(
        'PYQCD_EINSUM_ORACLE 仅支持 frozen 或 installed，'
        f'收到 {mode!r}')


def _target_module():
    module, missing = _target.probe()
    if missing:
        raise RuntimeError('pyqcd.einsum 未就绪: ' + '; '.join(missing))
    version_module = importlib.import_module('pyqcd.einsum._version')
    return module, {
        'version': getattr(version_module, '__version__', 'unknown'),
        'native_version': getattr(
            version_module, '__native_version__', 'unknown'),
        'path': os.path.abspath(module.__file__),
    }


def _native_availability() -> dict[str, bool]:
    """Probe the concrete native entry expected by each benchmark backend."""
    from pyqcd.einsum import _native

    return {
        'kupl': _native._kupl_entry() is not None,
        'cuda': _native._cuda_entry() is not None,
    }


def _cpu_count():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count()


def _device_info(backend):
    if backend == 'cpu':
        return {
            'type': 'cpu',
            'name': platform.processor() or platform.machine(),
            'threads': THREADS,
        }
    try:
        import cupy

        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props['name']
        if isinstance(name, bytes):
            name = name.decode()
        free, total = cupy.cuda.Device(0).mem_info
        return {
            'type': 'gpu',
            'name': str(name),
            'cupy': cupy.__version__,
            'free_mb': round(free / 2 ** 20, 1),
            'total_mb': round(total / 2 ** 20, 1),
        }
    except Exception as exc:  # noqa: BLE001 - 环境记录中如实报告不可用
        return {
            'type': 'gpu',
            'available': False,
            'error': f'{type(exc).__name__}: {exc}',
        }


def _environment(profiles, backends, equations, iterations, warmup,
                 optimize, mode, isolation, data_mode='real'):
    numpy_config = np.show_config(mode='dicts')
    blas = numpy_config.get('Build Dependencies', {}).get('blas', {})
    return {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hostname': platform.node(),
        'platform': platform.platform(),
        'python': platform.python_version(),
        'python_executable': sys.executable,
        'command': list(sys.argv),
        'numpy': np.__version__,
        'numpy_blas': blas,
        'dtype': DTYPE,
        'threads': THREADS,
        'thread_environment': {
            name: os.environ.get(name) for name in THREAD_ENV_VARS},
        'thread_pin_before_numpy_import': THREAD_PIN_BEFORE_NUMPY_IMPORT,
        'cpu_count': _cpu_count(),
        'devices': {name: _device_info(name) for name in backends},
        'optimize': optimize,
        'iterations': iterations,
        'warmup': warmup,
        'measurement_order': (
            'ABBA' if isolation == 'abba' else 'PROCESS_ISOLATED'),
        'isolation': isolation,
        'profiles': list(profiles),
        'backends': list(backends),
        'equations': list(equations),
        'mode': mode,
        'data_mode': data_mode,
        'data_scope': (
            SYNTHETIC_DATA_SCOPE if data_mode == 'synthetic' else DATA_SCOPE),
        'relative_error_tolerance': RELATIVE_ERROR_TOLERANCE,
        'speedup_target': SPEEDUP_TARGET,
    }


def _normalize_path(path):
    return [tuple(int(index) for index in step) for step in path]


def _path_digest(path):
    encoded = json.dumps(path, separators=(',', ':')).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()


def _plan_pair(eq, shapes, optimize, oracle_module, target_module,
               planner='production'):
    oracle_path, oracle_info = oracle_module.contract_path(
        eq, *shapes, shapes=True, optimize=optimize)
    qcd_path, qcd_info = target_module.contract_path(
        eq, *shapes, shapes=True, optimize=optimize)
    path = _normalize_path(oracle_path)
    qcd_path = _normalize_path(qcd_path)
    # production：两边各自按 optimize 规划一次，只计时稳态执行（各自用己方 path）
    # shared：两边共用 oracle path，只隔离执行层
    measured = path
    return measured, {
        'planner': planner,
        'path': path,
        'oracle_planner_path': path,
        'qcd_planner_path': qcd_path,
        'planner_path_match': path == qcd_path,
        'path_digest': _path_digest(path),
        'qcd_path_digest': _path_digest(qcd_path),
        'oracle_opt_cost': str(oracle_info.opt_cost),
        'qcd_opt_cost': str(qcd_info.opt_cost),
    }


def _optimize_for(planner, optimize, path, side_path=None):
    """Return the ``optimize`` argument one side should receive.

    ``production`` 下双方各自使用己方 planner 产生的 path（规划只做一次，
    不计入稳态计时）；``shared`` 下双方共用同一个 oracle path。
    """
    if planner == 'production' and side_path is not None:
        return _normalize_path(side_path)
    return _normalize_path(path)


def _to_backend(arrays, backend):
    if backend == 'gpu':
        import cupy

        return [cupy.asarray(array) for array in arrays]
    return list(arrays)


def _array_dtypes(arrays):
    return [str(array.dtype) for array in arrays]


def _sync(backend):
    if backend == 'gpu':
        import cupy

        cupy.cuda.Stream.null.synchronize()


def _to_numpy(array, backend):
    if backend == 'gpu':
        import cupy

        return cupy.asnumpy(array)
    return np.asarray(array)


def _norm_relative_error(actual, expected):
    """范数相对误差：max|Δ| / max|ref|（契约判定用的稳健口径）。

    逐元素 |Δ|/|ref| 在抵消严重的元素上（|ref| 接近 0）会被 1/|ref| 放大，
    与 dtype 精度无关，不能作为契约判据；它只保留为诊断量。
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return float('inf')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return float('inf')
    difference = np.abs(actual - expected)
    denominator = max(float(np.max(np.abs(expected), initial=0.0)),
                      np.finfo(np.float64).tiny)
    return float(np.max(difference, initial=0.0) / denominator)


def _max_relative_error(actual, expected):
    """逐元素 max |Δ|/|ref|：仅作诊断，不作契约判据。"""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return float('inf')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return float('inf')
    difference = np.abs(actual - expected)
    scale = np.maximum(
        np.abs(expected), np.finfo(np.float64).tiny)
    return float(np.max(difference / scale, initial=0.0))


def _max_absolute_error(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return float('inf')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return float('inf')
    return float(np.max(np.abs(actual - expected), initial=0.0))


def _abba_schedule(iterations):
    """每个完整 ABBA 周期给两边各两次样本。"""
    cycles = (iterations + 1) // 2
    schedule = []
    for _ in range(cycles):
        schedule.extend(('opt_einsum', 'qcd_einsum',
                         'qcd_einsum', 'opt_einsum'))
    return schedule


def _measure_pair(functions, backend, iterations, warmup):
    """预热后按 ABBA 交错计时，并返回双方统计量。"""
    for _ in range(max(1, warmup)):
        for name in ('opt_einsum', 'qcd_einsum',
                     'qcd_einsum', 'opt_einsum'):
            output = functions[name]()
            _sync(backend)
            del output

    samples = {'opt_einsum': [], 'qcd_einsum': []}
    schedule = _abba_schedule(iterations)
    for name in schedule:
        _sync(backend)
        start = time.perf_counter()
        output = functions[name]()
        _sync(backend)
        samples[name].append(time.perf_counter() - start)
        del output

    stats = {}
    for name, values in samples.items():
        stats[name] = {
            'samples_s': [round(value, 9) for value in values],
            'iterations': len(values),
            'median_s': round(statistics.median(values), 9),
            'min_s': round(min(values), 9),
            'max_s': round(max(values), 9),
            'stdev_s': round(
                statistics.pstdev(values) if len(values) > 1 else 0.0, 9),
        }
    return stats, schedule


def _operand_digest(arrays):
    """对 dtype、形状和 C-order 数据做摘要，用于跨进程操作数校验。"""
    digest = hashlib.sha256()
    for index, array in enumerate(arrays):
        contiguous = np.ascontiguousarray(array)
        digest.update(
            f'{index}:{contiguous.dtype}:{contiguous.shape}:'.encode('ascii'))
        digest.update(contiguous.tobytes(order='C'))
    return digest.hexdigest()


def _measure_single(function, backend, iterations, warmup):
    """在单个隔离进程中预热并统计一个实现。"""
    for _ in range(max(1, warmup)):
        output = function()
        _sync(backend)
        del output

    samples = []
    for _ in range(iterations):
        _sync(backend)
        start = time.perf_counter()
        output = function()
        _sync(backend)
        samples.append(time.perf_counter() - start)
        del output
    return {
        'samples_s': [round(value, 9) for value in samples],
        'iterations': len(samples),
        'median_s': round(statistics.median(samples), 9),
        'min_s': round(min(samples), 9),
        'max_s': round(max(samples), 9),
        'stdev_s': round(
            statistics.pstdev(samples) if len(samples) > 1 else 0.0, 9),
    }


def _loaded_oracle_snapshot():
    module = sys.modules.get('opt_einsum')
    if module is None:
        return {'loaded': False}
    return {
        'loaded': True,
        'version': getattr(module, '__version__', 'unknown'),
        'path': _package_dir(module),
    }


def _read_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _worker_benchmark(task):
    if task.get('schema') != 1:
        raise RuntimeError('不支持的 worker task schema')
    implementation = task['implementation']
    if implementation not in ('opt_einsum', 'qcd_einsum'):
        raise RuntimeError(f'未知 worker implementation: {implementation!r}')
    if task['dtype'] != DTYPE:
        raise RuntimeError(
            f"worker dtype={task['dtype']!r}，应为 {DTYPE!r}")
    if task['backend'] not in BACKEND_NAMES:
        raise RuntimeError(f"未知 worker backend: {task['backend']!r}")

    cfg = task['profile_config']
    peram = load_peram(task['t_src'], cfg['nvec'], task.get('data_mode', 'real'))
    arrays = build_operands(
        task['equation'], peram, cfg, seed=task['operand_seed'])
    shapes = [tuple(array.shape) for array in arrays]
    dtypes = _array_dtypes(arrays)
    if shapes != [tuple(shape) for shape in task['operand_shapes']]:
        raise RuntimeError(f'worker 操作数形状不一致: {shapes}')
    if dtypes != task['operand_dtypes']:
        raise RuntimeError(f'worker 操作数 dtype 不一致: {dtypes}')
    actual_operand_digest = _operand_digest(arrays)
    if actual_operand_digest != task['operand_digest']:
        raise RuntimeError(
            'worker 操作数摘要不一致: '
            f'{actual_operand_digest} != {task["operand_digest"]}')

    planner = task.get('planner', 'shared')
    path = _normalize_path(task['path'])
    if planner == 'shared' and _path_digest(path) != task['path_digest']:
        raise RuntimeError('worker 显式 path 摘要不一致')
    qcd_path = _normalize_path(task.get('qcd_path') or task['path'])
    optimize_argument = (
        _optimize_for(planner, task.get('optimize'), path, qcd_path)
        if implementation == 'qcd_einsum'
        else _optimize_for(planner, task.get('optimize'), path))

    if implementation == 'opt_einsum':
        module, loaded_oracle = resolve_oracle()
        loaded_oracle = dict(loaded_oracle)
        loaded_oracle['loaded'] = True
        contract = module.contract
        implementation_info = loaded_oracle
    else:
        module, implementation_info = _target_module()
        loaded_oracle = _loaded_oracle_snapshot()
        contract = module.contract

    backend = task['backend']
    backend_arrays = _to_backend(arrays, backend)
    actual_dtypes = _array_dtypes(backend_arrays)
    if actual_dtypes != task['operand_dtypes']:
        raise RuntimeError(f'worker backend 操作数 dtype 不一致: {actual_dtypes}')

    def function():
        return contract(
            task['subscripts'], *backend_arrays,
            optimize=optimize_argument, backend=BACKEND_NAMES[backend])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _sync(backend)
        output = function()
        _sync(backend)
    implementation_warnings = [
        f'{item.category.__name__}: {item.message}' for item in caught]
    output_host = _to_numpy(output, backend)
    output_dtype = str(output.dtype)
    output_shape = list(output.shape)
    del output

    np.save(task['output_path'], output_host, allow_pickle=False)
    del output_host, arrays, peram
    gc.collect()
    stats = _measure_single(function, backend, task['iterations'], task['warmup'])
    del backend_arrays
    return {
        'schema': 1,
        'implementation': implementation,
        'worker_pid': os.getpid(),
        'backend': backend,
        'dtype': task['dtype'],
        'threads': THREADS,
        'thread_environment': {
            name: os.environ.get(name) for name in THREAD_ENV_VARS},
        'thread_pin_before_numpy_import': THREAD_PIN_BEFORE_NUMPY_IMPORT,
        'numpy_version': np.__version__,
        'operand_seed': task['operand_seed'],
        'operand_digest': actual_operand_digest,
        'operand_shapes': [list(shape) for shape in shapes],
        'operand_dtypes': dtypes,
        'path': [list(step) for step in path],
        'path_digest': task['path_digest'],
        'planner': planner,
        'optimize_argument': (optimize_argument
                              if isinstance(optimize_argument, str)
                              else 'shared-path'),
        'output_dtype': output_dtype,
        'output_shape': output_shape,
        'implementation_warnings': implementation_warnings,
        'loaded_oracle': loaded_oracle,
        'implementation_info': implementation_info,
        'iterations': task['iterations'],
        'warmup': task['warmup'],
        'stats': stats,
    }


def _run_worker_task(task_path):
    task = _read_json(task_path)
    result_path = task.get('result_path')
    try:
        result = _worker_benchmark(task)
    except Exception as exc:  # noqa: BLE001 - 通过受控 JSON 回传硬失败
        result = {
            'schema': 1,
            'implementation': task.get('implementation'),
            'worker_pid': os.getpid(),
            'error': f'{type(exc).__name__}: {exc}',
        }
        if result_path:
            _write_json(result_path, result)
        return 1
    _write_json(result_path, result)
    return 0


def _run_isolated_worker(implementation, task):
    with tempfile.TemporaryDirectory(
            prefix=f'pyqcd-bench-{implementation}-') as directory:
        output_path = os.path.join(directory, 'output.npy')
        result_path = os.path.join(directory, 'result.json')
        task_path = os.path.join(directory, 'task.json')
        worker_task = dict(task)
        worker_task.update({
            'schema': 1,
            'implementation': implementation,
            'output_path': output_path,
            'result_path': result_path,
        })
        _write_json(task_path, worker_task)

        environment = os.environ.copy()
        environment.pop('PYQCD_BENCH_BOOTSTRAP_ACTIVE', None)
        environment['PYQCD_BENCH_THREAD_PINNED'] = '1'
        environment['PYQCD_BENCH_THREADS'] = str(THREADS)
        environment['PYQCD_BENCH_ISOLATION_WORKER'] = '1'
        for name in THREAD_ENV_VARS:
            environment[name] = str(THREADS)
        worker_script = os.path.join(os.path.dirname(__file__), 'bench_native.py')
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        completed = subprocess.run(
            [sys.executable, '-B', worker_script, '--worker-task', task_path],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        result = _read_json(result_path) if os.path.isfile(result_path) else None
        if completed.returncode != 0 or result is None or result.get('error'):
            detail = result.get('error') if result else ''
            stderr = completed.stderr.strip()
            if stderr:
                detail = f'{detail}; stderr={stderr[-2000:]}' if detail else stderr[-2000:]
            raise RuntimeError(
                f'{implementation} worker 失败（returncode='
                f'{completed.returncode}）: {detail or "无错误信息"}')
        if not isinstance(result.get('worker_pid'), int):
            raise RuntimeError(f'{implementation} worker 未返回有效 worker_pid')
        if not os.path.isfile(output_path):
            raise RuntimeError(f'{implementation} worker 未生成输出数组')
        output = np.load(output_path, allow_pickle=False)
        if str(output.dtype) != result.get('output_dtype'):
            raise RuntimeError(f'{implementation} worker 输出 dtype 不一致')
        if list(output.shape) != result.get('output_shape'):
            raise RuntimeError(f'{implementation} worker 输出 shape 不一致')
        return result, output


def _validate_isolated_result(result, task, oracle_info=None):
    implementation = result['implementation']
    if result.get('threads') != THREADS:
        raise RuntimeError(f'{implementation} worker threads 不一致')
    expected_thread_env = {name: str(THREADS) for name in THREAD_ENV_VARS}
    if result.get('thread_environment') != expected_thread_env:
        raise RuntimeError(f'{implementation} worker 线程环境不一致')
    if not result.get('thread_pin_before_numpy_import', False):
        raise RuntimeError(f'{implementation} worker 未在 NumPy 导入前钉住线程')
    if result.get('backend') != task['backend'] or result.get('dtype') != DTYPE:
        raise RuntimeError(f'{implementation} worker backend/dtype 不一致')
    if result.get('operand_digest') != task['operand_digest']:
        raise RuntimeError(f'{implementation} worker 操作数摘要不一致')
    if result.get('operand_seed') != task['operand_seed']:
        raise RuntimeError(f'{implementation} worker 操作数 seed 不一致')
    if _normalize_path(result.get('path', [])) != _normalize_path(task['path']):
        raise RuntimeError(f'{implementation} worker 未使用父进程显式 path')
    if result.get('path_digest') != task['path_digest']:
        raise RuntimeError(f'{implementation} worker path 摘要不一致')
    if result.get('iterations') != task['iterations']:
        raise RuntimeError(f'{implementation} worker iterations 不一致')
    if result.get('warmup') != task['warmup']:
        raise RuntimeError(f'{implementation} worker warmup 不一致')
    if result.get('stats', {}).get('iterations') != task['iterations']:
        raise RuntimeError(f'{implementation} worker 样本数不一致')
    if oracle_info is not None:
        loaded = result.get('loaded_oracle', {})
        for key in ('kind', 'version', 'path'):
            if loaded.get(key) != oracle_info.get(key):
                raise RuntimeError(
                    f'{implementation} worker loaded oracle {key} 不一致')


def _measure_abba_case(arrays, backend, eq, path, oracle_module,
                       target_module, iterations, warmup, oracle_info,
                       planner='production', optimize='dp', qcd_path=None):
    backend_arrays = _to_backend(arrays, backend)
    actual_dtypes = _array_dtypes(backend_arrays)
    if any(dtype != DTYPE for dtype in actual_dtypes):
        raise RuntimeError(
            f'{backend} 操作数 dtype 非 {DTYPE}: {actual_dtypes}')

    backend_name = BACKEND_NAMES[backend]
    oracle_argument = _optimize_for(planner, optimize, path)
    qcd_argument = _optimize_for(planner, optimize, path, qcd_path)
    functions = {
        'opt_einsum': (
            lambda: oracle_module.contract(
                eq, *backend_arrays, optimize=oracle_argument,
                backend=backend_name)),
        'qcd_einsum': (
            lambda: target_module.contract(
                eq, *backend_arrays, optimize=qcd_argument,
                backend=backend_name)),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _sync(backend)
        oracle_output = functions['opt_einsum']()
        _sync(backend)
        qcd_output = functions['qcd_einsum']()
        _sync(backend)
    implementation_warnings = [
        f'{item.category.__name__}: {item.message}' for item in caught]
    oracle_host = _to_numpy(oracle_output, backend)
    qcd_host = _to_numpy(qcd_output, backend)
    oracle_dtype = str(oracle_output.dtype)
    qcd_dtype = str(qcd_output.dtype)
    dtype_match = oracle_dtype == DTYPE and qcd_dtype == DTYPE
    relative_error = _norm_relative_error(qcd_host, oracle_host)
    elementwise_error = _max_relative_error(qcd_host, oracle_host)
    absolute_error = _max_absolute_error(qcd_host, oracle_host)
    relative_error_pass = relative_error <= RELATIVE_ERROR_TOLERANCE
    del oracle_output, qcd_output, oracle_host, qcd_host

    gc.collect()
    stats, schedule = _measure_pair(functions, backend, iterations, warmup)
    del backend_arrays
    return {
        'stats': stats,
        'measurement_sequence': schedule,
        'measurement_order': 'ABBA',
        'worker_pid': None,
        'loaded_oracle': dict(oracle_info),
        'worker_loaded_oracles': {
            'opt_einsum': dict(oracle_info),
            'qcd_einsum': {'loaded': 'in-process'},
        },
        'max_relative_error': relative_error,
        'max_elementwise_relative_error': elementwise_error,
        'max_absolute_error': absolute_error,
        'oracle_output_dtype': oracle_dtype,
        'qcd_output_dtype': qcd_dtype,
        'output_dtype_match': dtype_match,
        'correctness_pass': dtype_match and relative_error_pass,
        'implementation_warnings': implementation_warnings,
    }


def _measure_process_case(task, oracle_info):
    opt_result, oracle_host = _run_isolated_worker('opt_einsum', task)
    qcd_result, qcd_host = _run_isolated_worker('qcd_einsum', task)
    _validate_isolated_result(opt_result, task, oracle_info=oracle_info)
    _validate_isolated_result(qcd_result, task)

    oracle_dtype = opt_result['output_dtype']
    qcd_dtype = qcd_result['output_dtype']
    dtype_match = oracle_dtype == DTYPE and qcd_dtype == DTYPE
    relative_error = _norm_relative_error(qcd_host, oracle_host)
    elementwise_error = _max_relative_error(qcd_host, oracle_host)
    absolute_error = _max_absolute_error(qcd_host, oracle_host)
    relative_error_pass = relative_error <= RELATIVE_ERROR_TOLERANCE
    del oracle_host, qcd_host
    gc.collect()
    return {
        'stats': {
            'opt_einsum': opt_result['stats'],
            'qcd_einsum': qcd_result['stats'],
        },
        'measurement_sequence': ['opt_einsum', 'qcd_einsum'],
        'measurement_order': 'PROCESS_ISOLATED',
        'worker_pid': {
            'opt_einsum': opt_result['worker_pid'],
            'qcd_einsum': qcd_result['worker_pid'],
        },
        'loaded_oracle': opt_result['loaded_oracle'],
        'worker_loaded_oracles': {
            'opt_einsum': opt_result['loaded_oracle'],
            'qcd_einsum': qcd_result['loaded_oracle'],
        },
        'max_relative_error': relative_error,
        'max_elementwise_relative_error': elementwise_error,
        'max_absolute_error': absolute_error,
        'oracle_output_dtype': oracle_dtype,
        'qcd_output_dtype': qcd_dtype,
        'output_dtype_match': dtype_match,
        'correctness_pass': dtype_match and relative_error_pass,
        'implementation_warnings': (
            [f'opt_einsum: {item}' for item in
             opt_result['implementation_warnings']]
            + [f'qcd_einsum: {item}' for item in
               qcd_result['implementation_warnings']]),
    }


def _new_row(profile_name, cfg, backend, env, eq_name, meta, t_src,
             operand_shapes, operand_dtypes, path_info):
    row = {
        'profile': profile_name,
        'eq': eq_name,
        'desc': meta['desc'],
        'subscripts': meta['eq'],
        'line': meta['line'],
        'backend': backend,
        'device': env['devices'][backend],
        'dtype': DTYPE,
        'nvec': cfg['nvec'],
        't_src': t_src,
        'spatial': cfg['spatial'],
        'n_t': cfg['n_t'],
        'operand_seed': 1000 + 17 * t_src,
        'operand_shapes': operand_shapes,
        'operand_dtypes': operand_dtypes,
        'optimize': env['optimize'],
        'planner': env.get('planner', 'shared'),
        'measurement_order': env['measurement_order'],
        'isolation': env['isolation'],
        'worker_pid': None,
        'loaded_oracle': None,
        'operand_digest': None,
        'status': 'ok',
        'note': '',
    }
    row.update(path_info)
    return row


def run(profiles=DEFAULT_PROFILES, backends=DEFAULT_BACKENDS,
        equations=tuple(item['name'] for item in EQUATIONS),
        iterations=MIN_ACCEPTANCE_ITERATIONS, warmup=MIN_WARMUP,
        optimize='dp', mode='full', isolation='process',
        data_mode='real', planner='production', log=print):
    """执行基准并返回 ``(rows, env, problems)``。"""
    profiles = tuple(profiles)
    backends = tuple(backends)
    equations = tuple(equations)
    env = _environment(
        profiles, backends, equations, iterations, warmup, optimize, mode,
        isolation, data_mode)
    env['planner'] = planner
    problems = []

    unknown_profiles = sorted(set(profiles) - set(PROFILES))
    unknown_equations = sorted(set(equations) - set(EQUATION_BY_NAME))
    unknown_backends = sorted(set(backends) - set(BACKEND_NAMES))
    if unknown_profiles:
        problems.append(f'未知 profiles: {unknown_profiles}')
    if unknown_equations:
        problems.append(f'未知 equations: {unknown_equations}')
    if unknown_backends:
        problems.append(f'未知 backends: {unknown_backends}')
    if problems:
        return [], env, problems

    try:
        oracle_module, oracle_info = resolve_oracle()
    except Exception as exc:  # noqa: BLE001 - 转为硬失败报告
        env['opt_einsum'] = {
            'error': f'{type(exc).__name__}: {exc}'}
        return [], env, [env['opt_einsum']['error']]
    env['opt_einsum'] = oracle_info
    env['loaded_oracle'] = dict(oracle_info)
    log(f"oracle      : {oracle_info['label']} "
        f"{oracle_info['version']} ({oracle_info['path']})")

    try:
        target_module, target_info = _target_module()
    except Exception as exc:  # noqa: BLE001 - pyqcd 不可用也应硬失败
        env['pyqcd_einsum'] = {
            'error': f'{type(exc).__name__}: {exc}'}
        return [], env, [env['pyqcd_einsum']['error']]
    env['pyqcd_einsum'] = target_info
    env['native_available'] = _native_availability()
    log(f"pyqcd.einsum: {target_info['version']} ({target_info['path']})")
    log(f"线程/精度   : {THREADS} 线程, {DTYPE}, optimize={optimize!r}, "
        f"isolation={isolation}, iterations={iterations}")
    log('数据口径    : provisional - ' + env['data_scope']['note'])

    t_srcs = sorted({
        t_src for profile_name in profiles
        for t_src in PROFILES[profile_name]['t_src']
    })
    data_problems = data_status(t_srcs, data_mode)
    if data_problems:
        problems.extend(data_problems)
        for problem in data_problems[:5]:
            log(f'DATA ERROR: {problem}')
        return [], env, problems

    for backend in backends:
        if env['devices'][backend].get('available') is False:
            problem = (f'{backend} 不可用: '
                       f"{env['devices'][backend]['error']}")
            problems.append(problem)
            log(f'BACKEND ERROR: {problem}')

    available_backends = tuple(
        backend for backend in backends
        if env['devices'][backend].get('available') is not False)
    for backend in available_backends:
        native_name = 'cuda' if backend == 'gpu' else 'kupl'
        if not env['native_available'].get(native_name, False):
            problems.append(f'{backend} requested but native {native_name} is unavailable')
    rows = []
    for profile_name in profiles:
        cfg = dict(PROFILES[profile_name])
        for t_src in cfg['t_src']:
            load_start = time.perf_counter()
            peram = load_peram(t_src, cfg['nvec'], data_mode)
            load_s = time.perf_counter() - load_start
            log(f"\n[{profile_name}] Nvec={cfg['nvec']} t_src={t_src} "
                f"读盘={load_s:.2f}s - {cfg['note']}")
            for eq_name in equations:
                meta = EQUATION_BY_NAME[eq_name]
                arrays = build_operands(
                    eq_name, peram, cfg, seed=1000 + 17 * t_src)
                shapes = [tuple(array.shape) for array in arrays]
                dtypes = _array_dtypes(arrays)
                operand_digest = _operand_digest(arrays)
                if any(dtype != DTYPE for dtype in dtypes):
                    problems.append(
                        f'{profile_name}/{eq_name} 操作数 dtype 非 {DTYPE}: '
                        f'{dtypes}')
                    continue

                for backend in available_backends:
                    row = None
                    try:
                        path, path_info = _plan_pair(
                            meta['eq'], shapes, optimize,
                            oracle_module, target_module, planner)
                        row = _new_row(
                            profile_name, cfg, backend, env, eq_name, meta,
                            t_src, shapes, dtypes, path_info)
                        if isolation == 'process':
                            worker_task = {
                                'profile': profile_name,
                                'equation': eq_name,
                                'subscripts': meta['eq'],
                                't_src': t_src,
                                'profile_config': {
                                    'nvec': cfg['nvec'],
                                    'spatial': cfg['spatial'],
                                    'n_t': cfg['n_t'],
                                    't_sink': cfg['t_sink'],
                                },
                                'operand_seed': 1000 + 17 * t_src,
                                'data_mode': data_mode,
                                'planner': planner,
                                'optimize': optimize,
                                'operand_shapes': [
                                    list(shape) for shape in shapes],
                                'operand_dtypes': dtypes,
                                'operand_digest': operand_digest,
                                'path': ([] if path is None
                                         else [list(step) for step in path]),
                                'path_digest': path_info['path_digest'],
                                'qcd_path': [
                                    list(step) for step in
                                    path_info['qcd_planner_path']],
                                'backend': backend,
                                'dtype': DTYPE,
                                'iterations': iterations,
                                'warmup': warmup,
                            }
                            measurement = _measure_process_case(
                                worker_task, oracle_info)
                        else:
                            measurement = _measure_abba_case(
                                arrays, backend, meta['eq'], path,
                                oracle_module, target_module, iterations,
                                warmup, oracle_info, planner, optimize,
                                path_info['qcd_planner_path'])
                        opt_median = measurement['stats'][
                            'opt_einsum']['median_s']
                        qcd_median = measurement['stats'][
                            'qcd_einsum']['median_s']
                        speedup = opt_median / qcd_median
                        row.update({
                            'opt_einsum': measurement['stats']['opt_einsum'],
                            'qcd_einsum': measurement['stats']['qcd_einsum'],
                            'measurement_sequence':
                                measurement['measurement_sequence'],
                            'measurement_order':
                                measurement['measurement_order'],
                            'worker_pid': measurement['worker_pid'],
                            'loaded_oracle': measurement['loaded_oracle'],
                            'worker_loaded_oracles':
                                measurement['worker_loaded_oracles'],
                            'operand_digest': operand_digest,
                            'max_relative_error':
                                measurement['max_relative_error'],
                            'max_elementwise_relative_error':
                                measurement.get(
                                    'max_elementwise_relative_error'),
                            'max_absolute_error':
                                measurement['max_absolute_error'],
                            'oracle_output_dtype':
                                measurement['oracle_output_dtype'],
                            'qcd_output_dtype':
                                measurement['qcd_output_dtype'],
                            'output_dtype_match':
                                measurement['output_dtype_match'],
                            'relative_error_tolerance':
                                RELATIVE_ERROR_TOLERANCE,
                            'correctness_pass':
                                measurement['correctness_pass'],
                            'implementation_warnings':
                                measurement['implementation_warnings'],
                            'speedup_vs_opt': round(speedup, 6),
                            'speedup_target': SPEEDUP_TARGET,
                            'too_short': bool(
                                min(opt_median, qcd_median)
                                < MIN_MEANINGFUL_SECONDS),
                            'speedup_pass': speedup >= SPEEDUP_TARGET,
                        })
                        if not measurement['output_dtype_match']:
                            row['status'] = 'error'
                            row['note'] = (
                                '输出 dtype 不一致: '
                                f"oracle={measurement['oracle_output_dtype']}, "
                                f"qcd={measurement['qcd_output_dtype']}, "
                                f'expected={DTYPE}')
                        elif not measurement['correctness_pass']:
                            row['status'] = 'error'
                            row['note'] = (
                                'max relative error 超过容差: '
                                f"{measurement['max_relative_error']:.3e}")
                        elif speedup < SPEEDUP_TARGET:
                            row['note'] = (
                                f'低于 {SPEEDUP_TARGET:.1f}x 硬门禁')
                        log(f"  {eq_name}/{backend}: "
                            f"opt={opt_median:.6f}s "
                            f"qcd={qcd_median:.6f}s "
                            f"speedup={speedup:.3f}x "
                            f"max_rel={measurement['max_relative_error']:.3e} "
                            f"path={path_info['path_digest'][:12]}")
                    except Exception as exc:  # noqa: BLE001 - 单例失败写入报告
                        if row is None:
                            row = _new_row(
                                profile_name, cfg, backend, env, eq_name, meta,
                                t_src, shapes, dtypes, {
                                    'path': None,
                                    'path_digest': None,
                                    'planner_path_match': False,
                                })
                            row['operand_digest'] = operand_digest
                        row['status'] = 'error'
                        row['note'] = f'{type(exc).__name__}: {exc}'
                        row['correctness_pass'] = False
                        row['speedup_pass'] = False
                        log(f"  {eq_name}/{backend}: ERROR {row['note']}")
                    rows.append(row)
            del peram
            gc.collect()
    if isolation == 'process':
        env['worker_pids'] = sorted({
            pid for row in rows
            for pid in (row.get('worker_pid') or {}).values()
            if isinstance(pid, int)
        })
        env['worker_loaded_oracles'] = [
            row.get('worker_loaded_oracles') for row in rows
            if row.get('worker_loaded_oracles') is not None
        ]
    return rows, env, problems


def _acceptance(rows, env, problems, mode, requested_backends):
    reasons = list(problems)
    provisional_reasons = []
    oracle = env.get('opt_einsum', {})
    if oracle.get('frozen') is not True:
        provisional_reasons.append('oracle is not the frozen opt_einsum submodule')
    if env.get('data_scope', DATA_SCOPE).get('provisional', True):
        provisional_reasons.append('benchmark uses a proxy dataset rather than the requested target')
    if mode == 'smoke':
        reasons.append('smoke 只覆盖 T0n/CPU，明确不构成验收')
    else:
        missing_profiles = sorted(
            ACCEPTANCE_PROFILES - set(env.get('profiles', ()))
        )
        missing_equations = sorted(
            set(EQUATION_BY_NAME) - set(env.get('equations', ()))
        )
        missing_backends = sorted(
            set(BACKEND_NAMES) - set(env.get('backends', ()))
        )
        if missing_profiles:
            reasons.append(f'验收缺少 profiles: {missing_profiles}')
        if missing_equations:
            reasons.append(f'验收缺少 equations: {missing_equations}')
        if missing_backends:
            reasons.append(f'验收缺少 backends: {missing_backends}')
    if not env.get('thread_pin_before_numpy_import', False):
        reasons.append('numpy 在强制线程变量前已导入，计时口径无效')
    if env.get('iterations', 0) < MIN_ACCEPTANCE_ITERATIONS:
        reasons.append(
            f'中位数样本请求数 {env.get("iterations")} < '
            f'{MIN_ACCEPTANCE_ITERATIONS}')
    if env.get('isolation') not in ('abba', 'process'):
        reasons.append(f'未知 isolation={env.get("isolation")!r}')
    if not rows:
        reasons.append('没有可评估的基准行')
    measured_backends = {row['backend'] for row in rows
                         if row.get('status') == 'ok'}
    missing_backends = [
        backend for backend in requested_backends
        if backend not in measured_backends]
    if missing_backends:
        reasons.append(f'请求的 backend 未测成功: {missing_backends}')
    short_rows = []
    for row in rows:
        label = f"{row['profile']}/{row['eq']}/{row['backend']}"
        if row.get('status') != 'ok':
            reasons.append(f'{label} status={row.get("status")}')
            continue
        if not row.get('correctness_pass', False):
            reasons.append(f'{label} 数值校验失败')
        if (env.get('planner', 'shared') == 'shared'
                and not row.get('planner_path_match', False)):
            reasons.append(f'{label} 双方规划 path 不一致')
        native_name = 'cuda' if row['backend'] == 'gpu' else 'kupl'
        if not env.get('native_available', {}).get(native_name, False):
            reasons.append(f'{label} native {native_name} unavailable')
        fallback_warnings = [
            warning for warning in row.get('implementation_warnings', [])
            if 'using Python execution' in warning
            or 'native' in warning and 'unavailable' in warning
        ]
        if fallback_warnings:
            reasons.append(f'{label} used Python fallback: {fallback_warnings}')
        if env.get('isolation') == 'process':
            worker_pid = row.get('worker_pid')
            if not (isinstance(worker_pid, dict)
                    and isinstance(worker_pid.get('opt_einsum'), int)
                    and isinstance(worker_pid.get('qcd_einsum'), int)):
                reasons.append(f'{label} 缺少有效 worker_pid')
            if not row.get('loaded_oracle'):
                reasons.append(f'{label} 缺少 loaded oracle 信息')
        if row.get('too_short'):
            short_rows.append(label)
            continue
        if row['profile'] in ACCEPTANCE_PROFILES and not row.get('speedup_pass', False):
            reasons.append(
                f'{label} speedup={row.get("speedup_vs_opt")} < '
                f'{SPEEDUP_TARGET}')
    if short_rows:
        reasons.append(
            f'{len(short_rows)} 组双方中位耗时 < {MIN_MEANINGFUL_SECONDS}s，'
            f'记为 too_short 且不作为实战对照: {short_rows}')
    acceptance_rows = [row for row in rows if row['profile'] in ACCEPTANCE_PROFILES
                       and not row.get('too_short')]
    if mode != 'smoke' and not acceptance_rows:
        reasons.append('没有运行 T1/T2 生产规模 profile，不能作为性能验收')
    if mode == 'smoke':
        state = 'NON_ACCEPTANCE'
        passed = False
        return_code = 2
    elif mode != 'smoke' and not acceptance_rows:
        state = 'NON_ACCEPTANCE'
        passed = False
        return_code = 2
    elif reasons:
        state = 'FAIL'
        passed = False
        return_code = 1
    elif provisional_reasons:
        state = 'NON_ACCEPTANCE'
        passed = False
        return_code = 2
    else:
        state = 'PASS'
        passed = True
        return_code = 0
    return {
        'state': state,
        'passed': passed,
        'acceptance_evaluated': mode != 'smoke',
        'provisional': env.get('data_scope', DATA_SCOPE)['provisional'],
        'reasons': reasons + provisional_reasons,
        'return_code': return_code,
    }


def _checks(rows):
    checks = []
    for row in rows:
        checks.append({
            'profile': row['profile'],
            'acceptance_profile': row['profile'] in ACCEPTANCE_PROFILES,
            'eq': row['eq'],
            'backend': row['backend'],
            'isolation': row.get('isolation'),
            'status': row['status'],
            'path_digest': row.get('path_digest'),
            'planner_path_match': row.get('planner_path_match'),
            'operand_digest': row.get('operand_digest'),
            'worker_pid': row.get('worker_pid'),
            'loaded_oracle': row.get('loaded_oracle'),
            'output_dtype_match': row.get('output_dtype_match'),
            'max_relative_error': row.get('max_relative_error'),
            'max_elementwise_relative_error': row.get(
                'max_elementwise_relative_error'),
            'relative_error_tolerance': RELATIVE_ERROR_TOLERANCE,
            'correctness_pass': bool(row.get('correctness_pass', False)),
            'speedup_vs_opt': row.get('speedup_vs_opt'),
        'speedup_target': SPEEDUP_TARGET,
        'speedup_pass': bool(row.get('speedup_pass', False)),
        'pass': bool(row.get('correctness_pass', False)
                         and row.get('speedup_pass', False)
                         and row.get('planner_path_match', False)),
            'note': row.get('note', ''),
        })
    return checks


def artifact_dir(outdir=None):
    if outdir:
        path = os.path.abspath(outdir)
    else:
        stamp = datetime.now().strftime('%Y%m%d%H%M%S')
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'artifacts', f'bench_{stamp}')
    os.makedirs(path, exist_ok=True)
    return path


def write_artifacts(rows, env, acceptance, directory):
    payload = {
        'benchmark': 'qcd-einsum vs opt_einsum',
        'acceptance': acceptance,
        'checks': _checks(rows),
        'environment': env,
        'rows': rows,
    }
    with open(os.path.join(directory, 'bench.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(directory, 'env.json'), 'w',
              encoding='utf-8') as handle:
        json.dump(env, handle, indent=2, ensure_ascii=False)
    with open(os.path.join(directory, 'bench.md'), 'w',
              encoding='utf-8') as handle:
        handle.write(render_markdown(rows, env, acceptance))
    return directory


def render_markdown(rows, env, acceptance):
    oracle = env.get('opt_einsum', {})
    pyqcd = env.get('pyqcd_einsum', {})
    lines = [
        '# qcd-einsum vs opt_einsum 3x 硬门禁基准',
        '',
        '## 结论',
        '',
        f"- 状态：**{acceptance['state']}**",
        f"- acceptance evaluated：{acceptance['acceptance_evaluated']}",
        f"- provisional：{acceptance['provisional']}",
        f"- 说明：{env.get('data_scope', DATA_SCOPE)['note']}",
    ]
    for reason in acceptance['reasons']:
        lines.append(f'- 阻断：{reason}')
    lines += [
        '',
        '## 口径',
        '',
        f"- dtype：`{env['dtype']}`",
        f"- threads：`{env['threads']}`（numpy 导入前强制）",
        f"- optimize：`{env['optimize']}`",
        f"- isolation：`{env['isolation']}`",
        f"- acceptance profiles：`{sorted(ACCEPTANCE_PROFILES)}`；"
        f"diagnostic profiles：`{sorted(DIAGNOSTIC_PROFILES)}`",
        f"- 计时：预热后 {env['measurement_order']}，每实现不少于 "
        f"{env['iterations']} 次，取中位数",
        f"- planner：`{env.get('planner', 'shared')}`"
        + ('（双方各自规划一次，只计时稳态执行）'
           if env.get('planner') == 'production'
           else '（双方共用 oracle 显式 path，只隔离执行层）'),
        f"- 可比性门槛：双方中位耗时均 ≥ `{MIN_MEANINGFUL_SECONDS}` s，"
        '否则记 `too_short` 且不计入 3x 判定',
        '- 正确性：`max |actual-expected| / max(|expected|, tiny)` '
        f"≤ `{RELATIVE_ERROR_TOLERANCE}`",
        '- GPU：每次计时前后调用 default stream synchronize',
        '',
        '## 环境',
        '',
        f"- oracle：{oracle.get('label', 'error')}；version="
        f"`{oracle.get('version', 'unknown')}`；"
        f"frozen={oracle.get('frozen', False)}",
        f"- oracle path：`{oracle.get('path', oracle.get('error', ''))}`",
        f"- pyqcd.einsum：`{pyqcd.get('version', 'unknown')}`；"
        f"native=`{pyqcd.get('native_version', 'unknown')}`",
        f"- pyqcd path：`{pyqcd.get('path', pyqcd.get('error', ''))}`",
        f"- Python：`{env['python']}`；NumPy：`{env['numpy']}`",
        f"- CPU：`{env['cpu_count']}` logical CPUs",
    ]
    if env.get('worker_pids'):
        lines.append(f"- worker pids：`{env['worker_pids']}`")
    for backend, device in env['devices'].items():
        lines.append(f"- {backend} device：`{device}`")

    lines += [
        '',
        '## 结果',
        '',
        '| profile | eq | backend | isolation | worker pids | optimize | '
        'planner | opt median/s | qcd median/s | speedup | max rel err | '
        'planner path match | status |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for row in rows:
        opt = row.get('opt_einsum', {}).get('median_s')
        qcd = row.get('qcd_einsum', {}).get('median_s')
        speedup = row.get('speedup_vs_opt')
        verdict = ('too_short(no-comparison)' if row.get('too_short')
                   else 'PASS' if row.get('speedup_pass') else (
                       'DIAGNOSTIC(<3x)'
                       if row['profile'] in DIAGNOSTIC_PROFILES
                       else 'FAIL(<3x)'))
        relative = row.get('max_relative_error')
        lines.append(
            f"| {row['profile']} | {row['eq']} | {row['backend']} | "
            f"{row['isolation']} | {row.get('worker_pid') or ''} | "
            f"{row['optimize']} | {row.get('planner', '')} | "
            f"{opt if opt is not None else ''} | "
            f"{qcd if qcd is not None else ''} | "
            f"{speedup if speedup is not None else ''} | "
            f"{relative if relative is not None else ''} | "
            f"{row.get('planner_path_match', '')} | {row['status']} / {verdict} |")

    if rows:
        lines += ['', '## 显式 path', '']
        for row in rows:
            if row.get('path') is not None:
                lines.append(
                    f"- {row['profile']}/{row['eq']}/{row['backend']}: "
                    f"`{row['path']}`；digest=`{row['path_digest']}`")
    return '\n'.join(lines) + '\n'


def _print_summary(rows, env, acceptance, directory):
    print('\n──────────── benchmark 汇总 ────────────')
    print(f"模式: {env['mode']}；验收状态: {acceptance['state']}")
    print(f"oracle: {env.get('opt_einsum', {}).get('label', 'error')}")
    print(f"线程: {env['threads']}；dtype: {env['dtype']}；"
          f"optimize: {env['optimize']}；isolation: {env['isolation']}")
    for row in rows:
        if row['status'] == 'ok':
            verdict = 'PASS' if row.get('speedup_pass') else (
                'DIAGNOSTIC(<3x)' if row['profile'] in DIAGNOSTIC_PROFILES
                else 'FAIL(<3x)')
            print(
                f"  {row['profile']:<4s} {row['eq']:<3s} "
                f"{row['backend']:<4s} "
                f"speedup={row['speedup_vs_opt']:.3f}x "
                f"max_rel={row['max_relative_error']:.3e} "
                f"{verdict}")
        else:
            print(
                f"  {row['profile']:<4s} {row['eq']:<3s} "
                f"{row['backend']:<4s} ERROR {row['note']}")
    for reason in acceptance['reasons']:
        print(f'  阻断: {reason}')
    print(f'\n产物: {directory}')
    print('       bench.json / bench.md / env.json')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='qcd-einsum vs opt_einsum 严格 3x 蒸馏基准')
    parser.add_argument('--profiles', default=','.join(DEFAULT_PROFILES),
                        help='逗号列表：T0n,T0,T1,T2；T1/T2 为性能验收 profile，T0/T0n 为诊断')
    parser.add_argument('--backends', default=','.join(DEFAULT_BACKENDS),
                        help='逗号列表：cpu,gpu')
    parser.add_argument('--equations',
                        default=','.join(item['name'] for item in EQUATIONS))
    parser.add_argument('--iters', type=int,
                        default=MIN_ACCEPTANCE_ITERATIONS,
                        help='每实现中位数样本数，验收模式必须 >=10')
    parser.add_argument('--warmup', type=int, default=MIN_WARMUP,
                        help='每实现预热周期数')
    parser.add_argument('--optimize', default='dp',
                        help='收缩规划器；默认 dp（opt_einsum 常用策略）')
    parser.add_argument('--isolation', choices=('abba', 'process'),
                        default='process',
                        help='process=每实现全新解释器；abba=进程内交错计时')
    parser.add_argument('--planner', choices=('production', 'shared'),
                        default='production',
                        help='production=两边各自按 --optimize 规划（无缝替换语义）；'
                             'shared=两边共用 oracle 的显式 path（只隔离执行层）')
    parser.add_argument('--data', choices=DATA_MODES, default='real',
                        help='real=真实 perambulator；synthetic=固定种子合成操作数'
                             '（形状一致，正确性以 opt_einsum 对照为准）')
    parser.add_argument('--outdir', default=None,
                        help='产物目录（默认 artifacts/bench_<时间戳>/）')
    parser.add_argument('--smoke', action='store_true',
                        help='T0n/CPU 冒烟；报告明确 NON_ACCEPTANCE 并返回 2')
    parser.add_argument('--worker-task', default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker_task:
        return _run_worker_task(args.worker_task)

    if args.smoke:
        profiles = ('T0n',)
        backends = ('cpu',)
        equations = tuple(item['name'] for item in EQUATIONS)
        iterations = MIN_ACCEPTANCE_ITERATIONS
        warmup = max(1, args.warmup)
        mode = 'smoke'
    else:
        profiles = tuple(item for item in args.profiles.split(',') if item)
        backends = tuple(item for item in args.backends.split(',') if item)
        equations = tuple(item for item in args.equations.split(',') if item)
        iterations = args.iters
        warmup = args.warmup
        mode = 'full'
        if iterations < MIN_ACCEPTANCE_ITERATIONS:
            parser.error(
                f'验收模式 --iters 必须 >= {MIN_ACCEPTANCE_ITERATIONS}')
        if warmup < 1:
            parser.error('验收模式 --warmup 必须 >= 1')

    print('════════ qcd-einsum vs opt_einsum 3x 硬门禁 ════════')
    rows, env, problems = run(
        profiles=profiles, backends=backends, equations=equations,
        iterations=iterations, warmup=warmup, optimize=args.optimize,
        mode=mode, isolation=args.isolation, data_mode=args.data,
        planner=args.planner)
    acceptance = _acceptance(
        rows, env, problems, mode, requested_backends=backends)
    directory = artifact_dir(args.outdir)
    write_artifacts(rows, env, acceptance, directory)
    _print_summary(rows, env, acceptance, directory)
    return acceptance['return_code']


if __name__ == '__main__':
    raise SystemExit(main())
