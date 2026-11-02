"""Resumable production driver for the gradient-flow gluon TMD chain.

The module orchestrates existing PyQCD entry points.  It intentionally keeps
physics definitions in the production package and only adds workflow state,
resource guards, and canonical artifact discovery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_DEPS = PROJECT_ROOT / "data" / "deps"
if LOCAL_DEPS.is_dir() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

from pyqcd.operator._gluon_ope import read_gauge_lime
from pyqcd.pipeline._config import (
    ALttc, NEV, NEV1, NT, NX, get_gauge_path,
)
from pyqcd.pipeline._steps import (
    compute_2pt_for_config,
    compute_3pt_for_config,
    free_gpu_memory,
)
from pyqcd.pipeline._tmd9 import (
    _flow_cache_contract,
    _gauge_source_identity,
    compute_2pt_polarizations_multi,
    compute_tmd_ope_time,
    compute_tmd_ope_time_2d,
    compute_vertices_multi,
    flow_gauge_for_config,
)
from pyqcd.renorm import flow_action_density, wilson_flow_step
from pyqcd.tools import (
    get_backend, get_backend_name, set_backend, set_precision,
)
from pyqcd.tools._env import dump_env


CONF_IDS = (
    6250, 6450, 6650, 6850, 7050, 7250,
    7450, 7650, 7850, 8050, 8250, 8450,
)
MOMENTA_AXES = (
    (0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), (4, 0, 0),
)
FLOW_TIMES = (1.0, 2.0, 3.0)
Z_LIST = tuple(range(13))
B_LIST = tuple(range(7))
B_X_LIST = tuple(range(7))
B_Y_LIST = tuple(range(3))
STAPLE_LENGTH = 12
FLOW_EPS = 0.01
PRECISION = "complex64"
T_SEP_3PT = 12
DEFAULT_STAGES = ("distillation", "flow", "ope")
VALID_STAGES = frozenset(DEFAULT_STAGES)
NON_DISK_RESOURCE_FIELDS = (
    "cpu_percent", "memory_percent", "gpu_memory_percent",
)
RESOURCE_FIELDS = NON_DISK_RESOURCE_FIELDS + ("disk_percent",)
MAX_DISK_FRACTION = 0.95


class ResourceLimitError(RuntimeError):
    """Raised before a task when a declared resource exceeds its hard limit."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_fraction(max_fraction):
    if (isinstance(max_fraction, (bool, np.bool_))
            or not isinstance(max_fraction, (int, float, np.integer,
                                              np.floating))
            or not np.isfinite(max_fraction)
            or not 0.0 < float(max_fraction) <= 0.8):
        raise ValueError("max_fraction 必须是 (0, 0.8] 内的有限数")
    return float(max_fraction)


def _validate_disk_fraction(max_disk_fraction):
    if (isinstance(max_disk_fraction, (bool, np.bool_))
            or not isinstance(max_disk_fraction, (int, float, np.integer,
                                                   np.floating))
            or not np.isfinite(max_disk_fraction)
            or not 0.0 < float(max_disk_fraction) <= MAX_DISK_FRACTION):
        raise ValueError(
            "max_disk_fraction 必须是 (0, "
            f"{MAX_DISK_FRACTION:.2f}] 内的有限数")
    return float(max_disk_fraction)


def _throttle_backend_work(work_seconds, context):
    """Duty-cycle GPU kernels so wall-time occupancy stays at or below target."""
    target = float(context.get('gpu_duty_cycle', 0.8))
    if context.get('backend') == 'numpy' or target >= 1.0:
        return 0.0
    work_seconds = max(0.0, float(work_seconds))
    sleep_seconds = work_seconds * (1.0 / target - 1.0)
    if sleep_seconds > 0.0:
        time.sleep(sleep_seconds)
    return sleep_seconds


def validate_resource_snapshot(snapshot, max_fraction=0.8,
                               max_disk_fraction=None):
    """Reject process or disk use above their independent hard limits."""
    max_fraction = _validate_fraction(max_fraction)
    if max_disk_fraction is None:
        max_disk_fraction = max_fraction
    else:
        max_disk_fraction = _validate_disk_fraction(max_disk_fraction)
    for key in RESOURCE_FIELDS:
        value = snapshot.get(key)
        if value is None:
            continue
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer,
                                          np.floating))
                or not np.isfinite(value)
                or not 0.0 <= float(value) <= 100.0):
            raise ValueError(f"{key} 必须是 [0,100] 内的有限百分比")
        limit_fraction = (
            max_disk_fraction if key == 'disk_percent' else max_fraction)
        limit = 100.0 * limit_fraction
        if float(value) > limit:
            raise ResourceLimitError(
                f"{key}={float(value):.2f}% 超过硬上限 {limit:.2f}%")
    return snapshot


def _gpu_memory_percent(backend, device):
    if backend == "torch":
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            index = int(device)
            free, total = torch.cuda.mem_get_info(index)
            return 100.0 * (1.0 - float(free) / float(total))
        except (ImportError, RuntimeError, ValueError):
            return None
    if backend == "cupy":
        try:
            import cupy as cp
            with cp.cuda.Device(int(device)):
                free, total = cp.cuda.runtime.memGetInfo()
            return 100.0 * (1.0 - float(free) / float(total))
        except (ImportError, RuntimeError, ValueError):
            return None
    return None


def resource_snapshot(backend='numpy', device='cpu', path=None):
    vm = psutil.virtual_memory()
    target = Path(path) if path is not None else PROJECT_ROOT
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    disk = shutil.disk_usage(probe)
    return {
        'time_utc': _utc_now(),
        'cpu_percent': float(psutil.cpu_percent(interval=0.2)),
        'memory_percent': float(vm.percent),
        'disk_percent': 100.0 * float(disk.used) / float(disk.total),
        'gpu_memory_percent': _gpu_memory_percent(backend, device),
    }


def _assert_disk_reserve(path, max_disk_fraction, reserve_bytes):
    max_disk_fraction = _validate_disk_fraction(max_disk_fraction)
    target = Path(path)
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    disk = shutil.disk_usage(probe)
    projected = disk.used + max(0, int(reserve_bytes))
    if projected > max_disk_fraction * disk.total:
        raise ResourceLimitError(
            f'磁盘预留失败: projected={projected / 2**30:.2f} GiB, '
            f'limit={max_disk_fraction * disk.total / 2**30:.2f} GiB')


def _sanitize_worker_tag(value):
    if value is None:
        return None
    tag = str(value)
    if (not tag or len(tag) > 64
            or any(not (char.isalnum() or char in '._-') for char in tag)):
        raise ValueError('worker_tag 必须是 1-64 位字母、数字、点、下划线或连字符')
    return tag


def _checkpoint_name(backend, device, worker_tag=None):
    if worker_tag is not None:
        return f'worker_{worker_tag}.json'
    if str(device) == 'cpu':
        return 'cpu.json'
    device_tag = str(device).replace(':', '_').replace('/', '_')
    return f"worker_{backend}_{device_tag}.json"


def _validate_conf_ids(conf_ids):
    values = tuple(int(value) for value in conf_ids)
    if not values:
        raise ValueError("conf_ids 不能为空")
    if len(set(values)) != len(values):
        raise ValueError("conf_ids 不得重复")
    return values


def _validate_stages(stages):
    values = tuple(str(value) for value in stages)
    unknown = set(values) - VALID_STAGES
    if unknown:
        raise ValueError(f"未知阶段: {sorted(unknown)}")
    if not values:
        raise ValueError("stages 不能为空")
    return values


def _normalise_list(values, name, *, nonnegative=True):
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是非布尔整数序列") from exc
    if not result:
        raise ValueError(f"{name} 不能为空")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} 不得重复")
    if nonnegative and min(result) < 0:
        raise ValueError(f"{name} 不得为负")
    return result


def _read_raw_gauge(conf_id, precision):
    """Read one gauge configuration as an array on the active backend."""
    dtype = np.complex64 if precision == 'complex64' else np.complex128
    gauge = read_gauge_lime(get_gauge_path(int(conf_id)), NT, NX).astype(dtype)
    return get_backend().asarray(gauge)


def compute_flow_history(conf_id, *, tau_values=FLOW_TIMES, eps=FLOW_EPS,
                         backend='numpy', precision='complex128',
                         record_interval=0.1, device=0):
    """Flow one raw gauge field and return action-density diagnostics.

    The history is diagnostic only.  Production caches for each requested flow
    time are rebuilt independently by ``flow_gauge_for_config`` so no OPE input
    is chained from a late-flow field.
    """
    tau_values = tuple(float(value) for value in tau_values)
    if not tau_values or min(tau_values) <= 0.0:
        raise ValueError("tau_values 必须为有限正数")
    if len(set(tau_values)) != len(tau_values):
        raise ValueError("tau_values 不得重复")
    tau_values = tuple(sorted(tau_values))
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError('eps 必须为有限正数')
    max_tau = tau_values[-1]
    n_steps = int(math.ceil(max_tau / eps))
    step_size = max_tau / n_steps
    interval_steps = max(1, int(round(float(record_interval) / step_size)))

    device = 'cpu' if backend == 'numpy' else int(device)
    set_backend(backend, device=device)
    if backend == 'torch':
        set_precision(precision)
    gauge = _read_raw_gauge(int(conf_id), precision)
    times = [0.0]
    actions = [float(flow_action_density(gauge).mean())]
    checkpoint_action = []
    tau_index = 0
    current_t = 0.0
    for step in range(n_steps):
        gauge = wilson_flow_step(gauge, step_size)
        current_t = (step + 1) * step_size
        crossed = []
        while (tau_index < len(tau_values)
               and current_t + 1.0e-10 >= tau_values[tau_index]):
            crossed.append(tau_values[tau_index])
            tau_index += 1
        record = ((step + 1) % interval_steps == 0
                  or step == n_steps - 1 or crossed)
        if record:
            value = float(flow_action_density(gauge).mean())
            if times and abs(times[-1] - current_t) <= 1.0e-10:
                actions[-1] = value
            else:
                times.append(current_t)
                actions.append(value)
            checkpoint_action.extend([value] * len(crossed))
    flow_time = np.asarray(times, dtype=np.float64)
    action = np.asarray(actions, dtype=np.float64)
    if not np.isfinite(flow_time).all() or not np.isfinite(action).all():
        raise FloatingPointError('flow history 含非有限值')
    t2_action = flow_time ** 2 * action
    target = np.asarray(tau_values, dtype=np.float64)
    checkpoint = np.asarray(checkpoint_action, dtype=np.float64)
    if checkpoint.size != target.size:
        raise RuntimeError('未在每个请求流时记录 action density')
    t0 = np.nan
    t0_status = 'not_reached'
    crossing = np.flatnonzero(t2_action >= 0.3)
    if crossing.size:
        index = int(crossing[0])
        if index == 0:
            t0 = float(flow_time[index])
        else:
            t0 = float(np.interp(
                0.3,
                t2_action[index - 1:index + 1],
                flow_time[index - 1:index + 1],
            ))
        t0_status = 'interpolated'
    return {
        'flow_time': flow_time,
        'action_density': action,
        't2_action_density': t2_action,
        'tau_checkpoints': target,
        'checkpoint_action': checkpoint,
        't0': t0,
        't0_status': t0_status,
        'eps': step_size,
        'n_steps': n_steps,
    }


def _npz_complete(path, required):
    try:
        with np.load(path, allow_pickle=False) as handle:
            return all(name in handle.files for name in required)
    except (OSError, ValueError):
        return False


def _artifact_list_ready(paths):
    try:
        values = list(paths)
    except TypeError:
        return False
    if not values:
        return False
    for value in values:
        if isinstance(value, dict):
            path = Path(value.get('path', ''))
            expected_size = int(value.get('size_bytes', -1))
            expected_mtime = int(value.get('mtime_ns', -1))
            if (not path.is_file() or path.stat().st_size != expected_size
                    or path.stat().st_mtime_ns != expected_mtime):
                return False
        else:
            path = Path(value)
            if not path.is_file() or path.stat().st_size <= 0:
                return False
    return True


def _artifact_records(paths):
    records = []
    for value in paths:
        path = Path(value)
        stat = path.stat()
        if not path.is_file() or stat.st_size <= 0:
            raise RuntimeError(f'阶段产物缺失或为空: {path}')
        records.append({
            'path': str(path),
            'size_bytes': int(stat.st_size),
            'mtime_ns': int(stat.st_mtime_ns),
        })
    return records


def _h5_array_complete(path, shape, dtype):
    import h5py
    try:
        with h5py.File(path, 'r') as handle:
            if set(handle.keys()) != {'data'}:
                return False
            data = handle['data']
            if data.shape != tuple(shape) or data.dtype != np.dtype(dtype):
                return False
            return bool(np.isfinite(data[...]).all())
    except (OSError, ValueError, TypeError):
        return False


def _save_flow_history(run_dir, conf_id, payload, contract=None):
    path = Path(run_dir) / 'flow_history' / f'conf{int(conf_id)}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)
    if contract is not None:
        _atomic_json(path.with_suffix('.json'), contract)
    return path


def _flow_history_contract(conf_id, context):
    return {
        'schema': 'pyqcd.flow_history.v1',
        'conf_id': int(conf_id),
        'tau_values': list(context['tau_values']),
        'eps': float(context['eps']),
        'precision': context['precision'],
        'backend': context['backend'],
        'gauge_source': _gauge_source_identity(int(conf_id)),
    }


def _flow_history_complete(path, conf_id, context):
    required = (
        'flow_time', 'action_density', 't2_action_density',
        'tau_checkpoints', 'checkpoint_action',
    )
    path = Path(path)
    sidecar = path.with_suffix('.json')
    if not _npz_complete(path, required) or not sidecar.is_file():
        return False
    try:
        contract = json.loads(sidecar.read_text(encoding='utf-8'))
        expected = _flow_history_contract(conf_id, context)
    except (OSError, ValueError, TypeError, KeyError):
        return False
    if contract != expected:
        return False
    try:
        with np.load(path, allow_pickle=False) as handle:
            arrays = {name: np.asarray(handle[name]) for name in required}
    except (OSError, ValueError):
        return False
    if not all(np.isfinite(value).all() for value in arrays.values()):
        return False
    return np.allclose(
        arrays['tau_checkpoints'], np.asarray(context['tau_values']),
        rtol=0.0, atol=1.0e-12,
    )


def _run_distillation_stage(conf_id, context, check_resources):
    run_dir = Path(context['run_dir'])
    momenta = context['momenta']
    precision = context['precision']
    check_resources('vertices', conf_id)
    started = time.perf_counter()
    vertices = compute_vertices_multi(
        conf_id, run_dir, context['logger'], momenta,
        precision=precision, recompute=context['recompute'])
    _throttle_backend_work(time.perf_counter() - started, context)

    check_resources('polarized-2pt', conf_id)
    started = time.perf_counter()
    compute_2pt_polarizations_multi(
        conf_id, run_dir, context['logger'], vertices, momenta,
        precision=precision, recompute=context['recompute'])
    _throttle_backend_work(time.perf_counter() - started, context)

    generic_2pt = (
        'corr_pp_P0', 'corr_pp_P2', 'corr_pion_P0', 'corr_pion_P2')
    cdir = run_dir / 'data' / f'conf{conf_id}'
    generic_complete = all(
        _h5_array_complete(cdir / f'{name}_{conf_id}.h5', (NT,), np.float64)
        for name in generic_2pt
    )
    if context['recompute'] or not generic_complete:
        check_resources('connected-2pt', conf_id)
        started = time.perf_counter()
        compute_2pt_for_config(
            conf_id, run_dir, context['logger'], vertices,
            precision=precision, channels=('pp', 'pion'))
        _throttle_backend_work(time.perf_counter() - started, context)

    threept = (
        f'proton_P0_3pt_{conf_id}.h5',
        f'proton_P2_3pt_{conf_id}.h5',
        f'pion_P0_3pt_{conf_id}.h5',
        f'pion_P2_3pt_{conf_id}.h5',
    )
    threept_complete = all(
        _h5_array_complete(cdir / name,
                           (context['t_sep'] + 1, 4), np.float64)
        for name in threept
    )
    if context['recompute'] or not threept_complete:
        check_resources('connected-3pt', conf_id)
        started = time.perf_counter()
        compute_3pt_for_config(
            conf_id, run_dir, context['logger'], vertices,
            precision=precision, t_sep=context['t_sep'])
        _throttle_backend_work(time.perf_counter() - started, context)
    del vertices
    free_gpu_memory()
    artifact_paths = sorted(
        str(path) for path in cdir.glob('*')
        if path.is_file() and (
            path.name.startswith(('VdV_', 'VVV_'))
            or path.name.startswith('corr_pp_')
            or path.name.startswith('corr_pion_')
            or '3pt' in path.name
        )
    )
    return {
        'artifacts': _artifact_records(artifact_paths),
        'vertices': 'complete',
        'polarized_2pt': 'complete',
        'connected_2pt': 'recomputed' if not generic_complete else 'cache_hit',
        'connected_3pt': 'recomputed' if not threept_complete else 'cache_hit',
    }


def _save_contract_flow_gauge(conf_id, tau, context):
    run_dir = Path(context['run_dir'])
    flow_dir = run_dir / 'flowed'
    flow_dir.mkdir(parents=True, exist_ok=True)
    precision = context['precision']
    input_source = _gauge_source_identity(int(conf_id))
    contract = _flow_cache_contract(
        int(conf_id), float(tau), float(context['eps']), precision,
        input_source=input_source,
    )
    from pyqcd.pipeline._tmd9 import _flow_cache_tag
    path = flow_dir / (
        f"flowed_gauge_{_flow_cache_tag(contract)}_conf{int(conf_id)}.h5")
    gauge = flow_gauge_for_config(
        int(conf_id), float(tau), float(context['eps']), precision,
        save_dir=str(flow_dir), logger=context['logger'], save_gauge=True,
        input_source_identity=input_source,
    )
    del gauge
    free_gpu_memory()
    return str(path)


def _run_flow_stage(conf_id, context, check_resources):
    run_dir = Path(context['run_dir'])
    paths = []
    for tau in context['tau_values']:
        check_resources(f'flow-tau{tau:g}', conf_id)
        started = time.perf_counter()
        paths.append(_save_contract_flow_gauge(conf_id, tau, context))
        _throttle_backend_work(time.perf_counter() - started, context)
    history_path = run_dir / 'flow_history' / f'conf{int(conf_id)}.npz'
    if (context['recompute']
            or not _flow_history_complete(history_path, conf_id, context)):
        check_resources('flow-history', conf_id)
        started = time.perf_counter()
        history = compute_flow_history(
            conf_id, tau_values=context['tau_values'],
            eps=context['eps'], backend=context['backend'],
            precision=context['precision'], device=context['device'],
        )
        _throttle_backend_work(time.perf_counter() - started, context)
        _save_flow_history(
            run_dir, conf_id, history,
            contract=_flow_history_contract(conf_id, context))
    return {
        'artifacts': _artifact_records(paths + [
            str(history_path), str(history_path.with_suffix('.json'))]),
        'flow_gauge': paths,
        'flow_history': str(history_path),
    }


def _direction_grid(z_dir):
    if z_dir == 0:
        return 0, 1
    if z_dir == 1:
        return 1, 2
    return 2, 0


def _run_ope_stage(conf_id, context, check_resources):
    run_dir = Path(context['run_dir'])
    flow_dir = run_dir / 'flowed'
    completed = []
    for tau in context['tau_values']:
        for z_dir in (0, 1, 2):
            b_dir = _direction_grid(z_dir)[1]
            check_resources(f'full-ope-z{z_dir}-tau{tau:g}', conf_id)
            started = time.perf_counter()
            compute_tmd_ope_time(
                conf_id, run_dir, context['logger'], context['z_list'],
                context['b_list'], tau=tau, eps=context['eps'],
                precision=context['precision'], z_dir=z_dir,
                b_dir=b_dir, recompute=context['recompute'],
                gauge_flow_dir=str(flow_dir),
                staple_length=context['staple_length'], full_channels=True,
            )
            _throttle_backend_work(time.perf_counter() - started, context)
            free_gpu_memory()
            completed.append(('full', tau, z_dir))
        for z_dir in (0, 1, 2):
            b_dir = _direction_grid(z_dir)[1]
            check_resources(f'legacy-ope-z{z_dir}-tau{tau:g}', conf_id)
            started = time.perf_counter()
            compute_tmd_ope_time(
                conf_id, run_dir, context['logger'], context['z_list'],
                context['b_list'], tau=tau, eps=context['eps'],
                precision=context['precision'], z_dir=z_dir,
                b_dir=b_dir, recompute=context['recompute'],
                gauge_flow_dir=str(flow_dir),
                staple_length=context['staple_length'], full_channels=False,
            )
            _throttle_backend_work(time.perf_counter() - started, context)
            free_gpu_memory()
            completed.append(('legacy', tau, z_dir))
        for order in ((0, 1), (1, 0)):
            check_resources(
                f'2d-ope-order{order[0]}{order[1]}-tau{tau:g}', conf_id)
            started = time.perf_counter()
            compute_tmd_ope_time_2d(
                conf_id, run_dir, context['logger'], context['z_list'],
                context['b_x_list'], context['b_y_list'], tau=tau,
                eps=context['eps'], precision=context['precision'],
                z_dir=2, b_x_dir=0, b_y_dir=1,
                transverse_order=order, recompute=context['recompute'],
                gauge_flow_dir=str(flow_dir),
                staple_length=context['staple_length'],
            )
            _throttle_backend_work(time.perf_counter() - started, context)
            free_gpu_memory()
            completed.append(('2d', tau, order))
    return {
        'artifacts': _artifact_records(sorted(
            str(path) for path in (run_dir / 'data' / f'conf{conf_id}').glob(
                f'tmd_ope_*_conf{conf_id}.h5')
        )),
        'completed_calls': len(completed),
        'tau_values': list(context['tau_values']),
        'z_list': list(context['z_list']),
        'b_list': list(context['b_list']),
        'b_x_list': list(context['b_x_list']),
        'b_y_list': list(context['b_y_list']),
    }


def _default_stage_executor(stage, conf_id, context):
    checker = context['check_resources']
    if stage == 'distillation':
        return _run_distillation_stage(conf_id, context, checker)
    if stage == 'flow':
        return _run_flow_stage(conf_id, context, checker)
    if stage == 'ope':
        return _run_ope_stage(conf_id, context, checker)
    raise ValueError(f"未知阶段: {stage}")


def run_worker(run_dir, conf_ids=CONF_IDS, *, backend='torch', device=0,
               stages=DEFAULT_STAGES, precision=PRECISION,
               momenta=MOMENTA_AXES, tau_values=FLOW_TIMES,
               z_list=Z_LIST, b_list=B_LIST, b_x_list=B_X_LIST,
               b_y_list=B_Y_LIST, staple_length=STAPLE_LENGTH,
               eps=FLOW_EPS, t_sep=T_SEP_3PT, cpu_threads=4,
               recompute=False, max_resource_fraction=0.8,
               max_disk_fraction=0.8,
               gpu_duty_cycle=0.8,
               worker_tag=None,
               stage_executor=None, resource_checker=None,
               logger=print):
    """Run one disjoint configuration shard with stage-level checkpoints."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    conf_ids = _validate_conf_ids(conf_ids)
    stages = _validate_stages(stages)
    max_resource_fraction = _validate_fraction(max_resource_fraction)
    max_disk_fraction = _validate_disk_fraction(max_disk_fraction)
    gpu_duty_cycle = _validate_fraction(gpu_duty_cycle)
    worker_tag = _sanitize_worker_tag(worker_tag)
    if isinstance(cpu_threads, (bool, np.bool_)) or int(cpu_threads) <= 0:
        raise ValueError('cpu_threads 必须为正整数')
    cpu_threads = int(cpu_threads)
    if cpu_threads * 3 > int(psutil.cpu_count() * 0.8):
        raise ValueError('三个 worker 的 CPU 线程总数不得超过 80%')
    if backend not in ('torch', 'cupy', 'numpy'):
        raise ValueError("backend 必须是 torch/cupy/numpy")
    if backend == 'numpy':
        device = 'cpu'
    else:
        device = int(device)
    momenta = tuple(tuple(int(value) for value in mom) for mom in momenta)
    tau_values = tuple(sorted(float(value) for value in tau_values))
    z_list = _normalise_list(z_list, 'z_list')
    b_list = _normalise_list(b_list, 'b_list')
    b_x_list = _normalise_list(b_x_list, 'b_x_list')
    b_y_list = _normalise_list(b_y_list, 'b_y_list')

    set_backend(backend, device=None if backend == 'numpy' else device)
    if backend == 'torch':
        set_precision(precision)
        import torch
        torch.set_num_threads(cpu_threads)

    request_contract = {
        'schema_version': 3,
        'backend': backend,
        'device': device,
        'precision': precision,
        'conf_ids': list(conf_ids),
        'stages': list(stages),
        'momenta': [list(value) for value in momenta],
        'tau_values': list(tau_values),
        'z_list': list(z_list),
        'b_list': list(b_list),
        'b_x_list': list(b_x_list),
        'b_y_list': list(b_y_list),
        'staple_length': int(staple_length),
        'eps': float(eps),
        't_sep': int(t_sep),
        'gpu_duty_cycle': gpu_duty_cycle,
        'worker_tag': worker_tag,
        'cpu_threads': cpu_threads,
        'max_resource_fraction': max_resource_fraction,
        'max_disk_fraction': max_disk_fraction,
        'input_sources': {
            str(conf_id): _gauge_source_identity(int(conf_id))
            for conf_id in conf_ids
        },
    }
    request_bytes = json.dumps(
        request_contract, sort_keys=True, separators=(',', ':')).encode('utf-8')
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    checkpoint = run_dir / 'workers' / _checkpoint_name(
        backend, device, worker_tag)
    if checkpoint.exists():
        payload = json.loads(checkpoint.read_text(encoding='utf-8'))
        if payload.get('run_dir') != str(run_dir):
            raise ValueError('checkpoint run_dir 与当前运行不一致')
        previous_host = payload.get('hostname')
        previous_pid = payload.get('pid')
        if (payload.get('status') == 'running'
                and previous_host == socket.gethostname()
                and previous_pid is not None and int(previous_pid) != os.getpid()
                and psutil.pid_exists(int(previous_pid))):
            raise RuntimeError(
                f'worker 已由活动进程 PID={previous_pid} 占用同一 checkpoint')
        previous_request = payload.get('request_sha256')
        previous_version = int(
            payload.get('request_contract', {}).get('schema_version', 1))
        if (previous_version >= 3 and previous_request is not None
                and previous_request != request_sha256):
            raise ValueError('checkpoint request contract 与当前运行不一致')
        if previous_request is not None and previous_version < 3:
            payload.setdefault('migrations', []).append({
                'from_schema_version': previous_version,
                'to_schema_version': 3,
                'time_utc': _utc_now(),
            })
        payload['request_contract'] = request_contract
        payload['request_sha256'] = request_sha256
        payload['worker_tag'] = worker_tag
        payload['cpu_threads'] = cpu_threads
        payload['max_resource_fraction'] = max_resource_fraction
        payload['max_disk_fraction'] = max_disk_fraction
        payload['hostname'] = socket.gethostname()
        payload['pid'] = os.getpid()
        payload['status'] = 'running'
        payload['resumed_utc'] = _utc_now()
        payload.pop('finished_utc', None)
        payload.pop('failed_conf_id', None)
        payload.pop('failed_stage', None)
    else:
        payload = {
            'schema': 'pyqcd.tmd_full_production_worker.v1',
            'run_dir': str(run_dir),
            'backend': backend,
            'device': device,
            'precision': precision,
            'conf_ids': list(conf_ids),
            'stages': list(stages),
            'momenta': [list(value) for value in momenta],
            'tau_values': list(tau_values),
            'z_list': list(z_list),
            'b_list': list(b_list),
            'b_x_list': list(b_x_list),
            'b_y_list': list(b_y_list),
            'staple_length': int(staple_length),
            'eps': float(eps),
            't_sep': int(t_sep),
            'cpu_threads': cpu_threads,
            'max_resource_fraction': max_resource_fraction,
            'max_disk_fraction': max_disk_fraction,
            'gpu_duty_cycle': gpu_duty_cycle,
            'worker_tag': worker_tag,
            'request_contract': request_contract,
            'request_sha256': request_sha256,
            'started_utc': _utc_now(),
            'hostname': socket.gethostname(),
            'pid': os.getpid(),
            'status': 'running',
            'configurations': {},
        }

    resource_tag = worker_tag or (
        f"{backend}_{str(device).replace(':', '_')}")
    resource_path = run_dir / 'workers' / f'resources_{resource_tag}.jsonl'
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    default_checker = lambda: resource_snapshot(
        backend=backend, device=device, path=run_dir)
    checker_base = resource_checker or default_checker

    def check_resources(label, conf_id):
        snapshot = validate_resource_snapshot(
            checker_base(),
            max_fraction=max_resource_fraction,
            max_disk_fraction=max_disk_fraction,
        )
        reserve_gib = 0.0
        if label == 'vertices':
            reserve_gib = 3.2
        elif str(label).startswith('flow-tau'):
            reserve_gib = 0.35
        elif str(label).startswith(('full-ope', 'legacy-ope', '2d-ope')):
            reserve_gib = 0.03
        _assert_disk_reserve(
            run_dir, max_disk_fraction, reserve_gib * 2**30)
        snapshot.update({
            'label': str(label),
            'conf_id': int(conf_id),
            'backend': backend,
            'device': device,
        })
        with resource_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(snapshot, sort_keys=True) + '\n')
        return snapshot

    context = {
        'run_dir': run_dir,
        'backend': backend,
        'device': device,
        'precision': precision,
        'momenta': momenta,
        'tau_values': tau_values,
        'z_list': z_list,
        'b_list': b_list,
        'b_x_list': b_x_list,
        'b_y_list': b_y_list,
        'staple_length': int(staple_length),
        'eps': float(eps),
        't_sep': int(t_sep),
        'recompute': bool(recompute),
        'logger': logger,
        'check_resources': check_resources,
        'gpu_duty_cycle': gpu_duty_cycle,
    }
    executor = stage_executor or _default_stage_executor
    dump_env(
        run_dir / 'env' / f'{Path(checkpoint).stem}.json',
        root=str(PROJECT_ROOT),
    )
    _atomic_json(checkpoint, payload)
    logger(
        f"worker backend={backend} device={device} "
        f"configs={list(conf_ids)} stages={list(stages)}")
    for conf_id in conf_ids:
        record = payload['configurations'].setdefault(
            str(conf_id), {'stages': {}})
        for stage in stages:
            stage_record = record['stages'].get(stage, {})
            if (stage_record.get('status') == 'complete' and not recompute
                    and _artifact_list_ready(
                        stage_record.get('result', {}).get('artifacts', []))):
                logger(f"  conf={conf_id} stage={stage}: checkpoint hit")
                continue
            started = time.perf_counter()
            record['stages'][stage] = {'status': 'running', 'started_utc': _utc_now()}
            _atomic_json(checkpoint, payload)
            try:
                result = executor(stage, conf_id, context)
                stage_record = {
                    'status': 'complete',
                    'started_utc': record['stages'][stage]['started_utc'],
                    'finished_utc': _utc_now(),
                    'seconds': round(time.perf_counter() - started, 3),
                    'result': _json_safe(result),
                }
                record['stages'][stage] = stage_record
                record.pop('error', None)
                logger(
                    f"  conf={conf_id} stage={stage}: complete "
                    f"({stage_record['seconds']:.1f}s)")
            except BaseException as exc:
                record['stages'][stage] = {
                    'status': 'failed',
                    'started_utc': record['stages'][stage]['started_utc'],
                    'finished_utc': _utc_now(),
                    'seconds': round(time.perf_counter() - started, 3),
                    'error_type': type(exc).__name__,
                    'error': str(exc),
                }
                payload['status'] = 'failed'
                payload['failed_conf_id'] = conf_id
                payload['failed_stage'] = stage
                _atomic_json(checkpoint, payload)
                raise
            finally:
                payload['updated_utc'] = _utc_now()
                _atomic_json(checkpoint, payload)
    payload['status'] = 'complete'
    payload['hostname'] = socket.gethostname()
    payload['pid'] = os.getpid()
    payload.pop('failed_conf_id', None)
    payload.pop('failed_stage', None)
    payload['finished_utc'] = _utc_now()
    payload['checkpoint'] = str(checkpoint)
    payload['resource_log'] = str(resource_path)
    _atomic_json(checkpoint, payload)
    return payload


def _parse_int_list(text):
    return [int(value) for value in str(text).split(',') if value != '']


def _parse_float_list(text):
    return [float(value) for value in str(text).split(',') if value != '']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--conf-ids', default=','.join(map(str, CONF_IDS)))
    parser.add_argument('--backend', choices=('torch', 'cupy', 'numpy'),
                        default='torch')
    parser.add_argument('--device', default='0')
    parser.add_argument('--stages', default=','.join(DEFAULT_STAGES))
    parser.add_argument('--precision', choices=('complex64', 'complex128'),
                        default=PRECISION)
    parser.add_argument('--momenta', default=None,
                        help='semicolon-separated pz,py,px tuples')
    parser.add_argument('--tau-list', default=','.join(map(str, FLOW_TIMES)))
    parser.add_argument('--z-list', default=','.join(map(str, Z_LIST)))
    parser.add_argument('--b-list', default=','.join(map(str, B_LIST)))
    parser.add_argument('--bx-list', default=','.join(map(str, B_X_LIST)))
    parser.add_argument('--by-list', default=','.join(map(str, B_Y_LIST)))
    parser.add_argument('--staple-length', type=int, default=STAPLE_LENGTH)
    parser.add_argument('--eps', type=float, default=FLOW_EPS)
    parser.add_argument('--t-sep', type=int, default=T_SEP_3PT)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--max-resource-fraction', type=float, default=0.8)
    parser.add_argument('--max-disk-fraction', type=float, default=0.8)
    parser.add_argument('--gpu-duty-cycle', type=float, default=0.8)
    parser.add_argument('--worker-tag', default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    conf_ids = _parse_int_list(args.conf_ids)
    stages = tuple(value for value in args.stages.split(',') if value)
    momenta = MOMENTA_AXES
    if args.momenta:
        momenta = tuple(
            tuple(int(component) for component in item.split(','))
            for item in args.momenta.split(';')
        )
    device = args.device if args.device.lower() == 'cpu' else int(args.device)
    result = run_worker(
        args.run_dir,
        conf_ids,
        backend=args.backend,
        device=device,
        stages=stages,
        precision=args.precision,
        momenta=momenta,
        tau_values=_parse_float_list(args.tau_list),
        z_list=_parse_int_list(args.z_list),
        b_list=_parse_int_list(args.b_list),
        b_x_list=_parse_int_list(args.bx_list),
        b_y_list=_parse_int_list(args.by_list),
        staple_length=args.staple_length,
        eps=args.eps,
        t_sep=args.t_sep,
        cpu_threads=args.cpu_threads,
        recompute=args.recompute,
        max_resource_fraction=args.max_resource_fraction,
        max_disk_fraction=args.max_disk_fraction,
        gpu_duty_cycle=args.gpu_duty_cycle,
        worker_tag=args.worker_tag,
    )
    print(json.dumps({
        'status': result['status'],
        'checkpoint': result['checkpoint'],
        'resource_log': result['resource_log'],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
