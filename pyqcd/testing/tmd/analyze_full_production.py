"""Analyze the 12-configuration lattice-only gradient-flow gluon TMD run.

All outputs remain diagnostic: no soft/rapidity measurement, no perturbative
gluon-quark matching, and no continuum extrapolation are available here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = [
    'AR PL UMing CN', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_DEPS = PROJECT_ROOT / "data" / "deps"
if LOCAL_DEPS.is_dir() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

from pyqcd.analysis import (
    calculate_gradient_flow_tmd_ratios_2d_from_configurations,
    calculate_gradient_flow_tmd_ratios_from_configurations,
    physical_helicity_ratio,
    plateau_c0,
    resample,
    run_3pt_ratio,
    sem,
)
from pyqcd.analysis._bare_sumratio import (
    fit_forward_sumratio_difference,
    physical_tmin,
)
from pyqcd.pipeline._config import (
    ALttc, EIGEN_DIR, FM2GEV, NT, PERAM_DIR, get_gauge_path,
)
from pyqcd.pipeline._tmd9 import (
    MOMENTA_Z,
    load_full_channel_tmd_ope_2d_all,
    load_full_channel_tmd_ope_all,
    load_multi_2pt_polarizations,
    load_tmd_ope_all,
    momentum_tag,
    self_renormalize,
)
from pyqcd.renorm import (
    a_len_set,
    cs_kernel_two_momentum,
    fm_to_GeV,
    flow_scale_mask,
    pz_to_gev,
    quasi_pdf_gluon,
    quasi_tmd_pdf,
)
from pyqcd.testing.tmd.run_full_production import (
    B_LIST,
    B_X_LIST,
    B_Y_LIST,
    CONF_IDS,
    FLOW_EPS,
    FLOW_TIMES,
    MOMENTA_AXES,
    PRECISION,
    STAPLE_LENGTH,
    Z_LIST,
)


DIRECTION_MOMENTUM = {
    0: (0, 0, 2),
    1: (0, 2, 0),
    2: (2, 0, 0),
}
DIRECTION_LABEL = {0: 'x', 1: 'y', 2: 'z'}
DIRECTION_B = {0: 1, 1: 2, 2: 0}
TSEPS = tuple(range(5, 21))
PLATEAU_WINDOWS = ((5, 10), (7, 12), (9, 14), (11, 16))
PRIMARY_PLATEAU = (7, 12)
CUT = 6
SUM_RATIO_THRESHOLD_FM = 0.6
SUM_RATIO_MAX_TSEP = 18
JACKKNIFE_SAMPLES = 1000
X_GRID = np.linspace(-0.9, 0.9, 181)
ENERGY_WINDOWS = {
    'P000': (5, 14),
    'P200': (5, 12),
    'P400': (4, 10),
    'P020': (5, 12),
    'P002': (5, 12),
}


def _validate_analysis_conf_ids(conf_ids, *, allow_partial=False):
    expected = tuple(int(value) for value in CONF_IDS)
    values = tuple(int(value) for value in conf_ids)
    if not values:
        raise ValueError('分析组态列表不能为空')
    if len(set(values)) != len(values):
        raise ValueError('分析组态列表不能包含重复值')
    unknown = sorted(set(values) - set(expected))
    if unknown:
        raise ValueError(f'分析组态不在 12 组态目标集合中: {unknown}')
    if not allow_partial and values != expected:
        raise ValueError(
            '正式分析必须按目标顺序覆盖 12 个组态；部分运行需显式 --allow-partial')
    return values


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


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent,
            prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(
            json.dumps(
                _json_safe(payload), indent=2, ensure_ascii=False) + '\n')
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f'.{path.name}.',
            suffix='.tmp.npz', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _finite_mean_error(samples, jackknife=True):
    values = np.asarray(samples)
    if values.ndim < 2:
        raise ValueError('samples 必须至少含 sample 与 feature 两个轴')
    flat = values.reshape(values.shape[0], -1)
    mean_dtype = np.result_type(values.dtype, np.float64)
    error_dtype = np.result_type(np.abs(values).dtype, np.float64)
    mean_flat = np.full(flat.shape[1], np.nan, dtype=mean_dtype)
    error_flat = np.full(flat.shape[1], np.nan, dtype=error_dtype)
    for index in range(flat.shape[1]):
        column = flat[:, index]
        finite = column[np.isfinite(column)]
        if not finite.size:
            continue
        mean_flat[index] = np.mean(finite)
        if finite.size >= 2:
            error_flat[index] = float(sem(finite, jackknife=jackknife))
    mean = mean_flat.reshape(values.shape[1:])
    error = error_flat.reshape(values.shape[1:])
    return mean, error


def safe_self_renormalize(c0):
    c0 = np.asarray(c0)
    if c0.ndim != 3:
        raise ValueError('c0 必须为 (sample,z,b)')
    hR = np.full(c0.shape, np.nan, dtype=np.result_type(c0.dtype, np.complex128))
    norm = np.full((c0.shape[0], 1, c0.shape[2]), np.nan,
                   dtype=np.result_type(c0.dtype, np.complex128))
    for b_index in range(c0.shape[2]):
        column = c0[:, :, b_index:b_index + 1]
        if not np.isfinite(column).all():
            continue
        try:
            normalized, denominator = self_renormalize(column)
        except (ValueError, FloatingPointError):
            continue
        hR[:, :, b_index:b_index + 1] = normalized
        norm[:, :, b_index:b_index + 1] = denominator
    return hR, norm


def make_lattice_only_status(*, conf_ids, tau_values, **extra):
    payload = {
        'schema': 'pyqcd.tmd_full_lattice_analysis.v1',
        'status': 'diagnostic_only',
        'physical_acceptance': False,
        'perturbative_matching_applied': False,
        'soft_rapidity_measured': False,
        'n_configurations': len(conf_ids),
        'conf_ids': list(conf_ids),
        'tau_values': list(tau_values),
        'reasons': [
            'soft_rapidity_not_measured',
            'perturbative_gluon_quark_mixing_not_applied',
            'perturbative_light_cone_matching_not_applied',
            'single_lattice_spacing',
            'single_ensemble',
            'no_continuum_extrapolation',
            'no_staple_length_extrapolation',
            'finite_fourier_support',
        ],
    }
    payload.update(extra)
    return payload


def build_two_point_ratio_input(source_dt, tseps, *, n_conf=None,
                                n_polarization=1):
    """Select C2(source,tsep) and add polarization/direction/momentum axes."""
    data = np.asarray(source_dt, dtype=np.complex128)
    if data.ndim == 2:
        data = data[None]
    if data.ndim != 3:
        raise ValueError('source_dt 必须为 (source,dt) 或 (conf,source,dt)')
    if n_conf is not None and data.shape[0] != int(n_conf):
        raise ValueError(f'配置数不符: {data.shape[0]} != {n_conf}')
    tseps = tuple(int(value) for value in tseps)
    if not tseps or len(set(tseps)) != len(tseps):
        raise ValueError('tseps 必须非空且不重复')
    if min(tseps) < 0 or max(tseps) >= data.shape[2]:
        raise ValueError('tseps 超出 dt 轴')
    if (isinstance(n_polarization, (bool, np.bool_))
            or int(n_polarization) <= 0):
        raise ValueError('n_polarization 必须为正整数')
    selected = np.transpose(data[:, :, list(tseps)], (0, 2, 1))
    selected = selected[None, :, :, :, None]
    selected = np.repeat(selected, int(n_polarization), axis=0)
    selected = selected[:, None, :, :, :]
    return selected


def _axis_index(labels, value, name):
    try:
        return tuple(labels).index(value)
    except ValueError as exc:
        raise ValueError(f'{name} 未知: {value!r}') from exc


def select_ratio_observable(ratio, samples, error, *, channel,
                            orientation, component, direction):
    ratio = np.asarray(ratio)
    samples = np.asarray(samples)
    error = np.asarray(error)
    if ratio.ndim not in (9, 10):
        raise ValueError('ratio 必须为 1D-v2 或 2D-v3 轴数')
    if samples.ndim != ratio.ndim + 1 or error.shape != ratio.shape:
        raise ValueError('ratio/samples/error 轴数与形状不符')
    channel_index = _axis_index(
        ('unpolarized', 'helicity'), channel, 'channel')
    orientation_index = _axis_index(
        ('plus_z_raw', 'minus_z_raw', 'even_sum', 'odd_difference'),
        orientation, 'orientation')
    component_index = _axis_index(
        ('combined', 'Mtiti', 'Mijij', 'ti', 'tj', 'ij_single'),
        component, 'component')
    direction = int(direction)
    if direction < 0 or direction >= ratio.shape[3]:
        raise ValueError('direction 超出范围')
    index = (channel_index, orientation_index, component_index, direction)
    central = ratio[index][..., 0]
    sample = np.moveaxis(samples[index][..., 0, :], -1, 0)
    uncertainty = error[index][..., 0]
    return {
        'ratio': central,
        'samples': sample,
        'error': uncertainty,
    }


def plateau_from_ratio_samples(samples, *, tseps=None, tsep_start=7,
                               tsep_end=12, cut=CUT):
    """Return plateau samples with axes ``(sample,z,b)``."""
    samples = np.asarray(samples)
    if samples.ndim != 5:
        raise ValueError('samples 必须为 (sample,z,b,tsep,insertion)')
    nsample, nz, nb, ntsep, ninsertion = samples.shape
    if tseps is None:
        tseps = tuple(range(ntsep))
    tseps = tuple(int(value) for value in tseps)
    if len(tseps) != ntsep or min(tseps) < 0:
        raise ValueError('tseps 与 tsep 轴不一致')
    max_tsep = max(tseps)
    if ninsertion <= max_tsep:
        raise ValueError('insertion 轴不足以覆盖最大 tsep')
    legacy = np.full(
        (nsample, max_tsep + 1, max_tsep + 1, nz, nb),
        np.nan + 1j * np.nan,
        dtype=np.result_type(samples.dtype, np.complex128),
    )
    for index, tsep in enumerate(tseps):
        block = samples[:, :, :, index, :tsep + 1]
        legacy[:, tsep, :tsep + 1] = np.transpose(block, (0, 3, 1, 2))
    return plateau_c0(
        legacy, dt_max=max_tsep + 1, dt_start=tsep_start,
        dt_end=tsep_end, cut=cut,
    )


def expand_ratio_axes(values, *, tseps=TSEPS, sample_axis=False):
    """Embed ``(..., z, b, tsep, insertion)`` into physical tsep axes."""
    values = np.asarray(values)
    expected_ndim = 5 if sample_axis else 4
    if values.ndim != expected_ndim:
        raise ValueError('ratio 数组轴数与 sample_axis 不一致')
    tseps = tuple(int(value) for value in tseps)
    max_tsep = max(tseps)
    if values.shape[-2] != len(tseps):
        raise ValueError('tsep 轴长度与 tseps 不一致')
    nz, nb = values.shape[-4], values.shape[-3]
    leading = values.shape[:-4]
    output = np.full(
        leading + (max_tsep + 1, max_tsep + 1, nz, nb),
        np.nan,
        dtype=np.result_type(values.dtype, np.float64),
    )
    for index, tsep in enumerate(tseps):
        block = values[..., index, :tsep + 1]
        block = np.moveaxis(block, -1, block.ndim - 3)
        output[..., tsep, :tsep + 1, :, :] = block
    return output


def expand_sumratio_inputs(selected, *, tseps=TSEPS):
    """Expand selected ratio arrays without disturbing sample/insertion axes."""
    samples = np.asarray(selected['samples'])
    original = np.asarray(selected['ratio'])
    return (
        expand_ratio_axes(samples, tseps=tseps, sample_axis=True),
        expand_ratio_axes(original, tseps=tseps, sample_axis=False),
    )


def _load_h5_data(path):
    import h5py
    with h5py.File(path, 'r') as handle:
        if set(handle.keys()) != {'data'}:
            raise ValueError(f'HDF5 顶层不是唯一 data: {path}')
        return handle['data'][...]


def load_generic_correlators(run_dir, conf_ids):
    run_dir = Path(run_dir)
    corr2 = {}
    corr3 = {}
    names2 = ('corr_pp_P0', 'corr_pp_P2', 'corr_pion_P0', 'corr_pion_P2')
    names3 = ('proton_P0_3pt', 'proton_P2_3pt',
              'pion_P0_3pt', 'pion_P2_3pt')
    for conf_id in conf_ids:
        cdir = run_dir / 'data' / f'conf{conf_id}'
        corr2[conf_id] = {}
        corr3[conf_id] = {}
        for name in names2:
            array = _load_h5_data(cdir / f'{name}_{conf_id}.h5')
            if array.shape != (NT,):
                raise ValueError(f'{name} conf={conf_id} shape={array.shape}')
            corr2[conf_id][name] = array
        for name in names3:
            array = _load_h5_data(cdir / f'{name}_{conf_id}.h5')
            if array.ndim != 2 or array.shape[1] != 4:
                raise ValueError(f'{name} conf={conf_id} shape={array.shape}')
            corr3[conf_id][name] = array
    return corr2, corr3


def write_raw_input_manifest(run_dir, conf_ids):
    run_dir = Path(run_dir)
    entries = {}
    for conf_id in conf_ids:
        eigen_dir = Path(EIGEN_DIR) / str(conf_id)
        eigen_files = sorted(eigen_dir.glob(f'eigvecs_t*_{conf_id}'))
        peram_dir = Path(PERAM_DIR) / str(conf_id)
        peram_files = sorted(peram_dir.glob(f'perams.{conf_id}.*.*'))
        gauge_path = Path(get_gauge_path(conf_id))
        if len(eigen_files) != 72:
            raise ValueError(f'conf={conf_id} eigenvector 数={len(eigen_files)}')
        if len(peram_files) != 288:
            raise ValueError(f'conf={conf_id} perambulator 数={len(peram_files)}')
        if not gauge_path.is_file() or gauge_path.stat().st_size <= 0:
            raise ValueError(f'conf={conf_id} gauge 不可用')
        if any(path.stat().st_size != 66_355_200 for path in eigen_files):
            raise ValueError(f'conf={conf_id} eigenvector 大小异常')
        if any(path.stat().st_size != 46_080_000 for path in peram_files):
            raise ValueError(f'conf={conf_id} perambulator 大小异常')
        entries[str(conf_id)] = {
            'gauge_path': str(gauge_path),
            'gauge_size_bytes': gauge_path.stat().st_size,
            'eigenvector_count': len(eigen_files),
            'eigenvector_size_bytes_each': 66_355_200,
            'perambulator_count': len(peram_files),
            'perambulator_size_bytes_each': 46_080_000,
        }
    payload = {
        'schema': 'pyqcd.tmd_full_raw_input_manifest.v1',
        'conf_ids': list(conf_ids),
        'n_configurations': len(conf_ids),
        'entries': entries,
    }
    path = run_dir / 'analysis' / 'raw_input_manifest.json'
    _write_json(path, payload)
    return payload


def load_flow_histories(run_dir, conf_ids):
    histories = {}
    required = {
        'flow_time', 'action_density', 't2_action_density',
        'tau_checkpoints', 'checkpoint_action',
    }
    for conf_id in conf_ids:
        path = Path(run_dir) / 'flow_history' / f'conf{conf_id}.npz'
        with np.load(path, allow_pickle=False) as handle:
            missing = required - set(handle.files)
            if missing:
                raise ValueError(f'flow history 缺字段: {sorted(missing)}')
            histories[conf_id] = {
                key: np.asarray(handle[key]) for key in required
            }
        if not all(
                np.isfinite(value).all()
                for value in histories[conf_id].values()):
            raise ValueError(f'flow history conf={conf_id} 含非有限值')
        if not np.allclose(
                histories[conf_id]['tau_checkpoints'],
                np.asarray(FLOW_TIMES), rtol=0.0, atol=1.0e-12):
            raise ValueError(f'flow history conf={conf_id} 流时网格不符')
    return histories


def analyze_flow_histories(run_dir, histories, logger=print):
    out_dir = Path(run_dir) / 'analysis' / 'flow'
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.0, 3.0, 301)
    actions = []
    t2_actions = []
    t0_values = []
    summary = {}
    for conf_id, item in histories.items():
        t = np.asarray(item['flow_time'], dtype=np.float64)
        e = np.asarray(item['action_density'], dtype=np.float64)
        t2e = np.asarray(item['t2_action_density'], dtype=np.float64)
        actions.append(np.interp(grid, t, e))
        t2_actions.append(np.interp(grid, t, t2e))
        crossing = np.flatnonzero(t2e >= 0.3)
        if crossing.size:
            index = int(crossing[0])
            if index == 0:
                t0 = float(t[index])
            else:
                t0 = float(np.interp(
                    0.3, t2e[index - 1:index + 1],
                    t[index - 1:index + 1]))
            t0_values.append(t0)
        else:
            t0 = np.nan
        summary[str(conf_id)] = {
            't0': t0,
            'final_E': float(e[-1]),
            'final_t2E': float(t2e[-1]),
            'n_points': int(len(t)),
        }
    actions = np.asarray(actions)
    t2_actions = np.asarray(t2_actions)
    _write_npz(
        out_dir / 'flow_history_ensemble.npz',
        flow_time=grid,
        action_samples=actions,
        action_mean=np.nanmean(actions, axis=0),
        action_error=sem(actions, jackknife=True),
        t2_action_samples=t2_actions,
        t2_action_mean=np.nanmean(t2_actions, axis=0),
        t2_action_error=sem(t2_actions, jackknife=True),
        t0_values=np.asarray(t0_values, dtype=np.float64),
    )
    _write_json(out_dir / 'flow_summary.json', summary)
    _plot_flow_history(grid, actions, t2_actions, out_dir / 'flow_history.png')
    logger(
        f"  flow: t0={np.nanmedian(t0_values):.4f} "
        f"(N={len(t0_values)})")
    return summary


def _plot_flow_history(grid, actions, t2_actions, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for sample in actions:
        axes[0].plot(grid, sample, color='#457B9D', alpha=0.18, lw=0.7)
    axes[0].plot(grid, np.nanmean(actions, axis=0), color='#1D3557', lw=2.0)
    axes[0].set_yscale('log')
    axes[0].set_xlabel(r'$\tau=t/a^2$')
    axes[0].set_ylabel(r'$\langle E(\tau)\rangle$')
    axes[0].set_title('Wilson-flow 作用量密度')
    for sample in t2_actions:
        axes[1].plot(grid, sample, color='#E76F51', alpha=0.18, lw=0.7)
    axes[1].plot(grid, np.nanmean(t2_actions, axis=0), color='#9C2C13', lw=2.0)
    axes[1].axhline(0.3, color='k', ls='--', lw=0.9, label=r'$t_0$ target 0.3')
    axes[1].set_xlabel(r'$\tau=t/a^2$')
    axes[1].set_ylabel(r'$\tau^2\langle E\rangle$')
    axes[1].set_title('尺度设定诊断')
    axes[1].legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _source_mean(corr):
    value = np.asarray(corr)
    if value.ndim != 2:
        raise ValueError('source-resolved C2 必须为 (source,dt)')
    return np.mean(value, axis=0)


def _effective_mass_samples(samples):
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] < 3:
        raise ValueError('effective-mass samples 必须为 (sample,time>=3)')
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = (samples[:, 2:] + samples[:, :-2]) / (
            2.0 * samples[:, 1:-1])
        mass = np.full((samples.shape[0], samples.shape[1] - 2), np.nan)
        valid = ratio >= 1.0
        mass[valid] = np.arccosh(ratio[valid]) * (FM2GEV / ALttc)
    return mass


def _effective_mass_times(n_time):
    """Return the physical times represented by ``_effective_mass_samples``."""
    if int(n_time) < 3:
        raise ValueError('effective-mass time axis 至少需要 3 个点')
    return np.arange(1, int(n_time) - 1, dtype=np.int64)


def _plateau_mean_samples(samples, start, end):
    if end <= start or start < 0 or end > samples.shape[1]:
        raise ValueError('平台窗口无效')
    return np.nanmean(samples[:, start:end], axis=1)


def analyze_spectrum(run_dir, conf_ids, polarized_2pt, corr2, corr3,
                     logger=print):
    out_dir = Path(run_dir) / 'analysis' / 'spectrum'
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = [momentum_tag(momentum) for momentum in MOMENTA_AXES]
    summary = {}
    for polarization in ('nopol', 'pol35'):
        for tag in tags:
            stack = np.stack([
                _source_mean(polarized_2pt[cid][polarization][
                    f'corr_pp_{tag}'])
                for cid in conf_ids
            ])
            magnitude = np.abs(stack)
            jack = resample(magnitude, jackknife=True, axis=0)
            mass_samples = _effective_mass_samples(jack)
            mass_mean, mass_error = _finite_mean_error(mass_samples)
            start, end = ENERGY_WINDOWS.get(tag, (5, 12))
            mass_times = _effective_mass_times(magnitude.shape[1])
            window_indices = np.flatnonzero(
                (mass_times >= start) & (mass_times < end))
            if not window_indices.size:
                raise ValueError(f'effective-mass 平台窗为空: {tag} [{start}, {end})')
            plateau_samples = _plateau_mean_samples(
                mass_samples, int(window_indices[0]), int(window_indices[-1] + 1))
            energy = float(np.nanmean(plateau_samples))
            energy_error = float(sem(plateau_samples, jackknife=True))
            summary[f'{polarization}_{tag}'] = {
                'momentum_tag': tag,
                'polarization': polarization,
                'source_mean_shape': list(stack.shape),
                'effective_mass_GeV': mass_mean,
                'effective_mass_error_GeV': mass_error,
                'plateau_window': [start, end - 1],
                'plateau_times': mass_times[window_indices].tolist(),
                'energy_GeV': energy,
                'energy_error_GeV': energy_error,
                'energy_status': (
                    'diagnostic' if np.isfinite(energy) else 'unavailable'),
            }
            _write_npz(
                out_dir / f'effective_mass_{polarization}_{tag}.npz',
                source_mean=stack,
                jackknife=magnitude,
                effective_mass_samples=mass_samples,
                effective_mass_mean=mass_mean,
                effective_mass_error=mass_error,
                plateau_samples=plateau_samples,
            )
            _plot_effective_mass(
                stack, mass_mean, mass_error, mass_times, start, end,
                out_dir / f'effective_mass_{polarization}_{tag}.png',
                f'{polarization} {tag}',
            )
            logger(
                f"  spectrum {polarization} {tag}: "
                f"E={energy:.4f}±{energy_error:.4f} GeV")

    connected = run_3pt_ratio(
        corr2, corr3, conf_ids, t_sep=None, NT=NT, logger=logger)
    connected_payload = {}
    for key, item in connected.items():
        connected_payload[key] = {
            'R': item['R'],
            'R_err': item['R_err'],
            't_sep': item['t_sep'],
        }
        _plot_connected_ratio(
            item, out_dir / f'ratio_3pt_{key}.png', key)
    _write_json(out_dir / 'connected_3pt_ratio.json', connected_payload)
    _plot_dispersion(summary, out_dir / 'dispersion_diagnostic.png')
    _plot_spectrum_comparison(
        summary, out_dir / 'direction_momentum_comparison.png')
    _write_json(out_dir / 'spectrum_summary.json', summary)
    return summary


def _plot_effective_mass(source, mean, error, times, start, end, path, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    source = np.asarray(source)
    if source.ndim != 2:
        raise ValueError('effective-mass source 必须为 (configuration,time)')
    if np.shape(mean) != np.shape(times) or np.shape(error) != np.shape(times):
        raise ValueError('effective-mass mean/error 必须与 times 同形')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].errorbar(
        np.arange(source.shape[1]), np.abs(source.mean(axis=0)),
        yerr=sem(np.abs(source), jackknife=True), marker='o', ms=2.5,
        lw=1.0, capsize=1.5, color='#174A6E')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('t/a')
    axes[0].set_ylabel(r'$|C_2(t)|$')
    axes[0].set_title('两点关联函数')
    axes[1].errorbar(times, mean, yerr=error, marker='o', ms=2.5, lw=1.0,
                     capsize=1.5, color='#B24C32')
    window = (times >= start) & (times < end)
    if np.any(window):
        band = np.nanmean(mean[window])
        axes[1].axvspan(start - 0.5, end - 0.5, color='#E9C46A', alpha=0.2)
        axes[1].axhline(band, color='#264653', ls='--', lw=1.0)
    axes[1].set_xlabel('t/a')
    axes[1].set_ylabel(r'$m_{\rm eff}$ [GeV]')
    axes[1].set_title('有效质量与平台')
    fig.suptitle(label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_connected_ratio(item, path, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    values = np.asarray(item['R'])
    errors = np.asarray(item['R_err'])
    if values.ndim == 1:
        values = values[:, None]
        errors = errors[:, None]
    if values.ndim != 2 or errors.shape != values.shape or values.shape[1] > 4:
        raise ValueError('connected-3pt R/R_err 必须同形且最多包含四个分量')
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.0), constrained_layout=True)
    for component, axis in enumerate(axes.ravel()):
        if component >= values.shape[1]:
            axis.axis('off')
            continue
        x = np.arange(values.shape[0])
        axis.errorbar(x, values[:, component], yerr=errors[:, component],
                      marker='o', ms=2.5, lw=1.0, capsize=1.5,
                      color=['#264653', '#2A9D8F', '#E76F51', '#E9C46A'][component])
        axis.axhline(0.0, color='0.5', lw=0.8)
        axis.set_title(f'$\\gamma_{component + 1}$')
        axis.set_xlabel(r'$\tau/a$')
        axis.set_ylabel('R')
    fig.suptitle(f'连通 3pt/2pt ratio: {label}')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_dispersion(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    points = []
    for tag in ('P000', 'P200', 'P400', 'P020', 'P002'):
        item = summary.get(f'nopol_{tag}')
        if item is None or not np.isfinite(item['energy_GeV']):
            continue
        pz, py, px = next(
            momentum for momentum in MOMENTA_AXES
            if momentum_tag(momentum) == tag)
        p2 = float(pz * pz + py * py + px * px)
        points.append((p2, item['energy_GeV'], item['energy_error_GeV']))
    fig, axis = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    if points:
        points = np.asarray(points)
        axis.errorbar(points[:, 0], points[:, 1], yerr=points[:, 2],
                      marker='o', lw=1.2, capsize=2.0, color='#1D3557')
    axis.set_xlabel(r'$p^2/(2\pi/L)^2$')
    axis.set_ylabel(r'$E_{\rm plateau}$ [GeV]')
    axis.set_title('核子能量色散诊断')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_spectrum_comparison(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tags = ('P000', 'P200', 'P400', 'P020', 'P002')
    fig, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    for polarization, color in (('nopol', '#264653'), ('pol35', '#E76F51')):
        x, values, errors = [], [], []
        for index, tag in enumerate(tags):
            item = summary.get(f'{polarization}_{tag}')
            if item is None or not np.isfinite(item.get('energy_GeV', np.nan)):
                continue
            x.append(index)
            values.append(item['energy_GeV'])
            errors.append(item['energy_error_GeV'])
        if x:
            axis.errorbar(
                x, values, yerr=errors, marker='o', capsize=2.0,
                lw=1.2, color=color, label=polarization)
    axis.set_xticks(np.arange(len(tags)), tags)
    axis.set_xlabel('momentum / direction')
    axis.set_ylabel(r'$E_{\rm plateau}$ [GeV]')
    axis.set_title('核子谱学：方向与动量比较')
    axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _build_two_point_for_direction(polarized_2pt, conf_ids, direction,
                                   momentum=None):
    if momentum is None:
        momentum = DIRECTION_MOMENTUM[int(direction)]
    else:
        momentum = tuple(int(value) for value in momentum)
    tag = momentum_tag(momentum)
    nopol = np.stack([
        polarized_2pt[conf_id]['nopol'][f'corr_pp_{tag}']
        for conf_id in conf_ids
    ])
    pol35 = np.stack([
        polarized_2pt[conf_id]['pol35'][f'corr_pp_{tag}']
        for conf_id in conf_ids
    ])
    combined = np.concatenate((
        build_two_point_ratio_input(
            nopol, TSEPS, n_conf=len(conf_ids), n_polarization=1),
        build_two_point_ratio_input(
            pol35, TSEPS, n_conf=len(conf_ids), n_polarization=1),
    ), axis=0)
    return {
        int(conf_id): combined[:, :, index]
        for index, conf_id in enumerate(conf_ids)
    }


def _save_ratio_result(path, result):
    arrays = {}
    metadata = {}
    for key, value in result.items():
        if key in (
                'ratio_jackknife_samples',
                'ratio_jackknife_mean',
                'ratio_jackknife_bias_corrected'):
            continue
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif key in ('b_values', 'b_x_values', 'b_y_values', 'conf_ids',
                     'channel_labels', 'component_labels',
                     'z_orientation_labels', 'axes'):
            arrays[key] = np.asarray(value)
            metadata[key] = value
        elif isinstance(value, (str, int, float, bool, type(None), tuple,
                                list, dict, np.generic)):
            metadata[key] = value
    _write_npz(path, **arrays)
    _write_json(Path(path).with_suffix('.json'), metadata)
    return Path(path)


def analyze_ratio_1d(run_dir, conf_ids, polarized_2pt, tau_values,
                     logger=print):
    out_root = Path(run_dir) / 'analysis' / 'ratio_1d'
    outputs = {}
    for tau in tau_values:
        tau_tag = f'tau{tau:g}'
        for direction in (0, 1, 2):
            z_dir = int(direction)
            b_dir = DIRECTION_B[z_dir]
            momenta = [DIRECTION_MOMENTUM[z_dir]]
            if z_dir == 2:
                momenta.append((4, 0, 0))
            logger(
                f"  ratio 1D tau={tau:g} "
                f"direction={DIRECTION_LABEL[z_dir]}")
            ope = load_full_channel_tmd_ope_all(
                run_dir, conf_ids, Z_LIST, B_LIST, logger=logger,
                staple_length=STAPLE_LENGTH, tau=tau, eps=FLOW_EPS,
                precision=PRECISION, z_dir=z_dir, b_dir=b_dir,
            )
            missing = set(conf_ids) - set(ope)
            if missing:
                raise ValueError(f'缺少 1D OPE tau={tau} dir={z_dir}: {missing}')
            ope_by_conf = {
                conf_id: ope[conf_id]['tmd_full'] for conf_id in conf_ids
            }
            for momentum in momenta:
                tag = momentum_tag(momentum)
                twopt = _build_two_point_for_direction(
                    polarized_2pt, conf_ids, z_dir, momentum=momentum)
                result = calculate_gradient_flow_tmd_ratios_from_configurations(
                    ope_by_conf, twopt, conf_ids, B_LIST, TSEPS,
                    do_jackknife=True, store_jackknife_samples=True,
                    denominator_mode='complex',
                )
                stem = (
                    f'ratio_v2_{tau_tag}_{DIRECTION_LABEL[z_dir]}_{tag}')
                path = _save_ratio_result(
                    out_root / f'{stem}.npz', result)
                selected = select_ratio_observable(
                    result['ratio'], result['ratio_jackknife_samples'],
                    result['ratio_jackknife_error_real'],
                    channel='unpolarized', orientation='even_sum',
                    component='combined', direction=0,
                )
                selected['ratio'] = np.real(selected['ratio'])
                selected['samples'] = np.real(selected['samples'])
                selected['helicity_imag'] = physical_helicity_ratio(
                    result['ratio'])
                selected['helicity_imag_samples'] = np.moveaxis(
                    physical_helicity_ratio(
                        result['ratio_jackknife_samples']), -1, 0)
                _write_npz(
                    out_root / f'{stem}_selected.npz',
                    unpolarized_ratio=selected['ratio'],
                    unpolarized_samples=selected['samples'],
                    unpolarized_error=selected['error'],
                helicity_imag=selected['helicity_imag'],
                helicity_imag_samples=selected['helicity_imag_samples'],
                sample_axis=np.asarray('sample,z,b,tsep,insertion'),
                )
                _plot_ratio_slices(
                    selected['ratio'], out_root / f'{stem}_ratio.png',
                    f'{tau_tag} {DIRECTION_LABEL[z_dir]} direction {tag}')
                outputs[(float(tau), z_dir, tag)] = {
                    'path': str(path),
                    'selected_path': str(
                        out_root / f'{stem}_selected.npz'),
                    'result': selected,
                }
    return outputs


def _plot_ratio_slices(ratio, path, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    values = np.asarray(ratio)
    z_index = min(5, values.shape[0] - 1)
    b_index = min(5, values.shape[1] - 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for tsep_index, tsep in enumerate(TSEPS):
        if tsep > values.shape[2] - 1 + min(TSEPS):
            continue
        row = values[z_index, b_index, tsep_index]
        finite = np.isfinite(row.real)
        axes[0].plot(np.arange(row.size)[finite], row.real[finite],
                     marker='o', ms=2.0, lw=0.8, label=f'tsep={tsep}')
    axes[0].set_xlabel('insertion/t')
    axes[0].set_ylabel(r'Re $R$')
    axes[0].set_title(f'z={Z_LIST[z_index]}a, b={B_LIST[b_index]}a')
    axes[0].legend(ncol=2, fontsize=7)
    image = axes[1].imshow(
        np.real(np.nanmean(
            values[:, :, TSEPS.index(11), :12], axis=-1)).T,
        origin='lower', aspect='auto', cmap='RdBu_r')
    axes[1].set_xlabel('z/a')
    axes[1].set_ylabel('b/a')
    axes[1].set_title('tsep=11a 平均 Re R')
    fig.colorbar(image, ax=axes[1], shrink=0.8)
    fig.suptitle(label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_ratio_2d(run_dir, conf_ids, polarized_2pt, tau_values,
                     logger=print):
    out_root = Path(run_dir) / 'analysis' / 'ratio_2d'
    outputs = {}
    twopt = _build_two_point_for_direction(polarized_2pt, conf_ids, 2)
    for tau in tau_values:
        tau_tag = f'tau{tau:g}'
        for order in ((0, 1), (1, 0)):
            logger(f"  ratio 2D tau={tau:g} order={order}")
            ope = load_full_channel_tmd_ope_2d_all(
                run_dir, conf_ids, Z_LIST, B_X_LIST, B_Y_LIST,
                logger=logger, staple_length=STAPLE_LENGTH,
                tau=tau, eps=FLOW_EPS, precision=PRECISION,
                z_dir=2, b_x_dir=0, b_y_dir=1,
                transverse_order=order,
            )
            if set(ope) != set(conf_ids):
                raise ValueError(f'缺少 2D OPE tau={tau} order={order}')
            ope_by_conf = {
                conf_id: ope[conf_id]['tmd_full_2d']
                for conf_id in conf_ids
            }
            result = calculate_gradient_flow_tmd_ratios_2d_from_configurations(
                ope_by_conf, twopt, conf_ids, B_X_LIST, B_Y_LIST, TSEPS,
                z_dir=2, b_x_dir=0, b_y_dir=1,
                transverse_order=order, do_jackknife=True,
                store_jackknife_samples=True,
                denominator_mode='complex',
            )
            order_tag = ''.join(map(str, order))
            stem = f'ratio_v3_{tau_tag}_order{order_tag}'
            path = _save_ratio_result(out_root / f'{stem}.npz', result)
            selected = select_ratio_observable(
                result['ratio'], result['ratio_jackknife_samples'],
                result['ratio_jackknife_error_real'],
                channel='unpolarized', orientation='even_sum',
                component='combined', direction=0,
            )
            selected['ratio'] = np.real(selected['ratio'])
            selected['samples'] = np.real(selected['samples'])
            helicity_imag = physical_helicity_ratio(result['ratio'])
            helicity_imag_samples = np.moveaxis(
                physical_helicity_ratio(result['ratio_jackknife_samples']),
                -1, 0)
            outputs[(float(tau), order)] = {
                'path': str(path), 'selected': selected,
                'helicity_imag': helicity_imag,
                'helicity_imag_samples': helicity_imag_samples,
                'result': result,
            }
            _write_npz(
                out_root / f'{stem}_selected.npz',
                unpolarized_ratio=selected['ratio'],
                unpolarized_samples=selected['samples'],
                unpolarized_error=selected['error'],
                helicity_imag=helicity_imag,
                helicity_imag_samples=helicity_imag_samples,
                sample_axis=np.asarray(
                    'sample,z,b_x,b_y,tsep,insertion'),
            )
            _plot_ratio_2d_slices(
                selected['ratio'], out_root / f'{stem}_ratio.png',
                f'{tau_tag} order={order}')
    _plot_staircase_comparison(outputs, out_root / 'staircase_sensitivity.png')
    _write_json(
        out_root / 'staircase_summary.json',
        _staircase_summary(outputs),
    )
    return outputs


def _plot_ratio_2d_slices(ratio, path, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    values = np.asarray(ratio)
    z_index = min(5, values.shape[0] - 1)
    tsep_index = TSEPS.index(11)
    image = np.real(np.nanmean(
        values[z_index, :, :, tsep_index, :12], axis=-1))
    fig, axis = plt.subplots(figsize=(6.0, 4.6), constrained_layout=True)
    plot = axis.imshow(image.T, origin='lower', aspect='auto', cmap='RdBu_r')
    axis.set_xlabel(r'$b_x/a$')
    axis.set_ylabel(r'$b_y/a$')
    axis.set_title(f'{label}, z={Z_LIST[z_index]}a, tsep=11a')
    fig.colorbar(plot, ax=axis, shrink=0.8, label=r'Re $R$')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _staircase_summary(outputs):
    summary = {}
    for tau in sorted({key[0] for key in outputs}):
        a = outputs[(tau, (0, 1))]['selected']['ratio']
        b = outputs[(tau, (1, 0))]['selected']['ratio']
        finite = np.isfinite(a.real) & np.isfinite(b.real)
        if not np.any(finite):
            summary[f'tau{tau:g}'] = {
                'status': 'unavailable', 'finite_points': 0}
            continue
        delta = np.abs(a[finite] - b[finite])
        ref = np.linalg.norm(a[finite]) + 1.0e-30
        summary[f'tau{tau:g}'] = {
            'status': 'diagnostic',
            'finite_points': int(np.count_nonzero(finite)),
            'relative_l2': float(np.linalg.norm(delta) / ref),
            'median_abs_difference': float(np.median(delta)),
            'max_abs_difference': float(np.max(delta)),
        }
    return summary


def _plot_staircase_comparison(outputs, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    for tau in sorted({key[0] for key in outputs}):
        a = outputs[(tau, (0, 1))]['selected']['ratio']
        b = outputs[(tau, (1, 0))]['selected']['ratio']
        finite = np.isfinite(a.real) & np.isfinite(b.real)
        if np.any(finite):
            axis.scatter(a.real[finite], b.real[finite], s=9, alpha=0.6,
                         label=f'tau={tau:g}')
    axis.set_xlabel(r'Re $R$ order (0,1)')
    axis.set_ylabel(r'Re $R$ order (1,0)')
    axis.set_title('二维 staircase 离散化敏感性')
    axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _c0_windows(selected, tseps=TSEPS):
    windows = {}
    for start, end in PLATEAU_WINDOWS:
        windows[f'{start}_{end}'] = plateau_from_ratio_samples(
            selected['samples'], tseps=tseps,
            tsep_start=start, tsep_end=end, cut=CUT)
    return windows


def analyze_plateaus(run_dir, ratio_1d, logger=print):
    out_dir = Path(run_dir) / 'analysis' / 'plateaus'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    a_fm = float(a_len_set['L24x72']) * fm_to_GeV
    for key, payload in ratio_1d.items():
        tau, direction, momentum_tag_value = key
        selected = payload['result']
        windows = _c0_windows(selected)
        arrays = {}
        window_summary = {}
        for name, samples in windows.items():
            mean, error = _finite_mean_error(samples, jackknife=True)
            arrays[f'c0_{name}'] = samples
            arrays[f'c0_mean_{name}'] = mean
            arrays[f'c0_error_{name}'] = error
            window_summary[name] = {
                'window': [int(value) for value in name.split('_')],
                'sample_finite_fraction': float(np.mean(np.isfinite(samples))),
                'identifiable_fraction': float(np.mean(np.isfinite(mean))),
                'unavailable_cells': int(np.count_nonzero(~np.isfinite(mean))),
                'median_error': float(np.nanmedian(error)),
                'status': (
                    'identifiable' if np.isfinite(mean).any()
                    else 'statistically_unidentifiable'),
            }
        stem = f'tau{tau:g}_{DIRECTION_LABEL[direction]}_{momentum_tag_value}'
        primary = windows[f'{PRIMARY_PLATEAU[0]}_{PRIMARY_PLATEAU[1]}']
        _write_npz(out_dir / f'c0_{stem}.npz', **arrays)
        _plot_c0_windows(
            windows, out_dir / f'c0_{stem}_windows.png', stem)
        primary_mean, primary_error = _finite_mean_error(
            primary, jackknife=True)
        _plot_c0_heatmap(
            primary_mean, out_dir / f'c0_{stem}_heatmap.png', stem)
        sumratio_samples, sumratio_original = expand_sumratio_inputs(selected)
        sumratio = fit_forward_sumratio_difference(
            sumratio_samples,
            sumratio_original,
            tmin=physical_tmin(a_fm, SUM_RATIO_THRESHOLD_FM),
            cuts=(1, 2, 3), tsep_max=SUM_RATIO_MAX_TSEP,
        )
        _write_npz(out_dir / f'sumratio_{stem}.npz', **sumratio)
        summary[stem] = {
            'tau': tau,
            'direction': DIRECTION_LABEL[direction],
            'momentum': momentum_tag_value,
            'primary_window': list(PRIMARY_PLATEAU),
            'primary_c0_samples_shape': list(primary.shape),
            'primary_identifiable_fraction': window_summary[
                f'{PRIMARY_PLATEAU[0]}_{PRIMARY_PLATEAU[1]}'][
                    'identifiable_fraction'],
            'primary_status': window_summary[
                f'{PRIMARY_PLATEAU[0]}_{PRIMARY_PLATEAU[1]}']['status'],
            'windows': window_summary,
            'sumratio_quality_fraction': float(np.mean(
                np.asarray(sumratio['quality_accepted'], dtype=bool))),
        }
        logger(
            f"  plateau tau={tau:g} dir={DIRECTION_LABEL[direction]} "
            f"mom={momentum_tag_value}")
    _write_json(out_dir / 'plateau_summary.json', summary)
    _plot_sumratio_quality(summary, out_dir / 'sumratio_quality.png')
    return summary


def _plot_sumratio_quality(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    keys = sorted(summary)
    values = np.asarray([
        float(summary[key].get('sumratio_quality_fraction', np.nan))
        for key in keys
    ])
    finite = np.isfinite(values)
    fig, axis = plt.subplots(
        figsize=(8.0, max(3.0, 0.28 * len(keys))), constrained_layout=True)
    if finite.any():
        y = np.arange(len(keys))[finite]
        axis.barh(y, values[finite], color='#457B9D')
        axis.set_yticks(y, [keys[index] for index in y])
        axis.set_xlim(0.0, 1.0)
    else:
        axis.text(0.5, 0.5, 'Sumratio quality unavailable',
                  ha='center', va='center', transform=axis.transAxes)
        axis.set_yticks([])
    axis.set_xlabel('quality-accepted fraction')
    axis.set_title('Sumratio AIC 质量与 fallback 覆盖')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_plateaus_2d(run_dir, ratio_2d, logger=print):
    out_dir = Path(run_dir) / 'analysis' / 'plateaus_2d'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for (tau, order), payload in ratio_2d.items():
        selected = payload['selected']
        samples = np.asarray(selected['samples'])
        nsample, nz, nbx, nby, ntsep, ninsertion = samples.shape
        flattened = samples.reshape(
            nsample, nz, nbx * nby, ntsep, ninsertion)
        c0 = plateau_from_ratio_samples(
            flattened, tseps=TSEPS, tsep_start=PRIMARY_PLATEAU[0],
            tsep_end=PRIMARY_PLATEAU[1], cut=CUT)
        c0 = c0.reshape(nsample, nz, nbx, nby)
        mean, error = _finite_mean_error(c0, jackknife=True)
        order_tag = ''.join(map(str, order))
        stem = f'tau{tau:g}_order{order_tag}'
        _write_npz(
            out_dir / f'c0_{stem}.npz',
            c0_samples=c0,
            c0_mean=mean,
            c0_error=error,
            z_list=np.asarray(Z_LIST),
            b_x_list=np.asarray(B_X_LIST),
            b_y_list=np.asarray(B_Y_LIST),
        )
        _plot_c0_2d_heatmap(mean, out_dir / f'c0_{stem}_heatmap.png', stem)
        summary[stem] = {
            'tau': float(tau),
            'order': list(order),
            'shape': list(c0.shape),
            'finite_fraction': float(np.mean(np.isfinite(c0))),
            'median_error': float(np.nanmedian(error)),
        }
        logger(f"  2D plateau tau={tau:g} order={order}")
    _write_json(out_dir / 'plateau_2d_summary.json', summary)
    return summary


def _plot_c0_windows(windows, path, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(windows), figsize=(3.2 * len(windows), 3.4),
                             constrained_layout=True, squeeze=False)
    for axis, (name, samples) in zip(axes[0], windows.items()):
        mean, error = _finite_mean_error(samples, jackknife=True)
        b_index = min(5, mean.shape[1] - 1)
        axis.errorbar(Z_LIST, mean[:, b_index], yerr=error[:, b_index],
                      marker='o', ms=2.5, lw=1.0, capsize=1.5,
                      color='#2A6F97')
        axis.axhline(0.0, color='0.5', lw=0.8)
        axis.set_title(f'tsep {name.replace("_", "..")}')
        axis.set_xlabel('z/a')
        axis.set_ylabel(f'c0, b={B_LIST[b_index]}a')
    fig.suptitle(label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_c0_heatmap(mean, path, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    image = axis.imshow(
        np.real(mean).T, origin='lower', aspect='auto', cmap='RdBu_r')
    axis.set_xlabel('z/a')
    axis.set_ylabel('b/a')
    axis.set_title(f'c0 heatmap: {label}')
    fig.colorbar(image, ax=axis, shrink=0.8, label='c0')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_c0_2d_heatmap(mean, path, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    z_index = min(5, mean.shape[0] - 1)
    fig, axis = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)
    image = axis.imshow(
        np.real(mean[z_index]).T, origin='lower', aspect='auto',
        cmap='RdBu_r')
    axis.set_xlabel(r'$b_x/a$')
    axis.set_ylabel(r'$b_y/a$')
    axis.set_title(f'2D c0, z={Z_LIST[z_index]}a, {label}')
    fig.colorbar(image, ax=axis, shrink=0.8, label='c0')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_flow_windows(run_dir, ratio_1d, tau_values, logger=print):
    out_dir = Path(run_dir) / 'analysis' / 'flow_windows'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {'tau': {}}
    samples_by_tau = {}
    for tau in tau_values:
        selected = ratio_1d[(float(tau), 2, 'P200')]['result']
        z_mask = flow_scale_mask(
            np.asarray(Z_LIST, dtype=float), float(tau), include_local=False)
        b_mask = flow_scale_mask(
            np.asarray(B_LIST, dtype=float), float(tau), include_local=False)
        samples_by_tau[float(tau)] = selected['samples']
        summary['tau'][f'{tau:g}'] = {
            'flow_radius_over_a': float(np.sqrt(8.0 * tau)),
            'z_mask': z_mask.tolist(),
            'b_mask': b_mask.tolist(),
            'z_indices': np.flatnonzero(z_mask).tolist(),
            'b_indices': np.flatnonzero(b_mask).tolist(),
            'n_z': int(np.count_nonzero(z_mask)),
            'n_b': int(np.count_nonzero(b_mask)),
        }
    common_z = np.ones(len(Z_LIST), dtype=bool)
    common_b = np.ones(len(B_LIST), dtype=bool)
    for tau in tau_values:
        common_z &= np.asarray(summary['tau'][f'{tau:g}']['z_mask'], bool)
        common_b &= np.asarray(summary['tau'][f'{tau:g}']['b_mask'], bool)
    summary['common_window'] = {
        'z_indices': np.flatnonzero(common_z).tolist(),
        'b_indices': np.flatnonzero(common_b).tolist(),
        'n_points': int(np.count_nonzero(common_z) * np.count_nonzero(common_b)),
    }
    paired = {}
    if np.any(common_z) and np.any(common_b):
        reference_tau = min(tau_values)
        ref = samples_by_tau[float(reference_tau)]
        for tau in tau_values:
            if float(tau) == float(reference_tau):
                continue
            candidate = samples_by_tau[float(tau)]
            reference = ref
            candidate_win = candidate[:, common_z][:, :, common_b]
            reference_win = reference[:, common_z][:, :, common_b]
            diff = candidate_win - reference_win
            finite = np.isfinite(diff)
            if not np.any(finite):
                summary[f'paired_tau{tau:g}_minus_tau{reference_tau:g}'] = {
                    'status': 'unavailable'}
                continue
            error_real = sem(diff.real, jackknife=True)
            mean_real = np.nanmean(diff.real, axis=0)
            effect = np.abs(mean_real) / (
                np.abs(np.nanmean(reference_win.real, axis=0))
                + 1.0e-30)
            significant = np.abs(mean_real) / (error_real + 1.0e-30)
            paired_key = f'paired_tau{tau:g}_minus_tau{reference_tau:g}'
            summary[paired_key] = {
                'status': 'diagnostic',
                'median_paired_sigma': float(np.nanmedian(significant)),
                'median_relative_difference': float(np.nanmedian(effect)),
                'finite_fraction': float(np.mean(finite)),
            }
            paired[paired_key] = (mean_real, error_real)
    _write_json(out_dir / 'flow_window_summary.json', summary)
    _plot_flow_coverage(summary, out_dir / 'flow_window_coverage.png')
    _plot_paired_flow(paired, out_dir / 'flow_time_paired_difference.png')
    return summary


def _plot_flow_coverage(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    z = np.asarray(Z_LIST)
    fig, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    for index, (tau_text, item) in enumerate(summary['tau'].items()):
        axis.plot(z, np.full_like(z, index, dtype=float), '|', ms=15,
                  label=f'tau={tau_text}')
        mask = np.asarray(item['z_mask'], bool)
        axis.scatter(z[mask], np.full(np.count_nonzero(mask), index),
                     s=55, color='#E76F51')
    axis.set_yticks(range(len(summary['tau'])))
    axis.set_yticklabels([f"tau={key}" for key in summary['tau']])
    axis.set_xlabel('z/a')
    axis.set_title('Wilson-flow 有效 z 窗')
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_paired_flow(paired, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    for label, (mean, error) in paired.items():
        flat = np.arange(mean.size)
        axis.errorbar(flat, mean.ravel(), yerr=error.ravel(),
                      lw=0.8, capsize=1.0, label=label)
    axis.axhline(0.0, color='0.5', lw=0.8)
    axis.set_xlabel('flattened common-window cell')
    axis.set_ylabel(r'paired $\Delta R$')
    axis.set_title('跨流时配对 JK 差值')
    if paired:
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze_pdf(run_dir, conf_ids, ratio_1d, tau_values, logger=print):
    out_dir = Path(run_dir) / 'analysis' / 'pdf'
    out_dir.mkdir(parents=True, exist_ok=True)
    z_fm = np.asarray(Z_LIST, dtype=float) * ALttc
    b_fm = np.asarray(B_LIST, dtype=float) * ALttc
    summary = {}
    hR_by_momentum = {}
    for tau in tau_values:
        hR_by_momentum[float(tau)] = {}
        for pz_lattice in (2, 4):
            tag = f'P{pz_lattice}00'
            selected = ratio_1d[(float(tau), 2, tag)]['result']
            c0 = plateau_from_ratio_samples(
                selected['samples'], tseps=TSEPS,
                tsep_start=PRIMARY_PLATEAU[0],
                tsep_end=PRIMARY_PLATEAU[1], cut=CUT,
            )
            hR, normalization_norm = safe_self_renormalize(c0)
            pz_gev = pz_to_gev(pz_lattice, 'L24x72')
            quasi_samples = np.full(
                (hR.shape[0], len(X_GRID), len(B_LIST)), np.nan)
            quasi_pdf_samples = np.full(
                (hR.shape[0], len(X_GRID)), np.nan)
            for b_index in range(len(B_LIST)):
                for sample_index, sample in enumerate(hR[:, :, b_index]):
                    if not np.isfinite(sample).all():
                        continue
                    try:
                        _, qtmd = quasi_tmd_pdf(
                            sample[:, None], z_fm, b_fm[b_index:b_index + 1],
                            pz_gev, x_grid=X_GRID)
                    except (ValueError, FloatingPointError):
                        continue
                    qtmd = np.asarray(qtmd)
                    quasi_samples[sample_index, :, b_index] = (
                        qtmd[:, 0] if qtmd.ndim == 2 else qtmd)
            for sample_index, sample in enumerate(hR[:, :, 0]):
                if not np.isfinite(sample).all():
                    continue
                try:
                    _, qpdf = quasi_pdf_gluon(
                        np.real(sample), z_fm, pz_gev, x_grid=X_GRID)
                except (ValueError, FloatingPointError):
                    continue
                quasi_pdf_samples[sample_index] = qpdf
            stem = f'tau{tau:g}_{tag}'
            quasi_pdf_samples = np.asarray(quasi_pdf_samples)
            quasi_samples = np.asarray(quasi_samples)
            coverage = float(np.mean(np.isfinite(quasi_samples)))
            if np.isfinite(quasi_samples).any():
                qtmd_mean, qtmd_error = _finite_mean_error(
                    quasi_samples, jackknife=True)
                qpdf_mean, qpdf_error = _finite_mean_error(
                    quasi_pdf_samples, jackknife=True)
                status = (
                    'diagnostic' if coverage >= 0.05
                    else 'insufficient_coverage')
            else:
                qtmd_mean = qtmd_error = np.full(
                    (len(X_GRID), len(B_LIST)), np.nan)
                qpdf_mean = qpdf_error = np.full(len(X_GRID), np.nan)
                status = 'unavailable'
            _write_npz(
                out_dir / f'quasi_{stem}.npz',
                x_grid=X_GRID,
                z_grid_fm=z_fm,
                b_grid_fm=b_fm,
                hR_samples=hR,
                normalization_denominator=normalization_norm,
                quasi_pdf_samples=quasi_pdf_samples,
                quasi_tmd_pdf_samples=quasi_samples,
                quasi_pdf_mean=qpdf_mean,
                quasi_pdf_error=qpdf_error,
                quasi_tmd_pdf_mean=qtmd_mean,
                quasi_tmd_pdf_error=qtmd_error,
            )
            _plot_matrix_element(
                z_fm, hR, out_dir / f'hR_{stem}.png', stem)
            _plot_pdf(
                X_GRID, qpdf_mean, qpdf_error, qtmd_mean, qtmd_error, b_fm,
                out_dir / f'quasi_{stem}.png', stem)
            summary[stem] = {
                'product_group': 'quasi_pdf_tmd',
                'status': status,
                'tau': float(tau),
                'momentum': tag,
                'pz_gev': float(pz_gev),
                'n_finite_cells': int(np.count_nonzero(
                    np.isfinite(quasi_samples).all(axis=1))),
                'finite_fraction': coverage,
                'x_min': float(X_GRID[0]),
                'x_max': float(X_GRID[-1]),
            }
            hR_by_momentum[float(tau)][pz_lattice] = hR
            logger(f"  PDF tau={tau:g} {tag}: {status}")
        _cs_kernel(tau, hR_by_momentum[float(tau)], out_dir, summary)
    _write_json(out_dir / 'pdf_summary.json', summary)
    return summary


def _plot_matrix_element(z_fm, hR_samples, path, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mean, error = _finite_mean_error(hR_samples, jackknife=True)
    fig, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for b_index in (0, 3, 5, 6):
        if b_index >= mean.shape[1]:
            continue
        axis.errorbar(
            z_fm, np.real(mean[:, b_index]),
            yerr=np.real(error[:, b_index]), marker='o', ms=2.5,
            lw=1.0, capsize=1.5,
            label=f'b={B_LIST[b_index]}a')
    axis.axhline(0.0, color='0.5', lw=0.8)
    axis.set_xlabel(r'$z$ [fm]')
    axis.set_ylabel(r'$\mathrm{Re}\,h_R(z,b)$')
    axis.set_title(f'重整化矩阵元: {label}')
    axis.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _cs_kernel(tau, hR_by_momentum, out_dir, summary):
    if 2 not in hR_by_momentum or 4 not in hR_by_momentum:
        return
    h1 = hR_by_momentum[2]
    h2 = hR_by_momentum[4]
    p1 = pz_to_gev(2, 'L24x72')
    p2 = pz_to_gev(4, 'L24x72')
    radius = math.sqrt(8.0 * tau)
    z_ref = next((index for index in Z_LIST if index > radius), 1)
    ratio_error = np.full(len(B_LIST), np.finfo(np.float64).max)
    for b_index in range(len(B_LIST)):
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_samples = (
                np.real(h1[:, z_ref, b_index])
                / np.real(h2[:, z_ref, b_index]))
        finite = ratio_samples[np.isfinite(ratio_samples)]
        if finite.size >= 2:
            ratio_error[b_index] = float(sem(finite, jackknife=True))
    central_mean = np.full(len(B_LIST), np.nan)
    h1_mean = np.nanmean(np.real(h1), axis=0)
    h2_mean = np.nanmean(np.real(h2), axis=0)
    for b_index in range(len(B_LIST)):
        try:
            central_mean[b_index] = cs_kernel_two_momentum(
                h1_mean[:, b_index], h2_mean[:, b_index], p1, p2,
                z_ref=z_ref, errors=ratio_error[b_index:b_index + 1],
                min_significance=1.0, require_same_sign=True,
            )[0]
        except (ValueError, FloatingPointError, IndexError):
            continue
    valid_b = np.isfinite(central_mean)
    kernels = np.full((h1.shape[0], len(B_LIST)), np.nan)
    for sample_index in range(h1.shape[0]):
        for b_index in np.flatnonzero(valid_b):
            try:
                kernels[sample_index, b_index] = cs_kernel_two_momentum(
                    np.real(h1[sample_index, :, b_index]),
                    np.real(h2[sample_index, :, b_index]), p1, p2,
                    z_ref=z_ref, require_same_sign=True,
                )[0]
            except (ValueError, FloatingPointError, IndexError):
                pass
    mean, error = _finite_mean_error(kernels, jackknife=True)
    _write_npz(
        out_dir / f'cs_kernel_tau{tau:g}.npz',
        b_grid_fm=np.asarray(B_LIST, dtype=float) * ALttc,
        kernel_samples=kernels,
        kernel_mean=mean,
        kernel_error=error,
        z_ref=np.asarray(z_ref, dtype=np.int64),
        pz1_gev=np.asarray(p1),
        pz2_gev=np.asarray(p2),
    )
    summary[f'tau{tau:g}_cs_kernel'] = {
        'status': 'diagnostic' if np.isfinite(kernels).any() else 'unavailable',
        'z_ref': int(z_ref),
        'finite_fraction': float(np.mean(np.isfinite(kernels))),
        'imaginary_relative_max': float(np.nanmax(
            np.abs(np.imag(np.concatenate((h1, h2), axis=0)))
            / (np.abs(np.real(np.concatenate((h1, h2), axis=0))) + 1.0e-30))),
    }
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    if np.isfinite(mean).any():
        axis.errorbar(np.asarray(B_LIST) * ALttc, mean, yerr=error,
                      marker='o', capsize=2.0, lw=1.0, color='#6A4C93')
        axis.axhline(0.0, color='0.5', lw=0.8)
    else:
        axis.text(0.5, 0.5, 'CS kernel unavailable\n(significance/sign/flow gate)',
                  ha='center', va='center', transform=axis.transAxes)
    axis.set_xlabel(r'$b_\perp$ [fm]')
    axis.set_ylabel('K(b)')
    axis.set_title(f'CS 核诊断, tau={tau:g}, z_ref={z_ref}')
    fig.savefig(out_dir / f'cs_kernel_tau{tau:g}.png', dpi=180)
    plt.close(fig)


def _plot_pdf(x, qpdf_mean, qpdf_error, qtmd_mean, qtmd_error, b_fm,
              path, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].errorbar(x, qpdf_mean, yerr=qpdf_error, lw=1.0, capsize=1.0,
                     color='#264653')
    axes[0].axhline(0.0, color='0.5', lw=0.8)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel(r'$x\tilde g(x)$')
    axes[0].set_title('胶子准 PDF（sin 通道）')
    for index in (0, 3, 5, 6):
        if index >= qtmd_mean.shape[1]:
            continue
        axes[1].errorbar(
            x, qtmd_mean[:, index], yerr=qtmd_error[:, index],
            lw=1.0, capsize=0.8, label=f'b={b_fm[index]:.2f} fm')
    axes[1].axhline(0.0, color='0.5', lw=0.8)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel(r'$x\tilde g(x,b_\perp)$')
    axes[1].set_title('胶子准 TMD-PDF（cos 通道）')
    axes[1].legend(fontsize=7)
    fig.suptitle(label)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_error_budget(run_dir, plateau_summary, flow_summary,
                       staircase_summary, pdf_summary):
    out_dir = Path(run_dir) / 'analysis'
    statistical = []
    window_drift = []
    for item in plateau_summary.values():
        statistical.append(item['windows'][
            f'{PRIMARY_PLATEAU[0]}_{PRIMARY_PLATEAU[1]}']['median_error'])
        values = [entry['median_error'] for entry in item['windows'].values()]
        if len(values) > 1:
            window_drift.append(float(np.nanmax(values) - np.nanmin(values)))
    flow_effect = []
    for key, item in flow_summary.items():
        if key.startswith('paired_tau') and item.get('status') == 'diagnostic':
            flow_effect.append(item['median_relative_difference'])
    staircase = [
        item['relative_l2'] for item in staircase_summary.values()
        if item.get('status') == 'diagnostic'
    ]
    budget = {
        'schema': 'pyqcd.tmd_full_error_budget.v1',
        'status': 'diagnostic_only',
        'statistical_median': float(np.nanmedian(statistical))
        if statistical else None,
        'plateau_window_drift_median': float(np.nanmedian(window_drift))
        if window_drift else None,
        'flow_time_relative_effect_median': float(np.nanmedian(flow_effect))
        if flow_effect else None,
        'staircase_relative_l2_median': float(np.nanmedian(staircase))
        if staircase else None,
        'fourier_support': {'x_min': float(X_GRID[0]),
                            'x_max': float(X_GRID[-1])},
        'perturbative_matching': 'not_applied',
        'gluon_quark_mixing': 'not_applied',
        'soft_rapidity': 'not_measured',
        'continuum_limit': 'not_available_single_lattice_spacing',
        'pdf_products': len(pdf_summary),
    }
    _write_json(out_dir / 'error_budget.json', budget)
    with (out_dir / 'error_budget.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source', 'value', 'status'])
        for key in (
                'statistical_median', 'plateau_window_drift_median',
                'flow_time_relative_effect_median',
                'staircase_relative_l2_median'):
            writer.writerow([key, budget[key], 'diagnostic'])
        for key in ('perturbative_matching', 'gluon_quark_mixing',
                    'soft_rapidity', 'continuum_limit'):
            writer.writerow([key, budget[key], 'unavailable'])
    _plot_error_budget(budget, out_dir / 'error_budget.png')
    return budget


def _plot_error_budget(budget, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = []
    values = []
    for key, label in (
            ('statistical_median', 'statistical'),
            ('plateau_window_drift_median', 'window drift'),
            ('flow_time_relative_effect_median', 'flow time'),
            ('staircase_relative_l2_median', 'staircase')):
        value = budget.get(key)
        if value is not None:
            labels.append(label)
            values.append(float(value))
    fig, axis = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    axis.bar(labels, values, color=['#457B9D', '#E9C46A', '#E76F51', '#6A4C93'])
    axis.set_yscale('symlog', linthresh=1.0e-6)
    axis.set_ylabel('reported magnitude')
    axis.set_title('诊断性误差与系统效应账本')
    axis.tick_params(axis='x', rotation=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


_MANIFEST_TOP_LEVEL = {
    'data', 'flow_history', 'flowed', 'env', 'workers', 'analysis', 'plots',
}
_MANIFEST_EXCLUDED_PARTS = {'agents'}
_MANIFEST_EXCLUDED_NAMES = {
    'analysis_summary.json', 'artifact_manifest.json', 'finalize.log',
    '.finalize.lock', 'finalize_status.json',
}


def _manifest_entry(path, run_dir):
    relative = path.relative_to(run_dir)
    if not relative.parts or relative.parts[0] not in _MANIFEST_TOP_LEVEL:
        return None
    if any(part in _MANIFEST_EXCLUDED_PARTS for part in relative.parts):
        return None
    if path.name in _MANIFEST_EXCLUDED_NAMES:
        return None
    if relative.parts[0] in {'data', 'flow_history', 'flowed', 'env', 'workers'}:
        producer = 'production'
    elif relative.parts[0] in {'analysis', 'plots'}:
        producer = 'analysis'
    else:
        producer = 'metadata'
    return {
        'conf_id': next((
            int(parent.name[4:]) for parent in path.parents
            if parent.parent == run_dir / 'data'
            and parent.name.startswith('conf')
            and parent.name[4:].isdigit()
        ), None),
        'path': str(relative),
        'kind': path.suffix.lstrip('.') or 'unknown',
        'producer': producer,
        'size_bytes': path.stat().st_size,
    }


def write_artifact_manifest(run_dir, conf_ids, tau_values):
    run_dir = Path(run_dir)
    entries = []
    manifest_path = run_dir / 'analysis' / 'artifact_manifest.json'
    for path in sorted(run_dir.rglob('*')):
        if not path.is_file() or path == manifest_path:
            continue
        if path.name.endswith(('.tmp', '.tmp.npz')):
            continue
        entry = _manifest_entry(path, run_dir)
        if entry is not None:
            entries.append(entry)
    manifest = {
        'schema': 'pyqcd.tmd_full_artifact_manifest.v1',
        'created_utc': _utc_now(),
        'conf_ids': list(conf_ids),
        'tau_values': list(tau_values),
        'file_count': len(entries),
        'total_bytes': int(sum(item['size_bytes'] for item in entries)),
        'entries': entries,
    }
    _write_json(manifest_path, manifest)
    return manifest


def analyze_run(run_dir, conf_ids=CONF_IDS, tau_values=FLOW_TIMES,
                logger=print, allow_partial=False):
    started = time.perf_counter()
    run_dir = Path(run_dir).resolve()
    conf_ids = _validate_analysis_conf_ids(
        conf_ids, allow_partial=allow_partial)
    tau_values = tuple(float(value) for value in tau_values)
    logger(f"loading 12-configuration analysis inputs from {run_dir}")
    raw_manifest = write_raw_input_manifest(run_dir, conf_ids)
    polarized_2pt = load_multi_2pt_polarizations(
        run_dir, conf_ids,
        [momentum_tag(momentum) for momentum in MOMENTA_AXES],
        channels=('pp',), logger=logger, momenta=MOMENTA_AXES,
        precision=PRECISION, v_kind='VVV',
        polarizations=('nopol', 'pol35'),
    )
    if set(polarized_2pt) != set(conf_ids):
        raise ValueError('source-resolved 双极化 C2 未覆盖全部组态')
    corr2, corr3 = load_generic_correlators(run_dir, conf_ids)
    flow_histories = load_flow_histories(run_dir, conf_ids)
    flow_summary = analyze_flow_histories(
        run_dir, flow_histories, logger=logger)
    spectrum = analyze_spectrum(
        run_dir, conf_ids, polarized_2pt, corr2, corr3, logger=logger)
    ratio_1d = analyze_ratio_1d(
        run_dir, conf_ids, polarized_2pt, tau_values, logger=logger)
    ratio_2d = analyze_ratio_2d(
        run_dir, conf_ids, polarized_2pt, tau_values, logger=logger)
    plateaus = analyze_plateaus(run_dir, ratio_1d, logger=logger)
    plateaus_2d = analyze_plateaus_2d(
        run_dir, ratio_2d, logger=logger)
    windows = analyze_flow_windows(
        run_dir, ratio_1d, tau_values, logger=logger)
    staircase = _staircase_summary(ratio_2d)
    pdf = analyze_pdf(run_dir, conf_ids, ratio_1d, tau_values, logger=logger)
    budget = write_error_budget(
        run_dir, plateaus, windows, staircase, pdf)
    pdf_products = [
        item for item in pdf.values()
        if item.get('product_group') == 'quasi_pdf_tmd'
    ]
    n_pdf_products = sum(
        1 for item in pdf_products if item.get('status') == 'diagnostic')
    n_cs_products = sum(
        1 for key, item in pdf.items()
        if key.endswith('_cs_kernel') and 'status' in item)
    manifest = write_artifact_manifest(run_dir, conf_ids, tau_values)
    status = make_lattice_only_status(
        conf_ids=conf_ids,
        tau_values=tau_values,
        elapsed_seconds=round(time.perf_counter() - started, 3),
        n_flow_histories=len(flow_histories),
        raw_input_manifest='analysis/raw_input_manifest.json',
        raw_input_configurations=raw_manifest['n_configurations'],
        flow_history=flow_summary,
        n_ratio_1d=len(ratio_1d),
        n_ratio_2d=len(ratio_2d),
        n_pdf=n_pdf_products,
        n_pdf_requested=len(pdf_products),
        n_cs_kernel=n_cs_products,
        spectrum=spectrum,
        flow_windows=windows,
        staircase=staircase,
        plateau_2d=plateaus_2d,
        error_budget=budget,
        artifact_manifest='analysis/artifact_manifest.json',
        artifact_count=manifest['file_count'],
        artifact_bytes=manifest['total_bytes'],
    )
    _write_json(run_dir / 'analysis' / 'analysis_summary.json', status)
    logger(
        f"analysis complete in {status['elapsed_seconds']:.1f}s; "
        "status=diagnostic_only")
    return status


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--conf-ids', default=','.join(map(str, CONF_IDS)))
    parser.add_argument('--tau-list', default=','.join(map(str, FLOW_TIMES)))
    parser.add_argument(
        '--allow-partial', action='store_true',
        help='允许显式缩小组态集合，仅用于 smoke/调试，不用于正式报告')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    conf_ids = tuple(int(value) for value in args.conf_ids.split(',') if value)
    tau_values = tuple(
        float(value) for value in args.tau_list.split(',') if value)
    result = analyze_run(
        args.run_dir, conf_ids, tau_values,
        allow_partial=args.allow_partial)
    print(json.dumps({
        'status': result['status'],
        'physical_acceptance': result['physical_acceptance'],
        'summary': str(
            Path(args.run_dir).resolve() / 'analysis' / 'analysis_summary.json'),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
