"""Strict read-back verification for the full 12-configuration TMD run."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pyqcd.pipeline._steps import _load_strict_cache_mapping
from pyqcd.pipeline._tmd9 import (
    _vertex_cache_specs,
    load_full_channel_tmd_ope_2d_all,
    load_full_channel_tmd_ope_all,
    load_multi_2pt_polarizations,
    load_tmd_ope_all,
    momentum_tag,
)
from pyqcd.testing.tmd.analyze_full_production import (
    DIRECTION_B,
    DIRECTION_MOMENTUM,
    load_flow_histories,
    load_generic_correlators,
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


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def verify_run(run_dir, conf_ids=CONF_IDS, tau_values=FLOW_TIMES,
               logger=print, require_analysis=True):
    run_dir = Path(run_dir).resolve()
    conf_ids = tuple(int(value) for value in conf_ids)
    tau_values = tuple(float(value) for value in tau_values)
    started = time.perf_counter()
    report = {
        'schema': 'pyqcd.tmd_full_verification.v1',
        'run_dir': str(run_dir),
        'conf_ids': list(conf_ids),
        'tau_values': list(tau_values),
        'started_utc': _utc_now(),
        'checks': [],
        'failures': [],
    }

    def run_check(name, function):
        check_started = time.perf_counter()
        logger(f'check: {name}')
        try:
            detail = function()
        except BaseException as exc:
            record = {
                'name': name,
                'status': 'failed',
                'seconds': round(time.perf_counter() - check_started, 3),
                'error_type': type(exc).__name__,
                'error': str(exc),
            }
            report['checks'].append(record)
            report['failures'].append(record)
            logger(f'  FAIL: {type(exc).__name__}: {exc}')
            return False
        record = {
            'name': name,
            'status': 'passed',
            'seconds': round(time.perf_counter() - check_started, 3),
            'detail': detail,
        }
        report['checks'].append(record)
        logger(f"  PASS ({record['seconds']:.1f}s): {detail}")
        return True

    tags = [momentum_tag(momentum) for momentum in MOMENTA_AXES]

    def verify_vertices():
        shapes = {}
        for conf_id in conf_ids:
            specs = _vertex_cache_specs(
                conf_id, run_dir, MOMENTA_AXES, PRECISION)
            loaded = _load_strict_cache_mapping(specs, logger=logger)
            if loaded is None:
                raise ValueError(f'conf={conf_id} VdV/VVV strict cache miss')
            shapes[str(conf_id)] = {
                name: list(array.shape) for name, array in loaded.items()
            }
            del loaded
        return shapes

    def verify_c2():
        loaded = load_multi_2pt_polarizations(
            run_dir, conf_ids, tags, channels=('pp',), logger=logger,
            momenta=MOMENTA_AXES, precision=PRECISION,
            v_kind='VVV', polarizations=('nopol', 'pol35'),
        )
        if set(loaded) != set(conf_ids):
            raise ValueError(f'双极化 C2 覆盖 {len(loaded)}/{len(conf_ids)}')
        return {
            str(conf_id): {
                polarization: {
                    key: list(value.shape)
                    for key, value in loaded[conf_id][polarization].items()
                }
                for polarization in ('nopol', 'pol35')
            }
            for conf_id in conf_ids
        }

    def verify_generic():
        corr2, corr3 = load_generic_correlators(run_dir, conf_ids)
        return {
            'n_configurations': len(corr2),
            'n_2pt': sum(len(item) for item in corr2.values()),
            'n_3pt': sum(len(item) for item in corr3.values()),
        }

    def verify_flow_history():
        histories = load_flow_histories(run_dir, conf_ids)
        return {
            str(conf_id): {
                key: list(value.shape)
                for key, value in histories[conf_id].items()
            }
            for conf_id in conf_ids
        }

    def verify_1d_ope():
        counts = {}
        for tau in tau_values:
            for z_dir in (0, 1, 2):
                b_dir = DIRECTION_B[z_dir]
                full = load_full_channel_tmd_ope_all(
                    run_dir, conf_ids, Z_LIST, B_LIST, logger=logger,
                    staple_length=STAPLE_LENGTH, tau=tau, eps=FLOW_EPS,
                    precision=PRECISION, z_dir=z_dir, b_dir=b_dir,
                )
                legacy = load_tmd_ope_all(
                    run_dir, conf_ids, Z_LIST, B_LIST, logger=logger,
                    staple_length=STAPLE_LENGTH, tau=tau, eps=FLOW_EPS,
                    precision=PRECISION, z_dir=z_dir, b_dir=b_dir,
                )
                if set(full) != set(conf_ids) or set(legacy) != set(conf_ids):
                    raise ValueError(
                        f'1D OPE tau={tau} z_dir={z_dir} '
                        f'full={len(full)} legacy={len(legacy)}')
                counts[f'tau{tau:g}_z{z_dir}'] = {
                    'full_shape': list(full[conf_ids[0]]['tmd_full'].shape),
                    'legacy_shape': list(legacy[conf_ids[0]]['tmd'].shape),
                }
        return counts

    def verify_2d_ope():
        counts = {}
        for tau in tau_values:
            for order in ((0, 1), (1, 0)):
                loaded = load_full_channel_tmd_ope_2d_all(
                    run_dir, conf_ids, Z_LIST, B_X_LIST, B_Y_LIST,
                    logger=logger, staple_length=STAPLE_LENGTH,
                    tau=tau, eps=FLOW_EPS, precision=PRECISION,
                    z_dir=2, b_x_dir=0, b_y_dir=1,
                    transverse_order=order,
                )
                if set(loaded) != set(conf_ids):
                    raise ValueError(
                        f'2D OPE tau={tau} order={order} '
                        f'覆盖 {len(loaded)}/{len(conf_ids)}')
                counts[f'tau{tau:g}_order{order[0]}{order[1]}'] = list(
                    loaded[conf_ids[0]]['tmd_full_2d'].shape)
        return counts

    def verify_analysis_artifacts():
        analysis_dir = run_dir / 'analysis'
        summary_path = analysis_dir / 'analysis_summary.json'
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('status') != 'diagnostic_only':
            raise ValueError(f'analysis status={summary.get("status")}')
        if summary.get('physical_acceptance') is not False:
            raise ValueError('analysis 不得升级 physical_acceptance')
        if int(summary.get('n_pdf', 0)) <= 0:
            raise ValueError('没有可用的准 PDF/TMD-PDF 产品')
        ratio_1d = sorted((analysis_dir / 'ratio_1d').glob('*.npz'))
        selected_1d = [path for path in ratio_1d if path.stem.endswith('_selected')]
        ratio_2d = sorted((analysis_dir / 'ratio_2d').glob('*.npz'))
        selected_2d = [path for path in ratio_2d if path.stem.endswith('_selected')]
        if len(selected_1d) < len(conf_ids) or len(selected_2d) < 2 * len(tau_values):
            raise ValueError(
                f'ratio selected 产物不足: 1D={len(selected_1d)}, '
                f'2D={len(selected_2d)}')
        for path in selected_1d + selected_2d:
            with np.load(path, allow_pickle=False) as handle:
                if 'unpolarized_samples' not in handle.files:
                    raise ValueError(f'{path} 缺 unpolarized_samples')
                samples = np.asarray(handle['unpolarized_samples'])
                if samples.ndim < 2 or samples.shape[0] < 2:
                    raise ValueError(f'{path} sample 轴无效: {samples.shape}')
                if 'sample_axis' not in handle.files or not str(
                        handle['sample_axis'].item()).startswith('sample,'):
                    raise ValueError(f'{path} 缺 sample-first axis metadata')
        required_json = (
            analysis_dir / 'spectrum' / 'spectrum_summary.json',
            analysis_dir / 'flow' / 'flow_summary.json',
            analysis_dir / 'plateaus' / 'plateau_summary.json',
            analysis_dir / 'plateaus_2d' / 'plateau_2d_summary.json',
            analysis_dir / 'flow_windows' / 'flow_window_summary.json',
            analysis_dir / 'pdf' / 'pdf_summary.json',
            analysis_dir / 'error_budget.json',
            analysis_dir / 'artifact_manifest.json',
            analysis_dir / 'raw_input_manifest.json',
        )
        missing_json = [str(path) for path in required_json if not path.is_file()]
        if missing_json:
            raise FileNotFoundError(f'缺少分析产物: {missing_json}')
        pdf_npz = sorted((analysis_dir / 'pdf').glob('quasi_*.npz'))
        if len(pdf_npz) < 2 * len(tau_values):
            raise ValueError(f'PDF NPZ 数量不足: {len(pdf_npz)}')
        png = sorted(run_dir.rglob('*.png'))
        if len(png) < 40:
            raise ValueError(f'关键词图数量不足: {len(png)}')
        temporary = sorted(run_dir.rglob('*.tmp*'))
        if temporary:
            raise RuntimeError(f'存在临时残留: {[str(path) for path in temporary]}')
        return {
            'ratio_1d_npz': len(ratio_1d),
            'ratio_2d_npz': len(ratio_2d),
            'pdf_npz': len(pdf_npz),
            'png': len(png),
            'n_pdf': int(summary.get('n_pdf', 0)),
        }

    run_check('vertices', verify_vertices)
    run_check('polarized-2pt', verify_c2)
    run_check('generic-2pt-3pt', verify_generic)
    run_check('flow-history', verify_flow_history)
    run_check('full-and-legacy-1d-ope', verify_1d_ope)
    run_check('full-2d-ope-both-staircases', verify_2d_ope)
    if require_analysis:
        run_check('analysis-artifacts', verify_analysis_artifacts)
    report['finished_utc'] = _utc_now()
    report['elapsed_seconds'] = round(time.perf_counter() - started, 3)
    report['status'] = 'passed' if not report['failures'] else 'failed'
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--conf-ids', default=','.join(map(str, CONF_IDS)))
    parser.add_argument('--tau-list', default=','.join(map(str, FLOW_TIMES)))
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--skip-analysis', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    conf_ids = tuple(int(value) for value in args.conf_ids.split(',') if value)
    tau_values = tuple(
        float(value) for value in args.tau_list.split(',') if value)
    report = verify_run(
        args.run_dir, conf_ids, tau_values,
        require_analysis=not args.skip_analysis)
    output = args.out or (
        Path(args.run_dir) / 'analysis' / 'validation_report.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(output)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
