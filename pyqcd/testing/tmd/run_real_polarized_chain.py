"""Resumable real-data driver for the polarized finite-b TMD chain."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from pyqcd.pipeline._steps import free_gpu_memory
from pyqcd.pipeline._tmd9 import (
    MOMENTA_Z,
    compute_2pt_polarizations_multi,
    compute_tmd_ope_time,
    compute_tmd_ope_time_2d,
    compute_vertices_multi,
    load_full_channel_tmd_ope_all,
    load_full_channel_tmd_ope_2d_all,
    load_multi_2pt_polarizations,
)
from pyqcd.tools import set_backend, set_precision


DEFAULT_RUN_DIR = Path("data/runs/tmd-production-10conf")
Z_LIST = list(range(13))
B_LIST = list(range(5))
TAU = 3.0
EPS = 0.05
STAPLE_LENGTH = 12
B_X_LIST = list(range(5))
B_Y_LIST = list(range(2))
TRANSVERSE_ORDER = (0, 1)


def discover_configurations(run_dir):
    """Return sorted production configuration IDs without guessing extra data."""
    data_root = Path(run_dir) / "data"
    found = []
    if not data_root.is_dir():
        return found
    for path in data_root.glob("conf*"):
        suffix = path.name[4:]
        if path.is_dir() and suffix.isdigit():
            found.append(int(suffix))
    return sorted(found)


def _c2_complete(run_dir, conf_id, precision):
    loaded = load_multi_2pt_polarizations(
        run_dir, [conf_id], ['P000', 'P200', 'P400'],
        logger=lambda *_args: None,
        momenta=MOMENTA_Z, precision=precision)
    return conf_id in loaded


def _ope_complete(run_dir, conf_id, precision):
    loaded = load_full_channel_tmd_ope_all(
        run_dir, [conf_id], Z_LIST, B_LIST,
        logger=lambda *_args: None, staple_length=STAPLE_LENGTH,
        tau=TAU, eps=EPS, precision=precision)
    return conf_id in loaded


def _ope_2d_complete(run_dir, conf_id, precision, b_x_list, b_y_list,
                     transverse_order):
    loaded = load_full_channel_tmd_ope_2d_all(
        run_dir, [conf_id], Z_LIST, b_x_list, b_y_list,
        logger=lambda *_args: None, staple_length=STAPLE_LENGTH,
        tau=TAU, eps=EPS, precision=precision,
        transverse_order=transverse_order)
    return conf_id in loaded


def _write_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_chain(run_dir, conf_ids=None, *, precision='complex64',
              backend='torch', device=0, skip_c2=False, skip_ope=False,
              recompute_c2=False, recompute_ope=False,
              checkpoint=None, logger=print):
    """Compute missing polarized C2 and full-channel OPE stages in order."""
    run_dir = Path(run_dir).resolve()
    if conf_ids is None:
        conf_ids = discover_configurations(run_dir)
    conf_ids = [int(value) for value in conf_ids]
    if not conf_ids:
        raise ValueError("conf_ids 不能为空")
    if len(set(conf_ids)) != len(conf_ids):
        raise ValueError("conf_ids 不得重复")
    device_setting = (
        'cpu' if isinstance(device, str) and device.lower() == 'cpu'
        else device)
    set_backend(
        backend, device=device_setting if backend == 'torch' else None)
    if backend == 'torch':
        set_precision(precision)

    checkpoint = (
        Path(checkpoint) if checkpoint is not None
        else run_dir / 'analysis' / 'real_polarized_chain_status.json')
    payload = {
        'schema': 'pyqcd.real_polarized_chain_status.v1',
        'run_dir': str(run_dir),
        'precision': precision,
        'backend': backend,
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'running',
        'configurations': {},
    }
    _write_checkpoint(checkpoint, payload)
    logger(f"real polarized chain: {len(conf_ids)} configurations")

    for conf_id in conf_ids:
        started = time.perf_counter()
        record = payload['configurations'].setdefault(
            str(conf_id), {'c2': 'pending', 'ope': 'pending'})
        logger(f"\n=== conf {conf_id} ===")
        try:
            if skip_c2:
                record['c2'] = 'skipped'
            elif not recompute_c2 and _c2_complete(
                    run_dir, conf_id, precision):
                record['c2'] = 'cache_hit'
                logger(f"  conf={conf_id}: polarized C2 cache hit")
            else:
                stage_start = time.perf_counter()
                vertices = compute_vertices_multi(
                    conf_id, run_dir, logger, MOMENTA_Z,
                    precision=precision, recompute=False)
                compute_2pt_polarizations_multi(
                    conf_id, run_dir, logger, vertices, MOMENTA_Z,
                    precision=precision, recompute=recompute_c2)
                del vertices
                free_gpu_memory()
                record['c2'] = 'computed'
                record['c2_seconds'] = round(
                    time.perf_counter() - stage_start, 3)
                logger(f"  conf={conf_id}: polarized C2 computed in "
                       f"{record['c2_seconds']:.1f}s")

            if skip_ope:
                record['ope'] = 'skipped'
            elif not recompute_ope and _ope_complete(
                    run_dir, conf_id, precision):
                record['ope'] = 'cache_hit'
                logger(f"  conf={conf_id}: full-channel OPE cache hit")
            else:
                stage_start = time.perf_counter()
                compute_tmd_ope_time(
                    conf_id, run_dir, logger, Z_LIST, B_LIST,
                    tau=TAU, eps=EPS, precision=precision,
                    staple_length=STAPLE_LENGTH, full_channels=True,
                    recompute=recompute_ope)
                free_gpu_memory()
                record['ope'] = 'computed'
                record['ope_seconds'] = round(
                    time.perf_counter() - stage_start, 3)
                logger(f"  conf={conf_id}: full-channel OPE computed in "
                       f"{record['ope_seconds']:.1f}s")
            record['status'] = 'complete'
        except Exception as exc:
            record['status'] = 'failed'
            record['error'] = f"{type(exc).__name__}: {exc}"
            payload['status'] = 'failed'
            payload['failed_conf_id'] = conf_id
            payload['elapsed_seconds'] = round(
                time.perf_counter() - started, 3)
            _write_checkpoint(checkpoint, payload)
            raise
        finally:
            payload['updated_utc'] = datetime.now(timezone.utc).isoformat()
            _write_checkpoint(checkpoint, payload)

    payload['status'] = 'complete'
    payload['finished_utc'] = datetime.now(timezone.utc).isoformat()
    payload['checkpoint'] = str(checkpoint.resolve())
    _write_checkpoint(checkpoint, payload)
    logger(f"real polarized chain complete: {checkpoint}")
    return payload


def run_chain_2d(run_dir, conf_ids=None, *, precision='complex64',
                 backend='torch', device=0, b_x_list=B_X_LIST,
                 b_y_list=B_Y_LIST, transverse_order=TRANSVERSE_ORDER,
                 recompute=False, checkpoint=None, logger=print):
    """Compute missing two-component full-channel OPE caches in order."""
    run_dir = Path(run_dir).resolve()
    if conf_ids is None:
        conf_ids = discover_configurations(run_dir)
    conf_ids = [int(value) for value in conf_ids]
    if not conf_ids:
        raise ValueError("conf_ids 不能为空")
    device_setting = (
        'cpu' if isinstance(device, str) and device.lower() == 'cpu'
        else device)
    set_backend(
        backend, device=device_setting if backend == 'torch' else None)
    if backend == 'torch':
        set_precision(precision)
    checkpoint = (
        Path(checkpoint) if checkpoint is not None
        else run_dir / 'analysis' / 'real_polarized_chain_2d_status.json')
    payload = {
        'schema': 'pyqcd.real_polarized_chain_2d_status.v1',
        'run_dir': str(run_dir),
        'precision': precision,
        'backend': backend,
        'b_x_list': list(b_x_list),
        'b_y_list': list(b_y_list),
        'transverse_order': list(transverse_order),
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'running',
        'configurations': {},
    }
    _write_checkpoint(checkpoint, payload)
    logger(f"real 2D polarized chain: {len(conf_ids)} configurations")
    for conf_id in conf_ids:
        record = payload['configurations'].setdefault(
            str(conf_id), {'ope_2d': 'pending'})
        logger(f"\n=== conf {conf_id}: 2D OPE ===")
        try:
            stage_start = time.perf_counter()
            if not recompute and _ope_2d_complete(
                    run_dir, conf_id, precision, b_x_list, b_y_list,
                    transverse_order):
                record['ope_2d'] = 'cache_hit'
                logger(f"  conf={conf_id}: 2D OPE cache hit")
            else:
                compute_tmd_ope_time_2d(
                    conf_id, run_dir, logger, Z_LIST, b_x_list, b_y_list,
                    tau=TAU, eps=EPS, precision=precision,
                    z_dir=2, b_x_dir=0, b_y_dir=1,
                    transverse_order=transverse_order,
                    recompute=recompute, staple_length=STAPLE_LENGTH)
                free_gpu_memory()
                record['ope_2d'] = 'computed'
                record['ope_2d_seconds'] = round(
                    time.perf_counter() - stage_start, 3)
                logger(f"  conf={conf_id}: 2D OPE computed in "
                       f"{record['ope_2d_seconds']:.1f}s")
            record['status'] = 'complete'
        except Exception as exc:
            record['status'] = 'failed'
            record['error'] = f"{type(exc).__name__}: {exc}"
            payload['status'] = 'failed'
            payload['failed_conf_id'] = conf_id
            _write_checkpoint(checkpoint, payload)
            raise
        finally:
            payload['updated_utc'] = datetime.now(timezone.utc).isoformat()
            _write_checkpoint(checkpoint, payload)
    payload['status'] = 'complete'
    payload['finished_utc'] = datetime.now(timezone.utc).isoformat()
    payload['checkpoint'] = str(checkpoint.resolve())
    _write_checkpoint(checkpoint, payload)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        '--conf-ids', default=None,
        help='comma-separated configuration IDs; default: discover conf* dirs')
    parser.add_argument('--precision', default='complex64')
    parser.add_argument('--backend', default='torch')
    parser.add_argument(
        '--device', default='0',
        help="torch device: integer GPU index or 'cpu'")
    parser.add_argument('--skip-c2', action='store_true')
    parser.add_argument('--skip-ope', action='store_true')
    parser.add_argument('--recompute-c2', action='store_true')
    parser.add_argument('--recompute-ope', action='store_true')
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--dimension', choices=('1d', '2d'), default='1d')
    parser.add_argument('--bx-list', default='0,1,2,3,4')
    parser.add_argument('--by-list', default='0,1')
    parser.add_argument('--transverse-order', default='0,1')
    return parser.parse_args()


def main():
    args = parse_args()
    conf_ids = None
    if args.conf_ids:
        conf_ids = [int(value) for value in args.conf_ids.split(',') if value]
    device = (
        args.device if str(args.device).lower() == 'cpu'
        else int(args.device))
    if args.dimension == '2d':
        result = run_chain_2d(
            args.run_dir, conf_ids, precision=args.precision,
            backend=args.backend, device=device,
            b_x_list=[int(x) for x in args.bx_list.split(',')],
            b_y_list=[int(x) for x in args.by_list.split(',')],
            transverse_order=tuple(
                int(x) for x in args.transverse_order.split(',')),
            recompute=args.recompute_ope,
            checkpoint=args.checkpoint)
    else:
        result = run_chain(
            args.run_dir, conf_ids, precision=args.precision,
            backend=args.backend, device=device,
            skip_c2=args.skip_c2, skip_ope=args.skip_ope,
            recompute_c2=args.recompute_c2,
            recompute_ope=args.recompute_ope,
            checkpoint=args.checkpoint)
    print(json.dumps({
        'status': result['status'],
        'checkpoint': result['checkpoint'],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
