#!/usr/bin/env python3
"""Run a flow-window-restricted TMD disconnected diagnostic.

Unlike the production workflow, this diagnostic computes a caller-declared
``z_list``/``b_list`` so the nonlocal points can satisfy
``sqrt(8*tau) < |z|,|b|`` explicitly.  It never declares a physical PDF.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from pyqcd.analysis import run_disconnected_tmd_ratio
from pyqcd.pipeline._tmd9 import (
    compute_tmd_ope_time,
    load_multi_2pt,
    load_tmd_ope_all,
    momentum_tag,
)
from pyqcd.renorm import a_len_set, fm_to_GeV
from pyqcd.tools import set_backend


DEFAULT_CONF_IDS = [4150, 6250, 6450, 6650, 6850, 7050,
                    7250, 7450, 7650, 7850, 8050]
DEFAULT_Z_LIST = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
DEFAULT_B_LIST = [3, 4]
DEFAULT_MOMENTA = ((0, 0, 0), (2, 0, 0), (4, 0, 0))


def _integer_list(text):
    values = tuple(int(value) for value in text.split(',') if value != '')
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("list must contain non-negative integers")
    return values


def flow_window_indices(z_over_a, b_over_a, tau):
    """Return local/nonlocal masks for ``sqrt(8*tau) < |z|,|b|``."""
    z = np.asarray(z_over_a, dtype=float)
    b = np.asarray(b_over_a, dtype=float)
    if (z.ndim != 1 or b.ndim != 1 or z.size == 0 or b.size == 0
            or not np.isfinite(z).all() or not np.isfinite(b).all()
            or np.any(z < 0.0) or np.any(b < 0.0)):
        raise ValueError("z_over_a and b_over_a must be nonnegative finite vectors")
    tau = float(tau)
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError("tau must be finite and nonnegative")
    radius = float(np.sqrt(8.0 * tau))
    return {
        "radius_over_a": radius,
        "z_local": z == 0.0,
        "z_nonlocal": z > radius,
        "b_local": b == 0.0,
        "b_nonlocal": b > radius,
    }


def run_scan(*, source_run, out_dir, conf_ids, z_list, b_list, tau,
             eps=0.05, precision='complex64', bootstrap_samples=1000,
             backend='torch', logger=print):
    """Compute OPE and Sumratio diagnostics for one declared window."""
    conf_ids = [int(value) for value in conf_ids]
    z_list = [int(value) for value in z_list]
    b_list = [int(value) for value in b_list]
    if not conf_ids or not z_list or not b_list:
        raise ValueError("conf_ids, z_list and b_list must be nonempty")
    if (isinstance(bootstrap_samples, (bool, np.bool_))
            or not isinstance(bootstrap_samples, (int, np.integer))
            or bootstrap_samples <= 0):
        raise ValueError("bootstrap_samples must be a positive integer")
    masks = flow_window_indices(z_list, b_list, tau)
    invalid_z = [value for value, valid in zip(z_list, masks['z_nonlocal'])
                 if value != 0 and not valid]
    invalid_b = [value for value, valid in zip(b_list, masks['b_nonlocal'])
                 if value != 0 and not valid]
    if invalid_z or invalid_b:
        raise ValueError(
            f"requested nonlocal points violate flow window: z={invalid_z}, "
            f"b={invalid_b}, radius/a={masks['radius_over_a']:.6g}")

    source_run = os.path.abspath(source_run)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    set_backend(backend, device='cuda:0' if backend == 'torch' else None)
    if backend == 'torch':
        import torch
        torch.set_num_threads(8)
    for conf_id in conf_ids:
        logger(f"=== flow-window OPE conf{conf_id} tau={tau:g} ===")
        compute_tmd_ope_time(
            conf_id, out_dir, logger, z_list, b_list, tau=tau, eps=eps,
            precision=precision, gauge_flow_dir=os.path.join(out_dir, 'flowed'),
            staple_length=max(z_list))
    tags = [momentum_tag(momentum) for momentum in DEFAULT_MOMENTA]
    corr2 = load_multi_2pt(
        source_run, conf_ids, tags, channels=('pp',), logger=logger,
        momenta=DEFAULT_MOMENTA, precision=precision, source_resolved=True)
    ope = load_tmd_ope_all(
        out_dir, conf_ids, z_list, b_list, logger=logger,
        staple_length=max(z_list), tau=tau, eps=eps, precision=precision)
    results = {}
    for tag in tags:
        logger(f"=== flow-window analysis {tag} ===")
        result = run_disconnected_tmd_ratio(
            corr2, ope, conf_ids, out_dir, logger=logger,
            NT=72, nz=len(z_list), nb=len(b_list), dt_max=20,
            dt_start=7, dt_end=10, cut=6, momentum=tag,
            ratio_estimator='ensemble',
            a_fm=float(a_len_set['L24x72']) * fm_to_GeV,
            bootstrap_samples=int(bootstrap_samples), legacy_fit=False,
        )
        fit = result['proton']['sumratio_difference']
        results[tag] = {
            'quality_points': int(np.count_nonzero(fit['quality_accepted'])),
            'total_points': int(fit['quality_accepted'].size),
            'central_fallback_points': int(np.count_nonzero(
                fit['central_fallback_used'])),
            'replica_fallback_mean_fraction': float(np.mean(
                fit['replica_fallback_fraction'])),
            'replica_fallback_max_fraction': float(np.max(
                fit['replica_fallback_fraction'])),
            'estimate_available': bool(np.all(fit['estimate_available'])),
            'M_shape': list(np.asarray(fit['M']).shape),
            'M_boot_shape': list(np.asarray(fit['M_boot']).shape),
            'status': 'diagnostic_only',
            'physical_acceptance': False,
        }
    payload = {
        'schema': 'pyqcd.flow_window_scan.v1',
        'source_run': source_run,
        'out_dir': out_dir,
        'conf_ids': conf_ids,
        'z_list': z_list,
        'b_list': b_list,
        'tau': float(tau),
        'eps': float(eps),
        'bootstrap_samples': int(bootstrap_samples),
        'masks': {key: np.asarray(value).tolist() if isinstance(value, np.ndarray)
                  else value for key, value in masks.items()},
        'results': results,
        'physical_acceptance': False,
        'note': 'flow-window diagnostic only; no soft/rapidity, mixing, Z_R or continuum limit',
    }
    analysis_dir = os.path.join(out_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    with open(os.path.join(analysis_dir, 'flow_window_scan.json'), 'w',
              encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--conf-ids', default=','.join(map(str, DEFAULT_CONF_IDS)))
    parser.add_argument('--z-list', type=_integer_list,
                        default=tuple(DEFAULT_Z_LIST))
    parser.add_argument('--b-list', type=_integer_list,
                        default=tuple(DEFAULT_B_LIST))
    parser.add_argument('--tau', type=float, required=True)
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--precision', default='complex64',
                        choices=('complex64', 'complex128'))
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    parser.add_argument('--backend', default='torch', choices=('torch', 'numpy'))
    args = parser.parse_args()
    run_scan(
        source_run=args.source_run, out_dir=args.out,
        conf_ids=_integer_list(args.conf_ids),
        z_list=args.z_list, b_list=args.b_list, tau=args.tau, eps=args.eps,
        precision=args.precision, bootstrap_samples=args.bootstrap_samples,
        backend=args.backend,
    )


if __name__ == '__main__':
    main()
