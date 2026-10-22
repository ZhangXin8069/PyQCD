#!/usr/bin/env python3
"""
梯度流胶子 TMD-PDF 真实数据诊断链
=================================

调用 pyqcd 包（pyqcd.pipeline._tmd9 + renorm/analysis/operator）在真实
蒸馏数据（eigvec/peram/gauge，默认 11 组态 beta6.20_mu-0.2770_ms-0.2400_L24x72）
上跑通以下诊断链：

    1. 蒸馏 2pt（核子谱线，多动量：z 方向 [0,2,4] 或全方向 [0,2]）
    2. 梯度流：gauge → wilson_flow(τ=3a²) → flowed gauge
    3. 胶子 TMD 算符 O(z,b⊥)（flowed gauge 上，逐时间片空间求和）
    4. 不相连 3pt 因子化 + 真空扣除 + 比值 R
    5. 逐 (z,b) 拟合 → 裸矩阵元 c0(z,b,Pz)
    6. 自重整化（比值/混合方案）→ 准 TMD-PDF
    7. 一圈 gg-only 匹配 → 诊断的 x·g(x,b⊥) + CS 核
    8. 全图表输出 + JSON 汇总

这不是物理闭环：固定流时/格距/staple 长度、soft/rapidity 未测量，
默认只有一个或少量 Pz，且未处理胶子-夸克混合。输出统一标记为
``diagnostic_only``；只有补齐多尺度数据和误差账本后才可升级。

产出：`pyqcd/testing/tmd/artifacts/work/`（数据 + 图表 + 报告）。

用法：
    python -m pyqcd.testing.tmd.pdf_workflow --smoke
    python -m pyqcd.testing.tmd.pdf_workflow --conf-ids 6250 --momenta A
    python -m pyqcd.testing.tmd.pdf_workflow
    mpirun -np N python -m pyqcd.testing.tmd.pdf_workflow --full
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    '..', '..', '..'))

from pyqcd.pipeline._tmd9 import (
    MOMENTA_Z, MOMENTA_ALL, momentum_tag, parse_momentum_tag,
    compute_vertices_multi, compute_2pt_multi, compute_tmd_ope_time,
    load_multi_2pt, load_tmd_ope_all, self_renormalize,
)
from pyqcd.tools import set_backend, get_backend_name
from pyqcd.pipeline._config import get_gauge_path

DEFAULT_CONF_IDS = [4150, 6250, 6450, 6650, 6850, 7050,
                    7250, 7450, 7650, 7850, 8050]

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'artifacts', 'work')
LOG_ROOT = os.path.join(ROOT, 'logs', 'tmd_pdf')

# TMD 算符参数（格点单位）
Z_LIST = list(range(0, 13))          # z ∈ 0..12（Wilson 线纵向分离）
B_LIST = [0, 1, 2, 3, 4]             # b⊥ ∈ 0..4（横向位移）
STAPLE_LENGTH = max(abs(z) for z in Z_LIST)  # 全 z 扫描共享的固定臂长
TAU = 3.0                            # τ=3a² → 格点流时间 t=3
EPS = 0.05                           # RK3 步长（Luescher ≤0.05 保证 O(ε³)，60 步）
BOOTSTRAP_SAMPLES = 1000             # 与参考生产量级一致的非 jackknife replica 数


def parse_args():
    ap = argparse.ArgumentParser(description='梯度流胶子 TMD-PDF')
    ap.add_argument('--conf-ids', type=str, default=None,
                    help='逗号分隔组态（默认全 11 组态）')
    ap.add_argument('--momenta', choices=['A', 'B', 'Z'], default='A',
                    help='A=z 方向[0,2,4]; B=全方向{0,2}; Z=仅 z 方向（与A同）')
    ap.add_argument('--smoke', action='store_true',
                    help='冒烟：单组态(6250) 单动量(P200)')
    ap.add_argument('--smoke-mom', type=str, default='P200',
                    help='冒烟动量标签（默认 P200；多位/负分量如 P10_-2_0）')
    ap.add_argument('--skip-2pt', action='store_true', help='跳过蒸馏 2pt')
    ap.add_argument('--skip-ope', action='store_true', help='跳过梯度流 TMD 算符')
    ap.add_argument('--backend', default='torch', choices=['torch', 'numpy'])
    ap.add_argument('--precision', default='complex64',
                    choices=['complex64', 'complex128'])
    ap.add_argument('--cpu-threads', type=int, default=0,
                    help='Torch CPU 线程数；0 使用后端默认值')
    ap.add_argument('--staple-length', type=int, default=STAPLE_LENGTH,
                    help='所有 z 共享的 staple 臂长（格点单位，默认 12）')
    ap.add_argument('--tau', type=float, default=TAU,
                    help='无量纲 Wilson-flow 时间 tau=t/a^2（默认 3）')
    ap.add_argument('--eps', type=float, default=EPS,
                    help='RK3 积分最大步长 epsilon（默认 0.05）')
    ap.add_argument(
        '--normalization', choices=('local_spin2', 'z0_ratio'),
        default='local_spin2',
        help='local_spin2=局域绝对幅度诊断；z0_ratio=仅形状诊断')
    ap.add_argument('--out', default=None,
                    help='输出目录（默认 pyqcd/testing/tmd/artifacts/work）')
    ap.add_argument('--only-plot', action='store_true',
                    help='仅从已有数据重新出图+分析')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--compute-only', action='store_true',
                    help='只生成 2pt/OPE strict cache，不执行分析和作图')
    return ap.parse_args()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _attr_text(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return None if value is None else str(value)


def write_data_provenance(run_dir, conf_ids, logger=print):
    """Record input and cache identities without changing cache validity."""
    import h5py

    entries = {}
    for conf_id in conf_ids:
        gauge_path = get_gauge_path(conf_id)
        stat = os.stat(gauge_path)
        cdir = os.path.join(run_dir, 'data', f'conf{conf_id}')
        caches = []
        for pattern in (
                'corr_pp_*tmd9-strict-v2*.h5',
                'tmd_ope_*_conf*.h5'):
            for path in sorted(glob.glob(os.path.join(cdir, pattern))):
                with h5py.File(path, 'r') as handle:
                    caches.append({
                        'path': os.path.relpath(path, run_dir),
                        'contract_json': _attr_text(handle.attrs.get(
                            'pyqcd_cache_contract_json')),
                        'contract_sha256': _attr_text(handle.attrs.get(
                            'pyqcd_cache_contract_sha256')),
                    })
        entries[str(conf_id)] = {
            'gauge_path': gauge_path,
            'gauge_size_bytes': int(stat.st_size),
            'gauge_mtime_ns': int(stat.st_mtime_ns),
            'gauge_sha256': _sha256_file(gauge_path),
            'caches': caches,
        }
        logger(f"  provenance: conf{conf_id} gauge sha256="
               f"{entries[str(conf_id)]['gauge_sha256'][:12]}...")
    payload = {
        'schema': 'pyqcd.tmd_data_provenance.v1',
        'run_dir': os.path.abspath(run_dir),
        'entries': entries,
    }
    out_dir = os.path.join(run_dir, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'data_provenance.json')
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    return path


def select_run_scope(args):
    """把 CLI 选择规范化为组态列表和唯一有序动量列表。"""
    if args.smoke:
        return [6250], [parse_momentum_tag(args.smoke_mom)]
    if args.conf_ids:
        conf_ids = [int(value) for value in args.conf_ids.split(',')]
    else:
        conf_ids = list(DEFAULT_CONF_IDS)
    momenta = MOMENTA_Z if args.momenta in ('A', 'Z') else MOMENTA_ALL
    return conf_ids, list(momenta)


def main():
    args = parse_args()
    out_dir = args.out or OUT_ROOT
    log_dir = LOG_ROOT

    logger = lambda *a: print(*a)

    conf_ids, momenta = select_run_scope(args)

    mom_tags = [momentum_tag(m) for m in momenta]
    logger(f"=== 梯度流核子胶子 TMD-PDF ===")
    logger(f"  组态: {conf_ids}")
    logger(f"  动量: {list(momenta)} ({mom_tags})")
    logger(f"  z_list={Z_LIST}, b_list={B_LIST}, "
           f"staple_length={args.staple_length}, tau={args.tau}, "
           f"eps={args.eps}")
    logger(f"  输出: {out_dir}")
    logger(f"  backend={args.backend}, precision={args.precision}")

    if args.cpu_threads < 0:
        raise ValueError('--cpu-threads 必须为非负整数')
    if args.dry_run:
        return

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    set_backend(args.backend, device='cuda:0' if args.backend == 'torch' else None)
    if args.backend == 'torch' and args.cpu_threads > 0:
        import torch
        torch.set_num_threads(args.cpu_threads)

    run_dir = out_dir
    data_dir = os.path.join(run_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # ── 阶段 1：蒸馏顶点 + 2pt（多动量）─────────────────────────────
    if not args.only_plot and not args.skip_2pt:
        for cid in conf_ids:
            logger(f"\n─── 2pt: conf {cid} 动量 {mom_tags} ───")
            t0 = time.perf_counter()
            verts = compute_vertices_multi(cid, run_dir, logger, momenta,
                                           precision=args.precision)
            acc = compute_2pt_multi(cid, run_dir, logger, verts, momenta,
                                    precision=args.precision,
                                    channels=('pp',), source_resolved=True)
            logger(f"  2pt conf={cid} done in {time.perf_counter()-t0:.0f}s")
    else:
        logger("跳过蒸馏 2pt（--skip-2pt）")

    # ── 阶段 2：梯度流 + 胶子 TMD 算符矩阵元 ────────────────────────
    if not args.only_plot and not args.skip_ope:
        flow_dir = os.path.join(run_dir, 'flowed')
        for cid in conf_ids:
            logger(f"\n─── 梯度流 TMD 算符: conf {cid} ───")
            t0 = time.perf_counter()
            res = compute_tmd_ope_time(
                cid, run_dir, logger, Z_LIST, B_LIST,
                tau=args.tau, eps=args.eps, precision=args.precision,
                gauge_flow_dir=flow_dir,
                staple_length=args.staple_length)
            logger(f"  TMD OPE conf={cid} done in {time.perf_counter()-t0:.0f}s")
    else:
        logger("跳过梯度流 TMD 算符（--skip-ope）")

    if args.compute_only:
        logger("完成 compute-only；未执行 analysis/plots/report")
        return

    # ── 阶段 3：分析 + 出图 ────────────────────────────────────────
    run_analysis(conf_ids, momenta, run_dir, logger, args)
    write_data_provenance(run_dir, conf_ids, logger=logger)


def run_analysis(conf_ids, momenta, run_dir, logger, args):
    from pyqcd.analysis import (
        run_disconnected_tmd_ratio, plot_tmd_c0, plot_tmd_ratio,
    )
    from pyqcd.pipeline._tmd9 import load_tmd_ope_all
    from pyqcd.renorm import a_len_set, fm_to_GeV

    mom_tags = [momentum_tag(m) for m in momenta]
    corr2 = load_multi_2pt(
        run_dir, conf_ids, mom_tags, channels=('pp',), logger=logger,
        momenta=momenta, precision=args.precision, source_resolved=True)
    ope = load_tmd_ope_all(
        run_dir, conf_ids, Z_LIST, B_LIST, logger=logger,
        staple_length=args.staple_length,
        tau=getattr(args, 'tau', TAU), eps=getattr(args, 'eps', EPS),
        precision=args.precision)

    results = {}
    for tag in mom_tags:
        # 跳过缺失 2pt 的动量（如冒烟只算了 P200）
        has = any(f'corr_pp_{tag}' in corr2[cid] for cid in corr2)
        if not has:
            logger(f"  跳过 {tag}：无 2pt 数据")
            continue
        role = '零动量 hybrid 分母' if tag == 'P000' else '准 TMD 矩阵元'
        logger(f"\n=== 动量 {tag}：{role} ratio + 拟合 ===")
        res = run_disconnected_tmd_ratio(
            corr2, ope, conf_ids, run_dir, logger=logger,
            NT=72, nz=len(Z_LIST), nb=len(B_LIST),
            dt_max=20, dt_start=7, dt_end=10, cut=6,
            momentum=tag, ratio_estimator="ensemble",
            a_fm=float(a_len_set['L24x72']) * fm_to_GeV,
            bootstrap_samples=BOOTSTRAP_SAMPLES,
            legacy_fit=False)
        plot_tmd_c0(res, run_dir, tag, logger=logger,
                    nz=len(Z_LIST), nb=len(B_LIST))
        plot_tmd_ratio(res, run_dir, tag, logger=logger,
                       nz=len(Z_LIST), nb=len(B_LIST))
        if tag == 'P000':
            _write_zero_momentum_denominator_status(res, run_dir)
            continue
        results[tag] = {
            'c0_mean': np.asarray(res['proton']['c0']).mean(0).tolist(),
            'c0_err': None,
        }
        # 保存均值/误差（用 plateau 均值：11 组态统计下比 lsqfit 更稳健）
        c0 = np.asarray(res['proton']['c0'])
        from pyqcd.analysis import sem
        jackknife = res['proton'].get('resampling_kind') == 'jackknife'
        np.save(os.path.join(run_dir, 'analysis', 'tmd_ratio',
                             f'c0_mean_{tag}.npy'), c0.mean(0))
        np.save(os.path.join(run_dir, 'analysis', 'tmd_ratio',
                             f'c0_err_{tag}.npy'), sem(c0, jackknife))
        # plateau-mean 版本已由 run_disconnected_tmd_ratio 直接产出
        # （pyqcd.analysis.plateau_c0，fit 窗口内 ratio 均值，抗奇异协方差）

    # ── 阶段 4：自重整化 + 匹配 → TMD-PDF ──────────────────────────
    run_tmd_pdf_chain(
        conf_ids, run_dir, logger, args, momenta=momenta, corr2=corr2)
    write_summary(run_dir, conf_ids, momenta, logger, args)


def _write_zero_momentum_denominator_status(results, run_dir):
    """Persist the measured Pz=0 denominator gate without promoting fallbacks."""
    analysis_dir = os.path.join(run_dir, 'analysis', 'tmd_ratio')
    os.makedirs(analysis_dir, exist_ok=True)
    artifact_name = 'sumratio_difference_P000.npz'
    artifact_path = os.path.join(analysis_dir, artifact_name)
    payload = {
        'schema': 'pyqcd.zero_momentum_hybrid_denominator_status.v1',
        'momentum': 'P000',
        'role': 'zero_momentum_hybrid_denominator',
        'physical_status': 'unverified',
        'used_for_pdf': False,
        'artifact': artifact_name,
        'strict_gate_passed': False,
    }
    sumratio = results.get('proton', {}).get('sumratio_difference')
    if sumratio is None:
        payload['status'] = 'artifact_unavailable'
        payload['reason'] = 'sumratio_difference_missing'
    else:
        quality = np.asarray(sumratio['quality_accepted'], dtype=bool)
        available = np.asarray(
            sumratio['estimate_available'], dtype=bool)
        central_fallback = np.asarray(
            sumratio['central_fallback_used'], dtype=bool)
        replica_fallback = np.asarray(
            sumratio['replica_fallback_fraction'], dtype=float)
        finite = (
            np.isfinite(np.asarray(sumratio['M'])).all()
            and np.isfinite(np.asarray(sumratio['M_boot'])).all())
        strict = bool(
            quality.ndim == 2
            and available.shape == quality.shape
            and central_fallback.shape == quality.shape
            and replica_fallback.shape == quality.shape
            and finite
            and np.all(quality)
            and np.all(available)
            and not np.any(central_fallback)
            and not np.any(replica_fallback > 0.0))
        failed = [
            [int(z), int(b)]
            for z, b in np.argwhere(~quality)
        ]
        central_failed = [
            [int(z), int(b)]
            for z, b in np.argwhere(central_fallback)
        ]
        replica_failed = [
            [int(z), int(b)]
            for z, b in np.argwhere(replica_fallback > 0.0)
        ]
        fallback_status = np.asarray(
            sumratio.get('central_fallback_status', np.full(
                quality.shape, 'unavailable', dtype='<U80')))
        failure_cells = [
            [int(z), int(b), str(fallback_status[z, b])]
            for z, b in np.argwhere(~quality)
        ]
        payload.update({
            'status': 'strict_pass' if strict else 'quality_gate_failed',
            'strict_gate_passed': strict,
            'reason': (
                'all points pass quality/coverage'
                if strict else
                'quality or replica coverage gate failed; fallback values '
                'remain diagnostic'),
            'quality_points': int(np.count_nonzero(quality)),
            'total_points': int(quality.size),
            'central_fallback_points': int(np.count_nonzero(
                central_fallback)),
            'replica_fallback_mean_fraction': float(
                np.mean(replica_fallback)),
            'replica_fallback_max_fraction': float(
                np.max(replica_fallback)),
            'quality_failed_cells': failed,
            'quality_failure_cells': failure_cells,
            'central_fallback_cells': central_failed,
            'replica_fallback_cells': replica_failed,
            'estimator': 'forward_sumratio_difference_aic',
        })
    path = os.path.join(
        analysis_dir, 'zero_momentum_denominator_status.json')
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    return payload


def load_verified_plateau_inputs(analysis_dir, momenta, logger=print):
    """读取有状态、逐样本、纯纵向正动量 plateau，拒绝伪 fallback。"""
    analysis_dir = os.fspath(analysis_dir)
    verified = {}

    def log(message):
        if logger is not None:
            logger(message)

    for momentum in momenta:
        tag = momentum_tag(momentum)
        pz, py, px = parse_momentum_tag(tag)
        if pz <= 0 or py != 0 or px != 0:
            log(f"  [warn] {tag} is not a positive longitudinal momentum; "
                "PDF input unavailable")
            continue
        data_path = os.path.join(analysis_dir, f'c0_plateau_{tag}.npy')
        status_path = os.path.join(
            analysis_dir, f'c0_plateau_status_{tag}.npz')
        sumratio_path = os.path.join(
            analysis_dir, f'sumratio_difference_{tag}.npz')
        if os.path.isfile(sumratio_path):
            try:
                with np.load(sumratio_path, allow_pickle=False) as artifact:
                    required = (
                        'M', 'M_boot', 'quality_accepted',
                        'estimate_available', 'central_fallback_used')
                    missing = [key for key in required
                               if key not in artifact.files]
                    if missing:
                        raise ValueError(f'missing sumratio fields {missing}')
                    central = np.asarray(artifact['M'])
                    samples = np.asarray(artifact['M_boot'])
                    accepted = np.asarray(artifact['quality_accepted'], bool)
                    available = np.asarray(artifact['estimate_available'], bool)
                    fallback = np.asarray(
                        artifact['central_fallback_used'], bool)
                    replica_fallback = np.asarray(
                        artifact['replica_fallback_fraction'], dtype=float)
                    declared_samples = (
                        int(artifact['Nsample'])
                        if 'Nsample' in artifact.files else None)
                    if (declared_samples is not None
                            and declared_samples != samples.shape[0]):
                        raise ValueError(
                            'sumratio Nsample does not match M_boot axis')
                if (central.ndim != 2 or samples.ndim != 3
                        or samples.shape[1:] != central.shape
                        or samples.shape[0] < 2
                        or not np.isfinite(samples).all()
                        or not np.isfinite(central).all()
                        or not np.all(accepted)
                        or not np.all(available)
                        or np.any(fallback)
                        or np.any(replica_fallback > 0.0)):
                    raise ValueError('sumratio quality/coverage gate failed')
                verified[tag] = {
                    'c0': samples,
                    'c0_central': central,
                    'pz_lattice': pz,
                    'status': 'identifiable',
                    'quality_status': 'good',
                    'physical_status': 'unverified',
                    'estimator': 'forward_sumratio_difference_aic',
                }
                log(f"  {tag}: using strict Sumratio-difference AIC samples")
                continue
            except (OSError, ValueError, KeyError) as exc:
                log(f"  [warn] {tag} sumratio artifact unavailable: {exc}")
        if not os.path.isfile(data_path):
            log(f"  [warn] {tag} plateau data unavailable")
            continue
        if not os.path.isfile(status_path):
            log(f"  [warn] {tag} plateau status unavailable; "
                "numeric-only legacy data is not a verified PDF input")
            continue
        try:
            with np.load(status_path, allow_pickle=False) as metadata:
                if 'plateau_status' not in metadata.files:
                    raise ValueError('missing plateau_status')
                status = str(metadata['plateau_status'])
                quality_status = str(metadata['plateau_fit_quality_status']) \
                    if 'plateau_fit_quality_status' in metadata.files else None
                physical_status = str(
                    metadata['plateau_physical_result_status']) \
                    if 'plateau_physical_result_status' in metadata.files else None
            c0 = np.load(data_path, allow_pickle=False)
        except (OSError, ValueError, KeyError) as exc:
            log(f"  [warn] {tag} plateau artifact unavailable: {exc}")
            continue
        if status != 'identifiable':
            log(f"  [warn] {tag} plateau status={status}; PDF input unavailable")
            continue
        if (quality_status != 'good'
                or physical_status not in ('unverified', 'supported')):
            log(f"  [warn] {tag} plateau quality={quality_status}, "
                f"physical={physical_status}; PDF input unavailable")
            continue
        c0 = np.asarray(c0)
        if (c0.ndim != 3 or c0.shape[0] < 2
                or c0.shape[1] < 1 or c0.shape[2] < 1):
            log(f"  [warn] {tag} plateau sample shape must be "
                f"(Nsample>=2,nz>=1,nb>=1), got {c0.shape}")
            continue
        if not np.isfinite(c0).all():
            log(f"  [warn] {tag} plateau samples are nonfinite")
            continue
        verified[tag] = {
            'c0': c0,
            'pz_lattice': pz,
            'status': status,
            'quality_status': quality_status,
            'physical_status': physical_status,
            'estimator': 'z0_plateau_or_three_parameter_fit',
        }
    return verified


def estimate_proton_energy(corr2, conf_ids, tag, conf, pz_gev):
    """Estimate the nucleon Pt from same-configuration forward C2 data.

    This is a transparent diagnostic estimate, not a full correlated
    two-state spectral fit.  It uses the positive forward ratio
    ``|C2(t+1)|/|C2(t)|`` over a fixed window and falls back to the
    dispersion-mass estimate only when the input is unavailable.
    """
    from pyqcd.renorm import a_len_set

    values = []
    for cid in conf_ids:
        entry = corr2.get(cid, {}) if corr2 is not None else {}
        corr = entry.get(f'corr_pp_{tag}')
        if corr is None:
            continue
        corr = np.asarray(corr, dtype=np.complex128).real
        if corr.ndim == 2:
            corr = np.mean(corr, axis=0)
        corr = np.abs(corr)
        if corr.ndim != 1 or corr.size < 14:
            continue
        ratios = corr[9:14] / np.maximum(corr[8:13], 1e-30)
        finite = np.isfinite(ratios) & (ratios > 0.0)
        if np.any(finite):
            values.append(-np.mean(np.log(ratios[finite])))
    if values:
        e_lattice = float(np.median(values))
        e_gev = e_lattice / float(a_len_set[conf])
        if np.isfinite(e_gev) and e_gev > 0.0:
            return e_gev, 'same_configuration_C2_forward_log_ratio_window_8_12'
    return float(np.sqrt(0.94 ** 2 + pz_gev ** 2)), 'assumed_proton_mass_0.94_GeV'


def run_tmd_pdf_chain(conf_ids, run_dir, logger, args, *, momenta=None,
                      corr2=None):
    """从 c0(z,b,Pz) 走 shape-normalization → quasi-TMD → gg 匹配诊断。

    当前函数只写 ``diagnostic_*`` 文件名，状态明确是 ``diagnostic_only``：
    固定流时、固定 staple 长度、soft/rapidity 未测量、可能只有一个 Pz，
    且没有连续极限。它不会把输出升级为物理 TMD-PDF。
    """
    from pyqcd.renorm import (
        ioffe_time_distribution, tmd_matching_hybrid, flow_scale_mask,
        scheme_c0_coefficient, apply_scheme_c0_correction,
        match_one, pz_to_gev, fm_to_GeV,
        a_len_set, CA,
    )
    an_dir = os.path.join(run_dir, 'analysis', 'tmd_ratio')
    os.makedirs(an_dir, exist_ok=True)
    status_path = os.path.join(an_dir, 'chain_status.json')
    denominator_status = None
    denominator_status_path = os.path.join(
        an_dir, 'zero_momentum_denominator_status.json')
    if os.path.exists(denominator_status_path):
        try:
            with open(denominator_status_path, encoding='utf-8') as stream:
                denominator_status = json.load(stream)
        except (OSError, ValueError, TypeError) as exc:
            logger(f"  [warn] P000 denominator status unavailable: {exc}")

    def write_status(**extra):
        payload = {
            'schema': 'pyqcd.tmd_pdf_chain_status.v1',
            'status': 'diagnostic_only',
            'physical_acceptance': False,
            'workflow': 'gradient-flow gluon TMD-PDF in nucleon',
            'tau': float(args.tau),
            'eps': float(args.eps),
            'bootstrap_samples': int(BOOTSTRAP_SAMPLES),
            'z_list': list(Z_LIST),
            'b_list': list(B_LIST),
            'flow_radius_over_a': float(np.sqrt(8.0 * args.tau)),
        }
        payload.update(extra)
        if denominator_status is not None:
            payload['zero_momentum_denominator'] = denominator_status
            if not denominator_status.get('strict_gate_passed', False):
                payload.setdefault('reasons', [])
                if 'zero_momentum_denominator_quality_gate_failed' \
                        not in payload['reasons']:
                    payload['reasons'].append(
                        'zero_momentum_denominator_quality_gate_failed')
        with open(status_path, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
        return payload

    a_fm = float(a_len_set['L24x72']) * fm_to_GeV
    alpha_s_fixed = 0.25
    pz_scale_fixed = 2.0

    def convert_c0(c0_values):
        """Apply the reference one-loop flowed-OPE endpoint policy."""
        c0_values = np.asarray(c0_values)
        converted = np.empty_like(c0_values, dtype=np.complex128)
        metadata = None
        for sample_index in range(c0_values.shape[0]):
            for b_index in range(c0_values.shape[2]):
                raw = np.zeros(
                    (2, 2, 4, 6, c0_values.shape[1], 1),
                    dtype=np.complex128)
                # Use Mtiti alone; match_one reconstructs the unpolarized
                # combined component from Mtiti-Mijij.
                raw[1, 0, 0, 1, :, 0] = c0_values[sample_index, :, b_index]
                matched_raw, metadata = match_one(
                    raw, args.tau, a_fm, 2.0, alpha_s_fixed,
                    z0_policy='spin2_local', physical_acceptance=False)
                converted[sample_index, :, b_index] = (
                    matched_raw[1, 0, 0, 0, :, 0])
        return converted, metadata

    if momenta is None:
        momenta = MOMENTA_Z
    hR_store = load_verified_plateau_inputs(an_dir, momenta, logger=logger)
    if not hR_store:
        logger("  [warn] 无有状态的 Pz>0 逐样本矩阵元，跳过 PDF 链")
        reasons = ['no_verified_positive_longitudinal_plateau']
        if float(args.eps) > 0.01:
            reasons.append('flow_step_size_above_reference_0p01')
        if not np.all(flow_scale_mask(
                np.asarray(B_LIST, dtype=float), float(args.tau),
                include_local=False)[1:]):
            reasons.append('flow_radius_not_smaller_than_all_nonzero_b')
        write_status(reasons=reasons,
                     n_verified_momenta=0)
        return
    main_tag = next(iter(hR_store))

    conf = 'L24x72'
    pz_lattice = hR_store[main_tag]['pz_lattice']
    pz_gev = pz_to_gev(pz_lattice, conf)
    p_t_gev, p_t_source = estimate_proton_energy(
        corr2, conf_ids, main_tag, conf, pz_gev)
    p_t_lattice = p_t_gev * float(a_len_set[conf])
    z_grid = np.array(Z_LIST) * (a_len_set[conf] * fm_to_GeV)   # fm
    b_grid = np.array(B_LIST) * (a_len_set[conf] * fm_to_GeV)   # fm
    c0_samples = np.asarray(hR_store[main_tag]['c0'])
    hR_samples = None
    n_sample = int(c0_samples.shape[0])
    reasons = []
    normalization = getattr(args, 'normalization', 'local_spin2')
    if n_sample < 2:
        reasons.append('fewer_than_two_same_configuration_samples')
    if hR_store[main_tag].get('estimator') != 'forward_sumratio_difference_aic':
        reasons.append('non_sumratio_estimator')
    if any(np.asarray(item['c0']).shape[0] != n_sample
           for item in hR_store.values()):
        reasons.append('verified_momenta_have_different_sample_counts')
        logger('  [warn] verified momenta do not share a sample count; '
               'cross-momentum CS kernel is not trusted')

    # Convert each sample's z-dependent ratio through the same one-loop
    # flowed-OPE endpoint factors used by the reference.  The z=0 point uses
    # the explicitly selected local spin-2 proxy; it is not an endpoint limit.
    conversion_metadata = None
    normalization_status = 'z0_ratio_shape_diagnostic'
    try:
        converted_c0, conversion_metadata = convert_c0(c0_samples)
        if normalization == 'local_spin2':
            # Reference absolute-amplitude branch: keep the local spin-2
            # conversion and multiply by the tree-level kinematic factor
            # -2/(4*pi*alpha_s*Pt).  Do not divide by c0(z=0).
            alpha_s = alpha_s_fixed
            # R is a lattice matrix element; the reference factor uses the
            # dimensionless aE, not E in GeV.
            kinematic_factor = -2.0 / (
                4.0 * np.pi * alpha_s * p_t_lattice)
            hR_samples = converted_c0 * kinematic_factor
            normalization_status = 'local_spin2_absolute_diagnostic'
            reasons.append('absolute_normalization_one_loop_local_spin2')
        else:
            hR_samples, _ = self_renormalize(converted_c0)
        reasons.append('one_loop_endpoint_conversion_only')
    except (ValueError, FloatingPointError) as exc:
        reasons.append(f'flow_to_msbar_conversion_failed:{exc}')
        logger(f"  [warn] GF-to-MSbar conversion unavailable: {exc}")
    if hR_samples is None or not np.isfinite(hR_samples).all():
        write_status(
            reasons=reasons + ['renormalized_matrix_element_unavailable'],
            n_verified_momenta=len(hR_store), n_samples=n_sample,
            normalization=normalization)
        return

    # Complete signed-x support retains the ξ<0 antiquark branch.
    x_grid_full = np.linspace(-0.98, 0.98, 256)
    # Avoid an exact y=0 interpolation node: the one-loop kernel currently
    # requires avoid-zero grids, while an even node count spans the same support.
    y_grid_full = np.linspace(-3.0, 3.0, 600)
    x_min = 0.02
    physical_x = x_grid_full >= x_min
    x = x_grid_full[physical_x]
    quasi_samples = []
    convolution_samples = []
    scheme_correction_samples = []
    c_scheme = scheme_c0_coefficient(
        z_grid, mu_gev=2.0, zs_fm=0.25)
    for sample in hR_samples:
        _, value = ioffe_time_distribution(
            sample, z_grid, pz_gev=pz_gev, x_grid=x_grid_full,
            lambda_max=z_grid[-1] / fm_to_GeV * pz_gev)
        value = np.asarray(value)
        quasi_samples.append(value[:, None] if value.ndim == 1 else value)
        _, convolution = ioffe_time_distribution(
            sample, z_grid, pz_gev=pz_gev, x_grid=y_grid_full,
            lambda_max=z_grid[-1] / fm_to_GeV * pz_gev)
        convolution = np.asarray(convolution)
        convolution_samples.append(
            convolution[:, None] if convolution.ndim == 1 else convolution)
        _, scheme_x = ioffe_time_distribution(
            sample * c_scheme[:, None], z_grid, pz_gev=pz_gev,
            x_grid=x_grid_full,
            lambda_max=z_grid[-1] / fm_to_GeV * pz_gev)
        scheme_x = np.asarray(scheme_x)
        scheme_correction_samples.append(
            scheme_x[:, None] if scheme_x.ndim == 1 else scheme_x)
    quasi_samples = np.stack(quasi_samples, axis=0)
    convolution_samples = np.stack(convolution_samples, axis=0)
    scheme_correction_samples = np.stack(scheme_correction_samples, axis=0)
    quasi_full = quasi_samples.mean(axis=0)
    quasi_err_full = (np.std(quasi_samples, axis=0, ddof=1)
                      if n_sample > 1 else np.full_like(quasi_full, np.nan))
    scheme_coupling = alpha_s_fixed * CA / (2.0 * np.pi)
    quasi_hybrid_samples = apply_scheme_c0_correction(
        quasi_samples, scheme_correction_samples, alpha_s_fixed, ca=CA)
    quasi_hybrid_full = quasi_hybrid_samples.mean(axis=0)
    quasi_hybrid_err_full = (
        np.std(quasi_hybrid_samples, axis=0, ddof=1)
        if n_sample > 1 else np.full_like(quasi_hybrid_full, np.nan))
    xg = quasi_full[physical_x]
    xg_hybrid = quasi_hybrid_full[physical_x]
    xg_hybrid_err = quasi_hybrid_err_full[physical_x]
    xg_err = quasi_err_full[physical_x]

    flow_mask = flow_scale_mask(
        np.asarray(Z_LIST, dtype=float), float(args.tau), include_local=True)
    eligible_z_refs = np.flatnonzero(flow_mask & (np.asarray(Z_LIST) > 0))

    K_samples = None
    cs_conversion_metadata = None
    cs_z_ref = None
    if len(hR_store) >= 2:
        tags2 = list(hR_store)[:2]
        p1 = hR_store[tags2[0]]['pz_lattice']
        p2 = hR_store[tags2[1]]['pz_lattice']
        c01 = np.asarray(hR_store[tags2[0]]['c0'])
        c02 = np.asarray(hR_store[tags2[1]]['c0'])
        if p1 != p2 and c01.shape[0] == c02.shape[0] == n_sample:
            from pyqcd.renorm import cs_kernel_two_momentum
            try:
                converted1, cs_conversion_metadata = convert_c0(c01)
                converted2, _ = convert_c0(c02)
                p1_gev = pz_to_gev(p1, conf)
                p2_gev = pz_to_gev(p2, conf)
                e1_gev, _ = estimate_proton_energy(
                    corr2, conf_ids, tags2[0], conf, p1_gev)
                e2_gev, _ = estimate_proton_energy(
                    corr2, conf_ids, tags2[1], conf, p2_gev)
                e1_lattice = e1_gev * float(a_len_set[conf])
                e2_lattice = e2_gev * float(a_len_set[conf])
                h1 = converted1 * (-2.0 / (
                    4.0 * np.pi * alpha_s_fixed * e1_lattice))
                h2 = converted2 * (-2.0 / (
                    4.0 * np.pi * alpha_s_fixed * e2_lattice))
                if eligible_z_refs.size == 0:
                    raise ValueError(
                        "no nonzero z satisfies sqrt(8*tau)<z for CS extraction")
                z_ref = int(eligible_z_refs[0])
                cs_z_ref = z_ref
                K_samples = np.stack([
                    cs_kernel_two_momentum(
                        h1[i], h2[i], p1_gev, p2_gev, z_ref=z_ref)
                    for i in range(n_sample)
                ], axis=0)
            except (ValueError, FloatingPointError) as exc:
                reasons.append(f'cs_kernel_invalid:{exc}')
                logger(f"  [warn] CS kernel unavailable: {exc}")
    if len(hR_store) < 2:
        reasons.append('fewer_than_two_positive_longitudinal_momenta')
    if K_samples is None:
        reasons.append('cs_kernel_not_measured')
    if p_t_source.startswith('assumed_'):
        reasons.append('proton_energy_assumed_not_measured')
    reasons.extend([
        'soft_rapidity_not_measured',
        'reference_rapidity_scale_fixed_2gev_proxy',
        'single_fixed_flow_time_no_tau_extrapolation',
        'single_lattice_spacing_no_continuum_limit',
        'fixed_staple_length_no_ell_extrapolation',
        'fourier_finite_lambda_support_no_tail_model',
        'real_input_even_x_branch_only',
        'x_abs_lt_0.02_masked',
        'gg_only_no_gluon_quark_mixing',
    ])
    if float(args.eps) > 0.01:
        reasons.append('flow_step_size_above_reference_0p01')
    if not np.any(np.asarray(B_LIST) > 0):
        reasons.append('no_nonzero_transverse_separation')
    if not np.all(flow_mask[1:]):
        reasons.append('flow_radius_not_smaller_than_all_nonzero_z')
    if not np.all(flow_scale_mask(
            np.asarray(B_LIST, dtype=float), float(args.tau),
            include_local=False)[1:]):
        reasons.append('flow_radius_not_smaller_than_all_nonzero_b')

    matched_samples = []
    matching_metadata = None
    try:
        for sample_index in range(n_sample):
            K_sample = (0.0 if K_samples is None
                        else K_samples[sample_index])
            _, matched, metadata = tmd_matching_hybrid(
                x_grid_full, y_grid=y_grid_full, b_perp=b_grid, mu=2.0,
                pz_gev=pz_gev, pz_scale=pz_scale_fixed,
                cs_kernel=K_sample, soft_factor=1.0,
                x_tmd=convolution_samples[sample_index], lambda_s_fm=0.3,
                return_metadata=True, alpha_s=alpha_s_fixed)
            matched = apply_scheme_c0_correction(
                np.asarray(matched), scheme_correction_samples[sample_index],
                alpha_s_fixed, ca=CA)
            matched_samples.append(matched[:, None]
                                   if matched.ndim == 1 else matched)
            matching_metadata = metadata
    except (ValueError, FloatingPointError) as exc:
        reasons.append(f'matching_invalid:{exc}')
        logger(f"  [warn] matching unavailable: {exc}")
    if matched_samples:
        matched_samples = np.stack(matched_samples, axis=0)
        matched_full = matched_samples.mean(axis=0)
        matched_err_full = (np.std(matched_samples, axis=0, ddof=1)
                            if n_sample > 1 else np.full_like(matched_full, np.nan))
    else:
        matched_full = np.full_like(quasi_full, np.nan)
        matched_err_full = np.full_like(quasi_full, np.nan)
    xg_match = matched_full[physical_x]
    xg_match_err = matched_err_full[physical_x]
    K_mean = None if K_samples is None else K_samples.mean(axis=0)
    K_err = (None if K_samples is None or n_sample < 2
             else np.std(K_samples, axis=0, ddof=1))

    np.save(os.path.join(an_dir, 'diagnostic_tmd_pdf_x.npy'), x)
    np.save(os.path.join(an_dir, 'diagnostic_quasi_tmd_pdf_xg.npy'), xg)
    np.save(os.path.join(
        an_dir, 'diagnostic_quasi_hybrid_tmd_pdf_xg.npy'), xg_hybrid)
    np.save(os.path.join(
        an_dir, 'diagnostic_quasi_hybrid_tmd_pdf_xg_err.npy'),
        xg_hybrid_err)
    np.save(os.path.join(
        an_dir, 'diagnostic_quasi_tmd_pdf_xg_err.npy'), xg_err)
    np.save(os.path.join(
        an_dir, 'diagnostic_one_loop_gg_tmd_pdf_xg.npy'), xg_match)
    np.save(os.path.join(
        an_dir, 'diagnostic_one_loop_gg_tmd_pdf_xg_err.npy'), xg_match_err)
    if K_mean is not None:
        np.save(os.path.join(an_dir, 'cs_kernel_b.npy'), K_mean)
        if K_err is not None:
            np.save(os.path.join(an_dir, 'cs_kernel_b_err.npy'), K_err)
        np.save(os.path.join(an_dir, 'cs_kernel_bgrid.npy'), b_grid)
    with open(os.path.join(an_dir, 'matching_status.json'), 'w',
              encoding='utf-8') as stream:
        if matching_metadata is not None:
            matching_metadata['scheme_c0'] = {
                'status': 'perturbative_leading_twist_bridge_proxy',
                'zs_fm': 0.25,
                'mu_gev': 2.0,
                'alpha_s': alpha_s_fixed,
                'c0': np.asarray(c_scheme).tolist(),
            }
        json.dump(matching_metadata or {
            'status': 'matching_not_available',
            'physical_acceptance': False,
            'light_cone_matching': 'not_applied',
        }, stream, indent=2)
        stream.write('\n')
    np.save(os.path.join(an_dir, 'b_grid_fm.npy'), b_grid)
    np.save(os.path.join(an_dir, 'z_grid_fm.npy'), z_grid)
    np.savez(os.path.join(an_dir, 'pdf_chain_input_status.npz'),
             momentum_tag=np.asarray(main_tag),
             pz_lattice=np.asarray(pz_lattice, dtype=np.int64),
             input_status=np.asarray(hR_store[main_tag]['status']),
             input_estimator=np.asarray(hR_store[main_tag].get(
                 'estimator', 'z0_plateau_or_three_parameter_fit')),
             n_verified_momenta=np.asarray(len(hR_store), dtype=np.int64),
             n_samples=np.asarray(n_sample, dtype=np.int64),
             x_min=np.asarray(x_min, dtype=np.float64))
    status = write_status(
        reasons=reasons,
        n_verified_momenta=len(hR_store),
        n_samples=n_sample,
        main_momentum=main_tag,
        pz_gev=float(pz_gev),
        p_t_gev=float(p_t_gev),
        p_t_lattice=float(p_t_lattice),
        p_t_source=p_t_source,
        input_estimator=hR_store[main_tag].get(
            'estimator', 'z0_plateau_or_three_parameter_fit'),
        normalization=normalization,
        normalization_status=normalization_status,
        flow_window_mask=flow_mask.tolist(),
        flow_window_all_nonzero_z=bool(np.all(flow_mask[1:])),
        cs_kernel_samples=bool(K_samples is not None),
        cs_kernel_z_ref=cs_z_ref,
        flow_to_msbar=conversion_metadata,
        cs_flow_to_msbar=cs_conversion_metadata,
        soft_factor=1.0,
        scheme_c0_coefficient=np.asarray(c_scheme),
        scheme_c0_coupling=float(scheme_coupling),
        scheme_c0_status='perturbative_leading_twist_bridge_proxy',
        matching_status=(matching_metadata or {}).get('status'),
        convolution_x_support=[float(x_grid_full[0]), float(x_grid_full[-1])],
        convolution_y_support=[float(y_grid_full[0]), float(y_grid_full[-1])],
        x_min_abs=float(x_min),
        even_x_only=True,
        fixed_alpha_s=alpha_s_fixed,
        reference_rapidity_pz_gev=pz_scale_fixed,
        lattice_spacing_fm=a_fm,
        quasi_error_saved=True,
        matched_error_saved=True,
    )
    from pyqcd.analysis import plot_tmd_pdf
    plot_tmd_pdf(
        x, xg_hybrid, xg_match, b_grid, K_mean, main_tag, an_dir, logger,
        diagnostic=True)
    logger(f"  chain status: {status['status']} (physical_acceptance=False)")


def write_summary(run_dir, conf_ids, momenta, logger, args=None):
    summary = {
        'workflow': 'gradient-flow gluon TMD-PDF in nucleon',
        'conf_ids': conf_ids,
        'momenta': [list(m) for m in momenta],
        'z_list': Z_LIST, 'b_list': B_LIST,
        'tau': TAU if args is None else getattr(args, 'tau', TAU),
        'eps': EPS if args is None else getattr(args, 'eps', EPS),
        'backend': get_backend_name(),
        'time': datetime.now().isoformat(),
        'physical_acceptance': False,
        'status': 'diagnostic_only',
    }
    an_dir = os.path.join(run_dir, 'analysis', 'tmd_ratio')
    status_path = os.path.join(an_dir, 'chain_status.json')
    if os.path.exists(status_path):
        with open(status_path, encoding='utf-8') as stream:
            chain_status = json.load(stream)
        summary['chain_status'] = chain_status.get('status', 'diagnostic_only')
        summary['physical_acceptance'] = bool(
            chain_status.get('physical_acceptance', False))
        summary['chain_reasons'] = chain_status.get('reasons', [])
        summary['bootstrap_samples'] = chain_status.get(
            'bootstrap_samples', BOOTSTRAP_SAMPLES)
    denominator_path = os.path.join(
        an_dir, 'zero_momentum_denominator_status.json')
    if os.path.exists(denominator_path):
        with open(denominator_path, encoding='utf-8') as stream:
            summary['zero_momentum_denominator'] = json.load(stream)
    for f in os.listdir(an_dir):
        if f.endswith('.npy'):
            try:
                arr = np.load(os.path.join(an_dir, f))
                summary[f'shape_{f[:-4]}'] = list(arr.shape)
            except Exception:
                pass
    with open(os.path.join(run_dir, 'analysis', 'tmd_summary.json'), 'w') as fp:
        json.dump(summary, fp, indent=2, default=str)
    logger(f"  Summary -> {os.path.join(run_dir, 'analysis', 'tmd_summary.json')}")


if __name__ == '__main__':
    main()
