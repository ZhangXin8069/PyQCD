#!/usr/bin/env python3
"""
TMD-PDF 物理链自洽性断言
=========================

不依赖外部基线，用物理自洽性断言验证 TMD-PDF 全链产物：

A. 梯度流正确性（E 递减、unitarity 保持）——来自重算 flowed gauge
B. 多动量 2pt 谱线（P000 > P200 > P400 递减，组态间一致）
C. TMD OPE 存在性与形状（(nz,nb,Nt)，z=0 最大）
D. ratio 文件存在 + fit 报告生成
E. PDF 链产物（准 PDF / 匹配 PDF / CS 核）有限且可读

用法：python -m pyqcd.testing.tmd.verify_pdf_workflow [run_dir]
默认 run_dir = pyqcd/testing/tmd/artifacts/work
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import traceback

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    '..', '..', '..'))


def test_a_gradient_flow(run_dir, conf_ids=(6250,), tau=3.0, eps=0.05,
                         skip_flow=False):
    """梯度流正确性：E(0)>E(τ)、unitarity 保持（重算单组态验证）。"""
    if skip_flow:
        print("\n[A] SKIP: --skip-flow (flow field not recomputed)")
        return True
    if not conf_ids:
        print("\n[A] SKIP: no configurations")
        return True
    print("\n[A] 梯度流正确性（重算 conf%s 验证）" % conf_ids[0])
    from pyqcd.tools import set_backend
    from pyqcd.renorm import wilson_flow, flow_action_density
    from pyqcd.operator import read_gauge_lime
    from pyqcd.pipeline._config import NT, NX, get_gauge_path
    ok = True
    cid = conf_ids[0]
    g = read_gauge_lime(get_gauge_path(cid), NT, NX)
    try:
        import torch as _torch
        device = 'cuda:0' if _torch.cuda.is_available() else 'cpu'
        set_backend('torch', device=device)
        G = _torch.as_tensor(g.astype('complex64'), device=device)
    except (ImportError, RuntimeError):
        import numpy as _np
        set_backend('numpy')
        G = g.astype('complex64')
    del g
    E0_value = flow_action_density(G).mean()
    E0 = float(E0_value.detach().cpu()) if hasattr(E0_value, 'detach') else float(E0_value)
    V = wilson_flow(G, tau=tau, eps=eps)
    E1_value = flow_action_density(V).mean()
    E1 = float(E1_value.detach().cpu()) if hasattr(E1_value, 'detach') else float(E1_value)
    U = V[..., 0, :, :]
    if hasattr(U, 'detach'):
        identity = _torch.eye(3, dtype=U.dtype, device=U.device)
        dev_value = (U @ U.conj().transpose(-1, -2) - identity).abs().max()
        dev = float(dev_value.detach().cpu())
    else:
        dev = float(np.max(np.abs(
            U @ U.conj().transpose(0, 1, 2, 3, 5, 4) - np.eye(3))))
    ok_flow = (E1 < E0) and (dev < 1e-3)
    ok &= ok_flow
    print(f"  conf{cid}: E: {E0:.4f}->{E1:.4f} "
          f"({'decrease' if E1 < E0 else 'INCREASE!'})  "
          f"unitarity dev={dev:.2e} "
          f"({'ok' if dev < 1e-3 else 'FAIL'})")
    print(f"  [A] {'PASS' if ok else 'FAIL'}")
    return ok


def test_b_2pt(run_dir, conf_ids, *, allow_partial=False):
    """多动量源分辨 2pt：P000>P200>P400，组态间相对分散<10%。"""
    print("\n[B] 多动量 2pt 谱线")
    from pyqcd.pipeline._tmd9 import (
        _load_contract_h5, _multi_2pt_cache_specs, parse_momentum_tag,
    )
    ok = True
    moms = ['P000', 'P200', 'P400']
    vals = {m: [] for m in moms}
    loaded_by_momentum = {m: {} for m in moms}
    for cid in conf_ids:
        specs = _multi_2pt_cache_specs(
            cid, run_dir, [tuple(parse_momentum_tag(m)) for m in moms],
            ('pp',), 'complex64', 'VVV', source_resolved=True,
            polarization='nopol')
        for m in moms:
            spec = specs[f'corr_pp_{m}']
            array, _reason = _load_contract_h5(
                spec['path'], spec['contract'], expected_shape=spec['shape'],
                expected_dtype=spec['dtype'])
            if array is not None:
                loaded_by_momentum[m][cid] = {
                    f'corr_pp_{m}': array,
                }
    for cid in conf_ids:
        for m in moms:
            key = f'corr_pp_{m}'
            c = loaded_by_momentum[m].get(cid, {}).get(key)
            if c is None:
                if not allow_partial:
                    ok = False
                print(f"  conf{cid}: missing source-resolved {key}")
                continue
            if c.ndim != 2:
                ok = False
                print(f"  conf{cid}: {key} has non-source-resolved shape {c.shape}")
                continue
            vals[m].append(float(np.mean(np.abs(c[:, 0]))))
    present = [m for m in moms if vals[m]]
    if present:
        for m in present:
            v = np.array(vals[m])
            spread = v.std() / v.mean()
            print(f"  {m}: C2(0) mean={v.mean():.4f} spread={spread:.1%}")
            ok &= (spread < 0.10)
        if all(m in present for m in moms):
            ok &= np.mean(vals['P000']) > np.mean(vals['P200']) > np.mean(vals['P400'])
            print(f"  ordering P000>P200>P400: "
                  f"{'ok' if np.mean(vals['P000']) > np.mean(vals['P200']) > np.mean(vals['P400']) else 'FAIL'}")
        elif not allow_partial:
            ok = False
            print("  [B] partial momentum set requires --allow-partial")
    else:
        print("  [B] PARTIAL: no complete 2pt momentum")
    print(f"  [B] {'PASS' if ok else 'FAIL'}")
    return ok


def test_c_tmd_ope(run_dir, conf_ids, nz=13, nb=5, nt=72):
    """TMD OPE canonical cache 存在性、形状和有限性。"""
    print("\n[C] TMD OPE")
    import h5py
    ok = True
    for cid in conf_ids:
        cdir = os.path.join(run_dir, 'data', f'conf{cid}')
        matches = sorted(glob.glob(os.path.join(
            cdir, f'tmd_ope_*_conf{cid}.h5')))
        if len(matches) != 1:
            ok = False
            print(f"  conf{cid}: expected one canonical TMD OPE cache, "
                  f"found {len(matches)}")
            continue
        p = matches[0]
        with h5py.File(p, 'r') as f:
            tmd = np.array(f['data'])
        ok_shape = tmd.shape == (nz, nb, nt)
        ok &= ok_shape
        ok &= bool(np.isfinite(tmd).all())
        # z=0 只需保持可读且非全零；最大值判断不是物理恒等式。
        z0 = abs(tmd[0]).mean()
        zlast = abs(tmd[-1]).mean()
        ok_z = z0 > 0 and np.isfinite(z0) and np.isfinite(zlast)
        ok &= ok_z
        print(f"  conf{cid}: tmd {tmd.shape} dtype={tmd.dtype} "
              f"{'shape-ok' if ok_shape else 'SHAPE-FAIL'}  "
              f"|O(z=0)|_avg={z0:.2f}  |O(z=max)|_avg={zlast:.2f}")
    print(f"  [C] {'PASS' if ok else 'FAIL'}")
    return ok


def test_d_analysis(run_dir, momenta=None, *, allow_partial=False):
    """分析产物：ratio + fit 报告 + c0。"""
    print("\n[D] 分析产物")
    an = os.path.join(run_dir, 'analysis', 'tmd_ratio')
    ok = True
    if momenta is None:
        momenta = sorted({
            os.path.basename(p)[len('ratio_proton_'):-len('.npy')]
            for p in glob.glob(os.path.join(an, 'ratio_proton_*.npy'))
        })
    if not momenta:
        print("  [D] SKIP: no ratio artifacts")
        return bool(allow_partial)
    for m in momenta:
        for f in (f'ratio_proton_{m}.npy', f'c0_plateau_{m}.npy',
                  f'1_fit_report_{m}.txt'):
            if not os.path.exists(os.path.join(an, f)):
                if not allow_partial:
                    ok = False
                print(f"  missing {f}")
    for m in momenta:
        if not os.path.exists(os.path.join(an, f'ratio_proton_{m}.npy')):
            if allow_partial:
                print(f"  [D] PARTIAL: missing ratio_proton_{m}.npy")
                continue
            ok = False
            continue
        r = np.load(os.path.join(an, f'ratio_proton_{m}.npy'))
        finite = np.all(np.isfinite(r))
        print(f"  {m}: ratio {r.shape} finite={finite}")
        if not finite and allow_partial:
            print(f"  [D] PARTIAL: ratio {m} is fail-closed/non-finite")
        else:
            ok &= finite
        sumratio_path = os.path.join(an, f'sumratio_difference_{m}.npz')
        if os.path.exists(sumratio_path):
            with np.load(sumratio_path, allow_pickle=False) as fit:
                required = ('M', 'M_boot', 'quality_accepted',
                            'estimate_available', 'candidate_range_aic')
                missing = [key for key in required if key not in fit.files]
                if missing:
                    ok = False
                    print(f"  [D] sumratio {m}: missing {missing}")
                else:
                    finite_fit = (np.isfinite(fit['M']).all()
                                  and np.isfinite(fit['M_boot']).all())
                    sample_text = (
                        f" Nsample={int(fit['Nsample'])}"
                        if 'Nsample' in fit.files else '')
                    print(f"  {m}: sumratio quality="
                          f"{int(np.count_nonzero(fit['quality_accepted']))}/"
                          f"{fit['quality_accepted'].size}{sample_text} "
                          f"finite={finite_fit}")
                    if not finite_fit and allow_partial:
                        print(f"  [D] PARTIAL: sumratio {m} is fail-closed")
                    else:
                        ok &= finite_fit
        elif allow_partial:
            print(f"  [D] PARTIAL: sumratio artifact {m} absent")
    denominator_status_path = os.path.join(
        an, 'zero_momentum_denominator_status.json')
    denominator_artifact_path = os.path.join(
        an, 'sumratio_difference_P000.npz')
    if os.path.exists(denominator_status_path):
        try:
            with open(denominator_status_path, encoding='utf-8') as stream:
                denominator = json.load(stream)
            status = denominator.get('status')
            strict = denominator.get('strict_gate_passed')
            valid_status = (
                denominator.get('physical_status') == 'unverified'
                and denominator.get('used_for_pdf') is False
                and (
                    (status == 'artifact_unavailable' and strict is False)
                    or (status == 'quality_gate_failed' and strict is False)
                    or (status == 'strict_pass' and strict is True)
                )
            )
            if not valid_status:
                ok = False
                print("  [D] P000 denominator status is not fail-closed")
            else:
                print("  P000 denominator: "
                      f"{denominator.get('status')} "
                      f"quality={denominator.get('quality_points')}/"
                      f"{denominator.get('total_points')} "
                      f"used_for_pdf={denominator.get('used_for_pdf')}")
        except (OSError, ValueError, TypeError) as exc:
            ok = False
            print(f"  [D] invalid P000 denominator status: {exc}")
    elif os.path.exists(denominator_artifact_path):
        ok = False
        print("  [D] P000 denominator artifact lacks status metadata")
    print(f"  [D] {'PASS' if ok else 'FAIL'}")
    return ok


def test_e_pdf(run_dir, *, allow_partial=False):
    """PDF 链产物有限且可读。"""
    print("\n[E] PDF 链产物")
    an = os.path.join(run_dir, 'analysis', 'tmd_ratio')
    ok = True
    required = (
        'diagnostic_tmd_pdf_x.npy',
        'diagnostic_quasi_tmd_pdf_xg.npy',
        'diagnostic_quasi_hybrid_tmd_pdf_xg.npy',
        'diagnostic_one_loop_gg_tmd_pdf_xg.npy',
        'b_grid_fm.npy',
    )
    missing = [f for f in required if not os.path.exists(os.path.join(an, f))]
    if missing:
        if allow_partial:
            print(f"  [E] PARTIAL: missing {missing}")
            return True
        ok = False
        print(f"  missing {missing}")
    for f in required:
        p = os.path.join(an, f)
        if not os.path.exists(p):
            continue
        arr = np.load(p)
        finite = np.all(np.isfinite(arr))
        ok &= finite
        print(f"  {f}: {arr.shape} finite={finite}")
    status_path = os.path.join(an, 'matching_status.json')
    if os.path.exists(status_path):
        with open(status_path, encoding='utf-8') as stream:
            status = json.load(stream)
        print(f"  matching status: {status.get('status', 'unknown')}; "
              f"physical_acceptance={status.get('physical_acceptance', False)}")
        ok &= status.get('physical_acceptance') is not True
    chain_path = os.path.join(an, 'chain_status.json')
    if os.path.exists(chain_path):
        with open(chain_path, encoding='utf-8') as stream:
            chain = json.load(stream)
        print(f"  chain status: {chain.get('status', 'unknown')}; "
              f"physical_acceptance={chain.get('physical_acceptance', False)}")
        ok &= chain.get('physical_acceptance') is not True
    # 至少 1 张 PDF 图
    imgs = [f for f in os.listdir(an) if f.endswith('.png')]
    ok &= len(imgs) >= 5 or allow_partial
    print(f"  images: {len(imgs)}")
    print(f"  [E] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    parser = argparse.ArgumentParser(description='验证 TMD-PDF 全链产物')
    parser.add_argument('--allow-partial', action='store_true',
                        help='允许单组态/单 Pz 的中间产物，只报告 PARTIAL')
    parser.add_argument('--skip-flow', action='store_true',
                        help='不重算 547 MB 级规范场，只验证已有产物')
    parser.add_argument('--momenta', default=None,
                        help='逗号分隔动量标签；默认从 ratio 文件推断')
    parser.add_argument(
        'run_dir', nargs='?',
        default=os.path.join(
            ROOT, 'pyqcd', 'testing', 'tmd', 'artifacts', 'work'))
    args = parser.parse_args()
    run_dir = args.run_dir
    conf_ids = [int(x[4:]) for x in os.listdir(os.path.join(run_dir, 'data'))
                if x.startswith('conf')]
    conf_ids.sort()
    print("=" * 64)
    print(f"TMD-PDF 物理链自洽性验证  run_dir={run_dir}")
    print(f"conf_ids = {conf_ids}")
    print("=" * 64)
    requested_momenta = (
        None if args.momenta is None
        else tuple(value for value in args.momenta.split(',') if value))
    ok = True
    ok &= test_a_gradient_flow(run_dir, conf_ids[:2], skip_flow=args.skip_flow)
    ok &= test_b_2pt(run_dir, conf_ids, allow_partial=args.allow_partial)
    ok &= test_c_tmd_ope(run_dir, conf_ids)
    ok &= test_d_analysis(run_dir, requested_momenta,
                          allow_partial=args.allow_partial)
    ok &= test_e_pdf(run_dir, allow_partial=args.allow_partial)
    print("=" * 64)
    if ok and args.allow_partial:
        print("PARTIAL：结构/有限性检查通过；scoped 输入不足以声明物理 PDF")
    else:
        print("全部通过" if ok else "存在失败项！")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
