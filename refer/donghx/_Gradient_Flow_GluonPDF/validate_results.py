#!/usr/bin/env python3
"""Independently read back every stored correlator and benchmark time."""
from pathlib import Path
import argparse
import json
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('result',type=Path);a=p.parse_args();root=a.result
    meta=json.loads((root/'summary.json').read_text());assert meta['status']=='PASS'
    s=meta['settings'];assert s['conf']=='10000' and s['tsep']==5 and s['dt_list']==list(range(6))
    checks=[]
    for name,shape in [('C2',(4,3))]+[(f'C3_dt{i}',(2,2,4,3,21,4,4)) for i in range(6)]:
        with np.load(root/(name+'.npz'),allow_pickle=False) as d:
            ref=d['reference'];got=d['spin2'];assert ref.shape==got.shape==shape
            assert np.isfinite(ref).all() and np.isfinite(got).all()
            diff=abs(got-ref)
            rel=np.linalg.norm(diff.ravel())/np.linalg.norm(ref.ravel())
            peak=diff.max()/abs(ref).max();assert max(rel,peak)<2e-10
            if name!='C2':
                # Per (flavor, sign, pol, momentum): avoid domination by nopol.
                rr=np.linalg.norm(diff.reshape(2,2,4,3,-1),axis=-1)/np.maximum(
                    np.linalg.norm(ref.reshape(2,2,4,3,-1),axis=-1),1e-300)
                assert rr.max()<2e-10
            else:rr=np.array([rel])
            checks.append(dict(name=name,max_abs=float(diff.max()),relative_L2=float(rel),
                               relative_peak=float(peak),max_channel_relative_L2=float(rr.max())))
    for row in meta['records']:
        for method in ['reference','spin2']:
            values=row['timing_seconds'][method]
            assert len(values)==s['repeats'] and np.isfinite(values).all() and min(values)>0
            assert np.median(values)==row['median_seconds'][method]
    total_points=12+6*2*2*4*3*21*4*4
    report=dict(status='PASS',complex_numbers_compared=total_points,records=checks,
                timed_repetitions_verified=True,no_new_physical_estimator=True)
    (root/'readback_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    rows=[]
    for r in meta['records']:
        rows.append(f"| {r['label']} | {r['median_seconds']['reference']:.4f} | {r['median_seconds']['spin2']:.4f} | {r['speedup']:.3f} | {r['correctness']['relative_L2']:.2e} |")
    text=f'''# Conf 10000, tsep=5 实测结果

任务 {s['job']} 在 {s['gpu']}（{s['host']}）完成。source={s['source']}，sink={s['sink']}，flow={s['flow_time']}；Nev_src=200/Nev_current=400；三个动量、两种 Wilson-line 符号、z=0..20、四种极化、u/d 和全部 16 个电流 spin 分量。

两版采用相同 momentum_batch={s['momentum_batch']}、optimal 路径规划和 complex128 收缩。每版预热后交替测量 {s['repeats']} 次，GPU 计时前后同步；下表为中位数，单位秒。降维投影和相关中间量的创建包含在 spin2 耗时内。

| 项目 | 原始版 /s | 降维版 /s | 原始/降维 | 相对 L2 误差 |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(rows)+f'''

上述收缩中位耗时之和：原始 {meta['contraction_median_sum_reference']:.4f} s，降维 {meta['contraction_median_sum_spin2']:.4f} s，速度比 **{meta['contraction_speedup']:.3f}**。

本次实测的公共输入准备和 H 构建共 {meta['one_pass_measured_common_setup_seconds']:.3f} s。将其与各版一次收缩相加，速度比为 **{meta['common_setup_plus_contractions_speedup']:.3f}**。这只是本次单源测试的公共开销估计，包含两个 VVV 时刻重建和共享文件系统读取，不是完整生产 conf 的端到端加速。

独立读回 {total_points} 个复数通过，最大相对 L2={max(r['relative_L2'] for r in checks):.3e}，逐 flavor/sign/pol/momentum 最大相对 L2={max(r['max_channel_relative_L2'] for r in checks):.3e}。见 [readback_validation.json](readback_validation.json)、[summary.json](summary.json) 和 `C2.npz/C3_dt*.npz`。

实际计算保持原始时间边界符号；此源汇不绕回，但完整 flow window 含跨边界的 tau。尚未用真实绕回源汇、六方向和全部 96 个 source 做生产回归。本测试不改变拟合、ringed normalization、matching 或任何已有物理结果。

生成日期：2026-09-22。标题：外部正宇称降维实数据正确性与性能。
'''
    (root/'REPORT.md').write_text(text)

if __name__=='__main__':main()
