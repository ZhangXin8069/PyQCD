"""Interpret a completed, hash-verified audit without statistical claims."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import core as c

ROOT=Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def main():
    assert os.environ.get('SLURM_JOB_ID')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,digest in read(ROOT/'source_lock.json')['sha256'].items():
        assert c.sha(ROOT/name)==digest
    receipt=read(ROOT/'complete.json')
    assert receipt['status']=='PASS_MULTISTART_NUMERICAL_AUDIT'
    for name,digest in receipt['artifacts'].items():
        assert c.sha(ROOT/name)==digest,name
    audit,req=read(ROOT/'result_audit.json'),read(ROOT/'request.json')
    assert not audit['errors'] and len(audit['rows'])==8
    fits=audit['diagnostic_fits']
    report=['# FFT 多起点结果解读','',
            '全部32次真实组态规范固定完成，并通过严格theta<=1e-12及数值回读审计。',
            '每个ensemble仅一个组态；下表d为无权最小二乘的单组态诊断值，不附统计误差。','',
            '|ensemble|表示|长度窗口n|原起点|best2|best3|best4|best4减原起点|best4减best3|',
            '|---|---|---|---:|---:|---:|---:|---:|---:|']
    changes=[]
    for ens in req['ensembles']:
        for rep in req['diagnostic']['representations']:
            for lo,hi in req['diagnostic']['windows_a']:
                matching=[r for r in fits if r['ensemble']==ens['name'] and r['representation']==rep
                          and r['lo']==lo and r['hi']==hi and r['length_reference_a']==0]
                by_scheme={r['scheme']:r for r in matching}
                fields=['start0','best2','best3','best4']
                vals=[by_scheme[k].get('d_diag') for k in fields]
                if all(v is not None for v in vals):
                    delta=vals[3]-vals[0]
                    delta34=vals[3]-vals[2]
                    row=dict(ensemble=ens['name'],representation=rep,lo=lo,hi=hi,
                             original=vals[0],best2=vals[1],best3=vals[2],best4=vals[3],
                             best4_minus_original=delta,best4_minus_best3=delta34,
                             best4_fit_residual_rms=by_scheme['best4']['residual_rms'])
                    changes.append(row)
                    report.append(f"|{ens['name']}|{rep}|{lo}–{hi}|"+'|'.join(f'{v:.8f}' for v in vals)+f'|{delta:.8f}|{delta34:.8f}|')
                else:
                    report.append(f"|{ens['name']}|{rep}|{lo}–{hi}|INVALID|INVALID|INVALID|INVALID|||" )
    report+=['','|ensemble|flow index|tau|best1/2/3/4 selected start|', '|---|---:|---:|---|']
    for row in audit['rows']:
        report.append(f"|{row['ensemble']}|{row['task']%4}|{row['tau']:.8f}|{row['selected']}|")
    report+=['','选择规则是每个flow独立取前K个起点中规范泛函最大的严格收敛结果；所有长度和表示共用选择。',
        '有限候选内最大不等于绝对Landau规范，最后一个候选未改变结果也不证明更大的起点集合已饱和。',
        '全部4^4种跨flow候选组合的诊断范围见copy_envelope.csv；不是置信区间。',
        '基本表示/伴随表示分别计算，伴随量来自逐点局域字符。两个ensemble匹配了物理flow半径，',
        '但相同n窗口对应不同物理长度：L64的a=.0897fm，L96的a=.0775fm；本轮不把跨ensemble差异直接归因于格距效应。',
        '窗口和flow离散效应仍未控制，',
        '这里不进行small-flow-time或连续极限外推，也不施加矩阵元减除。',
        '', 'open_line_copy_bands.pdf显示原起点、best4和四候选包络；ratio_copy_comparison.pdf显示相同选择下的double ratio。']
    line_rows=list(csv.DictReader((ROOT/'copy_lines.csv').open()))
    ratio_rows=list(csv.DictReader((ROOT/'ratios.csv').open()))
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    ratio_fig,rax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ei,ens in enumerate(req['ensembles']):
        selection=[r for r in audit['rows'] if r['ensemble']==ens['name']]
        selected={r['task']%4:r['selected']['4'] for r in selection}
        for ri,rep in enumerate(req['diagnostic']['representations']):
            values=np.full((4,4,13),np.nan)
            for row in line_rows:
                if row['ensemble']==ens['name'] and row['representation']==rep:
                    values[int(row['flow_index']),int(row['start']),int(row['length_a'])]=float(row['value'])
            assert np.isfinite(values).all()
            for fi in range(4):
                color=f'C{fi}'
                axes[ei,ri].fill_between(range(13),values[fi].min(axis=0),values[fi].max(axis=0),color=color,alpha=.12)
                axes[ei,ri].plot(range(13),values[fi,0],ls='--',color=color,lw=.9)
                axes[ei,ri].plot(range(13),values[fi,selected[fi]],color=color,label=f"tau={ens['taus'][fi]:.3f}")
            for fi in range(3):
                for scheme,ls in [('start0','--'),('best4','-')]:
                    relevant=[r for r in ratio_rows if r['ensemble']==ens['name'] and r['representation']==rep
                              and r['scheme']==scheme and r['length_reference_a']=='0' and int(r['flow_index'])==fi]
                    relevant.sort(key=lambda r:int(r['length_a']))
                    ys=[float(r['log_ratio']) if r['status']=='VALID' else np.nan for r in relevant]
                    rax[ei,ri].plot([int(r['length_a']) for r in relevant],ys,color=f'C{fi}',ls=ls,
                                    label=f"tau={ens['taus'][fi]:.3f}" if scheme=='best4' else None)
            axes[ei,ri].set(title=ens['name']+' '+rep+'; solid best4, dashed original',xlabel='n = length / a',ylabel='Real volume/direction averaged line')
            rax[ei,ri].set(title=ens['name']+' '+rep+'; solid best4, dashed original',xlabel='n = length / a',ylabel='log double ratio (n0=0)')
            for ax in (axes[ei,ri],rax[ei,ri]):
                ax.grid(alpha=.2)
                ax.legend(fontsize=7)
    fig.savefig(ROOT/'open_line_copy_bands.pdf')
    ratio_fig.savefig(ROOT/'ratio_copy_comparison.pdf')
    plt.close(fig);plt.close(ratio_fig)
    (ROOT/'REVIEW.md').write_text('\n'.join(report)+'\n')
    c.write_json(ROOT/'review.json',dict(status='PASS_COMPLETED_RESULT_REVIEW',changes=changes,
        audit_sha256=c.sha(ROOT/'result_audit.json'),review_source_sha256=c.sha(ROOT/'review.py'),
        source_lock_sha256=c.sha(ROOT/'source_lock.json'),ncfg_per_ensemble=1,
        ensemble_d_F=None,ensemble_d_A=None,statistical_stability='NOT_ASSESSED',
        artifacts={n:c.sha(ROOT/n) for n in ['REVIEW.md','open_line_copy_bands.pdf','ratio_copy_comparison.pdf']}))
    print(json.dumps(dict(status='PASS_COMPLETED_RESULT_REVIEW',rows=len(changes))),flush=True)


if __name__=='__main__':
    main()
