"""CPU-only interpretive diagnostics; never promotes fallback or poor fits."""
import json
import os
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from common import CFG,ENSEMBLES,CHANNELS,FLAVORS,B,read,atomic_json,table,now
from analysis_cpu import path_cont


def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('CPU allocation required')
    bare=[];a_rows=[];flow_rows=[];mass=[];examples=[];sensitivity=[]
    for key in ENSEMBLES:
        for path in sorted((ROOT/key/'fits').glob('*.npz')):
            d=read(path);nonconstraint=~d['exact_symmetry_zero'];rep=d['fallback_used_boot'][:, :, :, :]
            fraction=rep.mean(axis=1)[nonconstraint]
            row=dict(ensemble=key,channel=str(d['channel']),flavor=str(d['flavor_combo']),tau=float(d['tau_a2']),
                entries=d['M_selected'].size,available=int(d['selected_available'].sum()),
                nonconstraint=int(nonconstraint.sum()),central_fallback=int((d['central_fallback_used']&nonconstraint).sum()),
                any_replica_fallback=int((d['fallback_used']&nonconstraint).sum()),
                median_bootstrap_fallback_fraction=float(np.median(fraction)),
                mean_bootstrap_fallback_fraction=float(np.mean(fraction)),max_bootstrap_fallback_fraction=float(np.max(fraction)),
                physical_result_status='diagnostic_no_ground_state_isolation_claim')
            bare.append(row)
    for ch in CHANNELS:
        for fl in FLAVORS:
            for t in CFG['flows64']:
                d=read(path_cont('08_continuum_fixed_flow',ch,fl,t));mask=~d['constraint_mask'];finite=np.isfinite(d['value'])&np.isfinite(d['samples']).all(axis=1)
                quality=finite&mask&(d['Q']>=.05)&(d['chi2_dof']<=3)
                raw=(d['a2_over_tf'] if 'a2_over_tf' in d else None)
                a_rows.append(dict(channel=ch,flavor=fl,tau64=t,tf_fm2=float(d['tf_fm2']),
                    finite_nonlocal=int((finite&mask).sum()),nonlocal=int(mask.sum()),central_quality_nonlocal=int(quality.sum()),
                    strict_upstream_quality_nonlocal=int(d['strict_quality_nonconstraint'].sum()),dof=1))
            for policy in CFG['policies']:
                d=read(path_cont('09_continuum_tau0',ch,fl,policy=policy));finite=np.isfinite(d['value'])&np.isfinite(d['samples']).all(axis=1);mask=~d['constraint_mask']
                quality=finite&mask&(d['Q']>=.05)&(d['chi2_dof']<=3)
                flow_rows.append(dict(channel=ch,flavor=fl,policy=policy,finite=int(finite.sum()),entries=d['value'].size,
                    finite_nonlocal=int((finite&mask).sum()),central_quality_nonlocal=int(quality.sum()),
                    strict_upstream_quality_nonlocal=int(d['strict_quality_nonconstraint'].sum())))
                for co,ip,iz in ((0,2,4),(1,2,4),(0,3,8),(1,3,8)):
                    examples.append(dict(channel=ch,flavor=fl,policy=policy,component=('Re_even','Im_odd')[co],
                        momentum_GeV=float(d['momentum_GeV'][ip]),z_fm=float(d['z_fm'][iz]),value=float(d['value'][co,ip,iz]),
                        error=float(d['error'][co,ip,iz]),Q=float(d['Q'][co,ip,iz]),chi2_dof=float(d['chi2_dof'][co,ip,iz]),
                        fit_status=str(d['status'][co,ip,iz]),strict_quality=bool(d['strict_quality_nonconstraint'][co,ip,iz])))
            mpath=ROOT/'continuum/10_continuum_m0_corrected/a2_linear/linear/RGR/mass_fits'/f'{ch}_{fl}.npz'
            d=read(mpath)
            mass.append(dict(channel=ch,flavor=fl,m0_GeV=float(d['m0_GeV']),error_GeV=float(np.std(d['m0_boot_GeV'],ddof=1)),
                fit_executed=bool(d['fit_executed']),quality_pass=bool(d['quality_pass']),Q=float(d['Q']),chi2_dof=float(d['chi2_dof']),status=str(d['status'])))
            primary=read(path_cont('09_continuum_tau0',ch,fl,policy='allflow_display_only'))
            for kind,model in (('quadratic','a2_linear'),('linear','a_linear_sensitivity')):
                control=read(path_cont('09_continuum_tau0',ch,fl,kind=kind,model=model,policy='allflow_display_only'))
                delta=control['value']-primary['value'];db=control['samples']-primary['samples'];err=np.std(db,axis=1,ddof=1)
                mask=np.isfinite(delta)&np.isfinite(err)&(err>0)&~primary['constraint_mask']
                sensitivity.append(dict(channel=ch,flavor=fl,control=f'{model}/{kind}',finite_nonlocal=int(mask.sum()),
                    median_absolute_shift=float(np.median(abs(delta[mask]))) if mask.any() else np.nan,
                    median_paired_pull=float(np.median(abs(delta[mask])/err[mask])) if mask.any() else np.nan,
                    max_paired_pull=float(np.max(abs(delta[mask])/err[mask])) if mask.any() else np.nan))
    out=ROOT/'00_audit/interpretation_v2';out.mkdir(exist_ok=True)
    for label,rows in (('bare_replica_fallback',bare),('continuum_quality',a_rows),('flow_quality',flow_rows),
                       ('m0_primary',mass),('examples_tau0',examples),('model_sensitivity',sensitivity)):
        table(out/(label+'.csv'),rows)
    summary=dict(timestamp=now(),job=os.environ['SLURM_JOB_ID'],bare_entries=sum(r['entries'] for r in bare),
        bare_available=sum(r['available'] for r in bare),continuum_primary_finite_nonlocal=sum(r['finite_nonlocal'] for r in a_rows),
        continuum_primary_central_quality_nonlocal=sum(r['central_quality_nonlocal'] for r in a_rows),
        continuum_primary_strict_upstream_quality_nonlocal=sum(r['strict_upstream_quality_nonlocal'] for r in a_rows),
        primary_m0_fits_executed=sum(r['fit_executed'] for r in mass),primary_m0_quality_pass=sum(r['quality_pass'] for r in mass),
        mass=mass,flow=flow_rows,physical_validated=False,controlled_continuum_limit=False,
        caveats='fixed dF candidate; mass/volume/Oa/excited-state/matching-scheme/strong-flow-window not closed; statistical errors conditional')
    atomic_json(out/'summary.json',summary)
    lines=['# 三格距数值诊断（CPU 汇总）','',summary['timestamp'],'',
        f'裸拟合selected覆盖：{summary["bare_available"]}/{summary["bare_entries"]}。',
        f'主a²/RGR/linear插值分支：有限非局域格距拟合{summary["continuum_primary_finite_nonlocal"]}，中心Q/chi2诊断通过{summary["continuum_primary_central_quality_nonlocal"]}，同时继承严格上游质量{summary["continuum_primary_strict_upstream_quality_nonlocal"]}。',
        f'连续分支主m0：执行{summary["primary_m0_fits_executed"]}/9，Q>=.05且chi2/dof<=3通过{summary["primary_m0_quality_pass"]}/9。','',
        'any_replica_fallback是至少一行回退的点数，不等于200行全部回退；逐点真实回退比例见bare_replica_fallback.csv。',
        '模型敏感性按同一bootstrap行的差值计算，不能把独立误差简单相加。','',
        '|channel|flavor|m0 [GeV]|stat.error|Q|chi2/dof|quality|','|---|---|---:|---:|---:|---:|---|']
    for r in mass:lines.append(f'|{r["channel"]}|{r["flavor"]}|{r["m0_GeV"]:.6g}|{r["error_GeV"]:.3g}|{r["Q"]:.3g}|{r["chi2_dof"]:.3g}|{r["quality_pass"]}|')
    lines+=['','有限值、低误差、代码通过和物理验收是不同状态；这些产物仍是共同中间S下的探索性结果，不是受控连续极限或最终PDF。']
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print('INTERPRETATION_SUMMARY_COMPLETE',json.dumps({k:v for k,v in summary.items() if k not in ('mass','flow')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
