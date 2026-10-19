#!/usr/bin/env python3
"""Paired finite-window audit. Never treat selected currents as configurations."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CONVENTIONS = Path('/public/group/lqcd/donghx/Diagram_GluonPDF/Fit_SKILL/C3_Ratio_Bootstrap')
sys.path.insert(0, str(CONVENTIONS))
from c3ratio.conventions import CHANNEL_LABELS, make_channel_specs, validate_gamma_basis

SCHEME = 'same_time_second_cross_time_independent_product'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def project(c3, labels, momenta):
    """w,f,sign,pol,p,z,m,o -> w,f,channel,sign,p,z (no phase/symmetry yet)."""
    specs = make_channel_specs(labels, ['z']*len(labels), ['nopol','pol15','pol25','pol35'])
    out = np.empty((3,3,4,2,len(momenta),21), complex)
    c3 = np.concatenate((c3, (c3[:,0]-c3[:,1])[:,None]), axis=1)
    for ip in range(len(momenta)):
        for ic in range(4):
            out[:,:,ic,:,ip] = np.einsum('wfszmo,mo->wfsz',
                c3[:,:,:,int(specs.pol_indices[ip,ic]),ip], specs.gamma[ip,ic])
    return out


def metrics(a,b):
    a,b = np.asarray(a),np.asarray(b)
    diff=b-a
    norm=float(np.linalg.norm(a))
    rms=float(np.sqrt(np.mean(abs(a)**2)))
    strong=abs(a)>max(1e-3*rms,1e-14)
    return dict(relative_L2=float(np.linalg.norm(diff)/max(norm,1e-300)),
        max_abs_shift=float(np.max(abs(diff))), reference_RMS=rms,
        strong_point_relative_max=(float(np.max(abs(diff[strong]/a[strong]))) if strong.any() else None),
        weak_point_count=int((~strong).sum()))


def load_case(path):
    receipt=json.loads((path.parent/'receipt.json').read_text())
    if receipt['status']!='PASS':
        raise ValueError(f'bad receipt {path}')
    with np.load(path,allow_pickle=False) as d:
        if str(d['weight_scheme'])!=SCHEME or int(d['weight_applied_count'])!=1:
            raise ValueError('wrong B weights')
        c3=d['c3_by_window_flavor']; c2=d['c2_common']
        moms=d['momenta_z'] if 'momenta_z' in d else np.array([3,4,5])
        labels=[str(x) for x in d['mom_labels']]
        windows=d['windows']
        if c3.shape != (3,2,2,4,len(moms),21,4,4) or c2.shape != (4,len(moms)):
            raise ValueError((path,c3.shape,c2.shape))
        if not np.isfinite(c3).all() or not np.isfinite(c2).all() or np.any(c2[0].real==0):
            raise ValueError(f'nonfinite/zero denominator {path}')
        return c3,c2,project(c3,labels,moms),moms,windows


def process(name, records, outdir):
    """Each record has conf/current/T/path. Separate source averaging precedes division."""
    groups={}
    raw_rows=[]
    inputs=[]
    denominator=[]
    for r in records:
        p=Path(r['file'])
        c3,c2,proj,moms,windows=load_case(p)
        inputs.append(dict(path=str(p),sha256=digest(p),conf=r['conf'],current=r['current'],tsep=r['tsep']))
        denominator.append(dict(conf=r['conf'],current=r['current'],tsep=r['tsep'],
            real_nopol=c2[0].real.tolist(),negative_count=int((c2[0].real<0).sum())))
        groups.setdefault((r['tsep'],r['conf']),[]).append((r['current'],proj,c2))
        # Raw open-current audit retains every pol and both Re/Im, not just a favored channel.
        ratio=c3/c2[0].real[None,None,None,None,:,None,None,None]
        ratio=np.concatenate((ratio,(ratio[:,0]-ratio[:,1])[:,None]),axis=1)
        for fi,fl in enumerate(('u','d','u-d')):
            for pol in range(4):
                for ip,mom in enumerate(moms):
                    raw_rows.append(dict(conf=r['conf'],current=r['current'],tsep=r['tsep'],
                        flavor=fl,pol=pol,momentum=int(mom),pair=f'{windows[-2]}->{windows[-1]}',
                        **metrics(ratio[-2,fi,:,pol,ip],ratio[-1,fi,:,pol,ip])))
    confs=sorted({r['conf'] for r in records},key=int)
    tseps=sorted({r['tsep'] for r in records})
    nums=[]; dens=[]
    for sep in tseps:
        ns=[];ds=[]
        for conf in confs:
            rows=sorted(groups[sep,conf])
            if [x[0] for x in rows] != [0,16,32,63]:
                raise ValueError('not the same four currents')
            ns.append(np.mean([x[1] for x in rows],axis=0))
            ds.append(np.mean([x[2][0].real for x in rows],axis=0))
        nums.append(ns);dens.append(ds)
    nums=np.asarray(nums);dens=np.asarray(dens)  # T,conf,w,f,ch,sign,p,z ; T,conf,p
    mean_num=nums.mean(axis=1);mean_den=dens.mean(axis=1)
    if np.any(mean_den==0): raise FloatingPointError('zero source/ensemble mean C2')
    ratio=mean_num/mean_den[:,None,None,None,None,:,None]
    shifts=ratio[:,-1]-ratio[:,-2]
    rows=[]
    for it,sep in enumerate(tseps):
        for fi,fl in enumerate(('u','d','u-d')):
            for ch,channel in enumerate(CHANNEL_LABELS):
                for ip,mom in enumerate(moms):
                    rows.append(dict(tsep=sep,flavor=fl,channel=channel,momentum=int(mom),
                        **metrics(ratio[it,-2,fi,ch,:,ip],ratio[it,-1,fi,ch,:,ip])))
    jk=None;err=None;drift_to_stat=None
    if len(confs)>1:
        n=len(confs)
        nj=(nums.sum(axis=1)[:,None]-nums)/(n-1)
        dj=(dens.sum(axis=1)[:,None]-dens)/(n-1)
        if np.any(dj==0): raise FloatingPointError('zero jackknife denominator')
        jk=nj/dj[:,:,None,None,None,None,:,None]
        err=np.sqrt((n-1)/n*np.sum((jk.real-jk.real.mean(axis=1,keepdims=True))**2,axis=1))
        err_im=np.sqrt((n-1)/n*np.sum((jk.imag-jk.imag.mean(axis=1,keepdims=True))**2,axis=1))
        # Ratios of paired numerator/denominator means; all 10 configs retained.
        numerator=np.stack((abs(shifts.real),abs(shifts.imag)))
        stat=np.stack((err[:,-1],err_im[:,-1]))
        valid=stat>1e-14
        drift_to_stat=np.full_like(stat,np.nan);np.divide(numerator,stat,out=drift_to_stat,where=valid)
    outdir.mkdir(parents=True,exist_ok=True)
    payload=dict(projected_numerator_source_mean=nums,c2_nopol_real_source_mean=dens,
        ratio=ratio,paired_window_shift=shifts,confs=np.asarray(confs),tseps=np.asarray(tseps),
        windows=windows,momenta_z=moms,channel_labels=np.asarray(CHANNEL_LABELS),
        flavor_labels=np.asarray(['u','d','u-d']),
        ratio_axes=np.asarray(['tsep','window','flavor','channel','sign','momentum','z']),
        estimator=np.asarray('ratio_of_separately_source_and_configuration_averaged_correlators'),
        transversity_status=np.asarray('pending_external_physics_confirmation'))
    if jk is not None:
        payload.update(ratio_jackknife=jk,error_real=err,error_imag=err_im,drift_over_stat=drift_to_stat)
    np.savez_compressed(outdir/'window_ratio_shift.npz',**payload)
    for filename,data in [('open_spin_curve_metrics.csv',raw_rows),('projected_curve_metrics.csv',rows)]:
        with (outdir/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    report=dict(name=name,configuration_count=len(confs),case_count=len(records),confs=confs,
        windows=windows.tolist(),flow_time_a2=float(records[0]['flow_time_a2']),
        projected_complex_curve_max_relative_L2=max(x['relative_L2'] for x in rows),
        open_spin_curve_max_relative_L2=max(x['relative_L2'] for x in raw_rows),
        projected_curve_max_abs_shift=max(x['max_abs_shift'] for x in rows),
        denominator_cases=denominator,source_ensemble_denominator=mean_den.tolist(),
        denominator_policy='signed real nopol denominator; no abs, clipping, or dropped configs',
        inputs=inputs,estimator=str(payload['estimator']),
        uncertainty=('paired delete-one configuration jackknife, recomputed ratio; no fit' if jk is not None else 'none: single configuration'),
        drift_over_stat_max=(float(np.nanmax(drift_to_stat)) if jk is not None else None),
        status='FINITE_WINDOW_DIAGNOSTIC_NOT_FULL_TIME_PROOF',
        caveats=['widest window is not the exact 64-time integral','four currents are not four independent configurations',
                 'no PDF/fit/Zchi/SFTE/LaMET claim; no phase or VDV symmetry averaging',
                 'near-zero components are judged also by absolute shifts, not unbounded relative change'])
    (outdir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(outdir/'window_ratio_comparison.pdf') as pdf:
        for ch,channel in enumerate(CHANNEL_LABELS):
            for component in ('real','imag'):
                fig,axes=plt.subplots(2,len(moms),figsize=(max(10,2.4*len(moms)),6),squeeze=False)
                for ip,mom in enumerate(moms):
                    ax=axes[0,ip]; ad=axes[1,ip]
                    for iw,w in enumerate(windows):
                        v=getattr(ratio[-1,iw,2,ch,:,ip],component).mean(axis=0)
                        ax.plot(np.arange(21),v,'.',label=f'w={w}')
                    for it,sep in enumerate(tseps):
                        ad.plot(np.arange(21),getattr(shifts[it,2,ch,:,ip],component).mean(axis=0),'.',label=f'T={sep}')
                    ax.set_title(f'Pz={mom}');ax.set_xlabel('z/a');ad.set_xlabel('z/a')
                    ax.grid(alpha=.2);ad.grid(alpha=.2)
                axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7)
                axes[0,0].set_ylabel(f'{component} C3 / Re C2');axes[1,0].set_ylabel('widest - previous')
                fig.suptitle(f'{name}: u-d {channel}; signs averaged for display only; Nconf={len(confs)}')
                fig.tight_layout(rect=(0,0,1,.94));pdf.savefig(fig);plt.close(fig)
    return {k:v for k,v in report.items() if k not in ('inputs','denominator_cases')}


def main():
    p=argparse.ArgumentParser();p.add_argument('--production-root',type=Path);a=p.parse_args()
    if 'SLURM_JOB_ID' not in os.environ: raise RuntimeError('CPU allocation required')
    validate_gamma_basis()
    summaries=[]
    if a.production_root:
        manifest=json.loads((a.production_root/'manifest.json').read_text())
        for flow in (.2,.5,.8):
            records=[]
            for task in manifest['tasks']:
                if task['flow_time_a2']!=flow: continue
                root=Path(task['reuse_result_root']) if task.get('reuse_result_root') else a.production_root
                for current in (0,16,32,63):
                    for sep in range(7,13):
                        records.append(dict(conf=task['conf'],current=current,tsep=sep,flow_time_a2=flow,
                            file=str(root/'C3_schemeB_Pz3456_tsep7to12'/task['tag']/task['conf']/f'tJ{current:03d}'/f'tsep{sep:02d}'/'window_comparison.npz')))
            summaries.append(process(f'20conf_tau{flow}',records,a.production_root/'summary'/f'tau{flow}'))
        dest=a.production_root/'summary'
    else:
        report=json.loads((BASE/'quark_l32x64_P4_B_memfix_20261015/summary/report.json').read_text())
        for flow in (.2,.5,.8):
            records=[dict(r,conf='10000') for r in report['cases'] if r['flow_time_a2']==flow]
            summaries.append(process(f'conf10000_tau{flow}',records,HERE/f'tau{flow}'))
        root=BASE/'quark_l32x64_schemeB_window_20261015'
        m=json.loads((root/'manifest.json').read_text())
        records=[dict(conf=r['conf'],current=c,tsep=7,flow_time_a2=.5,
                      file=str(root/'results'/r['conf']/f'tJ{c:03d}'/'window_comparison.npz'))
                 for r in m['configurations'] for c in m['currents']]
        summaries.append(process('legacy10conf_tau0.5',records,HERE/'legacy10conf_tau0.5'))
        dest=HERE
    (dest/'review_summary.json').write_text(json.dumps(dict(status='REVIEW_COMPLETE_RETAIN_WIDEST_AND_NESTED_WINDOWS',
        results=summaries,job=os.environ['SLURM_JOB_ID'],code_sha256=digest(__file__),
        conventions_sha256=digest(CONVENTIONS/'c3ratio/conventions.py'),
        recommendation='Extend same selected-current midpoint diagnostic to 20 conf, retain all nested windows; provisional widest reference w=5,6,8. No exact full-time convergence claimed.'),indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__': main()
