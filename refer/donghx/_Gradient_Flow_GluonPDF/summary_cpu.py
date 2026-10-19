#!/usr/bin/env python3
"""Validate all ratio/fit products; plot flow-offset bare values without lines."""
import csv
from datetime import datetime
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import *


def main():
    allocation();m=manifest();indices=shared_indices();rows=[];artifacts=[]
    plots=HERE/'plots';plots.mkdir(exist_ok=True)
    for task in range(72):
        it,rem=divmod(task,9);ic,jf=divmod(rem,3);tau=FLOWS[it][0]
        channel=CHANNELS[ic];flavor=FLAVORS[jf]
        receipt=json.loads((HERE/'status'/f'fit_{task}.json').read_text())
        rp=ratio_path(tau,channel,flavor);fp=fit_path(tau,channel,flavor)
        if receipt['status']!='PASS' or sha256(fp)!=receipt['output_sha256'] or sha256(rp)!=receipt['input_sha256']:
            raise ValueError('product SHA/receipt mismatch')
        with np.load(fp,allow_pickle=False) as d:
            np.testing.assert_array_equal(d['boot_indices'],indices);np.testing.assert_array_equal(d['confs'],CONFS)
            assert int(d['fit_Tmin'])==7 and float(d['q_min'])==Q_MIN and float(d['chi2_dof_max'])==CHI_MAX
            assert d['M_boot_selected'].shape==(2,100,3,21)
            w=d['candidate_weights'];wb=d['candidate_weights_boot']
            available=d['selected_available'];zero=d['exact_symmetry_zero']
            if not np.isfinite(d['M_boot_selected'][:, :, :, :]).all():
                # Missing finite-candidate fits remain explicit; they are not fabricated or hidden.
                print('REPORTED_MISSING_FIT_POINTS',task,int((~available).sum()),flush=True)
            np.testing.assert_allclose(w.sum(axis=0)[available&~zero],1.,rtol=2e-13,atol=2e-13)
            for co in range(2):
                for ip,pz in enumerate(PZ):
                    for z in range(21):
                        rows.append(dict(tau=tau,channel=channel,flavor=flavor,component=COMPONENTS[co],Pz=pz,z=z,
                            matrix=float(d['M_selected'][co,ip,z]),stat_error=float(d['statistical_error'][co,ip,z]),
                            window_systematic=float(d['window_systematic'][co,ip,z]),cut_systematic=float(d['cut_systematic'][co,ip,z]),
                            selected_available=bool(available[co,ip,z]),quality_accepted=bool(d['quality_accepted'][co,ip,z]),
                            fallback_used=bool(d['fallback_used'][co,ip,z]),fallback_fraction=float(d['fallback_fraction'][co,ip,z]),
                            status=str(d['fallback_status'][co,ip,z]),source=str(fp)))
        artifacts.extend([dict(path=str(rp),sha256=sha256(rp)),dict(path=str(fp),sha256=sha256(fp))])
    colors=plt.get_cmap('tab10')(np.arange(8));offsets=np.linspace(-.28,.28,8)
    for channel in CHANNELS:
        for flavor in FLAVORS:
            fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True,squeeze=False)
            for it,(tau,_) in enumerate(FLOWS):
                with np.load(fit_path(tau,channel,flavor),allow_pickle=False) as d:
                    for ip,pz in enumerate(PZ):
                        for co in range(2):
                            ax=axes[ip,co];x=np.arange(21)+offsets[it]
                            valid=d['selected_available'][co,ip];zero=d['exact_symmetry_zero'][co,ip]
                            diagnostic=d['fallback_used'][co,ip]|~d['quality_accepted'][co,ip]
                            for hollow in (False,True):
                                mask=valid&(diagnostic==hollow)&~zero
                                ax.errorbar(x[mask],d['M_selected'][co,ip,mask],yerr=d['statistical_error'][co,ip,mask],
                                    fmt='o',linestyle='none',markersize=3.0,capsize=1.2,
                                    color=colors[it],mfc='none' if hollow else colors[it],alpha=.85)
                            if zero.any():ax.plot(x[zero],d['M_selected'][co,ip,zero],marker='x',linestyle='none',color=colors[it],ms=4)
                            ax.set_title(f'Pz={pz} | {COMPONENTS[co]}',fontsize=10)
                            ax.set_xlim(-.5,20.5);ax.grid(alpha=.2);ax.axhline(0,color='gray',lw=.5)
                            if ip==2:ax.set_xlabel('z/a')
                            if co==0:ax.set_ylabel('Connected bare matrix element')
            handles=[Line2D([],[],marker='o',ls='none',color=colors[it],label=fr'$\tau={tau:g}$') for it,(tau,_) in enumerate(FLOWS)]
            handles.extend([Line2D([],[],marker='o',ls='none',color='black',mfc='none',label='diagnostic fallback / incomplete quality'),
                            Line2D([],[],marker='x',ls='none',color='black',label='exact symmetry zero (not fitted)')])
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.952),ncol=4,fontsize=8,frameon=False)
            fig.suptitle(f'L32x64 A | {channel} {flavor} | Nconf30, Nboot100 | Ratio AIC Tmin7\n'
                'Q >= 1e-3, chi2/dof <= 1e3; statistical error bars; positive-z diagnostic',fontsize=11,y=.995)
            fig.tight_layout(rect=(0,0,1,.88))
            p=plots/f'bare_{channel}_{flavor}_allflow.pdf';fig.savefig(p);plt.close(fig)
            artifacts.append(dict(path=str(p),sha256=sha256(p)))
    path=HERE/'bare_matrix_progress.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(HERE/'complete.json',dict(status='COMPLETE_A_SCHEME_RATIO_AND_BARE_FIT_DIAGNOSTICS',
        timestamp=datetime.now().astimezone().isoformat(),job=os.environ['SLURM_JOB_ID'],
        nconf=30,nboot=100,ratio_products=72,fit_products=72,plots=9,points=len(rows),
        selected_available=sum(r['selected_available'] for r in rows),quality_complete=sum(r['quality_accepted'] for r in rows),
        fallback_points=sum(r['fallback_used'] for r in rows),missing_finite_points=sum(not r['selected_available'] for r in rows),
        bootstrap_indices_sha256=m['bootstrap_indices_sha256'],artifacts=artifacts,
        remaining=['state isolation/window-systematic interpretation','transversity relative-sign external confirmation',
                   'full six-direction estimator','Zchi/Ward, line mass, SFTE/LaMET: not part of this run']))
    print('SUMMARY_COMPLETE',len(rows),'points,9 PDFs',flush=True)


if __name__=='__main__':main()
