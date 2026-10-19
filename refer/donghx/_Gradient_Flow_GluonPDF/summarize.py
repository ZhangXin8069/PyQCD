#!/usr/bin/env python3
"""Compute-node aggregation only. Never classify a scheme as biased by A/B difference."""
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'reference'))
from flow_common_l32x64 import atomic_json


def metrics(a,b):
    delta=b-a
    na,nb,nd=(float(np.linalg.norm(x.ravel())) for x in (a,b,delta))
    return dict(norm_A=na,norm_B=nb,norm_difference=nd,
        relative_L2_to_A=nd/max(na,1e-300),max_abs=float(np.max(abs(delta))),
        real_difference_norm=float(np.linalg.norm(delta.real.ravel())),
        imaginary_difference_norm=float(np.linalg.norm(delta.imag.ravel())))


def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    out=HERE/'summary';out.mkdir(exist_ok=True)
    missing=[];rows=[];c3_samples=[];c2_samples=[];conf_labels=[]
    fieldnames=['conf','t_current','sigma_a','sigma_b','time_type','wick',
        'norm_A','norm_B','norm_difference','relative_L2_to_A','max_abs',
        'real_difference_norm','imaginary_difference_norm']
    with (out/'ordered_block_wick_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fieldnames);writer.writeheader()
        for confrow in manifest['configurations']:
            conf=confrow['conf'];conf_c3=[];conf_c2=[]
            for tj in manifest['currents']:
                root=HERE/'results'/conf/f'tJ{tj:03d}'
                if not (root/'receipt.json').is_file() or not (root/'comparison.npz').is_file():
                    missing.append(dict(conf=conf,t_current=tj));continue
                receipt=json.loads((root/'receipt.json').read_text())
                if receipt.get('status')!='PASS':
                    missing.append(dict(conf=conf,t_current=tj));continue
                with np.load(root/'comparison.npz') as data:
                    terms=data['signed_wick_blocks']
                    times=data['sigma_times'];names=data['wick_names']
                    conf_c3.append(data['c3_by_scheme_flavor']);conf_c2.append(data['c2_common'])
                    for ia,ta in enumerate(times):
                        for ib,tb in enumerate(times):
                            for it,name in enumerate(names):
                                writer.writerow(dict(conf=conf,t_current=tj,sigma_a=int(ta),
                                    sigma_b=int(tb),time_type='same' if ta==tb else 'cross',
                                    wick=str(name),**metrics(terms[0,ia,ib,it],terms[1,ia,ib,it])))
                    rows.append(dict(conf=conf,t_current=tj,production_checks=receipt['checks'],
                        wick_metrics=receipt['wick_metrics'],c3_metrics=receipt['c3_metrics']))
            if len(conf_c3)==len(manifest['currents']):
                c3_samples.append(np.stack(conf_c3));c2_samples.append(np.stack(conf_c2));conf_labels.append(conf)
    report=dict(status='PASS' if not missing else 'INCOMPLETE',expected_cases=40,
        completed_cases=len(rows),complete_conf_count=len(conf_labels),missing=missing,cases=rows,
        scope=manifest['scope'],interpretation=[
            'Same-time changes, different-sector changes or failure to recover production A are implementation failures.',
            'Cross-time A/B differences are finite-realization estimator differences, not proof of bias.',
            'Shared source/sink stochastic-mode higher joint moments remain to be established.',
            'Only tau=0.5 and four selected current/source/sink cases were compared.'])
    if c3_samples:
        samples=np.stack(c3_samples) # conf,current,scheme,flavor,S,P,M,L,m,o
        # Four source-time cases averaged within a conf; configs remain paired.
        time_averages=samples.mean(axis=1)
        differences=time_averages[:,1]-time_averages[:,0]
        n=len(conf_labels)
        se_re=np.std(differences.real,axis=0,ddof=1)/np.sqrt(n) if n>1 else np.full(differences.shape[1:],np.nan)
        se_im=np.std(differences.imag,axis=0,ddof=1)/np.sqrt(n) if n>1 else np.full(differences.shape[1:],np.nan)
        np.savez_compressed(out/'paired_c3_c2_samples.npz',c3=samples,c2=np.stack(c2_samples),
            conf_labels=np.asarray(conf_labels),currents=np.asarray(manifest['currents']),
            c3_axes=np.asarray(['conf','case','scheme','flavor','sign','pol','mom','dz','m','o']),
            paired_difference_mean=differences.mean(axis=0),paired_difference_SE_real=se_re,
            paired_difference_SE_imag=se_im,
            paired_statistics_note=np.asarray('4 cases averaged per conf; 10 paired conf SEM; no bootstrap, not full production estimator'))
        # A simple tensor-norm view, retaining separate Wick topologies.
        fig,axes=plt.subplots(2,3,figsize=(12,6.5),sharex=True)
        for it,(name,ax) in enumerate(zip(('u1','u2','u3','u4','d1','d2'),axes.flat)):
            for ic,tj in enumerate(manifest['currents']):
                selected=[r for r in rows if r['t_current']==tj]
                xs=[int(r['conf']) for r in selected]
                ys=[r['wick_metrics'][name]['relative_L2_to_A'] for r in selected]
                ax.plot(xs,ys,marker='o',linestyle='none',label=f'tJ={tj}',alpha=.8)
            ax.set_title(name);ax.set_yscale('log');ax.set_ylabel(r'$\|B-A\|_2/\|A\|_2$')
            ax.tick_params(axis='x',rotation=45,labelsize=8);ax.grid(alpha=.2)
        for ax in axes[-1]:ax.set_xlabel('configuration')
        axes[0,0].legend(fontsize=8)
        fig.suptitle('L32x64 weight diagnostic; tau=0.5; all ordered blocks / open current spins')
        fig.tight_layout();fig.savefig(out/'wick_relative_differences.pdf');plt.close(fig)
    atomic_json(out/'report.json',report)
    print(json.dumps({k:report[k] for k in ('status','expected_cases','completed_cases','complete_conf_count','missing')},indent=2))
    if missing:raise SystemExit(2)


if __name__=='__main__':main()
