"""Explain noisy ratios without changing, clipping or selecting their replicas."""
import csv
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis_common import *

allocation(); code_check()
with np.load(HERE/'diagnostics/c2_tau0p10.npz',allow_pickle=False) as d:
    center, boots, snr, crossing = d['c2_original'], d['c2_boot'], d['real_snr'], d['bootstrap_real_sign_cross_fraction']
records=[]
for ip,p in enumerate(MOMENTA):
    for it,T in enumerate(TSEPS):
        dt = sorted({T//2,(T+1)//2})
        c=np.take(center[ip,it],dt).mean(); b=np.take(boots[:,ip,it],dt,axis=-1).mean(axis=-1)
        er,ei=np.std(b.real,ddof=1),np.std(b.imag,ddof=1)
        records.append(dict(signed_p=p,T=T,T_fm=T*A_FM,center_real=c.real,center_imag=c.imag,
            error_real=er,error_imag=ei,center_real_snr=abs(c.real)/er,
            center_real_sign_cross_fraction=float(np.mean(np.sign(b.real)!=np.sign(c.real))),
            center_min_abs_boot_over_abs_original=float(np.min(abs(b))/abs(c)),
            interior_min_real_snr=float(np.min(snr[ip,it,2:T-1])),
            interior_max_real_sign_cross_fraction=float(np.max(crossing[ip,it,2:T-1]))))
path=HERE/'diagnostics/c2_signal_by_momentum_tsep.csv'
with path.open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
fig,axes=plt.subplots(2,3,figsize=(12,7.5),squeeze=False)
for ip,p in enumerate(PZ):
    for si,s in enumerate((1,-1)):
        rows=[r for r in records if r['signed_p']==p*s]
        for row,comp in enumerate(('real','imag')):
            axes[row,ip].errorbar(np.array(TSEPS)+.06*(si-.5),[r[f'center_{comp}'] for r in rows],
                yerr=[r[f'error_{comp}'] for r in rows],fmt='o',ls='none',ms=4,capsize=2,
                label=f'P={p*s}')
            axes[row,ip].axhline(0,lw=.6,color='.7');axes[row,ip].set_xticks(TSEPS)
            axes[row,ip].set_title(f'nopol C2, {comp}, |P|={p}')
            axes[row,ip].set_xlabel('T/a');axes[row,ip].legend(frameon=False)
            axes[row,ip].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
fig.suptitle('L32x64 B, 50 cfg / 200 boot: complex dt-matched denominator, central insertion sampling')
fig.tight_layout(rect=(0,0,1,.94))
out=HERE/'plots/C2_signal_by_momentum_tsep.pdf';fig.savefig(out)
fig.savefig(HERE/'plots/C2_signal_by_momentum_tsep.png');plt.close(fig)
atomic_json(HERE/'status/c2_diagnostics.json',dict(status='PASS',records=records,
    csv=str(path),csv_sha256=sha(path),plot=str(out),plot_sha256=sha(out),job=os.environ['SLURM_JOB_ID'],
    source_sha256=sha(__file__),
    caveat='Re(C2) sign crossing is a noise diagnostic, not an assertion that the retained complex C2 is zero; no replicas removed.'))
print('C2_SIGNAL_DIAGNOSTICS_PASS',flush=True)
