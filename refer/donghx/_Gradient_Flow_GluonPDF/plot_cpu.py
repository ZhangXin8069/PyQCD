"""All flows offset in z, no connecting lines; quality/fallback remain distinct."""
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from common import *


def main():
    allocation();manifest();shared_indices();plots=[];rows=[]
    colors=plt.get_cmap('tab10')(np.arange(len(FLOWS)));offsets=np.linspace(-.28,.28,len(FLOWS))
    for channel in CHANNELS:
        for flavor in FLAVORS:
            datasets=[]
            for tau,_ in FLOWS:
                with np.load(fit_path(tau,channel,flavor),allow_pickle=False) as d:
                    datasets.append({k:d[k] for k in ('M_selected','statistical_error','selected_available',
                        'exact_symmetry_zero','fallback_used','quality_accepted','fallback_fraction',
                        'fallback_status','window_systematic','cut_systematic')})
                d=datasets[-1]
                for co,ip,z in np.ndindex(2,len(PZ),DZ):
                    rows.append(dict(ensemble=KEY,tau=tau,rf_fm=float(A_FM*np.sqrt(8*tau)),channel=channel,
                        flavor=flavor,component=COMPONENTS[co],Pz=PZ[ip],z_a=z,z_fm=z*A_FM,
                        matrix=float(d['M_selected'][co,ip,z]),statistical_error=float(d['statistical_error'][co,ip,z]),
                        window_systematic=float(d['window_systematic'][co,ip,z]),cut_systematic=float(d['cut_systematic'][co,ip,z]),
                        selected_available=bool(d['selected_available'][co,ip,z]),
                        quality_accepted=bool(d['quality_accepted'][co,ip,z]),fallback_used=bool(d['fallback_used'][co,ip,z]),
                        fallback_fraction=float(d['fallback_fraction'][co,ip,z]),status=str(d['fallback_status'][co,ip,z]),
                        source=str(fit_path(tau,channel,flavor))))
            path=HERE/'plots'/f'bare_{channel}_{flavor}_allflow_boot200.pdf'
            with PdfPages(path) as pdf:
                for fixed in (False,True):
                    fig,axes=plt.subplots(len(PZ),2,figsize=(12.5,3*len(PZ)+2),sharex=True,squeeze=False)
                    for it,((tau,_),d) in enumerate(zip(FLOWS,datasets)):
                        for ip,p in enumerate(PZ):
                            for co in range(2):
                                ax=axes[ip,co];x=np.arange(DZ)+offsets[it]
                                valid=d['selected_available'][co,ip];zero=d['exact_symmetry_zero'][co,ip]
                                diag=d['fallback_used'][co,ip]|~d['quality_accepted'][co,ip]
                                for hollow in (False,True):
                                    use=valid&~zero&(diag==hollow)
                                    ax.errorbar(x[use],d['M_selected'][co,ip,use],yerr=d['statistical_error'][co,ip,use],
                                        fmt='o',linestyle='none',color=colors[it],mfc='none' if hollow else colors[it],
                                        ms=3.1,capsize=1.1,elinewidth=.8,alpha=.85)
                                if zero.any():
                                    ax.plot(x[zero],d['M_selected'][co,ip,zero],marker='x',ls='none',color=colors[it],ms=4)
                                ax.set_title(fr'$P_z={p}$ | {COMPONENTS[co]}',fontsize=10)
                                ax.axhline(0,color='gray',lw=.5);ax.grid(alpha=.18);ax.set_xlim(-.5,20.5)
                                if fixed:ax.set_ylim((-1.,2.) if co==0 else (-2.,1.))
                                if ip==len(PZ)-1:ax.set_xlabel('$z/a$')
                                if co==0:ax.set_ylabel('Connected bare matrix element')
                    handles=[Line2D([],[],marker='o',ls='none',color=colors[i],
                        label=fr'$\tau={t:.4g},\ r_f={A_FM*np.sqrt(8*t):.4f}$ fm') for i,(t,_) in enumerate(FLOWS)]
                    handles.extend([Line2D([],[],marker='o',ls='none',color='black',label='quality-only, no fallback'),
                        Line2D([],[],marker='o',ls='none',color='black',mfc='none',label='diagnostic fallback / incomplete quality'),
                        Line2D([],[],marker='x',ls='none',color='black',label='exact odd-z0 zero (not fitted)')])
                    view='fixed viewport (full values retained in NPZ/CSV)' if fixed else 'full y-range (all statistical error bars)'
                    fig.suptitle(f'{KEY} B | {channel} {flavor} | {NCONF} conf, 200 bootstrap | {view}\n'
                        f'Ratio AIC: Tmin={TMIN} ({TMIN*A_FM:.4f} fm), Q >= 1e-3, chi2/dof <= 1e3; statistical errors',
                        fontsize=10.5,y=.993)
                    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.958),ncol=3,fontsize=7.8,frameon=False)
                    fig.subplots_adjust(left=.08,right=.98,bottom=.045,top=.855,hspace=.35,wspace=.23)
                    pdf.savefig(fig);plt.close(fig)
            plots.append(dict(path=str(path),sha256=sha256(path),pages=2))
    with (HERE/'bare_matrix_progress.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    atomic_json(HERE/'status/plots.json',dict(status='PASS',products=plots,point_rows=len(rows),
        csv_path=str(HERE/'bare_matrix_progress.csv'),csv_sha256=sha256(HERE/'bare_matrix_progress.csv'),
        plot_policy='all z=0..20; flow offsets; no connecting lines; page1 full range, page2 fixed viewport',
        job=os.environ['SLURM_JOB_ID']))
    print('PLOTS_COMPLETE',KEY,len(plots),flush=True)


if __name__=='__main__':
    main()
