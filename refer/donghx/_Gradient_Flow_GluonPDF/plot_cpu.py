"""CPU-only all-flow plots and before/after comparison at tau=0.5."""
import json
import os
import shutil
import subprocess
import numpy as np
from common import *


def main():
    require_cpu(); check_lock()
    verification = json.loads((HERE/'verification.json').read_text())
    assert verification['status']=='PASS_FULL_ARRAY_INDEPENDENT_READBACK'
    assert verification['complete_sha256']==sha(HERE/'complete.json')
    complete = json.loads((HERE/'complete.json').read_text())
    for item in complete['artifacts']:
        assert sha(item['path'])==item['sha256']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':11,
                         'pdf.fonttype':42,'savefig.dpi':130})
    colors = plt.get_cmap('tab10').colors
    paths=[]
    for branch,pz in BRANCHES.items():
        for channel in CHANNELS:
            for flavor in FLAVORS:
                for mode in ('after_allflow','before_after_tau0p50'):
                    height = 3*len(pz)+2.4
                    fig,axes = plt.subplots(len(pz),2,figsize=(13,height),squeeze=False,
                                            sharex=True,sharey=False)
                    flows = FLOWS if mode=='after_allflow' else (.5,)
                    for it,tau in enumerate(flows):
                        with np.load(output_path(branch,tau,channel,flavor),allow_pickle=False) as archive:
                            d = {key:archive[key] for key in archive.files}
                        x = np.arange(21)+(it-(len(flows)-1)/2)*.065
                        for ip,p in enumerate(pz):
                            for co,label in enumerate(COMPONENTS):
                                ax=axes[ip,co]
                                constraint=d['local_anchor_constraint'][co,ip]|d['exact_symmetry_zero'][co,ip]
                                strict=d['strict_quality_nonconstraint'][co,ip]
                                stages = [('H_line','line_statistical_error',colors[it],0.,'o',1.)]
                                if mode!='after_allflow':
                                    stages=[('H_local','statistical_error','#5995bd',-.07,'s',.55),
                                            ('H_line','line_statistical_error','#9a3300',.07,'o',1.)]
                                for value,error,color,shift,marker,alpha in stages:
                                    for mask,filled in ((~constraint&strict,True),(~constraint&~strict,False)):
                                        ax.errorbar(x[mask]+shift,d[value][co,ip,mask],
                                            yerr=d[error][co,ip,mask],fmt=marker,ls='none',
                                            ms=3.1,capsize=1.8,elinewidth=.8,color=color,
                                            mfc=color if filled else 'none',alpha=alpha)
                                    ax.plot(x[constraint]+shift,d[value][co,ip,constraint],
                                            marker='x',ls='none',ms=4,color=color,alpha=alpha)
                                ax.axhline(0,color='.65',lw=.6,zorder=0)
                                ax.set_xlim(-.5,20.5)
                                ax.set_title(rf'$P_z/(2\pi/L)={p}$; '+label.replace('_',' '))
                                ax.set_ylabel(r'$H_{\rm line}$' if mode=='after_allflow' else r'$H_{\rm local}$ or $H_{\rm line}$')
                                ax.grid(alpha=.15)
                    for ax in axes[-1]:
                        ax.set_xlabel(r'$z/a$ (horizontal offsets for comparison only)')
                    if mode=='after_allflow':
                        handles=[Line2D([],[],color=colors[it],marker='o',ls='none',
                                        label=rf'$\tau={tau:.1f}$') for it,tau in enumerate(FLOWS)]
                    else:
                        handles=[Line2D([],[],color='#5995bd',marker='s',ls='none',alpha=.55,
                                        label='before: local normalized'),
                                 Line2D([],[],color='#9a3300',marker='o',ls='none',
                                        label='after: fixed-dF line subtracted')]
                    handles.extend([
                        Line2D([],[],color='black',marker='o',mfc='none',ls='none',label='inherited fit/denominator diagnostic'),
                        Line2D([],[],color='black',marker='o',ls='none',label='upstream quality (not final physics)'),
                        Line2D([],[],color='black',marker='x',ls='none',label='local constraint / exact odd zero')])
                    desc='all selected flows' if mode=='after_allflow' else r'before / after at $\tau=0.5$'
                    fig.suptitle(f'L32x64 B | {channel} | {flavor.replace("_minus_"," - ")} | {desc}',
                                 y=1-.18/height,fontsize=13)
                    legend=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1-.65/height),
                                      ncol=4 if mode=='after_allflow' else 3,frameon=False,fontsize=9)
                    fig.text(.5,.025,r'20 conf | 100 paired bootstrap | fixed $d_F=0.374137$ trial; coefficient uncertainty NOT included',
                             ha='center',fontsize=9)
                    fig.text(.5,.008,'Local charge removed | no m0 / SFTE / LaMET | candidate dF and inherited fits are diagnostic',
                             ha='center',fontsize=8)
                    fig.subplots_adjust(top=1-1.95/height,bottom=.085 if len(pz)>1 else .18,
                                        left=.095,right=.98,hspace=.42,wspace=.28)
                    fig.canvas.draw()
                    renderer=fig.canvas.get_renderer()
                    lb=legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
                    title_top=max(ax.title.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y1
                                  for ax in axes[0])
                    if lb.y0<=title_top+.008:
                        raise RuntimeError(('legend / axes overlap',branch,channel,flavor,mode))
                    folder=HERE/'plots'/mode/branch; folder.mkdir(parents=True,exist_ok=True)
                    path=folder/f'{mode}_{channel}_{flavor}.pdf'
                    fig.savefig(path); plt.close(fig); paths.append(path)
        print('PLOTS_DONE',branch,flush=True)
    fig,ax=plt.subplots(figsize=(8,5))
    for it,tau in enumerate(FLOWS):
        ax.plot(np.arange(21),line_factor(np.arange(21),tau),marker='o',ls='none',ms=3,
                color=colors[it],label=rf'$\tau={tau:.1f}$')
    ax.set_yscale('log'); ax.set_xlim(-.5,20.5)
    ax.set_xlabel(r'$z/a$'); ax.set_ylabel(r'$\exp[d_F|z/a|/\sqrt{8\tau}]$')
    ax.legend(ncol=3,fontsize=9); ax.grid(alpha=.2)
    ax.set_title(r'Fixed $d_F=0.374137$: central values and statistical errors scale together')
    fig.tight_layout(); path=HERE/'plots/line_factor.pdf'; fig.savefig(path); plt.close(fig); paths.append(path)
    previews=HERE/'previews'; previews.mkdir(exist_ok=True)
    rendered=[]
    if not shutil.which('pdftoppm'):
        raise RuntimeError('pdftoppm required for visual verification')
    for mode,branch,channel in (('after_allflow','moving','unpolarized'),('after_allflow','moving','helicity'),
                               ('after_allflow','Pz0','transversity'),('before_after_tau0p50','moving','unpolarized')):
        path=HERE/'plots'/mode/branch/f'{mode}_{channel}_u_minus_d.pdf'
        prefix=previews/f'{branch}_{mode}_{channel}_u_minus_d'
        subprocess.run(['pdftoppm','-f','1','-singlefile','-scale-to','1400','-png',str(path),str(prefix)],check=True)
        rendered.append(str(prefix.with_suffix('.png')))
    check_lock()
    write_json(HERE/'plots_complete.json',dict(status='PASS_PDF_LAYOUT_AND_RENDER',created=now(),
        job=os.environ['SLURM_JOB_ID'],node=os.environ.get('SLURMD_NODENAME'),
        data_complete_sha256=sha(HERE/'complete.json'),verification_sha256=sha(HERE/'verification.json'),
        pdf_count=len(paths),legend_title_no_overlap=True,previews=rendered,
        artifacts=[dict(path=str(p),sha256=sha(p)) for p in paths]))
    print('PLOTS_PASS',len(paths),flush=True)


if __name__=='__main__':
    main()
