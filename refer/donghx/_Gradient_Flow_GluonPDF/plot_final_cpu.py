"""Final non-overlapping layouts; consumes immutable normalized data only."""
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from run_cpu import HERE, INPUTS, CHANNELS, FLAVORS, COMPONENTS, output_path, sha, write_json


def main():
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Plotting requires CPU allocation')
    validation=json.loads((HERE/'verification.json').read_text())
    assert validation['status']=='PASS_COMPLETE_INDEPENDENT_READBACK_AND_PDF_RENDER'
    assert validation['complete_sha256']==sha(HERE/'complete.json')
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':11,'pdf.fonttype':42})
    colors=plt.get_cmap('tab10').colors
    paths=[]
    for branch,(_,flows,pz) in INPUTS.items():
        folder=HERE/'plots_final'/branch
        folder.mkdir(parents=True,exist_ok=True)
        for channel in CHANNELS:
            for flavor in FLAVORS:
                height=3*len(pz)+2.1
                fig,axes=plt.subplots(len(pz),2,figsize=(13,height),sharex=True,sharey=False,squeeze=False)
                for it,tau in enumerate(flows):
                    with np.load(output_path(branch,tau,channel,flavor),allow_pickle=False) as d:
                        x=np.arange(21)+(it-(len(flows)-1)/2)*.065
                        for ip,p in enumerate(pz):
                            for co,label in enumerate(COMPONENTS):
                                ax=axes[ip,co]
                                good=d['selected_available'][co,ip]
                                constraint=d['local_anchor_constraint'][co,ip]|d['exact_symmetry_zero'][co,ip]
                                strict=d['strict_quality_nonconstraint'][co,ip]
                                for mask,filled in ((good&~constraint&strict,True),(good&~constraint&~strict,False)):
                                    if mask.any():
                                        ax.errorbar(x[mask],d['H_local'][co,ip,mask],yerr=d['statistical_error'][co,ip,mask],
                                            fmt='o',ls='none',ms=3,capsize=1.8,elinewidth=.8,color=colors[it],
                                            mfc=colors[it] if filled else 'none',alpha=.88)
                                mask=good&constraint
                                ax.plot(x[mask],d['H_local'][co,ip,mask],marker='x',ls='none',ms=4,color=colors[it])
                                ax.axhline(0,color='.7',lw=.6,zorder=0)
                                ax.set_xlim(-.5,20.5)
                                ax.set_title(rf'$P_z/(2\pi/L)={p}$; '+label.replace('_',' '))
                                ax.set_ylabel(r'$h_\Gamma(z,P,\tau)/h_\Gamma(0,P,\tau)$')
                                ax.grid(alpha=.15)
                for ax in axes[-1]:
                    ax.set_xlabel(r'$z/a$ (flow times shifted horizontally only)')
                handles=[Line2D([],[],color=colors[it],marker='o',ls='none',ms=4,label=rf'$\tau={tau:.1f}$')
                         for it,tau in enumerate(flows)]
                handles.extend([
                    Line2D([],[],color='black',marker='o',mfc='none',ls='none',label='inherited fit / denominator diagnostic'),
                    Line2D([],[],color='black',marker='o',ls='none',label='upstream quality, no fallback/warning'),
                    Line2D([],[],color='black',marker='x',ls='none',label='constraint / exact odd zero (not a fit)')])
                fig.suptitle(f'L32x64 B | {channel} | {flavor.replace("_minus_"," - ")} | same-P local normalization',
                             y=1-.18/height,fontsize=13)
                legend=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1-.65/height),
                                  ncol=4,frameon=False,fontsize=9,columnspacing=1.3)
                fig.text(.5,.022,'20 conf | 100 paired bootstrap | statistical errors only | local charge removed; line mass NOT subtracted',
                         ha='center',fontsize=9)
                fig.subplots_adjust(top=1-1.95/height,bottom=.085 if len(pz)>1 else .18,
                                    left=.09,right=.98,hspace=.40,wspace=.25)
                fig.canvas.draw()
                renderer=fig.canvas.get_renderer()
                legend_box=legend.get_window_extent(renderer).transformed(fig.transFigure.inverted())
                title_top=max(ax.title.get_window_extent(renderer).transformed(fig.transFigure.inverted()).y1
                              for ax in axes[0])
                if legend_box.y0<=title_top+.008:
                    raise RuntimeError(('legend/title overlap',branch,channel,flavor,legend_box.y0,title_top))
                path=folder/f'localnorm_{channel}_{flavor}_allflow.pdf'
                fig.savefig(path)
                plt.close(fig)
                paths.append(path)
        print('FINAL_LAYOUT_DONE',branch,flush=True)
    previews=HERE/'previews_final'
    previews.mkdir(exist_ok=True)
    rendered=[]
    for branch,channel in (('moving','unpolarized'),('moving','helicity'),('Pz0','transversity')):
        path=HERE/'plots_final'/branch/f'localnorm_{channel}_u_minus_d_allflow.pdf'
        prefix=previews/f'{branch}_{path.stem}'
        subprocess.run([shutil.which('pdftoppm'),'-f','1','-singlefile','-scale-to','1500','-png',str(path),str(prefix)],check=True)
        rendered.append(str(prefix.with_suffix('.png')))
    write_json(HERE/'final_plots.json',dict(status='PASS_FINAL_PDF_LAYOUT_AND_SOURCE_VALIDATION',
        generated=datetime.now().astimezone().isoformat(),job=os.environ['SLURM_JOB_ID'],
        node=os.environ.get('SLURMD_NODENAME'),nplots=len(paths),data_complete_sha256=sha(HERE/'complete.json'),
        independent_data_verification_sha256=sha(HERE/'verification.json'),
        plot_source_sha256=sha(__file__),legend_title_no_overlap_all_figures=True,
        final_previews=rendered,artifacts=[dict(path=str(p),sha256=sha(p)) for p in paths]))
    print('PASS_FINAL_PLOTS',len(paths),flush=True)


if __name__=='__main__':
    main()
