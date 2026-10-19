"""Plot measured 10-configuration ringed products with their paired errors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from quark_ringed_kinetic_v1.core import digest


def load_checked(source):
    meta=json.loads(source.with_name('manifest.json').read_text())
    if digest(source)!=meta['output_sha256']:
        raise ValueError('Paired product changed after its manifest')
    with np.load(source,allow_pickle=False) as f:d={k:np.array(f[k]) for k in f.files}
    if str(d['schema'])!='quark_ringed_paired_10conf_v1':raise ValueError('Expected measured paired ensemble product')
    np.testing.assert_array_equal(d['confs'],np.arange(10000,11000,100))
    if d['bootstrap_indices'].shape!=(50,10) or d['kinetic_noise'].shape!=(len(d['flow_times']),10,32):
        raise ValueError('Need full ten configurations, 32 noises and original 50 replicas')
    if not np.isfinite(d['kinetic_noise']).all() or not np.isfinite(d['ringed_bootstrap']).all():
        raise ValueError('Incomplete measurements')
    if not d['kinetic_imaginary_check_pass'].all():raise ValueError('Kinetic imaginary-part check failed')
    np.testing.assert_allclose(d['ringed_bootstrap'],d['bare_bootstrap']*d['Zchi_bootstrap'][:,:,None,None],rtol=1e-14,atol=1e-15)
    np.testing.assert_allclose(d['ringed_error'],d['ringed_bootstrap'].std(axis=1,ddof=1),rtol=1e-14,atol=1e-15)
    bare=Path(meta['input_bare']['path'])
    if digest(bare)!=meta['input_bare']['sha256']:raise ValueError('Changed original bare input')
    with np.load(bare,allow_pickle=False) as f:
        np.testing.assert_array_equal(d['bootstrap_indices'],f['bootstrap_indices'])
        np.testing.assert_array_equal(d['bare_central'],f['M_filled'] if 'M_filled' in f else f['M'])
    return d,meta


def plot(source,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42})
    d,meta=load_checked(source)
    output.mkdir(parents=True,exist_ok=False)
    flow,z,p=d['flow_times'],d['delta_z_values'],d['momenta_integer_pz']
    palette={.05:'#6a3d9a',.1:'#4266a3',.2:'#25849c',.5:'#159b78',.8:'#81a731',1.0:'#dc8b20'}
    colors=[palette[round(float(t),2)] for t in flow]
    shifts=np.linspace(-.20,.20,len(flow))
    method='Sumratio difference AIC' if len(flow)==6 else 'Direct ratio AIC (unweighted diagnostic)'
    records=[]

    def draw(ax,ip,selection,zmax):
        for i in selection:
            y,e=d['ringed_central'][i,ip],d['ringed_error'][i,ip]
            for good in (True,False):
                s=(d['fit_valid'][i,ip]==good)&(z<=zmax)
                ax.errorbar(z[s]+shifts[i],y[s],yerr=e[s],fmt='o',ls='none',ms=4,
                    capsize=2,elinewidth=.85,color=colors[i],mfc=colors[i] if good else 'white',
                    alpha=.95 if good else .65)
        pg=2*np.pi*p[ip]/32*.1973269804/float(d['a_fm'])
        ax.set(title=rf'$n_z={p[ip]},\ P_z={pg:.2f}$ GeV',xlabel=r'$z/a$',
            ylabel=r'$\mathrm{Re}\,h^{\rm ringed}_{u-d}=Z_\chi\,\mathrm{Re}\,h^{\rm GF}_{u-d}$' if ip==0 else '',
            xlim=(-.5,zmax+.5))
        ax.axhline(0,color='.6',ls='--',lw=.7);ax.grid(alpha=.16)
        ax.tick_params(direction='in',top=True,right=True)

    def page(pdf,selection,zmax,stem,index,subtitle):
        fig,axs=plt.subplots(1,3,figsize=(15.5,5.8))
        for ip,ax in enumerate(axs):draw(ax,ip,selection,zmax)
        fig.suptitle('After ringed normalization | '+method,y=.99)
        handles=[Line2D([],[],color=colors[i],marker='o',ls='none',label=rf'$t/a^2={flow[i]:g}$') for i in selection]
        fig.legend(handles=handles,ncol=len(selection),frameon=False,loc='upper center',bbox_to_anchor=(.5,.94))
        fig.text(.5,.84,subtitle,ha='center',fontsize=10)
        fig.text(.5,.025,'10 matched configurations; 32 noises/configuration; 50 original paired bootstrap replicas.\nOpen: fit fallback / unweighted diagnostic. Published critical valence mass; SFTE factor not applied.',ha='center',fontsize=9)
        fig.subplots_adjust(left=.075,right=.985,top=.76,bottom=.19,wspace=.28)
        pdf.savefig(fig);fig.savefig(output/f'{stem}_page{index}.png',dpi=150)
        records.append({'pdf':stem+'.pdf','page':index,'flow_times':flow[selection].tolist(),
                        'xlim':list(axs[0].get_xlim()),'ylim':[list(a.get_ylim()) for a in axs]})
        plt.close(fig)

    for stem,selection,subtitle in [
        ('after_Zchi_allflow',np.arange(len(flow)),'All available flow times; complete error-bar ranges'),
        ('after_Zchi_tau_ge_0p2',np.flatnonzero(flow>=.2-1e-12),'Display subset: t/a^2 >= 0.2; no data refit')]:
        with PdfPages(output/(stem+'.pdf')) as pdf:
            for i,zmax in enumerate((15,20),1):page(pdf,selection,zmax,stem,i,subtitle)
    with PdfPages(output/'after_Zchi_eachflow.pdf') as pdf:
        for i,t in enumerate(flow):page(pdf,np.array([i]),15,'after_Zchi_eachflow',i+1,rf'$t/a^2={t:g}$')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    axs[0].errorbar(flow,d['Zchi'],yerr=d['Zchi_error'],fmt='o',ls='none',capsize=3)
    axs[0].set(xlabel=r'$t/a^2$',ylabel=r'$Z_\chi$',title='10-configuration ensemble normalization')
    axs[1].plot(flow,100*d['Zchi_error']/d['Zchi'],'o',ls='none')
    axs[1].set(xlabel=r'$t/a^2$',ylabel='Relative bootstrap error [%]',title='50 shared configuration draws')
    for a in axs:a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'Zchi_10conf.pdf');plt.close(fig)
    report={'input':str(source.resolve()),'input_sha256':digest(source),'script_sha256':digest(__file__),
            'error':'std of 50 paired Zchi_b * h_b samples, ddof=1','SFTE_applied':False,
            'confs':d['confs'].tolist(),'flow_times':flow.tolist(),'Zchi':d['Zchi'].tolist(),
            'Zchi_error':d['Zchi_error'].tolist(),'offsets':shifts.tolist(),'connecting_lines':False,'pages':records,
            'source_physics_status':str(d['status'])}
    (output/'plot_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('analysis',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new figure directory')
    for kind in ('sixflow_sumdiff','fourflow_ratio'):
        report=plot(a.analysis/kind/'paired_matrix_elements.npz',a.output/kind)
        print(json.dumps({'method':kind,'Zchi':report['Zchi'],'error':report['Zchi_error']},indent=2),flush=True)


if __name__=='__main__':main()
