#!/usr/bin/env python3
"""Extend quasi support after the first no-z0 numerical-convergence audit.

Preserves the parent output and all original fit parameters/replica identities.
Only the numerical Fourier/convolution domain changes.
"""
import argparse
import json
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from lambda_fourier_quasi_no_z0_v1 import SOURCE,transform,plot_results,physical_pz,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE.parent/'lambda_fourier_quasi_no_z0_v1')
    p.add_argument('--out',type=Path)
    a=p.parse_args();out=a.out or a.source/'quasi_support_refined'
    if out.exists() and any(out.iterdir()):raise FileExistsError('refuse overwrite')
    m=json.loads((a.source/'manifest.json').read_text());args=SimpleNamespace(**m['settings'])
    with np.load(a.source/'no_z0_xg.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    x=d['x'];y=np.linspace(-12,12,2401);pabs=d['pabs_values'];h=d['H'];lam=d['lambda_data']
    E=d['energy_lattice'];tails=[];local=[];local_values=[];results=[];variants=[];checks=[]
    out.mkdir(parents=True);per=out/'per_momentum';per.mkdir()
    for ip,n in enumerate(pabs):
        with np.load(a.source/'per_momentum'/f'Pz{n}.npz',allow_pickle=False) as f:dd={k:f[k] for k in f.files}
        tail={k[5:]:v for k,v in dd.items() if k.startswith('tail_')};tails.append(tail)
        local.append({k[6:]:v for k,v in dd.items() if k.startswith('local_') and not k.startswith('local_unexpanded_')})
        local_values.append(dd['local_matched'])
        ha=h[ip].copy();ha[:,0]=np.r_[dd['local_unexpanded_params_selected'][0],dd['local_unexpanded_params_boot_selected'][:,0]]*dd['kinematic_factor']
        oldlocal=h[ip].copy();oldlocal[:,0]=d['source_R'][:,ip,0]*dd['kinematic_factor']
        P=float(physical_pz(n,args.a_fm));r=transform(h[ip],tail,lam[ip],P,x,y,args,xi_max=200.);results.append(r)
        var={};err=np.std(r['matched'][1:],axis=0,ddof=1);region=(x>=.2-1e-10)&(x<=.8+1e-10)
        specs=[('join7',h[ip],dict(join=7.),y),('join9',h[ip],dict(join=9.),y),
            ('cut2',h[ip],dict(cut=2),y),('local_unexpanded',ha,{},y),('old_endpoint_local',oldlocal,{},y),
            ('order16',h[ip],dict(order=16),y),('lambda320',h[ip],dict(stop=320.),y),
            ('kernel96',h[ip],dict(kernel_order=96),y),('xi400',h[ip],dict(xi_max=400.),y),
            ('y_halfstep',h[ip],{},np.linspace(-12,12,4801)),
            ('y_support16',h[ip],{},np.linspace(-16,16,3201)),('y_support24',h[ip],{},np.linspace(-24,24,4801))]
        for key,hh,kw,yy in specs:
            opts=dict(xi_max=200.);opts.update(kw)
            rr=transform(hh,tail,lam[ip],P,x,yy,args,**opts);var[key]=rr['matched']
            delta=rr['matched']-r['matched']
            checks.append(dict(n_z=int(n),variation=key,max_central_shift=float(np.max(abs(delta[0,region]))),
                max_shift_over_stat=float(np.max(abs(delta[0,region])/err[region])),
                max_paired_shift_error=float(np.max(np.std(delta[1:,region],axis=0,ddof=1)))))
        variants.append(var)
        # Do not carry obsolete numerical variations as if tested at the new domain.
        dd={k:v for k,v in dd.items() if not k.startswith('variation_')}
        dd.update(r);dd.update({'variation_'+k:v for k,v in var.items()});dd['quasi_grid']=y
        np.savez_compressed(per/f'Pz{n}.npz',**dd)
        numerical=[c['max_shift_over_stat'] for c in checks if c['n_z']==n and c['variation'] in
            {'order16','lambda320','kernel96','xi400','y_halfstep','y_support16','y_support24'}]
        print(f'n_z={n}, maximum numerical variation in x=.2..8: {max(numerical):.5f} sigma',flush=True)
    plot_results(out,pabs,x,h,lam,tails,results,variants,np.stack(local_values,axis=-1),local,E,d['c2_tsep'],d['c2'],args)
    d['matched']=np.stack([r['matched'] for r in results]);d['quasi_hybrid']=np.stack([r['quasi_hybrid'] for r in results])
    np.savez_compressed(out/'no_z0_xg.npz',**d)
    m.update(generated_utc=datetime.now(timezone.utc).isoformat(),schema='no_z0_absolute_gluon_support_refinement_v1',
        parent_output=str(a.source),parent_product_sha256=digest(a.source/'no_z0_xg.npz'),
        checks=checks,quasi_y_support=[-12.,12.],quasi_y_step=.01,xi_max=200.,
        refinement='Only Fourier/convolution numerical settings; all fits and replica identities unchanged')
    m['scripts'][str(Path(__file__).resolve())]=digest(__file__)
    (out/'manifest.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
    print('FINISHED '+str(out),flush=True)


if __name__=='__main__':main()
