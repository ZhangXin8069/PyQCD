#!/usr/bin/env python3
"""Redraw existing N406/Nboot1000 unpolarized GF-converted matrix elements.

No matching, refitting, resampling, or input mutation is performed. The main
operator is identified by its saved label, not by an assumed array index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from plot_matched_selected_taus_by_pz_v1 import DEFAULT_ROOT, load_products, product_path

BASE = Path(__file__).resolve().parent
LABEL = 'unpolarized_TU_minus_SU_even'
TAUS = np.array([.1,.2,.3,.4,.5,.6,1.,1.4,1.8,2.2,2.6,3.,3.4,3.8])
SELECTED = np.array([.5,1.,1.4,1.8])


def array_digest(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode()); h.update(a.dtype.str.encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def load_and_audit(root):
    products = load_products(root, TAUS)
    records, reference, settings = [], None, None
    for tau, row in products.items():
        path = product_path(root, tau)
        with np.load(path, allow_pickle=False) as f:
            labels = f['operator_labels'].astype(str).tolist()
            if labels.count(LABEL) != 1:
                raise ValueError(f'{path}: missing/duplicate unpolarized operator')
            row['operator_index'] = labels.index(LABEL)
            if f['operator_projection'][row['operator_index']] != 'real':
                raise ValueError('Unpolarized projection must be real')
            if list(f['axes_M']) != ['operator','momentum_abs','z']:
                raise ValueError('Unexpected matrix axes')
            if not np.isclose(float(f['flow_tau_t_over_a2']),tau,rtol=0,atol=1e-12):
                raise ValueError('Flow time does not match filename')
            if str(f['matching_status']) != 'one_loop_GF_to_MSbar_quasi_after_bare_fit':
                raise ValueError('Input is not the expected GF-to-MSbar product')
            indices = {'confs': f['confs'], 'bootstrap_indices': f['bootstrap_indices']}
            if reference is None:
                reference = indices
            else:
                for key in indices:
                    np.testing.assert_array_equal(reference[key], indices[key])
            this_settings = {key: float(f['matching_'+key])
                             for key in ['a_fm','mu_GeV','alpha_s','kappa_F']}
            if settings is None:
                settings = this_settings
            elif settings != this_settings:
                raise ValueError('Different matching parameters across flow times')
        if reference['bootstrap_indices'].shape != (1000,406):
            raise ValueError('Unexpected bootstrap indices')
        if not np.all(row['filled_statistical_error'][row['estimate_available']] >= 0):
            raise ValueError('Negative or nonfinite plotting error')
        receipt_path = Path(str(path) + '.done.json')
        receipt = json.loads(receipt_path.read_text())
        if receipt['status'] != 'complete':
            raise ValueError('Matching completion receipt is not complete')
        records.append({'path': str(path.resolve()), 'tau': tau,
                        'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns,
                        'output_sha256_from_completion_receipt': receipt['output_sha256'],
                        'plot_array_sha256': array_digest({k: v for k,v in row.items() if isinstance(v,np.ndarray)})})
    return products, records, settings, array_digest(reference)


def draw_series(ax, row, ip, zmask, color, shift, ylim):
    op = row['operator_index']
    z = row['z_values'][zmask]
    y = row['M_filled'][op,ip,zmask]
    e = row['filled_statistical_error'][op,ip,zmask]
    available = row['estimate_available'][op,ip,zmask] & np.isfinite(y) & np.isfinite(e)
    accepted = row['fit_accepted'][op,ip,zmask]
    fallback = row['central_fallback_used'][op,ip,zmask]
    cuts = row['usable_cut_count'][op,ip,zmask]
    masks = {'accepted_multi_cut': available & accepted & ~fallback & (cuts>=2),
             'accepted_single_cut': available & accepted & ~fallback & (cuts<2),
             'fallback': available & fallback,
             'other_diagnostic': available & ~accepted & ~fallback}
    for key, mask in masks.items():
        if not mask.any():
            continue
        marker = 'X' if key=='fallback' else ('D' if key=='other_diagnostic' else 'o')
        face = color if key in ('accepted_multi_cut','fallback') else 'white'
        ax.errorbar(z[mask]+shift,y[mask],yerr=e[mask],fmt=marker,linestyle='none',
                    color=color,mfc=face,mec=color,ms=4.,capsize=1.4,elinewidth=.65,alpha=.88)
    return {**{key:int(mask.sum()) for key,mask in masks.items()},
            'center_outside_ylim':int((available & ((y<ylim[0]) | (y>ylim[1]))).sum()),
            'errorbar_extends_outside_ylim':int((available & ((y-e<ylim[0]) | (y+e>ylim[1]))).sum())}


def plot_page(products, taus, settings, args, name):
    colors = (['#1f77b4','#d62728','#2ca02c','#9467bd'] if len(taus)==4
              else [plt.get_cmap('tab20')(i) for i in range(14)])
    shifts = np.linspace(-.25,.25,len(taus)) if len(taus)>4 else np.array([-.12,-.04,.04,.12])
    fig, axes = plt.subplots(1,3,figsize=(16.5,5.7),sharey=True)
    summary = {'page':name,'flow_times':taus.tolist(),'x_offsets':shifts.tolist(),'panels':[]}
    for ip,pz in enumerate([3,4,5]):
        ax=axes[ip]
        counts = {}
        for tau,c,shift in zip(taus,colors,shifts):
            row=products[float(tau)]
            zmask=(row['z_values']>=0)&(row['z_values']<=args.zmax)
            counts[str(float(tau))]=draw_series(ax,row,ip,zmask,c,shift,(args.ymin,args.ymax))
        ax.set_ylim(args.ymin,args.ymax)
        ax.set_xlim(-.5,args.zmax+.5)
        ax.set_xticks(np.arange(0,args.zmax+1,3))
        ax.set_yticks(np.arange(args.ymin,args.ymax+.001,.5))
        ax.axhline(0.,c='.5',lw=.7,ls='--')
        ax.grid(alpha=.2,lw=.6)
        ax.tick_params(direction='in',top=True,right=True)
        ax.set_xlabel(r'$z/a$',fontsize=12)
        ax.set_title(rf'$n_z={pz}$, $P_z=2\pi n_z/(32a)$',fontsize=12)
        summary['panels'].append({'momentum_integer':pz,'ylim':list(ax.get_ylim()),
                                   'xlim':list(ax.get_xlim()),'counts_by_tau':counts})
    axes[0].set_ylabel(r'$\mathrm{Re}\,h^{\mathrm{GF}\to\overline{\mathrm{MS}}}_{g,\,T_U-S_U}$',fontsize=12)
    handles=[Line2D([],[],c=c,marker='o',ls='none',ms=4,label=rf'$t_f/a^2={t:g}$')
             for t,c in zip(taus,colors)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.995),
               ncol=7 if len(taus)>4 else 4,frameon=False,fontsize=10)
    subtitle=(f'Nconf=406, Nboot=1000; mu={settings["mu_GeV"]:g} GeV, '
              f'alpha_s={settings["alpha_s"]:g}; y=[{args.ymin:g},{args.ymax:g}]')
    fig.suptitle('Unpolarized gluon: '+r'$(T_U-S_U)_{\rm even}$'
                 +' | '+('all 14 flow times' if len(taus)>4 else 'selected 4 flow times')
                 +'\n'+subtitle,y=.85,fontsize=12.5)
    fig.text(.5,.065,'Filled: accepted, >=2 cuts; open: accepted, 1 cut; X: fallback; diamond: other diagnostic.',
             ha='center',fontsize=9)
    fig.text(.5,.025,'Finite-a/finite-flow GF conversion, not a light-cone PDF. '
             'z=0: inherited formal conversion; clipping changes display only.',ha='center',fontsize=9,color='.35')
    fig.subplots_adjust(left=.075,right=.985,bottom=.2,top=.69,wspace=.09)
    return fig,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matched-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--out',type=Path,default=BASE/'bare_all_operators_matching_v1/plots_unpolarized_ylim_m1p5_1p0')
    p.add_argument('--ymin',type=float,default=-1.5)
    p.add_argument('--ymax',type=float,default=1.)
    p.add_argument('--zmax',type=int,default=15)
    args=p.parse_args()
    if not np.isfinite([args.ymin,args.ymax]).all() or args.ymin>=args.ymax or not 1<=args.zmax<=24:
        p.error('Invalid axis range')
    products,records,settings,index_hash=load_and_audit(args.matched_root)
    print('Audited 14 matched flow products; common 406 configs and 1000 bootstrap index rows.',flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    pdf_path=args.out/'unpolarized_TU_minus_SU_matched_all_Pz.pdf'
    pages=[]
    with PdfPages(pdf_path) as pdf:
        for name,taus in [('all14',TAUS),('selected4',SELECTED)]:
            fig,summary=plot_page(products,taus,settings,args,name)
            pdf.savefig(fig)
            fig.savefig(args.out/f'unpolarized_TU_minus_SU_matched_{name}.png',dpi=200)
            plt.close(fig)
            pages.append(summary)
    for record in records:
        s=Path(record['path']).stat()
        if (s.st_size,s.st_mtime_ns)!=(record['bytes'],record['mtime_ns']):
            raise RuntimeError('An input product changed during plotting')
    manifest={'status':'plot_complete_existing_GF_conversion_diagnostic_not_lightcone_PDF',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'pdf':str(pdf_path.resolve()),
        'operator':LABEL,'nconf':406,'nboot':1000,'matching':settings,'input_records':records,
        'common_config_and_bootstrap_index_sha256':index_hash,
        'checksum_scope':'plotted arrays and indices hashed; full input file hashes copied from completion receipts, not recomputed',
        'script':str(Path(__file__).resolve()),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'connecting_lines':False,'input_mutation':False,'error_definition':'saved filled_statistical_error, unchanged',
        'pages':pages}
    (args.out/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'pdf':str(pdf_path.resolve()),'ylim':[args.ymin,args.ymax],
                      'pages':[page['page'] for page in pages]},indent=2))


if __name__=='__main__':
    main()
