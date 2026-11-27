"""Compare direct two-state ratios with the previous adjacent/constant chain."""

import csv
import importlib.util
import json

import matplotlib
import numpy as np
from scipy.stats import chi2

from fit import ENSEMBLES, NBOOT, PREVIOUS, ROOT, design, digest, write_csv

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


BASE = ROOT.parent/'quark_gf_cut2_dz005_common_lambda_20261126'
CODE = ROOT.parent/'quark_gf_cut2_moving_continuum_20261125'/'analyze.py'
ORDER = list(ENSEMBLES.values())
LABELS = (1.5, 2., 2.5)
PALETTE = ('tab:blue','tab:orange','tab:green')


def load_verified(root, receipt, relative):
    path = root/relative
    assert digest(path)==receipt['outputs_sha256'][relative]
    return np.load(path,allow_pickle=False)


def normalize(pack):
    pars = pack['parameters']
    values = pars[...,0]+1j*pars[...,3]
    denominator = values[:,0].real
    norm_boot = denominator[:,1:]
    median = np.median(abs(norm_boot),axis=1)
    sd = norm_boot.std(axis=1,ddof=1)
    assert np.all(denominator != 0)
    sign_match = np.sign(norm_boot)==np.sign(denominator[:,:1])
    safe = ((abs(denominator[:,0]) >= 3*sd) & (sign_match.mean(axis=1)>=.99)
            & (np.quantile(abs(norm_boot),.01,axis=1)>=.05*median))
    normalized = values/denominator[:,None,:]
    good = pack['good'] & pack['good'][:,:1] & safe[:,None]
    replica_good = pack['good_boot'] & pack['good_boot'][:,:1] & safe[:,None,None] & sign_match[:,None,:]
    good[:,0] = True
    replica_good[:,0] = True
    np.testing.assert_allclose(normalized[:,0], 1, rtol=0, atol=1e-14)
    normalized[:,0] = 1
    assert np.isfinite(normalized).all()
    return normalized, good, replica_good, safe


def plot_bare(path, name, values, errors, z, radii, momenta, good):
    fig, axes = plt.subplots(2, len(momenta), figsize=(3.3*len(momenta),6.3), sharex=True, squeeze=False)
    colors = plt.cm.viridis(np.linspace(.12,.87,len(radii)))
    for p_index,momentum in enumerate(momenta):
        for component, axis in enumerate(axes[:,p_index]):
            for rf_index,(radius,color) in enumerate(zip(radii,colors)):
                for valid in (True,False):
                    show = good[rf_index,p_index]==valid
                    points = (values.real if component==0 else values.imag)[rf_index,p_index]
                    axis.errorbar(z[show],points[show],yerr=errors[rf_index,p_index,:,component][show],
                                  color=color,fmt='o',mfc=color if valid else 'white',ms=2.6,lw=.7,
                                  label=f'{radius:.3f}' if not valid else None)
            axis.set_yscale('symlog',linthresh=1.)
            axis.grid(alpha=.2)
            if component==0:
                axis.set_title(f'n={momentum}')
            else:
                axis.set_xlabel('z [fm]')
            if p_index==0:
                axis.set_ylabel('Re M (symlog)' if component==0 else 'Im M (symlog)')
    handles, labels = axes[0,-1].get_legend_handles_labels()
    fig.legend(handles,labels,title=r'$r_f$ [fm]',loc='lower center',ncol=len(radii),frameon=False)
    fig.suptitle(name+' — ordinary C3/C2 two-state bare matrix; open = diagnostic')
    fig.tight_layout(rect=(0,.11,1,.95))
    fig.savefig(path)
    plt.close(fig)


def plot_signal(pdf, name, pack):
    momenta = pack['pz_values']
    p_index = np.flatnonzero(momenta==4).item()
    sample_z = int(round(.45/pack['a_fm']))
    ts, ti = pack['T'],pack['tinsertion']
    pars = pack['parameters'][p_index]
    ratio = pack['origin_plus_boot_ratio'][:,p_index]
    fig, axes = plt.subplots(2,2,figsize=(10,6.4))
    for column,z_index in enumerate((0,sample_z)):
        for component,axis in enumerate(axes[:,column]):
            if component and z_index==0:
                axis.axis('off'); continue
            observed = ratio[:,z_index].real if not component else ratio[:,z_index].imag
            coeff = pars[z_index,0,component*3:component*3+3]
            gap = pars[z_index,0,6]*float(pack['a_fm'])/.1973269804
            for time,color in zip(np.unique(ts),plt.cm.viridis(np.linspace(.1,.85,len(np.unique(ts))))):
                used = ts==time
                axis.errorbar(ti[used], observed[0,used],yerr=observed[1:,used].std(axis=0,ddof=1),fmt='o',ms=3,color=color,label=f'T={time}')
                ins = np.linspace(2,time-2,100)
                features,_ = design(gap, np.full_like(ins,time),ins,1)
                axis.plot(ins,features@coeff,color=color,lw=.9)
            axis.set_title(f'{"Re" if not component else "Im"} $C_3/C_2$, z={z_index*float(pack["a_fm"]):.3f} fm, Q={pack["q"][p_index,z_index,0]:.2g}')
            axis.set_xlabel('insertion time / a')
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
    fig.suptitle(f'{name}, n=4, $r_f={float(pack["rf_fm"]):.5f}$ fm; fixed cut2 window')
    fig.tight_layout(rect=(0,0,1,.95))
    pdf.savefig(fig)
    plt.close(fig)


def load_sources(fit_receipt, old_receipt, continuum, core):
    target_p, common_rf = continuum['pz_GeV'],continuum['rf_fm']
    bare_dir = ROOT/'01_bare_two_state'
    factor_rows = PREVIOUS/'03_H3_subtracted_native_z/matrix_elements.csv'
    assert digest(factor_rows)==old_receipt['outputs_sha256'][str(factor_rows.relative_to(PREVIOUS))]
    factors = {}
    with factor_rows.open() as stream:
        for row in csv.DictReader(stream):
            factors[row['ensemble'],float(row['rf_fm'])] = float(row['H3_d'])
    sources, norm_rows = [], []
    with PdfPages(bare_dir/'ratio_fit_examples_P4.pdf') as pdf:
        for name in ORDER:
            flows, diagnostics = [], []
            with load_verified(PREVIOUS,old_receipt,f'01_bare_local_ratio/{name}.npz') as previous:
                z, native_p = previous['z_fm'],previous['pz_GeV']
                for rf_index,radius in enumerate(previous['rf_fm']):
                    tag = f'tau{previous["tau_over_a2"][rf_index]:.8f}'.replace('.','p')
                    with load_verified(ROOT,fit_receipt,f'01_bare_two_state/{name}_{tag}.npz') as pack:
                        assert abs(float(pack['rf_fm'])-radius)<1e-12
                        for field in ('confs','boot_indices'):
                            np.testing.assert_array_equal(pack[field],previous[field])
                        normalized,good,good_boot,safe = normalize(pack)
                        native_n = pack['pz_values']
                        bare = pack['parameters'][...,0]+1j*pack['parameters'][...,3]
                        diagnostics.append((bare[:,:,0],core.complex_columns(bare[:,:,1:]).std(axis=2,ddof=1),pack['good']))
                        if np.isclose(radius,.1794,atol=2e-6,rtol=0):
                            plot_signal(pdf,name,pack)
                        for p_index,momentum in enumerate(pack['pz_values']):
                            norm_rows.append(dict(ensemble=name,rf_fm=radius,n=int(momentum),
                                                  bare_M0=bare[p_index,0,0].real,
                                                  bare_M0_sd=bare[p_index,0,1:].real.std(ddof=1),
                                                  M0_two_state_quality=pack['good'][p_index,0], M0_division_stable=safe[p_index]))
                        if not np.any(np.isclose(radius, common_rf, atol=1e-12, rtol=0)):
                            continue
                        factor = np.exp(factors[name,radius]*abs(z)/radius)
                        aligned = []
                        for p_index, target in enumerate(target_p):
                            inds, weights, beyond = core.bracket(native_p[1:],target,True)
                            inds += 1
                            assert beyond == (name != 'F32P30' and p_index==0)
                            matched = np.einsum('p,pzb->zb',weights,normalized[inds])*factor[:,None]
                            good_p = good[inds].all(axis=0)
                            good_replica_p = good_boot[inds].all(axis=0)
                            interp = [core.bracket(z,position) for position in continuum['z_fm'][p_index]]
                            aligned.append((np.stack([w@matched[j] for j,w,_ in interp]),
                                            np.array([good_p[j].all() for j,_,_ in interp]),
                                            np.stack([good_replica_p[j].all(axis=0) for j,_,_ in interp])))
                        flows.append((radius, *[np.stack(items) for items in zip(*aligned)]))
                plot_bare(bare_dir/f'{name}_bare_matrix.pdf',name,*[np.stack(items) for items in zip(*diagnostics)][:2],
                          z,previous['rf_fm'],native_n,
                          np.stack([item[2] for item in diagnostics]))
            np.testing.assert_allclose([item[0] for item in flows],common_rf,atol=1e-12,rtol=0)
            sources.append(tuple(np.stack(items) for items in zip(*[item[1:] for item in flows])))
    write_csv(bare_dir/'local_normalization_audit.csv',norm_rows)
    return sources,norm_rows


def stable_continuum(core, center, draws, spacing):
    observations = core.complex_columns(center)
    samples = core.complex_columns(draws)
    deviations = samples - samples.mean(axis=1, keepdims=True)
    _, input_singular, input_vh = np.linalg.svd(deviations/np.sqrt(NBOOT-1), full_matrices=False)
    assert np.all(input_singular[:,1]>np.finfo(float).eps*NBOOT*input_singular[:,0])
    whitener = input_vh/input_singular[...,None]
    scaled = (spacing/spacing[0])**2
    design = np.concatenate((np.broadcast_to(np.eye(2), (3,2,2)),scaled[:,None,None]*np.eye(2)),axis=2)
    weighted = (whitener@design).reshape(6,4)
    scales = np.linalg.norm(weighted,axis=0)
    left, singular, right = np.linalg.svd(weighted/scales,full_matrices=False)
    assert singular[-1]>np.finfo(float).eps*6*singular[0]
    inverse = (right.T/singular)@left.T/scales[:,None]
    linear_map = np.einsum('fei,eij->fej',inverse.reshape(4,3,2),whitener)
    np.testing.assert_allclose(linear_map.reshape(4,6)@design.reshape(6,4),np.eye(4),atol=1e-7)
    beta = np.einsum('fei,ei->f',linear_map,observations)
    weighted_observations = (whitener@observations[...,None]).ravel()
    orthogonal_residual = weighted_observations - left@(left.T@weighted_observations)
    chisq = float(orthogonal_residual@orthogonal_residual)
    covariance = inverse@inverse.T
    increments = np.einsum('fei,ebi->ebf',linear_map,deviations)
    normalization = np.sqrt(np.outer(covariance.diagonal(),covariance.diagonal()))
    np.testing.assert_allclose(np.einsum('ebf,ebg->fg',increments,increments)/((NBOOT-1)*normalization),
                               covariance/normalization,rtol=1e-8,atol=1e-8)
    return dict(beta=beta, cov=covariance, source_draws=increments, chisq=chisq,
                Q=float(chi2.sf(chisq,2)), dof=2, input_cov_condition=max((input_singular[:,0]/input_singular[:,1])**2),
                scaled_design_condition=singular[0]/singular[-1])


def continuum_fit(core, old, sources):
    shape = old['beta'].shape[:-1]
    beta = np.zeros((*shape,4)); beta[...,0] = 1.
    covariance = np.zeros((*shape,4,4))
    increments = np.zeros((*shape,3,NBOOT,4))
    all_pass = np.zeros(shape,dtype=bool)
    fit_pass = np.zeros(shape,dtype=bool)
    rows, aligned_rows = [], []
    aligned = np.stack([item[0] for item in sources],axis=-1)
    aligned_good = np.stack([item[1] for item in sources],axis=-1)
    aligned_good_boot = np.stack([item[2] for item in sources],axis=-1)
    for rf_index,p_index,point_index in np.ndindex(shape):
        key = rf_index,p_index,point_index
        coord = dict(rf_fm=old['rf_fm'][rf_index], Pz_GeV=old['pz_GeV'][p_index], lambda_value=old['coordinate'][point_index],
                     z_fm=old['z_fm'][p_index,point_index])
        for e_index,name in enumerate(ORDER):
            central, replicates = aligned[key][:,e_index][0],aligned[key][:,e_index][1:]
            aligned_rows.append(dict(**coord,ensemble=name,**core.point_fields(central,core.complex_columns(replicates).std(axis=0,ddof=1)),
                                     new_input_pass=aligned_good[key][e_index],
                                     replica_chain_pass_fraction=aligned_good_boot[key][:,e_index].mean(),
                                     old_real=old['aligned_center'][key][e_index].real,
                                     old_imag=old['aligned_center'][key][e_index].imag))
        if point_index==0:
            np.testing.assert_allclose(aligned[key],1,atol=1e-14,rtol=0)
            continue
        equivalent_old = stable_continuum(core,old['aligned_center'][key],old['aligned_boot'][key],old['a_fm'])
        np.testing.assert_allclose(equivalent_old['beta'],old['beta'][key],rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(equivalent_old['cov'],old['parameter_covariance'][key],rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(equivalent_old['Q'],old['q'][key],atol=1e-10)
        fit = stable_continuum(core,aligned[key][0],aligned[key][1:].T,old['a_fm'])
        beta[key],covariance[key],increments[key] = fit['beta'],fit['cov'],fit['source_draws']
        all_pass[key] = aligned_good[key].all()
        fit_pass[key] = (fit['Q']>=.05 and fit['chisq']/fit['dof']<=3)
        paired_diff = np.sqrt(((increments[key]-old['independent_source_increments'][key])[:,:,:2]**2).sum(axis=(0,1))/(NBOOT-1))
        old_errors = np.sqrt(old['parameter_covariance'][key].diagonal()[:2])
        rows.append(dict(**coord,**core.point_fields(complex(*fit['beta'][:2]),np.sqrt(fit['cov'].diagonal()[:2])),
                         paired_delta_real_sd=paired_diff[0],paired_delta_imag_sd=paired_diff[1],
                         old_real=old['beta'][key][0],old_imag=old['beta'][key][1],old_real_sd=old_errors[0],old_imag_sd=old_errors[1],
                         old_Q=old['q'][key],chi2_dof=fit['chisq']/fit['dof'],Q=fit['Q'],fit_pass=fit_pass[key],
                         input_cov_condition=fit['input_cov_condition'],scaled_design_condition=fit['scaled_design_condition'],
                         C24_input_pass=aligned_good[key][0],E32_input_pass=aligned_good[key][1],F32_input_pass=aligned_good[key][2],
                         all_inputs_pass=all_pass[key],uses_momentum_extrapolation=p_index==0,
                         passes_core_gates=(all_pass[key] and fit_pass[key] and p_index>0)))
    out = ROOT/'02_fixed_lambda'
    out.mkdir(exist_ok=True)
    write_csv(out/'continuum.csv',rows); write_csv(out/'aligned_ensembles.csv',aligned_rows)
    np.savez_compressed(out/'continuum.npz',beta=beta,parameter_covariance=covariance,independent_source_increments=increments,
                        all_inputs_pass=all_pass,fit_pass=fit_pass,aligned_origin_plus_boot=aligned,aligned_good=aligned_good,
                        aligned_good_boot=aligned_good_boot,ensemble_order=ORDER,
                        **{key: old[key] for key in ('coordinate','z_fm','pz_GeV','a_fm','rf_fm')})
    with PdfPages(out/'two_state_vs_adjacent_fixed_lambda.pdf') as pdf:
        for rf_index,radius in enumerate(old['rf_fm']):
            fig,axes = plt.subplots(2,3,figsize=(12,6.3),sharex=True)
            for p_index,label in enumerate(LABELS):
                for component,axis in enumerate(axes[:,p_index]):
                    key = rf_index,p_index,slice(None),component
                    for array,cov,color,marker,method in ((old['beta'],old['parameter_covariance'],'0.45','o','Adjacent constant'),
                                                         (beta,covariance,PALETTE[p_index],'s','Direct two-state')):
                        axis.errorbar(old['coordinate'],array[key],yerr=np.sqrt(cov[key+(component,)]),
                                      color=color,fmt=marker+'--',mfc='white',ms=3,lw=.85,label=method)
                    axis.set_yscale('symlog',linthresh=1.)
                    axis.axhline(0,color='0.7',lw=.6)
                    axis.grid(alpha=.2)
                    if component==0: axis.set_title(f'$P_z={label:.1f}$ GeV')
                    else: axis.set_xlabel(r'$\lambda$')
                    if p_index==0: axis.set_ylabel('Re H (symlog)' if component==0 else 'Im H (symlog)')
            axes[0,0].legend(frameon=False)
            fig.suptitle(f'Finite-flow $a^2$ continuum diagnostics, $r_f={radius:.5f}$ fm')
            fig.tight_layout(rect=(0,0,1,.95))
            pdf.savefig(fig)
            plt.close(fig)
    crossings = zero_crossing_audit(out, old, beta, covariance)
    plot_continuum_examples(out, old, beta, covariance, aligned, rows)
    return rows, crossings, aligned_rows


def zero_crossing_audit(out, old, beta, covariance):
    rows = []
    p_index = LABELS.index(2.0)
    for rf_index, radius in enumerate(old['rf_fm']):
        row = {'rf_fm': float(radius)}
        for tag, results, errors in (('adjacent', old['beta'], old['parameter_covariance']),
                                      ('two_state', beta, covariance)):
            real = results[rf_index,p_index,:,0]
            sigma = np.sqrt(errors[rf_index,p_index,:,0,0])
            negative = np.flatnonzero(real[1:] < 0) + 1
            index = int(negative[0]) if len(negative) else None
            row[f'{tag}_first_negative_lambda'] = '' if index is None else old['coordinate'][index]
            row[f'{tag}_first_negative_z_fm'] = '' if index is None else old['z_fm'][p_index,index]
            row[f'{tag}_real_at_first_negative'] = '' if index is None else real[index]
            row[f'{tag}_sigma_at_first_negative'] = '' if index is None else sigma[index]
            row[f'{tag}_n_negative'] = len(negative)
            row[f'{tag}_n_negative_by_1sigma'] = int(np.count_nonzero(real[1:]+sigma[1:]<0))
        rows.append(row)
    write_csv(out/'P2_first_negative.csv',rows)
    return rows


def plot_continuum_examples(out, old, beta, covariance, aligned, rows):
    p_index = LABELS.index(2.0)
    indices = [np.argmin(abs(old['z_fm'][p_index]-length)) for length in (.30,.45,.50,.60)]
    spacing_sq = old['a_fm']**2
    a_grid = np.linspace(0, 1.08*spacing_sq.max(), 100)
    fitted_rows = {(row['rf_fm'],row['Pz_GeV'],row['z_fm']):row for row in rows}
    methods = (('Adjacent constant','0.40','o',old['beta'],old['parameter_covariance'],
                old['aligned_center'],old['aligned_boot']),
               ('Direct two-state','tab:orange','s',beta,covariance,aligned[...,0,:],
                np.swapaxes(aligned[...,1:,:],-1,-2)))
    with PdfPages(out/'P2_a2_fit_examples.pdf') as pdf:
        for rf_index,radius in enumerate(old['rf_fm']):
            fig,axes = plt.subplots(2,len(indices),figsize=(13,6.0),sharex=True)
            for column,index in enumerate(indices):
                key = rf_index,p_index,index
                length = old['z_fm'][p_index,index]
                row = fitted_rows[radius,old['pz_GeV'][p_index],length]
                for component,axis in enumerate(axes[:,column]):
                    for label,color,marker,params,cov,centers,draws in methods:
                        data = centers[key].real if component==0 else centers[key].imag
                        samples = draws[key].real if component==0 else draws[key].imag
                        intercept,slope = params[key][[component,component+2]]
                        projection = np.stack((np.ones_like(a_grid),a_grid/spacing_sq[0]),axis=-1)
                        sub = cov[key][np.ix_([component,component+2],[component,component+2])]
                        sigma = np.sqrt(np.maximum(np.einsum('xi,ij,xj->x',projection,sub,projection),0))
                        line = intercept+slope*projection[:,1]
                        axis.plot(a_grid,line,color=color,lw=1,label=label)
                        axis.fill_between(a_grid,line-sigma,line+sigma,color=color,alpha=.13,lw=0)
                        axis.errorbar(spacing_sq,data,yerr=samples.std(axis=-1,ddof=1),
                                      fmt=marker,mfc='white',color=color,ms=3,lw=.8)
                    axis.axhline(0,color='0.7',lw=.6)
                    axis.set_yscale('symlog',linthresh=1.)
                    axis.grid(alpha=.2)
                    if component==0:
                        axis.set_title(f'z={length:.2f} fm; λ={old["coordinate"][index]:.2f}\nQ old/new={row["old_Q"]:.2g}/{row["Q"]:.2g}')
                    else:
                        axis.set_xlabel(r'$a^2$ [fm$^2$]')
                    if column==0:
                        axis.set_ylabel('Re H (symlog)' if component==0 else 'Im H (symlog)')
            axes[1,0].legend(frameon=False,fontsize=8)
            fig.suptitle(f'$P_z$ ≈ 2.0 GeV; $r_f$={radius:.5f} fm; all two-state points are diagnostic')
            fig.tight_layout(rect=(0,0,1,.95))
            pdf.savefig(fig)
            plt.close(fig)


def main():
    fit_receipt = json.loads((ROOT/'fit_receipt.json').read_text())
    old_receipt = json.loads((PREVIOUS/'run_receipt.json').read_text())
    base_receipt = json.loads((BASE/'run_receipt.json').read_text())
    assert digest(CODE)==base_receipt['inputs_sha256'][str(CODE)]
    spec = importlib.util.spec_from_file_location('core',CODE)
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    with load_verified(BASE,base_receipt,'02_common_lambda/grid_and_continuum.npz') as old:
        sources,norm_rows = load_sources(fit_receipt,old_receipt,old,core)
        rows, crossings, aligned_rows = continuum_fit(core,old,sources)
    with (ROOT/'01_bare_two_state/fit_quality.csv').open() as stream:
        bare_rows = list(csv.DictReader(stream))
    summary = {ens: {key:sum(row[key]=='True' for row in bare_rows if row['ensemble']==ens)
                     for key in ('denominator_ok','gap_closed','residual_ok','quality_accepted')} |
                     {'total':sum(row['ensemble']==ens for row in bare_rows)} for ens in ORDER}
    continuum = [{key:sum(row[key] for row in rows if round(row['Pz_GeV'],1)==label)
                  for key in ('fit_pass','all_inputs_pass','passes_core_gates')} for label in LABELS]
    demo = [row for row in rows if round(row['Pz_GeV'],1)==2.0 and abs(row['rf_fm']-.1794)<1e-5 and abs(row['z_fm']-.5)<1e-10][0]
    demo_inputs = [row for row in aligned_rows if all(row[key]==demo[key] for key in ('Pz_GeV','rf_fm','z_fm'))]
    assert len(demo_inputs)==3 and len(rows)==357
    doc = ['# Cut2：普通 C3/C2 两态 → fixed-λ', '',
           '输入为同组态及同 500 副本的 unpolarized connected u−d 普通 C3/C2；对照为 adjacent/ref75 常数。',
           '固定 cut=2、0.8<Ta<1.1 fm：C24 T/a=8,9,10；E32=9,10,11,12；F32=11,12。',
           r'对称展开 $R=c_0+c_1[e^{-\Delta E\tau}+e^{-\Delta E(T-\tau)}]+c_2e^{-\Delta E T}$。',
           '仅用 ratio 拟合；每个 z 独立、实虚共享能隙；0.2–2 GeV 为搜索边界，不是先验。',
           '同一副本先以 c0(z=0) 归一化，再按旧动量方案对齐、在原始 z 做 H3 减除、插值到旧 fixed-λ 网格并做实虚联合 a² 外推。',
           '沿用相同 GLS 模型，SVD 直接作用于副本偏差和加权设计矩阵以避免正规方程放大病态；旧数据 357 点的截距、协方差与 Q 均复现。',
           '裸拟合质量门另外要求原 C2 分母稳定、中心 gap 剖面闭合且不触边、≥90% 的副本 gap 可辨识。z=0 归一化额外检查分母稳定。',
           '', '| Ensemble | C2 分母通过 | 中心 gap 剖面闭合 | 残差通过 | 全部通过 | 总数 |', '|---|---:|---:|---:|---:|---:|']
    for ens,row in summary.items():
        doc.append('| '+ens+' | '+' | '.join(str(item) for item in row.values())+' |')
    doc.extend(['', '| Pz [GeV] | 连续拟合优度 | 三组输入同时通过 | 全部通过 |', '|---|---:|---:|---:|'])
    for label,row in zip(LABELS,continuum):
        doc.append(f'| {label:.1f} | '+' | '.join(f'{item}/119' for item in row.values())+' |')
    doc.extend(['', '## 2.0 GeV 首个负实部网格点', '',
                '| rf [fm] | 旧 z [fm] | 两态 z [fm] | 两态实部低于零超过 1σ 的网格点数 |', '|---:|---:|---:|---:|'])
    for row in crossings:
        old_z, new_z = (f'{row[f"{tag}_first_negative_z_fm"]:.2f}' if row[f'{tag}_first_negative_z_fm']!='' else '—'
                        for tag in ('adjacent','two_state'))
        doc.append(f'| {row["rf_fm"]:.5f} | {old_z} | {new_z} | {row["two_state_n_negative_by_1sigma"]} |')
    doc.extend(['', '这张表统计中心值首次变负，不是过零位置的显著性测量；网格点之间相关。',
                f'代表点 Pz≈2 GeV、rf≈0.1794 fm、z=0.50 fm：实部 {demo["real"]:+.4g} ± {demo["real_sd"]:.4g}，Q={demo["Q"]:.4g}；旧常数结果 {demo["old_real"]:+.4g} ± {demo["old_real_sd"]:.4g}，Q={demo["old_Q"]:.4g}。',
                '两态的三个输入实部：'+ '；'.join(f'{entry["ensemble"]} {entry["real"]:+.4g} ± {entry["real_sd"]:.4g}' for entry in demo_inputs)+'。',
                '诊断点保留全部副本，统计误差可能受局域归一化的近零分母影响；图使用 symlog。连续优度不能替代裸拟合质量。未做零流时外推或 matching。', '',
                '本轮同时改变了 ratio 构造与拟合模型，不能将两者差异单独归因于两态修正。', '',
                '裸矩阵元、能隙诊断和 ratio 拟合图在 `01_bare_two_state/`；逐点矩阵元、对照图和 2.0 GeV 的 a² 拟合图在 `02_fixed_lambda/`。', ''])
    (ROOT/'RESULTS.zh-CN.md').write_text('\n'.join(doc),encoding='utf-8')
    paths = [ROOT/'RESULTS.zh-CN.md', ROOT/'fit_receipt.json', ROOT/'fit.py', ROOT/'compare.py',
             *sorted(ROOT.glob('0[12]_*/*'))]
    receipt = dict(status='COMPLETE_TWO_STATE_FIXED_LAMBDA_DIAGNOSTIC', bare_summary=summary, continuum_summary=continuum,
                   representative_point=demo, representative_inputs=demo_inputs, P2_crossing_audit=crossings,
                   normalization_stable=sum(row['M0_division_stable'] for row in norm_rows),
                   normalization_total=len(norm_rows), outputs_sha256={str(path.relative_to(ROOT)):digest(path) for path in paths})
    (ROOT/'run_receipt.json').write_text(json.dumps(receipt,indent=2,default=lambda obj:obj.item(),allow_nan=False)+'\n')
    print(json.dumps({key:receipt[key] for key in ('bare_summary','continuum_summary','representative_point')},indent=2,default=lambda obj:obj.item()))


if __name__=='__main__':
    main()
