#!/usr/bin/env python3
"""Correlated direct-Ratio plateau fits, per-replica AIC, separate quality/fallback."""
import argparse
import json
import os
import time
import numpy as np
from scipy.stats import chi2
from common import *
from barefit.config import StatisticsConfig
from barefit.covariance import correlated_constant_fit

STATS=StatisticsConfig(covariance_shrinkage='schaefer_strimmer_correlation_to_identity',
    svdcut=1e-12,covariance_relative_jitter=1e-12,covariance_absolute_jitter=1e-18,
    normal_rcond=1e-12,q_min=Q_MIN,correlated_chi2_dof_max=CHI_MAX,
    bootstrap_min_success=1.,minimum_accepted_windows=1,
    window_systematic='within_cut_AIC_weighted_variance_plus_separate_equal_cut_spread',
    fit_window_status='fixed_Tmin7_user_Q1em3_chi2dof1e3')


def weights(scores):
    s=np.asarray(scores,float);out=np.zeros_like(s);ok=np.isfinite(s)
    if ok.any():
        v=np.exp(-.5*(s[ok]-s[ok].min()));out[ok]=v/v.sum()
    return out


def candidate_fit(central,samples):
    """Require actual sample information before regularized one-parameter GLS."""
    y=np.asarray(central,float);b=np.asarray(samples,float)
    if y.ndim!=1 or b.shape!=(100,len(y)) or len(y)<=1:
        raise ValueError('bad shape or no positive data dof')
    if not np.isfinite(y).all() or not np.isfinite(b).all():raise ValueError('nonfinite candidate')
    variance=np.var(b,axis=0,ddof=1)
    if np.any(variance<=0):raise ValueError('zero bootstrap variance')
    standardized=(b-b.mean(axis=0))/np.sqrt(variance)
    raw_correlation=standardized.T@standardized/99.
    eigen=np.linalg.eigvalsh(raw_correlation)
    rank=int(np.count_nonzero(eigen>1e-12*eigen.max()))
    if rank<1:raise ValueError('sample information cannot identify one parameter')
    fit=correlated_constant_fit(y,b,STATS)
    if fit.dof<=0 or fit.covariance.rank<=1:raise ValueError('nonpositive fit dof')
    values=fit.parameter_boot[:,0]
    residual=b-values[:,None]
    chib=np.einsum('bi,ij,bj->b',residual,fit.covariance.inverse,residual,optimize=True)
    qb=chi2.sf(chib,fit.dof)
    if not np.isfinite(values).all():raise ValueError('incomplete candidate bootstrap')
    return dict(M=float(fit.parameters[0]),boot=values,chi=float(fit.chi2),dof=int(fit.dof),
        Q=float(fit.q_value),chi_boot=chib,Q_boot=qb,stat_error=float(np.std(values,ddof=1)),
        accepted=bool(fit.q_value>=Q_MIN and fit.chi2/fit.dof<=CHI_MAX),
        boot_accepted=np.isfinite(qb)&(qb>=Q_MIN)&(chib/fit.dof<=CHI_MAX),
        sample_rank=rank,effective_rank=fit.covariance.rank,model_rank=1,
        covariance_condition=fit.covariance.condition,shrinkage=fit.covariance.shrinkage_lambda,
        raw_correlation_eigenvalues=eigen,covariance_eigenvalues=fit.covariance.eigenvalues)


def observations(ro,rb,valid,component,ip,iz,separations,cut):
    central=[];boot=[]
    for sep in separations:
        it=TSEPS.index(sep);sl=slice(cut,sep-cut+1)
        if not valid[component,ip,iz,it,sl].all():raise ValueError('invalid point in fit candidate')
        r=ro[component,ip,iz,it,sl]
        b=rb[:,component,ip,iz,it,sl]
        central.append(r.real if component==0 else r.imag)
        boot.append(b.real if component==0 else b.imag)
    return np.concatenate(central),np.concatenate(boot,axis=1)


def fallback(finite,errors,chi):
    finite=np.asarray(finite,int)
    if not len(finite):return -1
    good=finite[np.isfinite(errors[finite])]
    if len(good):return min(good,key=lambda j:(errors[j],chi[j],j))
    return min(finite,key=lambda j:(chi[j],j))


def select(values,chi,aic,accepted,errors,cut_indices):
    """AIC within each cut; equal-weight usable cuts; no AIC comparison across cuts."""
    mixed=[];quality_mixed=[];all_weights=np.zeros(len(values));used=[];spread=[]
    for group in cut_indices:
        finite=group[np.isfinite(values[group])&np.isfinite(chi[group])]
        good=finite[accepted[finite]]
        if len(good):
            w=weights(aic[good]);value=float(w@values[good]);quality_mixed.append(value)
            wc=np.zeros(len(values));wc[good]=w
            spread.append(float(w@((values[good]-value)**2)))
        else:
            j=fallback(finite,errors,chi)
            if j<0:continue
            value=float(values[j]);wc=np.zeros(len(values));wc[j]=1.;used.append(j)
            spread.append(0.)
        mixed.append(value);all_weights+=wc
    if not mixed:
        return dict(value=np.nan,quality=np.nan,weights=all_weights,fallback=True,
                    fallback_indices=[],within=np.nan,cut_spread=np.nan)
    all_weights/=len(mixed)
    return dict(value=float(np.mean(mixed)),quality=float(np.mean(quality_mixed)) if quality_mixed else np.nan,
        weights=all_weights,fallback=bool(used),fallback_indices=used,
        within=float(np.sqrt(np.mean(spread))),cut_spread=float(np.std(mixed,ddof=0)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);arg=p.parse_args()
    allocation();manifest();started=time.monotonic()
    it,rem=divmod(arg.task,9);ic,jf=divmod(rem,3)
    tau=FLOWS[it][0];channel=CHANNELS[ic];flavor=FLAVORS[jf]
    source=ratio_path(tau,channel,flavor);output=fit_path(tau,channel,flavor)
    receipt_path=HERE/'status'/f'fit_{arg.task}.json'
    if receipt_path.exists():
        r=json.loads(receipt_path.read_text())
        if r['status']=='PASS' and r['output_sha256']==sha256(output) and r['input_sha256']==sha256(source):
            print('VALIDATED_FIT_SKIP',arg.task,flush=True);return
        raise ValueError('incompatible fit completion')
    source_hash=sha256(source)
    with np.load(source,allow_pickle=False) as d:
        ro=d['ratio_original'];rb=d['ratio_boot'];valid=d['ratio_valid_mask']
        if ro.shape!=(2,3,21,6,11) or rb.shape!=(100,2,3,21,6,11):raise ValueError('ratio axes changed')
        if str(d['estimator_id'])!=ESTIMATOR or str(d['weight_scheme'])!=SCHEME:raise ValueError('wrong estimator/scheme')
        np.testing.assert_array_equal(d['confs'],CONFS);np.testing.assert_array_equal(d['boot_indices'],shared_indices())
    nc=len(CANDIDATES);point=(2,3,21);shape=(nc,)+point;bs=(nc,100)+point
    fields={name:np.full(shape,np.nan) for name in ('M','chi','Q','stat_error','AIC','condition','shrinkage')}
    integer={name:np.full(shape,-1,dtype=np.int16) for name in ('dof','sample_rank','effective_rank','model_rank')}
    candboot=np.full(bs,np.nan);chiboot=np.full(bs,np.nan);qboot=np.full(bs,np.nan);aicboot=np.full(bs,np.nan)
    accept=np.zeros(shape,bool);acceptboot=np.zeros(bs,bool);reasons=np.full(shape,'not_attempted',dtype='U128')
    maxdata=max(sum(t-2*c+1 for t in ts) for _,ts,c in CANDIDATES)
    raw_eigen=np.full(shape+(maxdata,),np.nan);reg_eigen=np.full_like(raw_eigen,np.nan)
    exactzero=np.zeros(point,bool)
    for co in range(2):
        for ip in range(3):
            for iz in range(21):
                if co==1 and iz==0:
                    mask=valid[co,ip,iz]
                    if (np.all(ro[co,ip,iz][mask].imag==0)
                            and np.all(rb[:,co,ip,iz][:,mask].imag==0)):
                        exactzero[co,ip,iz]=True
                for j,(_,ts,cut) in enumerate(CANDIDATES):
                    if exactzero[co,ip,iz]:
                        reasons[j,co,ip,iz]='exact_VDV_odd_z0_symmetry_not_a_fit';continue
                    try:
                        y,b=observations(ro,rb,valid,co,ip,iz,ts,cut)
                        r=candidate_fit(y,b)
                    except (ValueError,np.linalg.LinAlgError,FloatingPointError) as exc:
                        reasons[j,co,ip,iz]=str(exc)[:128];continue
                    idx=(j,co,ip,iz);bi=(j,slice(None),co,ip,iz)
                    omitted=sum(t-2*cut+1 for t in range(max(ts)+1,11))
                    for key in ('M','chi','Q','stat_error'):fields[key][idx]=r[key]
                    fields['AIC'][idx]=r['chi']+2+2*omitted
                    fields['condition'][idx]=r['covariance_condition'];fields['shrinkage'][idx]=r['shrinkage']
                    for key in integer:integer[key][idx]=r[key]
                    candboot[bi]=r['boot'];chiboot[bi]=r['chi_boot'];qboot[bi]=r['Q_boot']
                    aicboot[bi]=r['chi_boot']+2+2*omitted;accept[idx]=r['accepted'];acceptboot[bi]=r['boot_accepted']
                    raw_eigen[idx][:len(y)]=r['raw_correlation_eigenvalues']
                    reg_eigen[idx][:len(y)]=r['covariance_eigenvalues']
                    reasons[idx]='identifiable_quality_pass' if r['accepted'] else 'identifiable_quality_rejected'
    selected=np.full(point,np.nan);quality=np.full(point,np.nan)
    sb=np.full((100,)+point,np.nan);qb=np.full_like(sb,np.nan)
    used=np.zeros((100,)+point,bool);central_fallback=np.zeros(point,bool)
    fidx=np.full((100,len(CUTS))+point,-1,dtype=np.int16)
    cweights=np.zeros(shape);bweights=np.zeros(bs)
    within=np.full(point,np.nan);cutsys=np.full(point,np.nan)
    status=np.full(point,'no_finite_candidate',dtype='U64')
    cut_indices=[np.array([j for j,(_,_,c) in enumerate(CANDIDATES) if c==cut]) for cut in CUTS]
    for co,ip,iz in np.ndindex(point):
        if exactzero[co,ip,iz]:
            selected[co,ip,iz]=0.;sb[:,co,ip,iz]=0.
            within[co,ip,iz]=cutsys[co,ip,iz]=0.;status[co,ip,iz]='exact_symmetry_zero_not_fitted';continue
        idx=(slice(None),co,ip,iz)
        r=select(fields['M'][idx],fields['chi'][idx],fields['AIC'][idx],accept[idx],fields['stat_error'][idx],cut_indices)
        selected[co,ip,iz]=r['value'];quality[co,ip,iz]=r['quality'];cweights[idx]=r['weights']
        central_fallback[co,ip,iz]=r['fallback'];within[co,ip,iz]=r['within'];cutsys[co,ip,iz]=r['cut_spread']
        for boot in range(100):
            bi=(slice(None),boot,co,ip,iz)
            br=select(candboot[bi],chiboot[bi],aicboot[bi],acceptboot[bi],fields['stat_error'][idx],cut_indices)
            sb[boot,co,ip,iz]=br['value'];qb[boot,co,ip,iz]=br['quality'];used[boot,co,ip,iz]=br['fallback']
            bweights[bi]=br['weights']
            for q,j in enumerate(br['fallback_indices']):fidx[boot,q,co,ip,iz]=j
        if np.isfinite(selected[co,ip,iz]) and np.isfinite(sb[:,co,ip,iz]).all():
            status[co,ip,iz]='selected_with_diagnostic_fallback' if central_fallback[co,ip,iz] or used[:,co,ip,iz].any() else 'all_quality_selected'
    available=np.isfinite(selected)&np.isfinite(sb).all(axis=0)
    qaccepted=np.isfinite(quality)&np.isfinite(qb).all(axis=0)&~exactzero
    statistical=np.std(sb,axis=0,ddof=1)
    atomic_npz(output,**common_metadata(tau),channel=np.asarray(channel),flavor_combo=np.asarray(flavor),
        component_labels=np.asarray(COMPONENTS),axes_matrix=np.asarray('component,p,z'),
        axes_boot=np.asarray('component,boot,p,z'),M_selected=selected,M_boot_selected=sb.transpose(1,0,2,3),
        M_quality=quality,M_boot_quality=qb.transpose(1,0,2,3),selected_available=available,
        quality_accepted=qaccepted,statistical_error=statistical,
        window_systematic=within,cut_systematic=cutsys,total_error=np.sqrt(statistical**2+within**2+cutsys**2),
        fallback_used=central_fallback|used.any(axis=0),fallback_used_boot=used.transpose(1,0,2,3),
        central_fallback_used=central_fallback,fallback_fraction=used.mean(axis=0),
        fallback_candidate_index=fidx.transpose(2,0,1,3,4),fallback_status=status,exact_symmetry_zero=exactzero,
        candidate_M=fields['M'],candidate_M_boot=candboot,candidate_chi2=fields['chi'],
        candidate_chi2_dof=fields['chi']/integer['dof'],candidate_Q=fields['Q'],candidate_dof=integer['dof'],
        candidate_Q_boot=qboot,candidate_chi2_boot=chiboot,candidate_accepted=accept,
        candidate_bootstrap_accepted=acceptboot,candidate_statistical_error=fields['stat_error'],
        candidate_AIC=fields['AIC'],candidate_AIC_boot=aicboot,
        candidate_weights=cweights,candidate_weights_boot=bweights,
        candidate_sample_rank=integer['sample_rank'],candidate_effective_rank=integer['effective_rank'],
        candidate_model_rank=integer['model_rank'],candidate_covariance_condition=fields['condition'],
        candidate_shrinkage=fields['shrinkage'],candidate_raw_correlation_eigenvalues=raw_eigen,
        candidate_regularized_covariance_eigenvalues=reg_eigen,candidate_status=reasons,
        candidate_labels=np.asarray([l for l,_,_ in CANDIDATES]),
        fit_Tmin=np.asarray(7),fit_Tmin_fm=np.asarray(7*A_FM),q_min=np.asarray(Q_MIN),chi2_dof_max=np.asarray(CHI_MAX),
        chis_definition=np.asarray('correlated chi2 / dof; NOT raw chi2'),
        input_ratio_sha256=np.asarray(source_hash),fit_model=np.asarray('R(T,t_insert)=M'),
        aic_policy=np.asarray('chi2+2*k+2*N_omit; within each cut, then equal usable cuts; independent per replica'),
        bootstrap_error_definition=np.asarray('std(M_boot,ddof=1); no division by sqrt(Nboot)'),
        covariance_svdcut=np.asarray(1e-12),covariance_absolute_jitter=np.asarray(1e-18),
        pipeline_version=np.asarray('L32x64_A_ratio_AIC_T7_boot100_v1'))
    with np.load(output,allow_pickle=False) as d:
        assert d['M_boot_selected'].shape==(2,100,3,21)
        np.testing.assert_array_equal(d['boot_indices'],shared_indices())
    atomic_json(receipt_path,dict(status='PASS',task=arg.task,tau=tau,channel=channel,flavor=flavor,
        input_sha256=source_hash,output=str(output),output_sha256=sha256(output),
        total_points=126,selected_available=int(available.sum()),quality_complete_points=int(qaccepted.sum()),
        fallback_points=int((central_fallback|used.any(axis=0)).sum()),exact_symmetry_zero_points=int(exactzero.sum()),
        no_finite_candidate_points=int((~available).sum()),seconds=time.monotonic()-started,
        job=os.environ['SLURM_JOB_ID'],physical_status='finite_flow_bare_diagnostic_no_state_isolation_claim'))
    print('FIT_COMPLETE',arg.task,tau,channel,flavor,'available',available.sum(),'quality',qaccepted.sum(),flush=True)


if __name__=='__main__':main()
