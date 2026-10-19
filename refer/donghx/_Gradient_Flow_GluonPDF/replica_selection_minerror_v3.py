"""Replica-local quality/AIC selection with explicitly labelled min-error fallback."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2 as chi2_dist


def select_cell(beta, boot, chi, chi_boot, dof, penalty, groups, numerical,
                qmin=1e-6, chi2max=10., allow_zero_dof=False):
    """Arrays: beta=(candidate,param), boot=(boot,candidate,param).

    Original data and every replica independently select candidates. Within
    each group (bare-fit cut), AIC-average quality-passing windows; then average
    the groups containing at least one passing window with equal weights.
    Fallback is global across groups, only if no candidate passes at all.
    Its error is std of the candidate's finite replicas, ddof=1, for param 0.
    """
    beta, boot = np.asarray(beta,float), np.asarray(boot,float)
    nc, nk = beta.shape
    nb = len(boot)
    if boot.shape != (nb,nc,nk):
        raise ValueError('invalid bootstrap axes')
    chi, chi_boot = np.asarray(chi,float), np.asarray(chi_boot,float)
    dof, penalty, groups = np.asarray(dof),np.asarray(penalty),np.asarray(groups)
    numerical = np.asarray(numerical,bool)
    values = np.concatenate([beta[None],boot],axis=0)
    chis = np.concatenate([chi[None],chi_boot],axis=0)
    finite = np.isfinite(values).all(axis=-1)&np.isfinite(chis)&numerical[None]
    safe_dof = np.maximum(dof,1)
    passing = finite&(dof[None]>0)&(chi2_dist.sf(chis,safe_dof[None])>=qmin)&(chis/safe_dof[None]<=chi2max)
    scores = chis+penalty[None]
    weights = np.zeros((nb+1,nc))
    n_groups = np.zeros(nb+1,int)
    for group in np.unique(groups):
        mask = passing&(groups==group)[None]
        has = mask.any(axis=1)
        n_groups += has
        rows = np.flatnonzero(has)
        if not len(rows):
            continue
        local = np.where(mask[rows],scores[rows],np.inf)
        w = np.exp(-.5*(local-local.min(axis=1,keepdims=True)))
        weights[rows] += w/w.sum(axis=1,keepdims=True)
    weights = np.divide(weights,n_groups[:,None],out=np.zeros_like(weights),where=n_groups[:,None]>0)
    quality = passing.any(axis=1)
    qvalues = np.einsum('rw,rwk->rk',weights,np.where(np.isfinite(values),values,0.))
    qvalues[~quality] = np.nan
    errors = np.full(nc,np.nan)
    finite_count = np.isfinite(boot[:,:,0]).sum(axis=0)
    for j in range(nc):
        b = boot[:,j,0]
        if finite_count[j]>=2:
            errors[j] = np.std(b[np.isfinite(b)],ddof=1)
    selected = qvalues.copy()
    fallback = np.zeros(nb+1,bool)
    indices = np.full(nb+1,-1,int)
    status = np.where(quality,'quality_aic','no_finite_candidate').astype('U64')
    for row in np.flatnonzero(~quality):
        pool = np.flatnonzero(finite[row]&(dof>0))
        zero = False
        if not len(pool) and allow_zero_dof:
            pool = np.flatnonzero(finite[row]&(dof==0))
            zero = True
        if not len(pool):
            continue
        with_error = pool[np.isfinite(errors[pool])]
        if len(with_error):
            j = with_error[np.lexsort((with_error,chis[row,with_error],errors[with_error]))[0]]
            label = 'minimum_candidate_statistical_error'
        else:
            j = pool[np.lexsort((pool,chis[row,pool]))[0]]
            label = 'minimum_chi2_no_finite_error'
        selected[row] = values[row,j]
        weights[row,j] = 1.
        fallback[row], indices[row] = True,j
        status[row] = ('zero_dof_' if zero else '')+label
    available = np.isfinite(selected).all(axis=1)
    spread = np.full(nk,np.nan)
    if quality[0]:
        spread = np.sqrt(np.einsum('w,wk->k',weights[0],
            np.where(np.isfinite(beta),beta-selected[0],0.)**2))
    error = np.std(selected[1:],axis=0,ddof=1) if available[1:].all() else np.full(nk,np.nan)
    return dict(params_quality=qvalues[0],params_boot_quality=qvalues[1:],
        params_selected=selected[0],params_boot_selected=selected[1:],
        quality_accepted=quality[0],bootstrap_quality_accepted=quality[1:],
        selected_available=available[0],bootstrap_selected_available=available[1:],
        central_fallback_used=fallback[0],replica_fallback_used=fallback[1:],
        central_fallback_candidate_index=indices[0],replica_fallback_candidate_index=indices[1:],
        central_status=status[0],replica_status=status[1:],
        replica_fallback_fraction=np.mean(fallback[1:]),stat_error=error,window_spread=spread,
        candidate_statistical_error=errors,candidate_finite_replica_count=finite_count,
        candidate_quality=passing[0],candidate_boot_quality=passing[1:],
        weights=weights[0],weights_boot=weights[1:],qualified_count=passing[0].sum())


def select_grid(beta,boot,chi,chi_boot,dof,penalty,groups,numerical,**kwargs):
    """Grid arrays: (candidate,p,z,param); bootstrap adds leading axis."""
    out = {}
    npz,nz = beta.shape[1:3]
    trailing_param = {'params_quality','params_boot_quality','params_selected','params_boot_selected','stat_error','window_spread'}
    for ip in range(npz):
        for iz in range(nz):
            r = select_cell(beta[:,ip,iz],boot[:,:,ip,iz],chi[:,ip,iz],chi_boot[:,:,ip,iz],
                            dof[:,ip,iz],penalty,groups,numerical[:,ip,iz],**kwargs)
            for key,value in r.items():
                value = np.asarray(value)
                if key not in out:
                    shape = (*value.shape[:-1],npz,nz,value.shape[-1]) if key in trailing_param else (*value.shape,npz,nz)
                    out[key] = np.empty(shape,dtype=value.dtype)
                if key in trailing_param:
                    out[key][...,ip,iz,:] = value
                else:
                    out[key][...,ip,iz] = value
    return out
