"""Independent output oracles; never call production ratio or selection functions."""
import numpy as np
from common import *


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=3e-10,atol=2e-11,equal_nan=True)


def independent_weights(value,chi,aic,accepted,statistical):
    answer=np.zeros_like(value);usable=np.zeros(value.shape[1:],dtype=np.int16)
    fallback=np.zeros(value.shape[1:],bool)
    for cut in range(len(CUTS)):
        sl=slice(5*cut,5*cut+5);v,x,score=value[sl],chi[sl],aic[sl]
        finite=np.isfinite(v)&np.isfinite(x);good=finite&accepted[sl]
        any_good=good.any(axis=0);any_finite=finite.any(axis=0)
        minimum=np.min(np.where(good,score,np.inf),axis=0)
        minimum=np.where(any_good,minimum,0.)
        exponent=np.zeros_like(score)
        np.subtract(score,minimum[None],out=exponent,where=good)
        out=np.zeros_like(score);np.exp(-.5*exponent,out=out,where=good)
        normalizer=out.sum(axis=0)
        np.divide(out,normalizer[None],out=out,where=normalizer[None]>0)
        errors=np.broadcast_to(statistical[sl,None],v.shape)
        any_error=(finite&np.isfinite(errors)).any(axis=0)
        error_key=np.where(any_error[None],np.where(np.isfinite(errors),errors,np.inf),0.)
        best_error=np.full(v.shape[1:],np.inf);best_chi=np.full(v.shape[1:],np.inf)
        best_index=np.full(v.shape[1:],-1,dtype=np.int16)
        for j in range(5):
            better=finite[j]&((best_index<0)|(error_key[j]<best_error)
                |((error_key[j]==best_error)&(x[j]<best_chi)))
            best_error=np.where(better,error_key[j],best_error)
            best_chi=np.where(better,x[j],best_chi);best_index=np.where(better,j,best_index)
        use_fallback=~any_good&any_finite
        for j in range(5): out[j]=np.where(use_fallback&(best_index==j),1.,out[j])
        answer[sl]=out;usable+=any_finite;fallback|=use_fallback
    np.divide(answer,usable[None],out=answer,where=usable[None]>0)
    return answer,fallback


def audit_selection(d):
    close(d['candidate_statistical_error'],np.std(d['candidate_M_boot'],axis=1,ddof=1))
    close(d['statistical_error'],np.std(d['M_boot_selected'],axis=1,ddof=1))
    for boot in (False,True):
        if boot:
            value,chi,aic=(d[k] for k in ('candidate_M_boot','candidate_chi2_boot','candidate_AIC_boot'))
            accepted=d['candidate_bootstrap_accepted'];q=d['candidate_Q_boot']
            stored_w=d['candidate_weights_boot'];stored=d['M_boot_selected'].transpose(1,0,2,3)
        else:
            value,chi,aic=(d[k][:,None] for k in ('candidate_M','candidate_chi2','candidate_AIC'))
            accepted=d['candidate_accepted'][:,None];q=d['candidate_Q'][:,None]
            stored_w=d['candidate_weights'][:,None];stored=d['M_selected'][None]
        dof=d['candidate_dof'][:,None]
        expected=np.isfinite(value)&np.isfinite(chi)&(q>=Q_MIN)&(chi<=CHI_MAX*dof)&(dof>0)
        np.testing.assert_array_equal(accepted,expected)
        w,fb=independent_weights(value,chi,aic,accepted,d['candidate_statistical_error'])
        close(w,stored_w)
        result=(np.where(w>0,value,0.)*w).sum(axis=0)
        result[:,~d['selected_available']]=np.nan
        close(result,stored)
        actual=d['fallback_used_boot'].transpose(1,0,2,3) if boot else d['central_fallback_used'][None]
        # Exact symmetry zeros have no fit and are not statistical fallbacks.
        fb[:,d['exact_symmetry_zero']]=False
        np.testing.assert_array_equal(fb,actual)


def audit_ratios():
    allocation();indices=shared_indices();checks=0
    np.testing.assert_array_equal(indices,np.load(PAIRED/INDEX_NAME,allow_pickle=False))
    for tau,_ in FLOWS:
        with np.load(HERE/'projected'/f'projected_{tau_tag(tau)}.npz',allow_pickle=False) as d:
            c3=d['c3_source_mean'];c2=d['c2_source_mean']
            np.testing.assert_array_equal(d['confs'],CONFS)
        # Explicit gathers, no bincount/tensordot estimator from production.
        samples=[np.arange(NCONF)]+list(indices)
        time_mask=np.arange(13)[None,:]<=np.asarray(TSEPS)[:,None]
        for ic,channel in enumerate(CHANNELS):
            for jf,flavor in enumerate(FLAVORS):
                numerator=c3[:,ic,jf] if jf<2 else c3[:,ic,0]-c3[:,ic,1]
                with np.load(ratio_path(tau,channel,flavor),allow_pickle=False) as d:
                    np.testing.assert_array_equal(d['boot_indices'],indices)
                    np.testing.assert_array_equal(d['pz_values'],[0])
                    mask=d['ratio_valid_mask'];original=d['ratio_original'];boot=d['ratio_boot']
                    assert not np.isfinite(original[~mask]).any() and not np.isfinite(boot[:,~mask]).any()
                    assert str(d['estimator_id'])==ESTIMATOR and str(d['weight_scheme'])==SCHEME
                    for ib,idx in enumerate(samples):
                        den=c2[idx].mean(axis=0);num=numerator[idx].mean(axis=0)
                        raw=np.full_like(num,np.nan+1j*np.nan)
                        np.divide(num,den[None,:,None,:,:],out=raw,where=time_mask[None,None,None,:,:])
                        expected=np.stack(((raw[0]+raw[1])/2,(raw[0]-raw[1])/2))
                        close(expected,original if ib==0 else boot[ib-1])
                        close(den,d['c2_original'] if ib==0 else d['c2_boot'][ib-1])
                        checks+=1
        print('PZ0_INDEPENDENT_RATIO_AUDIT_PASS',tau,flush=True)
    return checks
