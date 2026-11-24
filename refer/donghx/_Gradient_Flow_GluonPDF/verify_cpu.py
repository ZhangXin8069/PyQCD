"""Independent gather, AIC/fallback and artifact verification; no fit rerouting."""
import csv
from datetime import datetime
import re
from common import *


def pdf_pages(path):
    value=Path(path).read_bytes()
    assert value.startswith(b'%PDF-') and value.rstrip().endswith(b'%%EOF') and b'Matplotlib' in value
    pages=re.findall(rb'\d+ 0 obj\s*<< /Type /Page\b',value)
    counts=re.findall(rb'/Count\s+(\d+)\b',value)
    assert len(counts)==1 and int(counts[0])==len(pages)>0
    return len(pages)


def independent_weights(values,chi,aic,accepted,errors):
    """Vectorized oracle: register each cut independently, then equal usable cuts."""
    output=np.zeros_like(values);usable=np.zeros(values.shape[1:],int)
    fallback_indices=np.full((len(CUTS),)+values.shape[1:],-1,int)
    for ci,cut in enumerate(CUTS):
        group=np.asarray([j for j,(_,_,c) in enumerate(CANDIDATES) if c==cut])
        v,ch,scores=values[group],chi[group],aic[group]
        finite=np.isfinite(v)&np.isfinite(ch)
        good=finite&accepted[group]
        exists=finite.any(axis=0);usable+=exists
        minimum=np.where(good,scores,np.inf).min(axis=0)
        with np.errstate(invalid='ignore'):
            raw=np.where(good,np.exp(-.5*(scores-minimum[None])),0.)
        norm=raw.sum(axis=0)
        w=np.divide(raw,norm[None],out=np.zeros_like(raw),where=norm[None]>0)
        need=exists&~good.any(axis=0)
        err=np.broadcast_to(errors,values.shape)[group]
        err=np.where(finite&np.isfinite(err),err,np.inf)
        safechi=np.where(finite,ch,np.inf)
        order=np.broadcast_to(group.reshape((-1,)+(1,)*(v.ndim-1)),v.shape)
        best=np.lexsort((order,safechi,err),axis=0)[0]
        for local,j in enumerate(group):
            use=need&(best==local);w[local][use]=1.
            fallback_indices[ci][use]=j
        output[group]=w
    output=np.divide(output,usable[None],out=np.zeros_like(output),where=usable[None]>0)
    return output,fallback_indices,(fallback_indices>=0).any(axis=0)


def gather_oracle(c3,c2,indices,ch,jf,rep):
    zsel=(0,1,5,10,20)
    cs=c3[:,ch,jf] if jf<2 else c3[:,ch,0]-c3[:,ch,1]
    cs=np.take(cs,zsel,axis=3)
    if rep<0:
        numerator,denominator=cs.mean(axis=0),c2.mean(axis=0)
    else:
        numerator,denominator=cs[indices[rep]].mean(axis=0),c2[indices[rep]].mean(axis=0)
    with np.errstate(invalid='ignore'):
        raw=numerator/denominator[None,:,None,:,:]
    ans=np.empty((2,len(PZ),len(zsel),6,13),complex)
    for co,sgn in enumerate((1.,-1.)):
        paired=(raw[0]+sgn*raw[1])/2
        for ip,p in enumerate(PZ):
            ans[co,ip]=paired[MOMENTA.index(p)] if p==0 else (
                paired[MOMENTA.index(p)]+sgn*paired[MOMENTA.index(-p)])/2
    return ans


def main():
    allocation();m=manifest();indices=shared_indices();artifacts=[];coverage=[];c2ref=None
    test=json.loads((HERE/'status/cpu_tests.json').read_text());assert test['status']=='PASS'
    physical=np.arange(13)[None,:]<=np.asarray(TSEPS)[:,None]
    for it,(tau,_) in enumerate(FLOWS):
        receipt=json.loads((HERE/'status'/f'ratio_{tau_tag(tau)}.json').read_text())
        assert receipt['status']=='PASS' and receipt['bootstrap_indices_sha256']==sha256(index_path())
        projected=Path(receipt['projected']);assert sha256(projected)==receipt['projected_sha256']
        with np.load(projected,allow_pickle=False) as d:
            c3,c2=d['c3_source_mean'],d['c2_source_mean']
            np.testing.assert_array_equal(d['confs'],CONFS);np.testing.assert_array_equal(d['boot_indices'],indices)
        if c2ref is None:c2ref=c2.copy()
        np.testing.assert_array_equal(c2[...,physical],c2ref[...,physical])
        artifacts.append(dict(path=str(projected),sha256=receipt['projected_sha256']))
        saved={}
        for ch,channel in enumerate(CHANNELS):
            saved[channel]={}
            for jf,flavor in enumerate(FLAVORS):
                task=it*9+ch*3+jf;rp=ratio_path(tau,channel,flavor);fp=fit_path(tau,channel,flavor)
                rmark=next(r for r in receipt['products'] if r['path']==str(rp))
                fmark=json.loads((HERE/'status'/f'fit_{task}.json').read_text())
                assert fmark['status']=='PASS' and sha256(rp)==rmark['sha256']==fmark['input_sha256']
                assert sha256(fp)==fmark['output_sha256']
                with np.load(rp,allow_pickle=False) as d:
                    ro,rb,valid=d['ratio_original'],d['ratio_boot'],d['ratio_valid_mask']
                    assert rb.shape==(200,2,len(PZ),21,6,13)
                    assert str(d['axes_boot'])=='boot,vdv,p,z,tsep,insertion'
                    assert str(d['c2_mode'])=='complex_nopol_dt_matched'
                    assert str(d['estimator_id'])==ESTIMATOR and str(d['weight_scheme'])==SCHEME
                    np.testing.assert_array_equal(d['boot_indices'],indices)
                    np.testing.assert_array_equal(d['confs'],CONFS)
                    np.testing.assert_array_equal(valid,np.broadcast_to(physical,ro.shape))
                    assert np.isfinite(ro[valid]).all() and np.isfinite(rb[:,valid]).all()
                    assert np.isnan(ro[~valid]).all() and np.isnan(rb[:,~valid]).all()
                    np.testing.assert_array_equal(d['component_boot'][:,0],rb[:,0].real)
                    np.testing.assert_array_equal(d['component_boot'][:,1],rb[:,1].imag)
                    for rep in (-1,0,17,199):
                        value=ro if rep<0 else rb[rep]
                        np.testing.assert_allclose(np.take(value,(0,1,5,10,20),axis=2),
                            gather_oracle(c3,c2,indices,ch,jf,rep),rtol=3e-10,atol=3e-10,equal_nan=True)
                    saved[channel][flavor]=rb.copy()
                with np.load(fp,allow_pickle=False) as d:
                    np.testing.assert_array_equal(d['boot_indices'],indices)
                    np.testing.assert_array_equal(d['confs'],CONFS)
                    assert str(d['axes_boot'])=='component,boot,p,z'
                    assert int(d['nboot'])==200 and int(d['fit_Tmin'])==TMIN and float(d['fit_Tmin_fm'])>.6
                    assert float(d['q_min'])==Q_MIN and float(d['chi2_dof_max'])==CHI_MAX
                    assert list(d['candidate_labels'])==[l for l,_,_ in CANDIDATES]
                    selected=d['M_boot_selected'].transpose(1,0,2,3)
                    available=d['selected_available'];zero=d['exact_symmetry_zero']
                    assert selected.shape==(200,2,len(PZ),21)
                    np.testing.assert_array_equal(available,np.isfinite(d['M_selected'])&np.isfinite(selected).all(axis=0))
                    np.testing.assert_allclose(np.std(selected,axis=0,ddof=1),d['statistical_error'],equal_nan=True)
                    cb=d['candidate_M_boot'];cc=d['candidate_chi2_boot'];aa=d['candidate_AIC_boot']
                    accepted=d['candidate_bootstrap_accepted'];errs=d['candidate_statistical_error']
                    for j,(_,ts,cut) in enumerate(CANDIDATES):
                        penalty=2+2*sum(t-2*cut+1 for t in range(max(ts)+1,max(TSEPS)+1))
                        np.testing.assert_allclose(aa[j],cc[j]+penalty,equal_nan=True)
                    np.testing.assert_allclose(np.std(cb,axis=1,ddof=1),errs,equal_nan=True)
                    expected_accept=np.isfinite(cb)&(d['candidate_Q_boot']>=Q_MIN)&(
                        cc/d['candidate_dof'][:,None]<=CHI_MAX)
                    np.testing.assert_array_equal(accepted,expected_accept)
                    ow,of,used=independent_weights(cb,cc,aa,accepted,errs[:,None])
                    np.testing.assert_allclose(d['candidate_weights_boot'],ow,rtol=5e-12,atol=5e-12)
                    np.testing.assert_array_equal(d['fallback_candidate_index'].transpose(2,1,0,3,4),of)
                    np.testing.assert_array_equal(d['fallback_used_boot'].transpose(1,0,2,3),used)
                    np.testing.assert_allclose(d['fallback_fraction'],used.mean(axis=0))
                    expect=(ow*np.where(np.isfinite(cb),cb,0.)).sum(axis=0)
                    np.testing.assert_allclose(selected[:,available],expect[:,available],rtol=5e-11,atol=5e-11)
                    cw,cf,cu=independent_weights(d['candidate_M'],d['candidate_chi2'],d['candidate_AIC'],d['candidate_accepted'],errs)
                    np.testing.assert_allclose(d['candidate_weights'],cw,rtol=5e-12,atol=5e-12)
                    np.testing.assert_array_equal(d['central_fallback_used'],cu)
                    np.testing.assert_array_equal(d['fallback_used'],cu|used.any(axis=0))
                    for co,ip in np.ndindex(2,len(PZ)):
                        av=available[co,ip];zz=zero[co,ip];fb=d['fallback_used'][co,ip];qq=d['quality_accepted'][co,ip]
                        coverage.append(dict(ensemble=KEY,tau=tau,channel=channel,flavor=flavor,
                            component=COMPONENTS[co],Pz=PZ[ip],points=21,selected_available=int(av.sum()),
                            quality_envelope_complete=int(qq.sum()),diagnostic_fallback_points=int(fb.sum()),
                            quality_no_fallback_points=int((av&qq&~fb&~zz).sum()),exact_symmetry_zero=int(zz.sum()),
                            no_finite_candidate=int((~av).sum())))
                artifacts.extend((dict(path=str(rp),sha256=rmark['sha256']),dict(path=str(fp),sha256=fmark['output_sha256'])))
            np.testing.assert_allclose(saved[channel]['u_minus_d'],saved[channel]['u']-saved[channel]['d'],
                rtol=3e-10,atol=3e-10,equal_nan=True)
            saved[channel].clear()
        print('INDEPENDENT_RATIO_FIT_AIC_FALLBACK_PASS',KEY,tau,flush=True)
    plots=json.loads((HERE/'status/plots.json').read_text())
    assert plots['status']=='PASS' and len(plots['products'])==9
    assert sha256(plots['csv_path'])==plots['csv_sha256']
    for p in plots['products']:
        assert sha256(p['path'])==p['sha256'] and pdf_pages(p['path'])==p['pages']==2
        artifacts.append(p)
    with (HERE/'coverage_by_channel_flavor_momentum.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=coverage[0].keys());writer.writeheader();writer.writerows(coverage)
    total=lambda key:sum(r[key] for r in coverage)
    points=len(FLOWS)*9*2*len(PZ)*21
    assert total('points')==plots['point_rows']==points
    nfinite=total('no_finite_candidate')
    atomic_json(HERE/'complete.json',dict(status='COMPLETE_BOOT200_RATIO_AIC_BARE_DIAGNOSTICS' if not nfinite else
        'COMPLETE_WITH_REGISTERED_NO_FINITE_CANDIDATES',timestamp=datetime.now().astimezone().isoformat(),
        job=os.environ['SLURM_JOB_ID'],ensemble=KEY,nconf=NCONF,nboot=200,seed=SEED,confs=list(CONFS),
        flows=[t for t,_ in FLOWS],pz=list(PZ),fit_Tmin=TMIN,fit_Tmin_fm=TMIN*A_FM,
        q_min=Q_MIN,chi2_dof_max=CHI_MAX,ratio_products=len(FLOWS)*9,fit_products=len(FLOWS)*9,pdf_count=9,
        point_count=points,selected_available=total('selected_available'),
        quality_envelope_complete=total('quality_envelope_complete'),
        diagnostic_fallback_points=total('diagnostic_fallback_points'),
        quality_no_fallback_points=total('quality_no_fallback_points'),exact_symmetry_zero_points=total('exact_symmetry_zero'),
        no_finite_candidate_points=nfinite,bootstrap_indices_sha256=sha256(index_path()),
        input_manifest_sha256=sha256(HERE/'input_manifest.json'),code_lock_sha256=sha256(ROOT/'code_lock.json'),
        independent_checks=['explicit gather from configuration means: central and replicas0,17,199; five z values',
          'all200 replica AIC normalization and omission penalties','all200 min-statistical-error fallback indices/weights',
          'paired u-d ratio identity, complex division, even-real/odd-imag, NaN masks and axes',
          'cross-flow unflowed C2 equality','PDF header/EOF/Page-object counts; visual review separate'],
        artifacts=artifacts,limitations=m['limitations']))
    text=[f'# {KEY}: completed B-scheme bare matrix elements', '',
        f'- {NCONF} configurations; 200 iid configuration-bootstrap samples; seed 1115.',
        f'- Pz: {PZ}; fixed Tmin {TMIN} ({TMIN*A_FM:.4f} fm), Tmax {TMAXS}, cuts {CUTS}.',
        '- Direct constant Ratio fit; Q >= 1e-3, chi2/dof <= 1e3; correlated shrinkage/SVD.',
        f'- Available {total("selected_available")}/{points}; no finite candidate {nfinite}.',
        f'- Quality envelope complete: {total("quality_envelope_complete")}.',
        f'- Any diagnostic fallback: {total("diagnostic_fallback_points")}.',
        f'- All-quality without fallback (excluding exact zeros): {total("quality_no_fallback_points")}.',
        f'- Odd-z0 algebraic zeros (not fitted): {total("exact_symmetry_zero")}.',
        '- Quality-envelope and fallback counts can overlap; do not add them.',
        '- 200 resamples are NOT 200 independent configurations; no autocorrelation assessment yet.', '',
        '## Data', '', '- `ratios/`: complex paired 200-bootstrap ratios and real/imag projections.',
        '- `fits/`: complete candidate/quality/selected/fallback/AIC/covariance metadata.',
        '- `bare_matrix_progress.csv`: all fitted points and separate errors.',
        '- `coverage_by_channel_flavor_momentum.csv`: quality/diagnostic coverage.',
        '- `plots/`: 9 PDFs; all flows offset in z; no connecting lines. Page 1 full range, page 2 fixed viewport.',
        '- `complete.json`: independent verification and artifact SHA256.', '',
        'These are finite-flow connected bare diagnostics, not matched PDFs. No Zchi/local normalization,',
        'line subtraction, flow matching or continuum/flow-time/LaMET extrapolation was done.',
        'Transversity relative sign remains pending_external_physics_confirmation.']
    (HERE/'RESULTS.md').write_text('\n'.join(text)+'\n')
    print('INDEPENDENT_VERIFICATION_COMPLETE',KEY,total('selected_available'),points,flush=True)


if __name__=='__main__':
    main()
