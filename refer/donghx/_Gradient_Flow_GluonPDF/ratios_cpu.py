"""Reuse SHA-audited configuration projections, add P0, bootstrap from configs."""
import argparse
import csv
import time
from common import *
from conventions import make_channel_specs, PHASE_FACTORS


SPECS = make_channel_specs([f'P{"p" if p>=0 else "m"}{abs(p)}_z' for p in MOMENTA],
                          ['z']*len(MOMENTA), list(POLS))


def project_raw(raw, momenta):
    specs = make_channel_specs([f'P{"p" if p>=0 else "m"}{abs(p)}_z' for p in momenta],
                              ['z']*len(momenta), list(POLS))
    assert raw.shape == (2,2,4,len(momenta),DZ,4,4)
    out = np.empty((3,2,2,len(momenta),DZ), complex)
    for ip in range(len(momenta)):
        v = [np.einsum('fszmo,mo->fsz', raw[:,:,int(specs.pol_indices[ip,ch]),ip],
                       specs.gamma[ip,ch], optimize=True) for ch in range(4)]
        out[0,:,:,ip] = v[0]
        out[1,:,:,ip] = PHASE_FACTORS['helicity']*v[1]
        out[2,:,:,ip] = .5*PHASE_FACTORS['transversity']*(v[2]+v[3])
    return out


def audit_live_ledger(path, tau, c3, c2, offset):
    """Stat every saved raw payload; independently reproject one insertion/conf."""
    oracle = {}; count = 0
    with Path(path).open() as stream:
        for row in csv.DictReader(stream):
            rawpath = Path(row['path']); st = rawpath.stat()
            assert rawpath.is_file() and st.st_size > 0
            if row.get('size'):
                assert st.st_size == int(row['size']) and st.st_mtime_ns == int(row['mtime_ns'])
            ic = CONFS.index(row['conf']); sep = 8 if ic%2 == 0 else 9; dt = 3 if ic%2 == 0 else 1
            if int(row['tsep']) == sep and int(row['dt']) == dt:
                expected = row.get('npz_sha256', row.get('sha256'))
                assert sha256(rawpath) == expected
                with np.load(rawpath, allow_pickle=False) as d:
                    raw, raw2 = d['c3_flavor'], d['c2_common']
                    nm = raw.shape[3]
                    moms = (3,-3,4,-4,5,-5,6,-6) if nm == 8 else MOMENTA[offset:]
                    assert len(moms) == nm
                    pr = project_raw(raw, moms)
                    if nm == 8:
                        pr = pr[:,:,:,2:]; raw2 = raw2[:,2:]
                    assert str(d['weight_scheme']) == SCHEME and int(d['weight_applied_count']) == 1
                if ic not in oracle:
                    oracle[ic] = [np.zeros_like(pr), np.zeros_like(raw2[0]), set()]
                item = oracle[ic]; cur = int(row['current'])
                assert cur not in item[2]
                item[0] += pr/4; item[1] += raw2[0]/4; item[2].add(cur)
            count += 1
    for ic,(num,den,currents) in oracle.items():
        assert currents == set(CURRENTS)
        sep,dt = (8,3) if ic%2 == 0 else (9,1)
        it = TSEPS.index(sep)
        np.testing.assert_allclose(c3[ic,:,:, :,offset:,:,it,dt], num, rtol=2e-12, atol=1e-27)
        np.testing.assert_allclose(c2[ic,offset:,it,dt], den, rtol=2e-12, atol=1e-27)
    print('LIVE_LEDGER_AND_RAW_PROJECTION_ORACLES',KEY,tau,count,len(oracle),flush=True)
    return dict(path=str(path), sha256=sha256(path), live_payloads=count, oracle_confs=len(oracle),
                oracle_raw_payloads=4*len(oracle))


def compute_ratios(c3,c2,indices):
    """Config means -> complex ratios -> z even/odd -> aligned +/-P average."""
    dz = c3.shape[5]
    assert c3.shape == (NCONF,3,2,2,len(MOMENTA),dz,6,13)
    assert c2.shape == (NCONF,len(MOMENTA),6,13)
    weights = np.stack([np.bincount(row,minlength=NCONF)/NCONF for row in indices])
    full = np.concatenate((c3,(c3[:,:,0]-c3[:,:,1])[:,:,None]),axis=2)
    mean, den = full.mean(axis=0), c2.mean(axis=0)
    boots = np.tensordot(weights,full,axes=(1,0))
    denboot = np.tensordot(weights,c2,axes=(1,0))
    physical = np.arange(13)[None,:] <= np.asarray(TSEPS)[:,None]
    assert np.isfinite(c3[...,physical]).all() and np.isnan(c3[...,~physical]).all()
    assert np.isfinite(c2[...,physical]).all() and np.isnan(c2[...,~physical]).all()
    if np.any(den[...,physical] == 0) or np.any(denboot[...,physical] == 0):
        raise ValueError('Exactly zero complex C2; cannot clip, take abs, or discard replicas')
    with np.errstate(invalid='ignore', divide='raise'):
        raw = mean / den[None,None,None,:,None,:,:]
        rawboot = boots / denboot[:,None,None,None,:,None,:,:]
    directional = np.stack((.5*(raw[:,:,0]+raw[:,:,1]),.5*(raw[:,:,0]-raw[:,:,1])),axis=2)
    directionalboot = np.stack((.5*(rawboot[:,:,:,0]+rawboot[:,:,:,1]),
                                 .5*(rawboot[:,:,:,0]-rawboot[:,:,:,1])),axis=3)
    parity = np.asarray([1.,-1.]); values,samples = [],[]
    for p in PZ:
        ip = MOMENTA.index(p)
        if p == 0:
            values.append(directional[:,:,:,ip]);samples.append(directionalboot[:,:,:,:,ip])
        else:
            im = MOMENTA.index(-p)
            values.append(.5*(directional[:,:,:,ip]+directional[:,:,:,im]*parity[None,None,:,None,None,None]))
            samples.append(.5*(directionalboot[:,:,:,:,ip]+directionalboot[:,:,:,:,im]*parity[None,None,None,:,None,None,None]))
    return np.stack(values,axis=3),np.stack(samples,axis=4),den,denboot


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--flow-index',type=int,required=True)
    args=parser.parse_args();allocation();m=manifest();ind=shared_indices();start=time.monotonic()
    tau=FLOWS[args.flow_index][0];marker=HERE/'status'/f'ratio_{tau_tag(tau)}.json'
    if marker.exists():
        record=json.loads(marker.read_text());assert record['status']=='PASS'
        assert all(sha256(p['path'])==p['sha256'] for p in record['products'])
        assert sha256(record['projected'])==record['projected_sha256'];return
    source=m['parent_projected'][args.flow_index];assert sha256(source['path'])==source['sha256']
    assert sha256(source['ledger'])==source['ledger_sha256']
    with np.load(source['path'],allow_pickle=False) as d:
        previous_c3=d['c3_source_mean'];previous_c2=d['c2_source_mean']
    live=[];p0_cases=0
    if KEY=='L32x64':
        c3=np.full((NCONF,3,2,2,len(MOMENTA),DZ,6,13),np.nan+1j*np.nan)
        c2=np.full((NCONF,len(MOMENTA),6,13),np.nan+1j*np.nan)
        c3[:,:,:, :,1:]=previous_c3;c2[:,1:]=previous_c2
        old=json.loads((OLD64/'status'/f'ratio_{parent_tag(tau)}.json').read_text())
        assert sha256(old['raw_case_ledger'])==old['raw_case_ledger_sha256']
        live.append(audit_live_ledger(old['raw_case_ledger'],tau,c3,c2,1))
        live.append(audit_live_ledger(source['ledger'],tau,c3,c2,1))
        p0=load_isolated('frozen_p0_B50',P064/'common.py');pm=p0.load_manifest()
        rows={r['conf']:r for r in pm['tasks']+pm['reused_tasks'] if r['flow_time_a2']==tau}
        ledger_path=HERE/'projected'/f'pz0_raw_ledger_{tau_tag(tau)}.csv'
        with ledger_path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=('conf','tau','current','tsep','dt','path','sha256'));writer.writeheader()
            for ic,conf in enumerate(CONFS):
                row=rows[conf]
                for it,sep in enumerate(TSEPS):
                    for dt in range(sep+1):
                        num=np.zeros((3,2,2,1,DZ),complex);den=np.zeros(1,complex)
                        for cur in CURRENTS:
                            _,_,raw2,pr,receipt=p0.read_case(row,cur,sep,dt,verify_hash=True)
                            num+=pr/4;den+=raw2[0]/4;p0_cases+=1
                            writer.writerow(dict(conf=conf,tau=tau,current=cur,tsep=sep,dt=dt,
                                path=str(p0.output_dir(row,cur,sep,dt)/'correlators.npz'),sha256=receipt['npz_sha256']))
                        c3[ic,:,:, :,0,:,it,dt]=num[:,:,:,0];c2[ic,0,it,dt]=den[0]
                print('P0_CONF_REVERIFIED',tau,conf,ic+1,'/',NCONF,flush=True)
        assert p0_cases==12600
    else:
        c3,c2=previous_c3,previous_c2
        live.append(audit_live_ledger(source['ledger'],tau,c3,c2,0))
    projected=HERE/'projected'/f'projected_{tau_tag(tau)}_Nconf{NCONF}.npz'
    atomic_npz(projected,c3_source_mean=c3,c2_source_mean=c2,
        axes_c3=np.asarray('conf,channel,flavor_u_d,sign,signed_momentum,z,tsep,insertion'),
        axes_c2=np.asarray('conf,signed_momentum,tsep,insertion'),**common_metadata(tau))
    ro,rb,den,db=compute_ratios(c3,c2,ind)
    assert ro.shape==(3,3,2,len(PZ),DZ,6,13) and rb.shape==(NBOOT,)+ro.shape
    np.testing.assert_allclose(rb[:,:,2],rb[:,:,0]-rb[:,:,1],rtol=3e-10,atol=1e-11,equal_nan=True)
    mask=np.arange(13)[None,:]<=np.asarray(TSEPS)[:,None];products=[]
    for ch,channel in enumerate(CHANNELS):
        for jf,flavor in enumerate(FLAVORS):
            r,b=ro[ch,jf],rb[:,ch,jf];valid=np.broadcast_to(mask,r.shape).copy()
            assert np.isfinite(r[valid]).all() and np.isfinite(b[:,valid]).all()
            assert np.isnan(r[~valid]).all() and np.isnan(b[:,~valid]).all()
            if KEY=='L32x64':
                prior=PARENT64/'ratios'/f'ratio_{parent_tag(tau)}_{channel}_{flavor}_Nboot200_seed1115.npz'
                with np.load(prior,allow_pickle=False) as d:
                    np.testing.assert_array_equal(d['boot_indices'],ind)
                    np.testing.assert_allclose(r[:,1:],d['ratio_original'],rtol=2e-10,atol=2e-10,equal_nan=True)
                    np.testing.assert_allclose(b[:,:,1:],d['ratio_boot'],rtol=2e-10,atol=2e-10,equal_nan=True)
            else:
                prior=PARENT96/'ratios'/f'ratio_{tau_tag(tau)}_{channel}_{flavor}_Nboot100_seed1115.npz'
                with np.load(prior,allow_pickle=False) as d:
                    np.testing.assert_allclose(r,d['ratio_original'],rtol=2e-10,atol=2e-10,equal_nan=True)
                    np.testing.assert_allclose(b[:100],d['ratio_boot'],rtol=2e-10,atol=2e-10,equal_nan=True)
            path=ratio_path(tau,channel,flavor)
            atomic_npz(path,ratio_original=r,ratio_boot=b,ratio_valid_mask=valid,valid_tau_mask=mask,
                component_original=np.stack((r[0].real,r[1].imag)),
                component_boot=np.stack((b[:,0].real,b[:,1].imag),axis=1),
                axes_original=np.asarray('vdv,p,z,tsep,insertion'),axes_boot=np.asarray('boot,vdv,p,z,tsep,insertion'),
                axes_component_boot=np.asarray('boot,component,p,z,tsep,insertion'),
                channel=np.asarray(channel),flavor_combo=np.asarray(flavor),component_labels=np.asarray(COMPONENTS),
                gamma_matrices=SPECS.gamma,selected_pol_indices=SPECS.pol_indices,
                phase_factors=np.asarray([1.,-1j,1j]),c2_original=den,c2_boot=db,
                is_contact=(np.arange(13)[None,:]==0)|(np.arange(13)[None,:]==np.asarray(TSEPS)[:,None]),
                physical_result_status=np.asarray('finite_flow_connected_bare_ratio_not_matched_PDF'),**common_metadata(tau))
            products.append(dict(path=str(path),sha256=sha256(path)))
    atomic_json(marker,dict(status='PASS',tau=tau,nconf=NCONF,nboot=NBOOT,products=products,
        projected=str(projected),projected_sha256=sha256(projected),live_ledgers=live,
        Pz0_raw_cases_full_SHA_Wick_gamma_reverified=p0_cases,
        parent_bootstrap_regression='all 200 existing moving L64 replicas; first100 L96 replicas',
        bootstrap_indices_sha256=m['bootstrap_indices_sha256'],seconds=time.monotonic()-start,job=os.environ['SLURM_JOB_ID']))
    print('RATIO_BOOT200_COMPLETE',KEY,tau,flush=True)


if __name__=='__main__':
    main()
