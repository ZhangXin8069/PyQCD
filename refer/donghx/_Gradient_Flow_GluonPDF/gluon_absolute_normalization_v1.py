"""Normalization helpers: all assumptions documented by the no-z0 branch.

Connection-valued Clover: F_code=a^2*g*F_canonical at tree level.
Trace(Ta Tb)=delta_ab/2 cancels the unaveraged even-sum factor 2.
After the inherited gF*gF MS conversion, R=-g_MS^2*(aE)*I,
where I(lambda)=integral_0^1 dx xg(x) cos(x lambda) at leading twist.
The 1/pi cosine-transform input is H=2I, not I.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from replica_selection_minerror_v3 import select_cell
from extrapolate_unpolarized_tau_ge1_correlated_v2 import correlated_linear


def kinematic_factor(energy_lattice,alpha_s):
    E=np.asarray(energy_lattice,float)
    if alpha_s<=0 or np.any(E<=0) or not np.isfinite(E).all():
        raise ValueError('positive energy and coupling required')
    return -2/(4*np.pi*alpha_s*E)


def local_spin2_coefficient(tau,a_fm,mu,alpha_s,ca=3.):
    """Finite MSbar diagonal coefficient for the connection-field spin-2 part.

    Makino/Suzuki 1403.4772v5: zeta_11 Eq4.34 times Z_11 EqA.12,
    O_bare=Z O_R; subtract trace, set quark-block contributions aside.
    L=ln(2 mu^2 t)+gamma_E. Not the coefficient c1 of the TOTAL EMT.
    Applying this spin-2 coefficient to TU-SU requires its continuum forward
    invariant-amplitude projection; finite-a extra tensor effects unresolved.
    """
    t=np.asarray(tau)*(a_fm/.1973269804)**2
    if np.any(t<=0):raise ValueError('positive flow time required')
    L=np.log(2*mu**2*t)+np.euler_gamma
    return 1+alpha_s*ca/(4*np.pi)*((11/3)*L+7/3)


def whitening(boot,svdcut=1e-10):
    cov=np.cov(boot,rowvar=False,ddof=1);sd=np.sqrt(np.diag(cov))
    if np.any(sd<=0):raise ValueError('zero variance')
    corr=cov/sd[:,None]/sd[None,:];ev,vec=np.linalg.eigh((corr+corr.T)/2)
    if ev[0]<-1e-10*ev[-1]:raise ValueError('invalid covariance')
    raw=int((ev>1e-12*ev[-1]).sum());keep=ev>svdcut*ev[-1]
    W=vec[:,keep].T/np.sqrt(ev[keep])[:,None]/sd[None,:]
    return W,cov,ev,raw,int(keep.sum())


def fit_c2_window(t,curves,svdcut=1e-10):
    """C2=A exp[-E(t-tmin)] using fixed full covariance, no log fit.

    Forward-only approximation: T=96 and t<=15, thermal term omitted.
    Amplitude is profiled analytically; positive E is not a prior.
    """
    scale=curves[0,0];v=curves/scale
    W,cov,ev,raw,rank=whitening(v[1:],svdcut)
    if raw<2 or rank<=2:raise ValueError('C2 covariance rank insufficient')
    dt=t-t[0];wy=v@W.T;beta=[];chi=[];condition=[]
    for row in wy:
        def solve(E):
            phi=np.exp(-E*dt);wp=W@phi
            A=wp@row/(wp@wp);res=row-A*wp
            return res@res,A,phi
        opt=minimize_scalar(lambda E:solve(E)[0],bounds=(.1,2.),method='bounded',
                            options=dict(xatol=1e-10))
        if not opt.success or opt.x<=.10001 or opt.x>=1.99999:raise ValueError('C2 energy boundary')
        c,A,phi=solve(opt.x)
        J=W@np.column_stack([-A*dt*phi,phi]);J/=np.linalg.norm(J,axis=0)
        if np.linalg.matrix_rank(J,tol=1e-12)<2:raise ValueError('C2 model Jacobian rank')
        beta.append([opt.x,A*scale]);chi.append(c);condition.append(np.linalg.cond(J))
    return dict(beta=np.array(beta),chi=np.array(chi),covariance=cov*scale**2,
                correlation_eigenvalues=ev,raw_rank=raw,rank=rank,dof=rank-2,
                scaled_whitened_jacobian_condition=np.array(condition))


def fit_energies(t,curves,args):
    windows=np.array([(lo,hi) for lo in [8,9,10,11,12] for hi in [13,14,15] if hi-lo>=3])
    results=[];fits=[];penalty=4+2*(8-(windows[:,1]-windows[:,0]+1))
    for ip in range(curves.shape[-1]):
        fs=[]
        for lo,hi in windows:
            use=(t>=lo)&(t<=hi)
            fs.append(fit_c2_window(t[use],curves[:,use,ip],args.svdcut))
        b=np.stack([f['beta'] for f in fs],axis=1);c=np.stack([f['chi'] for f in fs],axis=1)
        r=select_cell(b[0],b[1:],c[0],c[1:],np.array([f['dof'] for f in fs]),
            penalty,np.zeros(len(fs),int),np.ones(len(fs),bool),qmin=args.qmin,chi2max=args.chi2max)
        if not r['selected_available'] or not r['bootstrap_selected_available'].all():raise ValueError('C2 incomplete')
        r.update(windows=windows,candidate_params=b,candidate_chi2=c,
                 candidate_dof=np.array([f['dof'] for f in fs]),
                 candidate_raw_rank=np.array([f['raw_rank'] for f in fs]),
                 candidate_retained_rank=np.array([f['rank'] for f in fs]),
                 candidate_model_condition=np.stack([f['scaled_whitened_jacobian_condition'] for f in fs]),
                 penalty=penalty)
        results.append(r);fits.append(fs)
    return results,fits


def fit_local_flow(times,curves,args):
    windows=np.array([(i,j) for i in range(len(times)-2) for j in range(i+3,len(times)+1)])
    fits=[correlated_linear(times[lo:hi],curves[0,lo:hi],curves[1:,lo:hi],args.svdcut) for lo,hi in windows]
    b=np.stack([f['beta'] for f in fits]);bb=np.stack([f['beta_boot'] for f in fits],axis=1)
    chi=np.array([f['chi2'] for f in fits]);cb=np.stack([f['chi2_boot'] for f in fits],axis=1)
    penalty=4+2*(len(times)-(windows[:,1]-windows[:,0]))
    r=select_cell(b,bb,chi,cb,np.array([f['dof'] for f in fits]),penalty,
        np.zeros(len(fits),int),np.ones(len(fits),bool),qmin=args.qmin,chi2max=args.chi2max)
    if not r['selected_available'] or not r['bootstrap_selected_available'].all():raise ValueError('local flow incomplete')
    r.update(windows=windows,candidate_params=b,candidate_params_boot=bb,
             candidate_chi2=chi,candidate_chi2_boot=cb,candidate_dof=np.array([f['dof'] for f in fits]),
             candidate_raw_rank=np.array([f['raw_rank'] for f in fits]),
             candidate_retained_rank=np.array([f['retained_rank'] for f in fits]),
             candidate_design_condition=np.array([f['scaled_whitened_design_condition'] for f in fits]),
             penalty=penalty)
    cov=np.full((len(fits),len(times),len(times)),np.nan)
    eig=np.full((len(fits),len(times)),np.nan)
    for k,((lo,hi),f) in enumerate(zip(windows,fits)):
        cov[k,lo:hi,lo:hi]=f['covariance'];eig[k,:hi-lo]=f['correlation_eigenvalues']
    r.update(covariance=cov,correlation_eigenvalues=eig)
    return r
