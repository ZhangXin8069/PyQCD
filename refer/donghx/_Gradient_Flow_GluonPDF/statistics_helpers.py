"""Paired ensemble loop/line log-flow slopes and explicit geometry conversion."""
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import traceback
import numpy as np
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
REP=['fundamental','adjoint']

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def write(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(clean(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def table(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',newline='') as stream:
  writer=csv.DictWriter(stream,keys);writer.writeheader()
  for r in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in clean(r).items()})
def jk_error(v):return np.sqrt((len(v)-1)/len(v)*np.sum((v-v.mean(axis=0))**2,axis=0))
def covariance(jk):
 d=jk-jk.mean(axis=0);return (len(jk)-1)/len(jk)*(d.T@d)
def log_stats(raw):
 n=len(raw);mean=raw.mean(axis=0);jk=(raw.sum(axis=0)-raw)/(n-1)
 valid=np.isfinite(mean)&(mean>0)&np.all(np.isfinite(jk)&(jk>0),axis=0)
 y=np.full(mean.shape,np.nan);jy=np.full(jk.shape,np.nan)
 y[valid]=np.log(mean[valid]);jy[:,valid]=np.log(jk[:,valid])
 return mean,y,jy,valid

def gls(y,jk,X,cut=1e-10):
 """Fixed full JK covariance, explicit correlation eigenmode deletion; no prior."""
 y=np.asarray(y);jk=np.asarray(jk);X=np.asarray(X)
 nd,np_=X.shape
 if nd<=np_ or not np.isfinite(y).all() or not np.isfinite(jk).all():
  return dict(status='statistically_unidentifiable',reason='insufficient/invalid data'),None
 cov=covariance(jk);s=np.sqrt(np.diag(cov))
 if not np.all(s>0):return dict(status='statistically_unidentifiable',reason='zero variance'),None
 cor=cov/np.outer(s,s);ev,U=np.linalg.eigh((cor+cor.T)/2)
 assert ev[0]>=-1e-10*max(ev[-1],1),ev
 raw_rank=int(np.sum(ev>ev[-1]*1e-12));keep=ev>ev[-1]*cut;rank=int(keep.sum())
 if raw_rank<np_ or rank<=np_:
  return dict(status='statistically_unidentifiable',reason='covariance rank',sample_rank=raw_rank,effective_rank=rank),None
 white=(U[:,keep]/np.sqrt(ev[keep])).T/s[None,:]
 A=white@X;scale=np.linalg.norm(A,axis=0);model_rank=int(np.linalg.matrix_rank(A/scale))
 if model_rank<np_:return dict(status='statistically_unidentifiable',reason='model rank'),None
 # QR/SVD solution; independently compare the normal-equation map.
 transform=np.linalg.lstsq(A,np.eye(rank),rcond=1e-12)[0]@white
 alt=np.linalg.solve(A.T@A,A.T)@white
 assert np.allclose(transform,alt,rtol=3e-6,atol=1e-7*max(1,np.max(abs(transform))))
 p=transform@y;pj=jk@transform.T;r=white@(y-X@p);cs=float(r@r);dof=rank-np_
 return dict(status='identifiable',params=p,errors=jk_error(pj),chi2=cs,dof=dof,chi2_dof=cs/dof,Q=float(chi2.sf(cs,dof)),
  sample_rank=raw_rank,effective_rank=rank,required_rank=np_,Ndata=nd,correlation_eigenvalues=ev,
  retained_correlation_condition=float(ev[-1]/ev[keep][0]),model_condition=float(np.linalg.cond(A/scale)),
  covariance_relative_cut=cut,quality='model_tension' if chi2.sf(cs,dof)<.01 else 'no_Q_flag',prior=None),pj

def tests():
 x=np.array([1.2,1.4,1.6,1.8]);X=np.column_stack([x,np.ones(4)])
 rng=np.random.default_rng(20261113);noise=rng.normal(size=(120,4))*1e-3
 y=-.7*x+.2;j=y+noise
 fit,p=gls(y,j,X);assert np.allclose(fit['params'],[-.7,.2],atol=1e-9)
 n=np.arange(4,9);d=.32;k=-d/np.sqrt(8)
 line=k*n+.13;loop=2*k*n-.9
 assert abs(np.polyfit(n,line,1)[0]-np.polyfit(n,loop,1)[0]/2)<1e-14
 assert abs(-np.sqrt(8)*np.polyfit(n,line,1)[0]-d)<1e-14
 mean,y,j,v=log_stats(np.array([[.2,.5],[.8,.5],[.5,.6]]))
 assert abs(y[0]-np.log(.5))<1e-14 and abs(y[0]-np.log([.2,.8,.5]).mean())>.01
 _,_,_,v=log_stats(np.array([[1.,-1.],[1.,-2.]]));assert not v[1]
 f,_=gls(np.ones(4),np.ones((5,4)),X);assert f['status']=='statistically_unidentifiable'
 assert np.allclose(jk_error(noise)**2,np.diag(covariance(noise)))
 write(ROOT/'tests.json',dict(status='PASS',checks=['GLS SVD versus normal equations','synthetic slope sign/sqrt8/two spatial edges',
  'log of ensemble mean, not mean log','nonpositive points not abs or clipped','zero variance gate','jackknife covariance prefactor']))

