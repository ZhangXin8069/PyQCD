"""CPU-only coefficient audit. No matrix elements, production output or line mass is changed."""
from pathlib import Path
import os,json,csv,hashlib,math
import mpmath as mp
ROOT=Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID'),'CPU Slurm allocation required'
mp.mp.dps=40
CF=mp.mpf(4)/3;CA=mp.mpf(3);TR=mp.mpf('.5')
l2=mp.log(2);l3=mp.log(3);pi=mp.pi;cl=mp.im(mp.polylog(2,mp.e**(1j*pi/3)));li2=mp.polylog(2,mp.mpf('.25'));li3=mp.polylog(3,mp.mpf('.25'));zeta2=mp.zeta(2)
def quark_series(L,nf=3,channel='V',ringed_constant=None):
 # arXiv:2610.40225v1 Eq.(40), squared to form Eq.(34).
 L=mp.mpf(L);cc=(-mp.mpf('23.7947')*CF*CA+mp.mpf('30.3914')*CF**2-mp.mpf('3.92255')*CF*TR*nf) if ringed_constant is None else mp.mpf(ringed_constant)
 h1=-CF*(mp.mpf('1.5')*L+1+mp.log(432)/2)
 h2=L**2*(mp.mpf(9)/8*CF**2-mp.mpf(11)/4*CF*CA+CF*TR*nf)
 h2+=L*(CF**2*(mp.mpf(11)/4-4*pi**2/3+mp.mpf(3)/4*mp.log(432))+CF*CA*(pi**2/3-mp.mpf(31)/4-mp.mpf(11)/6*mp.log(432))+CF*TR*nf*(3+mp.mpf(2)/3*mp.log(432)))
 h2+=CF**2*(2+4*pi**2/3*(l2-1)-4*pi*cl+12*l2-mp.mpf(3)/8*l3*(4+8*l2+3*l3)+mp.mpf(11)/2*li2-li3-mp.mpf(4)/3*mp.zeta(3))
 h2+=CF*CA*(-mp.mpf(239)/6-pi+pi**2/6-pi**2/3*l2+pi*cl-85*l2+84*l3+2*l2**2+17*li2-li3/2-mp.mpf(2)/3*mp.zeta(3))
 h2+=CF*TR*nf*(mp.mpf(20)/3+pi**2/3)+cc/2
 n1=2*h1;n2=2*h2+h1*h1
 # arXiv:2311.16799v2 Eqs.(3.9),(3.10); that paper uses alpha_s/pi.
 if channel=='V':
  j1=4*CF*(mp.mpf(1)/8-l2-mp.mpf(3)/4*l3)
  j2=cc+16*(CF**2*(-mp.mpf(41)/128-mp.mpf(5)/32*zeta2+mp.mpf(3)/8*l2+l2**2/4-mp.mpf(3)/32*l3+mp.mpf(3)/2*li2)+CA*CF*(-mp.mpf(763)/384-mp.mpf(5)/32*zeta2-mp.mpf(13)/4*l2+l2**2/4+mp.mpf(27)/8*l3+mp.mpf(21)/16*li2)+CF*TR*nf*(mp.mpf(35)/96+zeta2/8)+L*(CA*CF*(-mp.mpf(11)/12*l2-mp.mpf(11)/16*l3+mp.mpf(11)/96)+CF*TR*nf*(l2/3+l3/4-mp.mpf(1)/24)))
 elif channel=='T':
  j1=4*CF*(-l2-mp.mpf(3)/4*l3+L/4)
  j2=cc+16*(CF**2*(-mp.mpf(7)/12-zeta2/4+mp.mpf(4)/3*l2+l2**2/4-mp.mpf(3)/4*l3+li2)+CA*CF*(-mp.mpf(1159)/864-zeta2/16-mp.mpf(17)/6*l2+l2**2/4+3*l3+mp.mpf(5)/4*li2)+CF*TR*nf*(mp.mpf(47)/216+zeta2/12)+L*(CF**2*(-l2/4-mp.mpf(3)/16*l3-mp.mpf(19)/32)+CA*CF*(-mp.mpf(11)/12*l2-mp.mpf(11)/16*l3+mp.mpf(257)/288)+CF*TR*nf*(l2/3+l3/4-mp.mpf(13)/72))+L**2*(CF**2/32+mp.mpf(11)/96*CA*CF-CF*TR*nf/24))
 else:raise ValueError(channel)
 return float(n1),float(n2),float(j1-n1),float(j2-n2+n1*n1-j1*n1)
def gluon_endpoints(L,nf=3):
 L=mp.mpf(L)
 p1=mp.mpf(0);p2=CA**2*(-2*pi**2/3*L+2*pi**2/3*(l2-mp.mpf(5)/3)-4*pi/3*(-2+mp.sqrt(3)+cl))
 b1=CA*L
 b2=L**2*(mp.mpf(7)/3*CA**2-mp.mpf(2)/3*CA*TR*nf)+L*(CA**2*(mp.mpf(34)/9-2*pi**2/3)-mp.mpf(26)/9*CA*TR*nf)
 b2+=CA**2*(mp.mpf(196)/27+pi-pi**2/6-4*pi/3*cl+(72+2*pi**2/3)*l2-mp.mpf(10)/3*l2**2-mp.mpf(123)/2*l3+4*li2)-CA*TR*nf/27*(89+3*pi**2)
 return [float(x) for x in [p1,p2,b1,b2]]
def inverse_series(c1,c2,alpha):
 a=alpha/(4*math.pi);return 1-c1*a,1-c1*a+(c1*c1-c2)*a*a
checks=[]
for nf in [0,3,4]:
 for L in [-1,0,1,2]:
  for ch,p,c in [('V',3,2.5),('T',4,2)]:
   n1,n2,d1,d2=quark_series(L,nf,ch);assert abs(d1-float(CF)*(p*L+c))<1e-12
   assert abs(quark_series(L,nf,ch,ringed_constant=0)[3]-quark_series(L,nf,ch,ringed_constant=100)[3])<1e-11
   checks.append('quark NLO recovery and ringed cancellation nf=%d L=%d channel=%s'%(nf,L,ch))
  p1,p2,b1,b2=gluon_endpoints(L,nf)
  assert abs(2*p2-(-118.435*L-171.621))<.002
  bil2=2*b2+b1*b1;rounded=(51-2*nf)*L**2-(50.4353+26*nf/3)*L-164.226-13.1788*nf
  assert abs(bil2-rounded)<.003
  mix2=p2+b2;rounded=(21-nf)*L**2-(84.4353+13*nf/3)*L-167.924-6.58938*nf
  assert abs(mix2-rounded)<.003
  checks.append('gluon exact endpoint products vs rounded bilocal Eqs51-53 nf=%d L=%d'%(nf,L))
rows=[]
for alpha in [.25,.30]:
 for L in [0.,-.5,.5,1.]:
  for ch in ['V','T']:
   n1,n2,d1,d2=quark_series(L,3,ch);a=alpha/(4*math.pi)
   rows.append(dict(case='quark local ratio '+ch,alpha_s=alpha,nf=3,L=L,NLO_factor=1+a*d1,NNLO_strict_factor=1+a*d1+a*a*d2,NNLO_minus_NLO=a*a*d2,coefficient_a1=d1,coefficient_a2=d2))
  p1,p2,b1,b2=gluon_endpoints(L,3)
  for name,c1,c2 in [('gluon U: perp-perp pair',2*b1,2*b2+b1*b1),('gluon H: mixed pair',p1+b1,p2+b2+p1*b1),('gluon parallel pair',2*p1,2*p2+p1*p1)]:
   v1,v2=inverse_series(c1,c2,alpha)
   rows.append(dict(case=name,alpha_s=alpha,nf=3,L=L,NLO_factor=v1,NNLO_strict_factor=v2,NNLO_minus_NLO=v2-v1,coefficient_a1=c1,coefficient_a2=c2))
with (ROOT/'coefficient_probe.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Existing code is inspected as text only, not imported or changed.
status={'status':'PASS_COEFFICIENT_ALGEBRA_AUDIT','job':os.environ['SLURM_JOB_ID'],'checks':len(checks),'details':checks,'scope':'SU3 massless nonsinglet quark-ratio and gluon coefficients only; not a validated production matching','exact_source_equations':['2610.40225v1 Eq40 squared; Eqs43-44','2311.16799v2 Eqs3.9-3.10; alpha_s/pi -> alpha_s/(4pi) conversion'],'line_mass_included':False,'matrix_elements_read':False,'RGR_implemented':False,'local_axial_finite_gamma5_scheme_checked_in_code':False,'transversity_external_sign_confirmed':False,'example_at_L0_alpha025':[r for r in rows if r['L']==0 and r['alpha_s']==.25]}
(ROOT/'probe_report.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({k:v for k,v in status.items() if k!='details'},indent=2),flush=True)
