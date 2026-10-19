#!/public/home/xinghy/anaconda3-2023.03/bin/python

# test file to check whether first term has problem

import numpy as np
import fileinput
import math
import matplotlib.pyplot as plt
import copy
from scipy.optimize import curve_fit
import lsqfit
import gvar as gv
from html.entities import entitydefs
import seaborn as sns
from prettytable import PrettyTable
import matplotlib.patches as mpatches
from lsq_tools import *

state = "C24P29"
object = "Nucleon"  
Nt0 = 72
N_a = 0.077
N = 872
nbsamples=3000

if state=="C24P29":
    conf_name = "beta6.20_mu-0.2770_ms-0.2400_L24x72"
    Nx=24
    Nmom=5
    Nsamples=872
    beta=np.array([22,22,21,19,15])
    # start=np.array([14,14,13,12,11])
    start=np.array([7,6,6,5,5])
    tfit2 = np.array([0,0,0,0,0])  # choose time - start time
    Nev=100
if state=="C32P29":
    conf_name = "beta6.20_mu-0.2770_ms-0.2400_L32x64"
    Nx=32
    Nmom=5
    Nsamples=982
    beta=np.array([27,25,21,21,19])
    start=np.array([7,6,6,5,5])
    tfit2 = np.array([0,0,0,0,0])  # choose time - start time
    Nev=100
if state=="C48P29":
    conf_name = "beta6.20_mu-0.2770_ms-0.2400_L48x96"
    Nx=48
    Nmom=5
    Nsamples=284
    beta=np.array([28,26,26,27,20])
    start=np.array([7,6,6,6,5] )
    tfit2 = np.array([0,0,0,0,0])  # choose time - start time
    Nev=200

deltat=1

savefile = F"/public/home/gengyq/laph/NP/fitwork/picture/{state}/{state}_{object}_dispersion_twostate.pdf"
# -----------------------fit disperion
E_all = []
for p in range(Nmom):
    try:
        E0=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_E/{object}_mom{p}_{start[p]}_{beta[p]}_n{Nsamples}_nboot{nbsamples}_Nev{Nev}_E.npy")[tfit2[p]]
    except:
        if state=="C32P29":
            E0=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_E/{object}_mom{p}_{start[p]}_{beta[p]}_n428_nboot{nbsamples}_Nev{Nev}_E.npy")[tfit2[p]]
        elif state=="C48P29":
            E0=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_E/{object}_mom{p}_{start[p]}_{beta[p]}_n284_nboot{nbsamples}_Nev{Nev}_E.npy")[tfit2[p]]
    
    E_all.append(E0)


E_all=np.asarray(E_all)   # mom,confid
E_all=np.transpose(E_all,(1,0))  #  confid, mom

N=nbsamples

print("2EdE=",2*np.mean(E_all,0)*np.std(E_all,0))
E_all=E_all**2
E_all_ave = np.mean(E_all,0)
E_all_err = np.std(E_all,0)
P = (np.array([0, 1, np.sqrt(2), np.sqrt(3)])*2*np.pi/Nx) ** 2
if Nmom==5:
    P = (np.array([0, 1, np.sqrt(2), np.sqrt(3), 2])*2*np.pi/Nx) ** 2
print("E2=",E_all_ave)
print("dE2=",E_all_err)

print("P=",P)

def fcn(P, p):
    return p["a"] * P + p["b"]**2


pr = np.zeros([N, 2])
p0 = {"a":1, "b":E_all_ave[0]} #

M_ij = np.zeros((Nmom, Nmom))
for i in range(Nmom):
    M_ij[i] = np.sum(np.array([E_all[:, i] - E_all_ave[i]]).T * (E_all - E_all_ave),0,)

M_ij=M_ij / N
M_dig = np.tril(M_ij)
M_dig = np.triu(M_dig)

for fid in range(N):
    gv_corr = gv.gvar(E_all[fid], M_ij)
    fit = lsqfit.nonlinear_fit(data=(P, gv_corr), fcn=fcn, p0=p0)  # 拟合
    # print(fit)
    pr[fid][0] = fit.p["a"].mean
    pr[fid][1] = fit.p["b"].mean
    if fid==0:
        chi_=fit.chi2 / (fit.dof)
        print("chi2/dof=", chi_)
print("dof", fit.dof)
np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/Zsloop/{object}_disperison_Nev{Nev}_twostate.npy",pr) ####################################################

pr_aver_Z = np.mean(pr[:, 0], 0)
pr_aver_m0 = np.mean(pr[:, 1], 0)
print("z",pr[:, 0])
print("Z_mean=", pr_aver_Z)
print("m0_mean=", pr_aver_m0)

Z_err = np.std(pr[:, 0], 0) 
m0_err = np.std(pr[:, 1], 0) 
print("Z_error=", Z_err)
print("m0_error=", m0_err)

# -----------------------picture

# Y1 = (pr_aver_Z + Z_err) * P + (pr_aver_m0 + m0_err)**2
# Y2 = (pr_aver_Z - Z_err) * P + (pr_aver_m0 - m0_err)**2

fitE=[]
for p in P:
    fitE.append(pr[:, 0] * p + pr[:, 1]**2)
fitE=np.array(fitE)
Y1=np.mean(fitE,1)+np.std(fitE,1)
Y2=np.mean(fitE,1)-np.std(fitE,1)

fig2 = plt.figure()
plt.errorbar(
    x=P,
    y=E_all_ave,
    yerr=E_all_err,
    color="black",
    marker="",
    alpha=0.7,
    linestyle="none",
    capsize=2,
    capthick=1,
    #label=f"{object}",
)

plt.fill_between(P, Y1, Y2, facecolor="cornflowerblue", alpha=0.4)
plt.text(x=(P[2]+P[3])/2, y=E_all_ave[0]+(E_all_ave[4]-E_all_ave[0])*0.1, s="$c^2$ = %.3f(%.0f)" % (pr_aver_Z, Z_err * 1e3))
plt.text(x=(P[2]+P[3])/2, y=E_all_ave[0], s="$\chi^2/d.o.f.$ = %.2f" % (chi_))
# plt.errorbar(x=P,y=pr_aver*(P+E_all_ave[0]),yerr=M_err**2,color='blue',ecolor='blue',label='fitting',mec='blue',marker = 'o',alpha = 0.7,markerfacecolor = 'none',capsize =2,capthick=1)
# plt.plot(P_pion**2,P_pion**2++E_pion[0]**2,label='E-P function')
#plt.legend(loc=0)
plt.title("%s %s dispersion relation" % (state, object))
plt.xlabel("P$^2$")
plt.ylabel("E$^2$")
plt.savefig("/public/home/gengyq/test.jpg",dpi=400)
plt.savefig(F"{savefile}",dpi=400)
