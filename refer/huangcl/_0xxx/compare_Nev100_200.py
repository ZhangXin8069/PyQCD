import numpy as np
from lsq_tools import *
import matplotlib.pyplot as plt
import lsqfit
import gvar as gv

Nt=27
v100=bootstrap(np.load("/public/home/gengyq/NP/raw/C48P29_T1plus_Nev100.npy")[:,0,0,:Nt],3000)
v200=bootstrap(np.load("/public/home/gengyq/NP/raw/C48P29_T1plus_Nev200.npy")[:,0,0,:Nt],3000)

v100=v100/v100[0,0]
v200=v200/v100[0,0]

print(v100.shape)
print(v200.shape)

M_100 = v100 - np.mean(v100,0)
M_100 = np.matmul(M_100.T, M_100) / 3000
M_200 = v200 - np.mean(v200,0)
M_200 = np.matmul(M_200.T, M_200) / 3000

mass100=np.log(np.roll(v100,1,1)/v100)
mass200=np.log(np.roll(v200,1,1)/v200)

#print(np.mean(mass100,0))

p0={"E":1.07,"a":0.8,"dE":0.3}
def func(x,p):
    return p["a"]*np.exp(-p["E"]*x)+(1-p["a"])*np.exp(-(p["E"]+p["dE"]**2)*x)
t=np.arange(0,Nt,1)

s1=8
d1=17
s2=9
d2=16

N=500
param1=np.zeros([N,3])
param2=np.zeros([N,3])
for i in range(N):
    t1=t[s1:d1]
    y1=gv.gvar(v100[i],M_100)[s1:d1]
    fit1=lsqfit.nonlinear_fit(data=(t1,y1),fcn=func,p0=p0)
    p1=fit1.p
    param1[i,0]=p1["E"].mean
    param1[i,1]=p1["a"].mean
    param1[i,2]=p1["dE"].mean

    t2=t[s2:d2]
    y2=gv.gvar(v200[i],M_200)[s2:d2]
    fit2=lsqfit.nonlinear_fit(data=(t2,y2),fcn=func,p0=p0)
    p2=fit2.p
    param2[i,0]=p2["E"].mean
    param2[i,1]=p2["a"].mean
    param2[i,2]=p2["dE"].mean
    if i==0:
        print(fit1.chi2/fit1.dof)
        print(fit2.chi2/fit2.dof)
print(np.mean(param1,0))
print(np.mean(param2,0))

plott=np.arange(0,Nt,0.001)
plot1=[]
plot2=[]
for i in range(N):
    corr1=func(plott,{"E":param1[i,0],"a":param1[i,1],"dE":param1[i,2]})
    corr2=func(plott,{"E":param2[i,0],"a":param2[i,1],"dE":param2[i,2]})
    plot1.append(np.log(np.roll(corr1,1000)/corr1))
    plot2.append(np.log(np.roll(corr2,1000)/corr2))

y11=np.mean(plot1,0)+np.std(plot1,0)
y12=np.mean(plot1,0)-np.std(plot1,0)
y21=np.mean(plot2,0)+np.std(plot2,0)
y22=np.mean(plot2,0)-np.std(plot2,0)

plt.figure()

plt.fill_between(plott,y1=y11,y2=y12,color="red",alpha=0.1)
plt.fill_between(plott,y1=y21,y2=y22,color="gray",alpha=0.1)
plt.fill_between(plott[1000*s1+1000:1000*d1],y1=y11[1000*s1+1000:1000*d1],y2=y12[1000*s1+1000:1000*d1],color="red",alpha=0.3)
plt.fill_between(plott[1000*s2+400:1000*d2+400],y1=y21[1000*s2+400:1000*d2+400],y2=y22[1000*s2+400:1000*d2+400],color="gray",alpha=0.3)
plt.errorbar(t,np.mean(mass100,0),yerr=np.std(mass100,0),capsize=2,capthick=1,\
             color="red",marker='o',linestyle="none",mfc="none",mec="red",label=r"$N_{ev}$=100")
plt.errorbar(t+0.4,np.mean(mass200,0),yerr=np.std(mass200,0),mfc="none",mec="black",capsize=2,capthick=1,\
             color="black",marker='o',linestyle="none",label=r"$N_{ev}$=200")
plt.ylim(1.05,1.1)
plt.xlim(5,Nt)
plt.xticks([5,10,15,20,25],[5,10,15,20,25])
plt.legend(loc="lower left",fontsize=15)
plt.xlabel(r"$t/a$",fontsize=12)
plt.ylabel(r"$aM_{eff}$",fontsize=12)
plt.savefig("/public/home/gengyq/test.jpg",dpi=400)
plt.savefig("/public/home/gengyq/laph/NP/fitwork/picture/C48P29/Nev100-200.pdf")
