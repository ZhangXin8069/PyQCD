import lsqfit
import gvar as gv
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(113)


Mall1=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/C24P29/param_E/Nucleon_mom0_7_22_n872_nboot3000_Nev100_E.npy")[0]
Mall2=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/C32P29/param_E/Nucleon_mom0_7_27_n982_nboot3000_Nev100_E.npy")[0]
Mall3=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/C48P29/param_E/Nucleon_mom0_7_28_n750_nboot3000_Nev200_E.npy")[0]

# Mall1=np.random.normal(0.5465,0.0019,3000)
# Mall2=np.random.normal(0.5364,0.0012,3000)
# Mall3=np.random.normal(0.5375,0.0007,3000)

# Mall1=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/C24P29/param_E/weighted_E.npz")
# Mall2=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/C32P29/param_E/weighted_E.npz")
# Mall3=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/C48P29/param_E/weighted_E.npz")

merr1=np.std(Mall1) #np.sqrt(Mall1["stat"]**2+Mall1["sys"]**2)
merr2=np.std(Mall2) #np.sqrt(Mall2["stat"]**2+Mall2["sys"]**2)
merr3=np.std(Mall3) #np.sqrt(Mall3["stat"]**2+Mall3["sys"]**2)

# print(Mall1["stat"],Mall1["sys"])
# print(Mall2["stat"],Mall2["sys"])
# print(Mall3["stat"],Mall3["sys"])
# print(Mall1["mean"],Mall2["mean"],Mall3["mean"])
# print(merr1,merr2,merr3)
# Nsample=Mall1["weightEall"].shape[0]

mpi=0.1561917

def volumefn(L,p):
    return p["c"]*(np.exp(-mpi*L)/mpi/L)+p["Minf"]

fig,ax=plt.subplots()

# M1tot=np.random.normal(Mall1["mean"],merr1,3000)
# M2tot=np.random.normal(Mall2["mean"],merr2,3000)
# M3tot=np.random.normal(Mall3["mean"],merr3,3000)

ydata=np.array([gv.gvar(np.mean(Mall1),merr1),gv.gvar(np.mean(Mall2),merr2),gv.gvar(np.mean(Mall3),merr3)])
x=np.array([24,32,48])
p0={"c":1,"Minf":0.54}
fit=lsqfit.nonlinear_fit(data=(x,ydata),fcn=volumefn,p0=p0)
print(fit.p)
print(fit)
print("chi/dof=",fit.chi2)
total=np.zeros([3000,2])
for i in range(3000):
    ydata=np.array([gv.gvar(Mall1[i],merr1),gv.gvar(Mall2[i],merr2),gv.gvar(Mall3[i],merr3)])
    fit=lsqfit.nonlinear_fit(data=(x,ydata),fcn=volumefn,p0=p0)
    total[i,0]=(fit.p)["c"].mean
    total[i,1]=(fit.p)["Minf"].mean
print("total",np.mean(total,0),np.std(total,0))


xplot=np.arange(0,81,1)
y=[]
for x in xplot:
    y.append(volumefn(x,{"c":total[:,0],"Minf":total[:,1]}))
yploymean=np.mean(y,1)
yerr=np.std(y,1)





plt.errorbar([24,32,48],[np.mean(Mall1),np.mean(Mall2),np.mean(Mall3)],yerr=[merr1,merr2,merr3],linestyle="none",color="black",marker=".",capthick=0.7,capsize=2,zorder=100)
# plt.errorbar([24,32,48],[Mall1["mean"],Mall2["mean"],Mall3["mean"]],yerr=[Mall1["stat"],Mall2["stat"],Mall3["stat"]],linestyle="none",color="black",marker=".",capthick=0.7,capsize=2,zorder=100)
plt.fill_between(xplot,y1=yploymean+yerr,y2=yploymean-yerr,color="cornflowerblue",alpha=0.3,linewidth=0.5)
plt.errorbar(80,np.mean(total[:,1]),yerr=np.std(total[:,1]),linestyle="none",color="blue",marker=".",capthick=0.7,capsize=2,label="$m^\infty_N$")
plt.legend(loc="upper right",fontsize=15)



# M1stat=np.random.normal(Mall1["mean"],Mall1["stat"],3000)
# M2stat=np.random.normal(Mall2["mean"],Mall2["stat"],3000)
# M3stat=np.random.normal(Mall3["mean"],Mall3["stat"],3000)

# ydata=np.array([gv.gvar(Mall1["mean"],Mall1["stat"]),gv.gvar(Mall2["mean"],Mall2["stat"]),gv.gvar(Mall3["mean"],Mall3["stat"])])
# x=np.array([24,32,48])
# p0={"c":1,"Minf":0.54}
# fit=lsqfit.nonlinear_fit(data=(x,ydata),fcn=volumefn,p0=p0)
# print(fit.p)
# print(fit)
# print("chi/dof=",fit.chi2)
# stat=np.zeros([3000,2])
# for i in range(3000):
#     ydata=np.array([gv.gvar(M1stat[i],Mall1["stat"]),gv.gvar(M2stat[i],Mall2["stat"]),gv.gvar(M3stat[i],Mall3["stat"])])
#     fit=lsqfit.nonlinear_fit(data=(x,ydata),fcn=volumefn,p0=p0)
#     stat[i,0]=(fit.p)["c"].mean
#     stat[i,1]=(fit.p)["Minf"].mean
# print("stat",np.mean(stat,0),np.std(stat,0))


# xplot=np.arange(0,81,1)
# y=[]
# for x in xplot:
#     y.append(volumefn(x,{"c":stat[:,0],"Minf":stat[:,1]}))
# yerr=np.std(y,1)
# plt.plot(xplot,yploymean,color="cornflowerblue",alpha=0.9,linewidth=0.5)
# plt.fill_between(xplot,y1=yploymean+yerr,y2=yploymean-yerr,color="cornflowerblue",alpha=0.6,linewidth=0.5)
# plt.errorbar(80,np.mean(stat[:,1]),yerr=np.std(stat[:,1]),linestyle="none",color="blue",marker=".",capthick=0.7,capsize=2)

# np.savez("/public/home/gengyq/laph/NP/fitwork/picture/Volume_expasion.npz",tot=total,stat=stat)

# plt.subplots_adjust(left=0.15,right=0.95)
# ax.yaxis.set_label_coords(0.1, 0.5) 
#
plt.ylim(0.53,0.56)
plt.xlim(18,81)
plt.xticks([24,32,48,64,80],["24","32","48","...",r"$\infty$"])
plt.xlabel("L/a")
plt.ylabel(r"$am_N$", labelpad=5)

plt.savefig("/public/home/gengyq/test.jpg",dpi=400)
plt.savefig("/public/home/gengyq/laph/NP/fitwork/picture/Volume_expasion.pdf",dpi=400)