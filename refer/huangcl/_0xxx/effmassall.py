import numpy as np
import matplotlib.pyplot as plt


state="C48P29"

if state=="C24P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L24x72"
    Eexp=np.array([0.6,0.64,0.68,0.72,0.76])
    beta=[22,22,21,19,15]
    ymax=0.85 
    ymin=0.5 
    Lt = 50
    start1=[7,6,6,5,5] 
    Nsamples=872
    Nev=100
if state=="C32P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L32x64"
    Eexp=np.array([0.6,0.63,0.66,0.69,0.72])
    beta=[27,25,21,21,19]
    ymax=0.8
    ymin=0.45 
    Lt = 64
    start1=[7,6,6,5,5] 
    Nsamples=982
    Nev=100
if state=="C48P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L48x96"
    Eexp=np.array([0.54,0.56,0.58,0.6,0.62])
    beta=[28,26,26,27,20]
    ymax=0.8
    ymin=0.45 
    Lt = 64
    start1=[7,6,6,6,5] 
    Nmom = 5
    Nsamples=284
    Nev=200

jack=False
from lsq_tools import *
momlist=["Px0Py0Pz0","Px0Py0Pz1","Px0Py1Pz1","Px1Py1Pz1","Px0Py0Pz2"] 
color=["blue","green","orange","red","purple"]
dt=[-0.2,-0.1,0,0.1,0.2]
plt.figure()
for p in range(5):
    print(F"momentum {p} job starts!!!!")

    if state=="C24P29":
        a=read_dat_file(F"/public/group/lqcd/corr/twopt/xinghy/beta6.20_mu-0.2770_ms-0.2400_L24x72/Nev100/corr_Nucleon_pp_{momlist[p]}",".dat",True)
    elif state=="C32P29":
        a_=read_dat_file(F"/public/group/lqcd/corr/twopt/xinghy/beta6.20_mu-0.2770_ms-0.2400_L32x64/Nev100/corr_Nucleon_pp_{momlist[p]}",".dat",True)
        if p==0:
            b_=read_dat_file(F"/public/home/gengyq/laph/proton/C32P29/result/corr_Nucleon_pp_{momlist[p]}.conf",".dat",False)/64
            print(a_.shape,b_.shape)
            a=np.concatenate([a_,b_],0)
        else:
            a=a_
            Nsamples=428
    elif state=="C48P29":
        if p==0:
            Nsamples=750
        else:
            Nsamples=284
        a=read_dat_file(F"/public/home/gengyq/laph/proton/C48P29/result/corr_Nucleon_pp_{momlist[p]}.conf",".dat",False)
    elif state=="C48P14":
        a=read_dat_file(F"/public/group/lqcd/corr/twopt/xinghy/beta6.20_mu-0.2825_ms-0.2310_L48x96/corr_Nucleon_pp_{momlist[p]}",".dat",True)
    print('data set : ',a.shape)
    if jack==True:
        boot_corr = jackknife(a)
    else:
        boot_corr = bootstrap(a , 3000)
    st0 = time.time()
    
    contract = boot_corr[:,:Lt//2]
    print(contract.shape)
    N = contract.shape[0]  # your final number of samples
    print("N=", N)

    contract_sum = np.sum(contract, 0)
    sum_average = np.mean(contract, 0)

    meson_mass_average = np.log(np.roll(sum_average, 1) / (sum_average)) 
    meson_mass_jk =np.log(np.roll(contract, 1, 1) / (contract)) 
    jkerros = np.std(meson_mass_jk, 0)

    
    T = np.arange(Lt//2)
    plt.errorbar(
                x=T+dt[p],
                y=meson_mass_average,
                yerr=jkerros,
                ecolor=color[p],
                linestyle="none",
                mec=color[p],
                marker="o",
                alpha=0.7,
                markerfacecolor="none",
                capsize=2,
                capthick=1,
            )
    

    Efit=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_E/Nucleon_mom{p}_{start1[p]}_{beta[p]}_n{Nsamples}_nboot3000_Nev{Nev}_E.npy")[0]
    a=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_a/Nucleon_mom{p}_{start1[p]}_{beta[p]}_n{Nsamples}_nboot3000_Nev{Nev}_a.npy")[0]
    a1=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_a/Nucleon_mom{p}_{start1[p]}_{beta[p]}_n{Nsamples}_nboot3000_Nev{Nev}_a1.npy")[0]
    dE=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_a/Nucleon_mom{p}_{start1[p]}_{beta[p]}_n{Nsamples}_nboot3000_Nev{Nev}_dE.npy")[0]
    # else:
    #     Efit=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_E/Nucleon_mom{p}_{start1[p]}_{beta[p]}_n{Nsamples}_nboot3000_Nev{Nev}_E.npy")
    
    plotx = np.arange(start1[p]-1,beta[p],0.01)
    ploty=[]
    print(a.shape)
    for x in plotx:
        ploty.append(a * np.exp(-Efit *x) + a1* np.exp(-(Efit+dE**2)* x))
    ploty=np.array(ploty)
    mass=np.log(np.roll(ploty,100,0)/ploty)
    
    mean=np.mean(mass,1)
    err=np.std(mass)
    plt.fill_between(plotx[100:],y1=np.mean(mass,1)[100:]-np.std(mass,1)[100:],y2=np.mean(mass,1)[100:]+np.std(mass,1)[100:],color=color[p],alpha=0.4)
    
    if state=="C24P29":
        plt.ylim(0.45,1)
        plt.xlim(0.5,22.5)
        plt.xticks(np.arange(0,22,5),np.arange(0,22,5))
        plt.text(3,(1-0.45)*0.1+0.45,f"{state}",fontsize=15)
    elif state=="C32P29":
        plt.ylim(0.45,0.8)
        plt.xlim(0.5,28.5)
        plt.xticks(np.arange(0,28,5),np.arange(0,28,5))
        plt.text(3,(0.8-0.45)*0.1+0.45,f"{state}",fontsize=15)
    elif state=="C48P29":
        plt.ylim(0.5,0.65)
        plt.xlim(0.5,28.5)
        plt.xticks(np.arange(0,28,5),np.arange(0,28,5))
        plt.text(3,(0.65-0.5)*0.1+0.5,f"{state}",fontsize=15)

plt.xlabel("T/a")
plt.ylabel(r"$aM_{eff}$")


plt.savefig("/public/home/gengyq/test.jpg",dpi=400)
plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/effmass/{state}_Nucleon_effmass_twostate.pdf",dpi=400)