import numpy as np
import matplotlib.pyplot as plt


state="C32P29"


jack=False
from lsq_tools import *
momlist=["Px0Py0Pz0","Px0Py0Pz1","Px0Py1Pz1","Px1Py1Pz1","Px0Py0Pz2"] 
color=["blue","green","orange","red","purple"]
dt=[-0.2,-0.1,0,0.1,0.2]
plt.figure()
for p in range(1):
    print(F"momentum {p} job starts!!!!")

    if state=="C32P29":
        a_=read_dat_file(F"/public/group/lqcd/corr/twopt/xinghy/beta6.20_mu-0.2770_ms-0.2400_L32x64/Nev100/corr_Nucleon_pp_{momlist[p]}",".dat",True)
        
        b_=read_dat_file(F"/public/home/gengyq/laph/proton/C32P29/result/corr_Nucleon_pp_{momlist[p]}.conf",".dat",False)/64
        a=np.concatenate([a_,b_],0)
        print(np.mean(a_,0))
        print(np.mean(b_,0))
        print(np.mean(a,0))
   
    print('data set : ',a_.shape)
    print('data set : ',b_.shape)

    boot_corr1 = bootstrap(a_ , 3000)
    boot_corr2 = bootstrap(b_ , 3000)
    boot_corr3 = bootstrap(a , 3000)
    st0 = time.time()
    
    contract1 = boot_corr1[:,:32]
    contract2 = boot_corr2[:,:32]
    contract3 = boot_corr3[:,:32]
    

    contract_sum1 = np.sum(contract1, 0)
    sum_average1 = np.mean(contract1, 0)
    contract_sum2 = np.sum(contract2, 0)
    sum_average2 = np.mean(contract2, 0)
    contract_sum3 = np.sum(contract2, 0)
    sum_average3 = np.mean(contract2, 0)

    meson_mass_average1 = np.log(np.roll(sum_average1, 1) / (sum_average1)) 
    meson_mass_jk1 =np.log(np.roll(contract1, 1, 1) / (contract1)) 
    jkerros1 = np.std(meson_mass_jk1, 0)

    meson_mass_average2 = np.log(np.roll(sum_average2, 1) / (sum_average2)) 
    meson_mass_jk2 =np.log(np.roll(contract2, 1, 1) / (contract2)) 
    jkerros2 = np.std(meson_mass_jk2, 0)
    meson_mass_average3 = np.log(np.roll(sum_average3, 1) / (sum_average3)) 
    meson_mass_jk3 =np.log(np.roll(contract3, 1, 1) / (contract3)) 
    jkerros3 = np.std(meson_mass_jk3, 0)

    
    T = np.arange(64//2)
    plt.errorbar(
                x=T+dt[p],
                y=meson_mass_average1,
                yerr=jkerros1,
                ecolor=color[p],
                linestyle="none",
                mec=color[p],
                marker="o",
                alpha=0.7,
                markerfacecolor="none",
                capsize=2,
                capthick=1,
            )
    
    
    plt.errorbar(
                x=T+dt[p+1],
                y=meson_mass_average2,
                yerr=jkerros2,
                ecolor=color[p+1],
                linestyle="none",
                mec=color[p+1],
                marker="o",
                alpha=0.7,
                markerfacecolor="none",
                capsize=2,
                capthick=1,
            )

    plt.errorbar(
                x=T+dt[p+2],
                y=meson_mass_average3,
                yerr=jkerros3,
                ecolor=color[p+2],
                linestyle="none",
                mec=color[p+2],
                marker="o",
                alpha=0.7,
                markerfacecolor="none",
                capsize=2,
                capthick=1,
            )
    
    
    if state=="C24P29":
        plt.ylim(0.45,1)
        plt.xlim(0.5,22.5)
        plt.xticks(np.arange(0,22,5),np.arange(0,22,5))
        plt.text(3,(1-0.45)*0.1+0.45,f"{state}",fontsize=15)
    elif state=="C32P29":
        plt.ylim(0.52,0.56)
        plt.xlim(0.5,28.5)
        plt.xticks(np.arange(0,28,5),np.arange(0,28,5))
    elif state=="C48P29":
        plt.ylim(0.5,0.65)
        plt.xlim(0.5,28.5)
        plt.xticks(np.arange(0,28,5),np.arange(0,28,5))
        plt.text(3,(0.65-0.5)*0.1+0.5,f"{state}",fontsize=15)
plt.xlabel("T/a")
plt.ylabel(r"$aM_{eff}$")


plt.savefig("/public/home/gengyq/test.jpg",dpi=400)
# plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/effmass/{state}_Nucleon_effmass.pdf",dpi=400)