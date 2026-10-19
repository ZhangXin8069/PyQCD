#!/public/home/xinghy/anaconda3-2023.03/bin/python

# test file to check whether first term has problem

import numpy as np
import fileinput
import math
import matplotlib.pyplot as plt
import time
import matplotlib.gridspec as gridspec
# import copy
import lsqfit
import pandas as pd
import gvar as gv
from lsq_tools import *

# from html.entities import entitydefs
# import seaborn as sns
from prettytable import PrettyTable
import matplotlib.patches as mpatches

shape = "twostate"
method = "p0"
Dwave=False #True
object = "NP"

if object=="NP":
    irrep=r"$T_1^+$"
else:
    irrep=r"$A_1^+$"

N = 3000
nbsamples = 3000
px = 0
py = 0
pz = 0
Na = 0.077
state = "C32P29"
GEVP=True#False#

jack=False
SAVEdataandgraph=True#False#
SAVEchijpg=True#False#
effbandinchi=True#False#
TEST=True#False#
Ptest= 0#None#
deltat= 11  
if TEST==True:
    SAVEdataandgraph=False
else:
    if SAVEdataandgraph==True:
        deltat=  1 
        SAVEchijpg=False
    elif SAVEdataandgraph==False:
        deltat=  10 
        SAVEchijpg=True
if state=="C24P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L24x72"
    svdcuts=np.array([1e-12,1e-12,1e-12,1e-12])
    Nt = 72
    Nt0 = 72
    Nev=100
    id=np.arange(872)*50+4050
    Eexp=np.array([1.1,1.2 ])
    idALL=np.arange(872)*50+4050
    MAX=[1.12,1.2]
    MIN=[1.04,1.12]
    beta=[16,15 ]  
    start1=[7,8 ]  
    start2=[6,6 ]
    bin=    None
    if deltat==  1:
        start=start1
        start3=start1
    else: 
        start=start2
        start3=[11,12,14]
    Nsamples = 872
    Nmom=2

if state=="C32P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L32x64"
    svdcuts=np.array([1e-12,1e-12,1e-12,1e-12])
    id=np.arange(1000)*50+11000
    Nt = 64
    Nt0 = 64
    Nev=100
    Eexp=np.array([1.2,1.35,1.5 ])
    idALL=np.arange(1000)*50+11000
    MAX=[1.12,1.2]
    MIN=[1.04,1.12]
    beta=[17,17,17 ]  
    start1=[9,9,10]  
    start2=[7,7,7 ]
    bin=    None
    if deltat==  1:
        start=start1
        start3=start1
    else: 
        start=start2
        start3=[13,12,14]
    Nsamples = 431
    Nmom=3

if state=="C48P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L48x96"
    svdcuts=np.array([1e-12,1e-12,1e-12,1e-12])
    id=np.arange(500)*10+500
    Nt = 96
    Nt0 = 96
    Nev=100
    MAX=[1.12,1.15]
    MIN=[1.05,1.075]
    Eexp=np.array([1.2,1.35,1.5 ])
    idALL=np.arange(300)*10+500
    beta=[16,14,15 ]  
    start1=[5,5,6]  
    start2=[4,4,4 ]
    bin=    None
    if deltat==  1:
        start=start1
        start3=start1
    else: 
        start=start2
        start3=[10,9,14]
    Nsamples = 120
    Nmom=3

if state=="C48P14":
    conf_name="beta6.20_mu-0.2825_ms-0.2310_L48x96"
    svdcuts=np.array([1e-12,1e-12,1e-12,1e-12])
    id=np.arange(120)*10+500
    Nt = 96
    Nt0 = 96
    Nev=100
    Eexp=np.array([0.9,0.94,0.98 ])
    idALL=np.arange(200)*10+500
    beta=[14,14,17]  
    start1=[7,7,9]  
    start2=[5,5,6]
    bin=    None
    if deltat==  1:
        start=start1
    else: 
        start=start2
    Nsamples = 120
    Nmom=1

if bin!=None:
    Nsamples =Nsamples //bin
    print(Nsamples)


if state=="C48P14":
    ymax=1.4
    ymin=0.7
t00=2

def energy(m,L,n):
    return np.sqrt(m**2+(np.pi*2/L)**2*n)

M_1=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/C24P29/param_E/Nucleon_mom0_6_22_n872_nboot3000_Nev100_E.npy")[0]
M_2=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/C32P29/param_E/Nucleon_mom0_13_27_n982_nboot3000_Nev100_E.npy")[0]
M_3=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/C48P29/param_E/Nucleon_mom0_11_28_n281_nboot3000_Nev200_E.npy")[0]

if state=="C48P29":
    Eonemean=2*np.array([np.mean(M_3),np.mean(energy(M_3,48,1)),np.mean(energy(M_3,48,1))])
    Eoneerr=2*np.array([np.std(M_3),np.std(energy(M_3,48,1)),np.std(energy(M_3,48,1))])
elif state=="C32P29":
    Eonemean=2*np.array([np.mean(M_2),np.mean(energy(M_2,32,1)),np.mean(energy(M_2,32,1))])
    Eoneerr=2*np.array([np.std(M_2),np.std(energy(M_2,32,1)),np.std(energy(M_2,32,1))])
elif state=="C24P29":
    Eonemean=2*np.array([np.mean(M_1),np.mean(energy(M_1,24,1)),np.mean(energy(M_1,24,1))])
    Eoneerr=2*np.array([np.std(M_1),np.std(energy(M_1,24,1)),np.std(energy(M_1,24,1))]) 

st0 = time.time()
savefile = f"/public/home/gengyq/laph/{object}/pictureNP/{conf_name}"


if Ptest!=None:
    Pall=[Ptest]
else:
    Pall=np.arange(Nmom)
for p in Pall:
    ymax=MAX[p]
    ymin=MIN[p]
    if TEST and deltat!=1:
        deltat=int(beta[p]-start2[p]-2)
    print(F"momentum {p} fit job starts!!!!")
    st0 = time.time()
    savefile = f"/public/home/gengyq/laph/{object}/picture/{conf_name}"
    if GEVP==True:
        prefix=f"/public/home/gengyq/laph/NP/fitwork/GEVPcorr/{state}/{object}/Nev{Nev}/eigvals_{p}_t0_{t00}"
        print(prefix)
        if Dwave:
            prefix=f"/public/home/gengyq/laph/NP/fitwork/GEVPcorr/{state}/T1SDmix/eigvals_{p}_t0_{t00}"
        print("GEVP!!")
        if bin!=None:
            prefix=f"/public/home/gengyq/laph/{object}/GEVP_result/{conf_name}/sample{Nsamples}bin{bin}/sort_eigval/eigvals_{p}_t0_{t00}"
            print(f"GEVP bin={bin}")
        subfix=".dat"
        corr_2pt= read_dat_file(prefix,subfix,GEVP)
    # else:
    #     corr_2pt=np.array(contract_all[p])

    
    print("Nused=",corr_2pt.shape)

    if GEVP==True:  
        contract=corr_2pt
    else:
        t00=0
        if bin!=None:
            print(f"bin={bin}")
            Nsamples=corr_2pt.shape[0]//bin*bin
            print("Nsamples=",Nsamples)
            corr_2pt=corr_2pt[:Nsamples]
            contract = (corr_2pt[1::2] + corr_2pt[0::2])/2
            #np.save(f"/public/home/gengyq/laph/{object}/corr_GEVP/{conf_name}/bincorr/corr_{object}_p_p1_bin{bin}",contract)
        else:
            Nsamples=corr_2pt.shape[0]
            print("Nsamples=",Nsamples)
            contract=corr_2pt
        if jack==True:
            contract = jackknife(contract)
            print("jackknife!!")
        else:
            contract = bootstrap(contract,nbsamples)
            print("bootstrap!!")

    ed0=time.time()
    print(f"data load done, time used : %.3f s"%(ed0-st0))


    Nt = 25
    contract = contract[:, :Nt]
    contract = np.array(contract)
    print(contract.shape)
    #print(contract)
    N = contract.shape[0]  # your final number of samples
    print("usedN=", N)

    contract_sum = np.sum(contract, 0)
    sum_average = np.mean(contract, 0)


    #meson_mass_average = np.arccosh((np.roll(sum_average, -1) + np.roll(sum_average, 1)) / (2 * sum_average)) #/ Na* 0.1974
    #meson_mass_jk =np.arccosh((np.roll(contract, -1, 1) + np.roll(contract, 1, 1)) / (2 * contract)) #/ Na* 0.1974
    meson_mass_average = np.log(np.roll(sum_average, 1) / (sum_average)) #/ Na* 0.1974
    meson_mass_jk =np.log(np.roll(contract, 1, 1) / (contract))    #/ Na* 0.1974
    

    jkerros = np.std(meson_mass_jk, 0)
    if jack==True:
        jkerros = jkerros*np.sqrt(Nsamples-1)
    #print(meson_mass_average)
    #print(jkerros)
    # ------------------------fit
       
    M_ij = contract - sum_average
    M_ij = np.matmul(M_ij.T, M_ij) / N
    if jack==True:
        M_ij=M_ij*(Nsamples-1)
    #print(M_ij)
    M_dig = np.tril(M_ij)
    M_dig = np.triu(M_dig)


    def efunc(x, p):
        if shape == "onestate":
            return p["a0"] * np.exp(-p["E0"] * (x-t00))
        else:
            if GEVP:
                return p["a0"] * np.exp(-p["E0"] * (x-t00)) + (1-p["a0"]) * np.exp(-(p["E0"]+p["dE"]**2)* (x -t00))
            else:
                return p["a0"] * np.exp(-p["E0"] * (x)) + (p["a1"]) * np.exp(-(p["E0"]+p["dE"]**2)* (x))

    if TEST==True or SAVEchijpg==True:
        N=150
    pr_arr_corr = np.zeros([N, 2])
    T = np.arange(Nt)
    chi_square = np.zeros([1, deltat])
    mass_mean = np.zeros([1, deltat])
    error_mean = np.zeros([1, deltat])
    save_E = np.zeros([deltat,N])
    save_chi = np.zeros([deltat,N])
    save_dof = np.zeros([deltat])
    save_a0 = np.zeros([deltat,N])
    save_dE = np.zeros([deltat,N])
     
    if method == "p0":
        print("fit with p0 method")
        p0 = {}
        p0["a0"] = 0.3
        p0["E0"] = Eexp[p]
        if shape=="twostate":
            p0["dE"] = 0.5
            if GEVP!=True:
                p0["a1"] = 0.5
        print(p0)
    else:
        print("fit with prior method")
        prior = {}
        prior["a0"] = gv.gvar(0.3, 1e10)
        prior["E0"] = gv.gvar(Eexp[p], 2)
        if shape=="twostate":
            prior["dE"] = gv.gvar(0.5, 1e10)
            if GEVP!=True:
                prior["a1"] = 0.5
        print(prior)
        
    for t2 in range(1):  # right side
        for t1 in range(start[p], start[p] + deltat):  # left side
            st1=time.time()
            print("t1=", t1, "t2=", t2 + beta[p])
            gv_corr = gv.gvar(np.mean(contract,0), M_ij)
            if method == "p0":
                fit_mean = lsqfit.nonlinear_fit(data=(T[t1 : t2 + beta[p] + 1], gv_corr[t1 : t2 + beta[p] + 1]),fcn=efunc,svdcut=1e-12,p0=p0)
            else:
                fit_mean = lsqfit.nonlinear_fit(data=(T[t1 : t2 + beta[p] + 1], gv_corr[t1 : t2 + beta[p] + 1]),fcn=efunc, svdcut=1e-12,prior=prior)
            chi_mean = fit_mean.chi2/fit_mean.dof
            param_mean = fit_mean.p
            
            for fid in range(N):
                gv_corr = gv.gvar(contract[fid], M_ij)
                
                if method == "p0":
                    fit = lsqfit.nonlinear_fit(data=(T[t1 : t2 + beta[p] + 1], gv_corr[t1 : t2 + beta[p] + 1]),fcn=efunc,svdcut=1e-12,p0=p0)
                else:
                    fit = lsqfit.nonlinear_fit(data=(T[t1 : t2 + beta[p] + 1], gv_corr[t1 : t2 + beta[p] + 1]),fcn=efunc, svdcut=1e-12,prior=prior,)
                    
                pr_corr = fit.p
                #print(pr_corr["E0"].mean)
                pr_arr_corr[fid][0] = pr_corr["E0"].mean
                pr_arr_corr[fid][1] = pr_corr["a0"].mean
                
                #print("id=",fid, "E=",fit.p["E0"].mean, "chis = ", fit.chi2/ (fit.dof+1))
                save_E[t1 - start[p],fid] = pr_corr["E0"].mean
                save_a0[t1 - start[p],fid] = pr_corr["a0"].mean 
                if shape=="twostate":
                    save_dE[t1 - start[p],fid] = pr_corr["dE"].mean 
                save_chi[t1 - start[p],fid] = fit.chi2
            save_dof[t1 - start[p]] = fit.dof

            pr_aver_corr = np.mean(pr_arr_corr, 0)
            
            print(param_mean)
            print("dof",fit.dof,"chis = ", chi_mean)  #-2
            

            mass_mean[t2][t1 - start[p]] = pr_aver_corr[0]
            chi_square[t2][t1 - start[p]] = chi_mean

            error_mean[t2][t1 - start[p]] = np.std(pr_arr_corr, 0)[0]
            if jack==True:
                error_mean[t2][t1 - start[p]] = error_mean[t2][t1 - start[p]] *np.sqrt(Nsamples-1)
            ed1=time.time()
            print(f"{t1} to {beta[p]} fit done, time used : %.3f s"%(ed1-st1))
            
            ##-----------------------pictureNP
            end = beta[p]
            E_mean = mass_mean[0][t1 - start[p]] #/ Na * 0.1974
            E_error = error_mean[0][t1 - start[p]] #/ Na * 0.1974

            # Nt=int(Nt/2+1)
            T = np.arange(Nt)

            fig = plt.figure()
            # plt.plot([t1,t1],[0,10],color='black',alpha=0.3)
            
            plt.errorbar( x=T, y=meson_mass_average, yerr=jkerros, ecolor="cornflowerblue", linestyle="none",\
                mec="cornflowerblue",marker="o", alpha=0.7,markerfacecolor="none",capsize=2,capthick=1,label="Meff")
            plt.legend(loc='upper right')
            # left, bottom, width, height = (t1, (E_mean - E_error), end - t1, 2 * E_error)
            # rect = mpatches.Rectangle((left, bottom),width,height,# fill=False,\
            #     alpha=0.4,facecolor="red",label="fit")
            # plt.gca().add_patch(rect)

            # plt.text(6, (ymax-ymin)*0.9+ymin, 'fit $E$ = %.4f'%E_mean, fontsize=9)
            # plt.text(6, (ymax-ymin)*0.85 +ymin, 'error = %.4f'%E_error, fontsize=9) 
            # plt.text(6, (ymax-ymin)*0.8+ymin, '$\chi^2/d.o.f.$ = %.2f'%chi_mean, fontsize=9)                               
            #####  band  ####

            def plotfit(t,E,a,dE):
                if shape=="twostate":
                    return a * np.exp(-E * (t-t00)) + (1-a) * np.exp(-(E+dE**2)* (t-t00))
                else:
                    return a * np.exp(-E * (t-t00))
                    
            plotx=np.arange(5-0.495,21+0.01-0.495,0.01)
            plotdata=[]
            for i in plotx:
                plotdata.append(plotfit(i,save_E[t1-start[p]],save_a0[t1-start[p]],save_dE[t1-start[p]]))
            plotdata=np.array(plotdata)
            plotdata=100*np.log(np.roll(plotdata,1,0)/plotdata)
            plt.fill_between(plotx[1:]+0.495,np.mean(plotdata,1)[1:]+np.std(plotdata,1)[1:],np.mean(plotdata,1)[1:]-np.std(plotdata,1)[1:],alpha=0.2,facecolor="red")

            plotx=np.arange(t1-0.495,beta[p]+0.01-0.495,0.01)
            plotdata=[]
            for i in plotx:
                plotdata.append(plotfit(i,save_E[t1-start[p]],save_a0[t1-start[p]],save_dE[t1-start[p]]))
            plotdata=np.array(plotdata)
            plotdata=100*np.log(np.roll(plotdata,1,0)/plotdata)
            plt.fill_between(plotx[1:]+0.495,np.mean(plotdata,1)[1:]+np.std(plotdata,1)[1:],np.mean(plotdata,1)[1:]-np.std(plotdata,1)[1:],alpha=0.4,facecolor="red",label="fit")

            #################
   
            thre_up= Eonemean[p]+Eoneerr[p]
            thre_down= Eonemean[p]-Eoneerr[p]
            plt.fill_between([0,35],[thre_up,thre_up],[thre_down,thre_down],color="orange",alpha=0.2,label="free")
 
            # plt.text(t1-0.1, pion_mass_average[t1-start[p]], 't1', ha='center', va='bottom', fontsize=9)
            plt.legend(loc='upper right')
            if GEVP==True and Dwave==True:
                plt.title(r"%s %s $\lambda$=%s GEVP S-Dmix" % (state, irrep,p))
            elif GEVP==True and Dwave!=False:
                plt.title(r"%s %s $\lambda$=%s GEVP" % (state, irrep,p))
            else:
                plt.title(r"%s %s $\lambda$=%s" % (state, irrep,p))
            plt.xlabel("t")
            plt.ylabel("aE")
            plt.xlim(0, 25)
            plt.ylim(ymin, ymax)
            # plt.yticks([0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6])
            # plt.xticks([5,10,15,20,25])
            if SAVEdataandgraph==True and deltat==1:   
                if Dwave:
                    plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/2effmass/{object}_effmass_mom{p}_Dwave.pdf", format='pdf',dpi=400) 
                else:
                    plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/2effmass/{object}_effmass_mom{p}.pdf", format='pdf',dpi=400)        
            if TEST==True:
                plt.savefig(f"/public/home/gengyq/test.jpg",dpi=400) #,dpi=200 
        
        #####################################################  pretty table
        print("mass=", mass_mean)
        print("error=", error_mean)
        twostatefit = PrettyTable(
            ["start timeslice(end=" + str(beta[p]) + ")", "chi2/dof", "mass mean", "fit error"]
        )
        for i in range(deltat):
            twostatefit.add_row(
                [
                    np.arange(deltat)[i] + start[p],
                    chi_square[0][i],
                    mass_mean[0][i],
                    error_mean[0][i],
                ]
            )
        print(twostatefit)
        
        #####################################################  chi plot
        fig, (ax1, ax2) = plt.subplots(2, 1)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        selectid=start1[p]-start[p]
        selectid2=start3[p]-start[p]+1
        
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        #mass_mean = mass_mean #/ Na * 0.1974
        #error_mean = error_mean #/ Na * 0.1974
        ax1.errorbar(x=np.arange(deltat) + start[p],y=mass_mean[0],yerr=error_mean[0],ecolor="cornflowerblue",linestyle="none",\
            mec="cornflowerblue",marker="o", alpha=0.7,markerfacecolor="none",capsize=2,capthick=1,)
        ax1.errorbar(x=np.arange(selectid,selectid2)+start[p],y=mass_mean[0][selectid:selectid2],\
                     yerr=error_mean[0][selectid:selectid2],ecolor="brown",linestyle="none",\
            mec="brown",marker="o",markerfacecolor="none",capsize=2,capthick=1,)
        plotx=np.arange(deltat) + start[p]        

        from scipy.stats import chi2
        def weight(chi,dof,dEfit):
            p_value = chi2.sf(chi, dof)
            value=((1-2*abs(p_value-0.5))*np.min(dEfit)/dEfit)**2
            return value
        
        w=weight(save_chi[selectid:selectid2,0],save_dof[selectid:selectid2],np.std(save_E[selectid:selectid2],1))
        w=w/np.sum(w)
        print("weight",w)
        wa=weight(save_chi[selectid:selectid2,0],save_dof[selectid:selectid2],np.std(save_a0[selectid:selectid2],1))
        wa=wa/np.sum(wa)
        wde=weight(save_chi[selectid:selectid2,0],save_dof[selectid:selectid2],np.std(save_dE[selectid:selectid2],1))
        wde=wde/np.sum(wde)
        weightE=np.einsum("a,ab->b",w,save_E[selectid:selectid2])
        weightdE=np.einsum("a,ab->b",wde,save_dE[selectid:selectid2])
        weightA=np.einsum("a,ab->b",wa,save_a0[selectid:selectid2])
        weightmean=np.mean(weightE)
        print(weightmean)
        weightstd=np.std(weightE)
        print(weightstd)
        ax1.fill_between([plotx[0]-0.5,plotx[-1]+0.5],[weightmean+weightstd,weightmean+weightstd],\
            [weightmean-weightstd,weightmean-weightstd],alpha=0.2,color="red")

        ax1.fill_between([0,35],[thre_up,thre_up],[thre_down,thre_down],color="orange",alpha=0.2 )
        ax2.scatter(x=plotx,y=chi_square[0],color="cornflowerblue")
        ax2.scatter(x=np.arange(selectid,selectid2)+start[p],y=chi_square[0,selectid:selectid2],color="brown")
        # ax2.text(selectid+start[p],chi_square[0,selectid]+0.25,f"{chi_square[0,selectid]:0.2f}",ha='center')
        # ax1.text(selectid+start[p],mass_mean[0,selectid]+error_mean[0,selectid]*1.2,\
        #          f"{mass_mean[0,selectid]:0.4f}"+"("+"%0.0f"%(error_mean[0,selectid]*10000)+")",ha='center')
        ax2.plot(plotx,np.ones_like(plotx),color="grey",alpha=0.7,linestyle="--")
        auto_ymin=weightmean-weightstd*8
        auto_ymax=weightmean+weightstd*8
        ax2.set_xlabel("start t")
        ax1.set_ylabel("$fit$ $M_{eff}$")
        ax2.set_ylabel("$\chi^2/d.o.f.$")
        ax1.set_ylim(auto_ymin, auto_ymax)
        ax2.set_ylim(0, 3)  
        ax1.set_xlim(plotx[0]-0.5,plotx[-1]+0.5)
        ax2.set_xlim(plotx[0]-0.5,plotx[-1]+0.5)  
        #ax1.legend(loc=1)
        #ax2.legend(loc=1)
        Xset=[]
        for i in plotx:
            Xset.append(f"{i}")
        ax2.set_xticks(plotx,Xset)
        Yset=[]
        for i in np.arange(0,3,0.5):
            Yset.append(f"{i}")
        ax2.set_yticks(np.arange(0,3,0.5),Yset)
        plt.subplots_adjust(hspace=0)
        if Dwave:
            ax1.set_title(r"%s %s $\lambda$=%s (end t=%s)" % (state,irrep, p, beta[p]))
        else:
            ax1.set_title(r"%s %s $\lambda$=%s (end t=%s)" % (state,irrep, p, beta[p]))
        if SAVEchijpg==True and deltat!=1:
            if Dwave:
                plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/2chi/{object}_fitchi_mom{p}_Dwave.pdf", format='pdf',dpi=400)
            else:
                plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/2chi/{object}_fitchi_mom{p}.pdf", format='pdf',dpi=400)
        if TEST==True:          
            plt.savefig("/public/home/gengyq/test1.jpg",dpi=400) 
        ###############################
        fig = plt.figure()
        # plt.plot([t1,t1],[0,10],color='black',alpha=0.3)
        
        plt.errorbar( x=T, y=meson_mass_average, yerr=jkerros, ecolor="cornflowerblue", linestyle="none",\
            mec="cornflowerblue",marker="o", alpha=0.7,markerfacecolor="none",capsize=2,capthick=1,label="Meff")
        plt.legend(loc='upper right')
        # left, bottom, width, height = (start1[p], (weightmean - weightstd), end - start1[p], 2 * weightstd)
        # rect = mpatches.Rectangle((left, bottom),width,height,# fill=False,\
        #     alpha=0.4,facecolor="red",label="fit")
        # plt.gca().add_patch(rect)          

        def plotfit(t,E,a,dE):
            if shape=="twostate":
                return a * np.exp(-E * (t-t00)) + (1-a) * np.exp(-(E+dE**2)* (t-t00))
            else:
                return a * np.exp(-E * (t-t00))
                    
        plotx=np.arange(5-0.495,21+0.01-0.495,0.01)
        plotdata=[]
        for i in plotx:
            plotdata.append(plotfit(i,weightE,weightA,weightdE))
        plotdata=np.array(plotdata)
        plotdata=100*np.log(np.roll(plotdata,1,0)/plotdata)
        plt.fill_between(plotx[1:]+0.495,np.mean(plotdata,1)[1:]+np.std(plotdata,1)[1:],np.mean(plotdata,1)[1:]-np.std(plotdata,1)[1:],alpha=0.2,facecolor="red")

        plotx=np.arange(start1[p]-0.495,beta[p]+0.01-0.495,0.01)
        plotdata=[]
        for i in plotx:
            plotdata.append(plotfit(i,weightE,weightA,weightdE))
        plotdata=np.array(plotdata)
        plotdata=100*np.log(np.roll(plotdata,1,0)/plotdata)
        plt.fill_between(plotx[1:]+0.495,np.mean(plotdata,1)[1:]+np.std(plotdata,1)[1:],np.mean(plotdata,1)[1:]-np.std(plotdata,1)[1:],alpha=0.4,facecolor="red",label="fit")                 

        thre_up= Eonemean[p]+Eoneerr[p]
        thre_down= Eonemean[p]-Eoneerr[p]
        plt.fill_between([0,35],[thre_up,thre_up],[thre_down,thre_down],color="orange",alpha=0.2,label="free")

        plt.legend(loc='upper right')
        if GEVP==True and Dwave==True:
            plt.title(r"%s %s $\lambda$=%s GEVP S-Dmix" % (state, irrep,p))
        elif GEVP==True and Dwave!=False:
            plt.title(r"%s %s $\lambda$=%s GEVP" % (state, irrep,p))
        else:
            plt.title(r"%s %s $\lambda$=%s" % (state, irrep,p))
        plt.xlabel("t")
        plt.ylabel("aE")
        plt.xlim(0, 25)
        plt.ylim(ymin, ymax)

        if SAVEdataandgraph==True:   
            if Dwave:
                plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/2effmass/{object}_effmass_mom{p}_Dwave_weight.pdf", format='pdf',dpi=400) 
            else:
                plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/2effmass/{object}_effmass_mom{p}_weight.pdf", format='pdf',dpi=400)        
        if TEST==True:
            plt.savefig(f"/public/home/gengyq/test.jpg",dpi=400) #,dpi=200         

        ##############################        
     
            
    ed00=time.time()
    print(f"all p={p} job done, time used : %.3f s"%(ed00-st0))
    print("")
    print("")
    if SAVEdataandgraph==True:    
        if Dwave:
            np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_E/eigen{p}_SDmix.npy",save_E)
        else:
            np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_E/eigen{p}.npy",save_E)
        # np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_a/{object}_mom{p}_{start[p]}_{beta[p]}_{deltat}_n{Nsamples}_{method}_nb{nbsamples}_GEVP{GEVP}_bin{bin}_a.npy",save_a0)           
        # np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/chi/{object}_mom{p}_{start[p]}_{beta[p]}_{deltat}_n{Nsamples}_{method}_nb{nbsamples}_bin{bin}_GEVP{GEVP}_chi.npy",save_chi)
        # np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_dE/{object}_mom{p}_{start[p]}_{beta[p]}_{deltat}_n{Nsamples}_{method}_nb{nbsamples}_GEVP{GEVP}_bin{bin}_dE.npy",save_dE)
       
