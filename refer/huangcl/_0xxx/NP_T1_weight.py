#!/public/home/xinghy/anaconda3-2023.03/bin/python

# test file to check whether first term has problem

import numpy as np
import fileinput
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
Dwave = False
object = "NN"

if object=="NP":
    irrep=r"$T_1^+$"
    channel="T1"
else:
    irrep=r"$A_1^+$"
    channel="A1"

N = 3000
nbsamples = 3000
px = 0
py = 0
pz = 0
Na = 0.077
state = "C48P29"
GEVP=True#False#

jack=False
SAVEdataandgraph=True#False#
SAVEchijpg=True#False#
effbandinchi=True#False#
TEST=True#False#
Ptest= 0#None#
deltat=  6
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
    Nev=100
    Nt = 72
    Nt0 = 72
    ymax=1.75
    ymin=1.65
    id=np.arange(872)*50+4050
    Eexp=np.array([1,0.07 ])
    beta=[16,15 ]  
    start=[10,7,7]
    Nsamples = 872
    Nmom=2

if state=="C32P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L32x64"
    svdcuts=np.array([1e-12,1e-12,1e-12,1e-12])
    id=np.arange(1000)*50+11000
    Nev=100
    Nt = 64
    Nt0 = 64
    ymax=1.2
    ymin=1
    Eexp=np.array([1.064,1.14,1.15 ])
    beta=[19,17,14 ]  
    start=[9,9,7]
    Nsamples = 982
    Nmom=2

if state=="C48P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L48x96"
    svdcuts=np.array([1e-12,1e-12,1e-12,1e-12])
    id=np.arange(500)*10+500
    Nt = 96
    Nt0 = 96
    ymax=1.2
    ymin=1
    Nev=100
    Eexp=np.array([1.07,1.07,1.7 ])
    beta=[15,14,14 ]  
    start=[7,7,7]
    Nsamples = 450
    Nmom=1

# if bin!=None:
#     Nsamples =Nsamples //bin
#     print(Nsamples)

t00=2



def energy(m,L,n):
    return np.sqrt(m**2+(np.pi*2/L)**2*n)


st0 = time.time()
savefile = f"/public/home/gengyq/laph/{object}/pictureNP/{conf_name}"

if Ptest!=None:
    Pall=[Ptest]
else:
    Pall=np.arange(Nmom)


for p in Pall:


    print(F"momentum {p} fit job starts!!!!")
    st0 = time.time()
    savefile = f"/public/home/gengyq/laph/{object}/picture/{conf_name}"

    # prefix=f"/public/home/gengyq/laph/NP/fitwork/GEVPcorr/{state}/{object}/ratio/eigvals_{p}_t0_{t00}"
    prefix=f"/public/home/gengyq/laph/NP/fitwork/GEVPcorr/{state}/{object}/Nev100/eigvals_{p}_t0_{t00}"
    print("GEVP!!")
    subfix=".dat"
    corr_2pt= read_dat_file(prefix,subfix,GEVP)
    print("Nused=",corr_2pt.shape)
    contract=corr_2pt

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
    # ------------------------fitting
       
    M_ij = contract - sum_average
    M_ij = np.matmul(M_ij.T, M_ij) / N
    if jack==True:
        M_ij=M_ij*(Nsamples-1)
    #print(M_ij)
    M_dig = np.tril(M_ij)
    M_dig = np.triu(M_dig)


    def efunc(x, p):
        if shape == "onestate":
            return p["a0"]*10**2 * np.exp(-p["E0"] * (x))
        else:
            if GEVP:
                return p["a0"] * np.exp(-p["E0"] * (x-t00)) + (1-p["a0"]) * np.exp(-(p["E0"]+p["dE"]**2)* (x-t00 ))
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
    #save_dE = np.zeros([deltat,N])
     
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
            save_dof[t1 - start[p]] = fit_mean.dof
            
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
                #save_dE[t1 - start[p],fid] = pr_corr["dE"].mean 
                save_chi[t1 - start[p],fid] = fit.chi2 

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
                mec="cornflowerblue",marker="o", alpha=0.7,markerfacecolor="none",capsize=2,capthick=1)
            #plt.legend(loc='upper right')
            left, bottom, width, height = (t1, (E_mean - E_error), end - t1, 2 * E_error)
            rect = mpatches.Rectangle((left, bottom),width,height,# fill=False,\
                alpha=0.4,facecolor="red")
            plt.gca().add_patch(rect)
            # if E_error*10000//1<10:
            #     plt.text(1, (ymax-ymin)*0.15+ymin, f'$E$ = %.5f'%(E_mean), fontsize=9)
            #     plt.text(1, (ymax-ymin)*0.1 +ymin, f'error = %.5f'%(E_error), fontsize=9) 
            # else:
            #     plt.text(1, (ymax-ymin)*0.15+ymin, f'$E$ = %.4f'%(E_mean), fontsize=9)
            #     plt.text(1, (ymax-ymin)*0.1 +ymin, f'error = %.4f'%(E_error), fontsize=9) 
            #plt.text(1, (ymax-ymin)*0.05+ymin, '$\chi^2/d.o.f.$ = %.2f'%(chi_mean), fontsize=9) 

            #plt.fill_between([0,35],[deltaE[p]+Eoneerr[p],deltaE[p]+Eoneerr[p]],[deltaE[p]-Eoneerr[p],deltaE[p]-Eoneerr[p]],color="orange",alpha=0.2,label="free")
 
            # plt.text(t1-0.1, pion_mass_average[t1-start[p]], 't1', ha='cender', va='bottom', fontsize=9)
            #plt.legend(loc='upper right')
            # if GEVP==True and Dwave==True:
            #     plt.title(r"%s %s $\lambda$=%s GEVP S-Dmix" % (state, irrep,p))
            # elif GEVP==True and Dwave!=False:
            #     plt.title(r"%s %s $\lambda$=%s GEVP" % (state, irrep,p))
            # else:
            #     plt.title(r"%s %s $\lambda$=%s" % (state, irrep,p))
            plt.text(3,(ymax-ymin)*0.05+ymin,r"$^3S_1$ $\mathbf{p}^2=$%s"%p,fontsize=15)
            plt.xlabel(r"$t/a$")
            plt.ylabel(r"$aE$")
            plt.xlim(0, 20)
            plt.ylim(ymin, ymax)
            # plt.yticks([0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6])
            # plt.xticks([5,10,15,20,25])
            if SAVEdataandgraph==True and deltat ==1 :    
                plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/2effmass/{object}_effmass_mom{p}.pdf", format='pdf',dpi=250)        
                #plt.savefig(f"{savefile}/{shape}/corr_{object}_P{p}_t{t1}_{start[p]}_{beta[p]}_{deltat}_n{Nsamples}_nb{nbsamples}_GEVP{GEVP}_bin{bin}.jpg",dpi=400) #,dpi=200
            if TEST==True:
                plt.savefig(f"/public/home/gengyq/test.jpg",dpi=400) #,dpi=200 
        ### --------------pretty table
        print("mass=", mass_mean)
        print("error=", error_mean)
        twostatefit = PrettyTable(
            ["start timeslice(end=" + str(beta[p]) + ")", "chi2/dof", "mass mean", "fitting error"]
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
        
        
        fig, (ax1, ax2) = plt.subplots(2, 1)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        #mass_mean = mass_mean #/ Na * 0.1974
        #error_mean = error_mean #/ Na * 0.1974
        ax1.errorbar(x=np.arange(deltat) + start[p],y=mass_mean[0],yerr=error_mean[0],ecolor="red",linestyle="none",\
            mec="red",marker="o", alpha=0.7,markerfacecolor="none",capsize=2,capthick=1,)
        plotx=np.arange(deltat) + start[p]        
        if effbandinchi==True:
            from scipy.stats import chi2
            def calweight(chi,dof,dEfit):
                p_value = chi2.sf(chi, dof)
                #value=((1-2*abs(p_value-0.5))/dEfit)**2
                value=p_value/dEfit**2
                return value
            def calweight1(chi,dof,dEfit):
                p_value = chi2.sf(chi, dof)
                value=((1-2*abs(p_value-0.5))/dEfit)**2
                return value
            weight=calweight(chi_square[0]*save_dof,save_dof,np.std(save_E,1))
            weight1=calweight1(chi_square[0]*save_dof,save_dof,np.std(save_E,1))
            weight=weight/np.sum(weight)
            weight1=weight1/np.sum(weight1)
            print(weight,weight1)
            #weight=weight1
            id=np.argmax(weight)
            errsta1=np.std(save_E[id])
            errsta=np.sqrt(np.sum(weight*np.std(save_E,1)**2))
            print(errsta1,errsta)
            test=np.einsum("a,ab->b",weight,save_E)

            mean=np.sum(np.mean(save_E,1)*weight)
            print(weight)
            err=np.sqrt(np.sum(weight*(np.mean(save_E,1)-mean)**2))
            
            errall=np.sqrt(errsta**2+err**2)
            print(mean,errsta,err,errall,np.std(test))
            ax1.fill_between([plotx[0]-0.5,plotx[-1]+0.5],[mean+errall,mean+errall],\
                [mean-errall,mean-errall],alpha=0.2,color="red")
            ax1.fill_between([plotx[0]-0.5,plotx[-1]+0.5],[mean+errsta,mean+errsta],\
                [mean-errsta,mean-errsta],alpha=0.3,color="red")

            # left, bottom, width, height = (plotx[0], (mean - err),deltat-1, 2 * err)
            # rect = mpatches.Rectangle((left, bottom),width,height,# fill=False,\
            #     alpha=0.2,facecolor="red")
            # plt.gca().add_patch(rect)
        
        ax2.scatter(x=plotx,y=chi_square[0],color="cornflowerblue")
        # ax2.text(selectid+start[p],chi_square[0,selectid]+0.25,f"{chi_square[0,selectid]:0.2f}",ha='center')
        # ax1.text(selectid+start[p],mass_mean[0,selectid]+error_mean[0,selectid]*1.2,\
        #          f"{mass_mean[0,selectid]:0.4f}"+"("+"%0.0f"%(error_mean[0,selectid]*10000)+")",ha='center')
        ax2.plot([plotx[0]-0.5,plotx[-1]+0.5],np.ones(2),color="grey",alpha=0.7,linestyle="--")
        auto_ymin=mean-errall*8
        auto_ymax=mean+errall*8
        
        formatter = ticker.StrMethodFormatter("{x:.3f}")
        ax1.yaxis.set_major_formatter(formatter)
        ax2.set_xlabel(r"$t_{min}$")
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
        ax1.set_title(r"%s %s $\lambda$=%s ($t_{max}$=%s)" % (state,irrep, p, beta[p]))
        if SAVEchijpg==True and deltat != 1 :
            plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/2chi/{object}_fitchi_mom{p}.pdf", format='pdf',dpi=400)
            #plt.savefig(f"/public/home/gengyq/laph/{object}/pictureNP/{conf_name}/{object}_chi_Meff_P{p}_{start[p]}_{beta[p]}_n{Nsamples}_nboot{nbsamples}_GEVP{GEVP}_bin{bin}.jpg",dpi=400)
        if TEST==True:          
            plt.savefig("/public/home/gengyq/test1.jpg",dpi=400)  


#################################
    fig = plt.figure()
            
    plt.errorbar( x=T, y=meson_mass_average, yerr=jkerros, ecolor="cornflowerblue", linestyle="none",\
        mec="cornflowerblue",marker="o", alpha=0.7,markerfacecolor="none",capsize=2,capthick=1)
    #plt.legend(loc='upper right')
    left, bottom, width, height = (start[p], (mean - errall), beta[p] - start[p], 2 * errall)
    rect = mpatches.Rectangle((left, bottom),width,height,# fill=False,\
        alpha=0.2,facecolor="red")
    plt.gca().add_patch(rect)
    left, bottom, width, height = (start[p], (mean - errsta), beta[p] - start[p], 2 * errsta)
    rect = mpatches.Rectangle((left, bottom),width,height,# fill=False,\
        alpha=0.3,facecolor="red")
    plt.gca().add_patch(rect)

    #plt.legend(loc='upper right')
    if object=="NP":
        plt.text(7,(ymax-ymin)*0.1+ymin,r"%s  $^3S_1$ ($\mathbf{p}^2=$%s)"%(state,p),fontsize=15)
    else:
        plt.text(7,(ymax-ymin)*0.1+ymin,r"%s  $^1S_0$ ($\mathbf{p}^2=$%s)"%(state,p),fontsize=15)
    plt.xlabel(r"$t/a$",fontsize=10)
    plt.ylabel(r"$aE$",fontsize=10)
    plt.xlim(5, 20)
    plt.xticks([5,10,15,20],[5,10,15,20])
    plt.ylim(ymin, ymax)

    plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/2effmass/{state}_{channel}_mom{p}.pdf", format='pdf',dpi=250)        
    if TEST==True:
        plt.savefig(f"/public/home/gengyq/test2.jpg",dpi=400) #,dpi=200 
            
    ed00=time.time()
    print(f"all p={p} job done, time used : %.3f s"%(ed00-st0))
    print("")
    print("")
    if SAVEdataandgraph==True:    
        # np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_E/eigen{p}.npy",save_E)
        np.savez(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_E/weighted_E.npz",mean=mean,stat=errsta,sys=err,weightEall=test)
        #np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_a/{object}_mom{p}_{start[p]}_{beta[p]}_{deltat}_n{Nsamples}_{method}_nb{nbsamples}_GEVP{GEVP}_bin{bin}_a.npy",save_a0)           
        #np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/chi/{object}_mom{p}_{start[p]}_{beta[p]}_{deltat}_n{Nsamples}_{method}_nb{nbsamples}_bin{bin}_GEVP{GEVP}_chi.npy",save_chi)
        #np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_dE/{object}_mom{p}_{start[p]}_{beta[p]}_{deltat}_n{Nsamples}_{method}_nb{nbsamples}_GEVP{GEVP}_bin{bin}_dE.npy",save_dE)
       


