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

shape = "onestate"
method = "p0"

object = "NN"

Nt = 96
Nt0 = 96
N = 3000
nbsamples = 3000
px = 0
py = 0
pz = 0
# Na = 0.105
state = "C24P29"
GEVP=True#False#

jack=False
SAVEdataandgraph=True#False#
SAVEchijpg=True#False#
effbandinchi=True#False#
TEST=True#False#
Ptest= 0#None#
deltat= 1 
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
    idALL=np.arange(872)*50+4050
    ymax=0.15
    ymin=-0.15
    Eexp=np.array([-0.05,1.2 ])
    beta=[16,14 ]  
    start1=[12,8 ]  
    start2=[7,7 ]
    bin=    None
    if deltat==  1:
        start=start1
    else: 
        start=start2
    Nsamples = 872
    Nmom=2
if state=="C32P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L32x64"
    svdcuts=np.array([1e-12,1e-12,1e-12,1e-12])
    idALL=np.arange(431)*50+11000
    ymax=0.15
    ymin=-0.15
    Eexp=np.array([0.04,0.07,1.2 ])
    beta=[19,17,13 ]  
    start1=[10,12,8 ]  
    start2=[6,6,5]
    bin=    None
    if deltat==  1:
        start=start1
    else: 
        start=start2
    Nsamples = 430
    Nmom=3

if bin!=None:
    Nsamples =Nsamples //bin
    print(Nsamples)

t00=2

if state=="C24P29":
    conf_name = "beta6.20_mu-0.2770_ms-0.2400_L24x72"
    Nxone=24
    Nmomone=5
    Nsamplesone=872
    betaone=np.array([22,22,21,19,15])
    startone=np.array([6,6,5,5,5])
    tfit2one = np.array([0,0,0,0,0])  # choose time - start time
    Nevone=100
if state=="C32P29":
    conf_name = "beta6.20_mu-0.2770_ms-0.2400_L32x64"
    Nxone=32
    Nmomone=5
    Nsamplesone=428
    betaone=np.array([22,25,21,21,19])
    startone=np.array([6,6,5,5,5])
    tfit2one = np.array([0,0,0,0,0])  # choose time - start time
    Nevone=100


Eoneall=np.zeros([5])
Eoneerrall=np.zeros([5])
for p in range(5):
    Eall=np.load(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_E/Nucleon_mom{p}_{startone[p]}_{betaone[p]}_n{Nsamplesone}_nboot3000_Nev{Nevone}_E.npy")[0]
    Eoneall[p]=np.mean(Eall)
    Eoneerrall[p]=np.std(Eall)
deltaE=np.array([0,Eoneall[1]-Eoneall[0],Eoneall[2]-Eoneall[0],Eoneall[3]-Eoneall[0]])*2
print(deltaE)

threshold=2*np.load(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_E/Nucleon_mom{0}_{startone[0]}_{betaone[0]}_n{Nsamplesone}_nboot3000_Nev{Nevone}_E.npy")[0]
threshold_corr=[]
momlist=["Px0Py0Pz0"] 
for p in momlist:
    if state=="C24P29":
        a=read_dat_file(F"/public/group/lqcd/corr/twopt/xinghy/{conf_name}/Nev100/corr_Nucleon_pp_{p}",".dat",True)
    elif state=="C32P29":
        a=read_dat_file(F"/public/group/lqcd/corr/twopt/xinghy/{conf_name}/Nev100/corr_Nucleon_pp_{p}",".dat",True)
    threshold_corr.append(a)

data_all_one=np.array(threshold_corr[0]) 
print(data_all_one.shape)
data_all_one=data_all_one/data_all_one[0,0]
data_all_one=bootstrap(data_all_one , nbsamples)

st0 = time.time()
savefile = f"/public/home/gengyq/laph/{object}/picture/{conf_name}"

if Ptest!=None:
    Pall=[Ptest]
else:
    Pall=np.arange(Nmom)

if not GEVP:
    datapath=f"/public/home/gengyq/NP/result/spin/{conf_name}/"
    data=[]
    contract_all=[]
    for id in idALL:
        try:
            if state=="C24P29":
                data.append(np.load(f"{datapath}/{object}corr_spin_{id}.npy"))
            elif state=="C32P29":
                data.append(np.load(f"{datapath}/{object}corr_spin_mom20_conf{id}.npy"))
        except:
            continue
    data=np.array(data)
    print("samples",data.shape[0])
    contract_all.append(data[:,0,0,:,0,0,1,1]+data[:,0,0,:,1,1,0,0]-data[:,0,0,:,0,1,1,0]-data[:,0,0,:,1,0,0,1])

    corr1=0
    for p1 in range(1,4):
        for p2 in range(1,4):
            corr1=corr1+(data[:,p1,p2,:,0,0,1,1]+data[:,p1,p2,:,1,1,0,0]-data[:,p1,p2,:,0,1,1,0]-data[:,p1,p2,:,1,0,0,1])
    contract_all.append(corr1)

    if state=="C32P29":
        corr2=0
        for p1 in range(4,10):
            for p2 in range(4,10):
                corr2=corr2+(data[:,p1,p2,:,0,0,1,1]+data[:,p1,p2,:,1,1,0,0]-data[:,p1,p2,:,0,1,1,0]-data[:,p1,p2,:,1,0,0,1])
        contract_all.append(corr2)

for p in Pall:
    if TEST and deltat!=1:
        deltat=int(beta[p]-start2[p]-1)
    print(F"momentum {p} fit job starts!!!!")
    st0 = time.time()
    savefile = f"/public/home/gengyq/laph/{object}/picture/{conf_name}"
    if GEVP==True:
        prefix=f"/public/home/gengyq/laph/NP/fitwork/GEVPcorr/{state}/{object}/sort/eigvals_{p}_t0_{t00}"
        print("GEVP!!")
        if bin!=None:
            prefix=f"/public/home/gengyq/laph/{object}/GEVP_result/{conf_name}/sample{Nsamples}bin{bin}/sort_eigval/eigvals_{p}_t0_{t00}"
            print(f"GEVP bin={bin}")
        subfix=".dat"
        corr_2pt= read_dat_file(prefix,subfix,GEVP)
    else:
        corr_2pt=np.array(contract_all[p])

    
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
    contract = np.array(contract)/data_all_one[:,:Nt]**2
    print(contract.shape)
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
    # save_chi = np.zeros([deltat,N])
    save_a0 = np.zeros([deltat,N])
    # save_dE = np.zeros([deltat,N])
     
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
                # save_dE[t1 - start[p],fid] = pr_corr["dE"].mean 
                # save_chi[t1 - start[p],fid] = fit.chi2/ (fit.dof) 

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
            
            ##-----------------------picture
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
            
            left, bottom, width, height = (t1, (E_mean - E_error), end - t1, 2 * E_error)
            rect = mpatches.Rectangle((left, bottom),width,height,# fill=False,\
                alpha=0.4,facecolor="red",label="fitting")
            plt.gca().add_patch(rect)
            # plt.fill_between([t1,end],y1=[E_mean - E_error,E_mean - E_error],y2=[E_mean + E_error,E_mean + E_error],color="red",alpha=0.4,label="fitting")
            
            if E_error*10000//1<10:
                plt.text(1, (ymax-ymin)*0.15+ymin+deltaE[p], f'$\Delta E$ = %.5f'%(E_mean), fontsize=9)
                plt.text(1, (ymax-ymin)*0.1 +ymin+deltaE[p], f'error = %.5f'%(E_error), fontsize=9) 
            else:
                plt.text(1, (ymax-ymin)*0.15+ymin+deltaE[p], f'$\Delta E$ = %.4f'%(E_mean), fontsize=9)
                plt.text(1, (ymax-ymin)*0.1 +ymin+deltaE[p], f'error = %.4f'%(E_error), fontsize=9) 
            plt.text(1, (ymax-ymin)*0.05+ymin+deltaE[p], '$\chi^2/d.o.f.$ = %.2f'%(chi_mean), fontsize=9) 

            plt.fill_between([0,35],[deltaE[p]+Eoneerrall[p],deltaE[p]+Eoneerrall[p]],[deltaE[p]-Eoneerrall[p],deltaE[p]-Eoneerrall[p]],color="orange",alpha=0.2,label="free")
 
            # plt.text(t1-0.1, pion_mass_average[t1-start[p]], 't1', ha='cender', va='bottom', fontsize=9)
            plt.legend(loc='upper right')
            if GEVP==True and bin!=None:
                plt.title(r"%s %s $[0,0,0]$ $\tilde{q}^2$=%s GEVP bin=%s" % (state, object,p,bin))
            elif GEVP==True and bin==None:
                plt.title(r"%s %s $[0,0,0]$ $\tilde{q}^2$=%s GEVP" % (state, object,p))
            elif GEVP!=True and bin!=None:
                plt.title(r"%s %s $[0,0,0]$ $\tilde{q}^2$=%s bin=%.f" % (state, object,p,bin))
            else:
                plt.title(r"%s %s $[0,0,0]$ $\tilde{q}^2$=%s" % (state, object,p))
            plt.xlabel("t")
            plt.ylabel(r"$\Delta$ aE")
            plt.xlim(0, 25)
            plt.ylim(ymin+deltaE[p], ymax+deltaE[p])
            # plt.yticks([0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6])
            # plt.xticks([5,10,15,20,25])
            if SAVEdataandgraph==True and deltat==1:    
                plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/2effmass/{object}_effmass_mom{p}_ratio.pdf", format='pdf',dpi=250)        
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
            selectid=start1[p]-start[p]
            mean=mass_mean[0,selectid]
            err=error_mean[0,selectid]
            ax1.fill_between([plotx[0]-0.5,plotx[-1]+0.5],[mean+err,mean+err],\
                [mean-err,mean-err],alpha=0.2,color="red")

            # left, bottom, width, height = (plotx[0], (mean - err),deltat-1, 2 * err)
            # rect = mpatches.Rectangle((left, bottom),width,height,# fill=False,\
            #     alpha=0.2,facecolor="red")
            # plt.gca().add_patch(rect)
        formatter = ticker.StrMethodFormatter("{x:.3f}")
        ax1.yaxis.set_major_formatter(formatter)
        ax2.scatter(x=plotx,y=chi_square[0],color="cornflowerblue")
        ax2.scatter(x=selectid+start[p],y=chi_square[0,selectid],color="brown")
        ax2.text(selectid+start[p],chi_square[0,selectid]+0.25,f"{chi_square[0,selectid]:0.2f}",ha='center')
        ax1.text(selectid+start[p],mass_mean[0,selectid]+error_mean[0,selectid]*1.2,\
                 f"{mass_mean[0,selectid]:0.4f}"+"("+"%0.0f"%(error_mean[0,selectid]*10000)+")",ha='center')
        ax2.plot(plotx,np.ones_like(plotx),color="grey",alpha=0.7,linestyle="--")
        if p==0:
            ax1.plot([0,25],[deltaE[p],deltaE[p]],color="black",linewidth=0.5)
        else:
            ax1.plot([0,25],[deltaE[p],deltaE[p]],color="gray",linestyle="--",linewidth=0.5)
        auto_ymin=np.mean(mass_mean[0,selectid])-error_mean[0,selectid]*8
        auto_ymax=np.mean(mass_mean[0,selectid])+error_mean[0,selectid]*8
        ax2.set_xlabel("start t")
        ax1.set_ylabel("$fit$ $\Delta M_{eff}$")
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
        ax1.set_title(r"%s %s $\tilde{q}^2$=%s (end t=%s)" % (state, object, p, beta[p]))
        if SAVEchijpg==True and deltat!=1:
            plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/2chi/{object}_fitchi_mom{p}_ratio.pdf", format='pdf',dpi=400)
            #plt.savefig(f"/public/home/gengyq/laph/{object}/picture/{conf_name}/{object}_chi_Meff_P{p}_{start[p]}_{beta[p]}_n{Nsamples}_nboot{nbsamples}_GEVP{GEVP}_bin{bin}.jpg",dpi=400)
        if TEST==True:          
            plt.savefig("/public/home/gengyq/test1.jpg",dpi=400)  
            
    ed00=time.time()
    print(f"all p={p} job done, time used : %.3f s"%(ed00-st0))
    print("")
    print("")
    if SAVEdataandgraph==True:    
        np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_E/{object}_mom{p}_{start[p]}_{beta[p]}_E_ratio.npy",save_E)
        # np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/{shape}/param_a/{object}_mom{p}_{start[p]}_{beta[p]}_{deltat}_n{Nsamples}_{method}_nb{nbsamples}_bin{bin}_GEVP{GEVP}_a.npy",save_a0)           
        # np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/{shape}/chi/{object}_mom{p}_{start[p]}_{beta[p]}_{deltat}_n{Nsamples}_{method}_nb{nbsamples}_bin{bin}_GEVP{GEVP}_chi.npy",save_chi)
        # np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/{shape}/param_dE/{object}_mom{p}_{start[p]}_{beta[p]}_{deltat}_n{Nsamples}_{method}_nb{nbsamples}_GEVP{GEVP}_bin{bin}_dE.npy",save_dE)
       


