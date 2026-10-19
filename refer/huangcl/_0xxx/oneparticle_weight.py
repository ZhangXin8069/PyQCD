import numpy as np
import fileinput
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import time
import lsqfit
import gvar as gv
from prettytable import PrettyTable
from lsq_tools import *
import glob


nbsamples = 3000


state="C48P29"
object= "Nucleon"
svdcuts=np.array([1e-12,1e-12,1e-12,1e-12,1e-12])
jack=False#True#
shape = "onestate"
SAVEdataandgraph=True# False#          #used to control whether save data and graph
effbandinchi=True#False#
SAVEchijpg= True# False#               #used to control whether save chi graph
TEST=     True#False#       #used to control whether test
Ptest=   0 #None#           #used to control which momentum
deltat=  8 
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
    Eexp=np.array([0.6,0.64,0.68,0.72,0.76])
    beta=22
    ymax=0.85 
    ymin=0.5 
    Lt = 50
    start=11
    Nsamples = 879
    Nmom = 5
    Nev=100
if state=="C32P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L32x64"
    Eexp=np.array([0.6,0.63,0.66,0.69,0.72])
    beta=27
    ymax=0.8
    ymin=0.45 
    Lt = 64
    start=11
    Nsamples = 430
    Nmom = 5
    Nev=100
if state=="C48P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L48x96"
    Eexp=np.array([0.54,0.56,0.58,0.6,0.62])
    beta=28
    ymax=0.8
    ymin=0.45 
    Lt = 64
    start=10
    Nsamples = 500
    Nmom = 5
    Nev=200
 
momlist=["Px0Py0Pz0","Px0Py0Pz1","Px0Py1Pz1","Px1Py1Pz1","Px0Py0Pz2"] 

######################   bin1    #################################

Nt = Lt//2

for p in [0]:
    # if TEST and deltat!=1:
    #     deltat=int(beta-start-1)
    print(F"momentum {p} fit job starts!!!!")

    if state=="C24P29":
        a=read_dat_file(F"/public/group/lqcd/corr/twopt/xinghy/beta6.20_mu-0.2770_ms-0.2400_L24x72/Nev100/corr_Nucleon_pp_{momlist[p]}",".dat",True)
    elif state=="C32P29":
        a_=read_dat_file(F"/public/group/lqcd/corr/twopt/xinghy/beta6.20_mu-0.2770_ms-0.2400_L32x64/Nev100/corr_Nucleon_pp_{momlist[p]}",".dat",True)
        if p==0:
            b_=read_dat_file(F"/public/home/gengyq/laph/proton/C32P29/result/corr_Nucleon_pp_{momlist[p]}.conf",".dat",False)/64
            a=np.concatenate([a_,b_],0)
        else:
            a=a_
    elif state=="C48P29":
        a=read_dat_file(F"/public/home/gengyq/laph/proton/C48P29/result/corr_Nucleon_pp_{momlist[p]}.conf",".dat",False)
    elif state=="C48P14":
        a=read_dat_file(F"/public/group/lqcd/corr/twopt/xinghy/beta6.20_mu-0.2825_ms-0.2310_L48x96/corr_Nucleon_pp_{momlist[p]}",".dat",True)
    print('data set : ',a.shape[0])
    Nsamples=a.shape[0]
    if jack==True:
        boot_corr = jackknife(a)
    else:
        boot_corr = bootstrap(a , nbsamples)
    st0 = time.time()
    
    contract = boot_corr[:,:Nt]
    print(contract.shape)
    #print(contract)
    N = contract.shape[0]  # your final number of samples
    print("N=", N)
    

    contract_sum = np.sum(contract, 0)
    sum_average = np.mean(contract, 0)

    # meson_mass_average = np.arccosh((np.roll(sum_average, -1) + np.roll(sum_average, 1)) / (2 * sum_average)) #/ Na* 0.1974
    # meson_mass_jk =np.arccosh((np.roll(contract, -1, 1) + np.roll(contract, 1, 1)) / (2 * contract)) #/ Na* 0.1974
    meson_mass_average = np.log(np.roll(sum_average, 1) / (sum_average)) #/ Na* 0.1974
    meson_mass_jk =np.log(np.roll(contract, 1, 1) / (contract)) #/ Na* 0.1974
    jkerros = np.std(meson_mass_jk, 0)
    if jack==True:
        jkerros=jkerros*np.sqrt(Nsamples-1) 
    # ------------------------fitting
    M_ij = contract - sum_average
    M_ij = np.matmul(M_ij.T, M_ij) 
    if jack==True:
        M_ij = M_ij*(Nsamples-1)/Nsamples
        print("jack!!")
    else:
        M_ij = M_ij/nbsamples

    M_dig = np.tril(M_ij)
    M_dig = np.triu(M_dig)


    def efunc(x, p):
        t00=0
        if shape == "onestate":
            return p["a0"] * np.exp(-p["E0"] * (x-t00))
        else:
            return p["a0"] * np.exp(-p["E0"] * (x-t00)) + (p["a1"]) * np.exp(-(p["E0"]+p["dE"]**2)* (x -t00))
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
    save_a = np.zeros([deltat,N])
    for t2 in range(1):  # right side
        for t1 in range(start, start + deltat):  # left side
            st1=time.time()
            print("t1=", t1, "t2=", t2 + beta)
            
            p0 = {}
            p0["a0"] = 0.5
            p0["E0"] = Eexp[p]
            if shape=="twostate":
                p0["a1"]=0.3
                p0["dE"] = 0.5
            # prior = {}
            # prior["a0"] = gv.gvar(0.3, 1e10)
            # prior["E0"] = gv.gvar(Eexp[p], 2)
            #if shape=="twostate":
                # prior["dE"] = gv.gvar(0.5, 1e10)
            
            gv_corr = gv.gvar(np.mean(contract,0), M_ij)
            fit_mean = lsqfit.nonlinear_fit(data=(T[t1 : t2 + beta + 1], gv_corr[t1 : t2 + beta + 1]), fcn=efunc,svdcut=1e-12,p0=p0)
            chi_mean=fit_mean.chi2/fit_mean.dof
            save_dof[t1 - start] = fit_mean.dof
            for fid in range(N):

                gv_corr = gv.gvar(contract[fid], M_ij)
                fit = lsqfit.nonlinear_fit(data=(T[t1 : t2 + beta + 1], gv_corr[t1 : t2 + beta + 1]), fcn=efunc,svdcut=1e-12,p0=p0)
                pr_corr = fit.p
                pr_arr_corr[fid][0] = pr_corr["E0"].mean
                pr_arr_corr[fid][1] = pr_corr["a0"].mean
                
                #print("id=",fid, "E=",fit.p["E0"].mean, "chis = ", fit.chi2/ (fit.dof+1))
                save_E[t1 - start,fid] = pr_corr["E0"].mean
                save_a[t1 - start,fid] = pr_corr["a0"].mean
                save_chi[t1 - start,fid] = fit.chi2 
                
            
            pr_aver_corr = np.mean(pr_arr_corr, 0)
            print(fit.p, "chis = ", chi_mean)  #-2
            print("dof",fit.dof)

            mass_mean[t2][t1 - start] = pr_aver_corr[0]
            chi_square[t2][t1 - start] = chi_mean
            
            error_mean[t2][t1 - start] = np.std(pr_arr_corr, 0)[0]
            if jack==True:
                error_mean[t2][t1 - start]=error_mean[t2][t1 - start]*np.sqrt(Nsamples-1)
            
            ed1=time.time()
            print(f"{t1} to {beta} fit done, time used : %.3f s"%(ed1-st1))
            
            ##-----------------------picture
            end = beta
            E_mean = mass_mean[0][t1 - start] #/ Na * 0.1974
            E_error = error_mean[0][t1 - start] #/ Na * 0.1974

            # Nt=int(Nt/2+1)
            T = np.arange(Nt)

            fig = plt.figure()
            # plt.plot([t1,t1],[0,10],color='black',alpha=0.3)
            
            plt.errorbar(
                x=T,
                y=meson_mass_average,
                yerr=jkerros,
                ecolor="cornflowerblue",
                linestyle="none",
                mec="cornflowerblue",
                marker="o",
                alpha=0.7,
                markerfacecolor="none",
                capsize=2,
                capthick=1,
                label="Meff",
            )
            plt.legend(loc='upper right')
            left, bottom, width, height = (t1, (E_mean - E_error), end - t1, 2 * E_error)
            rect = mpatches.Rectangle(
                (left, bottom),
                width,
                height,
                # fill=False,
                alpha=0.4,
                facecolor="red",
                label="fitting",
            )
            plt.gca().add_patch(rect)
            if p >2:
                plt.text(5, (ymax-ymin)*0.2+ymin, 'fit $E$ = %.5f'%(E_mean), fontsize=6)
                plt.text(5, (ymax-ymin)*0.15+ymin, 'error = %.5f'%(E_error), fontsize=6) 
                plt.text(5, (ymax-ymin)*0.1+ymin, '$\chi^2/d.o.f.$ = %.2f'%(chi_mean), fontsize=6)                               
            else:
                plt.text(5, (ymax-ymin)*0.8+ymin, '$\chi^2/d.o.f.$ = %.2f'%(chi_mean), fontsize=6)
                plt.text(5, (ymax-ymin)*0.9+ymin, 'fit $E$ = %.5f'%(E_mean), fontsize=6)
                plt.text(5, (ymax-ymin)*0.85+ymin, 'error = %.5f'%(E_error), fontsize=6)
            plt.legend(loc='upper right') 
            plt.title("%s %s |P|$^2$=%s" % (state, object,p))
            plt.xlabel("t")
            plt.ylabel("aE")
            #plt.ylabel("%s P%s%s%s Meff/Gev" % (object, px, py, pz))
            plt.xlim(0, Nt)
            plt.ylim(ymin, ymax)
            if SAVEdataandgraph==True:
                plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/effmass/{object}_eff_mom{p}.pdf", format='pdf',dpi=400)
                #plt.savefig(f"/public/home/gengyq/laph/Lambda_c/picture/{conf_name}/lambda_c_P{p}_t{t1}_{start}_{beta}_{deltat}_n{Nsamples}_nboot{nbsamples}_Nev{Nev}.jpg",dpi=400) #,dpi=200
            if TEST==True:
                plt.savefig(f"/public/home/gengyq/test.jpg",dpi=400) #,dpi=200
        ### --------------pretty table
        print("mass=", mass_mean)
        print("error=", error_mean)
        twostatefit = PrettyTable(
            ["start timeslice(end=" + str(beta) + ")", "chi2/dof", "mass mean", "fitting error"]
        )
        for i in range(deltat):
            twostatefit.add_row(
                [
                    np.arange(deltat)[i] + start,
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
        ax1.errorbar(
            x=np.arange(deltat) + start,
            y=mass_mean[0],
            yerr=error_mean[0],
            ecolor="black",
            linestyle="none",
            mec="black",
            marker="o",
            alpha=0.7,
            markerfacecolor="none",
            capsize=2,
            capthick=1,
            #label="%s $|P|^2$=%s $M_{eff}$" % (object,p),
        )
        plotx=np.arange(deltat) + start
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
        ax2.scatter(x=plotx,y=chi_square[0],color="cornflowerblue")
        # ax2.text(selectid+start,chi_square[0,selectid]+0.25,f"{chi_square[0,selectid]:0.2f}",ha='center')
        # ax1.text(selectid+start,mass_mean[0,selectid]+error_mean[0,selectid]*1.2,\
        #          f"{mass_mean[0,selectid]:0.4f}"+"("+"%0.0f"%(error_mean[0,selectid]*10000)+")",ha='center')
        ax2.plot([plotx[0]-0.5,plotx[-1]+0.5],np.ones(2),color="grey",alpha=0.7,linestyle="--")
        auto_ymin=mean-errall*8
        auto_ymax=mean+errall*8
        ax2.set_xlabel(r"$t_{min}$")
        ax1.set_ylabel("$fit$ $M_{eff}$")
        ax2.set_ylabel("$\chi^2/d.o.f.$")
        ax1.set_ylim(auto_ymin, auto_ymax)
        ax2.set_ylim(0, 3)   
        ax1.set_xlim(plotx[0]-0.5,plotx[-1]+0.5)
        ax2.set_xlim(plotx[0]-0.5,plotx[-1]+0.5)
        Xset=[]
        for i in plotx:
            Xset.append(f"{i}")
        ax2.set_xticks(plotx,Xset)
        ax1.set_xticks(plotx,Xset)
        Yset=[]
        for i in np.arange(0,3,0.5):
            Yset.append(f"{i}")
        ax2.set_yticks(np.arange(0,3,0.5),Yset)
        plt.subplots_adjust(hspace=0)
        ax1.set_title(r"%s %s  $M_{eff}$ and $\chi^2$($t_{max}$=%s)" % (state, object, beta))
        if SAVEchijpg==True:
            plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/chi/{state}_{object}_mom{p}.pdf", format='pdf',dpi=400)
            #plt.savefig(f"/public/home/gengyq/laph/Lambda_c/picture/{conf_name}/lambda_c_chi_Meff_P{p}_{start}_{beta}_n{Nsamples}_nboot{nbsamples}_Nev{Nev}.jpg",dpi=400)
        if TEST==True:
            plt.savefig("/public/home/gengyq/test1.jpg",dpi=400)

    ed00=time.time()
    print(f"all p={p} job done, time used : %.3f s"%(ed00-st0))
    print("")
    print("")
            
    
    np.savez(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_E/weighted_E.npz",mean=mean,stat=errsta,sys=err,weightEall=test)
        # np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_a/{object}_mom{p}_{start}_{beta}_n{Nsamples}_nboot{nbsamples}_Nev{Nev}_a.npy",save_a)
        # np.save(F"/public/home/gengyq/laph/NP/fitwork/result/{state}/chi/{object}_mom{p}_{start}_{beta}_n{Nsamples}_nboot{nbsamples}_Nev{Nev}_chi.npy",save_chi)
    
    