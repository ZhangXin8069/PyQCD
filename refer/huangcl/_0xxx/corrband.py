#!/public/home/xinghy/anaconda3-2023.03/bin/python

# test file to check whether first term has problem

import numpy as np
import fileinput

import matplotlib.pyplot as plt
from lsq_tools import *


shape = "twostate"
method = "p0"

state = "C24P29"
object = "NN"



Na = 0.077
nbsamples=3000
datapath="/public/home/gengyq/laph/di_lambda_c/fit_result/"   

if state=="C24P29":
    Nmom=2
    Lx = 24
    Nt = 72
    Nt0 = 72
    Pmv=None
    beta=[16,15 ]  
    start=[8,10 ]  
    deltat=1
    Nsamples=879
    bin=None#2
    GEVP=True#False#
else:
    Nmom=3
    Lx = 32
    Nt = 64
    Nt0 = 64
    Pmv=None
    beta=[18,17,14 ]  
    start=[10,11,10] 
    deltat=1
    Nsamples=430
    bin=None#2
    GEVP=True#False#

E_all1=np.zeros((Nmom,nbsamples))
a0=np.zeros((Nmom,nbsamples))
dE=np.zeros((Nmom,nbsamples))
for p in range(Nmom):   
    E_all1[p] = np.load(\
        f"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_E/NP_mom0_8_16_1_n872_p0_nb3000_GEVPTrue_binNone_E.npy")[0]              
    a0[p] = np.load(\
        f"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_a/NP_mom0_8_16_1_n872_p0_nb3000_GEVPTrue_binNone_a.npy")[0]              
    dE[p] = np.load(\
        f"/public/home/gengyq/laph/NP/fitwork/result/{state}/param_dE/NP_mom0_8_16_1_n872_p0_nb3000_GEVPTrue_binNone_dE.npy")[0]         
print("a",np.mean(a0,1))     
print("E",np.mean(E_all1,1)) 
   
Nt=25


if Pmv!=None :
    E_all1=np.sqrt((E_all1)**2+(2*np.pi/Lx)**2*Pmv)    

momset=["01","12","14","23"]
for i in range(1):
    if GEVP:
        t00=2
        prefix=f"/public/home/gengyq/laph/NP/fitwork/GEVPcorr/{state}/{object}/sort/eigvals_{i}_t0_{t00}"
        subfix=".dat"
        corr=read_dat_file(prefix,subfix,GEVP)[:,:25]
        print("GEVP!")
    else:
        corr=np.load(f"/public/home/gengyq/laph/NP/{state}/samples{Nsamples}/corr_{Nsamples}_6.npy")[:,i,i]
        corr=bootstrap(corr,3000)
        t00=0
        print("raw data")
    print(corr.shape)

    corr_error=np.std(corr,0)
    Efit=np.mean(E_all1[i])
    print(Efit)
    fiterr=np.std(E_all1[i])
    
    data_sum_ave = np.zeros(Nt)
    data_M_err = np.zeros(Nt)
    epslon=0.001
    T_corr = np.arange(0,Nt,epslon)
    T = np.arange(Nt)
    data_M = np.zeros((nbsamples, T_corr.shape[0]))  # fitting data N*Nt
    for n in range(nbsamples):
        data_M[n] = a0[i,n]*np.exp(-E_all1[i,n] * (T_corr - t00)) + \
            (1-a0[i,n])*np.exp(-(E_all1[i,n]+dE[i,n]**2) * (T_corr - t00))
        #data_M[n] = pr_arr_corr[n, 1] + (1-pr_arr_corr[n, 1])*np.exp(-(pr_arr_corr[n, 3]**2) * (T_corr - t00))

    data_M= np.exp(Efit * T_corr)*data_M
    #print(data_M)
    data_sum_ave = np.mean(data_M, 0)  # data mean
    data_M_err = np.std(data_M, 0)  # data error
    

    fig = plt.figure()
    plt.errorbar(x=T,y=np.mean(corr,0)*np.exp(Efit * (T)),yerr=corr_error*np.exp(Efit * T),ecolor="cornflowerblue",linestyle="none",\
        mec="cornflowerblue", marker="o", alpha=0.7, markerfacecolor="none",capsize=2,capthick=1,label="corr")
    
    y1=data_sum_ave[int(start[i]/epslon):int(beta[i]/epslon)]-data_M_err[int(start[i]/epslon):int(beta[i]/epslon)]
    y2=data_sum_ave[int(start[i]/epslon):int(beta[i]/epslon)]+data_M_err[int(start[i]/epslon):int(beta[i]/epslon)]
    plt.plot(T_corr[int(start[i]/epslon):int(beta[i]/epslon)],data_sum_ave[int(start[i]/epslon):int(beta[i]/epslon)],alpha=0.7,color="red")
    plt.fill_between(x=T_corr[int(start[i]/epslon):int(beta[i]/epslon)],y1=y1,y2=y2,alpha=0.5,color="red",label="fit")
    #plt.yscale('log')
    plt.legend(loc='upper right')
    #plt.title("%s %s |P|$^2$=%s corr" % (state, object,p))
    plt.xlabel("t")
    plt.ylabel("corr")
    plt.xlim(1, 40)
    
    ymax=np.max(((np.mean(corr,0)+corr_error)*np.exp(Efit * T))[start[i]:beta[i]])
    ymin=np.min(((np.mean(corr,0)-corr_error)*np.exp(Efit * T))[start[i]:beta[i]])
    
    plotymin=1.5*ymin-ymax*0.5
    plotymax=-0.5*ymin+ymax*1.5
    
    #plt.yscale("log")
    plt.ylim(plotymin,plotymax)
    #plt.ylim(0,5000)
    #plt.savefig("/public/home/gengyq/test2.jpg",dpi=400)
    # plt.savefig(\
    #     f"/public/home/gengyq/laph/di_lambda_c/001/result/{conf_name}/A1/picture/corrband/eigen{i}.pdf",dpi=400)
    plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/corr/eigen{i}_t{start[p]}.pdf",dpi=400)





