import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lsq_tools import *

state = "C48P29"
object="NN"
t0=10

if state=="C24P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L24x72"
    idALL=np.arange(872)*50+4050
    Lt=72
    bin=    None
    Nsamples = 879
    Nmom=2
if state=="C32P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L32x64"
    idALL=np.arange(431)*50+11000
    Lt=64
    bin=    None
    Nsamples = 430
    Nmom=3

Nmom=5
Lt=96
input_dir=F"/public/home/gengyq/laph/NP/fitwork/GEVPcorr/C48P29/T1SDmix/"
contract_all=np.load("%s/corr_matrix_boots.npz" %input_dir)['corr_matrix'].transpose(0,2,3,1)


dataplot=np.real(np.mean(contract_all[:,:,:,t0],0))
datamatrix=np.zeros([Nmom,Nmom,Lt])   
for i in range(Nmom):
  for j in range(Nmom):
    if j >= i:
      datamatrix[i,j] = abs(dataplot[i,j]/np.sqrt(abs(dataplot[i,i]*dataplot[j,j])))
      datamatrix[j,i] = datamatrix[i,j]
print(datamatrix[...,t0])    

#-------------------------seaborn.heatmap
fig=plt.figure(facecolor='white')
sns.set()

if state=="C24P29":
  ax = sns.heatmap(datamatrix[:,:,t0], annot=True,cmap='YlGnBu_r',center=0.5,vmin=0, vmax=1,\
                  xticklabels=['$\lambda$0','$\lambda$1'],\
                    yticklabels=['$\lambda$0','$\lambda$1'],fmt=".3f") # fmt=".2e",
elif state=="C32P29":
    ax = sns.heatmap(datamatrix[:,:,t0], annot=True,cmap='YlGnBu_r',center=0.5,vmin=0, vmax=1,\
                  xticklabels=['$\lambda$0','$\lambda$1','$\lambda$2'],\
                    yticklabels=['$\lambda$0','$\lambda$1','$\lambda$2'],fmt=".3f") # fmt=".2e",
elif state=="C48P29":
    ax = sns.heatmap(datamatrix[:,:,t0], annot=True,cmap='YlGnBu_r',center=0.5,vmin=0, vmax=1,\
                  xticklabels=['S(q=0)','S(q=1)','S(q=2)','D(q=1)','D(q=2)'],\
                    yticklabels=['S(q=0)','S(q=1)','S(q=2)','D(q=1)','D(q=2)'],fmt=".3f")
ax.xaxis.tick_top()
ax.set_facecolor('white')
for i in range(1,datamatrix.shape[0]):
    plt.axhline(i,0, Nmom,color='white', lw=2, linestyle='--')
    plt.axvline(i,Nmom,0, color='white', lw=2, linestyle='--')
# plt.xticks(np.arange(Nmom)+0.5,['P0','P1','P2','P3'])
# plt.yticks(np.arange(Nmom)+0.5,['P0','P1','P2','P3'])
ax.set_xlim(0,Nmom)
ax.set_ylim(Nmom,0)
ax.tick_params(axis='both', which='both', length=0, width=0)



plt.xlabel("%s $NN$ $^3S_1-^3D_1$"%(state))
plt.gca().xaxis.labelpad = 10

# plt.title(F"{state} $P$=[0,0,0] t0={t0}")
# plt.savefig("/public/home/gengyq/laph/di_lambda_c/picture/beta6.41_mu-0.2295_ms-0.2050_L48x96/hotmap.jpg",dpi=400)

plt.savefig("/public/home/gengyq/test.jpg",dpi=400)
plt.savefig(f"/public/home/gengyq/laph/NP/fitwork/picture/{state}/NN_P000_t{t0}.pdf",dpi=400)