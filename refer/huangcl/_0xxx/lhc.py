import numpy as np
import matplotlib.pyplot as plt
from lsq_tools import *

state="F48P30"
object="NP"
Nev=100
data=np.load(f"/public/home/gengyq/laph/NP/fitwork/GEVPcorr/{state}/{object}/Nev{Nev}/eigvals_0_t0_2.npy")
mass=np.log(data/np.roll(data,-1,1))

mpi=304/197.33*0.07746
mN=0.417028079395258
mlhc1=np.sqrt(mN**2-mpi**2/4)*2
mlhc2=np.sqrt(mN**2+mpi**2/4)*2

Time=np.arange(96)
plt.plot([0,96],[mlhc1,mlhc1],color="grey")
plt.plot([0,96],[mlhc2,mlhc2],color="grey",label="lhc")
plt.plot([0,96],[mN*2,mN*2],color="black",label="NN")

plt.xlim(0,48)
plt.ylim(0.78,0.88)
plt.errorbar(Time,np.mean(mass,0),yerr=np.std(mass,0),linestyle="none",label="F48P30")
data=np.load(f"/public/home/gengyq/laph/NP/fitwork/GEVPcorr/{state}/{object}/Nev{Nev}/eigvals_1_t0_2.npy")
mass=np.log(data/np.roll(data,-1,1))
plt.errorbar(Time,np.mean(mass,0),yerr=np.std(mass,0),linestyle="none")
plt.legend(loc="lower right")
plt.savefig("/public/home/gengyq/test.jpg",dpi=400)

