


import fileinput
import numpy as np
import time
from gamma_matrix_cupy_DP import *
from opt_einsum import contract
from readin import *
from VVV_tools import *
from matrixes import *
import cupy as cp


infile=fileinput.input()
for line in infile:
    tmp=line.split()
    if(tmp[0]=='Lt'):
        Lt=int(tmp[1])
    if(tmp[0]=='Lx'):
        Lx=int(tmp[1])
    if(tmp[0]=='conf_id'):
        conf_id=int(tmp[1])
    if(tmp[0]=='conf_name'):
        conf_name=tmp[1]
    if(tmp[0]=='Nev'):
        Nev=int(tmp[1])
    if(tmp[0]=='Nev1'):  #number of eigenvectors used in contraction
        Nev1=int(tmp[1])
    if(tmp[0]=='peram_dir'):
        peram_dir=tmp[1]
    if(tmp[0]=='VVV_dir'):
        VVV_dir=tmp[1]
    if(tmp[0]=='eig_dir'):
        eig_dir=tmp[1]
    if(tmp[0]=='result_dir'):
        result_dir=tmp[1]
        


###############################################################################

Px = [0, 0, 0, 1, 0, 0, 0,-1]
Py = [0, 0, 1, 0, 0, 0,-1, 0]
Pz = [0, 1, 0, 0, 0,-1, 0, 0]
# Px = [0,0]
# Py = [0,0]
# Pz = [0,0]
Nmom = len(Px)
st = time.time()

################################   VVV load   #####################################
eig_dir = f"/public/group/lqcd/eigensystem/{conf_name}/{conf_id}/"
VVV = np.zeros([Nmom, Lt, Nev1, Nev1, Nev1], "<c16")
# saveVVV=f"/public/group/lqcd/corr/twopt/gengyq/{conf_name}/{conf_id}/"
for p in range(Nmom):
    st00=time.time()
    try:    
        VVV[p]=readin_VVV(VVV_dir, Nev, Nev1, Lt, conf_id, Px[p], Py[p], Pz[p])
    except:
        VVV_=VVV_cal(eig_dir,Lx,Lt,Nev,conf_id, Px[p], Py[p], Pz[p])
        VVV_ = cp.asnumpy(VVV_)
        VVV[p] = VVV_[:,:Nev1,:Nev1,:Nev1]
    ed00=time.time()
    print("element p=%s%s%s load done, time used: %.3f s" % (Px[p],Py[p],Pz[p],ed00 - st00))
VVV = np.transpose(VVV,(1,0,2,3,4))  # Lt,id,Ne,Ne,Ne

ed = time.time()
print("element load done, time used: %.3f s" % (ed - st))

###############################  peramb path set   #######################################

# st = time.time()
peram_u_dir = f"{peram_dir}" #/light
Cgamma5=cp.asarray(gamma(7))
# cp.cuda.Device().synchronize()
# ed = time.time()
# print("read peram done, time used: %.3f s" % (ed - st))

#######################  quark spin and eigenid set for wick contraction  ##################################
NPnum=len(NP_indexes)
# NNnum=len(NN_indexes)
print("NP diagrams",NPnum)
# print("NN diagrams",NNnum)

###############################################################################

P_snk=0
P_src=0

di_baryon_NP = cp.zeros((Nmom//2,Nmom,Lt,2,2,2,2), "<c16")
# di_baryon_NN = cp.zeros((Nmom//2,Nmom//2,Lt,2,2,2,2), "<c16")
tmpNP=cp.zeros((4,4,4,4),"<c16")
# tmpNN=cp.zeros((4,4,4,4),"<c16")

st0=time.time()
for t_src in np.arange(1):  # Lt, if use debug-gpu, set 1
    st1 = time.time()
    
    peramb_all_u=cp.array(readin_peram_cpu(peram_u_dir, conf_id, t_src, Lt, Nev1))

    for p2 in range(Nmom):
        if p2==0:
            P_src=0
        elif p2==4:
            continue
        else:
            P_src=1

            
        VVV_src1 = cp.conj(cp.asarray(VVV[t_src,p2]))
        VVV_src2 = cp.conj(cp.asarray(VVV[t_src,int((p2 + Nmom // 2) % Nmom)]))
        
        for t_snk in range(Lt):
            if int((t_snk-t_src)%Lt) < 24: 
                peramb_u=peramb_all_u[(t_snk-t_src)%Lt]

                for p1 in range(Nmom//2):
                    VVV_snk1=cp.array(VVV[t_snk,p1])
                    VVV_snk2=cp.array(VVV[t_snk,int((p1 + Nmom // 2) % Nmom)])
                    if p1==0:
                        P_snk=0
                    else:
                        P_snk=1

                        
                    if P_snk <= P_src:
                        for id1 in range(NPnum):
                            tmpNP +=  -1*NP_coeff[id1]*contract(
                            NP_indexes[id1]+",mo,ace,np,bdf,uw,gik,vx,hjl->rqzy",
                            peramb_u,peramb_u,peramb_u,peramb_u,peramb_u,peramb_u,Cgamma5,VVV_src1,Cgamma5,VVV_snk1,
                            Cgamma5,VVV_src2,Cgamma5,VVV_snk2
                            ) 
                        # for id2 in range(NNnum):
                        #     tmpNN += -1* NN_coeff[id2]*contract(
                        #     NN_indexes[id2]+",mo,ace,np,bdf,uw,gik,vx,hjl->rqzy",
                        #     peramb_u,peramb_u,peramb_u,peramb_u,peramb_u,peramb_u,Cgamma5,VVV_src1,Cgamma5,VVV_snk1,
                        #     Cgamma5,VVV_src2,Cgamma5,VVV_snk2
                        #     ) 
                            
                        # where r is spin = 1/2 and z is -1/2 for particle 1 at sink while q,y for particle 2 at source.

                        di_baryon_NP[p1, p2, (t_snk - t_src) % Lt] += tmpNP[:2,:2,:2,:2]
                        # di_baryon_NN[p1, p2, (t_snk - t_src) % Lt] += tmpNN[:2,:2,:2,:2]
                        
                        tmpNP=cp.zeros((4,4,4,4),"<c16")
                        # tmpNN=cp.zeros((4,4,4,4),"<c16")
   
    ed1 = time.time()
    print(f"t_src {t_src} caululate done , time used: %.3f s" % (ed1 - st1))

np.save("%s/NPcorr_conf%s.npy"%(result_dir,conf_id),di_baryon_NP)
ed0=time.time()
print(f"all t_src caululate and save done, time used: %.3f s" % (ed0 - st0))