#

import numpy as np
import torch
import os
import fileinput
import math as math
from gamma_matrix_DR_torch import *
from input_output_4_torch import *
from opt_einsum import contract
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

infile=fileinput.input()
for line in infile:
    tmp=line.split()
    if(tmp[0]=='Nt'):
        Nt=int(tmp[1])
    if(tmp[0]=='Nx'):
        Nx=int(tmp[1])
    if(tmp[0]=='conf_id'):
        conf_id=tmp[1]
    if(tmp[0]=='Nev1'):    #number of eigenvectors used in contraction
        Nev1=int(tmp[1])
    if(tmp[0]=='Px'):
        Px=int(tmp[1])
    if(tmp[0]=='Py'):
        Py=int(tmp[1])
    if(tmp[0]=='Pz'):
        Pz=int(tmp[1])
    if(tmp[0]=='peram_u_dir'):
        peram_u_dir=tmp[1]
    if(tmp[0]=='VVV_dir'):
        VVV_dir=tmp[1]
    if(tmp[0]=='corr_nucl_dir'):
        corr_nucl_dir=tmp[1]
    if(tmp[0]=='eigen_dir'):
        eigen_dir=tmp[1]
#peram_size=Nt*4*Nev*Nev*2

Nc = 3

def phase_calc(Mom): # Mom = P(z,y,x)
  phase_factor = np.zeros(Nx*Nx*Nx, dtype=complex)
  for z in range(0,Nx):
    for y in range(0,Nx):
      for x in range(0,Nx):
        Pos = np.array([z,y,x])
        phase_factor[z*Nx*Nx + y*Nx + x] = np.exp( -np.dot(Mom,Pos)*2*np.pi*1j/Nx )
  return torch.from_numpy(phase_factor)

def VVV_timeslice_calc(Mom, eigvecs_cupy):
    VVV_timeslice_device = torch.zeros((Nev1, Nev1, Nev1), dtype=torch.cdouble)
    phase_factor_cupy=phase_calc(Mom)
    for xi in range(0, Nx): 
        VVV_timeslice_device = VVV_timeslice_device + \
            contract("x,ax,bx,cx->abc", phase_factor_cupy[xi*(Nx**2):(xi+1)*(Nx**2)], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),0], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),1], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),2] ) + \
            contract("x,ax,bx,cx->abc", phase_factor_cupy[xi*(Nx**2):(xi+1)*(Nx**2)], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),1], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),2], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),0] ) + \
            contract("x,ax,bx,cx->abc", phase_factor_cupy[xi*(Nx**2):(xi+1)*(Nx**2)], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),2], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),0], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),1] ) - \
            contract("x,ax,bx,cx->abc", phase_factor_cupy[xi*(Nx**2):(xi+1)*(Nx**2)], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),2], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),1], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),0] ) - \
            contract("x,ax,bx,cx->abc", phase_factor_cupy[xi*(Nx**2):(xi+1)*(Nx**2)], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),0], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),2], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),1] ) - \
            contract("x,ax,bx,cx->abc", phase_factor_cupy[xi*(Nx**2):(xi+1)*(Nx**2)], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),1], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),0], eigvecs_cupy[:,xi*(Nx**2):(xi+1)*(Nx**2),2] )
    VVV_timeslice_host = VVV_timeslice_device.numpy()
    return VVV_timeslice_host

def VDV_assemble(eig_dir, t, Nev1, conf_id, Nx, Px, Py, Pz):
    Mom = np.array([Pz,Py,Px])
    
    exp_diag = np.zeros(Nx*Nx*Nx*Nc, dtype=np.complex128)   # M
    for z in range(0,Nx):
        for y in range(0,Nx):
            for x in range(0,Nx):
                Pos = np.array([z,y,x])
                exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3] = np.exp( -np.dot(Mom,Pos)*2*math.pi*1j/Nx )
                exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3 + 1] = exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3]
                exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3 + 2] = exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3]

    eigvecs_cupy = readin_eigvecs_cpu(eig_dir, t, Nev1,conf_id, Nx).reshape(Nev1,Nx*Nx*Nx*3)
    VDV_cupy = np.zeros((Nev1, Nev1), dtype=np.complex128)
    exp_diag_cupy=np.asarray(exp_diag)
    VDV_cupy=np.matmul((eigvecs_cupy * exp_diag_cupy), np.conj(eigvecs_cupy.T))
    VDV_cupy = np.conj(VDV_cupy.T)

    return VDV_cupy

st00 = time.time()

matrix_pplus = 0.5*(gamma(0)+gamma(4))#positive parity projection
matrix_pplus =matrix_pplus.cpu().numpy()
matrix_pminus = 0.5*(gamma(0)-gamma(4))#negative parity projection
matrix_pminus =matrix_pminus.cpu().numpy()

corr_nucl_pp=np.zeros((1,Nt), dtype=complex)
corr_nucl_pm=np.zeros((1,Nt), dtype=complex)

contrac_nucl_matrix_1 = torch.zeros((Nt,Nt,4,4), device=device, dtype=torch.cdouble)
contrac_nucl_matrix_2 = torch.zeros((Nt,Nt,4,4), device=device, dtype=torch.cdouble)
contrac_nucl_matrix_3 = torch.zeros((Nt,Nt,4,4), device=device, dtype=torch.cdouble)
contrac_nucl_matrix = torch.zeros((Nt,Nt,4,4), device=device, dtype=torch.cdouble)
# st = time.time()
# VVV_mid = readin_eigvecs_device(eigen_dir, t, Nev1, conf_id, Nx)
# VVV_sink = VVV_timeslice_calc([0,0,1],)                                                   # readin_VVV_all_device(VVV_dir, Nev1,Nt,conf_id, Px, Py, Pz)
# ed = time.time()
# print("read VVV_sink done, time used: %.3f s" %(ed-st))

VVV_sink = torch.zeros((Nt,Nev1, Nev1, Nev1), dtype=torch.cdouble)
VdV_sink = torch.zeros((Nt,Nev1,Nev1), dtype=torch.complex128)
st = time.time()
peram_cpu_all_light_u=readin_peram_all_cpu(peram_u_dir, conf_id, Nt, Nev1)
ed = time.time()
print("read peram_u done, time used: %.3f s" %(ed-st))
st10 = time.time()
# for t in range(0,Nt):
#   VVV_mid = readin_eigvecs_cpu(eigen_dir, t, Nev1, conf_id, Nx)
#   VVV_sink[t] = torch.from_numpy(VVV_timeslice_calc([Pz,Py,Px] , VVV_mid))
  
#   del VVV_mid
for ti in range(Nt):
    VdV_sink[ti] = torch.from_numpy(VDV_assemble(eigen_dir, ti, Nev1, conf_id, Nx, Px, Py, Pz))
VVV_sink = readin_VVV_all_device(VVV_dir, Nev1, Nt, conf_id, Px, Py, Pz)

ed10 = time.time()

print("creat VVV has complished, time used: %.5f "%(ed10-st10))
qq_ones = torch.ones((4,4), dtype = torch.complex128)
# ------------------------------------------------------------------------------------------------

VVV_sink = VVV_sink.cuda()
VdV_sink = VdV_sink.cuda()
qq_ones = qq_ones.cuda()

st = time.time()

for t_source in range(0,Nt):
    st0 = time.time()
 
    st1 = time.time()
    # peram_u=readin_peram_v2(peram_u_dir, conf_id, Nt, Nev, Nev1, t_source)
    peram_u=peram_cpu_all_light_u[t_source].cuda()
    torch.cuda.synchronize()
    ed1 = time.time()
    print("transfer perambulator data from cpu to device, time used:  %.5f s" %(ed1-st1))
    
    
    st1 = time.time()
    CG5_peram_d_CG5=contract("gh,thkbe,jk->tgjbe", gamma(7), peram_u, gamma(7))
    VVV_source = torch.conj(VVV_sink[t_source])
    torch.cuda.synchronize()
    ed1 = time.time()
    print("some trival calculation, time used:  %.5f s" %(ed1-st1))
    

    st1 = time.time()
    for t_sink in range(0,Nt):
        contrac_nucl_matrix_1[t_sink,t_source] = \
        contract("ijk,vqbp,ab,qnia,hnjo,hmkl,pol->vm", VVV_sink[t_sink], peram_u[t_sink], VdV_sink[t_sink], peram_u[t_sink], CG5_peram_d_CG5[t_sink], peram_u[t_sink], VVV_source)- \
        contract("ijk,vqbp,ab,hnil,hnjo,qmka,pol->vm", VVV_sink[t_sink], peram_u[t_sink], VdV_sink[t_sink], peram_u[t_sink], CG5_peram_d_CG5[t_sink], peram_u[t_sink], VVV_source)
        
        contrac_nucl_matrix_2[t_sink,t_source] = \
        contract("ijk,hqbl,ab,vmkp,hnjo,qnia,pol->vm", VVV_sink[t_sink], peram_u[t_sink], VdV_sink[t_sink], peram_u[t_sink], CG5_peram_d_CG5[t_sink], peram_u[t_sink], VVV_source)- \
        contract("ijk,vnip,ab,hqbl,hnjo,qmka,pol->vm", VVV_sink[t_sink], peram_u[t_sink], VdV_sink[t_sink], peram_u[t_sink], CG5_peram_d_CG5[t_sink], peram_u[t_sink], VVV_source)
        
        # contrac_nucl_matrix_3[t_sink,t_source] = \
        # contract("ijk,qg,qgab,ab,hnip,hnjo,vmkl,pol->vm", VVV_sink[t_sink], qq_ones, peram_u[t_sink], VdV_sink[t_sink], peram_u[t_sink], CG5_peram_d_CG5[t_sink], peram_u[t_sink], VVV_source)- \
        # contract("ijk,qg,qgab,ab,hmkp,hnjo,vnil,pol->vm", VVV_sink[t_sink], qq_ones, peram_u[t_sink], VdV_sink[t_sink], peram_u[t_sink], CG5_peram_d_CG5[t_sink], peram_u[t_sink], VVV_source)
    
    torch.cuda.synchronize()
    ed1 = time.time()
    print("main calculation, time used:  %.5f s" %(ed1-st1))
    
    ed0 = time.time()
    print("****************all complete for t_source %d, time used: %.3f s****************" %(t_source, ed0-st0))
    
contrac_nucl_matrix = contrac_nucl_matrix_1 - contrac_nucl_matrix_2 + contrac_nucl_matrix_3

contrac_nucl_matrix=contrac_nucl_matrix.cpu().numpy()
ed = time.time()
print("All calculation on GPU done, time used: %.3f s" %(ed-st))

# ------------------------------------------------------------------------------------------------
st0 = time.time()
contrac_nucl_pp=contract("li,yxil->yx",matrix_pplus,contrac_nucl_matrix)
contrac_nucl_pm=contract("li,yxil->yx",matrix_pminus,contrac_nucl_matrix)	


for t_source in range(0,Nt):
    for t_sink in range(0,Nt):
        if(t_sink<t_source):
            contrac_nucl_pp[t_sink,t_source]=-1.0*contrac_nucl_pp[t_sink,t_source]

        if(t_sink>t_source):
            contrac_nucl_pm[t_sink,t_source]=-1.0*contrac_nucl_pm[t_sink,t_source]

        corr_nucl_pp[0,(t_sink-t_source+Nt)%Nt] = corr_nucl_pp[0,(t_sink-t_source+Nt)%Nt] + contrac_nucl_pp[t_sink,t_source]
        corr_nucl_pm[0,(t_sink-t_source+Nt)%Nt] = corr_nucl_pm[0,(t_sink-t_source+Nt)%Nt] + contrac_nucl_pm[t_sink,t_source]

ed0 = time.time()
print("****************final calculation on CPU done, time used: %.3f s****************" %( ed0-st0))

# ------------------------------------------------------------------------------------------------

ed00 = time.time()
print("All io and calculation done, time used: %.3f s" %(ed00-st00))


write_data_ascii(corr_nucl_pp, Nt, Nx, "%s/corr_uud_1_plus_Px%sPy%sPz%s.conf%s_torch.dat"%(corr_nucl_dir,Px,Py,Pz,conf_id))
write_data_ascii(corr_nucl_pm, Nt, Nx, "%s/corr_uud_1_nega_Px%sPy%sPz%s.conf%s_torch.dat"%(corr_nucl_dir,Px,Py,Pz,conf_id))

print("succeed")