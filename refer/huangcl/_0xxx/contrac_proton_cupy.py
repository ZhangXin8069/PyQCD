#!/beegfs/home/liuming/software/install/python/bin/python3

#test file to check whether first term has problem

import numpy as np
import cupy as cp
import math
import os
import fileinput
from gamma_matrix_cupy_DR import *
from input_output_4_cupy import *
from opt_einsum import contract
import time

print("Job started")
Nc=3
# ------------------------------------------------------------------------------
infile=fileinput.input()
for line in infile:
  tmp=line.split()
  if(tmp[0]=='Nt'):
    Nt=int(tmp[1])
  if(tmp[0]=='Nx'):
    Nx=int(tmp[1])
  if(tmp[0]=='conf_id'):
    conf_id=tmp[1]
  if(tmp[0]=='Nev1'):  #number of eigenvectors used in contraction
    Nev1=int(tmp[1])
  if(tmp[0]=='Px'):
    Px=int(tmp[1])
  if(tmp[0]=='Py'):
    Py=int(tmp[1])
  if(tmp[0]=='Pz'):
    Pz=int(tmp[1])
  if(tmp[0]=='peram_u_dir'):
    peram_u_dir=tmp[1]
  if(tmp[0]=='eigen_dir'):
    eig_dir=tmp[1]
  if(tmp[0]=='corr_nucl_dir'):
    corr_nucl_dir=tmp[1]
  if(tmp[0]=='VVV_dir'):
    VVV_dir=tmp[1]


corr_proton_plus=np.zeros((1,Nt), dtype=complex)
corr_proton_nega=np.zeros((1,Nt), dtype=complex)

Mom=np.array([Pz,Py,Px])

# gamma matrix
pplus=(gamma(0)+gamma(4))/2
pnega=(gamma(0)-gamma(4))/2

# calculate the exp part
phase_exp = np.zeros(Nx**3, dtype=complex)
for z in range(0,Nx):
  for y in range(0,Nx):
    for x in range(0,Nx):
      ni = np.array([z,y,x])
      phase_exp[z*(Nx**2)+y*Nx+x] = np.exp(-2*np.pi*1j*np.dot(Mom,ni)/Nx)
      
# calculate the multiply part
def Mom_VVV_sin_t(Mom,mid_VVV):
  VVV_timeslice_t_1 = cp.zeros((Nev1,Nev1,Nev1), dtype=cp.complex128)
  VVV_timeslice_t_1 = VVV_timeslice_t_1 + \
    contract("x,ax,bx,cx->abc", phase_exp, mid_VVV[:,:,0], mid_VVV[:,:,1], mid_VVV[:,:,2]) + \
    contract("x,ax,bx,cx->abc", phase_exp, mid_VVV[:,:,1], mid_VVV[:,:,2], mid_VVV[:,:,0]) + \
    contract("x,ax,bx,cx->abc", phase_exp, mid_VVV[:,:,2], mid_VVV[:,:,0], mid_VVV[:,:,1]) - \
    contract("x,ax,bx,cx->abc", phase_exp, mid_VVV[:,:,0], mid_VVV[:,:,2], mid_VVV[:,:,1]) - \
    contract("x,ax,bx,cx->abc", phase_exp, mid_VVV[:,:,1], mid_VVV[:,:,0], mid_VVV[:,:,2]) - \
    contract("x,ax,bx,cx->abc", phase_exp, mid_VVV[:,:,2], mid_VVV[:,:,1], mid_VVV[:,:,0])
  VVV_timeslice_t = cp.asnumpy(VVV_timeslice_t_1)
  return VVV_timeslice_t

#caculate VdV
def VDV_assemble(eig_dir, t, Nev1, conf_id, Nx, Px, Py, Pz):
    Mom = np.array([Pz,Py,Px])
    
    exp_diag = np.zeros(Nx*Nx*Nx*Nc, dtype=complex)   # M
    for z in range(0,Nx):
        for y in range(0,Nx):
            for x in range(0,Nx):
                Pos = np.array([z,y,x])
                exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3] = np.exp( -np.dot(Mom,Pos)*2*np.pi*1j/Nx )
                exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3 + 1] = exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3]
                exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3 + 2] = exp_diag[z*Nx*Nx*3 + y*Nx*3 + x*3]

    eigvecs_cupy = readin_eigvecs_device(eig_dir, t, Nev1, conf_id, Nx)
    VDV_cupy = cp.zeros((Nev1, Nev1), dtype=complex)
    exp_diag_cupy=cp.asarray(exp_diag)
    
    VDV_cupy=cp.matmul((cp.conj(cp.transpose(eigvecs_cupy)) * exp_diag_cupy), eigvecs_cupy) 
    
    return VDV_cupy

# read in the eigenvector and calculate the mom sink
st_vvv = time.time()           

VVV_sink = cp.zeros((Nt, Nev1, Nev1, Nev1), dtype=cp.complex128)

for t in range(0,Nt):
  
  eigvecs_device = readin_eigvecs_device(eig_dir,t,Nev1,conf_id,Nx)
  VVV_sink[t] = Mom_VVV_sin_t(Mom, eigvecs_device)
  
  del eigvecs_device

VdV_sink = cp.zeros((Nt,Nev1,Nev1), dtype=cp.complex128)

for ti in range(0,Nt):
  VdV_sink[t] = VDV_assemble(eig_dir, ti, Nev1, conf_id, Nx, Px, Py, Pz)

# VVV_sink=readin_VVV_all_device(VVV_dir, Nev1, Nt, conf_id, Px, Py, Pz) 

ed_vvv = time.time()  
print("VVV and VdV done, time used: %.3f s" %(ed_vvv-st_vvv))                       

corr_proton_matrix=cp.zeros((Nt,Nt,4,4), dtype=complex)

for t_source in range(0,Nt):
  
    st0 = time.time()

    VVV_source = cp.conj(VVV_sink[t_source])  # [t_source].T
    peram_u=readin_peram_device(peram_u_dir, conf_id, Nt, Nev1 , t_source)

    CG5_peram_d_CG5=contract("gh,thkbe,jk->tgjbe", gamma(7), peram_u, gamma(7))
    
    for t_sink in range(0,Nt):
      
      corr_proton_matrix[t_sink,t_source] = \
        contract("ijk,hnip,hnjo,vmkl,pol->vm", VVV_sink[t_sink], peram_u[t_sink], CG5_peram_d_CG5[t_sink], peram_u[t_sink], VVV_source)- \
        contract("ijk,hmil,hnjo,vnkp,pol->vm", VVV_sink[t_sink], peram_u[t_sink], CG5_peram_d_CG5[t_sink], peram_u[t_sink], VVV_source)

    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()
    
    del peram_u
    
    ed0 = time.time()
    print("****************all complete for t_source %d, time used: %.3f s****************" %(t_source, ed0-st0))

corr_proton_matrix_fw = contract("mv,xyvm->xy",pplus,corr_proton_matrix)
corr_proton_matrix_re = contract("mv,xyvm->xy",pnega,corr_proton_matrix)

# corr_proton_matrix_fw=cp.asnumpy(corr_proton_matrix_fw)  
# corr_proton_matrix_re=cp.asnumpy(corr_proton_matrix_re)  #Returns an array on the host memory from an arbitrary source array.

for t_source in range(0,Nt):
    for t_sink in range(0,Nt):
      
      if(t_sink<t_source):
        corr_proton_matrix_fw[t_source,t_sink]=-1.0*corr_proton_matrix_fw[t_source,t_sink]
      if(t_sink>t_source):
        corr_proton_matrix_re[t_source,t_sink]=-1.0*corr_proton_matrix_re[t_source,t_sink]
        
        
      corr_proton_plus[0,(t_sink-t_source+Nt)%Nt] = corr_proton_plus[0,(t_sink-t_source+Nt)%Nt] + corr_proton_matrix_fw[t_source,t_sink]
      corr_proton_nega[0,(t_sink-t_source+Nt)%Nt] = corr_proton_nega[0,(t_sink-t_source+Nt)%Nt] + corr_proton_matrix_re[t_source,t_sink]
        


write_data_ascii(corr_proton_plus, Nt, Nx, "%s/corr_uud_plus_Px%sPy%sPz%s_conf%s_cupy.dat"%(corr_nucl_dir,Px,Py,Pz,conf_id))
write_data_ascii(corr_proton_nega, Nt, Nx, "%s/corr_uud_nega_Px%sPy%sPz%s_conf%s_cupy.dat"%(corr_nucl_dir,Px,Py,Pz,conf_id))

print("succeed")