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
from lattice import GaugeFieldIldg
print("Job started")
Nc=3
Nd=4
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
    
  if(tmp[0]=='Nev_max'):  #number of eigenvectors used in contraction
    Nev_max=int(tmp[1])
  if(tmp[0]=='ENV_gap'):
    ENV_gap=int(tmp[1])
  if(tmp[0]=='ENV_stare'):
    ENV_stare=int(tmp[1])
  if(tmp[0]=='ENV_end'):
    ENV_end=int(tmp[1])
    
  if(tmp[0]=='Px'):
    Px=int(tmp[1])
  if(tmp[0]=='Py'):
    Py=int(tmp[1])
  if(tmp[0]=='Pz'):
    Pz=int(tmp[1])
    
  if(tmp[0]=='t_sep'):
    t_sep=int(tmp[1])
  if(tmp[0]=='link_dir'):
    link_dir=tmp[1]
  if(tmp[0]=='link_max'):
    link_max=int(tmp[1])
    
  if(tmp[0]=='corr_2pt'):
    corr_2pt=int(tmp[1])
  if(tmp[0]=='corr_3pt'):
    corr_3pt=int(tmp[1])
  if(tmp[0]=='part_of_3pt'):
    part_of_3pt=int(tmp[1])
    
  if(tmp[0]=='peram_u_dir'):
    peram_u_dir=tmp[1]
  if(tmp[0]=='eigen_dir'):
    eig_dir=tmp[1]
  if(tmp[0]=='conf_dir'):
    conf_dir=tmp[1]
  if(tmp[0]=='corr_dir'):
    corr_dir=tmp[1]
Mom=np.array([Pz,Py,Px])
# P matrix
pplus=(gamma(0)+gamma(4))/2
pnega=(gamma(0)-gamma(4))/2
G5Gx_sink = gamma(0)
GxG5_source = gamma(0)
#read in gauge link
gauge_field = GaugeFieldIldg(f"{conf_dir}", ".lime", [Nt, Nx, Nx, Nx, Nd, Nc, Nc]) # (Nt, Nz, Ny, Nx, 4(方向 x y z t), 3, 3)
gauge_link = cp.asarray(gauge_field.load(str(conf_id))[:,:,:,:,2,:,:])
# read in the eigenvector and calculate the mom sink
st_vdv = time.time()           
if link_dir=='0':
  VdV_sink_link = cp.zeros((Nt, Nev_max, Nev_max), dtype=cp.complex128)
  VdV_sink_link[:]=cp.identity(Nev_max)
else :
  VdV_sink_link = cp.zeros((Nt, 2*link_max+1, Nev_max, Nev_max), dtype=cp.complex128)
  for ti in range(0,Nt):
    VdV_sink_link[ti] = VdV_sink_t_link(gauge_link[ti],eig_dir, ti, Nev_max, conf_id, Nx, link_dir, link_max)
ed_vdv = time.time()                                                        # time_end
print("VdV_link done, time used: %.3f s" %(ed_vdv-st_vdv))                       # output the time of use 


st_vvv = time.time()   
VVV_sink = cp.zeros((Nt, Nev_max, Nev_max, Nev_max), dtype=cp.complex128)
VVV_source = cp.zeros((Nt, Nev_max, Nev_max, Nev_max), dtype=cp.complex128)
phase_exp_VVV = phase_exp_3pt(Mom, Nx)
for t in range(0,Nt):
  VVV_sink[t] = Mom_VVV_sink_t(eig_dir, t, Nev_max, conf_id, Nx, phase_exp_VVV, '0', link_max)
  VVV_source[t] = cp.conj(VVV_sink[t].T)
ed_vvv = time.time()                                                        # time_end
print("VVV_sink, VVV_source done, time used: %.3f s" %(ed_vvv-st_vvv))                       # output the time of use 

# read in the peram 
st_peram = time.time()
peram_u = cp.zeros((Nt,Nt,4,4,Nev_max,Nev_max),dtype=cp.complex128)
CG5_peram_d_CG5 = cp.zeros((Nt,Nt,4,4,Nev_max,Nev_max),dtype=cp.complex128)
for t_source in range(Nt):
  peram_u[t_source]=readin_peram_device(peram_u_dir, conf_id, Nt, Nev_max, t_source)
  CG5_peram_d_CG5[t_source]=contract("gh,thkbe,jk->tgjbe", gamma(7), peram_u[t_source], gamma(7))
ed_peram = time.time()
print("complete read in the peram, time used %.3f"%(ed_peram - st_peram))

# the main caculation part
for ENV in range(ENV_stare, ENV_end+1, ENV_gap):
  st_ENV = time.time()
  
  corr_proton_2pt_matrix=cp.zeros((Nt,Nt,4,4), dtype=cp.complex128)
  proton_3pt_matrix_U = cp.zeros((Nt,Nt), dtype=cp.complex128)
  proton_3pt_matrix_D = cp.zeros((Nt,Nt), dtype=cp.complex128)
  corr_proton_3pt_matrix_all = cp.zeros((2*link_max+1,6,Nt,Nt,4,4), dtype=cp.complex128)

  for t_source in range(Nt):
    st0 = time.time()
    
    # Pion 2pt loop over t_source:
    if(corr_2pt==1):
      for t_sink in range(0,Nt):                # il,be,ag,bcij,eflk,cf,jk->t

        corr_proton_2pt_matrix[t_source,t_sink] = \
          contract("ijk,hnip,hnjo,vmkl,pol->vm", VVV_sink[t_sink,0:ENV,0:ENV,0:ENV], peram_u[t_source,t_sink,:,:,0:ENV,0:ENV], CG5_peram_d_CG5[t_source,t_sink,:,:,0:ENV,0:ENV], peram_u[t_source,t_sink,:,:,0:ENV,0:ENV], VVV_source[t_source,0:ENV,0:ENV,0:ENV])- \
          contract("ijk,hmil,hnjo,vnkp,pol->vm", VVV_sink[t_sink,0:ENV,0:ENV,0:ENV], peram_u[t_source,t_sink,:,:,0:ENV,0:ENV], CG5_peram_d_CG5[t_source,t_sink,:,:,0:ENV,0:ENV], peram_u[t_source,t_sink,:,:,0:ENV,0:ENV], VVV_source[t_source,0:ENV,0:ENV,0:ENV])
   
    ed_2pt = time.time()
    print("all complete of 2pt for t_source %d, time used: %.3f s" %(t_source, ed_2pt-st0))
    
    # Pion 3pt loop over t_source:
    if corr_3pt==1:
      for link_indx in range(2*link_max+1):
        for t_curr in range(0,Nt):
          
          corr_proton_3pt_matrix_all[link_indx,0,t_source,t_curr] = \
              contract("ijk,hnjo,hnip,vgka,gq,ab,qmbl,pol->vm", VVV_sink[(t_sep+t_source)%Nt,0:ENV,0:ENV,0:ENV], 
                      CG5_peram_d_CG5[t_source,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], 
                      peram_u[t_source,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], 
                      peram_u[(t_curr+t_source)%Nt,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], gamma(4), VdV_sink_link[(t_curr+t_source)%Nt,link_indx,0:ENV,0:ENV], peram_u[t_source,(t_curr+t_source)%Nt,:,:,0:ENV,0:ENV], 
                      VVV_source[t_source,0:ENV,0:ENV,0:ENV])
              
          corr_proton_3pt_matrix_all[link_indx,1,t_source,t_curr] = \
              contract("ijk,hnjo,hmil,vgka,gq,ab,qnbp,pol->vm", VVV_sink[(t_sep+t_source)%Nt,0:ENV,0:ENV,0:ENV], 
                      CG5_peram_d_CG5[t_source,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], 
                      peram_u[t_source,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], 
                      peram_u[(t_curr+t_source)%Nt,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], gamma(4), VdV_sink_link[(t_curr+t_source)%Nt,link_indx,0:ENV,0:ENV], peram_u[t_source,(t_curr+t_source)%Nt,:,:,0:ENV,0:ENV], 
                      VVV_source[t_source,0:ENV,0:ENV,0:ENV])
              
          corr_proton_3pt_matrix_all[link_indx,2,t_source,t_curr] = \
              contract("ijk,hnjo,hgia,gq,ab,qnbp,vmkl,pol->vm", VVV_sink[(t_sep+t_source)%Nt,0:ENV,0:ENV,0:ENV], 
                      CG5_peram_d_CG5[t_source,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], 
                      peram_u[(t_curr+t_source)%Nt,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], gamma(4), VdV_sink_link[(t_curr+t_source)%Nt,link_indx,0:ENV,0:ENV], peram_u[t_source,(t_curr+t_source)%Nt,:,:,0:ENV,0:ENV], 
                      peram_u[t_source,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], 
                      VVV_source[t_source,0:ENV,0:ENV,0:ENV])
              
          corr_proton_3pt_matrix_all[link_indx,3,t_source,t_curr] = \
              contract("ijk,hnjo,hgia,gq,ab,qmbl,vnkp,pol->vm", VVV_sink[(t_sep+t_source)%Nt,0:ENV,0:ENV,0:ENV], 
                      CG5_peram_d_CG5[t_source,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], 
                      peram_u[(t_curr+t_source)%Nt,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], gamma(4), VdV_sink_link[(t_curr+t_source)%Nt,link_indx,0:ENV,0:ENV], peram_u[t_source,(t_curr+t_source)%Nt,:,:,0:ENV,0:ENV], 
                      peram_u[t_source,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], 
                      VVV_source[t_source,0:ENV,0:ENV,0:ENV])
              
          corr_proton_3pt_matrix_all[link_indx,4,t_source,t_curr] = \
            contract("ijk,hnip,hc,cgja,gq,ab,qfbo,nf,vmkl,pol->vm", VVV_sink[(t_sep+t_source)%Nt,0:ENV,0:ENV,0:ENV], 
                    peram_u[t_source,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], 
                    gamma(7),peram_u[(t_curr+t_source)%Nt,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], gamma(4), VdV_sink_link[(t_curr+t_source)%Nt,link_indx,0:ENV,0:ENV], peram_u[t_source,(t_curr+t_source)%Nt,:,:,0:ENV,0:ENV],gamma(7),
                    peram_u[t_source,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], 
                    VVV_source[t_source,0:ENV,0:ENV,0:ENV])
            
          corr_proton_3pt_matrix_all[link_indx,5,t_source,t_curr] = \
            contract("ijk,hmil,hc,cgja,gq,ab,qfbo,nf,vnkp,pol->vm", VVV_sink[(t_sep+t_source)%Nt,0:ENV,0:ENV,0:ENV], 
                    peram_u[t_source,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], 
                    gamma(7),peram_u[(t_curr+t_source)%Nt,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], gamma(4), VdV_sink_link[(t_curr+t_source)%Nt,link_indx,0:ENV,0:ENV], peram_u[t_source,(t_curr+t_source)%Nt,:,:,0:ENV,0:ENV],gamma(7),
                    peram_u[t_source,(t_sep+t_source)%Nt,:,:,0:ENV,0:ENV], 
                    VVV_source[t_source,0:ENV,0:ENV,0:ENV])
          
    cp._default_memory_pool.free_all_blocks()
    cp.cuda.Device().synchronize()
    ed0 = time.time()
    print("all complete for t_source %d, time used: %.3f s" %(t_source, ed0-st0))
    print("**********************************************************************")
  
  st_pp = time.time()
  # P matrix * corr
  if corr_2pt==1:
    corr_proton_2pt_matrix_fw = contract("vm,xyvm->xy",pplus,-1.0*corr_proton_2pt_matrix)
    corr_proton_2pt_matrix_re = contract("vm,xyvm->xy",pnega,-1.0*corr_proton_2pt_matrix)
  if corr_3pt==1:
    corr_proton_3pt_matrix_fw = contract("vm,lnxyvm->lnxy",pplus,corr_proton_3pt_matrix_all)
  ed_pp = time.time()
  print("P matrix contracted use time %.3f"%(ed_pp-st_pp))
 
  # translate type to asnumpy
  if corr_2pt==1:
    corr_proton_2pt_matrix_fw = cp.asnumpy(corr_proton_2pt_matrix_fw)
  if corr_3pt==1:
    corr_proton_3pt_matrix_fw = cp.asnumpy(corr_proton_3pt_matrix_fw)
    proton_3pt_matrix_U = -1.0*(corr_proton_3pt_matrix_fw[:,0] - corr_proton_3pt_matrix_fw[:,1] + corr_proton_3pt_matrix_fw[:,2] - corr_proton_3pt_matrix_fw[:,3])
    proton_3pt_matrix_D = -1.0*(corr_proton_3pt_matrix_fw[:,4] - corr_proton_3pt_matrix_fw[:,5])
    if part_of_3pt==0:
      del corr_proton_3pt_matrix_fw

  proton_2pt_plus = np.zeros((1,Nt),dtype=np.complex128)
  
  st_sum = time.time()
  #Returns an array on the host memory from an arbitrary source array.
  if corr_2pt==1:
    for t_source in range(0,Nt):
      for t_sink in range(0,Nt):
        proton_2pt_plus[0,(t_sink-t_source+Nt)%Nt] = proton_2pt_plus[0,(t_sink-t_source+Nt)%Nt] + corr_proton_2pt_matrix_fw[t_source,t_sink]
  del corr_proton_2pt_matrix_fw
  if corr_3pt==1:          
    proton_3pt_U = np.sum(proton_3pt_matrix_U,axis=1).reshape(2*link_max+1,1,Nt)
    proton_3pt_D = np.sum(proton_3pt_matrix_D,axis=1).reshape(2*link_max+1,1,Nt)
    del proton_3pt_matrix_U
    del proton_3pt_matrix_D
    if part_of_3pt==1:
      proton_3pt = np.sum(corr_proton_3pt_matrix_fw,axis=2).reshape(2*link_max+1,6,1,Nt)
      del corr_proton_3pt_matrix_fw
  ed_sum = time.time()
  print("loop over source t use time %.3f"%(ed_sum-st_sum))
  # save the data
  if corr_2pt==1:
    write_data_ascii(proton_2pt_plus, Nt, Nx, "%s/ENV_%d_%dx%d/corr_uuu_conf%s_2pt.dat"%(corr_dir,ENV,Nx,Nt,conf_id))
    del proton_2pt_plus
  if corr_3pt==1:
    for link_indx_save in range(2*link_max+1):
      write_data_ascii(proton_3pt_U[link_indx_save], Nt, Nx, "%s/ENV_%d/corr_uuu_conf%s_gamma%d_3pt_tseq%d_link%d_U.dat"%(corr_dir,ENV,Nx,Nt,conf_id,4,t_sep,(link_indx_save-link_max)))
      write_data_ascii(proton_3pt_D[link_indx_save], Nt, Nx, "%s/ENV_%d/corr_uuu_conf%s_gamma%d_3pt_tseq%d_link%d_D.dat"%(corr_dir,ENV,Nx,Nt,conf_id,4,t_sep,(link_indx_save-link_max)))
    del proton_3pt_U
    del proton_3pt_D
    if part_of_3pt==1:
      for i in range(6):
        for link_indx_save in range(2*link_max+1):
          if i<=3:flavour='U'
          else:flavour='D'
          write_data_ascii(proton_3pt[link_indx_save,i], Nt, Nx, "%s/ENV_%d_%dx%d/corr_uuu_conf%s_gamma%d_3pt_tseq%d_link%d_%s%d.dat"%(corr_dir,ENV,Nx,Nt,conf_id,4,t_sep,(link_indx_save-link_max),flavour,i))
          del proton_3pt

    
  ed_ENV = time.time()
  print("****************complete for ENV %d, time used: %.3f s****************" %(ENV, ed_ENV-st_ENV))
  
print("all caculations are succeed")
