#!/beegfs/home/liuming/software/install/python/bin/python3

#test file to check whether first term has problem
import numpy as np
import cupy as cp
import math
import os
import sys as sy
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
  if(tmp[0]=='link_dir'):
    link_dir=tmp[1]
  if(tmp[0]=='link_max'):
    link_max=tmp[1]
  if(tmp[0]=='peram_u_dir'):
    peram_u_dir=tmp[1]
  if(tmp[0]=='eigen_dir'):
    eig_dir=tmp[1]
  if(tmp[0]=='corr_nucl_dir'):
    corr_nucl_dir=tmp[1]
  if(tmp[0]=='VVV_dir'):
    VVV_dir=tmp[1]

    
Mom=np.array([Pz,Py,Px])
# P matrix
pplus=(gamma(0)+gamma(4))/2
pnega=(gamma(0)-gamma(4))/2

# read in the eigenvector and calculate the mom sink
st_vvv = time.time()           
if (link_dir=='0'):
  VVV_sink = cp.zeros((Nt, Nev1, Nev1, Nev1), dtype=complex)
  VdV_sink = cp.zeros((Nt, Nev1, Nev1), dtype=complex)
else:
  VVV_sink = cp.zeros((Nt, Nx, Nev1, Nev1, Nev1), dtype=complex)
  VdV_sink = cp.zeros((Nt, Nx, Nev1, Nev1), dtype=complex)
  
phase_exp_VVV = phase_exp_3pt(Mom, Nx)
phase_exp_VV  = phase_exp_2pt(Mom, Nx)


for t in range(0,Nt):
  VVV_sink[t] = Mom_VVV_sink_t(eig_dir, t, Nev1, conf_id, Nx, phase_exp_VVV, link_dir, link_max)

if Px==0 and Py==0 and Pz==0 :
    for t in range(0,Nt):
        VdV_sink[t]=cp.identity(Nev1)
else :
  for ti in range(0,Nt):
    VdV_sink[ti] = Mom_VdV_sink_t(eig_dir, ti, Nev1, conf_id, Nx, phase_exp_VV, link_dir, link_max)

ed_vvv = time.time()  
print("VVV and VdV done, time used: %.3f s" %(ed_vvv-st_vvv))
                 
corr_2pt = 1 # if corr_2pt==0 don't calculate corr_2pt else if corr_2pt==1 do it
corr_3pt = 1 # if corr_3pt==0 don't calculate corr_3pt else if corr_3pt==1 do it
part_3pt = 0 # if ==0 don't save the part of 3pt (total number of 3pt is 4 ) else if ==1 save it
t_sep = 8 #

corr_proton_3pt_matrix_1=cp.zeros((1,t_sep+1,4,4), dtype=complex)
corr_proton_3pt_matrix_2=cp.zeros((1,t_sep+1,4,4), dtype=complex)
corr_proton_3pt_matrix_3=cp.zeros((1,t_sep+1,4,4), dtype=complex)
corr_proton_3pt_matrix_4=cp.zeros((1,t_sep+1,4,4), dtype=complex)
corr_proton_3pt_matrix_5=cp.zeros((1,t_sep+1,4,4), dtype=complex)
corr_proton_3pt_matrix_6=cp.zeros((1,t_sep+1,4,4), dtype=complex)
corr_proton_2pt_matrix=cp.zeros((Nt,Nt,4,4), dtype=complex)

for gi in range(4,5):

  for t_source in range(0,Nt):
      st0 = time.time()

      VVV_source = cp.conj(VVV_sink[t_source].T)  # [t_source].T
      peram_u_source=readin_peram_device(peram_u_dir, conf_id, Nt, Nev1 , t_source)
      CG5_peram_d_CG5=contract("gh,thkbe,jk->tgjbe", gamma(7), peram_u_source, gamma(7))
      
      if (corr_2pt==1):
        for t_sink in range(0,Nt):
          corr_proton_2pt_matrix[t_source,t_sink] = \
            contract("ijk,hnip,hnjo,vmkl,pol->vm", VVV_sink[t_sink], peram_u_source[t_sink], CG5_peram_d_CG5[t_sink], peram_u_source[t_sink], VVV_source)- \
            contract("ijk,hmil,hnjo,vnkp,pol->vm", VVV_sink[t_sink], peram_u_source[t_sink], CG5_peram_d_CG5[t_sink], peram_u_source[t_sink], VVV_source)
      
      if (t_source==0 and corr_3pt==1):
        for t_curr in range(0,t_sep+1):
          peram_u_curr=readin_peram_device(peram_u_dir, conf_id, Nt, Nev1 , t_curr)
          corr_proton_3pt_matrix_1[t_source,t_curr] = \
              contract("ijk,hnjo,hnip,vgka,gq,ab,qmbl,pol->vm", VVV_sink[t_sep], CG5_peram_d_CG5[t_sep], peram_u_source[t_sep], peram_u_curr[t_sep], gamma(gi), VdV_sink[t_curr], peram_u_source[t_curr], VVV_source)
          corr_proton_3pt_matrix_2[t_source,t_curr] = \
              contract("ijk,hnjo,hmil,vgka,gq,ab,qnbp,pol->vm", VVV_sink[t_sep], CG5_peram_d_CG5[t_sep], peram_u_source[t_sep], peram_u_curr[t_sep], gamma(gi), VdV_sink[t_curr], peram_u_source[t_curr], VVV_source)
          corr_proton_3pt_matrix_3[t_source,t_curr] = \
              contract("ijk,hnjo,hgia,gq,ab,qnbp,vmkl,pol->vm", VVV_sink[t_sep], CG5_peram_d_CG5[t_sep], peram_u_curr[t_sep], gamma(gi), VdV_sink[t_curr], peram_u_source[t_curr], peram_u_source[t_sep], VVV_source)
          corr_proton_3pt_matrix_4[t_source,t_curr] = \
              contract("ijk,hnjo,hgia,gq,ab,qmbl,vnkp,pol->vm", VVV_sink[t_sep], CG5_peram_d_CG5[t_sep], peram_u_curr[t_sep], gamma(gi), VdV_sink[t_curr], peram_u_source[t_curr], peram_u_source[t_sep], VVV_source)
          corr_proton_3pt_matrix_5[t_source,t_curr] = \
            contract("ijk,hnip,hc,cgja,gq,ab,qfbo,nf,vmkl,pol->vm", VVV_sink[t_sep], peram_u_source[t_sep], gamma(7), peram_u_curr[t_sep], gamma(gi), VdV_sink[t_curr], peram_u_source[t_curr], gamma(7), peram_u_source[t_sep], VVV_source)
          corr_proton_3pt_matrix_6[t_source,t_curr] = \
            contract("ijk,hmil,hc,cgja,gq,ab,qfbo,nf,vnkp,pol->vm", VVV_sink[t_sep], peram_u_source[t_sep], gamma(7), peram_u_curr[t_sep], gamma(gi), VdV_sink[t_curr], peram_u_source[t_curr], gamma(7), peram_u_source[t_sep], VVV_source)
          # corr_proton_3pt_matrix_1[t_source,t_curr] = \
          #   contract("ijk,vnip,nhjo,hgka,gq,ab,qmbl,pol->vm", VVV_sink[t_sep], CG5_peram_d_CG5[t_sep], peram_u_source[t_sep], peram_u_curr[t_sep], gamma(gi), VdV_sink[t_curr], peram_u_source[t_curr], VVV_source)
          # corr_proton_3pt_matrix_2[t_source,t_curr] = \
          #   contract("ijk,vnip,nmjl,hgka,gq,ab,qhbo,pol->vm", VVV_sink[t_sep], CG5_peram_d_CG5[t_sep], peram_u_source[t_sep], peram_u_curr[t_sep], gamma(gi), VdV_sink[t_curr], peram_u_source[t_curr], VVV_source)
          # corr_proton_3pt_matrix_3[t_source,t_curr] = \
          #   contract("ijk,vnip,ngja,gq,ab,qhbo,hmkl,pol->vm", VVV_sink[t_sep], CG5_peram_d_CG5[t_sep], peram_u_curr[t_sep], gamma(gi), VdV_sink[t_curr], peram_u_source[t_curr], peram_u_source[t_sep], VVV_source)
          # corr_proton_3pt_matrix_4[t_source,t_curr] = \
          #   contract("ijk,vnip,ngja,gq,ab,qmbl,hhko,pol->vm", VVV_sink[t_sep], CG5_peram_d_CG5[t_sep], peram_u_curr[t_sep], gamma(gi), VdV_sink[t_curr], peram_u_source[t_curr], peram_u_source[t_sep], VVV_source)
          del peram_u_curr
      cp._default_memory_pool.free_all_blocks()
      cp.cuda.Device().synchronize()
      
      del peram_u_source

      
      ed0 = time.time()
      print("****************all complete for t_source %d, time used: %.3f s****************" %(t_source, ed0-st0))
  
  st0 = time.time()
  # P matrix * corr
  if (corr_2pt==1):
    corr_proton_2pt_matrix_fw = contract("vm,xyvm->xy",pplus,-1.0*corr_proton_2pt_matrix)
    corr_proton_2pt_matrix_re = contract("vm,xyvm->xy",pnega,-1.0*corr_proton_2pt_matrix)
  if (corr_3pt==1):
    # U  
    corr_proton_3pt_matrix_fw_1 = contract("vm,xyvm->xy",pplus,corr_proton_3pt_matrix_1)
    corr_proton_3pt_matrix_fw_2 = contract("vm,xyvm->xy",pplus,corr_proton_3pt_matrix_2)
    corr_proton_3pt_matrix_fw_3 = contract("vm,xyvm->xy",pplus,corr_proton_3pt_matrix_3)
    corr_proton_3pt_matrix_fw_4 = contract("vm,xyvm->xy",pplus,corr_proton_3pt_matrix_4)
    # D
    corr_proton_3pt_matrix_fw_5 = contract("vm,xyvm->xy",pplus,corr_proton_3pt_matrix_5)
    corr_proton_3pt_matrix_fw_6 = contract("vm,xyvm->xy",pplus,corr_proton_3pt_matrix_6)
  ed0 = time.time()
  print("****************complete for PP matrix, time used: %.3f s****************" %(ed0-st0))

  # translate type to asnumpy
  st0 = time.time()
  corr_proton_2pt_matrix_fw = cp.asnumpy(corr_proton_2pt_matrix_fw)
  corr_proton_3pt_matrix_fw_1 = cp.asnumpy(corr_proton_3pt_matrix_fw_1)
  corr_proton_3pt_matrix_fw_2 = cp.asnumpy(corr_proton_3pt_matrix_fw_2)
  corr_proton_3pt_matrix_fw_3 = cp.asnumpy(corr_proton_3pt_matrix_fw_3)
  corr_proton_3pt_matrix_fw_4 = cp.asnumpy(corr_proton_3pt_matrix_fw_4)
  corr_proton_3pt_matrix_fw_5 = cp.asnumpy(corr_proton_3pt_matrix_fw_5)
  corr_proton_3pt_matrix_fw_6 = cp.asnumpy(corr_proton_3pt_matrix_fw_6)
  corr_proton_3pt_matrix_fw_U = -1.0*(corr_proton_3pt_matrix_fw_1 - corr_proton_3pt_matrix_fw_2 + corr_proton_3pt_matrix_fw_3 - corr_proton_3pt_matrix_fw_4)
  corr_proton_3pt_matrix_fw_D = -1.0*(corr_proton_3pt_matrix_fw_5 - corr_proton_3pt_matrix_fw_6)
  corr_proton_2pt_plus = np.zeros((1,Nt), dtype=complex)
  for t_source in range(0,Nt):
      for t_sink in range(0,Nt):
        
        if (corr_2pt==1):
          if(t_sink<t_source):
            corr_proton_2pt_matrix_fw[t_source,t_sink]=-1.0*corr_proton_2pt_matrix_fw[t_source,t_sink]
          if(t_sink>t_source):
            corr_proton_2pt_matrix_re[t_source,t_sink]=-1.0*corr_proton_2pt_matrix_re[t_source,t_sink]
          corr_proton_2pt_plus[0,(t_sink-t_source+Nt)%Nt] = corr_proton_2pt_plus[0,(t_sink-t_source+Nt)%Nt] + corr_proton_2pt_matrix_fw[t_source,t_sink]
          # corr_proton_2pt_nega[0,(t_sink-t_source+Nt)%Nt] = corr_proton_2pt_nega[0,(t_sink-t_source+Nt)%Nt] + corr_proton_2pt_matrix_re[t_source,t_sink]
  ed0 = time.time()
  print("****************complete for trace the matrix, time used: %.3f s****************" %(ed0-st0))
  
  # save the data
  if (corr_2pt==1):
    write_data_ascii(corr_proton_2pt_plus, Nt, Nx, "%s/corr_uud_plus_Px%sPy%sPz%s_conf%s_2pt_1_cupy.dat"%(corr_nucl_dir,Px,Py,Pz,conf_id))
  
  if (part_3pt==1):
    write_data_ascii(corr_proton_3pt_matrix_fw_1, t_sep+1, Nx, "%s/corr_uud_plus_Px%sPy%sPz%s_conf%s_gamma%d_3pt_tseq%d_1_cupy.dat"%(corr_nucl_dir,Px,Py,Pz,conf_id,gi,t_sep))
    write_data_ascii(corr_proton_3pt_matrix_fw_2, t_sep+1, Nx, "%s/corr_uud_plus_Px%sPy%sPz%s_conf%s_gamma%d_3pt_tseq%d_2_cupy.dat"%(corr_nucl_dir,Px,Py,Pz,conf_id,gi,t_sep))
    write_data_ascii(corr_proton_3pt_matrix_fw_3, t_sep+1, Nx, "%s/corr_uud_plus_Px%sPy%sPz%s_conf%s_gamma%d_3pt_tseq%d_3_cupy.dat"%(corr_nucl_dir,Px,Py,Pz,conf_id,gi,t_sep))
    write_data_ascii(corr_proton_3pt_matrix_fw_4, t_sep+1, Nx, "%s/corr_uud_plus_Px%sPy%sPz%s_conf%s_gamma%d_3pt_tseq%d_4_cupy.dat"%(corr_nucl_dir,Px,Py,Pz,conf_id,gi,t_sep))
    write_data_ascii(corr_proton_3pt_matrix_fw_5, t_sep+1, Nx, "%s/corr_uud_plus_Px%sPy%sPz%s_conf%s_gamma%d_3pt_tseq%d_5_cupy.dat"%(corr_nucl_dir,Px,Py,Pz,conf_id,gi,t_sep))
    write_data_ascii(corr_proton_3pt_matrix_fw_6, t_sep+1, Nx, "%s/corr_uud_plus_Px%sPy%sPz%s_conf%s_gamma%d_3pt_tseq%d_6_cupy.dat"%(corr_nucl_dir,Px,Py,Pz,conf_id,gi,t_sep))
  
  if (corr_3pt==1):
    write_data_ascii(corr_proton_3pt_matrix_fw_U, t_sep+1, Nx, "%s/corr_uud_plus_Px%sPy%sPz%s_conf%s_gamma%d_3pt_tseq%d_U_cupy.dat"%(corr_nucl_dir,Px,Py,Pz,conf_id,gi,t_sep))
    write_data_ascii(corr_proton_3pt_matrix_fw_D, t_sep+1, Nx, "%s/corr_uud_plus_Px%sPy%sPz%s_conf%s_gamma%d_3pt_tseq%d_D_cupy.dat"%(corr_nucl_dir,Px,Py,Pz,conf_id,gi,t_sep))
print("succeed") 