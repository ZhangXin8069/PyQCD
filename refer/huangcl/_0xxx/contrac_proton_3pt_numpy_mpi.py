#!/beegfs/home/liuming/software/install/python/bin/python3

#test file to check whether first term has problem
import numpy as np
import math
import os
import fileinput
from gamma_matrix_numpy_DR import *
from input_output_4_numpy import *
from opt_einsum import contract
import time
from mpi4py import MPI
print("Job started")
Nc=3
# ------------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
rank_size = comm.Get_size()

infile=fileinput.input()
read_io = read_input(infile)

t_rank = read_io.Nt / rank_size
if type(t_rank) != int:
  print('error: Nt / mpi_rank is not int, it is %.3f'%(t_rank))
  exit()
Mom=np.array([read_io.Pz,read_io.Py,read_io.Px])
# P matrix
pplus=(gamma(0)+gamma(4))/2
pnega=(gamma(0)-gamma(4))/2
G5Gx_sink = gamma(0)
GxG5_source = gamma(0)
# read in the eigenvector and calculate the mom sink
st_vdv = time.time()           
if rank == 0:
  if read_io.link_dir=='0':
    VdV_sink_link = np.zeros((read_io.Nt, 1, read_io.Nev_max, read_io.Nev_max), dtype=np.complex128)
    for t in range(0,read_io.Nt):
        VdV_sink_link[t]=np.identity(read_io.Nev_max)
  else :
    VdV_sink_link = np.zeros((read_io.Nt, 2*read_io.link_max+1, read_io.Nev_max, read_io.Nev_max), dtype=np.complex128)
    for ti in range(0,read_io.Nt):
      VdV_sink_link[ti] = Mom_VdV_sink_t(read_io.eig_dir, ti, read_io.Nev_max, read_io.conf_id, read_io.Nx, read_io.link_dir, read_io.link_max)
  ed_vdv = time.time()                                                        # time_end
  print("VdV_link done, time used: %.3f s" %(ed_vdv-st_vdv))                       # output the time of use 


  st_vvv = time.time()   
  VVV_sink = np.zeros((read_io.Nt, read_io.Nev_max, read_io.Nev_max, read_io.Nev_max), dtype=np.complex128)
  VVV_source = np.zeros((read_io.Nt, read_io.Nev_max, read_io.Nev_max, read_io.Nev_max), dtype=np.complex128)
  phase_exp_VVV = phase_exp_3pt(Mom, read_io.Nx)
  for t in range(0,read_io.Nt):
    VVV_sink[t] = Mom_VVV_sink_t(read_io.eig_dir, t, read_io.Nev_max, read_io.conf_id, read_io.Nx, phase_exp_VVV, read_io.link_dir, read_io.link_max)
    VVV_source[t] = np.conj(VVV_sink[t].T)
  ed_vvv = time.time()                                                        # time_end
  print("VVV_sink, VVV_source done, time used: %.3f s" %(ed_vvv-st_vvv))                       # output the time of use 

if rank == 1:
  # read in the peram 
  st_peram = time.time()
  peram_u = np.zeros((read_io.Nt,read_io.Nt,4,4,read_io.Nev_max,read_io.Nev_max),dtype=np.complex128)
  CG5_peram_d_CG5 = np.zeros((read_io.Nt,read_io.Nt,4,4,read_io.Nev_max,read_io.Nev_max),dtype=np.complex128)
  for t_source in range(read_io.Nt):
    peram_u[t_source]=readin_peram_device(read_io.peram_u_dir, read_io.conf_id, read_io.Nt, read_io.Nev_max, t_source)
  ed_peram = time.time()
  print("complete read in the peram, time used %.3f"%(ed_peram - st_peram))

mpi_rece_vvv = np.zeros()

mpi_rece_peram = np.zeros((t_rank,read_io.Nt,4,4,read_io.Nev_max,read_io.Nev_max), dtype=np.complex128)
comm.scatter(peram_u, mpi_rece_peram, root=1)
# the main caculation part
for ENV in range(read_io.ENV_stare, read_io.ENV_end+1, read_io.ENV_gap):
  st_ENV = time.time()
  for link_indx in range(2*read_io.link_max+1):
    if rank == 0:
      corr_proton_2pt_matrix=np.zeros((read_io.Nt,read_io.Nt,4,4), dtype=np.complex128)
      proton_3pt_matrix_U = np.zeros((read_io.Nt,read_io.Nt), dtype=np.complex128)
      proton_3pt_matrix_D = np.zeros((read_io.Nt,read_io.Nt), dtype=np.complex128)
      corr_proton_3pt_matrix=np.zeros((read_io.Nt,read_io.Nt,4,4), dtype=np.complex128)
      
    for rank_indx in range(rank_size):
      if rank == rank_indx:
        
        for t_source in range(read_io.Nt):
          st0 = time.time()
          
          # Pion 2pt loop over t_source:
          if(read_io.corr_2pt==1):
            for t_sink in range(0,read_io.Nt):                # il,be,ag,bcij,eflk,cf,jk->t

              corr_proton_2pt_matrix[t_source,t_sink] = \
                contract("ijk,hnip,hnjo,vmkl,pol->vm", VVV_sink[t_sink,0:ENV,0:ENV,0:ENV], peram_u[t_source,t_sink,:,:,0:ENV,0:ENV], CG5_peram_d_CG5[t_source,t_sink,:,:,0:ENV,0:ENV], peram_u[t_source,t_sink,:,:,0:ENV,0:ENV], VVV_source[t_source,0:ENV,0:ENV,0:ENV])- \
                contract("ijk,hmil,hnjo,vnkp,pol->vm", VVV_sink[t_sink,0:ENV,0:ENV,0:ENV], peram_u[t_source,t_sink,:,:,0:ENV,0:ENV], CG5_peram_d_CG5[t_source,t_sink,:,:,0:ENV,0:ENV], peram_u[t_source,t_sink,:,:,0:ENV,0:ENV], VVV_source[t_source,0:ENV,0:ENV,0:ENV])
        
          ed_2pt = time.time()
          print("all complete of 2pt for t_source %d, time used: %.3f s" %(t_source, ed_2pt-st0))
          
          # Pion 3pt loop over t_source:
          if read_io.corr_3pt==1:
            for t_curr in range(0,read_io.Nt):
              current_contract[t_source,(t_curr+t_source)%read_io.Nt] = contract("vgka,gq,ab,qmbl->vmkl",
                peram_u[(t_curr+t_source)%read_io.Nt,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], gamma(4), VdV_sink_link[(t_curr+t_source)%read_io.Nt,link_indx,0:ENV,0:ENV], peram_u[t_source,(t_curr+t_source)%read_io.Nt,:,:,0:ENV,0:ENV])
              
              corr_proton_3pt_matrix_1[link_indx,t_source,t_curr] = \
                  contract("ijk,hc,cdjo,nd,hnip,vgka,gq,ab,qmbl,pol->vm", VVV_sink[(read_io.t_sep+t_source)%read_io.Nt,0:ENV,0:ENV,0:ENV], 
                          gamma(7), peram_u[t_source,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], gamma(7),
                          peram_u[t_source,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], 
                          peram_u[(t_curr+t_source)%read_io.Nt,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], gamma(4), VdV_sink_link[(t_curr+t_source)%read_io.Nt,link_indx,0:ENV,0:ENV], peram_u[t_source,(t_curr+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], 
                          VVV_source[t_source,0:ENV,0:ENV,0:ENV])
                  
              corr_proton_3pt_matrix_2[link_indx,t_source,t_curr] = \
                  contract("ijk,hc,cdjo,nd,hmil,vgka,gq,ab,qnbp,pol->vm", VVV_sink[(read_io.t_sep+t_source)%read_io.Nt,0:ENV,0:ENV,0:ENV], 
                          gamma(7), peram_u[t_source,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], gamma(7),
                          peram_u[t_source,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], 
                          peram_u[(t_curr+t_source)%read_io.Nt,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], gamma(4), VdV_sink_link[(t_curr+t_source)%read_io.Nt,link_indx,0:ENV,0:ENV], peram_u[t_source,(t_curr+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], 
                          VVV_source[t_source,0:ENV,0:ENV,0:ENV])
                  
              corr_proton_3pt_matrix_3[link_indx,t_source,t_curr] = \
                  contract("ijk,hc,cdjo,nd,hgia,gq,ab,qnbp,vmkl,pol->vm", VVV_sink[(read_io.t_sep+t_source)%read_io.Nt,0:ENV,0:ENV,0:ENV], 
                          gamma(7), peram_u[t_source,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], gamma(7),
                          peram_u[(t_curr+t_source)%read_io.Nt,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], gamma(4), VdV_sink_link[(t_curr+t_source)%read_io.Nt,link_indx,0:ENV,0:ENV], peram_u[t_source,(t_curr+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], 
                          peram_u[t_source,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], 
                          VVV_source[t_source,0:ENV,0:ENV,0:ENV])
                  
              corr_proton_3pt_matrix_4[link_indx,t_source,t_curr] = \
                  contract("ijk,hc,cdjo,nd,hgia,gq,ab,qmbl,vnkp,pol->vm", VVV_sink[(read_io.t_sep+t_source)%read_io.Nt,0:ENV,0:ENV,0:ENV], 
                          gamma(7), peram_u[t_source,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], gamma(7),
                          peram_u[(t_curr+t_source)%read_io.Nt,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], gamma(4), VdV_sink_link[(t_curr+t_source)%read_io.Nt,link_indx,0:ENV,0:ENV], peram_u[t_source,(t_curr+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], 
                          peram_u[t_source,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], 
                          VVV_source[t_source,0:ENV,0:ENV,0:ENV])
                  
              corr_proton_3pt_matrix_5[link_indx,t_source,t_curr] = \
                contract("ijk,hnip,hc,cgja,gq,ab,qfbo,nf,vmkl,pol->vm", VVV_sink[(read_io.t_sep+t_source)%read_io.Nt,0:ENV,0:ENV,0:ENV], 
                        peram_u[t_source,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], 
                        gamma(7),peram_u[(t_curr+t_source)%read_io.Nt,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], gamma(4), VdV_sink_link[(t_curr+t_source)%read_io.Nt,link_indx,0:ENV,0:ENV], peram_u[t_source,(t_curr+t_source)%read_io.Nt,:,:,0:ENV,0:ENV],gamma(7),
                        peram_u[t_source,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], 
                        VVV_source[t_source,0:ENV,0:ENV,0:ENV])

              corr_proton_3pt_matrix_6[link_indx,t_source,t_curr] = \
                contract("ijk,hmil,hc,cgja,gq,ab,qfbo,nf,vnkp,pol->vm", VVV_sink[(read_io.t_sep+t_source)%read_io.Nt,0:ENV,0:ENV,0:ENV], 
                        peram_u[t_source,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], 
                        gamma(7),peram_u[(t_curr+t_source)%read_io.Nt,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], gamma(4), VdV_sink_link[(t_curr+t_source)%read_io.Nt,link_indx,0:ENV,0:ENV], peram_u[t_source,(t_curr+t_source)%read_io.Nt,:,:,0:ENV,0:ENV],gamma(7),
                        peram_u[t_source,(read_io.t_sep+t_source)%read_io.Nt,:,:,0:ENV,0:ENV], 
                        VVV_source[t_source,0:ENV,0:ENV,0:ENV])
                
          ed0 = time.time()
          print("all complete for t_source %d, time used: %.3f s" %(t_source, ed0-st0))
          print("**********************************************************************")
      
    st_pp = time.time()
    # P matrix * corr
    if (read_io.corr_2pt==1):
      corr_proton_2pt_matrix_fw = contract("vm,xyvm->xy",pplus,-1.0*corr_proton_2pt_matrix)
      corr_proton_2pt_matrix_re = contract("vm,xyvm->xy",pnega,-1.0*corr_proton_2pt_matrix)
    if (read_io.corr_3pt==1):
      # U  
      corr_proton_3pt_matrix_fw_1 = contract("vm,xyvm->xy",pplus,corr_proton_3pt_matrix_1)
      corr_proton_3pt_matrix_fw_2 = contract("vm,xyvm->xy",pplus,corr_proton_3pt_matrix_2)
      corr_proton_3pt_matrix_fw_3 = contract("vm,xyvm->xy",pplus,corr_proton_3pt_matrix_3)
      corr_proton_3pt_matrix_fw_4 = contract("vm,xyvm->xy",pplus,corr_proton_3pt_matrix_4)
      # D
      corr_proton_3pt_matrix_fw_5 = contract("vm,xyvm->xy",pplus,corr_proton_3pt_matrix_5)
      corr_proton_3pt_matrix_fw_6 = contract("vm,xyvm->xy",pplus,corr_proton_3pt_matrix_6)
      
    del corr_proton_2pt_matrix
    del corr_proton_3pt_matrix_1
    del corr_proton_3pt_matrix_2
    del corr_proton_3pt_matrix_3
    del corr_proton_3pt_matrix_4
    del corr_proton_3pt_matrix_5
    del corr_proton_3pt_matrix_6
    
    ed_pp = time.time()
    print("P matrix contracted use time %.3f"%(ed_pp-st_pp))
  
    # translate type to asnumpy
    proton_3pt_matrix_U = -1.0*(corr_proton_3pt_matrix_fw_1 - corr_proton_3pt_matrix_fw_2 + corr_proton_3pt_matrix_fw_3 - corr_proton_3pt_matrix_fw_4)
    proton_3pt_matrix_D = -1.0*(corr_proton_3pt_matrix_fw_5 - corr_proton_3pt_matrix_fw_6)
  
    proton_2pt_plus = np.zeros((1,read_io.Nt),dtype=np.complex128)
    proton_3pt_U = np.zeros((1,read_io.Nt),dtype=np.complex128)
    proton_3pt_D = np.zeros((1,read_io.Nt),dtype=np.complex128)
    
    st_sum = time.time()
    #Returns an array on the host memory from an arbitrary source array.
    if read_io.corr_2pt==1:
      for t_source in range(0,read_io.Nt):
        for t_sink in range(0,read_io.Nt):
          proton_2pt_plus[0,(t_sink-t_source+read_io.Nt)%read_io.Nt] = proton_2pt_plus[0,(t_sink-t_source+read_io.Nt)%read_io.Nt] + corr_proton_2pt_matrix_fw[t_source,t_sink]
    if read_io.corr_3pt==1:          
      proton_3pt_U = np.sum(proton_3pt_matrix_U,axis=0).reshape(1,read_io.Nt)
      proton_3pt_D = np.sum(proton_3pt_matrix_D,axis=0).reshape(1,read_io.Nt)
      if read_io.part_of_3pt==1:
        proton_3pt_U1 = np.sum(corr_proton_3pt_matrix_fw_1,axis=0).reshape(1,read_io.Nt)
        proton_3pt_U2 = np.sum(corr_proton_3pt_matrix_fw_2,axis=0).reshape(1,read_io.Nt)
        proton_3pt_U3 = np.sum(corr_proton_3pt_matrix_fw_3,axis=0).reshape(1,read_io.Nt)
        proton_3pt_U4 = np.sum(corr_proton_3pt_matrix_fw_4,axis=0).reshape(1,read_io.Nt)
        
        proton_3pt_D1 = np.sum(corr_proton_3pt_matrix_fw_5,axis=0).reshape(1,read_io.Nt)
        proton_3pt_D2 = np.sum(corr_proton_3pt_matrix_fw_6,axis=0).reshape(1,read_io.Nt)
        
    del corr_proton_2pt_matrix_fw
    del corr_proton_3pt_matrix_fw_1
    del corr_proton_3pt_matrix_fw_2
    del corr_proton_3pt_matrix_fw_3
    del corr_proton_3pt_matrix_fw_4
    del corr_proton_3pt_matrix_fw_5
    del corr_proton_3pt_matrix_fw_6
    ed_sum = time.time()
    print("loop over source t use time %.3f"%(ed_sum-st_sum))

    # save the data
    if read_io.corr_2pt==1:
      write_data_ascii(proton_2pt_plus, read_io.Nt, read_io.Nx, "%s/ENV_%d_%dx%d/corr_uuu_conf%s_2pt.dat"%(read_io.corr_dir,ENV,read_io.Nx,read_io.Nt,read_io.conf_id))
    if read_io.corr_3pt==1:
      write_data_ascii(proton_3pt_U, read_io.Nt, read_io.Nx, "%s/ENV_%d_%dx%d/corr_uuu_conf%s_gamma%d_3pt_tseq%d_U.dat"%(read_io.corr_dir,ENV,read_io.Nx,read_io.Nt,read_io.conf_id,4,read_io.t_sep))
      write_data_ascii(proton_3pt_D, read_io.Nt, read_io.Nx, "%s/ENV_%d_%dx%d/corr_uuu_conf%s_gamma%d_3pt_tseq%d_D.dat"%(read_io.corr_dir,ENV,read_io.Nx,read_io.Nt,read_io.conf_id,4,read_io.t_sep))
      if read_io.part_of_3pt==1:
        write_data_ascii(proton_3pt_U1, read_io.Nt, read_io.Nx, "%s/ENV_%d_%dx%d/corr_uuu_conf%s_gamma%d_3pt_tseq%d_U1.dat"%(read_io.corr_dir,ENV,read_io.Nx,read_io.Nt,read_io.conf_id,4,read_io.t_sep))
        write_data_ascii(proton_3pt_U2, read_io.Nt, read_io.Nx, "%s/ENV_%d_%dx%d/corr_uuu_conf%s_gamma%d_3pt_tseq%d_U2.dat"%(read_io.corr_dir,ENV,read_io.Nx,read_io.Nt,read_io.conf_id,4,read_io.t_sep))
        write_data_ascii(proton_3pt_U3, read_io.Nt, read_io.Nx, "%s/ENV_%d_%dx%d/corr_uuu_conf%s_gamma%d_3pt_tseq%d_U3.dat"%(read_io.corr_dir,ENV,read_io.Nx,read_io.Nt,read_io.conf_id,4,read_io.t_sep))
        write_data_ascii(proton_3pt_U4, read_io.Nt, read_io.Nx, "%s/ENV_%d_%dx%d/corr_uuu_conf%s_gamma%d_3pt_tseq%d_U4.dat"%(read_io.corr_dir,ENV,read_io.Nx,read_io.Nt,read_io.conf_id,4,read_io.t_sep))
        write_data_ascii(proton_3pt_D1, read_io.Nt, read_io.Nx, "%s/ENV_%d_%dx%d/corr_uuu_conf%s_gamma%d_3pt_tseq%d_D1.dat"%(read_io.corr_dir,ENV,read_io.Nx,read_io.Nt,read_io.conf_id,4,read_io.t_sep))
        write_data_ascii(proton_3pt_D2, read_io.Nt, read_io.Nx, "%s/ENV_%d_%dx%d/corr_uuu_conf%s_gamma%d_3pt_tseq%d_D2.dat"%(read_io.corr_dir,ENV,read_io.Nx,read_io.Nt,read_io.conf_id,4,read_io.t_sep))
  
    
  ed_ENV = time.time()
  print("****************complete for ENV %d, time used: %.3f s****************" %(ENV, ed_ENV-st_ENV))
  
print("all caculations are succeed")
