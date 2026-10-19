import numpy as np
import cupy as cp

from opt_einsum import contract
import time

print("Job started!")

# ------------------------------------------------------------------------------

def readin_eigvecs(eig_dir, t, Nev1,conf_id, Nx):
    f=open("%s/eigvecs_t%03d_%s"%(eig_dir, t, conf_id),'rb')
    eigvecs=np.fromfile(f,dtype='f8')
    eigvecs_size=eigvecs.size
    Nev=int(eigvecs_size/(Nx*Nx*Nx*3*2))
    eigvecs=eigvecs.reshape(Nev,Nx*Nx*Nx,3,2)
    eigvecs=eigvecs[...,0]+eigvecs[...,1]*1j
    eigvecs=eigvecs[0:Nev1,:,:]
    return cp.asarray(eigvecs)



# ------------------------------------------------------------------------------
def phase_calc(Mom,Nx):
    phase_factor = np.zeros(Nx * Nx * Nx, dtype=complex)
    for z in range(0, Nx):
        for y in range(0, Nx):
            for x in range(0, Nx):
                Pos = np.array([z, y, x])
                phase_factor[z * Nx * Nx + y * Nx + x] = np.exp(
                    -np.dot(Mom, Pos) * 2 * np.pi * 1j / Nx
                )
    return cp.asarray(phase_factor)


# ------------------------------------------------------------------------------

def VVV_cal(eig_dir,Nx,Nt,Nev1,conf_id,Px,Py,Pz):
    VVV = cp.zeros((Nt, Nev1, Nev1, Nev1), dtype=complex)
    Mom = np.array([Pz, Py, Px])
    #print(Mom)
    for t in range(0, Nt):
        #st1 = time.time()
        try:
            eigvecs_cupy = readin_eigvecs(eig_dir, t, Nev1,conf_id, Nx)
        except:
            eigvecs_cupy =np.load(f"/public/home/sunp/sunpeng/eigen_vector/beta6.20_mu-0.2770_ms-0.2400_L48x96/vector/{conf_id}/beta6.20_mu-0.2770_ms-0.2400_L48x96_cfg_{conf_id}_smear_HYP_alp1_0.5_alp2_0.6_alp3_0.6_{t}.eigenvector.npy")[:Nev1]
            eigvecs_cupy = eigvecs_cupy.reshape(Nev1,Nx*Nx*Nx,3)
        #ed1 = time.time()
        #print("Read-in eigenvector done , time used: %.3f s" % (ed1 - st1))
        # eigen_x_sum = np.transpose(eigvecs_cupy)
        # eigen_x_sum=cp.sum(eigvecs_cupy,axis=1)
        # eigen_1=eigen_sum[1]
        #print(cp.shape(eigvecs_cupy))
        #st2 = time.time()
        phase_factor_cupy = phase_calc(Mom,Nx)
        for xi in range(
            0, Nx
        ):  # I did this becasue the intermediate array is too large for a single GPU to handle, loop z
            VVV[t] += (
                contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                )
                + contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                )
                + contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                )
                - contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                )
                - contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                )
                - contract(
                    "x,ax,bx,cx->abc",
                    phase_factor_cupy[xi * (Nx**2) : (xi + 1) * (Nx**2)],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 1],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 0],
                    eigvecs_cupy[:, xi * (Nx**2) : (xi + 1) * (Nx**2), 2],
                )
            )
        #ed2 = time.time()
        #print("t= %s VVV Contraction done , time used: %.3f s" % (t,ed2 - st2))
        
        del eigvecs_cupy
    return VVV

