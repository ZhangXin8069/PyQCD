import numpy as np
import math
import os
import fileinput
from mpi4py import MPI
from mpi4py.util import pkl5
from gamma_matrix import *
from input_output import *
from opt_einsum import contract
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print("Job started")
Nc = 3
# ------------------------------------------------------------------------------
infile = fileinput.input()
for line in infile:
    tmp = line.split()
    if tmp[0] == "Nt":
        Nt = int(tmp[1])
    if tmp[0] == "Nx":
        Nx = int(tmp[1])
    if tmp[0] == "conf_id":
        conf_id = tmp[1]
    if tmp[0] == "Nev":
        Nev = int(tmp[1])
    if tmp[0] == "Nev1":  # number of eigenvectors used in contraction
        Nev1 = int(tmp[1])
    if tmp[0] == "Px":
        Px = int(tmp[1])
    if tmp[0] == "Py":
        Py = int(tmp[1])
    if tmp[0] == "Pz":
        Pz = int(tmp[1])
    if tmp[0] == "path_eigenvectors":
        eig_dir = tmp[1]
    if tmp[0] == "peram_u_dir":
        peram_u_dir = tmp[1]
    if tmp[0] == "VdV_dir":
        VdV_dir = tmp[1]
    if tmp[0] == "corr_nucl_dir":
        corr_nucl_dir = tmp[1]


st00 = time.time()

##______________________________________________________________________
##______________________________________________________________________

# calculate the exp part


def readin_eigvecs(eig_dir, t, Nev1, conf_id, Nx):
    f = open("%s/eigvecs_t%03d_%s" % (eig_dir, t, conf_id), "rb")
    eigvecs = np.fromfile(f, dtype="f8")
    eigvecs_size = eigvecs.size
    Nev = int(eigvecs_size / (Nx * Nx * Nx * 3 * 2))
    eigvecs = eigvecs.reshape(Nev, Nx * Nx * Nx * 3, 2)
    eigvecs = eigvecs[..., 0] + eigvecs[..., 1] * 1j
    eigvecs = eigvecs[0:Nev1, :]
    eigvecs = np.transpose(eigvecs)
    # eigvecs = eigvecs.reshape(Nev1, Nx, Nx, Nx, Nc)
    return eigvecs


def readin_peram(peram_dir, conf_id, Nt, Nev1, t_source):
    f = open("%s/perams.%s.0.%i" % (peram_dir, conf_id, t_source), "rb")
    peram = np.fromfile(f, dtype="f8")
    f.close()

    for d_source in range(1, 4):
        f = open("%s/perams.%s.%i.%i" % (peram_dir, conf_id, d_source, t_source), "rb")
        temp = np.fromfile(f, dtype="f8")
        peram = np.append(peram, temp)
        temp = None
        f.close()
    peram_size = peram.size
    Nev = int(np.sqrt(peram_size / (4 * 4 * Nt * 2)))
    peram = peram.reshape(
        4, Nt, Nev, 4, Nev, 2
    )  # d_source, t_sink, ev_source, d_sink, ev_sink, complex
    peram = peram.transpose(
        1, 3, 0, 4, 2, 5
    )  # t_sink, d_sink, d_source, ev_sink, ev_souce,  complex
    peram = peram[..., 0] + peram[..., 1] * 1j
    peram = peram[:, :, :, 0:Nev1, 0:Nev1]

    return peram


def VDV_assemble(eig_dir, t, Nev1, conf_id, Nx, Px, Py, Pz):
    Mom = np.array([Pz, Py, Px])

    exp_diag = np.zeros(Nx * Nx * Nx * Nc, dtype=complex)
    for z in range(0, Nx):
        for y in range(0, Nx):
            for x in range(0, Nx):
                Pos = np.array([z, y, x])
                exp_diag[z * Nx * Nx * 3 + y * Nx * 3 + x * 3] = np.exp(
                    -np.dot(Mom, Pos) * 2 * math.pi * 1j / Nx
                )
                exp_diag[z * Nx * Nx * 3 + y * Nx * 3 + x * 3 + 1] = exp_diag[
                    z * Nx * Nx * 3 + y * Nx * 3 + x * 3
                ]
                exp_diag[z * Nx * Nx * 3 + y * Nx * 3 + x * 3 + 2] = exp_diag[
                    z * Nx * Nx * 3 + y * Nx * 3 + x * 3
                ]

    eigvecs = readin_eigvecs(eig_dir, t, Nev1, conf_id, Nx)
    VDV = np.zeros((Nev1, Nev1), dtype=complex)

    VDV = np.matmul((np.conj(np.transpose(eigvecs)) * exp_diag), eigvecs)

    return VDV


###############################################################################

# load VdV
# if momentum is zero, the VdV is a unit matrix

VdV_sink = np.zeros((Nt, Nev1, Nev1), dtype=complex)

st_vdv_all = time.time()
if rank != 0:
    for t in range(int(Nt * (rank - 1) / (size - 1)), int(Nt * (rank) / (size - 1))):
        st_vdv = time.time()
        if Px == 0 and Py == 0 and Pz == 0:
            VdV_sink[t] = np.identity(Nev1)
        else:
            VdV_sink[t] = VDV_assemble(eig_dir, t, Nev1, conf_id, Nx, Px, Py, Pz)

        ed_vdv = time.time()
        print("VDV t %s done, time used: %.3f s" % (t, ed_vdv - st_vdv))

comm.Barrier()

VdV_sink_all = None
if rank == 0:
    VdV_sink_all = np.empty((size, VdV_sink.size), dtype=complex)
comm.Gather(VdV_sink, VdV_sink_all, root=0)

if rank == 0:
    VdV_sink = np.sum(VdV_sink_all, axis=0)
    VdV_sink = VdV_sink.reshape(Nt, Nev1, Nev1)
    # VdV in rank0 = sum(VdV)
    ed_vdv_all = time.time()
    print("VDV all t done, time used: %.3f s" % (ed_vdv_all - st_vdv_all))

VdV_sink = comm.bcast(VdV_sink, root=0)
# Broadcast VdV from rank0
# print(rank, VdV_sink.shape)
################################################################################

# construct empty array

corr_pion_matrix = np.zeros((Nt, Nt), dtype=complex)
corr_pion = np.zeros((1, Nt), dtype=complex)

print("Contract start")
if rank != 0:
    for t_source in range(
        int(Nt * (rank - 1) / (size - 1)), int(Nt * (rank) / (size - 1))
    ):
        st0 = time.time()
        # load perambulator
        peram_u = readin_peram(peram_u_dir, conf_id, Nt, Nev1, t_source)
        # construct anti-perambulator
        g5_peramc_g5 = contract(
            "lm,tmnfe,ni->tlife", gamma(5), np.conj(peram_u), gamma(5)
        )
        # contruct (VdV)^dagger
        VdV_source = np.conj(VdV_sink[t_source].T)
        # correlation function

        for t_sink in range(Nt):
            corr_pion_matrix[t_source, t_sink] = contract(
                "mo,np,nmba,podc,db,ac->",
                gamma(5),
                gamma(5),
                peram_u[t_sink],
                g5_peramc_g5[t_sink],
                VdV_sink[t_sink],
                VdV_source,
            )
        # which is equal to:
        # for t_sink in range(Nt):
        #     corr_pion_matrix[t_source,t_sink] = \
        #         contract("nmba,nmdc,db,ac->",  peram_u[t_sink],\
        #                   np.conj(peram_u[t_sink]),VdV_sink[t_sink], VdV_source )

        # clear useless data
        del peram_u
        ed0 = time.time()
        print(
            "****************all complete for t_source %d, time used: %.3f s****************"
            % (t_source, ed0 - st0)
        )

comm.Barrier()

print("Save start")

corr_pion_matrix_all = None
if rank == 0:
    corr_pion_matrix_all = np.empty((size, corr_pion_matrix.size), dtype=complex)
comm.Gather(corr_pion_matrix, corr_pion_matrix_all, root=0)

if rank == 0:
    corr_pion_matrix = np.sum(corr_pion_matrix_all, axis=0)
    corr_pion_matrix = corr_pion_matrix.reshape(Nt, Nt)
    # sum the same delta time-slime correlation function
    for t_source in range(0, Nt):
        for t_sink in range(0, Nt):
            corr_pion[0, (t_sink - t_source + Nt) % Nt] = (
                corr_pion[0, (t_sink - t_source + Nt) % Nt]
                + corr_pion_matrix[t_source, t_sink]
            )

    print(corr_pion[0])
    # save correlation function
    write_data_ascii(
        corr_pion,
        Nt,
        Nx,
        "%s/corr_uu_gamma5_Px%sPy%sPz%s_conf%s_mpi.dat"
        % (corr_nucl_dir, Px, Py, Pz, conf_id),
    )

    print("Save done")
    ed00 = time.time()
    print(
        "****************final calculation done, time used: %.3f s****************"
        % (ed00 - st00)
    )
