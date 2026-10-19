#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
import sys
import numpy as np
import matplotlib
import utils

t0=2

state = "C24P29"
object="NP"

eig_input_dir=f"/public/home/gengyq/laph/NP/fitwork/GEVPcorr/{state}/{object}/"
output_path=f"/public/home/gengyq/laph/NP/fitwork/GEVPcorr/{state}/{object}/sort/"

eig_vecs=np.load("%s/eigvecs_t0_%s.npz" %(eig_input_dir,t0))["eig_vecs"]
print(eig_vecs.shape)

corr_matrix=np.load("%s/corr_matrix_boots.npz" %(eig_input_dir))["corr_matrix"]
print(corr_matrix.shape)
nboots = corr_matrix.shape[0]
Nt = corr_matrix.shape[1]
dim=corr_matrix.shape[2]

eig_vals = np.zeros((nboots, Nt, dim))
for _d in range(dim):
	infile="%s/eigvals_%i_t0_%s.dat" %(eig_input_dir, _d,t0)
	eig_vals[:,:,_d] = np.genfromtxt(infile, skip_header=1, usecols=(1)).reshape(nboots, Nt)

Ct0=corr_matrix[:,t0]

tmax=40
# order, order_eig_vals, order_eig_vecs=utils.reorder_by_evec_eval(eig_vals, eig_vecs, corr_matrix[:,t0,:,:], t0, tmax)
order, order_eig_vals, order_eig_vecs=utils.reorder_by_eval(eig_vals, eig_vecs, t0, tmax)




for i in range(dim):
	utils.write_data_ascii(order_eig_vals[:,:,i], "%s/eigvals_%d_t0_%d.dat" %(output_path, i, t0))

np.savez("%s/eigvecs_t0_%d.npz" %(output_path, t0), eig_vecs=order_eig_vecs, order=order)

