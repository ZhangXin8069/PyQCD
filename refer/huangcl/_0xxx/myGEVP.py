#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
import sys
import numpy as np
import utils
from utils.analyze_fcts import bootstrap

contract_all=np.zeros([300,72,2,2])
contract_all[:,:,0,0]=?
contract_all[:,:,1,1]=?

def main(dim, t0):
    nboots = 3000
    output_path = (f"/public/home/gengyq/laph/NP/fitwork/GEVPcorr/{state}/{object}/Nev{Nev}")
    
    corr_data=contract_all #[Nconf,Nt,dim,dim]

    corr_matrix=bootstrap(corr_data,nboots)
    # corr_matrix=corr_matrix.transpose(0,3,1,2)  #confid,nt,dim,dim

    np.savez("%s/corr_matrix_boots.npz" % output_path, corr_matrix=corr_matrix)
    print(corr_matrix.shape)
    tmax = 25
    eig_vals, eig_vecs = utils.calculate_gevp(corr_matrix, t0, tmax)
    # 	order_eig_vals, order_eig_vecs=utils.reorder_by_eval(eig_vals, eig_vecs, t0)
    # 	order_eig_vals, order_eig_vecs=utils.reorder_by_evec(eig_vals, eig_vecs, corr_matrix[:,t0,:,:], t0, tmax)

    
    # order, order_eig_vals, order_eig_vecs=utils.reorder_by_evec_eval(eig_vals, eig_vecs, corr_matrix[:,t0,:,:], t0, tmax)
    order, order_eig_vals, order_eig_vecs=utils.reorder_by_eval(eig_vals, eig_vecs, t0, tmax)

    for i in range(dim):
        utils.write_data_ascii(order_eig_vals[:,:,i], "%s/eigvals_%d_t0_%d.dat" %(output_path, i, t0))
        np.save("%s/eigvals_%d_t0_%d.npy"%(output_path, i, t0),order_eig_vals[:,:,i])
    np.savez("%s/eigvecs_t0_%d.npz" %(output_path, t0), eig_vecs=order_eig_vecs, order=order)


# make this script importable, according to the Google Python Style Guide
if __name__ == "__main__":
    try:
        main(2, 4)
    except KeyboardInterrupt:
        pass
