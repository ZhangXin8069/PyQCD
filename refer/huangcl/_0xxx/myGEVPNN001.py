#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
import sys
import numpy as np
import utils
from utils.analyze_fcts import bootstrap


state = "C48P29"
object="NN"
Nev=100

if state=="C24P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L24x72"
    idALL=np.arange(879)*50+4050
    Lt=72
    bin=    None
    Nmom=1
if state=="C32P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L32x64"
    idALL=np.arange(1000)*50+11000
    Lt=64
    bin=    None
    Nmom=1
if state=="C48P29":
    conf_name="beta6.20_mu-0.2770_ms-0.2400_L48x96"
    idALL=np.arange(751)*10+500
    Lt=96
    bin=    None
    Nmom=1



datapath=f"/public/home/gengyq/NP/result/boost001/"
dataNN=[]
dataNP=[]
for id in idALL:
    try:
        if state=="C32P29":
            # dataNN.append(np.load(f"{datapath}/{conf_name}/NNcorr_spin_mom20_conf{id}.npy")[:1,:1])
            dataNP.append(np.load(f"{datapath}/{conf_name}/NPcorr_spin_mom2_conf{id}.npy"))
        elif state=="C48P29":
            dataNP.append(np.load(f"{datapath}/{conf_name}/NPcorr_spin_mom2_conf{id}.npy"))

    except:
        print(id)
        continue
dataNP=np.array(dataNP)   # confid, Nmom, Nmom, t , r,q,z,y
# dataNN=np.array(dataNN)
print(dataNP.shape)

if object=="NP":
    dataNP=(dataNP+dataNP.transpose(0,1,2,3,4,7,6,5)+dataNP.transpose(0,1,2,3,6,5,4,7)+dataNP.transpose(0,1,2,3,6,7,4,5))/4
else:
    dataNP=(dataNP-dataNP.transpose(0,1,2,3,4,7,6,5)-dataNP.transpose(0,1,2,3,6,5,4,7)+dataNP.transpose(0,1,2,3,6,7,4,5))/4


print("samples",dataNP.shape[0])

N1=dataNP.shape[1]
N2=dataNP.shape[2]

print(N1)
print(N2)
contract_all1=np.zeros([dataNP.shape[0],Nmom,Nmom,Lt],"<c16")
contract_all2=np.zeros([dataNP.shape[0],Nmom,Nmom,Lt],"<c16")
contract_all3=np.zeros([dataNP.shape[0],Nmom,Nmom,Lt],"<c16")
for p1 in range(N1):
    # if p1==0:
    ptot1=0
    # elif p1 <=3:
    #     ptot1=1
    for p2 in range(N1):
        if p2!=3:
            ptot2=0
            # if p2==0:
            #     ptot2=0
            # elif p2 <=3:
            #     ptot2=1
            # elif p2 <=5 and p2 >3:
            #     ptot2=1
            if object=="NN":
                irrep="A1_001"
                contract_all1[:,ptot1,ptot2,:]+=((dataNP[:,p1,p2,:,0,0,1,1]+dataNP[:,p1,p2,:,1,1,0,0]-dataNP[:,p1,p2,:,0,1,1,0]-dataNP[:,p1,p2,:,1,0,0,1]))/2*3
                # contract_all2[:,ptot1,ptot2,:]+=(dataNN[:,p1,p2,:,0,0,1,1]+dataNN[:,p1,p2,:,1,1,0,0]-dataNN[:,p1,p2,:,0,1,1,0]-dataNN[:,p1,p2,:,1,0,0,1])/2
                # contract_all3[:,ptot1,ptot2,:]+=(dataNN[:,p1,p2,:,0,0,1,1]+dataNN[:,p1,p2,:,1,1,0,0]-dataNN[:,p1,p2,:,0,1,1,0]-dataNN[:,p1,p2,:,1,0,0,1])/2
            if object=="NP":
                irrep="E2_001"
                contract_all1[:,ptot1,ptot2,:]+=(dataNP[:,p1,p2,:,0,0,1,1]+dataNP[:,p1,p2,:,1,1,0,0]+dataNP[:,p1,p2,:,0,1,1,0]+dataNP[:,p1,p2,:,1,0,0,1])/2
                contract_all2[:,ptot1,ptot2,:]+=dataNP[:,p1,p2,:,0,0,0,0]
                contract_all3[:,ptot1,ptot2,:]+=dataNP[:,p1,p2,:,1,1,1,1]
# print(np.mean(np.log(np.roll(contract_all1[:,0,0,10:25],1,1)/contract_all1[:,0,0,10:25]),0))
# print(np.mean(np.log(np.roll(contract_all2[:,0,0,10:25],1,1)/contract_all2[:,0,0,10:25]),0))
# print(np.mean(np.log(np.roll(contract_all3[:,0,0,10:25],1,1)/contract_all3[:,0,0,10:25]),0))

contract_all=(contract_all1+contract_all2+contract_all3)/3
for p1 in range(Nmom):
    for p2 in range(Nmom):
        if p2<p1:
            contract_all[:,p1,p2,:]=contract_all[:,p2,p1,:] 
# print(contract_all[20,0,0,:25])
np.save(f"/public/home/gengyq/NP/raw/{state}_{irrep}_Nev{Nev}",contract_all)
print(contract_all.shape)

# print(contract_all[10,0,1])
# print(contract_all[10,1,0])

def main(dim, t0):
    nboots = 3000
    output_path = (f"/public/home/gengyq/laph/NP/fitwork/GEVPcorr/{state}/{object}/P001")
    
    corr_data=contract_all #[Nconf,Nt,dim,dim]

    # corr_matrix=bootstrap((corr_data[1::2,:4,:4,:]+corr_data[0::2,:4,:4,:])/2,nboots)
    corr_matrix=bootstrap(corr_data,nboots)
    corr_matrix=corr_matrix.transpose(0,3,1,2)  #confid,nt,dim,dim

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
        main(Nmom, 2)
    except KeyboardInterrupt:
        pass
