#!/Library/Frameworks/Python.framework/Versions/3.10/bin/python3
import sys
import numpy as np
import matplotlib
import fileinput
# import utils
matplotlib.use('Agg') # has to be imported before the next lines
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

sys.path.append("/public/home/yanzc/GEVP/GEVP_v2")
import utils

def read_dat(path):
    infile = fileinput.input(path)
    #infile = fileinput.input('%s/corr_uu_gamma5_Px1Py0Pz0_conf%05d.dat'%(meason_id,13500+fid*50))
    flist = []
    for line in infile :
        tmp=line.split()
        temp = float(tmp[0])
        #temp =float(tmp[fid]).real
        #print('temp:',temp)
        flist.append(temp)

    corr = np.array(flist)
    # print(corr)
	
    return corr[:nsample]

t0=2
tref=10
NMOM = 5

lattice="beta6.41_mu-0.2295_ms-0.2050_L32x96"
input_dir=F"/public/home/gengyq/laph/NP/fitwork/GEVPcorr/C48P29/T1SDmix/"
results_dir="%s/fit_results" %input_dir

#read in C(t0)
corr_matrix=np.load("%s/corr_matrix_boots.npz" %input_dir)['corr_matrix']
Ct0=corr_matrix[:,t0,:,:]
print(Ct0[0])
nsample=corr_matrix.shape[0]
Nt=corr_matrix.shape[1]
dim=corr_matrix.shape[2]
Ct0 = Ct0[:nsample]

# Ct0=np.zeros_like(Ct0_new)
# Ct0[:,0:4,0:4]=Ct0_new[:,1:5,1:5]
# Ct0[:,4,0:5]=Ct0_new[:,0,0:5]
# Ct0[:,0:5,4]=Ct0_new[:,0:5,0]
print(Ct0[0])

ev_=np.load("%s/eigvecs_t0_2.npz" %(input_dir))['eig_vecs']
# ev_=np.load("%s/sort_eigval/eigvecs_t0_4.npz" %(input_dir))['eig_vecs']
#ev=np.real(ev['eig_vecs'])
print(ev_.shape)
# ev=np.zeros_like(ev_)
# ev[:,:,:4]=ev_[:,:,1:5]
# ev[:,:,4]=ev_[:,:,0]
ev=ev_
#test=(ev[0,t0+1].T)@(Ct0[0])@ev[0,t0+1]
#print(test)

mass=np.array([1.072715295,1.101872036,1.106352397,1.136750672,1.135666449],dtype=float)
print(np.mean(mass,0))

Z=np.zeros((nsample, dim, dim), dtype='float') # dim(states) dim(ops)
coef=np.sqrt(2.0*mass)*np.exp(0.5*mass*t0)

print('coef:',coef.shape)
print('ev:',ev.shape)
print('Ct0:',Ct0.shape)

for n in range(dim):
	Z[:,n]=coef[n]*np.einsum("ai,aij->aj",  ev[:,tref,:,n], Ct0) #a:sample, i: ops, n: states
print("Z factor done")

mean_Z = np.mean(Z, axis=0)
std_Z = np.std(Z, axis=0)

norm_Z = np.amax(np.abs(mean_Z), axis=1)
#print(norm_Z[15])
#print(mean_Z[15])

norm_mean_Z = np.zeros_like(mean_Z)
norm_std_Z = np.zeros_like(std_Z)
for _n in range(dim):
	for _j in range(dim):
		norm_mean_Z[_n,_j] = mean_Z[_n,_j]/norm_Z[_n]
		norm_std_Z[_n,_j] = std_Z[_n,_j]/norm_Z[_n]


# print(mean_Z)
color=['r','r','r','b','b']
print(dim)
output=f"/public/home/gengyq/laph/NP/fitwork/picture/C48P29/overlap.pdf"
print(output)
pdfplot=PdfPages(output)
fig, ax = plt.subplots(1,dim,figsize=( 9,3), gridspec_kw={'wspace': 0.25})
# fig.set_size_inches(4,3)
for _n in range(5):
	for _i in range(dim):
		ax[_n].add_patch(patches.Rectangle((_i*0.1, 0.0), 0.0965, np.abs(norm_mean_Z[_n, _i]), fc=color[_i], color=color[_i], alpha=1.0, linewidth=1, linestyle='solid'))

	labely=-0.075
	fs=10
	
	ax[_n].spines['top'].set_color('none')
	ax[_n].spines['bottom'].set_color('none')
	ax[_n].spines['left'].set_color('none')
	ax[_n].spines['right'].set_color('none')
	ax[_n].get_xaxis().set_visible(False)
	ax[_n].get_yaxis().set_visible(False)
	ax[_n].set_xlim(0,0.5)
	# if np.std(mass,0)[_n]*100000<100:
	# 	ax[_n].text(0.1, labely, r"%.5f(%.0f)"%(np.mean(mass,0)[_n],np.std(mass,0)[_n]*100000), fontsize=fs)
	# else:
	# 	ax[_n].text(0.1, labely, r"%.4f(%.0f)"%(np.mean(mass,0)[_n],np.std(mass,0)[_n]*10000), fontsize=fs)
		#ax[_n].set_xlabel(r"%.4f(%.0f)"%(np.mean(mass,0)[_n],np.std(mass,0)[_n]*10000), fontsize=fs)
		# plt.gcf().text(0.35, labely, r"%.4f(%.0f)"%(np.mean(mass,0)[_n],np.std(mass,0)[_n]*10000), fontsize=fs)
	# plt.gcf().text(0.60, labely, r"$J/\Psi K^*$", fontsize=fs)
	# plt.gcf().text(0.71, labely, r"$\eta_c K^*$", fontsize=fs)



ax[0].text(0.1, labely, r"$S(\tilde{q}^2=0)$", fontsize=fs)
ax[1].text(0.1, labely, r"$S(\tilde{q}^2=1)$", fontsize=fs)
ax[2].text(0.1, labely, r"$D(\tilde{q}^2=1)$", fontsize=fs)
ax[3].text(0.1, labely, r"$D(\tilde{q}^2=2)$", fontsize=fs)
ax[4].text(0.1, labely, r"$S(\tilde{q}^2=2)$", fontsize=fs)

# ax[0].text(0.1, labely, r"$\Xi_{cc}N (\tilde{q}^2=0)$", fontsize=fs)
# ax[1].text(0.1, labely, r"$\Lambda_c\Lambda_c (\tilde{q}^2=0)$", fontsize=fs)
# ax[2].text(0.1, labely, r"$\Lambda_c\Lambda_c (\tilde{q}^2=1)$", fontsize=fs)
# ax[3].text(0.1, labely, r"$\Xi_{cc}N (\tilde{q}^2=1)$", fontsize=fs)



pdfplot.savefig()
plt.clf()
pdfplot.close()
	
