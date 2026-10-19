import numpy as np
from lqcddb import gamma, cached_contract, loop_tsrc

A = np.load('/public/home/sush/distillation/lambdaproton/result/C24P29/Px0Py0Pz1/ENV_50/conf10000/corr_3pt.npy')
B = np.load('/public/home/guoj/work/a.npy')
A = A.sum(0)
print(A[0, 0, 0, 0, 0, 0])
print(B[0, 0, 0, 0, 0, 0])


curr = (gamma(1) @ gamma(5))
projector = (gamma(0) + gamma(4))/2

B = cached_contract('...abcd, ab,cd->...', A, curr, projector)

print(B[0, :6])