"""Distinct x/y/z inputs, shuffled global momenta, nonzero accumulator test."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'reference'))
from flowed_perambulator_factorization import accumulate_triplet, build_flowed_sequential
from flow_common_l32x96 import spin_objects
from spin2_contractions import Contractor,accumulate_triplet_reduced


def main():
    rng=np.random.default_rng(20260923)
    def rand(shape):return rng.normal(size=shape)+1j*rng.normal(size=shape)
    c=Contractor('greedy');g,_,q=spin_objects()
    p=rand((4,4,3,3));b=rand((2,4,4,2,3));a=rand((2,4,4,3,2))
    h={d:build_flowed_sequential(rand((2,2,2,3,2,2)),a,c) for d in ['z','y','x']}
    vs=rand((3,3,3,3));vr=rand(vs.shape)
    shape=(2,4,7,3,4,4)
    ru=rand(shape);rd=rand(shape);su=ru.copy();sd=rd.copy()
    args=(h,['x','z','y'],p,b,g,q,vs,vr)
    accumulate_triplet(*args,ru,rd,[5,1,3],c)
    accumulate_triplet_reduced(*args,su,sd,[5,1,3],c)
    for label,ref,got in [('u',ru,su),('d',rd,sd)]:
        np.testing.assert_allclose(got,ref,rtol=2e-11,atol=1e-8)
        print(label,'relative_L2',np.linalg.norm(ref-got)/np.linalg.norm(ref))
    print('PASS multi-direction, shuffled momentum indices, nonzero accumulators')


if __name__=='__main__':main()
