# ═══════════════════════════════════════════════════════
# Multi-hadron 2pt: dibaryon_p_lambda_local_2pt <- source
#   Hadrons in sink:   2
#   Hadrons in source: 2
#   Topologies: 12
# ═══════════════════════════════════════════════════════

import cupy as cp
from opt_einsum import contract
# Epsilon tensor (3D color anti-symmetric)
epsilon = cp.zeros((3, 3, 3), dtype=cp.float64)
epsilon[0,1,2] = epsilon[1,2,0] = epsilon[2,0,1] = 1.0
epsilon[0,2,1] = epsilon[2,1,0] = epsilon[1,0,2] = -1.0

# Gamma matrices for baryon spin/parity structure
from pyquda_utils import core, io, gamma, source
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus

# Sink block: sum over 12 Wick topology/ies

try:
    try:
        two_pt_site = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHBhb,wtzyxJEje,wtzyxLDld,wtzyxKFkf->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 0 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 0: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        two_pt_site = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHBhb,wtzyxJEje,wtzyxLDld,wtzyxKFkf->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 0 dp
    print('topo=0 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 0: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHBhb,wtzyxJFjf,wtzyxLDld,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 1 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 1: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHBhb,wtzyxJFjf,wtzyxLDld,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 1 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=1 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 1: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHEhe,wtzyxJBjb,wtzyxLDld,wtzyxKFkf->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 2 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 2: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHEhe,wtzyxJBjb,wtzyxLDld,wtzyxKFkf->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 2 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=2 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 2: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHEhe,wtzyxJFjf,wtzyxLDld,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 3 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 3: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHEhe,wtzyxJFjf,wtzyxLDld,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 3 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=3 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 3: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHFhf,wtzyxJBjb,wtzyxLDld,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 4 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 4: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHFhf,wtzyxJBjb,wtzyxLDld,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 4 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=4 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 4: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHFhf,wtzyxJEje,wtzyxLDld,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 5 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 5: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHFhf,wtzyxJEje,wtzyxLDld,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 5 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=5 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 5: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHBhb,wtzyxJEje,wtzyxLAla,wtzyxKFkf->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 6 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 6: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHBhb,wtzyxJEje,wtzyxLAla,wtzyxKFkf->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 6 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=6 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 6: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHBhb,wtzyxJFjf,wtzyxLAla,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 7 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 7: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHBhb,wtzyxJFjf,wtzyxLAla,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 7 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=7 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 7: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHEhe,wtzyxJBjb,wtzyxLAla,wtzyxKFkf->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 8 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 8: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHEhe,wtzyxJBjb,wtzyxLAla,wtzyxKFkf->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 8 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=8 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 8: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHEhe,wtzyxJFjf,wtzyxLAla,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 9 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 9: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHEhe,wtzyxJFjf,wtzyxLAla,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 9 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=9 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 9: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHFhf,wtzyxJBjb,wtzyxLAla,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 10 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 10: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHFhf,wtzyxJBjb,wtzyxLAla,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 10 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=10 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 10: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHFhf,wtzyxJEje,wtzyxLAla,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 11 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 11: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHFhf,wtzyxJEje,wtzyxLAla,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_s.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 11 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=11 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 11: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()


# Trace spatial volume → time-slice
two_pt_local = two_pt_site

# MPI gather
from pyquda_comm import array
two_pt_result = core.gatherLattice(array.arrayAsNumpy(two_pt_local, backend='cupy'), [0, -1, -1, -1])
