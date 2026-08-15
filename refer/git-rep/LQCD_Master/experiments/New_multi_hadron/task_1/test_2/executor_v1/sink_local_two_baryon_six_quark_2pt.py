# ═══════════════════════════════════════════════════════
# Multi-hadron 2pt: local_two_baryon_six_quark_2pt <- source
#   Hadrons in sink:   2
#   Hadrons in source: 2
#   Topologies: 36
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

# Sink block: sum over 36 Wick topology/ies

try:
    try:
        two_pt_site = - contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIAia,wtzyxHChc,wtzyxJDjd,wtzyxLFlf,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 0 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 0: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        two_pt_site = - contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIAia,wtzyxHChc,wtzyxJDjd,wtzyxLFlf,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIAia,wtzyxHChc,wtzyxJFjf,wtzyxLDld,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 1 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 1: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIAia,wtzyxHChc,wtzyxJFjf,wtzyxLDld,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 1 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=1 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 1: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIAia,wtzyxHEhe,wtzyxJDjd,wtzyxLFlf,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 2 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 2: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIAia,wtzyxHEhe,wtzyxJDjd,wtzyxLFlf,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 2 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=2 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 2: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIAia,wtzyxHEhe,wtzyxJFjf,wtzyxLDld,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 3 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 3: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIAia,wtzyxHEhe,wtzyxJFjf,wtzyxLDld,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 3 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=3 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 3: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIDid,wtzyxHChc,wtzyxJAja,wtzyxLFlf,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 4 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 4: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIDid,wtzyxHChc,wtzyxJAja,wtzyxLFlf,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIDid,wtzyxHChc,wtzyxJFjf,wtzyxLAla,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 5 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 5: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIDid,wtzyxHChc,wtzyxJFjf,wtzyxLAla,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIDid,wtzyxHEhe,wtzyxJAja,wtzyxLFlf,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 6 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 6: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIDid,wtzyxHEhe,wtzyxJAja,wtzyxLFlf,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIDid,wtzyxHEhe,wtzyxJFjf,wtzyxLAla,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 7 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 7: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIDid,wtzyxHEhe,wtzyxJFjf,wtzyxLAla,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIFif,wtzyxHChc,wtzyxJAja,wtzyxLDld,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 8 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 8: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIFif,wtzyxHChc,wtzyxJAja,wtzyxLDld,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 8 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=8 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 8: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIFif,wtzyxHChc,wtzyxJDjd,wtzyxLAla,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 9 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 9: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIFif,wtzyxHChc,wtzyxJDjd,wtzyxLAla,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 9 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=9 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 9: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIFif,wtzyxHEhe,wtzyxJAja,wtzyxLDld,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 10 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 10: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIFif,wtzyxHEhe,wtzyxJAja,wtzyxLDld,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 10 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=10 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 10: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIFif,wtzyxHEhe,wtzyxJDjd,wtzyxLAla,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 11 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 11: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGBgb,wtzyxIFif,wtzyxHEhe,wtzyxJDjd,wtzyxLAla,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 11 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=11 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 11: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHBhb,wtzyxJDjd,wtzyxLFlf,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 12 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 12: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHBhb,wtzyxJDjd,wtzyxLFlf,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 12 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=12 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 12: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHBhb,wtzyxJFjf,wtzyxLDld,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 13 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 13: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHBhb,wtzyxJFjf,wtzyxLDld,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 13 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=13 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 13: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHEhe,wtzyxJDjd,wtzyxLFlf,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 14 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 14: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHEhe,wtzyxJDjd,wtzyxLFlf,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 14 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=14 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 14: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHEhe,wtzyxJFjf,wtzyxLDld,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 15 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 15: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIAia,wtzyxHEhe,wtzyxJFjf,wtzyxLDld,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 15 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=15 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 15: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHBhb,wtzyxJAja,wtzyxLFlf,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 16 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 16: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHBhb,wtzyxJAja,wtzyxLFlf,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 16 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=16 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 16: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHBhb,wtzyxJFjf,wtzyxLAla,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 17 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 17: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHBhb,wtzyxJFjf,wtzyxLAla,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 17 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=17 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 17: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHEhe,wtzyxJAja,wtzyxLFlf,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 18 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 18: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHEhe,wtzyxJAja,wtzyxLFlf,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 18 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=18 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 18: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHEhe,wtzyxJFjf,wtzyxLAla,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 19 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 19: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIDid,wtzyxHEhe,wtzyxJFjf,wtzyxLAla,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 19 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=19 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 19: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIFif,wtzyxHBhb,wtzyxJAja,wtzyxLDld,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 20 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 20: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIFif,wtzyxHBhb,wtzyxJAja,wtzyxLDld,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 20 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=20 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 20: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIFif,wtzyxHBhb,wtzyxJDjd,wtzyxLAla,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 21 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 21: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIFif,wtzyxHBhb,wtzyxJDjd,wtzyxLAla,wtzyxKEke->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 21 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=21 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 21: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIFif,wtzyxHEhe,wtzyxJAja,wtzyxLDld,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 22 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 22: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIFif,wtzyxHEhe,wtzyxJAja,wtzyxLDld,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 22 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=22 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 22: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIFif,wtzyxHEhe,wtzyxJDjd,wtzyxLAla,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 23 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 23: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGCgc,wtzyxIFif,wtzyxHEhe,wtzyxJDjd,wtzyxLAla,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 23 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=23 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 23: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIAia,wtzyxHBhb,wtzyxJDjd,wtzyxLFlf,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 24 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 24: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIAia,wtzyxHBhb,wtzyxJDjd,wtzyxLFlf,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 24 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=24 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 24: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIAia,wtzyxHBhb,wtzyxJFjf,wtzyxLDld,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 25 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 25: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIAia,wtzyxHBhb,wtzyxJFjf,wtzyxLDld,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 25 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=25 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 25: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIAia,wtzyxHChc,wtzyxJDjd,wtzyxLFlf,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 26 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 26: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIAia,wtzyxHChc,wtzyxJDjd,wtzyxLFlf,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 26 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=26 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 26: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIAia,wtzyxHChc,wtzyxJFjf,wtzyxLDld,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 27 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 27: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIAia,wtzyxHChc,wtzyxJFjf,wtzyxLDld,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 27 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=27 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 27: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIDid,wtzyxHBhb,wtzyxJAja,wtzyxLFlf,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 28 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 28: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIDid,wtzyxHBhb,wtzyxJAja,wtzyxLFlf,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 28 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=28 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 28: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIDid,wtzyxHBhb,wtzyxJFjf,wtzyxLAla,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 29 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 29: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIDid,wtzyxHBhb,wtzyxJFjf,wtzyxLAla,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 29 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=29 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 29: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIDid,wtzyxHChc,wtzyxJAja,wtzyxLFlf,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 30 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 30: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIDid,wtzyxHChc,wtzyxJAja,wtzyxLFlf,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 30 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=30 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 30: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIDid,wtzyxHChc,wtzyxJFjf,wtzyxLAla,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 31 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 31: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIDid,wtzyxHChc,wtzyxJFjf,wtzyxLAla,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 31 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=31 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 31: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIFif,wtzyxHBhb,wtzyxJAja,wtzyxLDld,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 32 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 32: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIFif,wtzyxHBhb,wtzyxJAja,wtzyxLDld,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 32 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=32 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 32: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIFif,wtzyxHBhb,wtzyxJDjd,wtzyxLAla,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 33 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 33: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIFif,wtzyxHBhb,wtzyxJDjd,wtzyxLAla,wtzyxKCkc->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 33 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=33 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 33: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIFif,wtzyxHChc,wtzyxJAja,wtzyxLDld,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 34 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 34: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIFif,wtzyxHChc,wtzyxJAja,wtzyxLDld,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 34 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=34 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 34: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIFif,wtzyxHChc,wtzyxJDjd,wtzyxLAla,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        )  # topo 35 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 35: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,HI,hig,KL,klj,GC,JF,wtzyxGEge,wtzyxIFif,wtzyxHChc,wtzyxJDjd,wtzyxLAla,wtzyxKBkb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Tmat,
            Tmat, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data,
        optimize='dp',
        )  # topo 35 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=35 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 35: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()


# Trace spatial volume → time-slice
two_pt_local = two_pt_site

# MPI gather
from pyquda_comm import array
two_pt_result = core.gatherLattice(array.arrayAsNumpy(two_pt_local, backend='cupy'), [0, -1, -1, -1])
