# ═══════════════════════════════════════════════════════
# Multi-hadron 2pt: pnL <- pnL
#   Hadrons in sink:   3
#   Hadrons in source: 3
#   Topologies: 576
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

# Sink block: sum over 576 Wick topology/ies

try:
    try:
        two_pt_site = - contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 0 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 0: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        two_pt_site = - contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 1 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 1: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 2 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 2: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 3 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 3: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 4 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 4: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 5 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 5: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 6 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 6: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 7 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 7: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 8 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 8: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 9 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 9: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 10 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 10: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 11 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 11: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 12 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 12: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 13 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 13: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 14 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 14: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 15 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 15: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 16 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 16: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 17 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 17: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 18 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 18: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 19 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 19: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 20 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 20: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 21 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 21: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 22 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 22: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 23 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 23: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 24 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 24: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 25 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 25: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 26 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 26: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 27 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 27: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 28 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 28: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 29 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 29: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 30 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 30: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 31 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 31: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 32 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 32: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 33 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 33: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 34 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 34: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 35 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 35: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
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

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 36 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 36: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 36 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=36 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 36: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 37 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 37: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 37 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=37 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 37: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 38 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 38: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 38 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=38 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 38: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 39 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 39: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 39 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=39 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 39: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 40 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 40: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 40 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=40 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 40: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 41 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 41: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 41 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=41 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 41: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 42 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 42: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 42 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=42 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 42: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 43 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 43: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 43 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=43 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 43: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 44 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 44: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 44 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=44 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 44: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 45 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 45: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 45 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=45 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 45: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 46 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 46: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 46 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=46 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 46: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 47 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 47: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 47 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=47 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 47: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 48 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 48: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 48 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=48 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 48: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 49 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 49: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 49 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=49 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 49: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 50 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 50: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 50 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=50 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 50: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 51 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 51: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 51 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=51 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 51: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 52 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 52: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 52 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=52 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 52: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 53 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 53: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 53 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=53 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 53: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 54 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 54: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 54 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=54 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 54: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 55 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 55: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 55 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=55 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 55: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 56 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 56: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 56 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=56 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 56: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 57 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 57: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 57 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=57 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 57: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 58 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 58: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 58 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=58 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 58: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 59 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 59: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 59 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=59 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 59: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 60 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 60: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 60 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=60 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 60: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 61 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 61: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 61 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=61 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 61: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 62 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 62: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 62 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=62 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 62: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 63 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 63: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 63 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=63 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 63: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 64 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 64: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 64 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=64 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 64: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 65 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 65: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 65 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=65 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 65: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 66 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 66: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 66 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=66 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 66: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 67 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 67: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 67 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=67 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 67: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 68 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 68: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 68 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=68 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 68: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 69 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 69: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 69 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=69 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 69: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 70 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 70: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 70 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=70 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 70: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 71 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 71: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 71 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=71 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 71: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 72 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 72: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 72 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=72 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 72: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 73 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 73: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 73 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=73 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 73: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 74 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 74: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 74 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=74 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 74: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 75 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 75: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 75 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=75 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 75: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 76 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 76: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 76 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=76 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 76: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 77 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 77: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 77 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=77 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 77: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 78 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 78: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 78 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=78 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 78: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 79 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 79: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 79 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=79 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 79: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 80 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 80: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 80 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=80 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 80: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 81 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 81: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 81 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=81 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 81: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 82 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 82: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 82 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=82 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 82: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 83 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 83: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 83 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=83 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 83: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 84 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 84: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 84 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=84 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 84: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 85 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 85: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 85 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=85 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 85: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 86 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 86: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 86 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=86 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 86: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 87 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 87: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 87 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=87 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 87: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 88 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 88: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 88 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=88 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 88: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 89 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 89: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 89 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=89 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 89: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 90 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 90: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 90 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=90 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 90: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 91 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 91: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 91 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=91 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 91: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 92 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 92: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 92 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=92 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 92: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 93 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 93: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 93 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=93 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 93: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 94 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 94: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 94 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=94 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 94: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 95 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 95: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 95 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=95 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 95: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 96 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 96: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 96 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=96 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 96: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 97 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 97: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 97 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=97 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 97: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 98 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 98: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 98 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=98 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 98: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 99 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 99: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 99 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=99 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 99: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 100 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 100: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 100 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=100 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 100: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 101 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 101: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 101 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=101 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 101: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 102 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 102: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 102 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=102 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 102: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 103 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 103: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 103 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=103 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 103: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 104 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 104: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 104 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=104 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 104: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 105 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 105: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 105 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=105 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 105: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 106 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 106: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 106 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=106 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 106: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 107 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 107: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 107 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=107 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 107: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 108 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 108: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 108 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=108 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 108: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 109 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 109: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 109 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=109 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 109: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 110 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 110: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 110 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=110 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 110: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 111 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 111: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 111 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=111 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 111: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 112 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 112: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 112 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=112 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 112: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 113 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 113: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 113 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=113 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 113: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 114 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 114: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 114 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=114 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 114: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 115 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 115: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 115 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=115 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 115: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 116 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 116: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 116 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=116 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 116: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 117 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 117: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 117 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=117 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 117: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 118 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 118: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 118 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=118 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 118: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 119 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 119: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 119 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=119 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 119: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 120 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 120: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 120 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=120 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 120: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 121 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 121: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 121 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=121 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 121: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 122 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 122: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 122 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=122 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 122: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 123 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 123: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 123 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=123 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 123: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 124 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 124: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 124 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=124 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 124: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 125 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 125: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 125 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=125 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 125: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 126 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 126: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 126 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=126 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 126: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 127 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 127: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 127 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=127 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 127: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 128 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 128: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 128 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=128 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 128: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 129 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 129: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 129 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=129 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 129: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 130 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 130: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 130 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=130 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 130: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 131 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 131: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 131 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=131 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 131: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 132 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 132: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 132 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=132 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 132: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 133 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 133: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 133 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=133 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 133: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 134 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 134: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 134 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=134 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 134: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 135 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 135: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 135 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=135 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 135: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 136 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 136: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 136 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=136 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 136: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 137 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 137: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 137 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=137 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 137: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 138 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 138: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 138 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=138 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 138: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 139 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 139: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 139 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=139 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 139: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 140 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 140: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 140 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=140 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 140: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 141 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 141: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 141 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=141 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 141: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 142 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 142: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 142 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=142 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 142: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 143 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 143: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLAla,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 143 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=143 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 143: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 144 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 144: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 144 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=144 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 144: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 145 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 145: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 145 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=145 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 145: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 146 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 146: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 146 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=146 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 146: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 147 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 147: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 147 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=147 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 147: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 148 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 148: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 148 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=148 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 148: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 149 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 149: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 149 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=149 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 149: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 150 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 150: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 150 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=150 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 150: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 151 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 151: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 151 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=151 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 151: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 152 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 152: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 152 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=152 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 152: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 153 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 153: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 153 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=153 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 153: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 154 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 154: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 154 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=154 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 154: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 155 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 155: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 155 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=155 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 155: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 156 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 156: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 156 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=156 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 156: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 157 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 157: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 157 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=157 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 157: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 158 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 158: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 158 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=158 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 158: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 159 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 159: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 159 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=159 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 159: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 160 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 160: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 160 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=160 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 160: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 161 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 161: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 161 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=161 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 161: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 162 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 162: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 162 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=162 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 162: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 163 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 163: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 163 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=163 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 163: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 164 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 164: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 164 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=164 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 164: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 165 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 165: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 165 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=165 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 165: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 166 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 166: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 166 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=166 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 166: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 167 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 167: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 167 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=167 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 167: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 168 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 168: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 168 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=168 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 168: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 169 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 169: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 169 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=169 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 169: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 170 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 170: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 170 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=170 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 170: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 171 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 171: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 171 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=171 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 171: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 172 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 172: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 172 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=172 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 172: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 173 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 173: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 173 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=173 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 173: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 174 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 174: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 174 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=174 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 174: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 175 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 175: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 175 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=175 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 175: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 176 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 176: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 176 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=176 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 176: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 177 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 177: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 177 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=177 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 177: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 178 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 178: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 178 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=178 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 178: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 179 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 179: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 179 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=179 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 179: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 180 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 180: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 180 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=180 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 180: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 181 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 181: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 181 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=181 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 181: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 182 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 182: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 182 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=182 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 182: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 183 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 183: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 183 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=183 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 183: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 184 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 184: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 184 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=184 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 184: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 185 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 185: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 185 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=185 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 185: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 186 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 186: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 186 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=186 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 186: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 187 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 187: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 187 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=187 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 187: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 188 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 188: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 188 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=188 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 188: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 189 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 189: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 189 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=189 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 189: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 190 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 190: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 190 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=190 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 190: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 191 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 191: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 191 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=191 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 191: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 192 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 192: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 192 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=192 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 192: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 193 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 193: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 193 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=193 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 193: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 194 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 194: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 194 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=194 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 194: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 195 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 195: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 195 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=195 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 195: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 196 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 196: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 196 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=196 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 196: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 197 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 197: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 197 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=197 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 197: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 198 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 198: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 198 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=198 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 198: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 199 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 199: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 199 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=199 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 199: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 200 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 200: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 200 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=200 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 200: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 201 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 201: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 201 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=201 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 201: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 202 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 202: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 202 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=202 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 202: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 203 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 203: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 203 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=203 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 203: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 204 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 204: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 204 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=204 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 204: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 205 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 205: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 205 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=205 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 205: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 206 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 206: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 206 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=206 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 206: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 207 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 207: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 207 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=207 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 207: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 208 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 208: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 208 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=208 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 208: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 209 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 209: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 209 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=209 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 209: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 210 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 210: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 210 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=210 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 210: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 211 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 211: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 211 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=211 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 211: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 212 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 212: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 212 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=212 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 212: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 213 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 213: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 213 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=213 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 213: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 214 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 214: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 214 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=214 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 214: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 215 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 215: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 215 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=215 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 215: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 216 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 216: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 216 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=216 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 216: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 217 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 217: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 217 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=217 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 217: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 218 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 218: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 218 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=218 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 218: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 219 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 219: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 219 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=219 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 219: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 220 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 220: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 220 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=220 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 220: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 221 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 221: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 221 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=221 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 221: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 222 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 222: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 222 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=222 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 222: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 223 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 223: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 223 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=223 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 223: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 224 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 224: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 224 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=224 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 224: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 225 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 225: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 225 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=225 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 225: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 226 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 226: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 226 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=226 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 226: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 227 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 227: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 227 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=227 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 227: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 228 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 228: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 228 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=228 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 228: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 229 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 229: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 229 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=229 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 229: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 230 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 230: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 230 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=230 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 230: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 231 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 231: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 231 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=231 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 231: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 232 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 232: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 232 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=232 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 232: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 233 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 233: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 233 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=233 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 233: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 234 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 234: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 234 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=234 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 234: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 235 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 235: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 235 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=235 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 235: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 236 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 236: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 236 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=236 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 236: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 237 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 237: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 237 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=237 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 237: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 238 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 238: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 238 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=238 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 238: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 239 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 239: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 239 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=239 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 239: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 240 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 240: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 240 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=240 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 240: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 241 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 241: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 241 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=241 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 241: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 242 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 242: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 242 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=242 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 242: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 243 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 243: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 243 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=243 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 243: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 244 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 244: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 244 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=244 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 244: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 245 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 245: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 245 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=245 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 245: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 246 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 246: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 246 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=246 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 246: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 247 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 247: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 247 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=247 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 247: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 248 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 248: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 248 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=248 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 248: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 249 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 249: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 249 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=249 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 249: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 250 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 250: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 250 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=250 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 250: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 251 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 251: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 251 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=251 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 251: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 252 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 252: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 252 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=252 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 252: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 253 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 253: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 253 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=253 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 253: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 254 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 254: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 254 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=254 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 254: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 255 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 255: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 255 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=255 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 255: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 256 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 256: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 256 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=256 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 256: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 257 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 257: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 257 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=257 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 257: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 258 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 258: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 258 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=258 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 258: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 259 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 259: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 259 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=259 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 259: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 260 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 260: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 260 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=260 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 260: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 261 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 261: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 261 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=261 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 261: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 262 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 262: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 262 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=262 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 262: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 263 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 263: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 263 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=263 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 263: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 264 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 264: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 264 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=264 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 264: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 265 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 265: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 265 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=265 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 265: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 266 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 266: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 266 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=266 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 266: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 267 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 267: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 267 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=267 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 267: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 268 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 268: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 268 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=268 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 268: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 269 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 269: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 269 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=269 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 269: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 270 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 270: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 270 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=270 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 270: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 271 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 271: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 271 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=271 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 271: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 272 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 272: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 272 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=272 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 272: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 273 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 273: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 273 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=273 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 273: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 274 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 274: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 274 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=274 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 274: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 275 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 275: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 275 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=275 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 275: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 276 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 276: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 276 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=276 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 276: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 277 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 277: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 277 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=277 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 277: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 278 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 278: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 278 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=278 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 278: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 279 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 279: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 279 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=279 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 279: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 280 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 280: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 280 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=280 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 280: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 281 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 281: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 281 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=281 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 281: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 282 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 282: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 282 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=282 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 282: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 283 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 283: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 283 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=283 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 283: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 284 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 284: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 284 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=284 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 284: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 285 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 285: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 285 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=285 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 285: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 286 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 286: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 286 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=286 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 286: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 287 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 287: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLEle,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 287 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=287 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 287: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 288 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 288: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 288 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=288 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 288: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 289 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 289: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 289 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=289 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 289: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 290 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 290: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 290 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=290 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 290: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 291 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 291: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 291 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=291 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 291: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 292 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 292: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 292 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=292 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 292: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 293 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 293: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 293 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=293 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 293: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 294 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 294: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 294 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=294 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 294: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 295 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 295: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 295 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=295 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 295: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 296 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 296: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 296 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=296 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 296: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 297 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 297: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 297 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=297 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 297: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 298 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 298: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 298 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=298 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 298: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 299 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 299: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 299 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=299 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 299: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 300 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 300: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 300 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=300 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 300: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 301 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 301: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 301 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=301 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 301: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 302 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 302: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 302 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=302 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 302: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 303 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 303: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 303 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=303 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 303: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 304 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 304: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 304 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=304 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 304: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 305 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 305: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 305 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=305 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 305: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 306 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 306: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 306 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=306 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 306: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 307 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 307: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 307 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=307 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 307: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 308 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 308: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 308 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=308 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 308: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 309 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 309: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 309 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=309 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 309: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 310 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 310: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 310 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=310 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 310: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 311 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 311: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 311 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=311 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 311: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 312 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 312: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 312 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=312 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 312: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 313 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 313: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 313 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=313 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 313: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 314 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 314: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 314 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=314 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 314: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 315 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 315: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 315 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=315 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 315: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 316 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 316: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 316 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=316 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 316: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 317 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 317: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 317 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=317 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 317: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 318 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 318: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 318 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=318 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 318: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 319 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 319: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 319 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=319 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 319: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 320 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 320: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 320 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=320 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 320: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 321 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 321: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 321 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=321 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 321: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 322 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 322: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 322 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=322 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 322: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 323 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 323: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKBkb,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 323 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=323 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 323: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 324 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 324: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 324 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=324 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 324: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 325 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 325: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 325 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=325 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 325: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 326 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 326: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 326 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=326 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 326: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 327 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 327: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 327 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=327 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 327: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 328 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 328: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 328 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=328 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 328: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 329 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 329: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 329 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=329 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 329: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 330 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 330: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 330 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=330 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 330: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 331 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 331: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 331 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=331 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 331: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 332 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 332: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 332 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=332 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 332: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 333 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 333: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 333 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=333 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 333: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 334 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 334: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 334 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=334 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 334: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 335 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 335: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 335 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=335 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 335: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 336 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 336: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 336 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=336 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 336: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 337 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 337: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 337 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=337 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 337: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 338 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 338: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 338 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=338 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 338: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 339 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 339: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 339 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=339 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 339: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 340 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 340: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 340 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=340 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 340: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 341 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 341: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 341 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=341 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 341: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 342 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 342: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 342 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=342 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 342: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 343 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 343: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 343 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=343 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 343: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 344 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 344: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 344 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=344 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 344: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 345 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 345: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 345 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=345 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 345: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 346 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 346: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 346 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=346 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 346: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 347 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 347: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 347 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=347 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 347: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 348 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 348: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 348 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=348 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 348: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 349 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 349: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 349 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=349 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 349: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 350 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 350: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 350 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=350 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 350: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 351 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 351: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 351 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=351 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 351: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 352 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 352: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 352 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=352 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 352: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 353 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 353: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 353 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=353 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 353: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 354 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 354: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 354 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=354 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 354: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 355 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 355: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 355 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=355 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 355: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 356 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 356: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 356 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=356 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 356: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 357 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 357: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 357 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=357 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 357: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 358 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 358: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 358 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=358 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 358: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 359 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 359: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKDkd,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 359 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=359 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 359: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 360 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 360: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 360 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=360 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 360: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 361 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 361: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 361 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=361 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 361: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 362 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 362: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 362 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=362 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 362: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 363 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 363: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 363 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=363 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 363: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 364 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 364: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 364 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=364 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 364: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 365 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 365: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 365 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=365 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 365: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 366 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 366: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 366 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=366 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 366: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 367 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 367: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 367 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=367 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 367: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 368 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 368: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 368 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=368 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 368: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 369 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 369: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 369 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=369 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 369: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 370 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 370: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 370 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=370 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 370: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 371 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 371: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 371 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=371 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 371: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 372 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 372: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 372 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=372 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 372: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 373 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 373: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 373 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=373 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 373: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 374 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 374: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 374 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=374 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 374: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 375 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 375: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 375 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=375 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 375: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 376 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 376: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 376 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=376 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 376: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 377 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 377: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 377 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=377 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 377: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 378 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 378: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 378 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=378 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 378: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 379 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 379: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 379 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=379 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 379: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 380 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 380: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 380 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=380 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 380: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 381 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 381: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 381 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=381 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 381: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 382 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 382: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 382 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=382 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 382: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 383 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 383: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 383 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=383 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 383: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 384 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 384: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 384 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=384 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 384: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 385 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 385: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 385 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=385 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 385: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 386 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 386: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 386 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=386 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 386: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 387 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 387: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 387 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=387 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 387: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 388 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 388: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 388 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=388 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 388: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 389 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 389: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 389 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=389 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 389: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 390 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 390: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 390 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=390 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 390: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 391 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 391: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 391 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=391 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 391: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 392 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 392: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 392 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=392 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 392: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 393 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 393: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 393 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=393 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 393: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 394 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 394: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 394 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=394 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 394: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 395 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 395: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKHkh,wtzyxMGmg,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 395 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=395 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 395: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 396 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 396: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 396 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=396 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 396: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 397 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 397: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 397 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=397 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 397: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 398 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 398: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 398 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=398 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 398: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 399 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 399: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 399 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=399 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 399: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 400 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 400: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 400 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=400 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 400: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 401 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 401: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 401 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=401 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 401: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 402 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 402: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 402 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=402 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 402: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 403 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 403: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 403 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=403 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 403: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 404 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 404: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 404 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=404 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 404: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 405 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 405: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 405 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=405 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 405: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 406 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 406: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 406 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=406 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 406: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 407 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 407: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 407 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=407 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 407: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 408 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 408: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 408 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=408 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 408: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 409 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 409: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 409 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=409 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 409: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 410 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 410: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 410 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=410 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 410: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 411 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 411: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 411 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=411 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 411: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 412 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 412: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 412 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=412 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 412: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 413 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 413: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 413 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=413 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 413: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 414 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 414: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 414 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=414 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 414: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 415 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 415: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNGng,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 415 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=415 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 415: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 416 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 416: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRGrg,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 416 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=416 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 416: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 417 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 417: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRGrg,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 417 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=417 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 417: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 418 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 418: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 418 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=418 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 418: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 419 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 419: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNGng,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 419 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=419 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 419: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 420 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 420: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 420 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=420 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 420: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 421 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 421: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 421 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=421 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 421: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 422 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 422: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 422 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=422 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 422: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 423 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 423: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 423 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=423 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 423: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 424 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 424: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 424 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=424 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 424: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 425 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 425: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 425 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=425 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 425: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 426 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 426: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 426 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=426 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 426: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 427 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 427: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 427 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=427 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 427: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 428 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 428: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 428 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=428 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 428: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 429 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 429: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 429 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=429 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 429: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 430 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 430: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 430 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=430 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 430: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 431 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 431: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLFlf,wtzyxKIki,wtzyxMGmg,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 431 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=431 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 431: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 432 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 432: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 432 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=432 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 432: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 433 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 433: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 433 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=433 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 433: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 434 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 434: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 434 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=434 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 434: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 435 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 435: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 435 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=435 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 435: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 436 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 436: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 436 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=436 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 436: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 437 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 437: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 437 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=437 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 437: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 438 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 438: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 438 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=438 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 438: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 439 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 439: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 439 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=439 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 439: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 440 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 440: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 440 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=440 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 440: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 441 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 441: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 441 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=441 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 441: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 442 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 442: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 442 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=442 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 442: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 443 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 443: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 443 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=443 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 443: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 444 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 444: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 444 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=444 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 444: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 445 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 445: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 445 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=445 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 445: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 446 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 446: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 446 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=446 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 446: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 447 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 447: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 447 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=447 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 447: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 448 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 448: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 448 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=448 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 448: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 449 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 449: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 449 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=449 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 449: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 450 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 450: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 450 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=450 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 450: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 451 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 451: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 451 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=451 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 451: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 452 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 452: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 452 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=452 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 452: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 453 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 453: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 453 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=453 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 453: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 454 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 454: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 454 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=454 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 454: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 455 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 455: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 455 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=455 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 455: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 456 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 456: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 456 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=456 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 456: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 457 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 457: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 457 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=457 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 457: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 458 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 458: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 458 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=458 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 458: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 459 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 459: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 459 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=459 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 459: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 460 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 460: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 460 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=460 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 460: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 461 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 461: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 461 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=461 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 461: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 462 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 462: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 462 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=462 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 462: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 463 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 463: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 463 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=463 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 463: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 464 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 464: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 464 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=464 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 464: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 465 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 465: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 465 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=465 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 465: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 466 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 466: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 466 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=466 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 466: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 467 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 467: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKBkb,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 467 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=467 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 467: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 468 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 468: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 468 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=468 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 468: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 469 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 469: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 469 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=469 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 469: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 470 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 470: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 470 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=470 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 470: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 471 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 471: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 471 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=471 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 471: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 472 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 472: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 472 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=472 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 472: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 473 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 473: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 473 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=473 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 473: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 474 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 474: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 474 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=474 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 474: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 475 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 475: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 475 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=475 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 475: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 476 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 476: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 476 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=476 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 476: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 477 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 477: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 477 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=477 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 477: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 478 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 478: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 478 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=478 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 478: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 479 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 479: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 479 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=479 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 479: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 480 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 480: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 480 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=480 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 480: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 481 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 481: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 481 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=481 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 481: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 482 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 482: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 482 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=482 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 482: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 483 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 483: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 483 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=483 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 483: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 484 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 484: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 484 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=484 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 484: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 485 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 485: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 485 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=485 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 485: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 486 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 486: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 486 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=486 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 486: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 487 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 487: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 487 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=487 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 487: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 488 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 488: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 488 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=488 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 488: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 489 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 489: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 489 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=489 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 489: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 490 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 490: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 490 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=490 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 490: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 491 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 491: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 491 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=491 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 491: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 492 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 492: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 492 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=492 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 492: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 493 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 493: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 493 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=493 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 493: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 494 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 494: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 494 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=494 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 494: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 495 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 495: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 495 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=495 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 495: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 496 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 496: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 496 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=496 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 496: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 497 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 497: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 497 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=497 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 497: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 498 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 498: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 498 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=498 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 498: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 499 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 499: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 499 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=499 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 499: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 500 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 500: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 500 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=500 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 500: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 501 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 501: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 501 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=501 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 501: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 502 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 502: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 502 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=502 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 502: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 503 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 503: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKDkd,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 503 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=503 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 503: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 504 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 504: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 504 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=504 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 504: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 505 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 505: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 505 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=505 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 505: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 506 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 506: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 506 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=506 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 506: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 507 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 507: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 507 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=507 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 507: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 508 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 508: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 508 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=508 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 508: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 509 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 509: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 509 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=509 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 509: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 510 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 510: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 510 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=510 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 510: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 511 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 511: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 511 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=511 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 511: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 512 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 512: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 512 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=512 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 512: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 513 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 513: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 513 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=513 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 513: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 514 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 514: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 514 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=514 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 514: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 515 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 515: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMAma,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 515 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=515 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 515: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 516 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 516: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 516 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=516 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 516: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 517 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 517: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 517 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=517 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 517: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 518 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 518: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 518 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=518 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 518: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 519 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 519: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 519 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=519 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 519: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 520 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 520: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 520 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=520 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 520: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 521 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 521: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 521 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=521 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 521: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 522 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 522: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 522 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=522 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 522: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 523 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 523: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 523 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=523 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 523: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 524 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 524: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 524 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=524 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 524: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 525 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 525: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 525 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=525 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 525: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 526 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 526: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 526 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=526 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 526: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 527 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 527: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMEme,wtzyxOIoi,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 527 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=527 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 527: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 528 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 528: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 528 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=528 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 528: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 529 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 529: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 529 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=529 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 529: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 530 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 530: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 530 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=530 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 530: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 531 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 531: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 531 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=531 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 531: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 532 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 532: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 532 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=532 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 532: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 533 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 533: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPIpi,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 533 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=533 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 533: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 534 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 534: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQIqi->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 534 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=534 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 534: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 535 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 535: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPIpi,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 535 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=535 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 535: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 536 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 536: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 536 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=536 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 536: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 537 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 537: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 537 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=537 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 537: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 538 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 538: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 538 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=538 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 538: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 539 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 539: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKHkh,wtzyxMFmf,wtzyxOIoi,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 539 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=539 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 539: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 540 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 540: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 540 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=540 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 540: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 541 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 541: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 541 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=541 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 541: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 542 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 542: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 542 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=542 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 542: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 543 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 543: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 543 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=543 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 543: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 544 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 544: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 544 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=544 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 544: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 545 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 545: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 545 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=545 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 545: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 546 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 546: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 546 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=546 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 546: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 547 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 547: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 547 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=547 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 547: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 548 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 548: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 548 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=548 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 548: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 549 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 549: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 549 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=549 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 549: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 550 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 550: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 550 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=550 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 550: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 551 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 551: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMAma,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 551 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=551 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 551: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 552 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 552: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 552 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=552 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 552: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 553 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 553: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 553 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=553 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 553: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 554 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 554: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 554 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=554 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 554: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 555 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 555: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOBob,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 555 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=555 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 555: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 556 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 556: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 556 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=556 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 556: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 557 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 557: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 557 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=557 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 557: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 558 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 558: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 558 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=558 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 558: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 559 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 559: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxODod,wtzyxNFnf,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 559 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=559 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 559: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 560 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 560: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxRFrf,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 560 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=560 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 560: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 561 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 561: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxRFrf,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 561 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=561 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 561: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 562 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 562: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 562 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=562 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 562: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 563 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 563: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMEme,wtzyxOHoh,wtzyxNFnf,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 563 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=563 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 563: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 564 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 564: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 564 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=564 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 564: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 565 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 565: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 565 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=565 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 565: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 566 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 566: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 566 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=566 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 566: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 567 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 567: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOBob,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 567 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=567 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 567: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 568 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 568: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 568 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=568 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 568: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 569 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 569: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNAna,wtzyxPHph,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 569 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=569 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 569: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 570 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 570: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQHqh->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 570 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=570 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 570: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 571 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 571: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxODod,wtzyxNEne,wtzyxPHph,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 571 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=571 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 571: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 572 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 572: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPBpb,wtzyxREre,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 572 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=572 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 572: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 573 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 573: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNAna,wtzyxPDpd,wtzyxREre,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 573 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=573 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 573: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 574 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 574: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPBpb,wtzyxRAra,wtzyxQDqd->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 574 dp
    two_pt_site += res_gpu
    del res_gpu
    print('topo=574 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 574: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

try:
    try:
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        )  # topo 575 greedy
    except (cp.cuda.memory.OutOfMemoryError, MemoryError):
        print('topo 575: OOM, retry dp...')
        cp.get_default_memory_pool().free_all_blocks()
        res_gpu = contract('AB,abc,DE,def,GH,ghi,KL,klj,NO,nom,QR,qrp,JC,MF,PI,wtzyxJCjc,wtzyxLGlg,wtzyxKIki,wtzyxMFmf,wtzyxOHoh,wtzyxNEne,wtzyxPDpd,wtzyxRAra,wtzyxQBqb->t',
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Cg5, epsilon, Cg5,
            epsilon, Cg5, epsilon,
            Tmat, Tmat, Tmat,
            prop_s.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
            prop_l.data, prop_l.data, prop_l.data,
        optimize='dp',
        )  # topo 575 dp
    two_pt_site -= res_gpu
    del res_gpu
    print('topo=575 is finished')
except (cp.cuda.memory.OutOfMemoryError, MemoryError):
    print('topo 575: both greedy and dp OOM, skipping')
cp.cuda.Stream.null.synchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()


# Trace spatial volume → time-slice
two_pt_local = two_pt_site

# MPI gather
from pyquda_comm import array
two_pt_result = core.gatherLattice(array.arrayAsNumpy(two_pt_local, backend='cupy'), [0, -1, -1, -1])
