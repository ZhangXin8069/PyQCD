# wick 专用内核：25 个可套用族（实现模板清单）

由 `pyqcd/einsum/_wick_tasklist.py` 的 8638 条任务按
「元数 + 各操作数 rank 多重集 + 输出 rank」分组得到。
同族内实现可套用同一份模板，差异只在指标槽位的连接关系（由步表描述）。

| 族 | 条数 | 元数 | 输出rank | ranks 多重集 | 代表性式子 |
|---:|---:|---:|---:|---|---|
| F01 | 1440 | 16 | 2 | 2,2,2,2,2,2,2,2,3,3,4,4,4,4,4,4 | `T00001` L1008: `cadb,egfh,mknl,oqpr,isjt,wuxv,ac,mo,gi,kq,uw,bd,fnp,hj,lrt,vx->es` |
| F02 | 1440 | 16 | 4 | 2,2,2,2,2,3,3,3,4,4,4,4,4,4,4,4 | `T00001` L3768: `cadb,egfh,mknl,oqpr,isjt,wuxv,ac,mo,gi,kq,uw,Nbd,Nfnp,Nhj,Mlrt,Mvx->MNes` |
| F03 | 1440 | 16 | 5 | 2,2,2,2,3,3,3,3,4,4,4,4,4,4,4,4 | `T00001` L4678: `cadb,egfh,mknl,oqpr,isjt,wuxv,ac,mo,Ggi,kq,uw,Nbd,Nfnp,Nhj,Mlrt,Mvx->MNGes` |
| F04 | 720 | 14 | 4 | 2,2,2,2,3,3,3,3,4,4,4,4,4,4 | `T00001` L213: `aibj,ckdl,eqfr,msnt,oupv,gwhx,ce,og,ik,su,bdf,nph,jlr,tvx->aqmw` |
| F05 | 720 | 14 | 6 | 2,2,2,2,4,4,4,4,4,4,4,4,4,4 | `T00001` L2977: `aibj,ckdl,eqfr,msnt,oupv,gwhx,ce,og,ik,su,Nbdf,Nnph,Mjlr,Mtvx->NMaqmw` |
| F06 | 720 | 16 | 3 | 2,2,2,2,2,2,3,3,3,3,4,4,4,4,4,4 | `T00001` L7528: `cadb,egfh,mknl,ospt,iujv,qwrx,ac,mo,gi,kq,su,bd,fnp,Mhj,Mlr,tvx->Mew` |
| F07 | 720 | 16 | 5 | 2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4 | `T00001` L8423: `cadb,egfh,mknl,ospt,iujv,qwrx,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew` |
| F08 | 360 | 13 | 2 | 2,2,2,2,2,2,3,3,4,4,4,4,4 | `T00001` L80: `cadb,egfh,minj,okpl,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| F09 | 360 | 13 | 4 | 2,2,2,2,3,3,4,4,4,4,4,4,4 | `T00001` L2848: `cadb,egfh,minj,okpl,sqtr,ac,mo,gi,qs,Nbd,Nfnp,Mhjl,Mrt->NMek` |
| F10 | 240 | 13 | 5 | 2,2,2,3,3,3,4,4,4,4,4,4,4 | `T00001` L5407: `cadb,egfh,mknl,oqpr,isjt,ac,mo,Ggi,kq,Nbd,Nfnp,Nhj,Mlrt->MNGes` |
| F11 | 120 | 13 | 3 | 2,2,2,2,3,3,3,3,4,4,4,4,4 | `T00001` L7399: `ambn,cgdh,ekfl,oqpr,isjt,ce,mo,gi,kq,bdf,Mnp,Mhj,lrt->Mas` |
| F12 | 120 | 13 | 5 | 2,2,3,3,3,3,3,3,4,4,4,4,4 | `T00001` L8294: `ambn,cgdh,ekfl,oqpr,isjt,ce,Gmo,Hgi,kq,bdf,Mnp,Mhj,lrt->MGHas` |
| F13 | 48 | 10 | 4 | 2,2,2,3,4,4,4,4,4,4 | `T00001` L3735: `ambn,cgdh,eifj,okpl,ce,mo,gi,Nbdf,Nnp,Mhjl->MNak` |
| F14 | 48 | 10 | 5 | 2,2,3,3,4,4,4,4,4,4 | `T00001` L4645: `ambn,cgdh,eifj,okpl,ce,Gmo,gi,Nbdf,Nnp,Mhjl->MNGak` |
| F15 | 24 | 10 | 2 | 2,2,2,2,3,3,4,4,4,4 | `T00001` L969: `ambn,cgdh,eifj,okpl,ce,mo,gi,bdf,np,hjl->ak` |
| F16 | 24 | 12 | 0 | 2,2,2,2,2,2,2,2,4,4,4,4 | `T00001` L1743: `cadb,menf,gohp,kilj,ac,em,og,ik,bd,fn,ph,jl->` |
| F17 | 24 | 12 | 1 | 2,2,2,2,2,2,3,3,4,4,4,4 | `T00001` L7366: `cadb,menf,gohp,kilj,ac,em,og,ik,bd,Mfn,Mph,jl->M` |
| F18 | 24 | 12 | 3 | 2,2,2,2,3,3,3,3,4,4,4,4 | `T00001` L8261: `cadb,menf,gohp,kilj,ac,Gem,Hog,ik,bd,Mfn,Mph,jl->MGH` |
| F19 | 12 | 9 | 2 | 2,2,2,3,3,3,4,4,4 | `T00001` L3720: `cadb,menf,gohp,ac,em,og,Nbd,Nfn,Mph->MN` |
| F20 | 12 | 9 | 3 | 2,2,3,3,3,3,4,4,4 | `T00001` L4630: `cadb,menf,gohp,ac,Gem,og,Nbd,Nfn,Mph->MNG` |
| F21 | 6 | 7 | 2 | 2,2,3,3,4,4,4 | `T00001` L61: `ambn,codp,egfh,ce,mo,bdf,nph->ag` |
| F22 | 6 | 9 | 0 | 2,2,2,2,2,2,4,4,4 | `T00001` L948: `cadb,menf,gohp,ac,em,og,bd,fn,ph->` |
| F23 | 6 | 7 | 4 | 2,2,4,4,4,4,4 | `T00001` L2831: `ambn,codp,egfh,ce,mo,Mbdf,Nnph->MNag` |
| F24 | 2 | 6 | 0 | 2,2,2,2,4,4 | `T00001` L46: `cadb,menf,ac,em,bd,fn->` |
| F25 | 2 | 6 | 2 | 2,2,3,3,4,4 | `T00001` L2820: `cadb,menf,ac,em,Mbd,Nfn->MN` |

合计：8638 条 / 25 族。
