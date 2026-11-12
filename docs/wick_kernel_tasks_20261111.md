# wick_12cases 专用内核任务清单

来源：`refer/sush/wick_12cases.md`，共 **8638** 条收缩式（基础表 2582 + 附录 6056），
每条一个专用内核任务。行号为该文件中的实际行号。

**机器可读任务表**：`pyqcd/einsum/_wick_tasklist.py`（惰性解析，`T.N_TASKS` / `T.TASKS` / `T.EQUATIONS` / `T.by_section()`），供批量流水线直接消费；不再另存 JSON 副本。

## 分组统计

| 情形 | 子表 | 条数 |
|---|---|---:|
| 1. 介子两点函数 |  | 2 |
| 10. 重子四点函数 |  | 120 |
| 13. 五夸克到三夸克三点函数 |  | 120 |
| 14. 五夸克到三夸克四点函数 |  | 720 |
| 2. 重子两点函数 |  | 6 |
| 3. 五夸克多粒子两点函数 |  | 120 |
| 4. 六夸克多粒子两点函数 |  | 720 |
| 5. 介子三点函数 |  | 6 |
| 6. 重子三点函数 |  | 24 |
| 7. 五夸克多粒子三点函数 |  | 720 |
| 9. 介子四点函数 |  | 24 |
| 动量指标与流分量指标 | 情形 10：重子四点函数 | 240 |
| 动量指标与流分量指标 | 情形 13：五夸克到三夸克三点函数 | 480 |
| 动量指标与流分量指标 | 情形 14：五夸克到三夸克四点函数 | 1440 |
| 动量指标与流分量指标 | 情形 1：介子两点函数 | 2 |
| 动量指标与流分量指标 | 情形 2：重子两点函数 | 6 |
| 动量指标与流分量指标 | 情形 3：五夸克多粒子两点函数 | 120 |
| 动量指标与流分量指标 | 情形 4：六夸克多粒子两点函数 | 720 |
| 动量指标与流分量指标 | 情形 5：介子三点函数 | 24 |
| 动量指标与流分量指标 | 情形 6：重子三点函数 | 96 |
| 动量指标与流分量指标 | 情形 7：五夸克多粒子三点函数 | 2880 |
| 动量指标与流分量指标 | 情形 9：介子四点函数 | 48 |

## 任务清单（前 40 条示例，全量见 wick_tasks.json）

| # | 行号 | 情形 | T号 | 元数 | 收缩式 |
|---:|---:|---|---|---:|---|
| 1 | 46 | 1. | T00001 | 6 | `cadb,menf,ac,em,bd,fn->` |
| 2 | 47 | 1. | T00002 | 6 | `cedf,manb,ac,em,bd,fn->` |
| 3 | 61 | 2. | T00001 | 7 | `ambn,codp,egfh,ce,mo,bdf,nph->ag` |
| 4 | 62 | 2. | T00002 | 7 | `ambn,cgdh,eofp,ce,mo,bdf,nph->ag` |
| 5 | 63 | 2. | T00003 | 7 | `aobp,cmdn,egfh,ce,mo,bdf,nph->ag` |
| 6 | 64 | 2. | T00004 | 7 | `aobp,cgdh,emfn,ce,mo,bdf,nph->ag` |
| 7 | 65 | 2. | T00005 | 7 | `agbh,cmdn,eofp,ce,mo,bdf,nph->ag` |
| 8 | 66 | 2. | T00006 | 7 | `agbh,codp,emfn,ce,mo,bdf,nph->ag` |
| 9 | 80 | 3. | T00001 | 13 | `cadb,egfh,minj,okpl,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 10 | 81 | 3. | T00002 | 13 | `cadb,egfh,minj,oqpr,sktl,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 11 | 82 | 3. | T00003 | 13 | `cadb,egfh,mknl,oipj,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 12 | 83 | 3. | T00004 | 13 | `cadb,egfh,mknl,oqpr,sitj,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 13 | 84 | 3. | T00005 | 13 | `cadb,egfh,mqnr,oipj,sktl,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 14 | 85 | 3. | T00006 | 13 | `cadb,egfh,mqnr,okpl,sitj,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 15 | 86 | 3. | T00007 | 13 | `cadb,eifj,mgnh,okpl,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 16 | 87 | 3. | T00008 | 13 | `cadb,eifj,mgnh,oqpr,sktl,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 17 | 88 | 3. | T00009 | 13 | `cadb,eifj,mknl,ogph,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 18 | 89 | 3. | T00010 | 13 | `cadb,eifj,mknl,oqpr,sgth,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 19 | 90 | 3. | T00011 | 13 | `cadb,eifj,mqnr,ogph,sktl,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 20 | 91 | 3. | T00012 | 13 | `cadb,eifj,mqnr,okpl,sgth,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 21 | 92 | 3. | T00013 | 13 | `cadb,ekfl,mgnh,oipj,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 22 | 93 | 3. | T00014 | 13 | `cadb,ekfl,mgnh,oqpr,sitj,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 23 | 94 | 3. | T00015 | 13 | `cadb,ekfl,minj,ogph,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 24 | 95 | 3. | T00016 | 13 | `cadb,ekfl,minj,oqpr,sgth,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 25 | 96 | 3. | T00017 | 13 | `cadb,ekfl,mqnr,ogph,sitj,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 26 | 97 | 3. | T00018 | 13 | `cadb,ekfl,mqnr,oipj,sgth,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 27 | 98 | 3. | T00019 | 13 | `cadb,eqfr,mgnh,oipj,sktl,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 28 | 99 | 3. | T00020 | 13 | `cadb,eqfr,mgnh,okpl,sitj,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 29 | 100 | 3. | T00021 | 13 | `cadb,eqfr,minj,ogph,sktl,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 30 | 101 | 3. | T00022 | 13 | `cadb,eqfr,minj,okpl,sgth,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 31 | 102 | 3. | T00023 | 13 | `cadb,eqfr,mknl,ogph,sitj,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 32 | 103 | 3. | T00024 | 13 | `cadb,eqfr,mknl,oipj,sgth,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 33 | 104 | 3. | T00025 | 13 | `cgdh,eafb,minj,okpl,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 34 | 105 | 3. | T00026 | 13 | `cgdh,eafb,minj,oqpr,sktl,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 35 | 106 | 3. | T00027 | 13 | `cgdh,eafb,mknl,oipj,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 36 | 107 | 3. | T00028 | 13 | `cgdh,eafb,mknl,oqpr,sitj,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 37 | 108 | 3. | T00029 | 13 | `cgdh,eafb,mqnr,oipj,sktl,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 38 | 109 | 3. | T00030 | 13 | `cgdh,eafb,mqnr,okpl,sitj,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 39 | 110 | 3. | T00031 | 13 | `cgdh,eifj,manb,okpl,sqtr,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
| 40 | 111 | 3. | T00032 | 13 | `cgdh,eifj,manb,oqpr,sktl,ac,mo,gi,qs,bd,fnp,hjl,rt->ek` |
