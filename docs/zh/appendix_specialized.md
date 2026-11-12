# 附录：所有支持的特化路径

本附录列出的收缩全部按**拓扑签名**分派（与字母无关），因此 `ij,jk->ik` 与
`ij,jb->ib` 走同一条路径。数字由语料与注册表自动生成，不是手写。

* pyqcd 版本：`0.3.5`（后端 `cuda`，数组后端 `numpy`）
* 语料规模：**8638** 条收缩（基础表 2582 + 附录变体 6056）
* 拓扑索引：**8638** 个签名，分 **24** 个桶

## 1. 语料覆盖（按情形）

| 情形 | 收缩条数 |
|---|---:|
| 1. | 2 |
| 10. | 120 |
| 13. | 120 |
| 14. | 720 |
| 2. | 6 |
| 3. | 120 |
| 4. | 720 |
| 5. | 6 |
| 6. | 24 |
| 7. | 720 |
| 9. | 24 |
| 动量指标与流分量指标 | 6056 |

## 2. 覆盖（按算子元数）

| 算子数 | 收缩条数 |
|---:|---:|
| 6 | 4 |
| 7 | 12 |
| 9 | 30 |
| 10 | 120 |
| 12 | 72 |
| 13 | 1200 |
| 14 | 1440 |
| 16 | 5760 |

## 3. 字面注册表（手工列出的形态）

这些先按**精确字符串**匹配，再走拓扑查表。`kind` 表示实际执行什么：
`kernel(...)` 是原生专用内核，`plan-cache` 只复用计划。

| 名称 | 类型 | 方程 |
|---|---|---|
| gauge-propagator(E4) | `plan-cache` | `zyxcTabn,zyxgTdef,Gae,Tzyxgc->TGdbfn` |
| gamma-sandwich(E5) | `plan-cache` | `ab,...bce,cd->...dae` |
| sush-2pt(S1) | `plan-cache` | `eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN` |
| sush-3pt-munu(S2) | `plan-cache` | `gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG` |
| sush-2pt-mu(S3) | `plan-cache` | `manb,cedf,bd,Mfn,ac,MFem->MF` |

## 4. 怎么判断走了哪条路

| 情形 | 日志 | 速度 |
|---|---|---|
| 拓扑命中且有原生内核 | `专用路径: topology:<摘要> [kernel: kupl_dedicated]` | 比通用快 2–4× |
| 拓扑命中但无内核 | `专用路径: ... [plan-cache]` | 约等于通用 |
| 未命中特化 | `通用路径: 未命中特化形态 (...)` | 基线 |

两种提示都是**每形态每进程一次**。用 `PYQCD_EINSUM_NOTICE=0` 关闭；
用 `PYQCD_EINSUM_SPECIALIZE_VERBOSE=1` 让特化提示**每次命中**都打印。

## 5. 自行核验

```bash
python -m pyqcd.einsum._specialize                 # 列出注册形态并自检
python -m pyqcd.testing._topology_full_bench --dim 3  # 全语料跑一遍
python -m pyqcd.testing._topology_perf_multidim       # 多尺寸加速比
```
