# 格点 QCD 特化方案进度简报（2026-11-08 08:00）

> "特化"= 针对格点 QCD 常见收缩形态的**专用执行路径**（方程匹配 → 专用内核 / 计划缓存），
> 命中时打印提示，可用参数关闭以便 A/B 对照。

## 1. 一句话结论

特化框架已可用，但**真正跑赢通用路径的只有 2 条**（E5 夹心、E4 规范场，且 E4 仅限极小规模）；
另有 1 条（E4）在大规模下**反而劣化 33 倍**（已登记为 B5 bug）；
其余 5 条只是计划缓存，**没有专用内核**。

## 2. 特化清单与实测收益

| 形态 | 类型 | 实装内容 | 实测（vs 通用路径） | 是否值得接线 |
|---|---|---|---|---|
| **E5** `ab,...bce,cd->...dae`（γ 夹心） | fused | sandwich kernel（CPU+GPU） | CPU L24：**142.7 ms vs 1188.9 ms = 8.33x**；GPU：快 3.3–3.4x | **是**（已生效） |
| **E4** `zyxcTabn,zyxgTdef,Gae,Tzyxgc->TGdbfn` | fused | e4_fused kernel（CUDA） | spatial=1：0.0003 vs 0.0009 s（**3.6x**）；**spatial≥2 全线劣化** | 仅极小规模可用，**需加收益判据（B5）** |
| E1 `gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M` | plan-cache | 仅复用计划缓存 | 无专用执行 | 无收益（标注如实） |
| E2 `Geifj,ckdl,agbh,ce,gi,Mbdf,Mhjl,Gka->GM` | plan-cache | 同上 | 无 | 无 |
| S1 `eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN` | plan-cache | 同上 | 无 | 无 |
| S2 `gahb,menf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG` | plan-cache | 同上 | 无 | 无 |
| S3 `manb,cedf,bd,Mfn,ac,MFem->MF` | plan-cache | 同上 | 无 | 无 |
| P1 `...15 算子...->MNGLxy` | multi-step | 仅标记（命中打印） | 无专用执行 | 待 bg-17/bg-19 |
| P2 `...15 算子（置换型）...` | multi-step | 同上 | 无 | 待 bg-17/bg-19 |

> 命名已修正：E1/E2/S1/S2/S3 原先标为 `fused-plan`（暗示有专用内核），现已改为如实的 `plan-cache`。

## 3. 机制与开关

| 能力 | 实现 |
|---|---|
| 注册表 | `pyqcd/einsum/_specialize.py`（9 条形态） |
| 命中提示 | `[pyqcd.einsum] 专用路径: <name> [<kind>] <note>`（每形态每进程一次；`PYQCD_EINSUM_SPECIALIZE_VERBOSE=1` 每次打印） |
| 单次关闭 | `contract(..., specialize=False)` |
| 进程关闭 | `PYQCD_EINSUM_SPECIALIZE=0` |
| 内核级开关 | `PYQCD_EINSUM_DISABLE_FUSED=1`（CUDA 在 `initialize_context`、KUPL 在 `try_sandwich_fast_path` 检查） |
| 公开查询 | `pe.specializations()` |

**A/B 已验证双向生效**（修复 B2 后）：

| 后端 | `specialize=False` | `True` | `False` | `True` |
|---|---|---|---|---|
| CUDA E4 | 1.808 ms | **0.735 ms** | 1.611 ms | **0.746 ms** |
| KUPL E5 | 1266.82 ms | **112.68 ms** | 1006.71 ms | — |

数值在四次调用中完全一致（CUDA `out_sum=9.410919e+03`，KUPL `-1.343492e+04`）。

## 4. 关键缺陷：E4 特化在大规模下失效（B5）

| spatial s | z·y·x | fused (s) | generic (s) | fused/generic |
|---|---|---|---|---|
| 1 | 1 | 0.0003 | 0.0009 | **0.28（融合快 3.6x）** |
| 2 | 8 | 0.0040 | 0.0028 | 1.44 |
| 3 | 27 | 0.0706 | 0.0073 | **9.65** |
| 4 | 64 | 0.2891 | 0.0131 | **22.1** |
| 8 | 512 | 1.0812 | 0.0415 | **26.1** |
| 12 | 1728 | 9.4835 | 0.4977 | **19.1** |
| 24（L24 生产） | 13824 | **36.44** | **1.104** | **33.0** |

- 按用户口径（劣于 opt_einsum 即 bug）：L24 上 opt ≈ 1.09–1.14 s，融合 36.4 s → **慢 32–33 倍**。
- 子代理 bg-6 另记录：默认融合路径下 `qcd_einsum` worker `returncode=-11`（SIGSEGV），
  关闭 fused 后不崩且回到 0.980x。
- **修复方案（待落地）**：在 `match_e4_fused` 加收益判据——当前数据只在 `z·y·x == 1` 时融合有优势，
  其余规模应回落通用多步。

## 5. 子代理在做的 4 个专用内核

| 代理 | 形态 | 现状 | 验收判据（你的规则） |
|---|---|---|---|
| bg-17 | P1/P2（KUPL，15 算子） | `.wip`，运行期段错误待修 | 必须 < opt 的 183.65 ms |
| bg-18 | E1/S1（CUDA 2pt） | `.wip`，5 个编译错误 | 必须优于 opt 的 380–480 ms |
| bg-19 | P1/P2（CUDA，15 算子） | `.wip`，100+ 编译错误 | 必须优于 opt，且不得让构建变红 |
| bg-20 | E1/S1（KUPL 2pt 优化） | 可用但 0.78–0.89x 通用路径 | N=100 必须 < 471 ms |

验收工具：`python -m pyqcd.testing.einsum.simple.kernel_acceptance`
（判据：慢于 opt → BUG；快于 opt 但慢于通用 → OK-WITH-CAVEAT；快于两者 → PASS）。

当前唯一可判定的 kernel：

```
[kupl_nucleon] nvec=50 rel=1.92e-15
  kernel 45.19 ms | generic 40.40 ms (0.89x) | opt 129.10 ms (2.86x)
  → OK-WITH-CAVEAT（快于 opt，慢于通用路径，暂不接线）
```

## 6. 特化在真实生产中的价值（按耗时占比）

| 组 | 占 pyqcd 生产时间 | 当前特化状态 |
|---|---|---|
| E1（2pt） | 42.6% | 无专用内核（plan-cache） |
| P1（3pt 主项） | 42.1% | 无（bg-17/bg-19 在做） |
| P2 | 8.3% | 无（同上） |
| E4 | 6.7% | fused 仅在 z·y·x=1 有效，L24 需回落 |
| E5 | 0.4% | **fused 已生效，8.3x** |

→ 结论：**占 84.7% 的 E1+P1+P2 目前没有可用的专用内核**，这是特化方案的主要缺口。

## 7. 下一步

1. **修 B5**：加收益判据，让 L24 回落通用路径（≈opt 1.0x），同时消除崩溃风险。
2. 收 bg-17/18/19/20 的 kernel，用 `kernel_acceptance` 逐个判定，不合格打回。
3. 只接线"快于通用路径"的特化；快于 opt 但慢于通用的保持零引用并登记。
4. 接线后重跑 `verify_native` 双后端 + `verify_compat` 980 项 + 生产规模 9 组方程。

---

## 8. 更新（08:30）：E1 专用 kernel 已接线并生效

### 8.1 变化

| 项 | 之前 | 现在 |
|---|---|---|
| E1（核子 2pt） | `plan-cache`（无专用执行） | **`fused`：KUPL 专用 kernel 已接线** |
| 注册表 `kind` | plan-cache | `fused`（`pe.specializations()` 已同步） |

### 8.2 实现

| 环节 | 文件 |
|---|---|
| kernel 实现 | `cpp/kupl/einsum/src/nucleon2pt.cpp`（bg-20 优化后版本） |
| 符号导出 | `cpp/kupl/einsum/make.sh` 的 version script 新增 `qcd_kupl_nucleon_2pt` |
| ctypes 绑定 + 严格形态校验 | 新增 `pyqcd/einsum/_kernel_kupl.py` |
| 分派 | `pyqcd/einsum/_api.py::contract`（命中且 `out is None` 时尝试，失败即回退） |

**严格校验条件**（任一不满足即回退通用路径，不猜测）：
下标规范化后必须等于 `gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M`；
8 个操作数、必须为 numpy `ndarray`；`complex128` 且 C 连续；
3 个 peram 均为 `(4,4,N,N)` 且 N 相同；两个 γ 与投影为 `(4,4)`；
两个 VVV 为 `(M,N,N,N)` 且 M 相同。

### 8.3 实测

| Nvec | kernel | 通用路径 | opt_einsum | kernel/通用 | kernel/opt | rel |
|---|---|---|---|---|---|---|
| 50 | 34.59–43.37 ms | 58.25–72.41 ms | 157–169 ms | **1.34–2.09x** | 3.62–4.87x | 1.56–1.92e-15 |
| **100（生产）** | 458.78–464.34 ms | 504.53–652.78 ms | 3331–4324 ms | **1.10–1.41x** | **7.17–9.42x** | 1.81–3.85e-15 |

### 8.4 回退验证与回归

| 场景 | 结果 |
|---|---|
| 非连续（Fortran order）数组 | 回退，rel 1.01e-15 |
| `complex64` 输入 | 回退，dtype 保持，rel 1.66e-06（单精度正常量级） |
| `PYQCD_EINSUM_KERNEL_KUPL=0` | 回退，rel 7.52e-16 |
| `verify_native --backend kupl` | 16/16 |
| `verify_compat` 全量 | **980/980 PASS** |
| `simple/run-all.sh` | 78 passed，exit=0 |

### 8.5 仍未接线的形态

| 形态 | 状态 |
|---|---|
| E1（S1 变体，variant=1） | kernel 声明支持，但尚未验证与接线 |
| CUDA 2pt / 3pt | `.wip`（K1/K2） |
| KUPL 3pt | 文件被 bg-17 取回重做中（K3） |
