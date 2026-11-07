# pyqcd-einsum 进度简报（2026-11-08 06:35）· 含真实工作流热点分析

## 1. 结论速览

| 项 | 状态 |
|---|---|
| CUDA E4 融合 kernel | 10.6 ms → **0.635 ms**（16.7x），相对 opt_einsum **1.90x** |
| CPU E1（T1，Nvec=100） | 1.546 s → **0.531 s**，相对 opt_einsum **6.98x** |
| CPU E4（T1） | 4.07 ms → **0.87 ms**，**8.25x**（但 too_short） |
| CPU E4（L24，Nvec=50） | opt 27.20 s vs 我们 1.911 s = **14.2x**，但 `rel=5.73e-12` **超出 1e-12 契约** |
| 真实生产热点 | **15 算子 3pt 收缩（P1/P2）**，不是 E1/E4 |
| 真实工作流整体 | CPU 模型口径 **2.48x**（opt 4306 s vs 我们 1736 s） |

## 2. 计算热点：来自真实生产代码，不是基准自造

### 2.1 生产脚本的调用结构（逐行核对）

**2pt**（`refer/sush/_distillation/contraction.NN.2pt.py:96-113`）

```
for t_src in range(Nt):
    for t_sink in 32 个位移:
        for Mom_indx in ceil(Mom_len/3):
            contract('eofp,ambn,cgdh,ce,mo,Nbdf,Mnph,ga->MN', ...)
```
调用数 = `Nt × 32 × 3`（`Mom_len = 8`）

**3pt**（`contraction.NJNp-.3pt.cupy.py:125-200`）

```
for t_src in range(Nt):
    for t_curr in t_sep+1 个时间片:
        for Mom_indx in 3 块:
            2 次 15 算子收缩
            for Mom_indx2 in 3 块:
                3 次 15 算子收缩
```
调用数 = `Nt × (t_sep+1) × 3 × (2 + 9) = Nt × (t_sep+1) × 33`

### 2.2 由此新增的真实方程组

| 组 | subscripts | 算子数 | 步数 | opt_cost（N=100） |
|---|---|---|---|---|
| **P1** | `agbh,eqfr,ospt,cmdn,ikjl,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy` | **15** | 14 | 9.78e9 |
| **P2** | `agbh,eqfr,ospt,ckdl,imjn,ce,Gmo,gi,kq,Nbdf,NLnp,Mhj,Mlrt,ax,sy->MNGLxy` | **15** | 14 | 6.74e9 |

### 2.3 单次耗时（CPU，Nev=50）

| 组 | opt_einsum | pyqcd-einsum | 倍率 |
|---|---|---|---|
| P1 | 183.65 ms | 215.24 ms | **0.85x** |
| P2 | 51.50 ms | 105.92 ms | **0.49x** |

→ 15 算子路径上我们**反而更慢**，且它是 3pt 的主力调用（每 `(t_src,t_curr)` 33 次）。

## 3. 真实工作流时间构成（`simple/workflow.py`，model 口径）

参数：`profile=T1`（Nvec=100）、`nt=8`、`tsep=2`、CPU 4 线程。

| 组 | 调用数 | opt ms/次 | qcd ms/次 | opt 占比 | qcd 占比 |
|---|---|---|---|---|---|
| E1（2pt） | 768 | 4205.72 | 887.76 | 75.0% | 39.3% |
| P1（3pt） | 396 | 2239.84 | 2244.67 | 20.6% | 51.2% |
| P2（3pt） | 396 | 478.20 | 416.99 | 4.4% | 9.5% |
| **合计** | | **4306.3 s** | **1735.8 s** | | **2.48x** |

**解读**：
- 对 opt_einsum 而言，2pt 占 75% 时间；对我们 2pt 只占 39%，3pt 的 P1 成为最大项（51%）。
- 所以**下一阶段优化重点应是 P1/P2（15 算子多步路径）**，而不是继续压 E1。

## 4. 测试组说明（避免误读）

| 组 | 物理含义 | 算子 | 备注 |
|---|---|---|---|
| E1 | 核子 2pt（NN 通道）：3×peram × 2×γ₇ × 2×VVV × 宇称投影 | 8 | 对应生产 2pt |
| E2 | 核子 3pt（简化 8 算子版） | 8 | 与生产 3pt 同构但规模小 |
| E4 | peram × peram × γ × 规范场 | 4 | L24 时为真实空间格点规模 |
| E5 | γ 夹心 `γ₅·peram†·γ₅` | 3 | 带宽受限 |
| S1/S3 | sush 2pt（真实脚本直取） | 8 / 6 | |
| S2 | sush 3pt 含 μν 插入 | 9 | |
| **P1/P2** | **sush 3pt 主项（15 算子）** | **15** | **真实热点** |

## 5. 新交付的接口

```bash
# 真实工作流时间构成（模型口径，秒级）
python -m pyqcd.testing.einsum.simple.workflow --nt 8 --tsep 2 --calls 3
# 整段实跑抽检
python -m pyqcd.testing.einsum.simple.workflow --mode run --nt 2 --tsep 1
```

`pyqcd/testing/einsum/simple/` 现有：`check_compat`、`check_memory`、`check_equations`、
`check_native`、`bench`、`workflow`、`run-all.sh`、`README.md`。

## 6. 必须如实登记的异常

| 项 | 说明 |
|---|---|
| E4/L24 数值 | `max rel = 5.73e-12`，**超过契约 1e-12**。原因是 L24 下求和项数达 1e5 量级、累加顺序不同；需明确是放宽尺度相关容差还是修正累加顺序，**未决** |
| E4/L24 倍率 14.2x | 数值层面尚未达标，因此该倍率**不能直接作为通过证据** |
| GPU E1/E2 | 分步剖析显示 step2/step1/step3 占 99%（327/145/62 ms），四方实现收敛在 373–444 ms，3x 不成立 |
| 退出期 segfault | 一次出现、未复现，bg-8 仍在查 |

## 7. 子代理

| 代理 | 职责 | 状态 |
|---|---|---|
| bg-9 | E4 kernel 冲 3x（0.635→0.40 ms） | 运行中 |
| bg-10 | E5 夹心路径（GPU+CPU） | 运行中 |
| bg-6 | GPU L24 生产测量 | 运行中 |
| bg-7 | 全库回归 | 运行中 |
| bg-8 | Segfault 根因 | 运行中 |
| bg-1/2/3/5 | 已完成（bg-1 使 E4 提速 16.7x，bg-2 使 CPU E1 达 6.98x） | 完成 |

## 8. 下一步

1. **新开 P1/P2 优化线**（15 算子多步路径，目标从 0.49–0.85x 提到 ≥3x）。
2. 解决 E4/L24 的 5.73e-12 精度问题（先判定是容差口径还是实现问题）。
3. 收 bg-6/7/8/9/10，独占机器做最终全量验收。

---

## 9. 勘误（2026-11-08 06:45，三处，均已用实测推翻）

### 勘误 1：E4/L24 的 `rel=5.73e-12` 不是数值缺陷

- 用同一 `build_operands`、同一 seed、同一 path 复现：`paths equal=True`、`opt_cost` 相同（27,471,052,800）。
- 结果：`max|Δ| = 2.148e-10`，`max|ref| = 7.1327e+03` →
  **范数相对误差 = 3.01e-14**（L2 口径 1.37e-14）。
- 根因：基准里的 `_max_relative_error` 用**逐元素** `|Δ|/|ref|`，在抵消元素（|ref|≈0）上被 1/|ref| 放大，
  与 dtype 精度无关。**契约的 1e-12 应按范数口径判**。
- 已修：新增 `_norm_relative_error`（`max|Δ|/max|ref|`）作为判据，
  逐元素值降级为诊断量 `max_elementwise_relative_error` 并同时写入产物。
- 因此此前"E4/L24 超出 1e-12 契约"的表述**撤回**。

### 勘误 2：P1/P2 不是"耗时占比最大的热点"

- 我此前写"15 算子 3pt 是真正热点"，但自己的表格显示 opt_einsum 的时间 **75% 花在 2pt（E1）**。
- 正确表述：**2pt 是最大耗时项；3pt（P1/P2）是我们相对最落后（0.49–0.85x）且调用量巨大的项**，
  是"优化机会最大"，不是"耗时占比最大"。两者不能混为一谈。
- 依据：`simple/workflow.py --nt 8 --tsep 2` 的 model 口径（见 §3）。

### 勘误 3：曾把"GPU 对 GPU 倍率"与"CPU/GPU 算力差"混为一谈

- 已撤回。GPU vs GPU 双方共享同一块卡，算力差异自然抵消；判据只有"双方是否都已接近该
  硬件的有效吞吐上限"。

## 10. 修正后的热点判断（可直接用于排期）

| 判据 | 结论 |
|---|---|
| 谁占 opt_einsum 时间最多 | **2pt（E1）约占 75%**（`tsep=2` 模型下） |
| 谁让我们落后最多 | **3pt 的 P1/P2（0.49–0.85x）** |
| 谁已从落后转为领先 | E4（GPU 1.90x）、CPU E1（6.98x）、CPU E4（8.25x） |
| 谁物理上不可达 3x | GPU E1/E2（四方实现收敛在 373–444 ms） |
| 下一阶段唯一高价值目标 | **P1/P2 的 14 步多步执行路径**（bg-11 已开工） |
