# pyqcd-einsum · GPU · L24 生产规模性能矩阵（2026-11-08）

> 口径：单卡 RTX 4060 Laptop（8 GB）· 单 GPU 对单 GPU · complex128 · `PYQCD_EINSUM_NATIVE=cuda` ·
> oracle = 站点包 opt_einsum 3.4.0（`PYQCD_EINSUM_ORACLE=installed`，非冻结）·
> `PYQCD_BENCH_THREADS=4` · `--iters 10 --warmup 2` · 进程隔离 · `--data synthetic`（provisional）
>
> 本文件只记录实测，不含任何源码修改。所有数字均来自 `pyqcd/testing/einsum/artifacts/gpu_l24_*` 下的
> `bench.json`（可直接复核）。

---

## 0. 一句话结论（不美化）

| 结论 | 证据 |
|---|---|
| **唯一真实落后：E4，pyqcd 慢 17.8–18.9 倍（0.053–0.056x）** | 2 个构建、4 次成功运行一致：qcd 20.37–20.74 s vs opt 1.09–1.14 s |
| **E4 同时会 SIGSEGV**：默认融合路径下 `qcd_einsum` worker `returncode=-11` | 在 ≥5 个构建状态上复现，见 §8 异常 1；关闭 fused 后不再崩且回到 0.980x |
| **唯一真实领先：E5，pyqcd 快 3.27–3.43x**，且**全部收益来自 fused γ-sandwich kernel** | 7 次运行中 6 次 3.27–3.43x；关闭 fused 后 E5 = 0.956–0.958x（持平） |
| **其余 5 组（E1/E2/S1/S2/S3）全部 `too_short`（双方中位均 < 100 ms）**，干净测量下比值 0.95–1.12 ≈ 持平 | §3、§4 |
| **规划器差异 = 0**：7 条方程的 opt/qcd path **逐条完全相同**（`planner_path_match=True`，digest 相同） | §3/§4 的 digest 列；`oracle_opt_cost == qcd_opt_cost` |
| **本轮未发生 OOM**：L24 全程未降级到 L24n | 峰值显存约 3.5 GB / 8188 MiB |

---

## 1. 机器与软件

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU（compute capability 8.9，8188 MiB） |
| 驱动 / CUDA UMD | 617.42（NVIDIA-SMI 615.78.05）/ CUDA UMD 13.4 |
| CuPy CUDA runtime | cudart 12090（CUDA 12.9），driver API 13040 |
| 系统 / 内核 | Linux 6.18.54.1-microsoft-standard-WSL2（glibc 2.43） |
| CPU | 13th Gen Intel(R) Core(TM) i7-13620H，8 逻辑核（WSL 可见） |
| 线程数 | `threads=4`，numpy 导入前强制（`thread_pin_before_numpy_import=true`，OMP/OPENBLAS/MKL/NUMEXPR/VECLIB 全为 4） |
| Python / NumPy | 3.12.4 / 2.5.4 |
| oracle | opt_einsum **3.4.0**，`kind=installed`、`frozen=false`（站点包，非冻结子模块），`provisional=true` |
| pyqcd.einsum | 0.1.0（native 0.1.0），`/home/zhangxin/PyQCD/pyqcd/einsum/__init__.py` |
| native 可用性 | `{"kupl": true, "cuda": true}` |
| 迭代 / 预热 | `--iters 10 --warmup 2`，`isolation=process`（每实现独立解释器），中位数 |
| dtype / optimize | complex128 / `--optimize dp` |
| 数据 | `--data synthetic`（真实 perambulator 缺失，固定种子合成；结论标记 provisional） |
| 可比性门槛 | `min(opt中位, qcd中位) < 0.1 s` → `too_short`，不计入实战对照 |

> **测量期间源码树被其他代理持续修改**：`pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so` 在 05:48–07:45 之间
> 至少重建 9 次（见 §8 异常 3）。每张表都在**同一构建内跑完**（运行前后 md5 相同），构建 md5 已在表头标注。

---

## 2. 表 A · `--planner production`（双方各自规划，只计稳态执行）

构建：`pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so` md5 = `9d34488b03cbd360c3b4e097282352bc`（2026-11-08 06:55:22）
L24 = Nvec 50、spatial 24、n_t 2（prop 单算子 1.06 GB）

| 方程 | opt 中位 (s) | qcd 中位 (s) | 倍率（opt/qcd） | max rel err | path digest | `too_short` |
|---|---|---|---|---|---|---|
| E1 | 0.081124 | 0.081842 | **0.991x** | 1.470e-15 | `16ac286c07d6` | ✅ 是 |
| E2 | 0.080350 | 0.080895 | **0.993x** | 2.026e-15 | `16ac286c07d6` | ✅ 是 |
| E4 | 1.085611 | 20.425015 | **0.053x** | 6.066e-14 | `e77630bb320a` | 否 |
| E5 | 0.392599 | 0.163137 | **2.407x** | 0 | `eda94defbc2a` | 否 |
| S1 | 0.078961 | 0.082803 | **0.954x** | 2.819e-17 | `16ac286c07d6` | ✅ 是 |
| S2 | 0.002626 | 0.002660 | **0.988x** | 1.641e-15 | `89a98a4aedd6` | ✅ 是（2.6 ms） |
| S3 | 0.002552 | 0.004941 | **0.516x**（离群，见 §7） | 1.271e-15 | `6562510463e4` | ✅ 是（2.6 ms） |

- 全部 7 行 `planner_path_match = True`（opt plan 与 qcd plan 逐条一致）。
- E4 行数据来自 `gpu_l24_v3_production_E4_a1`：**测量本身成功**，但该次 python 进程在退出时 `SIGSEGV`（rc=139），
  产物完整写出。另两次同配置运行直接 `worker returncode=-11` 报错（§8 异常 1）。
- E5 的 `warmup=2` 不足：opt 样本单调衰减 0.6875→0.3772 s（1.82x），qcd 0.2004→0.1356 s（1.48x）；
  中位数因此偏低，重复测量给出 3.27–3.43x（§7）。

---

## 3. 表 B · `--planner shared`（双方共用 oracle 显式 path，只隔离执行层）

构建：同一 md5 `9d34488b03cbd360c3b4e097282352bc`

| 方程 | opt 中位 (s) | qcd 中位 (s) | 倍率（opt/qcd） | max rel err | path digest | `too_short` |
|---|---|---|---|---|---|---|
| E1 | 0.080753 | 0.078990 | **1.022x** | 1.470e-15 | `16ac286c07d6` | ✅ 是 |
| E2 | 0.079908 | 0.081796 | **0.977x** | 2.026e-15 | `16ac286c07d6` | ✅ 是 |
| E4 | 1.135255 | 20.366656 | **0.056x** | 6.066e-14 | `e77630bb320a` | 否 |
| E5 | 0.398580 | 0.116421 | **3.424x** | 0 | `eda94defbc2a` | 否 |
| S1 | 0.081524 | 0.082990 | **0.982x** | 2.819e-17 | `16ac286c07d6` | ✅ 是 |
| S2 | 0.002646 | 0.002794 | **0.947x** | 1.641e-15 | `89a98a4aedd6` | ✅ 是（2.6 ms） |
| S3 | 0.002642 | 0.002629 | **1.005x** | 1.271e-15 | `6562510463e4` | ✅ 是（2.6 ms） |

> shared 与 production 的 path digest 完全相同，故两表**不存在规划器层面的差异**；
> 两表之间的距离（例如 E5 2.407x vs 3.424x）全部是 run-to-run 噪声，不是 planner 效应（§6）。

---

## 4. `too_short` 标注（不可作为实战对照）

按 harness 规则 `min(双方中位) < 0.1 s`：

- **E1、E2、S1**：约 79–83 ms（双方均 < 100 ms）
- **S2、S3**：约 2.6 ms（双方均 < 100 ms）

即 **5 / 7 组落在 `<100 ms` 区间**，只有 **E4（1.09 s / 20.4 s）与 E5（0.39 s / 0.12 s）** 双方都超过 100 ms，
构成合格对照。其余组只能说明"量级相同"，不能用于宣称快慢。

---

## 5. 对照结论（逐组，不美化）

**合格对照（双方 ≥ 100 ms）**

| 方程 | production | shared | 判定 |
|---|---|---|---|
| **E4** | 0.053x（1.086 s → 20.425 s） | 0.056x（1.135 s → 20.367 s） | **我们慢 17.8–18.9 倍**。在 `9d34488b` 与 `04c3e08c` 两个构建、4 次成功运行上均为 0.053–0.056x（20.37 / 20.43 / 20.58 / 20.74 s）。这是本轮唯一的实质性性能缺陷。 |
| **E5** | 2.407x（0.393 s → 0.163 s） | 3.424x（0.399 s → 0.116 s） | **我们快 2.4–3.4 倍**；7 次运行（v3/v4/v6 + 两次全量跑）中 6 次落在 3.27–3.43x，1 次 2.41x（该次 qcd 退化到 0.163 s 且 warmup 未收敛）。结论取 **≈3.3x**，不取单次最高值。 |

**不可作对照（`too_short`）**

| 方程 | production | shared | 观测 |
|---|---|---|---|
| E1 | 0.991x | 1.022x | 重复测量 1.123x（v4）→ 1.0 ± 0.13，属噪声 |
| E2 | 0.993x | 0.977x | 全量运行中出现 0.435x（qcd 建立稳态慢，样本 0.259→0.144 单调下降）；逐方程运行 0.993x |
| S1 | 0.954x | 0.982x | 重复 1.015x；单次样本中有 2 个尖峰（0.174/0.119 s） |
| S2 | 0.988x | 0.947x | 2.6 ms 量级，±5% 抖动 |
| S3 | 0.516x | 1.005x | 重复 0.996x / 0.960x；0.516x 为该次 qcd 样本双峰（4.9/6.8 ms）所致，属离群，已重跑（§7） |

**归因（见 §6）**：E4 的落后与 E5 的领先**都来自各自的 fused kernel**，不是通用执行层。
对 E4/E5 做 `PYQCD_EINSUM_DISABLE_FUSED=1` 开关实验后二者分别回到 **0.980x / 0.956–0.958x**；
其余 5 条方程本来就不走 fused kernel（专用命中的 `kind` 是计划缓存，无专用 kernel），
其 0.95–1.12x 说明通用执行层与 opt_einsum 同级。

---

## 6. 归因实验（**非默认口径**，仅用于定位）

环境变量 `PYQCD_EINSUM_DISABLE_FUSED=1`（关闭 E4/E5 的专用融合 kernel，rest 不变）：

| 方程 | 默认（fused on） | `PYQCD_EINSUM_DISABLE_FUSED=1` | 构建 md5 |
|---|---|---|---|
| E4 | 0.053x，且会 SIGSEGV | **0.980x**（1.2364 s → 1.2622 s），不再崩溃 | `04c3e08c…` |
| E5 | 3.27–3.43x | **0.958x**（0.4426 s → 0.4618 s）；重复 **0.956x**（0.3875 s → 0.4051 s） | `04c3e08c…` / `42510634…` |

结论：
1. **E4 的 19 倍落后 100% 来自 `cuda_e4_fused` 融合 kernel**（通用多步路径 0.98x，正常）。
   该 kernel 同时是 SIGSEGV 的来源。
2. **E5 的 3.3 倍领先 100% 来自 sandwich 融合 kernel**（通用路径 0.96x）。
   因此"我们更快"仅限于 E5 这一条专用路径，不能外推到其他方程。
3. 由于 E4/E5 是专用路径命中（`_specialize.REGISTRY`），两表的倍率与 planner 无关（path 相同），
   与执行层具体 kernel 强相关。

---

## 7. 重复测量与离群记录（按"重跑一次并记录两次结果"执行）

| 组 | 构建 md5 | 第 1 次 | 第 2 次（及后续） | 处理 |
|---|---|---|---|---|
| E5 / production | `9d34488b` → `42510634` | 2.407x（0.3926→0.1631） | 3.408x（0.3904→0.1146）、3.271x（0.3975→0.1215） | 全部保留；结论用 6/7 次一致值 ≈3.3x，并标注 2.407x 那次的 warmup 未收敛 |
| E5 / shared | `9d34488b` → `04c3e08c` | 3.424x（0.3986→0.1164） | 3.302x（0.3890→0.1178） | 一致 |
| S3 / production | `9d34488b` → `42510634` | 0.516x（qcd 样本双峰 4.9/6.8 ms） | 0.996x、0.960x | 判定 0.516x 为离群，取 ≈0.96–1.00x |
| S1 / production | `9d34488b` | 0.954x（qcd 含 0.174/0.119 s 尖峰） | 1.015x | ≈1.0x |
| E1 / production | `9d34488b` → `04c3e08c` | 0.991x（opt 0.0811） | 1.123x（opt 0.0886） | 1.0 ± 0.13，`too_short` |
| E4 / production | `9d34488b`、`04c3e08c` | 0.053x（20.425 s） | 0.053x（20.740 s） | 一致，稳定复现 |
| E2 / production（全量跑） | `04c3e08c` | 0.435x（qcd 0.1897，样本 0.259→0.144） | 0.993x（逐方程，`9d34488b`） | 0.435x 为 warmup/构建差异，取 0.99x |

**离群判据**：同配置相邻版本中位数相差 > 3x，或运行内 `max/min > 1.8` 且样本单调衰减 → 重跑。
E5（第 1 次）与 S3（第 1 次）命中该判据，均已重跑并如上表记录。

---

## 8. 异常清单（全部如实记录）

1. **E4 默认融合路径 SIGSEGV（最严重）**
   - 现象 A：`qcd_einsum` worker `returncode=-11`，行状态 `status=error`，`note="RuntimeError: qcd_einsum worker 失败（returncode=-11）: 无错误信息"`。
     复现于：`gpu_l24_20261108_production`（06:16）、单方程探针（06:22，scratchpad `e4_only`）、
     `gpu_l24_20261108_production_attempt2`（06:31）、`gpu_l24_v4_production_E4_a1/a2`（07:20、07:22）、
     `gpu_l24_full_shared_a1`（07:41）—— 覆盖 ≥5 个不同构建状态。
   - 现象 B：测量成功但进程退出时崩溃（rc=139），产物完整：`gpu_l24_v3_production_E4_a1`、`gpu_l24_full_production_a1`、
     `gpu_l24_v4_production_E4_a3`。
   - **间歇性**：同一个构建 `04c3e08c` 上，07:20/07:22 两次是 `-11` 报错，07:31 却成功测到 20.58 s。
   - faulthandler 定位（`--isolation abba` + `CUDA_LAUNCH_BLOCKING=1`）：
     ```
     [pyqcd.einsum] 专用路径: gauge-propagator(E4) [fused] … CUDA E4 融合 kernel
     Fatal Python error: Segmentation fault
       File "pyqcd/einsum/cuda/_program.py", line 117 in _call
       File "pyqcd/einsum/cuda/_program.py", line 59 in __call__
       File "pyqcd/einsum/_api.py", line 641 in contract
     ```
     即在 `applyCompiledQcd` 入口内崩溃；**不是 OOM**（未抛 cupy 内存异常，且显存充足：free 7.1 GB / total 8.2 GB）。
   - `PYQCD_EINSUM_DISABLE_FUSED=1` 下不崩（0.980x），说明崩溃点在 E4 fused kernel。

2. **E4 性能落后 17.8–18.9 倍**（0.053–0.056x）：qcd 20.37–20.74 s/iter，opt 1.09–1.14 s/iter，在 `9d34488b`、`04c3e08c`
   两个构建上稳定复现。等同 1.3 GFLOP/s 量级的有效吞吐（opt_cost 2.747e10），属 kernel 级问题，不是噪声。

3. **构建漂移（外部干扰，非本次测量可控）**：测量窗口内 `libpyqcdeinsum_cuda.so` 被其他代理反复重建：

   | 时间 | md5 前 8 位 | 备注 |
   |---|---|---|
   | 05:48:41 | `a6025928` | 起始状态 |
   | 06:16 前 | `e4b4005c` | 全量跑 pre（跑中变为 `88b74d3`，该次作废） |
   | 06:19:49 | `88b74d3c` | |
   | 06:26:52 | `b388cbe3` | |
   | 06:33 | `95aee9be` | |
   | 06:42:56 | `f794ad0d` | 体积 8.26 MB（其余约 1.3–1.9 MB） |
   | **06:55:22** | **`9d34488b`** | **表 A / 表 B 使用的构建** |
   | **07:23:13** | **`04c3e08c`** | 全量跑 + fused-off 归因实验 |
   | **07:41:22** | **`42510634`** | 最终重复测量 |

   影响：**同一方程在不同构建上可差 2.5 倍**。例如 S3 的 qcd 中位
   `9d34488b`=2.63 ms（1.005x）、`04c3e08c`=6.58/7.41 ms（0.384x/0.347x）、`42510634`=2.72 ms（0.960x）。
   因此本报告的所有对比都限定在**同一构建内**（运行前后 md5 相同才接受），绝不跨构建比较。

4. **并发负载**：测量期间机器上有其他代理在编译（`nvcc`/`cicc`/`cc1plus`，同时 0–6 个）并运行**同一个**
   `bench_native` L24 GPU 基准（例如 06:45–06:53、07:10 附近的 `bench_native --profiles L24`）。
   应对：每行测量前等待 GPU 计算进程清零（最多等 300 s），并采样 `nvidia-smi --query-compute-apps` 留证
   （`$COMMANDCODE_SCRATCHPAD/logs/*_interf.log`）。仍无法排除的残余干扰体现在样本离散度上
   （多数行 `max/min` = 1.04–1.25；E5 的 opt 侧 1.82–1.84 来自 warmup 未收敛）。
   未遇到/未执行 MPI、多进程吞吐、多卡对比。

5. **`warmup=2` 不足（系统性）**：E5 的 opt 样本 `0.6875, 0.5839, 0.5210, 0.4548, 0.4061, 0.3778, …` 单调衰减 1.82x；
   E2 在全量跑中 qcd `0.2592 … 0.1436`（1.81x）。中位数受前几个瞬态样本影响，故本报告对 E5/E2 给出了重复测量区间。
   这属于口径限制，未修改 harness。

6. **显存**：L24 全程未 OOM，**没有降级到 L24n**。观测峰值约 3.5 GB / 8188 MiB（E4/E5 段），
   与 PEOF 说明的"prop 单算子 1.06 GB、峰值约 3.2 GB"一致。

7. **进程退出码噪声**：多次运行 `rc=2`（harness 的 NON_ACCEPTANCE 返回码，正常）与 `rc=139`
   （解释器退出阶段 SIGSEGV）；后者已分别归入异常 1。

8. **Python 层也被并发修改**（不仅是 `.so`）：
   - `pyqcd/einsum/_api.py` mtime `07:32:59`：新增 `_fused_disabled()` 并加入 `_native_plan_key` / `_try_native_contract` /
     `_contract_signature` 的缓存 key（只影响缓存区分，不改变数值路径）。
   - `pyqcd/einsum/_specialize.py` mtime `07:44:57`：把 E1/E2/S1/S2/S3 的 `kind` 由 `fused-plan` 改名为 `plan-cache`
     并注明"仅复用计划缓存，无专用 kernel"（语义/标签，无行为变化；E4/E5 的 `kind: fused` 未变）。
   - 对照时间线：表 A / 表 B（06:54–07:02）在两处改动**之前**；全量跑（07:31:54）跨过 `_api.py` 改动；
     最终重复测量（v6，07:46+）在两处改动**之后**。已核对 diff：上述改动对本次测量语义无影响，
     但"同一份代码"这一点在窗口内不成立，故单列记录。

---

## 9. 复现命令与产物目录

### 9.1 规定的原始命令（全量 7 条）

> 本报告用它跑了 4 次：2 次（06:16、06:31）因测量中途 `.so` 被重建而只作异常证据；
> 2 次（07:31 production、07:41 shared）作交叉验证，结果与表 A/B 一致（见 §7）。

```bash
cd /home/zhangxin/PyQCD
PYQCD_EINSUM_ORACLE=installed PYQCD_EINSUM_NATIVE=cuda PYQCD_BENCH_THREADS=4 \
  python -m pyqcd.testing.einsum.bench_native \
  --data synthetic --profiles L24 --backends gpu \
  --equations E1,E2,E4,E5,S1,S2,S3 \
  --iters 10 --warmup 2 --planner production
```

把 `--planner production` 换成 `--planner shared` 即表 B。

### 9.2 本报告两张主表实际使用的逐方程命令

> 由于构建每 5–10 分钟被外部重建一次，全量跑（约 5–8 分钟）经常跨越两次构建而被污染。
> 因此表 A / 表 B 采用**同一命令逐方程执行**（每次约 40 s–4.5 min），
> 并用「运行前后 `.so` md5 相同」作为接受条件；测量语义（同一 worker、iters=10、warmup=2、process 隔离）完全不变。

```bash
cd /home/zhangxin/PyQCD
for EQ in E1 E2 E4 E5 S1 S2 S3; do
  for PL in production shared; do
    PYQCD_EINSUM_ORACLE=installed PYQCD_EINSUM_NATIVE=cuda PYQCD_BENCH_THREADS=4 \
      python -m pyqcd.testing.einsum.bench_native \
      --data synthetic --profiles L24 --backends gpu --equations "$EQ" \
      --iters 10 --warmup 2 --planner "$PL" \
      --outdir "pyqcd/testing/einsum/artifacts/gpu_l24_v3_${PL}_${EQ}_a1"
  done
done
```

归因实验（§6，非默认口径）：

```bash
cd /home/zhangxin/PyQCD
for EQ in E4 E5; do
  PYQCD_EINSUM_ORACLE=installed PYQCD_EINSUM_NATIVE=cuda PYQCD_BENCH_THREADS=4 \
    PYQCD_EINSUM_DISABLE_FUSED=1 \
    python -m pyqcd.testing.einsum.bench_native \
    --data synthetic --profiles L24 --backends gpu --equations "$EQ" \
    --iters 10 --warmup 2 --planner production \
    --outdir "pyqcd/testing/einsum/artifacts/gpu_l24_v5nofused_production_${EQ}_a1"
done
```

### 9.3 产物目录（均在 `/home/zhangxin/PyQCD/pyqcd/testing/einsum/artifacts/`）

| 用途 | 目录 |
|---|---|
| 表 A（production，`9d34488b`） | `gpu_l24_v3_production_{E1,E2,E4,E5,S1,S2,S3}_a1` |
| 表 B（shared，`9d34488b`） | `gpu_l24_v3_shared_{E1,E2,E4,E5,S1,S2,S3}_a1` |
| 规定命令·全量 production（`04c3e08c`） | `gpu_l24_full_production_a1` |
| 规定命令·全量 shared（`04c3e08c`→`42510634`，跨构建，仅参考） | `gpu_l24_full_shared_a1` |
| 重复测量 | `gpu_l24_v4_production_E1_a1`、`gpu_l24_v4_production_S1_a1`、`gpu_l24_v4_production_S3_a1`、`gpu_l24_v4_production_E5_a1`、`gpu_l24_v4_shared_E5_a1`、`gpu_l24_v4_production_E4_a1`（崩溃）、`…_E4_a2`（崩溃）、`…_E4_a3`（成功）、`gpu_l24_v6_production_E5_a1`、`gpu_l24_v6_production_S3_a1` |
| 归因实验（fused off） | `gpu_l24_v5nofused_production_E4_a1`、`gpu_l24_v5nofused_production_E5_a1` |
| 早期被构建漂移污染、仅作异常证据 | `gpu_l24_20261108_production`、`gpu_l24_20261108_production_attempt2` |
| E4 崩溃的 in-process 复现（faulthandler 栈） | 会话 scratchpad `$COMMANDCODE_SCRATCHPAD/e4_abba`（非项目产物，含 bench.json + 崩溃栈） |

每个目录含 `bench.json` / `bench.md` / `env.json`；`env.json` 记录 `threads=4`、`dtype=complex128`、
`iterations=10`、`warmup=2`、`isolation=process`、`planner`、`devices.gpu.name`、`worker_pids`。
接受条件已逐项复核：24 个 `v3/v4/v5nofused` 产物目录 **全部** 满足
`threads==4` ∧ `dtype==complex128` ∧ `devices.gpu.name=RTX 4060 Laptop GPU` ∧
`worker_pid.opt_einsum != worker_pid.qcd_einsum`（均为 int）∧ `planner` 与目录一致。
表 A / 表 B 的 14 个数值已用脚本与 `bench.json` 逐格比对（opt/qcd/倍率/max rel err/digest/too_short **完全一致**）。

---

## 10. 局限与未做项

- **数据为 synthetic（provisional）**：真实 perambulator 缺失，形状与真实蒸馏一致，正确性以 opt_einsum 逐元素对照
  （max rel err ≤ 6.1e-14，容差 1e-12，全部通过）。性能绝对值不依赖数据内容，但仍是 provisional。
- **oracle 非冻结**：站点包 opt_einsum 3.4.0（`frozen=false`），非子模块冻结副本。
- 仅 10 次迭代、2 次预热；E5/E2 的 warmup 瞬态使中位数偏低，已在 §7 用重复测量补偿，但未改 harness 口径。
- 未测 T1/T2、未测 CPU、未测 P1/P2（不在本次任务范围）。
- 未做多卡 / MPI / 多进程吞吐（明确禁止）。
- 测量时段源码树处于活跃开发中（其他代理正在改 E4 fused kernel），**本报告结论只对应 §8 异常 3 表中标注的构建**。
