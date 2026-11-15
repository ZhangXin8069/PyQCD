# dev60 独立复测（2026-11-15，agent: configure 子代理）

只读复测，未修改任何生产代码。产物：本目录下 JSON + 日志。

## 0. 口径与一个必须记录的环境事实

| 项 | 值 |
|---|---|
| 复测对象 | `HEAD = aa048e8a`（dev60，修复提交 `82641d51`） |
| 生产尺寸 | `_bench_scene.dims("distill")` = Nev 100 / Ns 4 / Mom 1 / Lt 1；`dims("distill_curr")` = Nev 400 / Mom 9 |
| KUPL 原生库 | **由本代理用 HEAD 源码在隔离目录重建**：`git archive HEAD cpp/kupl/einsum` → `python -m pyqcd.testing._topology_kernel_gen`（md5 `16f3c3d426d996b2b9e79fbb41e46975`，2582 kernels，与 dev60 日志记录的 `16f3c3d4…` 一致）→ `make.sh`（SUCCESS，35 s） |
| 为什么要重建 | 复测开始约 30 分钟后，**并行任务把 `cpp/kupl/einsum/src/topology_kernels.inc` 扩到 8638 条并重建了仓库内的 `.so`**（17:48/17:49，`.so` 4.8 MB）。此后仓库内库的路线分类变成 **19 v3 / 0 v2 / 1 declined**（见 `log_expanded_lib_default20.log`），dev60 的 v3/v2 分野不复存在。为公平复测 dev60，全程用 `PYQCD_EINSUM_KUPL_LIB=<隔离重建的 2582 表库>` |
| CUDA 侧 | 锁定 V100（`CUDA_VISIBLE_DEVICES=GPU-be23deb4-…`）。**必须 `LD_PRELOAD` NVRTC 12.4**（`/home/zhangxin/miniforge3/lib/libnvrtc.so.12.4.127`）：本机 conda 环境的 NVRTC 12.9 对 `sm_70` 报 `invalid value for --gpu-architecture`，任何 cupy 逐元素 kernel（含 `_probe_dtype`）都会失败并 fail closed。这是环境限制，不是代码缺陷（证据：`cupy.arange(4)` 在 3 块卡上全部 `CompileException`） |

## A. `_label_sizes` 轴位绑定与 batch 能力

（`part_a_label_sizes.json`；口径：`_label_sizes` + `_vertex_axes` + `plan_steps` + `kupl._dedicated_step`/`decompose` 双口径互校）

| seq | 形态 | 档 | 顶点指标段 | 绑定 | `max count_batch` | `batch_rank` |
|---|---|---|---|---|---|---|
| 2591 | 附录 | distill | `Nbd` / `Nfnp` / `Mhjl` / `Mrt` | 大写→Mom=1，其余→Nev=100 | 1 | 1 |
| 2591 | 附录 | distill_curr | 同上 | 大写→Mom=**9**，其余→Nev=400 | **9** | 1 |
| 3437 | 附录 | distill_curr | `Nbdf` / `Nnp` / `Mhjl` | 同上 | **9** | 1 |
| 2711 | 附录 | distill_curr | `Nbdf`/`Nnph`/`Mjlr`/`Mtvx` | 同上 | **81** | 2 |
| 1 / 3 / 9 | 基础表 | distill & distill_curr | `bd`→(1,1)/(1,9)、`fnp`→(1,1,100)/(1,9,400) | 首两轴 Lt, Mom | **1** | **0（恒）** |

反证（基础表）：seq 1/3/9 在两档下**每一步** `batch_rank == 0`、`count_batch == 1`，未被修复波及。
`count_batch` 的两个口径（`_dedicated_step["batch"]` 乘积 vs `decompose` 轴数×自算乘积）逐 step 相等。

## B. topology 路线与真 v3 性能（dev60 的 2582 表库）

### B1 route 分类（`DEFAULT_SEQS` 20 条，distill，`PYQCD_EINSUM_TOPOLOGY_REAL=1`）

| dtype | v3 | v2 | declined | v3 seq |
|---|---|---|---|---|
| complex128 | 6 | 13 | 1 | 1,3,9,849,855,1599 |
| complex64 | 6 | 13 | 1 | 同上 |
| float64 | 6 | 13 | 1 | 同上 |
| float32 | 7 | 13 | 0 | 1,3,9,**129**,849,855,1599 |

declined = seq 129，`status=4`，理由 `topology declined: pack-bound on CPU (3456077824 pack bytes, 216326912 FLOPs)`（float32 因 4 B/元素只到 0.86 GiB 而放行）。

### B2 真 v3 的 topology/dedicated 比值（13 条基础表 seq × 4 dtype，热态 3 轮中位）

我的 13 seq：`1,3,9,129,849,855,1599,1623,1700,1743,1863,2000,2582`（seq≤2582 全在编译表内）。

| dtype | n | 中位 | 最优 | 最差 | 快于 dedicated |
|---|---:|---:|---:|---:|---:|
| complex128 | 12 | 2.24× | 0.21× | 4.56× | 4/12 |
| complex64 | 12 | 2.41× | 0.09× | 3.95× | 3/12 |
| float64 | 12 | 2.60× | 0.24× | 4.01× | 3/12 |
| float32 | 13 | 2.85× | 0.34× | 5.48× | 3/13 |

逐 seq（complex128）：seq9 0.212、seq3 0.513、seq1700 0.768、seq2000 0.728（v3 胜），其余 2.2–4.6×。
数值对拍：全部格子与 dedicated 输出一致（rel ≤ 1.4e-15 / 复数单精度 ≤ 6.4e-7），0 超容差。

### B3 trace 交叉验证（`PYQCD_EINSUM_TOPOLOGY_TRACE=1`，逐 seq 独立进程）

| seq | route | trace 是否打出该方程的 `topology-specialized:` |
|---|---|---|
| 1 / 9 / 855（v3） | `kupl_topology` | **打出**（`steps=5/12/9`） |
| 2583 / 2591（v2） | `kupl_topology_v2` | **不打出**（只有探针那行 `0123,4567,10,54,32,76-> steps=5`） |

另：把路线符号的绑定缓存伪造成"缺失"后，`try_topology_apply` 返回 `None` 并给出
`… qcd_kupl_topology_last_specialized missing …`，**不写任何 `kupl_topology` 标签**（fail closed 成立）。

## C. CUDA 侧（V100，NVRTC 12.4 预载）

### C1 高元数（arity>6）命中（对比 `cupy.einsum`，生产 distill 尺寸）

| seq | arity | dtype | path | rel |
|---|---:|---|---|---|
| 3 | 7 | complex128 / complex64 | `cuda_topology` | 8.25e-16 / 4.57e-07 |
| 849 | 9 | 同上 | `cuda_topology` | 6.72e-16 / 7.21e-08 |
| 9 | 13 | 同上 | `cuda_topology` | 7.79e-16 / **4.41e-07** |
| 1599 | 12 | 同上 | `cuda_topology` | 5.14e-16 / 1.89e-07 |
| 855 | 10 | 同上 | **status=4 拒绝** | — |
| 1863 | 16 | 同上 | **status=4 拒绝** | — |

855 / 1863 的拒绝与预测一致：`result.json` 的 `ratio_vs_greedy` = 60.57 / 52.89（> 8）。

### C2 收缩序守卫（`table.tsv` 的 ratio 最小/最大各 2 条）

| seq | arity | table ratio | ratio_vs_greedy | 实测 | 与预测一致 |
|---|---:|---:|---:|---|---|
| 9 | 13 | 0.528 | 1.042 | accepted | ✅ |
| 15 | 13 | 0.528 | 1.042 | accepted | ✅ |
| 630 | 14 | 38104 | 3.81e4 | **status=4 rejected** | ✅ |
| 654 | 14 | 38104 | 3.81e4 | **status=4 rejected** | ✅ |

（`>8 拒、≤8 放行` 在 6 个格子上全部命中：上面 4 格 + C1 的 855/1863。）

### C3 三路径耗时（5 轮中位，含 deviceSync；生产 distill / distill_curr）

| seq | 档 | arity | 操作数 | generic | dedicated | topology | topology path |
|---|---:|---:|---:|---:|---:|---:|---|
| 1 | distill | 6 | 0.0 MiB | 0.716 ms | 0.362 ms | 1.141 ms | `cuda_topology` |
| 849 | distill | 9 | 0.0 MiB | 1.058 ms | 0.483 ms | 1.115 ms | `cuda_topology` |
| 1599 | distill | 12 | 0.0 MiB | 1.592 ms | 0.490 ms | 1.291 ms | `cuda_topology` |
| 9 | distill | 13 | 61.0 MiB | 2.294 ms | 0.916 ms | 1.690 ms | `cuda_topology` |
| 855 | distill | 10 | 0.6 MiB | 1.938 ms | 0.545 ms | 拒绝(status 4) | — |
| 9 | distill_curr | 13 | 976.6 MiB | 5.249 ms | 5.706 ms | 6.945 ms | `cuda_topology` |

这三条路径的 rel（vs generic）：topology ≤ 4e-16（complex128）/ ≤ 7e-7（单精度）。

## D. CPU pack-bound 守卫的「逐侧记账」

D1：把 `_topology_step_records` 的逐步判定与原生 `topology-gemm-step` 诊断行逐行对账
（生产 distill 尺寸）：

| seq | python 步数 / 原生行数 | 逐行一致 | path 序列 |
|---|---|---|---|
| 3 | 6 / 6 | 6/6 | 一致 |
| 855 | 9 / 9 | 9/9 | 一致 |
| 2 | 5 / 5 | 5/5 | 一致 |

D3：逐侧记账 vs 「两侧都 pack」的旧上界（complex128, distill）：

| seq | 步数 | 全直连步 | 逐侧字节 | 两侧上界 | 差 |
|---|---:|---:|---:|---:|---:|
| 3 | 6 | 2 | 359,936 | 130,953,984 | **364×** |
| 9 | 12 | 4 | 2,910,400 | 134,130,816 | 46× |
| 2582 | 15 | 4 | 424,992 | 3,112,416 | 7.3× |
| 855 | 9 | 3 | 411,168 | 1,778,016 | 4.3× |
| 1863 | 15 | 3 | 450,624 | 2,498,016 | 5.5× |
| 2000 | 15 | 6 | 129,012,800 | 132,930,528 | 1.03× |

D2：seq 129 的守卫字节数与「按记录逐侧求和」完全相等（complex128 3,456,077,824；
c64/f64 1,728,038,912；f32 864,019,456），且 f32 低于 1 GiB 门槛因此**不**被拒。

## 结论速览（6 项修复逐条）

| # | 修复 | 复测结论 |
|---|---|---|
| 1 | `_label_sizes` 轴位绑定 | **确认**（附录大写→Mom、其余→Nev；基础表逐位不变；count_batch 9/9/81） |
| 2 | topology 路线可观测性 | **确认**（v3/v2 分流准确 + trace 交叉验证 + 缺符号 fail closed） |
| 3 | CUDA 摘要 96→128 位 | **确认**（arity 7/9/12/13 全部命中 `cuda_topology`，rel 1e-16/1e-7） |
| 4 | v3 输入侧 direct 直连 | **确认（单侧/trans 分支）**；`count_batch > 1` 分支在 dev60 表内**不可达**（基础表 batch_rank 恒 0），与文档自述一致 |
| 5 | CPU pack-bound 逐侧记账 | **确认**（逐行与原生 trace 相同；量级差 1.03×–364×） |
| 6 | CUDA 收缩序守卫（>8 拒） | **确认**（6/6 与预测量化一致；2582 条语料拒 1692 的结论与此一致） |

**不一致 / 需注意**：文档那 4 行真 v3 比值表所用的 13 条 seq 列表**未被归档**，
我用自己选定的 13 条基础表 seq 复测得到方向一致但绝对值不同的分布
（中位 2.24–2.85× vs 文档 2.71–3.03×，最优 0.09–0.34× vs 0.04–0.20×，
最差 3.95–5.48× vs 4.29–4.72×）。同 `n` 模式（12/12/12/13）与「seq=129 仅对
float32 不被拒」这一细节完全复现，说明两套 seq 选择规则相同，差异来自具体 seq
与计时噪声，不是口径错误。

**附带发现（旧 v2 数字已作废，不属于本轮 6 项修复）**：`_label_sizes` 修正后，
附录 seq 的 `distill` 形状从错绑的 `(1,1,100)` 变成物理的 `(1,100,100)`，
v2 通用执行器的耗时随之上抬 3 个数量级 —— seq=2591 complex128（操作数 160 MiB）
本次 **首调 11.9 s / 热态中位 12.17 s（3 轮）**，而
`artifacts/topo_vs_dedicated_distill_20261115132117`（13:21，修复前）记的是 **7.5 ms**；
同一格 dedicated 直接以「plan arena 19.6 GB > 8 GiB 预算」拒绝。
因此凡引用那份旧产物里的 v2 格倍数（以及混合中位 4.89–8.18×）都不再可信；
v3 格不受影响（基础表绑定逐位未变，复测的 v3 耗时全在 0.3–24 ms 量级）。

