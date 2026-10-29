# TMD real-data GPU production runbook

## 适用范围

在真实组态上批量生产 source-resolved C2、1D/2D full-channel OPE，或需要判断 CUDA
设备映射、GPU 显存、checkpoint 与断点恢复时读取本文件。它记录 2026-10-29 的实测经验，
不替代 `geometry.md`、`renormalization.md` 与 `validation.md` 的物理门。

## 设备身份先探测

本机的 `nvidia-smi` 物理索引与 CUDA runtime 可见索引不一致：

| `nvidia-smi` 物理索引 | 设备 | capability | PyTorch 可见映射 |
|---:|---|---:|---|
| 0 | P100 16 GiB | 6.0 | `CUDA_VISIBLE_DEVICES=1/2` 可暴露 |
| 1 | P100 16 GiB | 6.0 | `CUDA_VISIBLE_DEVICES=1/2` 可暴露 |
| 2 | V100 32 GiB | 7.0 | `CUDA_VISIBLE_DEVICES=0` 暴露 |

不要按 `nvidia-smi` 索引直接填 `CUDA_VISIBLE_DEVICES`。运行前执行：

```bash
for cuda_id in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$cuda_id" python3 - <<'PY'
import torch
print([(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
       for i in range(torch.cuda.device_count())])
PY
done
```

实测教训（2026-10-29）：按 `nvidia-smi` 索引写 `CUDA_VISIBLE_DEVICES=2` 期望选中 V100，
实际拿到仍是一块 P100，启动 kernel 立即报 `no kernel image is available`。
CUDA 默认按「先快后慢」枚举设备，本机 V100 是 runtime 的 device 0。任何 GPU 作业都
先把「runtime 索引 → 设备名」这一行打进日志，再开始计算；日志里没有设备名就无法
区分「跑在 P100 上失败」和「跑在 V100 上出错」。

CuPy 14.0.1 仍带 sm_60 内核，可在 P100 上执行 kernel（实测 `cp.sum` 正常），
但 pyqcd 的 OPE 管线是否全程数值等价必须先用小网格 numpy vs cupy 逐元素校验，
不能只凭 kernel 能启动就用于生产缓存。

### P100 + CuPy 的实测结论（2026-10-29）

验证方法不要用「单线程 numpy 重算同一网格」：24³×72 的 Wilson flow 在单线程
numpy 上要几十分钟，而且参考值本身没有独立来源。**用磁盘上已有的 torch 生产
缓存作参考**：让 CuPy 在 P100 上重算 `(z=0..12, b=0..2, tau=1, L=12)`（独立
contract 键，不覆盖生产），再与生产缓存的对应切片逐元素比较。

| 量 | 结果 |
|---|---|
| 有限值模式 | 完全一致（134784 格点） |
| 中位相对差 | `7.8e-7`（complex64 舍入级别） |
| 最大绝对差 | `1.2e-3`，出现在 `|参考值|≈0` 的格点（相对差无意义） |
| Wilson flow 耗时 | 95 s（P100/CuPy，tau=1；V100/torch 独占约 40 s） |
| OPE 耗时 | 124 s（b=3），按 b 线性外推 b=7 约 290 s |
| 单任务总耗时比 | P100 ≈ **0.42 ×** V100 |

结论：CuPy/P100 可以作为独立设备参与生产，但会**破坏与 torch 产品的逐位
可比性**（差 1e-6 相对量级）。若某条回归判据要求逐位一致，就把那部分任务
（例如参考流时 tau=3）固定在同一后端上算，其余任务再分给 P100；报告里必须
按后端分别写明「逐位」还是「float32 等价」。把这一点写进产物 metadata，
不要只写在日志里。

### 混合后端生产：分级判据与任务分配

批量生产里「哪个文件是哪个后端算的」必须可追溯，否则回归判据无法自洽。

- worker 每次完成一个 `(tau, conf)` 就把 `backend` 与耗时写进自己的完成 JSON；
  汇总脚本从这些 JSON 重建 `{(tau, conf): backend}` 的 `backend_map`。
  **注意：完成 JSON 只在 worker 整个队列跑完后才落盘**，生产中途去读会得到空表，
  让所有单元格落到 `backend=unknown` 并被降级判据。中途要用「日志首行的
  `tag device=... backend=torch`」＋「`workers/jobs_<tag>.txt` 的任务清单」
  重建归属，并与最终 JSON **合并**（JSON 优先）而不是互相替代——两者覆盖面不同，
  本轮还额外有一个不进任何队列的基准任务（`v100bench`，tau=2/conf=6250），
  只能从它自己的日志补。
- 回归条目按该任务的后端集合分级：全部 `torch` 且 `max|Δ| = 0` → `criteria="bit_exact"`；
  全部 `torch` 但差 `1e-14` 级（归约顺序）→ `same_backend_1e-12`（`tolerance=1e-12`）；
  含 `cupy` → `criteria="float32_equivalent"`、`tolerance=1e-5`。把
  `criteria`/`tolerance`/`exactly_equal`/`backends` 四个字段一起写进产物。
  只写 `passes: true` 会让后来的人以为做过逐位比较；把 `2e-14` 标成 `bit_exact`
  则是另一种失真——标签必须与实测值一致。
- **参考流时留在最快的设备上。** 队列按 `tau` 降序排（tau=3 先跑），全部交给
  V100/torch，保证与已发布产品的逐位回归不被后端差异污染。
- 剩余任务按**实测速度**静态分配（本轮权重 `v100:1.0,p100a:0.42,p100b:0.42`），
  不按卡数均分。分配前先解析已完成日志（`tau=.. conf=.. done in ..s`），
  避免重复分配；重复分配虽然会被缓存挡住，但两个进程同时写同一缓存有风险。
- 每个 `(tau,conf)` 完成后立刻做一次旧切片对照（见 `validation.md`
  的「需要重算的批次轴扩展」），不要等全部跑完才发现某个后端系统性偏移。
- **复现性抽查要用空闲卡做「真重算」。** 直接 `--jobs <tau>:<conf>` 重跑不会
  触发计算：cache 命中会立刻返回，看起来「通过」其实什么都没算。正确做法是把
  该缓存挪到不匹配 glob 的临时路径（保留备份），再用另一张空闲卡重算，逐位比较
  后保留一份、删掉备份。同后端同设备应当逐位相同；跨设备（P100a vs P100b）
  而结果逐位相同，则说明 CuPy 路径在该驱动下是确定性的。
  状态为运行中的生产缓存不要这么做——先确认该 `(tau,conf)` 不被任何下游读取。

当前 PyTorch `2.10.0+cu128` 的编译架构为 sm_70 及以上；sm_60 P100 会在 kernel 启动时
报 `no kernel image is available`。本机生产使用：

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -B -m pyqcd.testing.tmd.run_real_polarized_chain \
  --run-dir <run_dir> --backend torch --device 0 ...
```

设备映射是运行环境事实，不是缓存身份。每次换机器、驱动、容器或 PyTorch 版本后重新探测。

## 2026-10-29 实测成本

系综为 `beta6.20_mu-0.2770_ms-0.2400_L24x72`，9 个组态，单进程 V100，`complex64`。

| 阶段 | 几何/参数 | 单组态耗时 | 输出 | 观测显存 |
|---|---|---:|---|---:|
| v3 双极化 C2 | `P000/P200/P400`，`nopol/pol35` | 279–368 s | 6 × `(72,72) complex128` | 约 2.7 GiB |
| 1D full-channel OPE | `z=0..12`，`b=0..4`，`tau=3`，`L=12` | 208–211 s | `(2,4,6,13,5,72) complex64` | 约 5.0 GiB |
| 2D staircase OPE | `z=0..12`，`bx=0..4`，`by=0..1`，order `(0,1)` | 4150 为 297.6 s | `(2,4,6,13,5,2,72) complex64` | 约 5.0 GiB |

C2 的 279–368 s 包含 3 动量和两极化的一次共享 contraction；总 9 组态时间约
`2645 s`。1D/2D OPE 均包含约 119 s 的 `tau=3` Wilson flow；当前
`compute_tmd_ope_time*` 使用 `save_gauge=False`，所以 1D 与 2D 批次会各自重新流化。
预估长批次时要把这部分重复成本计入。

## 批次与断点

- 一个 GPU 仅运行一个 worker；CPU 线程固定 4，避免与 18 核主机上的下载和其他任务争抢。
  单卡多进程实测（2026-10-29，V100 32 GiB，full-channel OPE `b=0..6`）：
  3 个进程各占约 4.9 GiB、GPU util 100\%，但单任务从 207~s 退化到约
  670~s（流 240~s + OPE 430~s），三进程合计吞吐约为单进程的 0.7--1.0 倍。
  OPE 是 launch/延迟受限而不是带宽受限，**同卡堆进程不划算**；
  要提速只能增加设备（本机另有两块 P100）。多卡时按设备实测速度静态分配任务，
  不要按卡数均分。
- 组态顺序写入 checkpoint；每个组态完成后才标记 `complete`，失败组态保留错误并停止队列。
- checkpoint 只在组态边界刷新。任务进行中若显示旧状态，先看同一进程的实时输出；最终以
  canonical HDF5、最终 checkpoint 和 audit 三方复核。
- C2 与 OPE 使用独立 checkpoint；已完成阶段可以不重算而继续下一阶段。
- 失败尝试不发布最终 HDF5；不要手工复制“看似完成”的文件。
- **后台分析作业显式给 `export PYTHONPATH=/root/PyQCD`，并统一 `python3 -u`。**
  用绝对路径调脚本不会让 `import pyqcd` 成功：本轮会话 cwd 在
  `products/pipeline/data`，三条 `plateau_combination_jk.py` /
  `b00_tsep_plateau.py` 作业一启动就 `ModuleNotFoundError: No module named 'pyqcd'`，
  主日志只剩一条 traceback，极易被误读成脚本坏了。`-u` 是为了重定向到文件时
  stdout 不被块缓冲（否则几分钟内日志 0 字节，看着像进程没起来）。

## 审计与状态

真实有限 b 输入在 ratio 前执行：

```bash
python3 -B -m pyqcd.testing.tmd.audit_finite_b_inputs \
  --run-dir <run_dir> --conf-ids <ids> --dimension 1d --out <audit.json>

python3 -B -m pyqcd.testing.tmd.audit_finite_b_inputs \
  --run-dir <run_dir> --conf-ids <ids> --dimension 2d \
  --bx-list 0,1,2,3,4 --by-list 0,1 --transverse-order 0,1 \
  --out <audit.json>
```

审计通过只表示 C2 与 full-channel OPE 的结构/极化/几何覆盖完整。它不验证 `t_sep`
平台、协方差正秩、soft/rapidity、CS、matching 或物理接受。报告必须继续写
`diagnostic_only`，直到 validation reference 的全部真实数据门通过。

## 失败模式

| 现象 | 判断与动作 |
|---|---|
| `no kernel image is available` | 先探测 capability；换到 sm_70+ 设备，不重启同一 P100 |
| `nvidia-smi` 查询 compute apps 为 `[N/A]` | 容器可能不暴露进程归属；结合进程、显存总量和耗时判断 |
| OPE 重复 Wilson flow | 检查 `save_gauge`；当前生产入口未持久化 gauge，需将流化成本计入每个 batch |
| 任务中 checkpoint 看似落后 | 只在组态完成后写入；以主进程实时日志和最终文件为准 |
| HDF5 已存在但 audit 失败 | 按 contract、shape、dtype、finite、极化/几何字段判定 miss，不能用文件名代替身份 |
| 结构 audit ready | 只能进入 ratio/统计门，不能据此发布 TMD-PDF |
