---
name: pyqcd-infra
description: |
  PyQCD 基础设施技能：numpy/cupy/torch 三后端切换与精度管理（set_backend/
  set_precision/get_backend 适配层）、h5py 张量 IO 与旧格式回退读取、ASCII
  关联函数读写对、V†V/VVV 预计算顶点积 reader、MPI 元任务并行（显存公式
  N·a=n·b 规划、GPU 绑定、任务后释放）。触发于："切 torch 后端"、"精度切换"、
  "h5 读写"、"读 VdV/VVV"、"MPI 并行"、"GPU 规划"、"显存不够"、"并行管线"。
metadata:
  openclaw:
    emoji: 🧱
---

# pyqcd-infra — 后端 / IO / 并行基础设施

## 目的与边界

为全部物理链与管线提供底座：计算后端切换、张量/关联函数 IO、MPI 元任务并行。
本技能不含物理逻辑；性能结论以文内实测基准为准，不凭感觉调参。

## 计算后端 — `pyqcd/tools/_backend.py` + `_torch_backend.py`

```python
from pyqcd.tools import set_backend, set_precision, get_backend
set_backend('torch')          # 别名 gpu/cuda 等价；全面替换 numpy/cupy
set_precision('complex64')    # 或 'complex128'；复数遵循全局精度
```

- numpy/cupy 输入自动转 torch；`get_backend()` 返回 numpy-like 适配层：
  einsum / roll(axis=) / transpose 任意轴 / take(axis=) / linalg / cos/sin /
  arccos/isnan/clip/maximum 标量 / argwhere / identity / append / random 均已包装。
- torch.Tensor 补丁：transpose/astype/.T/repeat(axis=)/get/二元运算。
- **实测基准**（勿凭感觉改）：torch CPU 梯度流 2.7–4.8× vs numpy（逐位一致
  max|d|~1e-15）；CPU 自动 8 线程最快（16 线程过并行反而慢 40%）；
  vertex conf6250 GPU 36s（峰值 176MB）、2pt 337s（峰值 570MB）。
- 一致性排查顺序：结果超差先核对后端与精度设置再查算法。

## 张量 IO — `pyqcd/tools/_io.py`

| 需求 | API |
|---|---|
| 管线产物读写（通用三后端） | `save_tensor_h5(arr, path, dataset)` / `load_tensor_h5(path)` |
| 旧产物兼容 | `_load_any`：优先 .h5，回退 .npy/.npz |
| L.Liu ASCII 关联函数对（.gz 自动压缩） | `write_data_ascii(data, T, L, filename)` / `read_data_ascii(filename)` |
| V†V / VVV 预计算顶点积二进制 reader | `readin_vdv_all(vdv_dir, nev, nev1, Nt, ...)` / `readin_vvv_all` / `readin_vvv`（f8 交错复数，Nev 自探测+截断 Nev1） |

## MPI 元任务并行 — `pyqcd/parallel/`

**规划公式（用户给定）**：$N\cdot a = n\cdot b$
（a=单元任务显存，b=单卡可用显存 80%，n=GPU 数 → 进程数 N；
批次 X=m/N；每卡进程 Y=N/n）。

```bash
python -m pyqcd.parallel --dry-run --confs 6250,6450   # 规划预览
mpirun -np N python -m pyqcd.parallel --confs ...      # 正式并行
```

- `plan_parallel(m, a_mem_mb, resources, n_gpu, force_y, ...)` 给出 N/X/Y；
  本机 1 卡+内存紧张时公式自动收敛 N=1。
- `run_parallel_pipeline(steps=('env','vertex','2pt',...), ...)`:元任务
  （step,conf）round-robin 调度 + GPU 绑定（rank mod n）+ 每任务后自动释放
  （empty_cache+gc）；**analysis/plots/report 仅 rank 0**（分析作图不并行）。

## 工作流程

1. 新脚本默认写 numpy 语义，由 set_backend 无缝切 torch/GPU；
   显存敏感处先 dry-run 规划再上并行。
2. 产物统一 h5；读侧永远走 _load_any 兼容链；对外交换数据用 ASCII 对。
3. 并行作业结束核对：各任务退出码 + rank0 产物完整性（模板守卫）。

## 错误处理

| 场景 | 处理 |
|---|---|
| GPU OOM | 公式降批次/进程数；或回 CPU（torch CPU 仍 2.7–4.8×） |
| 后端行为不一致 | 核对适配层覆盖面；缺失函数补包装而非绕过 |
| VVV 维度不符 | 确认 Nev 自探测与 Nev1 截断参数；f8 交错复数布局 |
| 并行残缺产物 | 用 _validate.check_files_existence 守卫补跑缺任务 |

## 与其他技能配合

- 管线级使用 → `pyqcd-pipeline`；分析/拟合 → `pyqcd-analysis`；
- 物理链计算主体 → `pyqcd-tmd-chain`；传播子生产（PyQUDA 侧）→ `pyqcd-propagator`。
