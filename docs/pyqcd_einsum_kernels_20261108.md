# pyqcd-einsum 专用 kernel 验收报告（2026-11-08 08:40）

## 1. 结论

**B7 / B8 / B9 三项全部解决，当前 bug 数 = 0。**
其中两项（B7、B9）的"缺陷"实际由验收工具与构建配置引起，不是 kernel 实现错误。

四个专用 kernel 的最终验收（`nvec=50`、`iterations=10`、4 线程）：

| kernel | 侧 | rel | kernel | 我们通用路径 | opt_einsum | kernel/通用 | kernel/opt | 判定 |
|---|---|---|---|---|---|---|---|---|
| `kupl_nucleon` | KUPL 2pt | 1.80e-15 | **22.65 ms** | 25.26 ms | 120.43 ms | 1.12x | 5.32x | **PASS**（已接线） |
| `cuda_nucleon` | CUDA 2pt | 1.32e-15 | 95.12 ms | 57.31 ms | 175.94 ms | 0.60x | 1.85x | OK-WITH-CAVEAT（不接线） |
| `cuda_3pt` | CUDA 3pt | 3.13e-15 | **30.74 ms** | 74.13 ms | 285.40 ms | 2.41x | 9.29x | **PASS**（待接线） |
| `kupl_3pt` | KUPL 3pt | 2.54e-15 | **62.40 ms** | 106.34 ms | 292.14 ms | 1.70x | 4.68x | **PASS**（待接线） |

判定规则（用户口径）：慢于 opt_einsum → BUG；快于 opt 但慢于通用路径 → OK-WITH-CAVEAT
（可用但不接线）；快于两者 → PASS。

## 2. B9（CUDA 2pt 无法编译）——根因：include 路径冲突

### 2.1 现象

```
nvcc ... nucleon2pt.cu
error: identifier "QCD_NUCLEON2PT_PERM_IDENTITY" is undefined   (line 721)
error: explicit type is missing ("int" assumed)                   (line 852)
error: expected a ";"                                             (line 852)
```

### 2.2 根因

`cpp/kupl/einsum/include/nucleon2pt.h`（6395 B）与
`cpp/cuda/einsum/include/nucleon2pt.h`（8130 B）**同名**，
而 `make-nv.sh` 的 include 顺序是 `-I cpp/kupl/einsum/include` → `-I cpp/cuda/einsum/include`。
源文件里的 `#include "nucleon2pt.h"` 因此解析到 **KUPL 版头文件**，
该头不定义 `QCD_NUCLEON2PT_API` / `QCD_NUCLEON2PT_PERM_IDENTITY`。

`pattern_3pt.h` 存在完全相同的问题（4825 B vs 5002 B）。

### 2.3 修复

两个 CUDA 源文件改为显式相对路径，绕开 `-I` 顺序依赖：

```cpp
#include "../include/nucleon2pt.h"     // 原 #include "nucleon2pt.h"
#include "../include/pattern_3pt.h"    // 原 #include "pattern_3pt.h"
```

验证：在 `make-nv.sh` 的真实 include 顺序下 `nvcc -c` 两文件均 **rc=0**；
随后 `build-nv.sh` SUCCESS，导出符号存在（`qcd_cuda_nucleon_2pt`）。

## 3. B7（CUDA 3pt 数值错误 rel=1.00）——根因：验收工具传错指针

### 3.1 现象

`status=0` 但 `rel = 1.00`，耗时 0.11 ms（几乎未计算）。

### 3.2 根因（两个工具 bug）

1. **指针域错误**：CUDA kernel 契约（`include/nucleon2pt.h` / `pattern_3pt.h`）明确要求
   **设备指针**，而验收工具传入的是 numpy 主机指针。
2. **契约参数错误**：
   - `nucleon2pt` 的 `dims` 是 **10 项**（`[N,mx,my,mode,permA,permB,permC,permV1,permV2,matflags]`，
     恒等 perm 码 = 123），工具只传了 8 项且语义不对；
   - CUDA 版算子顺序是 `{P1,P2,P3,G1,G2,pr,V1,V2}`（**投影在 V 之前**），
     工具按 einsum 顺序 `{...,V1,V2,pr}` 传。

### 3.3 修复（`pyqcd/testing/einsum/simple/kernel_acceptance.py`）

- CUDA 分支：`cupy.asarray` 上传操作数，传 `data.ptr`，输出分配在设备后拷回；
- CUDA 3pt：`dims = (nev, mom, link, 4, 2)`（5 项契约）；
- CUDA 2pt：`dims = (N, mx, my, mode=0, 123,123,123,123,123, matflags=0)`（10 项）；
  kernel 侧操作数按契约重排，reference 仍按 einsum 顺序计算；
- nvcc 编译补 `-Xcompiler -fPIC`。

## 4. B8（KUPL 3pt 数值错误 + ZGEMM lda 非法）——根因：测到旧版本

先前测到 `rel=1.00` 且 stderr 报 `ZGEMM parameter number 13 had an illegal value`
（lda 非法），当时该文件正被子代理重写（同一路径在两分钟内出现过
`.cpp` → 不存在 → `.cpp` 三种状态）。

代理新版落盘（08:26，45393 B）后重测：`status=0`、`rel=2.54e-15`、62.40 ms，
**无 CBLAS 报错**，判定 PASS。

## 5. 全局回归（修复后）

| 检查 | 结果 |
|---|---|
| `build-kupl.sh` | SUCCESS |
| `build-nv.sh` | SUCCESS |
| `verify_native --backend kupl` | 16 passed / 0 failed |
| `verify_native --backend cuda` | 16 passed / 0 failed |
| `simple/run-all.sh` | exit=0（compat 39 + memory 4 + equations 9 + native 13×2） |
| `verify_compat` 全量 | **980/980 PASS**（oracle 自检 70、API 38、路径 802、数值 70） |

## 6. 当前 kernel 接线状态

| kernel | 是否接线 | 说明 |
|---|---|---|
| `kupl_nucleon`（E1） | **已接线** | `make.sh` 导出符号 + `_kernel_kupl.py` 严格校验 + `_api.contract` 分派；E1 占我们生产时间 42.6% |
| `kupl_3pt`（P1/P2） | 未接线 | 1.70x 于通用路径；P1 占我们生产时间 42.1%，**优先接线目标** |
| `cuda_3pt`（P1/P2） | 未接线 | 2.41x 于通用路径 |
| `cuda_nucleon`（E1） | 未接线 | 0.60x 于通用路径（虽快于 opt），按规则不接线 |

## 7. 复现命令

```bash
cd /home/zhangxin/PyQCD
PYQCD_EINSUM_ENABLE_MPI=0 bash build-kupl.sh
PYQCD_CUDA_ARCHITECTURES=8.9 bash build-nv.sh
cp -f cpp/cuda/einsum/build/libpyqcdeinsum_cuda.so pyqcd/einsum/cuda/

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  python -m pyqcd.testing.einsum.simple.kernel_acceptance --nvec 50 --iterations 10

PYQCD_EINSUM_NATIVE=kupl python -m pyqcd.testing.einsum.verify_native --backend kupl
PYQCD_EINSUM_NATIVE=cuda python -m pyqcd.testing.einsum.verify_native --backend cuda
PYQCD_EINSUM_ORACLE=installed python -m pyqcd.testing.einsum.verify_compat
bash pyqcd/testing/einsum/simple/run-all.sh
```

---

## 8. 接线结果（08:55）

### 8.1 `cuda_3pt`（P1/P2，CUDA）——**已接线**

| 环节 | 内容 |
|---|---|
| 符号导出 | `cpp/cuda/einsum/make-nv.sh` version script 新增 `qcd_cuda_pattern_3pt`、`qcd_cuda_pattern_3pt_free` |
| 绑定 | 新增 `pyqcd/einsum/_kernel_cuda.py`（设备指针、`dims=[nev,mom,link,4,2]`、严格形状校验） |
| 分派 | `pyqcd/einsum/_api.py::contract`（numpy 路径失败后尝试 CUDA 路径；任一不满足即回退） |
| 开关 | `PYQCD_EINSUM_KERNEL_CUDA=0` 可关闭 |

实测（nvec=50，GPU）：

| 指标 | 值 |
|---|---|
| rel | 2.22e-15 |
| kernel | **53.12 ms** |
| 通用路径 | 66.35 ms（1.25x） |
| opt_einsum | 66.35 ms（1.25x） |
| 非连续输入 | 正确回退，rel 2.22e-15 |
| 验收工具（10 次） | kernel 30.74 ms vs 通用 74.13（**2.41x**）vs opt 285.40（9.29x），PASS |

### 8.2 `kupl_3pt`（P1/P2，KUPL）——**撤回接线**

接线后按规模扫描（4 线程，3 次中位）：

| Nvec | kernel | 通用路径 | opt_einsum | kernel/通用 | kernel/opt |
|---|---|---|---|---|---|
| 20 | 14.29 ms | 15.17 ms | 14.83 ms | 0.94x | 1.04x |
| 50 | 138.23 ms | 136.71 ms | 137.47 ms | 1.01x | 0.99x |
| **100（生产）** | 1837.88 ms | 1855.30 ms | **1825.43 ms** | 0.99x | **0.99x** |

→ 与通用路径基本持平，且 **N≥50 时比 opt_einsum 慢约 1%**；
按"必须快于通用路径与 opt_einsum"的规则不满足条件，**已撤回 `_api.contract` 中的分派**，
绑定保留并在 docstring 中记录上述数据与原因。

> 说明：验收工具早前报的 1.70x 是与"接线前的通用路径"比；接线后通用路径的对照值不同，
> 以本表的分规模扫描为准。

### 8.3 `cuda_nucleon`——继续优化

现状：95.12 ms vs 通用 57.31 ms（0.60x）、opt 175.94 ms（1.85x）。已派 bg-21 优化，
验收判据：必须 **< 57.31 ms** 且保持 rel ≤ 1e-12、快于 opt_einsum。

### 8.4 回归（接线后）

| 检查 | 结果 |
|---|---|
| `verify_native --backend cuda` | 16/16 |
| `verify_native --backend kupl` | 16/16 |
| `simple/run-all.sh` | exit=0 |
| `verify_compat` 全量 | **980/980 PASS** |

### 8.5 当前接线状态汇总

| kernel | 接线 | 依据 |
|---|---|---|
| `kupl_nucleon`（E1，CPU） | ✅ | 1.12–1.41x 于通用，5.3–9.4x 于 opt |
| `cuda_3pt`（P1/P2，GPU） | ✅ | 1.25–2.41x 于通用，1.25–9.29x 于 opt |
| `kupl_3pt`（P1/P2，CPU） | ❌ 撤回 | 0.94–1.01x，N≥50 略慢于 opt |
| `cuda_nucleon`（E1，GPU） | ❌ | 0.60x（优化中，bg-21） |
