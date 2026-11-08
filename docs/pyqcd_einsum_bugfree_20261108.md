# pyqcd-einsum 缺陷清零报告（2026-11-08 09:00）

> 目标：在推进性能接线前，把已知缺陷全部消除并给出可复现证据。

## 1. 结论

| 检查 | 结果 |
|---|---|
| KUPL / CUDA 构建 | **均 SUCCESS** |
| `verify_native` kupl / cuda | **16/16 + 16/16** |
| `verify_compat` 全量（980 项） | **980/980 PASS** |
| `simple/run-all.sh`（78 项） | **78 passed，exit=0** |
| `python -m pyqcd.testing`（全库） | **115 passed / 1 skipped / 0 failed** |
| 生产规模方程（T1，9 组） | **9/9**，最差 `rel = 1.444e-15` |

唯一 skip：`test_pipeline_runtime_failure_contracts`（`xelatex` 不可用，环境依赖）。
**当前已知缺陷：0。**

## 2. 本轮发现并修复的缺陷

### 2.1 `transpose_copy` 在输出全 1 维时返回未初始化内存（KUPL）

- 触发：输出所有维为 1 **且**需要输出轴重排
- 最小复现：`abcde,cedb->ac`，L(1,5,1,5,5)、R(1,5,5,5) → `rel = 1.00`、返回 `5.27e-310`
- 影响：E2 方程 `Geifj,ckdl,agbh,ce,gi,Mbdf,Mhjl,Gka->GM` 在 `greedy`/`auto`（opt_einsum 默认用法）下全错
- 根因：`extent == 1` 的轴被全部跳过后 `axes.empty()` 直接 `return true`，目标缓冲未写
- 修复：校验 `source_count == destination_count == 1` 后 `memcpy` 单元素
- 验证：最小复现 rel=0；E2 三 planner = 1.90e-16 / 9.47e-16 / 5.91e-16；
  **`verify_compat` 数值面由 69/70 提升到 70/70**

### 2.2 专用路径开关的缓存键缺项（CUDA 与 KUPL 都受影响）

- 触发：先 `specialize=False` 再切 `True`，专用路径不生效
- 现象：CUDA E4 `False 1.376 ms → True 1.159 ms`（纯净 True 应约 0.68 ms）
- 根因：原生程序缓存键未包含"是否禁用融合"状态，先前编译的通用程序被复用
  （涉及 `_native_plan_key`、`_contract_signature`、`contract_expression` 的 `native_cache`）
- 修复：新增 `_fused_disabled()`，并入上述三处键
- 验证（同一进程内反复切换，数值完全一致）：

| 后端 | False | True | False | True |
|---|---|---|---|---|
| CUDA E4 | 1.808 ms | **0.735 ms** | 1.611 ms | **0.746 ms** |
| KUPL E5 | 1266.82 ms | **112.68 ms** | 1006.71 ms | — |

### 2.3 测试硬编码 `/contract` 路径导致全库测试失败

- `pyqcd/testing/_mpi_planning_contract.py` 用 `run_dir='/contract/run'`，
  非特权环境无法创建 `/contract` → `PermissionError`
- 修复：新增 `_writable_run_dir()`（`tempfile.mkdtemp`），替换硬编码路径
- 验证：该模块 20 个测试 **0 失败**；全库由 `114 passed / 1 failed` 变为
  **`115 passed / 0 failed`**

### 2.4 `verify_compat` 脚手架缺口

- 缺 `pytest` 时 `opt_einsum.testing` 导入失败，整套差分无法运行
- 处置：安装 `pytest 9.2.0`；`verify_compat` 现在可完整运行（980/980）
- **前置条件（写进文档）**：跑 `verify_compat` 需要 `pytest`；
  冻结子模块不可用时必须设 `PYQCD_EINSUM_ORACLE=installed`

## 3. 系统性缺陷普查（未发现问题）

| 普查项 | 用例数 | KUPL | CUDA |
|---|---|---|---|
| 退化用例（输出全 1 维+重排、标量、广播） | 9 | 全对 | 全对 |
| 随机差分（2–4 算子，尺寸 1–3，greedy） | 200 | — | 0 失败 |
| 随机差分（≤6 算子，greedy/dp/auto） | 210 | — | 0 失败 |
| 对角/同一操作数重复标签（含 planner 3 种） | 240 | **0 失败** | **0 失败** |
| ellipsis（5 种形态） | 5 | 全对 | — |
| 四 dtype（complex128/64、float64/32） | 80+40 | 0 失败 | 0 失败 |
| `out=` 身份与跨步写入 | 2 | 全对 | 全对 |
| `memory_limit` 行为对齐 opt_einsum | 10 | **逐项一致** | — |
| `contract_path(shapes=True)` | 1 | 一致 | — |
| 融合路径 E4（spatial 1–3、Nvec 4–7、T 2–3） | 12 | — | 12/12 |
| 融合路径 E5 夹心 | 8 | — | 8/8 |
| `contract_expression` 换数组复用 | 5 轮 | 全对 | 全对 |

## 4. 已隔离的未完成 kernel（不构成缺陷）

| 文件 | 状态 | 原因 |
|---|---|---|
| `cpp/cuda/einsum/src/nucleon2pt.cu.wip` | 隔离 | 5 个编译错误 |
| `cpp/cuda/einsum/src/pattern_3pt.cu.wip` | 隔离 | 100+ 编译错误 |
| `cpp/kupl/einsum/src/pattern_3pt.cpp.wip` | 隔离 | 运行期段错误 |
| `cpp/kupl/einsum/src/nucleon2pt.cpp` | 保留但零引用 | 数值正确，但 N=100 时 612.77 ms > 现有 517.65 ms（0.84x，更慢） |

隔离后构建与全部测试均通过。

## 5. 复现命令

```bash
cd /home/zhangxin/PyQCD
PYQCD_EINSUM_ENABLE_MPI=0 bash build-kupl.sh
PYQCD_CUDA_ARCHITECTURES=8.9 bash build-nv.sh
cp -f cpp/cuda/einsum/build/libpyqcdeinsum_cuda.so pyqcd/einsum/cuda/

PYQCD_EINSUM_NATIVE=kupl python -m pyqcd.testing.einsum.verify_native --backend kupl
PYQCD_EINSUM_NATIVE=cuda python -m pyqcd.testing.einsum.verify_native --backend cuda
PYQCD_EINSUM_ORACLE=installed python -m pyqcd.testing.einsum.verify_compat     # 需 pytest
bash pyqcd/testing/einsum/simple/run-all.sh
python -m pyqcd.testing
```
