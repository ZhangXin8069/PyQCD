# PyQCD 全库回归 + einsum 子系统验证报告（2026-11-08）

本报告为**独立验证**结论：不修改任何源码，只执行任务书列出的 6 条检查并记录退出码、
PASS/FAIL 计数与失败详情，再与"改动前基线"（`1e2eea2` 的父提交 `1150334` 的完整代码树）做逐项对比。

## 0. 结论速览

| # | 检查 | 退出码 | 结果 | 判定 |
|---|---|---|---|---|
| 1 | `verify_native --backend kupl` | 0 | 16 passed / 0 skipped / 0 failed | **PASS** |
| 2 | `verify_native --backend cuda` | 0 | 16 passed / 0 skipped / 0 failed | **PASS** |
| 3 | `verify_compat --smoke` | 1 | 0 断言执行（`FileNotFoundError`） | **未执行**：环境问题（参考实现子模块未初始化） |
| 4 | `verify_compat`（全量） | 1 | 0 断言执行（同上，0.30 s 即退出） | **未执行**：同 3 |
| 5 | `python -m pyqcd.testing` | 1 | 114 passed / 1 skipped / 1 failed | **1 项失败**：`test_parallel_mpi_planning_contracts`（环境问题，基线同样失败） |
| 6 | `bash pyqcd/testing/einsum/simple/run-all.sh` | 0 | 76 passed / 0 skipped / 0 failed | **PASS** |

补充执行（任务书未列，用于把"检查 3/4 未执行"补成可判定的结论，见 §3.1、§5）：
用**冻结参考实现**（`opt_einsum` @ `fccdd5aa`，与 `git submodule` 钉住的 commit 一致）跑
差分测试 —— 全量 `PASS 979/980`，唯一失败项 `wick_3pt`，且**改动前基线同样失败**（同
`979/980`、同一用例），故判定为**既有产品缺陷，非本次改动引入**。

**总判定：本次改动未破坏任何既有行为。** 1 项全库失败、1 项（含 1 条断言）差分失败均为
既有问题（分别属环境依赖与既存数值缺陷），与本次改动无关。

## 1. 环境与版本快照

| 项 | 值 |
|---|---|
| 主机 | `Linux-6.18.54.1-microsoft-standard-WSL2-x86_64`（WSL2，8 核 / 15 GB） |
| Python | 3.12.4（`/home/zhangxin/miniforge3/bin/python`） |
| numpy | 2.5.4 |
| CuPy / GPU | 14.2.0 / NVIDIA GeForce RTX 4060 Laptop GPU（1 device） |
| 站点 opt_einsum | 3.4.0（`refer/git-rep/opt_einsum` 子模块**未初始化**，目录为空） |
| MPI | `prterun` / `mpiexec` 可用（OpenMPI，2 rank 测试可跑） |
| pytest | **未安装**（`import pytest` → `ModuleNotFoundError`） |

Git 状态：

- 验证开始时 HEAD = `1e2eea2`（2026-11-08 05:50:50，改动主体：`cpp/kupl/einsum/src/apply_contract.cpp`、
  `cpp/cuda/einsum/src/e4_fused.cu`、`pyqcd/einsum/_api.py`、`pyqcd/einsum/cuda/_program.py`、
  `pyqcd/testing/einsum/bench_distillation.py`）。
- 验证过程中另一代理于 **06:02:31** 提交 `fa2e74a`（新增 `pyqcd/testing/einsum/simple/**`、C++ buffer-pool
  调整）。该提交只新增文件与 C++ 源码，`pyqcd/einsum/*.py` 未变，故 Python 侧结论对两个 commit 都成立。
- 06:02:31（`fa2e74a` 提交）时工作树干净；之后另一代理仍在跑基准（`artifacts/bench_20261108060403/` 等陆续出现）。
  本次验证**只新增本报告文件**，未改任何源码、未改任何测试期望。

原生库指纹（`_native.library_path()` 实际加载的文件）：

| 后端 | 文件 | 构建时间 | SHA-256（前 16 位） |
|---|---|---|---|
| kupl | `cpp/kupl/einsum/build/libpyqcdeinsum_kupl.so` | 05:56:23 | `26e5a8da1388e0fc` |
| cuda | `pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so`（包内副本） | 05:56:00 | `9f23111e830edb6b` |

> 并发提示：`cpp/cuda/einsum/build/libpyqcdeinsum_cuda.so` 在 06:01:26 被另一代理重建成
> `9ed71972e7dc807d`（1301752 B），但**包内副本未刷新**，因此本次所有 CUDA 检查加载的始终是
> `9f23111e830edb6b`（1322232 B）。检查 1/2 在 06:01:59–06:02:00 复跑，运行前后哈希不变。
> 期间另有代理在跑 `bench_native.py`（`artifacts/bench_2026…` 目录 05:51–05:59 陆续出现），
> CPU 存在竞争，但**未遇到 `.so` 被替换导致的瞬时 ImportError**，因此没有触发"重试一次"分支。

## 2. 逐条检查记录

### 2.0 改动覆盖面（哪条检查真正触达了改动文件）

| 改动文件 | 触达它的检查 | 关键断言 |
|---|---|---|
| `cpp/kupl/einsum/src/apply_contract.cpp` | 1、5、6 | `kupl/*`（16 项）、`test_einsum_*`、`native[kupl]` 13 项 |
| `cpp/cuda/einsum/src/e4_fused.cu` | 2、5、6 | `cuda/*`（16 项）、`test_einsum_*`、`native[cuda]` 13 项 |
| `pyqcd/einsum/_api.py` | 1、2、3/4（未到断言）、5、6 | 全部；新增 `out=` 由 `out-contract` / `out/identity` / `out/values` 覆盖，新增 context 缓存由 `cache-value-1/2/3`（同对象签名、三次不同数值）与 `expression/reuse-1/2` 覆盖 |
| `pyqcd/einsum/cuda/_program.py` | 2、5、6 | `cuda/*`、`check_native --backend cuda`、`E*/cpu` 方程 |
| `pyqcd/testing/einsum/bench_distillation.py` | 无（6 条检查都不加载） | 仅补 `import` + `--help` 冒烟（见本节末） |
| `pyqcd/testing/einsum/simple/**`（`fa2e74a` 新增） | 6 | 76 项 |

> 说明：本次改动的新能力（`out=` 直写、原生 context 缓存）在上述断言中均为 PASS，
> 包括"同签名连续三次不同数值"（`cache-value-1/2/3`）这类容易踩缓存复用坑的用例。

### 检查 1：`verify_native --backend kupl`

```
$ cd /home/zhangxin/PyQCD
$ python -m pyqcd.testing.einsum.verify_native --backend kupl
```

- 退出码：**0**
- 耗时：0.27 s
- 输出（16 条 PASS，末行汇总）：

```
PASS kupl/matmul/complex128      PASS kupl/cache-value-1
PASS kupl/matmul/complex64       PASS kupl/cache-value-2
PASS kupl/matmul/float64         PASS kupl/cache-value-3
PASS kupl/matmul/float32         PASS kupl/out-contract
PASS kupl/chain                  PASS kupl/contract-expression
PASS kupl/batch                  PASS kupl/repeated-axis-fallback
PASS kupl/unary-permute          PASS kupl/native-path-used
PASS kupl/dot
PASS kupl/sandwich
kupl: 16 passed, 0 skipped, 0 failed
```

- 计数：**PASS 16 / SKIP 0 / FAIL 0**

### 检查 2：`verify_native --backend cuda`

```
$ PYQCD_EINSUM_NATIVE=cuda python -m pyqcd.testing.einsum.verify_native --backend cuda
```

- 退出码：**0**
- 耗时：0.81 s
- 输出：与检查 1 同构的 16 条 `PASS cuda/…`，末行 `cuda: 16 passed, 0 skipped, 0 failed`
- 计数：**PASS 16 / SKIP 0 / FAIL 0**

### 检查 3：`verify_compat --smoke`

```
$ python -m pyqcd.testing.einsum.verify_compat --smoke
```

- 退出码：**1**
- 耗时：0.26 s
- 关键输出（完整栈底）：

```
  File "/home/zhangxin/PyQCD/pyqcd/testing/einsum/verify_compat.py", line 323, in run
    oracle_version, oracle_path = _oracle.oracle_version()
  File "/home/zhangxin/PyQCD/pyqcd/testing/einsum/_oracle.py", line 51, in oracle
    raise FileNotFoundError(f'缺少参考实现目录: {ORACLE_PACKAGE}')
FileNotFoundError: 缺少参考实现目录: /home/zhangxin/PyQCD/refer/git-rep/opt_einsum/opt_einsum
```

- 计数：**PASS 0 / SKIP 0 / FAIL 0**（崩溃发生在任何断言之前）
- 判定：**环境问题**（详见 §3.1），非产品缺陷。

### 检查 4：`verify_compat`（全量）

```
$ python -m pyqcd.testing.einsum.verify_compat
```

- 退出码：**1**
- 耗时：0.30 s
- 关键输出：与检查 3 完全相同的 `FileNotFoundError`（同一个 `_oracle.oracle_version()` 调用点）
- 计数：**PASS 0 / SKIP 0 / FAIL 0**
- 任务书的"超 60 分钟则跳过"条款**未触发**：进程 0.30 s 即退出。
- 判定：**环境问题**（同 §3.1）。

### 检查 5：`python -m pyqcd.testing`（全库）

```
$ python -m pyqcd.testing
```

- 退出码：**1**
- 耗时：92.17 s（`fa2e74a` 提交后用同一命令复跑：**77.57 s，114 passed / 1 skipped / 1 failed，
  PASS 集合与首跑逐行相同**，见 §7）
- 汇总行：`114 passed, 1 skipped, 1 failed`
- 失败项（1）：`test_parallel_mpi_planning_contracts`
- 跳过项（1）：`test_pipeline_runtime_failure_contracts: xelatex is unavailable`
- einsum 相关子项全部 PASS：

```
PASS test_einsum_cuda_mpi_partition_contracts
PASS test_einsum_mixed_native_backends
PASS test_einsum_native_contracts
PASS test_einsum_zero_extent_and_noncontiguous_contracts
```

- 失败项的完整错误：

```
Traceback (most recent call last):
  File "/home/zhangxin/PyQCD/pyqcd/testing/conftest.py", line 134, in main
    fn()
  File "/home/zhangxin/PyQCD/pyqcd/testing/__init__.py", line 2907, in test_parallel_mpi_planning_contracts
    test()
  File "/home/zhangxin/PyQCD/pyqcd/testing/_mpi_planning_contract.py", line 556, in test_cupy_binds_rank_device_before_selecting_backend
    backend_events, observed = _run_backend_meta_task(
  File "/home/zhangxin/PyQCD/pyqcd/testing/_mpi_planning_contract.py", line 531, in _run_backend_meta_task
    mpi.run_parallel_pipeline(
  File "/home/zhangxin/PyQCD/pyqcd/parallel/_mpi.py", line 606, in run_parallel_pipeline
    _collective_failure_rendezvous(comm, setup_failure)
  File "/home/zhangxin/PyQCD/pyqcd/parallel/_mpi.py", line 103, in _collective_failure_rendezvous
    raise RuntimeError(f'MPI collective failure: {details}')
RuntimeError: MPI collective failure: phase=setup rank=1 type=PermissionError message=[Errno 13] Permission denied: '/contract'
```

- 判定：**环境问题**（详见 §3.2），基线同样失败。

### 检查 6：`pyqcd/testing/einsum/simple/run-all.sh`

说明：任务开始时该目录尚不存在（`ls: cannot access 'pyqcd/testing/einsum/simple': No such file or directory`），
05:57 由另一代理写入、06:02:31 随 `fa2e74a` 提交；执行时已就绪，故按"存在则执行"处理。

```
$ bash pyqcd/testing/einsum/simple/run-all.sh
```

- 退出码：**0**
- 耗时：3.25 s
- 关键输出（五段汇总）：

```
===== python3 -m pyqcd.testing.einsum.simple.check_compat =====
compat: 39 passed, 0 skipped, 0 failed
===== python3 -m pyqcd.testing.einsum.simple.check_memory =====
memory: 4 passed, 0 skipped, 0 failed
===== python3 -m pyqcd.testing.einsum.simple.check_equations =====
PASS E1/cpu  PASS E2/cpu  PASS E4/cpu  PASS E5/cpu  PASS S1/cpu  PASS S2/cpu  PASS S3/cpu
equations: 7 passed, 0 skipped, 0 failed
===== env PYQCD_EINSUM_NATIVE=kupl python3 -m …check_native --backend kupl =====
native[kupl]: 13 passed, 0 skipped, 0 failed
===== env PYQCD_EINSUM_NATIVE=cuda python3 -m …check_native --backend cuda =====
native[cuda]: 13 passed, 0 skipped, 0 failed

===== simple suite exit=0 =====
```

- 计数：**PASS 76 / SKIP 0 / FAIL 0**（39 + 4 + 7 + 13 + 13）

### 附加冒烟（不属于 6 条检查，用于覆盖未被任何检查触达的新增文件）

`pyqcd/testing/einsum/bench_distillation.py` 只被基准流程使用，6 条检查都不加载它，故补一条可导入性冒烟：

```
$ python -c "import pyqcd.testing.einsum.bench_distillation"
import OK
$ python -m pyqcd.testing.einsum.bench_distillation --help
usage: bench_distillation.py [-h] [--profiles PROFILES] [--backends BACKENDS] …
```

结论：**可导入、CLI 正常**。未运行实际基准（不在本次任务范围，且机器上有并发基准任务）。

## 3. 失败项详情与判定

### 3.1 检查 3 / 4：参考实现（oracle）缺失 → 环境问题

- 现象：`_oracle.oracle()` 在 `refer/git-rep/opt_einsum/opt_einsum` 不存在时直接
  `raise FileNotFoundError`；`verify_compat.main()` 没有兜底，因此**整个差分测试一条断言都没跑**。
- 根因：`refer/git-rep/opt_einsum` 是 git 子模块（`git ls-files -s` → `160000 fccdd5aa…`），
  本机所有子模块均未初始化（`git submodule status` 全部以 `-` 开头），目录为空。
  证据：`ls -la refer/git-rep/opt_einsum` → 只有 `.` / `..`。
- 与本次改动的关系：`pyqcd/testing/einsum/_oracle.py`、`verify_compat.py` 均**未被本次改动触及**
  （最近一次提交为 `eff826b`），且 `docs/pyqcd_einsum_performance_status_20261108.md` §4 已记录
  上一轮同样无法初始化该子模块（当时是 `GnuTLS recv error (-110)`）。
- 判定：**环境问题（既有）**，不属于产品缺陷，也不是本次改动引入。
- 附带发现（非阻塞）：`verify_compat.py` 对"oracle 目录缺失"没有像"目标包未就绪"那样输出
  `SKIP` 并以 exit 0 退出，而是直接抛栈，容易在 CI 中被误读成产品回归。属**既有**测试脚手架健壮性缺口。
- 补救（不改变仓库状态）：见 §3.3 与 §5 —— 把钉住的参考实现 clone 到**会话临时目录**，
  用 `PYTHONPATH` 前置 + `PYQCD_EINSUM_ORACLE=installed` 运行，绕开空子模块，
  从而得到可判定的差分结论。

### 3.2 检查 5：`test_parallel_mpi_planning_contracts` → 环境问题

- 现象：MPI 2 rank 运行时，rank 1 在 setup 阶段抛
  `PermissionError: [Errno 13] Permission denied: '/contract'`。
- 根因：`pyqcd/testing/_mpi_planning_contract.py:526/533` 把 `run_dir` 硬编码为
  `'/contract/run'`，`run_parallel_pipeline` 在 setup 阶段会真的创建该目录；
  本机 `/` 对当前用户不可写（`test -w /` → false，`/contract` 不存在）。
- 与本次改动的关系：该测试文件最近一次提交是 `2d57a28`（2026-08-31），**未被本次改动触及**；
  该项依赖 MPI/文件系统权限，与 einsum 无关。**改动前基线（`1150334` 树）跑同一套件得到
  完全相同的失败**（见 §4），证实非本次引入。
- 判定：**环境问题（既有，跑测环境 `/` 不可写）**。若要本地全绿，需要可写的
  `/contract`（容器内以 root 运行或预先 `mkdir /contract` 并授权）——属环境侧修复，不建议为它改测试期望。
- 最小复现：

```bash
cd /home/zhangxin/PyQCD
python -c "from pyqcd.testing.conftest import test_parallel_mpi_planning_contracts as t; t()"
# RuntimeError: MPI collective failure: phase=setup rank=1 type=PermissionError
#   message=[Errno 13] Permission denied: '/contract'
```

### 3.3 差分测试唯一失败用例 `wick_3pt` → 产品缺陷（既有，非本次引入）

- 用例：`verify_compat.CURATED_CASES` 的 `('distill', 'wick_3pt',
  'Geifj,ckdl,agbh,ce,gi,Mbdf,Mhjl,Gka->GM', …)`
- 现象：`rel = 1.0`，`max|d| = 4160.312306572801`；pyqcd 侧返回的是**未写入的输出缓冲**
  （值随进程/调用变化，形如 `5.1e-310+0j`，范数≈0），而 numpy / opt_einsum 结果一致
  （numpy greedy 与 oracle 的 `max|d| = 2.8e-12`）。
- 定位（只读探测，未改源码）：
  - `optimize='greedy' / 'auto' / 'branch-2'` → 失败；`'optimal' / 'dp' / False` → 通过。
  - pyqcd 规划出的 path 与 opt_einsum **逐位相同**
    （`[(0,3),(1,2),(0,3),(0,4),(1,3),(1,2),(0,1)]`，`opt_cost=3078`）；
    把该 path 交给 opt_einsum 执行 → `rel = 1.2e-16`（**路径正确**）。
  - 打桩禁用原生执行（`_api._try_native_contract → (False, None)`）后结果仍为未初始化缓冲
    → 问题在**纯 Python 执行层**（`_execute.py` / `_plan_py.py`，两者均未被本次改动触及，
    最近提交分别为 `534e244` / `eff826b`），不是 KUPL/CUDA kernel。
  - `PYQCD_EINSUM_CONTEXT_CACHE_LIMIT=0`（关掉本次新增的 context 缓存）后现象不变。
- 基线对比：`1150334` 树上同一用例同样 `rel=1.0`、同样返回未初始化缓冲（见 §4）→ **既有缺陷**。
- 判定：**产品缺陷（既有）**，与本次改动无关；但值得单独立项修复（`greedy/branch-2` 路径下
  某一步的输出缓冲未写满/未写）。本次按"如实报告、不修改测试期望"处理。
- 最小复现（不需要 oracle、不需要额外依赖，直接跑在仓库根）：

```bash
cd /home/zhangxin/PyQCD
python - <<'PY'
import numpy as np
from pyqcd.testing.einsum import _oracle, verify_compat
from pyqcd.einsum import contract
eq = 'Geifj,ckdl,agbh,ce,gi,Mbdf,Mhjl,Gka->GM'
shapes = [(1,2,2,3,3),(2,2,3,3),(2,2,3,3),(2,2),(2,2),(1,3,3,3),(1,3,3,3),(1,2,2)]
arrays = _oracle.rand_arrays({'eq': eq, 'shapes': shapes, 'name': 'wick_3pt'},
                             seed=verify_compat._seed_of('wick_3pt'))
ref = np.einsum(eq, *arrays, optimize='greedy')
for opt in ('greedy', 'dp'):
    got = np.asarray(contract(eq, *arrays, optimize=opt))
    print(opt, 'max|d| =', np.abs(got - ref).max(), got.ravel()[:1], ref.ravel()[:1])
PY

# 实测输出：
# greedy max|d| = 4160.312306572804 [5.11781043e-310+0.j] [-3748.59405671-1804.5057734j]
# dp     max|d| = 3.410605131648481e-12
```

## 4. 与"改动前基线"的差异

基线构造方式（**全程不修改仓库、不动 git 索引、不建 worktree**）：

- "本次改动"= commit `1e2eea2`（验证开始时的 HEAD）；"改动前基线"= 它的父提交 `1150334`
  （`git archive 1e2eea2~1`）。两者之差就是 `1e2eea2` 提交本身：任务书点名的 5 个文件
  （`apply_contract.cpp`、`e4_fused.cu`、`_api.py`、`cuda/_program.py`、`bench_distillation.py`）
  外加 `AGENTS.md`/`docs`/`logs`/基准产物。

```bash
# 1) 取出改动前的树（1e2eea2~1 = 1150334）到会话临时目录
git archive --format=tar 1e2eea2~1 | tar -x -C <scratch>/baseline
# 2) 把同一批已编译原生库复制进去（.so 被 .gitignore 忽略，不会随 archive 带出）
cp cpp/kupl/einsum/build/libpyqcdeinsum_kupl.so      <baseline>/cpp/kupl/einsum/
cp pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so          <baseline>/pyqcd/einsum/cuda/
cp pyqcd/einsum/cuda/einsum.cpython-312-*.so         <baseline>/pyqcd/einsum/cuda/
# 3) 用同一 frozen oracle 跑，只把 <baseline> 放到 PYTHONPATH 首位
PYTHONPATH=<baseline>:<frozen-oracle> PYQCD_EINSUM_ORACLE=installed python -m pyqcd.testing.einsum.verify_compat
# 4) 全库同理（cwd 切到 <baseline>，python -m 会把它的 pyqcd 放到 sys.path[0]）
cd <baseline> && PYTHONPATH=<baseline>:<frozen-oracle> PYQCD_EINSUM_ORACLE=installed python -m pyqcd.testing
```

| 检查 | 本次（`1e2eea2` / `fa2e74a`） | 基线（`1150334` 树） | 差异 | 判定 |
|---|---|---|---|---|
| 1 kupl native | 16 / 0 / 0，exit 0 | 同（同一 `.so`） | 无 | 未破坏 |
| 2 cuda native | 16 / 0 / 0，exit 0 | 同（同一 `.so`） | 无 | 未破坏 |
| 3/4 compat | 0 断言执行（oracle 缺失） | 同样 0 断言执行（基线树实测 `FileNotFoundError`） | 无 | 环境问题 |
| 3′ compat `--smoke`（frozen oracle） | `PASS 142/143`，1 FAIL（`wick_3pt`） | `PASS 142/143`，同一 FAIL | **无差异** | 既有缺陷 |
| 4′ compat 全量（frozen oracle） | `PASS 979/980`；自检 70/70、API 38/38、路径 802/802、数值 69/70；路径面 800 网络**逐位一致 800** | `PASS 979/980`；同一分布、同一 FAIL | **无差异** | 既有缺陷 |
| 5 全库 | 114 passed / 1 skipped / 1 failed（`fa2e74a` 上复跑同值） | 114 passed / 1 skipped / 1 failed | PASS 集合逐行 `diff` **完全相同**；失败同为 `test_parallel_mpi_planning_contracts`，跳过同为 `test_pipeline_runtime_failure_contracts` | 既有环境问题 |
| 6 simple suite | 76 / 0 / 0，exit 0 | 基线树无 `simple/**`（该目录 05:57 才出现） | 不适用 | 新增子系统自测通过 |

> 结论：**没有任何一项检查出现"本次失败、基线通过"的差异**，因此不存在"本次改动引入的回归"。
> 另用站点 `opt_einsum 3.4.0` 复跑全量差分同样得到 `PASS 979/980`（同一 `wick_3pt` 失败），
> 说明该失败与参考实现版本无关。

## 5. 未执行 / 跳过的检查及原因（明确清单）

1. **检查 3、4 的字面命令未执行任何断言**（0/980）：`refer/git-rep/opt_einsum` 子模块未初始化。
   判定为环境问题；已用"钉住 commit 的参考实现副本 + `PYQCD_EINSUM_ORACLE=installed`"补充执行
   （§4 的 3′/4′），结果单列，**不与字面命令的结果混同**。
   补充执行的两处环境偏差（如实记录）：
   - 参考实现来自 `git clone --depth 1 https://github.com/dgasmith/opt_einsum.git`（HEAD =
     `fccdd5aa2a933c8bd1452f44689d15e737982cc0`，与子模块钉住的 commit **完全一致**，
     版本号 `3.4.1.dev25+g5b54bc1dc`），放在**会话临时目录**，未写入仓库 `refer/`。
   - `opt_einsum.testing` 在 import 时 `import pytest`，而基础环境**没有 pytest**；因此在会话临时
     venv（`--system-site-packages` + 仅装 `pytest 9.2.0`）里运行上述补充命令，基础环境未被改动。
2. **检查 4 的"超 60 分钟"条款未触发**：进程 0.30 s 退出（oracle 缺失，早退）。
3. **`bench_distillation.py` 的实际基准未运行**：不属 6 条检查范围，机器上另有代理在跑并发基准；
   仅做了 import + `--help` 冒烟（通过）。
4. 其余检查（1、2、5、6）均已按任务书执行，无跳过项。

## 6. 复现命令汇总

```bash
cd /home/zhangxin/PyQCD

# 1)
python -m pyqcd.testing.einsum.verify_native --backend kupl
# 2)
PYQCD_EINSUM_NATIVE=cuda python -m pyqcd.testing.einsum.verify_native --backend cuda
# 3) / 4) 需要先初始化参考实现子模块（本机未初始化，故 exit 1）
python -m pyqcd.testing.einsum.verify_compat --smoke
python -m pyqcd.testing.einsum.verify_compat
# 5)
python -m pyqcd.testing
# 6)
bash pyqcd/testing/einsum/simple/run-all.sh

# 补充：用钉住的参考实现跑差分（仓库状态不变）
#   若要用仓库内的子模块实现，先执行：git submodule update --init refer/git-rep/opt_einsum
git clone --depth 1 https://github.com/dgasmith/opt_einsum.git /tmp/opt_einsum_frozen
PYQCD_EINSUM_ORACLE=installed PYTHONPATH=/tmp/opt_einsum_frozen \
  python -m pyqcd.testing.einsum.verify_compat --smoke
```

## 7. 时间线（并发环境记录）

| 时间 | 事件 |
|---|---|
| 05:50:50 | HEAD = `1e2eea2`（本次要验证的改动主体） |
| 05:51–05:56 | 另一代理产出 `artifacts/bench_2026…` 基准目录；重建 `libpyqcdeinsum_kupl.so`(05:56:23)、包内 `libpyqcdeinsum_cuda.so`(05:56:00) |
| 05:56:5x | 检查 1、2 执行（exit 0，16/16） |
| 05:57 | 检查 3、4 执行（exit 1，oracle 缺失，0 断言）；`simple/**` 出现 |
| 05:57:5x–05:59:2x | 检查 5 执行（92.17 s，114/1/1） |
| 05:59–06:01 | 补充差分（frozen oracle）：142/143、979/980；基线复跑：完全相同 |
| 06:01:26 | 另一代理重建 `cpp/cuda/einsum/build/libpyqcdeinsum_cuda.so`（包内副本未刷新） |
| 06:02:00 | 检查 1、2 复跑确认（16/16，运行前后 `.so` 哈希不变） |
| 06:02:31 | 另一代理提交 `fa2e74a`（新增 `simple/**`、C++ buffer-pool 调整） |
| 06:0x | 检查 6 执行（exit 0，76/76） |
| 06:03–06:04 | 在 `fa2e74a` 上用 `python -m pyqcd.testing` 复跑全库：**114 passed / 1 skipped / 1 failed，77.57 s**，失败/跳过项与首跑相同，PASS 集合逐行 `diff` 相同 |
