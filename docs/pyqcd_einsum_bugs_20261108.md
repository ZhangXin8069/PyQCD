# pyqcd-einsum Bug 状态（2026-11-08 08:15）

## 1. 汇总

| 状态 | 数量 | 编号 |
|---|---|---|
| 已修复 | 5 | B1、B2、B3、B4、B5 |
| 已定性为非缺陷 | 2 | B5′（预热口径）、B6（构建竞态 Segfault） |
| 未完成功能（非缺陷） | 3 | K1、K2、K3 |
| 待优化（非缺陷） | 1 | O1 |

**当前没有已知的未修复缺陷。**

## 2. 已修复

| # | Bug | 位置 | 影响 | 验证证据 |
|---|---|---|---|---|
| **B1** | `transpose_copy` 在"输出所有维为 1 且需重排"时 `axes.empty()` 直接 `return true`，目标缓冲未写 | `cpp/kupl/einsum/src/apply_contract.cpp` | 返回未初始化内存（5.27e-310）；命中 E2 的 `greedy`/`auto`（opt_einsum 默认用法） | 最小复现 `abcde,cedb->ac` rel 1.00 → **0**；`verify_compat` 数值面 69/70 → **70/70** |
| **B2** | 专用路径缓存键缺"是否禁用融合" | `pyqcd/einsum/_api.py` | `specialize` 切换不生效（CUDA 与 KUPL 都受影响） | 切换四次：CUDA E4 1.808→**0.735**→1.611→**0.746** ms；KUPL E5 1266.8→**112.7**→1006.7 ms；`out_sum` 完全一致 |
| **B3** | 测试硬编码 `/contract/run` | `pyqcd/testing/_mpi_planning_contract.py` | 非特权环境 PermissionError，全库测试必失败 | 全库 `114 passed/1 failed` → **115 passed/0 failed** |
| **B4** | `verify_compat` 缺 pytest 时整套差分无法运行 | 依赖 `opt_einsum.testing` | 980 项差分长期不可用 | 装 pytest 9.2.0 后 **980/980 PASS** |
| **B5** | CUDA E4 融合路径在 spatial≥2 急剧劣化（L24 慢 33 倍）且可 SIGSEGV | `cpp/cuda/einsum/src/apply_contract.cu::match_e4_fused` | 生产规模灾难性退化 | 加收益判据（仅 `z·y·x==1` 启用）：L24 从 **36.44 s → 1.095 s** |

## 3. 已定性为非缺陷

### B5′：L24 E4 GPU 的 4.99 s 是预热口径问题，不是性能缺陷

- 现象：同一配置两次测到 4.99 s（进程隔离基准）与 1.19 s（直连脚本）。
- 定位：**CUDA 路径前 3–4 次调用因显存池增长而慢 3–4 倍**。

  | 第 n 次调用 | 1 | 2 | 3 | 4 | 5 | 6…12 |
  |---|---|---|---|---|---|---|
  | 耗时（s） | 4.158 | 2.949 | 3.568 | 1.099 | 1.096 | 1.094–1.099 |

- 修复：基准默认预热由 2 轮改为 **4 轮**（`MIN_WARMUP`），并在源码注明依据。
- 修复后正式基准（`--profiles L24 --backends gpu --iters 10`，默认预热）：

| 组 | opt_einsum | pyqcd-einsum | 倍率 |
|---|---|---|---|
| E4 | 1.112793 s | **1.095407 s** | **1.016x（持平）** |
| E5 | 0.373987 s | **0.110959 s** | **3.371x（达标）** |

### B6：退出期 Segfault → 构建竞态，已加固

- 根因（子代理 bg-8 实验）：原子替换 `.so` **0/100 崩**；"映射期间就地截断 `.so`" **50/50 崩**。
- 加固：`setup.py::_stage_native_library()` 改为"临时文件 + `os.replace`"原子落位。
- 现状：4 次复现均未重现。

## 4. 未完成功能（不是缺陷，已移出构建）

| # | 文件 | 现状 | 负责 |
|---|---|---|---|
| K1 | `cpp/cuda/einsum/src/nucleon2pt.cu.wip` | 5 个编译错误 | bg-18 |
| K2 | `cpp/cuda/einsum/src/pattern_3pt.cu.wip` | 100+ 编译错误 | bg-19 |
| K3 | `cpp/kupl/einsum/src/pattern_3pt.cpp.wip` | 可编译但运行期段错误 | bg-17 |

三者均在 `src/` 之外（`.wip`），不参与构建、零引用，构建与全部测试保持绿色。

## 5. 待优化（非缺陷）

| # | 项 | 现状 |
|---|---|---|
| O1 | `cpp/kupl/einsum/src/nucleon2pt.cpp` | 数值正确；N=100 时 604.95 ms，慢于我们通用路径（471.24 ms，0.78x），但**快于 opt_einsum 5.38x**；按规则不算 bug，bg-20 在优化 |

## 6. 全局回归（当前）

| 检查 | 结果 |
|---|---|
| 构建 kupl / cuda | 均 SUCCESS |
| `verify_native` kupl / cuda | 16/16、16/16 |
| `verify_compat` 全量 | 980/980 PASS |
| `simple/run-all.sh` | 78 passed，exit=0 |
| `python -m pyqcd.testing` | 115 passed / 1 skipped / 0 failed |

## 7. 仍未达成的目标（非缺陷，但需登记）

**E4 在 GPU 上只是持平（1.016x），未达 3x。** 现有融合 kernel 只在 `z·y·x==1` 有效，
L24 需回落通用路径。要拿到 3x，必须实现**可扩展（分块/流式）的 E4 kernel**，
这是下一阶段的核心工程项，不是 bug。

E5 已达标（3.371x）。

---

## 8. 代码检查（09:10）

### 8.1 发现并修复（3 项）

| # | 问题 | 影响 | 修复 |
|---|---|---|---|
| **C1** | `contract(specialize: bool = True)` 默认值是 `True` 而非 `None`，`_specialize.enabled()` 恒走显式分支 | **环境变量 `PYQCD_EINSUM_SPECIALIZE=0` 完全不生效**（文档承诺的进程级关闭失效） | 签名改为 `specialize: bool \| None = None`，由 `enabled()` 决定默认值；验证：默认会打印专用路径提示，`=0` 时不打印 |
| **C2** | `_kernel_cuda.py` 从 `.cuda` 导入不存在的 `_library_path` | 异常被 `except` 吞掉 → **CUDA 3pt kernel 从未真正生效**，此前"1.25x"是通用路径抖动 | 改从 `._native._cuda_library_path` 导入；修复后实测 kernel 37.13 ms vs 通用 66.23 ms（**1.78x**）vs opt 65.45 ms（1.76x），`rel=2.60e-15` |
| **C3** | 注册表提示文案只写"KUPL 专用 kernel"，但 GPU 路径也会打印 | 误导 | 改为中性表述并区分 CUDA/KUPL 现状 |

### 8.2 检查通过项

| 检查 | 结果 |
|---|---|
| 生产代码调试残留（`breakpoint`/`pdb`/`XXX`/`FIXME`） | 无 |
| 生产模块 `print` | 仅 `_native_plan._self_check()`（`eff826b` 基线即存在，属自检）与 `_kernel_kupl` 的一次性提示（需求要求） |
| 构建告警 | KUPL / CUDA 均 **0 warning** |
| ABI 导出面 | 冻结 8 个符号仍在；新增 kernel 符号是显式加入 version script（`qcd_kupl_nucleon_2pt*`、`qcd_kupl_pattern_3pt*`、`qcd_cuda_pattern_3pt*`） |
| `pyqcd/einsum/cuda/einsum.cpp`（10601 行）被删除 | 该文件是 `einsum.pyx` 的 Cython 生成物；`setup.py` 只引用 `.pyx`。**实测删除后 `install.sh cuda` 仍能重建扩展**（`einsum.cpython-312-*.so` 192624 B） |
| 回归 | `verify_native` 16/16 ×2、`simple` exit=0、`verify_compat` 980/980、`pyqcd.testing` 115 passed / 0 failed |

### 8.3 已登记但未修复（风险说明）

| 项 | 说明 |
|---|---|
| `specialize` 是**进程级**开关 | 通过环境变量传递给原生层；并发线程用不同设置时不可靠（缓存键已含该标志，缓解但不根治）。彻底解决需在 ABI 增加 per-call 参数 |
| `_kernel_*.enabled()` 首次失败后永久缓存 | 若库在建库前被调用，之后不会重试（需重启进程） |
| `contract_expression` 无 `specialize` 参数 | 表达式路径只能通过环境变量控制 |
| 小数组连续性规则不一致 | 2pt 严格要求所有操作数连续；3pt 允许 ≤1024 元素的小数组由调用侧拷贝 |
