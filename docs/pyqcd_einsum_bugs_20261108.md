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

---

## 9. 二次复查发现并修复（10:05）

### 9.1 【严重】CUDA 2pt kernel 缺失连续性校验 → 非连续输入算出错误结果

| 项 | 内容 |
|---|---|
| 现象 | 传入 **cupy Fortran-order**（非 C 连续）数组时，`try_nucleon_2pt` 仍返回命中，且结果 `rel = 1.69`（完全错误） |
| 根因 | `_kernel_cuda._nucleon_shape_ok` 只校验了类型与 dtype，**漏了 `flags['C_CONTIGUOUS']`**；而 3pt 的 `_shape_ok` 有该检查，两者不一致 |
| 危害 | 设备端不做 stride 解释，按连续内存读取 → 静默错误结果（比崩溃更危险） |
| 修复 | 补上连续性校验：非连续即返回 `None` 回退通用路径 |
| 验证 | 非连续：`hit=False`、`rel=1.35e-15`；连续：`hit=True`、`rel=3.83e-14` |

**已加入回归**（`simple/check_kernels.py` 新增 2 项）：
`cuda/nucleon-noncontiguous`（不应命中）与 `cuda/nucleon-noncontiguous-correct`（结果仍正确）。
kernels 子项由 16 → **18 项**。

### 9.2 边界行为复核（全部通过）

| 场景 | 结果 |
|---|---|
| `nmom=2`（输出长度 2） | shape=(2,) 与参照一致，`rel=4.05e-14`，kernel 命中 |
| `out=` 传入 | 不使用 kernel（`r is out` 为 True），`rel=2.00e-15` |
| `optimize=False` | `rel=2.40e-14` |
| `PYQCD_EINSUM_KERNEL_CUDA=0` | 回退，`rel=2.00e-15` |
| 注册表一致性 | E1/P1/P2 = `fused(needs-optimization)`；E4/E5 = `fused`；其余 `plan-cache` |
| 静态检查 | 语法通过、无 `breakpoint`/`pdb`/`TODO`/`FIXME` 残留 |

### 9.3 回归（修复后）

| 检查 | 结果 |
|---|---|
| `verify_native` kupl / cuda | 16/16、16/16 |
| `simple/run-all.sh` | **96 passed，exit=0** |
| `verify_compat` 全量 | **980/980 PASS** |
| `python -m pyqcd.testing` | **115 passed / 1 skipped / 0 failed** |
| `git diff --check` | 干净 |

---

## 10. 三次复查发现并修复（10:20）

### 10.1 【严重】专用 kernel 异常会抛给调用方，违反回退契约

| 项 | 内容 |
|---|---|
| 契约要求 | `pyqcd/einsum/AGENTS.md` §7："任何原生路径失败 → 记录一次 warning，回退纯 Python，**不抛给调用方**" |
| 现象 | 注入 kernel 异常后，`contract()` 直接抛出 `RuntimeError`，没有回退 |
| 根因 | `_api.contract` 中对 `_kernel_kupl/_kernel_cuda` 的调用**没有 try/except** |
| 修复 | 新增 `_run_specialized_kernel()`（依次尝试四个 kernel）与 `_warn_specialized_kernel_once()`；异常 → warning 一次 → 继续走通用路径 |
| 验证 | 注入失败后：结果 shape 正确、warning 内容匹配、**第二次调用不再重复告警** |

已加入回归：`kupl/nucleon-exception-fallback` 与 `kupl/nucleon-exception-warn`（kernels 18 → **20 项**）。

### 10.2 KUPL 连续性行为复核（正确，无需修改）

| 场景 | 2pt | 3pt |
|---|---|---|
| 全部连续 | hit=True，rel 4.05e-15 | hit=True，rel 8.97e-16 |
| **大张量**非连续 | hit=False（回退），rel 0 | hit=False（回退），rel 0 |
| **小张量**非连续（如 4×2 投影） | hit=True（内部拷贝），rel 4.05e-15 | hit=True，rel 8.97e-16 |

### 10.3 回归（修复后）

| 检查 | 结果 |
|---|---|
| `verify_native` kupl / cuda | 16/16、16/16 |
| `simple/run-all.sh` | **98 passed，exit=0**（compat 39 + memory 4 + equations 9 + native 13×2 + kernels 20） |
| `verify_compat` 全量 | **980/980 PASS** |
| `python -m pyqcd.testing` | **115 passed / 1 skipped / 0 failed** |

---

## 11. 四次复查（10:35）—— 未发现新缺陷，补两项实证

### 11.1 开关与分派路径验证（全部正确）

| 场景 | 结果 |
|---|---|
| `PYQCD_EINSUM_SPECIALIZE=0` | **kernel 调用次数 = 0**（完全不进专用路径），结果正确 |
| `optimize` 传显式 path | rel 6.19e-16（kernel 命中，结果正确） |
| `optimize=False` | rel 6.19e-16 |
| `contract_expression(...)` 调用 | rel 5.50e-16（正确） |

### 11.2 已知限制（实证补充）：`contract_expression` 不走专用 kernel

实测统计（monkeypatch 计数器）：

| 入口 | `_run_specialized_kernel` 调用次数 |
|---|---|
| `contract(...)` | **1** |
| `contract_expression(...)` 返回的表达式调用 | **0** |

→ 表达式路径结果正确，但**不享受专用 kernel 加速**（`_expr.py` 直接编译原生程序，
不经过 `contract()` 的形态分派）。属已知限制，已在 §8.3 登记；修复需要把 `specialize`
语义传入表达式编译层，暂未实施。

### 11.3 资源泄漏检查（通过）

- 连续 200 次 kernel 路径调用：GPU 空闲内存 **7088.0 MiB → 7088.0 MiB（变化 0）**
- 200 次后结果与首次一致（`np.allclose` 为 True）
- 进程峰值 RSS 436.9 MiB
