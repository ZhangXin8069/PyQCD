---
name: pyqcd-einsum
description: Use when implementing, debugging, or benchmarking pyqcd.einsum against opt_einsum, especially native CUDA/KUPL execution, contraction-plan ABI, distillation workloads, vectorization, MPI, or CUDA streams.
---

# pyqcd-einsum - 原生兼容收缩引擎

## 目的与边界

本技能维护 `pyqcd.einsum` 的兼容 API、收缩规划、原生 ABI、CUDA/KUPL 执行、
蒸馏实战基准与性能门禁。它不定义 LapH/Wick 物理语义，也不替代 `pyqcd-infra`
对 backend、显存、MPI 和计时平台的统一契约。

状态词：

- `implemented`：接口或文件存在。
- `tested`：有当前命令、数值断言或 ABI 输出。
- `performance-validated`：同后端、同 dtype、同输入下稳定超过目标倍率。
- `unverified`：没有本次可复现证据，不得由其他后端的成功外推。

## 按需参考

- 修改原生计划、ABI、CUDA/KUPL 或性能时，读
  [references/native-implementation.md](references/native-implementation.md)。
- **动构建、性能测量或接线专用 kernel 之前必读**
  [references/measured-pitfalls.md](references/measured-pitfalls.md)：
  记录了 2026-11-08 会话中实测踩到的构建坑（CUDA 架构默认值、同名头文件冲突、MPI 探测误判）、
  测量坑（预热不足、对照变成 kernel 比 kernel、用现编 .so 冒充生产库、逐元素误差误判）、
  正确性坑（输出全 1 维未写、设备 kernel 缺连续性校验、异常不回退、开关未进缓存键）
  与已知限制（`contract_expression` 不走 kernel、CUDA MPI 是 stub、E4 融合仅极小规模生效）。
- 只做 API/物理缩并语义时，沿用 `pyqcd-distillation` 与
  `pyqcd-infra`，不要复制其完整规则。

## 强制工作流

1. **先固定兼容面**：以 `pyqcd/einsum/AGENTS.md` 和固定 opt_einsum 版本为
   oracle。规划路径、代价、`PathInfo`、`contract_expression` 与错误行为分别验证。
2. **先规划后执行**：生产路径必须复用 contraction list；禁止每次调用重新在 Python
   中展开所有步骤。固定形状优先使用 `contract_expression` 或等价原生计划缓存。
3. **K 轴最低维**：二元收缩统一 pack/transpose 为
   `A[B,M,K] x B[B,N,K]^T -> C[B,M,N]`。K 必须连续，pack/reduce/output kernel
   必须保证相邻线程或 SIMD lane 访问相邻 K 元素。
4. **一次原生调用**：Python 只负责生成一次计划、校验 dtype/连续性/设备和传入指针；
   多步收缩在 C++/CUDA 内完成。中间结果与工作区必须复用，避免逐步 Python dispatch。
5. **后端独立**：KUPL 不链接 CUDA，CUDA 不调用 PyQCU 库；两边共享 ABI 与计划格式，
   数值实现保持独立。生产代码不得 import `refer/` 或 `pyqcd.testing`。
6. **并行分层**：CUDA 为 MPI rank → 单 device context → 多 stream 批任务；KUPL 为
   MPI rank → 多进程/OpenMP → SIMD。多 rank 只切独立 batch，任何跨 rank 求和必须用
   对称 collective；不得让 rank 提前返回。
7. **性能门禁**：CPU 对 CPU、GPU 对 GPU，固定 dtype、线程、设备、输入、规划与
   stream；预热后取至少 10 次中位数。目标为 opt_einsum 的 3x，且结论需多轮复测。
   低于目标必须返回非零，不能只打印 warning。
8. **失败关闭**：重复轴、混合精度、非连续输入、越界尺寸、缺 kernel、CUDA/cuBLAS
   错误必须显式失败或安全回退；不得把零输出或缺步骤当作成功。

## 当前已验证经验

- 共享执行计划 v2 已 `implemented, tested`：header、对齐尺寸表、初始 shape、
  逐步 positions/labels 与最终输出共同描述 contraction list。
- K-lowest 布局分析已由
  `cpp/kupl/einsum/tests/test_runtime.cpp` 的 C++ 合约测试覆盖。
- KUPL 四 dtype、batch、广播、一元 reduce/permute 和多步收缩已 `tested`；
  当前全量入口记录为 `111 passed, 1 skipped`。
- CUDA 编译、动态导出与 Cython 加载已 `implemented`，但一般 cuBLAS GEMM 路径仍
  `unverified`，不得用逐元素路径通过代替 batch GEMM 通过。
- 开发窗口 CPU T0 四条方程约 `3.0x-18x`，CUDA V100/P100 E5 约
  `3.49x-4.07x`；但固定 oracle、进程隔离、MPI/MP 与目标系综仍未全部闭合，
  因此只能标为 provisional，不能升级为 `performance-validated`。

## 验收门

1. `nm -D --defined-only` 只暴露冻结四入口及 compiled-context 三入口；KUPL/CUDA
   计划结果逐位一致，内部 kernel 符号不得导出。
2. 固定小方程和四条 distillation 方程分别与 NumPy/opt_einsum 对照；complex128
   相对误差 `<= 1e-12`，同时检查 shape、dtype、NaN/Inf 和 `out` 身份。
3. 非连续输入、`size-1` 广播、隐式输出、trace/Hadamard、多步中间结果和空归约
   逐类覆盖；不支持项必须走显式回退。
4. CUDA 同 plan 在 `sm_60` 与 `sm_70` 实机执行；检查 device synchronize、stream
   依赖、错误码与中间 buffer 释放。
5. KUPL 检查 OpenMP 线程数、SIMD 连续 K、MPI rank 对称性和工作区复用。
6. 性能 JSON 必须保存 oracle commit/version、设备 UUID、线程、stream、warmup、
   iterations、path hash、median/IQR、峰值内存和正确性摘要；所有 3x 项 PASS 才可
   声明 `performance-validated`。

## 常见错误

- 仅把 NumPy/CuPy `einsum` 包一层就宣称原生加速。
- 只优化规划，不接通 `applyContractQcd`，公共 `contract` 仍走旧执行层。
- 将 K 放在高维或依赖隐式跨步，导致 SIMD lane 和 CUDA warp 访存离散。
- 用逐元素 CUDA 正确性替代 cuBLAS batched GEMM 正确性。
- benchmark 固定执行顺序、无预热、无同步、混用 `optimal/dp` 或低倍率仍返回 0。
- 把 MPI/多 stream 的吞吐加速冒充算法单流加速；两者必须分别报告。

## 交接

- 物理方程与 LapH 轴序：`pyqcd-distillation`、`pyqcd-physics-correlator`。
- backend、设备、MPI、显存和计时基础：`pyqcd-infra`。
- CUDA/KUPL 构建参考经验：PyQCU 原理吸收后自包含实现，不在运行时引用。
- 报告或 PDF 验收：`pyqcd-docs`。
