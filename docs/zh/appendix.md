# 附录：对比测试工作流与性能说明

> **范围。** 本附录属于源码仓库内容。PyPI 包不包含 MatX/Einsums 桥、
> 基准驱动，也不包含这里讨论的原始日志。

## 接入一个新的对比引擎

MatX 与 Einsums 的接入遵循同一套工作流，可直接复用于下一个引擎。

1. 将上游源码固定为 `refer/git-rep/<name>` 下的 Git submodule，并在
   `refer/git-rep/README.md` 记录来源、用途与固定 commit。
2. 在 `cpp/<device>/<name>/` 下新增 C ABI 桥：一个带状态码与平坦
   `int64`/`int32` 槽位数组的头文件、一个实现、一个带导出符号版本脚本的
   构建脚本，以及目录 `AGENTS.md`。
3. 构建必须失败关闭：缺少上游时产出桩库，合法调用返回 `UNAVAILABLE(4)`
   并给出可操作原因，绝不伪造成功。
4. 在 `pyqcd/testing/<name>/` 下新增 ctypes 绑定，并从官方文档提炼契约语料；
   所有未经实测的结论都要显式标注。
5. 在 `pyqcd/testing/__init__.py` 与 `pyqcd/testing/conftest.py` 注册验证入口。
6. 将引擎接入 `pyqcd/testing/einsum/bench_four_engines.py`，登记支持矩阵、
   精度标签与 `N/A` 原因。
7. 测试代码不得进入发行物：在 `find_packages` 中排除，并在 `MANIFEST.in`
   中 prune 其源码。
8. 验收通过才算完成：桩构建；依赖具备时的真机构建；对独立 oracle 的正确性；
   同线程、同预热的性能；`form-audit`；工作树干净。

## 四引擎性能摘要

测量环境为 i7-13620H（8 线程）与 RTX 4060 Laptop（8 GB）。每例 3 轮交替，
先验证正确性再计时；桥使用预分配输出；CPU 线程固定为 `OMP=8`、`BLAS=8`。
`opt_einsum` 运行在 NumPy/CuPy 之上，因此其数字表示“规划器 + 后端”。

### CPU（NumPy 数组，毫秒）

| 用例 | qcd | opt | einsums |
|---|---:|---:|---:|
| matmul 2D (real) | 718.7 | 713.0 | 944.8 |
| matmul 2D (complex) | 386.1 | 339.4 | 453.5 |
| element-wise | 309.0 | 351.7 | 1547.6 |
| outer product | 63.8 | 102.4 | 294.8 |
| full reduction | 58.0 | 63.3 | N/A |
| partial reduction | 114.7 | 121.2 | N/A |
| chain 3 ops | 597.5 | 589.7 | N/A |
| chain 4 ops | 282.1 | 275.5 | N/A |
| hyper-edge | 91.2 | 99.2 | N/A |
| batched GEMM | 374.3 | 3802.7 | 401.0 |
| ellipsis | 658.1 | 6510.7 | N/A |
| batch × matrix | 49.1 | 49.5 | N/A |
| many small batches | 105.9 | 221.1 | 179.1 |
| nucleon 2pt (kernel) | 2487.4 | 43142.1 | N/A |
| gamma sandwich (kernel) | 245.8 | 5681.2 | N/A |

### GPU（CuPy 数组，毫秒）

| 用例 | qcd | opt | matx |
|---|---:|---:|---:|
| matmul 2D (real) | 654.3 | 649.8 | 649.9 |
| matmul 2D (complex) | 326.8 | 325.0 | 327.0 |
| element-wise | 15.4 | 15.5 | 195.5 |
| outer product | 2.6 | 2.7 | 31.8 |
| full reduction | 9.5 | 20.1 | 282.2 |
| partial reduction | 8.8 | 8.9 | 17.9 |
| chain 3 ops | 575.0 | 584.7 | N/A |
| chain 4 ops | 252.4 | 251.4 | N/A |
| hyper-edge | 98.3 | 102.0 | N/A |
| batched GEMM | 169.7 | 176.8 | 163.4 |
| ellipsis | 334.0 | 324.3 | 324.2 |
| batch × matrix | 88.6 | 93.7 | N/A |
| many small batches | 25.0 | 25.1 | 23.7 |
| nucleon 2pt (kernel) | 875.7 | 2340.8 | N/A |
| gamma sandwich (kernel) | 796.4 | 1096.1 | N/A |

`matx` 行为 32 位计算。`N/A` 表示桥无法表达该用例：MatX 支持 1–2 个操作数、
rank 不超过 4；Einsums 只支持两个同 rank 操作数、rank 1–4、输出 rank 至少 1，
且不支持省略号。

### 时间花在哪里（Einsums）

| 用例 | 纯计算 (ms) | 张量分配 (ms) | 计算占比 |
|---|---:|---:|---:|
| GEMM 4096 | 578.9 | 134.2 | 81.2% |
| element-wise 12000 | 75.4 | 1103.7 | 6.4% |
| outer 8000 | 70.0 | 173.3 | 28.8% |
| batched 16×1024 | 173.5 | 136.4 | 56.0% |

element-wise 的端到端 1.55 s 中只有 6.4% 是计算，其余是逐次张量分配与拷贝。
Einsums 内核本身约 75 ms，比同用例的 qcd 与 opt_einsum 都快。

## 局限性

* `opt_einsum` 不是执行引擎：CPU 上测的是 NumPy/BLAS，GPU 上测的是
  CuPy/cuBLAS。
* MatX `einsum` 把 cuTensorNet 的 compute type 固定为 32F；MatX `matmul`
  才是真 FP64（256² 用例 `rel = 1.9e-16`）。
* 数据来自单机、每个家族单一尺寸、每例 3 轮 best-of-inner；选定用例的
  跨轮波动为 10–26%。
* 桥与引擎的成本边界不同：qcd/opt 在调用内分配输出；桥使用预分配输出，
  Einsums 桥还会额外分配输入张量。
* 未覆盖多卡/MPI、多 stream、非连续输入、混合精度、complex64/float32
  与生产尺度格点。
* CPU 约 15%、GPU 约 25% 以下的差异落在观测波动内，不应据此排名。

## 展望：在更多计算环境对比

* 在服务器 CPU（NUMA、更多核）与数据中心 GPU（HBM）上复测。
* 增加 MPI/多 stream 与分布式 batch 分片。
* 对齐精度：提供 MatX 64F 路径，或让所有引擎统一跑 32F。
* 做尺寸扫描，定位 cuTensorNet 开销被摊薄的 crossover。
* 测量冷启动、峰值内存、尾延迟与能耗，而不只是中位耗时。
* 对每个引擎做线程扫描，同时报告固定档与最佳档。
* 在第二台机器复核后再推广任何小倍率结论。

完整报告、原始 JSON 与组件探针位于源码仓库：
`docs/report_matx_einsums_bridge_20261110.md`、
`logs/einsum-four-engine-20261110.txt`、
`cpp/cpu/cpu-einsums/tests/bench_components.cpp`。
