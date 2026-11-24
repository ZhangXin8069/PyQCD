# L32x96 quark PDF：PyQCD contraction 实测报告

更新：2026-11-09T19:39:24+08:00

## 结论

本次真实输入测试数值通过，但仅替换 contraction 引擎没有观测到有意义的加速。
保持原来的生产代码和共享 conda 环境不变；PyQCD 仅安装在本目录隔离 venv 中。

## L32x96 生产任务状态

GPU 补算数组 3931190：72 个子任务均已 COMPLETED，退出码 0:0。
CPU 最终检查 3931191：已 COMPLETED，退出码 0:0。

新增 30 conf 的检查：52920 / 52920 cases，210 个 conf-flow 任务。

合并完成回执：`COMPLETE_PREVIOUS20_PLUS_EXPANSION30`，总计 `50` conf。
最终检查对新增 30 conf 逐结果校验，对原 20 conf 核对既有完成汇总的身份；不是重新运行所有原始计算。
完成回执：`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x96_B_resume_20261105/status/complete.json`。

## 安装环境及安装限制

官方 PyPI 分发包：pyqcd 0.2.1；生产 Python 3.10.18、CuPy 14.1.1、opt_einsum 3.4.0。
NumPy 2.2.6、SciPy 1.15.3 沿用生产环境；隔离安装了 h5py 3.14.0 兼容二进制包。
初次 venv 安装成功，但旧 Bash 在没有 OpenBLAS 开发文件时触发空数组错误，未编出原生库。
第二次 venv_native 安装加载 OpenBLAS 模块，原样使用官方源码，显式编译 sm80 CUDA。
CUDA 库及 Cython 接口均已编出；安装日志仍警告缺少 qcd_cuda_hadamard provider。
CPU KUPL 库另有 `undefined symbol: ompi_mpi_byte` 动态链接问题；CPU 数值测试走了安全 fallback。
因此 CPU 接口数值通过不能写成“CPU 原生加速验证通过”。这不影响已经测量的 CUDA 数值一致性。
详见 metadata/installed_environment_native.json、logs/native_install_3960815.err 和 GPU JSON 的 warnings。

## 数值验证和性能

测试状态：**PASS**。

CUDA diagnostic probe: CUDA fast path is active on NVIDIA A800-SXM4-80GB.

CPU 接口校验（含 fallback）：PASS；最大相对 L2 差异 0.000e+00。

GPU 小尺寸校验：PASS；最大相对 L2 差异 4.410e-16。

| Stage | opt_einsum warm median (s) | PyQCD warm median (s) | Speedup old/new | Max relative L2 |
| --- | ---: | ---: | ---: | ---: |
| real_C2_reduced | 0.0698323 | 0.0691174 | 1.01 | 4.275e-16 |
| real_H_one_sigma_b | 0.119889 | 0.120574 | 0.9943 | 0.000e+00 |
| real_C3_six_Wick_one_sigma_b | 0.32963 | 0.329625 | 1 | 1.421e-14 |
| real_full_sigma_sum_C3_one_Pz_all_z | 5.8699 | 5.87269 | 0.9995 | 4.554e-15 |

完整 C3 与已有生产文件的比较：原引擎相对差异 0.000e+00，
PyQCD 相对差异 4.554e-15；验收阈值 2e-10。

上述表格来自每个引擎一次额外 warmup 后的五次同步计时，交错运行顺序，取 wall-time 中位数。
冷启动（含本轮表达式规划/首次执行）及每个表达式的规划时间另外保存在 JSON 内。

## 实际输入及范围

conf=13100，Pz=4，tsep=7，dt=3，t_current=0。
tau_L32x96=0.267923912591，匹配的 tau_L32x64=0.2。
nev_source=200、nev_current=400，全部 21 个 z、正负 Wilson line、四个 baryon projectors、
六个 Wick 项和完整的所缓存原始时间双求和。权重仍为 B 方案，恰好施加一次。
输入读取、B-weighted vertex 及 VVV 重建共 204.278 秒；
这部分未替换引擎，单列而不算入 contraction 加速倍率。
真实参考结果：`/public/group/lqcd/donghx/Gradient_Flow_GluonPDF/quark_l32x96_B_expand50_matched_radius_20261028/C3_C2_schemeB/thin_tf0p26792391_eps0p00669810_tw6_Nv400_dz21/13100/tJ000/tsep07/dt03/correlators.npz`。

## 为什么当前没有明显加速

当前生产代码已经缓存 contract_expression，并做了外部旋量降维、H 时间求和和 Wick 分块；
本次只替换缓存表达式的实现，不改变这些算法。不能假定文档中其他 contraction 的专用核收益直接适用。
本次 21 个实际表达式与安装包的专用方程登记表静态精确比较：
命中数 0；且 ContractExpression 入口绕过 contract() 的专用方程分派。
GPU probe 的核子 2pt 专用核可用，不等于我们的降维 C2/C3 方程调用了同一个专用核。
静态检查不是运行时调用轨迹；本报告不把所有实际表达式武断归类为原生或 fallback。

设备内存每 50 ms 采样，包含不经 CuPy 分配的 native 工作空间。两引擎处于同一进程，
后测引擎的采样包含已保留的缓存/上下文，不能把 JSON 中的绝对内存峰值直接解释成独立内存倍率。
整个完整 C3 计时与既有生产输出一致，但并非完整四 GPU、九动量生产作业的总 wall-time 对照。
数值等价不独立证明 B 权重的统计处方、有限时间窗的收敛或 transversity 的外部物理符号约定。

## 代码及复现

`benchmark.py`：同一 Contractor 缓存契约，只将 `import opt_einsum` 替换为
`import pyqcd.einsum as opt_einsum`。生产目录任何源码/结果均未修改。
`gpu.slurm`：na800-sug、donghx_223、1 GPU、4 CPU；已完成任务 3960818。
`install_native.slurm`：已完成 CPU 安装任务 3960815。
`results/gpu_result.json`：数值差异、分阶段重复计时、dtype/shape/stride、规划路径和警告。

```bash
source /public/home/donghx/load_a100_PyQuad.sh
module load openblas/0.3.20-gcc-10.3.0
source /public/group/lqcd/donghx/Gradient_Flow_GluonPDF/pyqcd_einsum_benchmark_L32x96_20261109/venv_native/bin/activate
# 科学计算只能在已申请的计算节点执行，不要直接在登录节点运行 benchmark.py。
```

建议保留原生产引擎。若继续追求提速，应针对实际降维方程做分阶段热点分析或专用内核适配，
并再次通过六个 Wick 项和生产文件回归，不应仅凭其他案例的加速倍率全量替换。

## Slurm snapshot

```text
JobID|JobName|Partition|State|ExitCode|Elapsed|NodeList
3931190_0|q96_B50_resume|na800-sug|COMPLETED|0:0|02:44:35|gpu043
3931190_1|q96_B50_resume|na800-sug|COMPLETED|0:0|03:32:24|gpu043
3931190_2|q96_B50_resume|na800-sug|COMPLETED|0:0|04:31:04|gpu044
3931190_3|q96_B50_resume|na800-sug|COMPLETED|0:0|03:40:07|gpu044
3931190_4|q96_B50_resume|na800-sug|COMPLETED|0:0|04:12:49|gpu043
3931190_5|q96_B50_resume|na800-sug|COMPLETED|0:0|04:19:28|gpu043
3931190_6|q96_B50_resume|na800-sug|COMPLETED|0:0|05:34:39|gpu044
3931190_7|q96_B50_resume|na800-sug|COMPLETED|0:0|06:21:15|gpu044
3931190_8|q96_B50_resume|na800-sug|COMPLETED|0:0|06:40:32|gpu043
3931190_9|q96_B50_resume|na800-sug|COMPLETED|0:0|03:20:26|gpu043
3931190_10|q96_B50_resume|na800-sug|COMPLETED|0:0|03:31:17|gpu044
3931190_11|q96_B50_resume|na800-sug|COMPLETED|0:0|04:11:26|gpu044
3931190_12|q96_B50_resume|na800-sug|COMPLETED|0:0|04:23:19|gpu043
3931190_13|q96_B50_resume|na800-sug|COMPLETED|0:0|05:32:26|gpu044
3931190_14|q96_B50_resume|na800-sug|COMPLETED|0:0|06:27:15|gpu043
3931190_15|q96_B50_resume|na800-sug|COMPLETED|0:0|06:47:20|gpu044
3931190_16|q96_B50_resume|na800-sug|COMPLETED|0:0|03:29:31|gpu043
3931190_17|q96_B50_resume|na800-sug|COMPLETED|0:0|03:46:44|gpu044
3931190_18|q96_B50_resume|na800-sug|COMPLETED|0:0|04:09:04|gpu043
3931190_19|q96_B50_resume|na800-sug|COMPLETED|0:0|04:23:12|gpu043
3931190_20|q96_B50_resume|na800-sug|COMPLETED|0:0|05:39:15|gpu044
3931190_21|q96_B50_resume|na800-sug|COMPLETED|0:0|06:17:31|gpu044
3931190_22|q96_B50_resume|na800-sug|COMPLETED|0:0|06:36:56|gpu043
3931190_23|q96_B50_resume|na800-sug|COMPLETED|0:0|03:35:17|gpu043
3931190_24|q96_B50_resume|na800-sug|COMPLETED|0:0|03:35:05|gpu044
3931190_25|q96_B50_resume|na800-sug|COMPLETED|0:0|04:11:59|gpu043
3931190_26|q96_B50_resume|na800-sug|COMPLETED|0:0|04:20:47|gpu044
3931190_27|q96_B50_resume|na800-sug|COMPLETED|0:0|05:30:18|gpu043
3931190_28|q96_B50_resume|na800-sug|COMPLETED|0:0|06:25:20|gpu044
3931190_29|q96_B50_resume|na800-sug|COMPLETED|0:0|06:45:34|gpu043
3931190_30|q96_B50_resume|na800-sug|COMPLETED|0:0|03:24:43|gpu044
3931190_31|q96_B50_resume|na800-sug|COMPLETED|0:0|03:42:56|gpu043
3931190_32|q96_B50_resume|na800-sug|COMPLETED|0:0|04:10:13|gpu044
3931190_33|q96_B50_resume|na800-sug|COMPLETED|0:0|04:25:05|gpu044
3931190_34|q96_B50_resume|na800-sug|COMPLETED|0:0|05:37:17|gpu043
3931190_35|q96_B50_resume|na800-sug|COMPLETED|0:0|06:15:23|gpu043
3931190_36|q96_B50_resume|na800-sug|COMPLETED|0:0|06:35:40|gpu044
3931190_37|q96_B50_resume|na800-sug|COMPLETED|0:0|03:23:56|gpu044
3931190_38|q96_B50_resume|na800-sug|COMPLETED|0:0|03:33:21|gpu043
3931190_39|q96_B50_resume|na800-sug|COMPLETED|0:0|04:06:30|gpu043
3931190_40|q96_B50_resume|na800-sug|COMPLETED|0:0|04:25:54|gpu044
3931190_41|q96_B50_resume|na800-sug|COMPLETED|0:0|05:37:41|gpu044
3931190_42|q96_B50_resume|na800-sug|COMPLETED|0:0|06:20:16|gpu043
3931190_43|q96_B50_resume|na800-sug|COMPLETED|0:0|06:34:28|gpu043
3931190_44|q96_B50_resume|na800-sug|COMPLETED|0:0|03:28:02|gpu044
3931190_45|q96_B50_resume|na800-sug|COMPLETED|0:0|03:28:58|gpu044
3931190_46|q96_B50_resume|na800-sug|COMPLETED|0:0|04:11:41|gpu044
3931190_47|q96_B50_resume|na800-sug|COMPLETED|0:0|04:25:48|gpu043
3931190_48|q96_B50_resume|na800-sug|COMPLETED|0:0|05:38:55|gpu043
3931190_49|q96_B50_resume|na800-sug|COMPLETED|0:0|06:16:28|gpu044
3931190_50|q96_B50_resume|na800-sug|COMPLETED|0:0|06:39:52|gpu044
3931190_51|q96_B50_resume|na800-sug|COMPLETED|0:0|03:26:33|gpu043
3931190_52|q96_B50_resume|na800-sug|COMPLETED|0:0|03:28:23|gpu043
3931190_53|q96_B50_resume|na800-sug|COMPLETED|0:0|04:10:10|gpu044
3931190_54|q96_B50_resume|na800-sug|COMPLETED|0:0|04:23:07|gpu043
3931190_55|q96_B50_resume|na800-sug|COMPLETED|0:0|05:38:24|gpu044
3931190_56|q96_B50_resume|na800-sug|COMPLETED|0:0|06:18:30|gpu043
3931190_57|q96_B50_resume|na800-sug|COMPLETED|0:0|06:38:58|gpu044
3931190_58|q96_B50_resume|na800-sug|COMPLETED|0:0|03:25:51|gpu043
3931190_59|q96_B50_resume|na800-sug|COMPLETED|0:0|03:41:50|gpu044
3931190_60|q96_B50_resume|na800-sug|COMPLETED|0:0|04:12:23|gpu043
3931190_61|q96_B50_resume|na800-sug|COMPLETED|0:0|04:19:29|gpu043
3931190_62|q96_B50_resume|na800-sug|COMPLETED|0:0|05:34:19|gpu044
3931190_63|q96_B50_resume|na800-sug|COMPLETED|0:0|06:22:38|gpu044
3931190_64|q96_B50_resume|na800-sug|COMPLETED|0:0|06:41:12|gpu043
3931190_65|q96_B50_resume|na800-sug|COMPLETED|0:0|03:27:42|gpu043
3931190_66|q96_B50_resume|na800-sug|COMPLETED|0:0|03:31:19|gpu044
3931190_67|q96_B50_resume|na800-sug|COMPLETED|0:0|04:09:02|gpu043
3931190_68|q96_B50_resume|na800-sug|COMPLETED|0:0|04:24:46|gpu044
3931190_69|q96_B50_resume|na800-sug|COMPLETED|0:0|05:35:02|gpu043
3931190_70|q96_B50_resume|na800-sug|COMPLETED|0:0|06:16:08|gpu044
3931190_71|q96_B50_resume|na800-sug|COMPLETED|0:0|06:31:35|gpu043
3931191|q96_resume_final|i72c512g|COMPLETED|0:0|01:21:57|cpu052
3960805|q96_pyqcd_install|i72c512g|COMPLETED|0:0|00:00:32|cpu053
3960809|q96_pyqcd_cpu|i72c512g|FAILED|1:0|00:00:03|cpu053
3960815|q96_pyqcd_native|i72c512g|COMPLETED|0:0|00:01:53|cpu053
3960818|q96_pyqcd_bench|na800-sug|COMPLETED|0:0|00:05:12|gpu043
```
