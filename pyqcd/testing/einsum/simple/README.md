# pyqcd.einsum simple 实测接口

一条命令即可自查，不需要读基准框架源码。

## 快速开始

```bash
cd /home/zhangxin/PyQCD

# 1) API 兼容面（对照 opt_einsum 3.4）
python -m pyqcd.testing.einsum.simple.check_compat

# 2) 显存/内存预算（只做算术，不分配大张量）
python -m pyqcd.testing.einsum.simple.check_memory

# 3) 7 条蒸馏方程数值正确性（默认 T0n：Nvec=6，秒级）
python -m pyqcd.testing.einsum.simple.check_equations

# 4) 原生后端逐项检查
PYQCD_EINSUM_NATIVE=kupl python -m pyqcd.testing.einsum.simple.check_native --backend kupl
PYQCD_EINSUM_NATIVE=cuda python -m pyqcd.testing.einsum.simple.check_native --backend cuda

# 5) 性能对照（自己指定 profile 与后端）
python -m pyqcd.testing.einsum.simple.bench --profile T1  --backend cpu
python -m pyqcd.testing.einsum.simple.bench --profile L24 --backend cuda --iters 10

# 6) 全套
bash pyqcd/testing/einsum/simple/run-all.sh
```

## 方程组说明

| 组 | 物理含义 | 操作数 | 算力特征 |
|---|---|---|---|
| E1 | 核子 2pt（NN）：3×peram × 2×γ₇ × 2×VVV × 宇称投影 | 8 | opt_cost ≈ 1.92e10（Nvec=100） |
| E2 | 核子 3pt：current peram + 源端 peram×2 + γ₇×2 + VVV×2 + 投影 | 8 | ≈1.92e10 |
| E4 | peram × peram × γ × 规范场（含空间格点） | 4 | spatial=1 时毫秒级；L24 时 2.75e10 |
| E5 | γ 夹心 `γ₅·peram†·γ₅` | 3 | 带宽受限；L24 时 1.06e9 |
| S1 | sush NN 2pt（`refer/sush/_distillation/contraction.NN.2pt.py:119`） | 8 | ≈1.92e10 |
| S2 | sush 3pt 含 μν 插入（`contraction.block.numpy.32x96.compensation.copy.py:402`） | 9 | 6.7e7 |
| S3 | sush 2pt 含 μ 插入（同文件 :301） | 6 | 6.7e7 |

## profile

| 名称 | nvec | spatial | 说明 |
|---|---|---|---|
| `T0n` | 6 | 1 | 冒烟，秒级 |
| `T0` | 20 | 1 | CI 规模 |
| `T1` | 100 | 1 | 生产 Nvec |
| `T2` | 100 | 1（2 个 t_src） | 持续吞吐 |
| `L24` | 50 | 24 | 真实空间格点；prop 单算子 1.06 GiB |
| `L24n` | 24 | 24 | 显存保守档；prop 单算子 0.51 GiB |

`L24` 且 `Nvec=100` 时单算子 2.12 GiB、E4 峰值 >6 GiB，`check_memory` 会明确判为 UNSAFE。

## bench 读数注意

- **倍率不是唯一指标**：双方中位耗时都 <100 ms 的组标 `too_short`，不构成实战对照。
- opt_einsum 个别 path 不走 BLAS（例如 E2 在 T1 单次 2.18 s），倍率会虚高，必须同时看绝对耗时。
- 本机为单卡 RTX 4060 Laptop，**禁止**用多卡/多进程结果与单卡 opt_einsum 对比。

## 数据说明

`/public/group/lqcd` 与 `*.lime` 在本机不存在，因此操作数由固定种子合成
（形状与真实蒸馏一致），正确性以 opt_einsum / numpy 逐元素对照为准（`rel ≤ 1e-12`）。
