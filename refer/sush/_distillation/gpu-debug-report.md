# 五夸克到三夸克三点函数一致性报告

日期：2026-11-17（Asia/Shanghai）  
节点：`gpu024`，NVIDIA A100-PCIE-40GB，40 GiB，driver 550.54.15  
环境：`/public/home/sush/envs/C12P12`，CuPy 14.1.1，CUDA 12.6.1  
作业：`4017455`，单任务、单 GPU，complex128，`Nev_src=100`、`Nev_cur=400`。

## 测试对象

`test.py` 现在使用项目文档中“**五夸克到三夸克三点函数**”的同一物理定义：

- sink：`['|', 'u^d', 'gamma_M', 'u', '|', '|', 'u', 'u', 'C * gamma_B', 'u', '|']`；
- source：`['|', 'u^d', 'C * gamma_BH', 'u^d', 'u^d', '|']`；
- current：`['|', 'u^d', 'gamma_J1', 'u', '|']`；
- `Cpt='3pt'`，`Vindex=['', '', '', '']`，`Gindex=['', '', '', '']`；
- 输出方程为 `->es`，四个 V 区域按动态实现的时间区域编号注册：
  `VDV_0@tsink`、`VVV_0@tsink`、`VDV_0@tcur0`、`VVV_0@tsrc`。

dynamic 计划包含 120 个 nonzero terms。每个 term 的系数、Wick 方程、perambulator、gamma 和 V 张量都从同一计划解析，然后分别交给 dynamic、`opt_einsum.contract`、`pyqcd.einsum.contract`；因此三种结果计算的是同一组 120 张图。

## 结果

GPU 作业 `4017455` 完成，退出码 `0:0`。一次辅助计时如下，主要验收标准是图数量、特化命中和数值一致性：

| 项目 | 时间 |
|---|---:|
| `lqcddb_new.dynamic` | 3.720881 s |
| `lqcddb_update.dynamic` | 2.065072 s |
| `opt_einsum.contract` | 2.060477 s |
| `pyqcd.einsum.contract(specialize=False)` | 1.119381 s |
| `pyqcd.einsum.contract(specialize=True)` | 1.116918 s |

`pyqcd.specialization_matches=120/120`。四种结果的输出形状均为 `(4, 4)`；相对 `opt_einsum` 的最大误差为：

- `lqcddb_new.dynamic`：$5.642\times10^{-15}$；
- `lqcddb_update.dynamic`：$0$；
- pyqcd `specialize=False`：$5.642\times10^{-15}$；
- pyqcd `specialize=True`：$5.642\times10^{-15}$。

因此这次三点函数的 direct pyqcd 路径约为 opt_einsum 的 `0.54x`，数值完全一致。`specialize=True` 与 `False` 的差异小于 1%，不能把这次差异称为 specialize 开关带来的独立收益。

完整输出：

- [gpu-debug-4017455.log](/public/home/sush/distillation/test/gpu-debug-4017455.log)
- [gpu-debug-4017455.err](/public/home/sush/distillation/test/gpu-debug-4017455.err)

## 原因解释

前面的两点函数测试和这次三点函数测试不是同一个 topology。两点函数解析出的 12 条 direct 方程没有命中特化注册表，所以不能用它判断用户给出的三点 topology 是否加速。

用户给出的第一条三点方程

```text
cadb,egfh,mknl,oqpr,isjt,ac,mo,gi,kq,bd,fnp,hj,lrt->es
```

在 pyqcd 注册表中对应 `topology:67e8d341`，`matched=True`。单独复现这条方程时，A100 上 `opt_einsum.contract_expression(optimize='greedy')` 约为 41.81 ms，pyqcd `specialize=False` 约为 3.81 ms，说明 pyqcd 的 generic CUDA/CuPy 执行路线本身可以带来明显加速。这个加速不等同于 native topology kernel 已成功执行。

A100 的 `pyqcd --gpu-probe` 仍报告 native 2×2 probe 与 NumPy 不一致，实际 topology warning 也报告 native kernel fallback。因此本次 120 图测试中可以确认：

1. direct 方程已经命中特化注册表；
2. native 专用 CUDA kernel 仍未通过自检；
3. `specialize=False` 仍然明显更快，是 pyqcd 的 generic CUDA/CuPy 路径相对于 opt_einsum 路径的收益；
4. `specialize=True` 在 native fallback 后回到 generic 路径，所以与 `False` 基本相同。

`lqcddb_new.dynamic` 还包含 StructuralEinsum 匹配、轴变换和 native 失败回退，因此比 `lqcddb_update.dynamic` 慢；`lqcddb_update.dynamic` 的时间与 direct opt_einsum 接近。

本轮没有安装、重编译或覆盖 pyqcd/CUDA 环境；GPU 脚本只在作业进程中加载现有 C12P12 环境和设置变量。
