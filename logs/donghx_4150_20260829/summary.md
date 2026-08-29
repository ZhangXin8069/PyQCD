# donghx / PyQCD 组态 4150 对照证据

本目录记录本轮可提交的轻量证据索引；原始规范场、本征矢量、perambulator、HYP
场和参考产物均只读于用户明确给出的 `/public/group/lqcd/...` 路径，未复制到仓库。
大型 VVV/2pt 数组保留在本机生成目录或 `data/cmp1_4150/`，不进入 Git。

## 资产盘点

盘点命令：

```text
python examples/pyqcd/cmp1/manifest_4150.py --output examples/pyqcd/cmp1/v202608291424/manifest.json
python examples/pyqcd/cmp1/verify_manifest_4150.py
```

| 资产 | 当前事实 |
|---|---|
| Clover 规范场 | 存在，573439152 bytes |
| eigenvectors/4150 | 存在，72 个文件，4777574400 bytes |
| light perambulators/4150 | 存在，288 个文件，13271040000 bytes |
| HYP 3D 1/3/5 次、4D 10 次 | 均存在，各 7 个记录文件，约 573 MB；4D10 已完成三方向 OPE 对照 |
| 2pt_Result 系综目录 | 存在，322 个文件；含 `momsmear0_Cg5` 与 `momsmear0_Cg5g4` |
| TMD OPE x/y/z 结果 | 存在，58 个文件；含横向位移参数，未与本轮直线 OPE 混合比较 |
| `Result_hpy_4D_10times` | 当前路径不存在 |
| `Peram_code_2505`、`Peram_mpi`、`Peram_result` | 当前用户给定路径不存在 |

## 低层真实对照

结果：`examples/pyqcd/cmp1/v202608291500_lowlevel/results.json`。

| 对象 | 形状 | 相对差 | 状态 |
|---|---|---:|---|
| eigvec | `(8, 13824, 3)` | 0 | pass |
| phase | `(13824,)` | 0 | pass |
| VdV | `(1, 4, 4)` | 1.31e-15 | pass |
| VVV | `(4, 4, 4)` | 4.75e-16 | pass |
| Clover | `(4, 4, 2, 24, 24, 24, 3, 3)` | 2.45e-16 | pass |
| dual | `(4, 4, 2, 24, 24, 24, 3, 3)` | 1.89e-16 | pass |
| OPE | `(2, 2)` | 1.36e-15 | pass |

比较基准是同一 4150 规范场上独立重写的 donghx 参考公式；这七项不等同于
“外部逐时间 VVV 文件”对照。

## HYP 规范场与非极化 OPE 真实对照

运行入口：

```text
python examples/pyqcd/cmp1/run_4150_hyp_ope.py --smear 4d10 --directions x,y,z --outdir examples/pyqcd/cmp1/v20260829_hyp_ope_xyz
python examples/pyqcd/cmp1/run_4150_hyp_ope.py --smear 3d1 --directions z --outdir examples/pyqcd/cmp1/v20260829_hyp_ope_3d1_z
python examples/pyqcd/cmp1/run_4150_hyp_ope.py --smear 3d3 --directions z --outdir examples/pyqcd/cmp1/v20260829_hyp_ope_3d3_z
python examples/pyqcd/cmp1/run_4150_hyp_ope.py --smear 3d5 --directions z --outdir examples/pyqcd/cmp1/v20260829_hyp_ope_3d5_z
```

输入是各 `.lime.contents/msg02.rec04.ildg-binary-data`，读取后为
`(Nt,Nz,Ny,Nx,4,3,3)=(72,24,24,24,4,3,3)` 的 `complex128` 规范场；
`4d10` 的最大链接幺正性偏差为 `1.11e-15`。比较调用
`gluon_ope_operator_z0(..., second_insert="F")`，即 donghx 非极化
`F\,W\,F\,W` 通道；参考 TMD 成品只取横向位移为零的直线截面，不能据此宣称
非零横向 staple 已验证。

### 4D HYP 10 次：三方向 9 个通道

结果：`examples/pyqcd/cmp1/v20260829_hyp_ope_xyz/results.json`，摘要：
`examples/pyqcd/cmp1/v20260829_hyp_ope_xyz/summary.md`。

| 方向 | 通道数 | 输出形状 | 最大相对 L2 | 最大绝对差 | 状态 |
|---|---:|---|---:|---:|---|
| x | 3 | `(72,9)` | `3.30e-15` | `5.69e-13` | pass |
| y | 3 | `(72,9)` | `3.31e-15` | `5.69e-13` | pass |
| z | 3 | `(72,12)` | `7.50e-17` | `5.69e-14` | pass |

9/9 通道均为 `pass`，整体 runner 退出码为 0。

### 3D HYP 1/3/5 次：真实运行但无可配对参考成品

三次 z 向运行均成功构造并计算 3 个通道，输出形状均为 `(72,12)`，规范场均为
`complex128` 且最大幺正性偏差为 `1.11e-15`；由于用户给出的参考 OPE 成品只属于
`4d10`，三个结果按设计记为 `unverified`，runner 退出码均为 2。结果目录分别为：

```text
examples/pyqcd/cmp1/v20260829_hyp_ope_3d1_z/
examples/pyqcd/cmp1/v20260829_hyp_ope_3d3_z/
examples/pyqcd/cmp1/v20260829_hyp_ope_3d5_z/
```

## 费米子 2pt 真实对照

主结果：`examples/pyqcd/cmp1/v202608291710_fermion_cg5g4/`；
`Cg5` 结果：`examples/pyqcd/cmp1/v202608291705_fermion_cg5/`。
主结果为 `Nev=100`、`t_source=0`、`Delta t=2..36` 的 35 个时间对；其余动量
配置取 `Delta t=2`，且均复用经过形状/dtype 校验的本地 VVV 缓存。

| 通道/动量 | contract 相对差 | nopol 相对差 | 参考精度 |
|---|---:|---:|---|
| `Cg5g4`, `(Pz,Py,Px)=(0,0,0)` | 3.50e-15 | 1.27e-15 | complex128 |
| `Cg5`, `(0,0,0)` | 2.29e-15 | 1.53e-15 | complex128 |
| `Cg5g4`, `Pz=1`（单时间对） | 2.53e-15 | 8.56e-16 | complex128 |
| `Cg5g4`, `Pz=2`（单时间对） | 2.31e-15 | 8.26e-16 | complex128 |
| `Cg5g4`, `Pz=5`（单时间对） | 1.66e-15 | 1.42e-16 | complex128 |
| `Cg5g4`, `Px=1`（单时间对） | 2.64e-06 | 1.42e-06 | complex64，tol=1e-5 |
| `Cg5g4`, `Py=1`（单时间对） | 2.59e-06 | 5.94e-07 | complex64，tol=1e-5 |

每个 runner 的第三项 `vvv` 均为 `unverified(reference_output_missing)`：用户给定
参考结果目录没有可直接读取的逐时间 `VVV.t*.Px*Py*Pz*` 中间文件。因此整体状态
必须是 `unverified`，但已有 contract/nopol 数值比较仍为 `pass`。

### 无 momentum smear 的方向/大小矩阵

在同一组态、同一标准 light perambulator 上，按 `P=(Pz,Py,Px)` 分别覆盖
`Pz=0..5`、`Py=1..5`、`Px=1..5`。每个非零动量取 `Delta t=2`；`Cg5g4` 的
`P=0` 另取 `Delta t=2..36` 的 35 对。结果由各版本目录的 `results.json` 汇总：

| 变体 | 动量配置数 | 选定时间对 | contract/nopol | 参考 dtype | 最大 contract 相对差 | 最大 nopol 相对差 |
|---|---:|---:|---|---|---:|---:|
| `Cg5g4` | 16 | 50 | 16/16 pass | c128 12，c64 4 | 2.6388e-6 | 1.5445e-6 |
| `Cg5` | 16 | 16 | 16/16 pass | c128 16 | 3.0761e-15 | 1.7189e-15 |

其中每个 `contract` 输出形状为 `(72,72,4,4)`，每个 `nopol_pp` 输出形状为
`(72,72)`；`Cg5g4` 的 c64 参考采用 `tol=1e-5`，最大误差仍在容差内。上述
矩阵只证明无 momentum smear 的 2pt 中间/投影对象；`momsmear±2{x,y,z}` 仍需
独立 smeared perambulator，不能由标准 perambulator 推断。

## 本轮断言

```text
verify_manifest_4150: PASS 4/4
verify_4150_lowlevel: PASS 3/3
verify_4150_fermion: PASS 3/3
verify_4150_fermion_runner: PASS 12/12
verify_4150_hyp_ope_runner: PASS 3/3
```

完整主回归和报告编译结果见最终交付说明；本文件只记录可追溯数值，不把未运行的
3pt、ratio、barematrix 或完整 TMD 链写成通过。

## 最终回归（2026-08-29）

```text
python examples/pyqcd/conftest.py: 42 passed, 0 failed
verify_manifest_4150.py: PASS 4/4
verify_4150_lowlevel.py: PASS 3/3
verify_4150_fermion.py: PASS 3/3
verify_4150_fermion_runner.py: PASS 12/12
verify_4150_hyp_ope_runner.py: PASS 3/3
```
