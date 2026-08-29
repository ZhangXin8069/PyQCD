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
| HYP 3D 1/3/5 次、4D 10 次 | 均存在，各 7 个记录文件，约 573 MB |
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

## 费米子 2pt 真实对照

主结果：`examples/pyqcd/cmp1/v202608291710_fermion_cg5g4/`；
`Cg5` 结果：`examples/pyqcd/cmp1/v202608291705_fermion_cg5/`。
两者均为 `Nev=100`、`t_source=0`、`Delta t=2..36` 的 35 个时间对，且复用
同一经过形状/dtype 校验的本地 VVV 缓存。

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

## 本轮断言

```text
verify_manifest_4150: PASS 4/4
verify_4150_lowlevel: PASS 3/3
verify_4150_fermion: PASS 3/3
verify_4150_fermion_runner: PASS 7/7
```

完整主回归和报告编译结果见最终交付说明；本文件只记录可追溯数值，不把未运行的
3pt、ratio、barematrix 或完整 TMD 链写成通过。

## 最终回归（2026-08-29）

```text
python examples/pyqcd/conftest.py: 41 passed, 0 failed
verify_manifest_4150.py: PASS 4/4
verify_4150_lowlevel.py: PASS 3/3
verify_4150_fermion.py: PASS 3/3
verify_4150_fermion_runner.py: PASS 7/7
```
