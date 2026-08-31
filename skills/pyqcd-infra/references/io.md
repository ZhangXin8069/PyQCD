# I/O reference — HDF5 / ASCII / VdV / VVV / Ω / environment

## 适用范围

任务涉及 PyQCD 管线产物、旧格式兼容、L.Liu ASCII 关联函数、预计算顶点积或运行环境
快照时读取。本文件定义数据交换和 Ω 收缩权重的元数据边界，不定义统计估计量或物理
重整化。

## 接口表

| 数据 | API | 纪律 |
|---|---|---|
| 管线张量 | `save_tensor_h5(arr, path, dataset)` / `load_tensor_h5(path)` | 保存 dtype、shape 和 dataset 名 |
| 旧产物 | `_load_any` | 读取优先 `.h5`，再回退 `.npy/.npz` |
| ASCII 关联函数 | `write_data_ascii(data, T, L, filename)` / `read_data_ascii(filename)` | `.gz` 自动压缩/读取；离散标签列在 Re/Im 前 |
| V†V | `readin_vdv_all(vdv_dir, nev, nev1, Nt, ..., byteorder="native")` | f8 pair、精确平方 Nev、按 nev1 截断 |
| VVV | `readin_vvv_all(..., byteorder="native")` / `readin_vvv(..., byteorder="native")` | f8 pair、精确立方 Nev、按 nev1 截断 |
| Ω 收缩权重 | `create_omega_accelerate(...)` | 记录分区/抽样参数；不得当作物理重整化因子 |
| 环境 | `pyqcd.tools._env.dump_env` | 每次管线运行保存 git/包/XeLaTeX/GPU/cmdline |

## 管线 HDF5 完成边界

`save_tensor_h5` 是底层序列化接口；需要断点续跑语义时使用
`pyqcd.pipeline._steps.save_array`，由管线在同目录临时文件写完后以原子替换发布最终
`.h5`。消费端必须打开 `data` 数据集并核对预期 shape、数值 dtype 和可读性；空文件、损坏
HDF5、缺少 `data`、空数组或 shape/dtype 不符都不是“已完成”。`_load_any` 优先 `.h5`，
因此损坏的首选 HDF5 不能被同名旧 `.npy` 静默掩盖。具体 2pt 完成门与 `recompute_2pt`
语义见 `pyqcd-pipeline` 的 runbook。

OPE strict cache 的单文件 loader 在同一个 HDF5 handle、同一份已加载数组上完成
shape/dtype/finite/payload SHA 校验；四文件逐一发布完成后再做一次 source identity 检查。
这些是 `pyqcd-pipeline` 的缓存与发布契约，本 reference 不重复定义命中编排；没有协作锁时，
该检查不能证明完整 ABA 或跨文件线性化，且只捕捉检查前仍可观察到的 source identity 变化。

## VdV/VVV 二进制契约

`byteorder` 只接受 `native`、`little`、`big`，分别对应 `=f8`、`<f8`、`>f8`；默认
`native` 保持旧 reader 的本机端序行为，不做自动探测。真实 huangcl 文件的端序目前没有
证据，因而不得把默认值表述为“已确认的文件端序”；端序选错也未必触发尺寸错误，必须用
已知数值或独立产物校验内容。

每个复数严格存为两个连续 float64 `[real, imag]`，即每个复数 16 字节。reader 的守卫为：

1. `Nt`、`nev`（接口含此参数时）和 `nev1` 必须是正整数；`Px/Py/Pz` 必须是有符号整数，
   负动量按原值进入文件名。上述参数中的布尔值都不算整数；`nev1` 不得超过显式或由文件
   推得的 Nev。
2. 文件字节数必须同时对齐单个 f8 和完整 f8 pair。除以 16 得到复数个数后，整块文件还
   必须可被 `Nt` 整除。
3. VdV 每时间片复数个数必须是某个正整数 Nev 的精确平方；整块 VVV 必须是精确立方。
   `readin_vdv_all` / `readin_vvv` 还要求该精确根等于显式 `nev`，等价于文件大小分别严格为
   `16*Nt*nev**2` / `16*Nt*nev**3` 字节。
4. `readin_vvv_all` 要求 `t=0,...,Nt-1` 的每个文件分别给出精确立方 Nev，并对每片检查
   `nev1 <= Nev`；不能以向下取整、补零或部分文件代替缺失/损坏数据。

读取使用只读 `memmap` 映射完整逻辑形状，但只把各本征模轴前 `nev1` 的逻辑前缀复制到
`complex128` 输出；这避免在用户态复制完整 Nev 张量。它不等于“只发生前缀大小的物理
I/O”：页粒度映射、访问步长和 OS 页缓存仍可能触及更多数据。RSS、页缓存行为及真实文件
I/O 性能均未实测，不得据此声称固定内存降幅或加速比。

这里的 `complex128` 是 f8-pair 文件格式的显式解码结果，不是“任意输入 dtype 原样
round-trip”。若下游计算选择 `complex64`，必须在 reader 边界后显式转换并把转换记录到
元数据；不得把该格式事实与适配层静默升精度混为一谈。

含相位 basis 的 VdV/VVV 或 perambulator 还必须绑定 `lattice_shape=(Lz,Ly,Lx)`、
C-order 展平公式 `z*Ly*Lx+y*Lx+x`、`p_basis_sink/source`、独立的 `q_sink`、Fourier
符号、`Nev`、backend/precision 和组态 ID。文件名中的动量或单独的 perambulator 根目录
都不足以证明 basis 身份；缺失这些字段时只能兼容读取，不能严格 cache hit。

## Ω 权重的语义边界

`create_omega_accelerate` 产生的是无放回抽样的逆包含概率/方差压缩权重。某分区从
`space` 个候选中保留 `sum` 个标签，同一分区有 `k` 个互异指标时，其校正因子为
`prod_{j=0}^{k-1}(space-j)/(sum-j)`；对角重合由实现作相应修正。Ω 用于恢复抽样收缩的
无偏权重，不是算符、矩阵元或 TMD 的物理重整化因子。

当 `normal=True` 且 `dim=2` 时，实现以正对角矩阵 `D` 做 `D Ω D` 对称行和平衡，使每行
和为 `Nev`，同时保持矩阵对称；这是数值归一/平衡，不改变上述物理边界。交接时记录
`n_voxel`、`exact`、`N_eigen/N_sum/N_extract`、`noise`、`conserved`、`normal` 和 `dim`，
不要只保存 Ω 数组而丢失抽样定义。

## 验证顺序

检查输出目录、覆盖策略、shape、dtype、端序、元数据和读写 round-trip。basename-only
路径不得依赖隐式目录创建；跨版本读取必须保留原始后缀和回退路径证据。任何转换
（复数到实数、float 精度或轴重排）都在元数据中说明，不能静默丢失信息。
