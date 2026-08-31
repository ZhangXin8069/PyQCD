# Backend reference — numpy / cupy / torch

## 适用范围

只在任务涉及计算后端、精度切换、numpy-like 适配层或 CPU/GPU 一致性时读取。本文件
描述 PyQCD 的实际接口边界；不替代物理算法或显存规划。

## 接口

```python
from pyqcd.tools import set_backend, set_precision, get_backend

set_backend("torch")             # gpu/cuda 为别名
set_precision("complex64")       # 或 complex128
xp = get_backend()                # numpy-like 适配层
```

- numpy/cupy 输入可自动转 torch；复数 dtype 遵循全局精度。
- `xp` 已覆盖 `einsum`、`roll(axis=)`、任意轴 `transpose`、`take(axis=)`、`linalg`、
  `cos/sin/arccos/isnan/clip/maximum`、`argwhere`、`identity`、`append` 和 `random`。
- torch Tensor 适配了 `transpose`、`astype`、`.T`、`repeat(axis=)`、`get` 和二元运算。

## eigcompress 精度契约

`compress_matrix_V3`、`compress_matrix_V4` 和 `create_noise` 的输入若为
`complex64`，输出必须保持 `complex64`；NumPy 路径不能因随机矩阵或宿主侧临时数组静默
升为 `complex128`，Torch 路径也必须先用 `set_precision("complex64")` 建立对应 dtype。
同一 seed 的调用应可复现，合法的 `complex64` 舍入误差不能被固定的 `complex128` 容差
误报为正交性失败。正交检查的受控容差为
`max(1e-8, 16*eps(real_dtype))`；它是数值验收阈值，不是物理误差预算。

```bash
python -m pyqcd.testing._eigcompress_dtype_contract
```

该契约覆盖 NumPy/Torch（依赖缺失时显式 skip），并分别检查 dtype、seed 和正交性；仍须
在目标 GPU 与实际 `Nev/volume` 上重新测量显存、wall time 和数值误差。

## 选择与验证

1. 默认先用 numpy 语义写算法，再显式设置 backend；不要在业务代码中散落 `if torch`。
2. 切换后用同一小输入比较形状、dtype、有限性和数值误差；结果超差时先核对后端与
   precision，再查算法。
3. 记录 backend、precision、设备和比较容差。混合后端的 einsum 或隐式 CPU/GPU 拷贝
   必须显式暴露。

全局 Torch `device` 是把 NumPy/CuPy 等外部输入转换为 Tensor 时的默认目的地，不得覆盖
业务入口已经收到的 Torch Tensor 所在设备。当前 `pyqcd.gauge` 的公开观测量采用输入
Tensor device/dtype 优先规则，并只接受 `float32/float64/complex64/complex128`；其
逐观测量 dtype 映射由 `pyqcd-gauge/references/observables.md` 定义。其他模块不能未经
各自契约核实就外推这条 dtype 白名单。

## Vertex 与 GPU 计时契约

`phase_exp_2pt/phase_exp_3pt`、`Mom_VdV_sink_t` 和 `Mom_VVV_sink_t` 的计算结果必须保持
输入 backend、设备和复数 dtype；业务层不得用 `np.asarray` 把 Torch/CuPy 张量隐式拉回
host。矩形空间格点使用显式 `lattice_shape=(Lz,Ly,Lx)`，动量顺序固定为
`(pz,py,px)`；旧单个 `Nx` 入口只能作为立方格点兼容形式。

步骤计时必须包住 GPU 完成边界：CuPy 同步当前 null stream，Torch 只在配置 device 为
`cuda*` 时同步对应 CUDA device，Torch CPU 不触碰 CUDA runtime。主计算已经抛出时，
后同步错误不得覆盖主异常；主计算成功后的同步错误必须上抛。可执行门：

```bash
python -B -m pyqcd.testing._momentum_smearing_contract
python -B -m pyqcd.testing._pipeline_runtime_contract
```

本技能不保留固定加速倍率。性能结论必须在当前代码、输入、线程数和设备上重新测量，
同时报告 wall time、峰值内存与同输入数值误差；历史倍率不能升级为当前验收证据。
