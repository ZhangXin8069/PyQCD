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

## 选择与验证

1. 默认先用 numpy 语义写算法，再显式设置 backend；不要在业务代码中散落 `if torch`。
2. 切换后用同一小输入比较形状、dtype、有限性和数值误差；结果超差时先核对后端与
   precision，再查算法。
3. 记录 backend、precision、设备和比较容差。混合后端的 einsum 或隐式 CPU/GPU 拷贝
   必须显式暴露。

历史基准仅作待复测参考：torch CPU 梯度流相对 numpy 约 2.7–4.8×，已知样例
`max|Δ|≈1e-15`；性能随线程、输入和版本变化，不能直接当验收标准。
