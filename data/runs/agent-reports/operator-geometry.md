# 梯度流胶子算符与几何只读审计

审计日期：2026-10-22  
工作目录：`/root/PyQCD`  
范围：`pyqcd/renorm/_gradient_flow.py`、`pyqcd/renorm/_tmd.py`、`pyqcd/operator/_gradient_flow_gluon_ope.py`，以及用户指定的 donghx 参考和技能几何契约。  
写入边界：本报告是唯一写入文件；未修改生产代码、`refer/`、缓存或 Git 索引。

## 结论

未发现会推翻当前 `z_dir=2` 非极化或螺旋度组合通道的高严重度错误。Wilson flow RK3 系数、6-staple、四叶 Clover、dual `1/2`、正负 Wilson line、staple 三段路径、端点、无迹投影、基础/伴随归一和求和轴语义均与参考或独立 oracle 一致。

发现 3 个可复现问题：

| ID | 严重级别 | 位置 | 结论 | 当前测试 |
|---|---|---|---|---|
| G1 | 中 | `pyqcd/operator/_gradient_flow_gluon_ope.py:195-217` | `z_dir=1` 时原始 `ti/tj` 采用升序横向方向，偏离 donghx 的循环指派 `i=(z_dir+1)%3`、`j=(z_dir+2)%3`。组合通道因求和不变，但方向分量标签错误。 | 全套相关测试通过，但未检查 `z_dir=1` 的原始 `ti/tj` 方向。 |
| G2 | 低 | `pyqcd/renorm/_tmd.py:162-183` | `compute_dtype` 被读取后完全未使用；请求 `complex128` 或非法 `float32` 都不会改变输出或报错。 | 当前测试通过，但没有覆盖 `M_mu_lambda_nu_rho(..., compute_dtype=...)`。 |
| G3 | 低 | `pyqcd/renorm/_tmd.py:303-310` | 空间求和后才做虚部校验，空间格点上的相反虚部可在求和中抵消并被接受；`spatial_sum=False` 对同一数据会拒绝，行为不一致。 | 当前测试覆盖“时间平均不得掩盖虚部”，但未覆盖“空间求和不得掩盖虚部”。 |

另有一项非生产路径残余风险：`su3_exp` 是四阶 Taylor 加行列式归一，不是严格 SU(3) 指数。生产默认和参考建议的 `eps <= 0.05` 下误差很小，但公开接口接受任意正 `eps`；`eps=0.5` 在小随机 SU(3) 场上可产生约 `2.5e-3` 的幺正性破坏。详见 R1。

## 候选问题

### G1：`z_dir=1` 的原始横向分量方向语义偏离参考

PyQCD 使用：

```python
transverse = [direction for direction in range(3) if direction != z_dir]
i, j = transverse
```

因此 `z_dir=1` 得到 `i=0, j=2`。donghx 的 `flowed_gluon_ope.cpp:301-302` 明确使用：

```cpp
const int i=(zdir+1)%3, j=(zdir+2)%3;
```

同一约定也记录在 `refer/donghx/_Gradient_Flow_GluonPDF/ope_components.py:44`。`z_dir=0,2` 时两者一致；只有 `z_dir=1` 交换 `component=ti/tj`。`Mtiti=ti+tj` 和最终组合不受影响，但原始方向分量及其下游物理解释会错位。

复现命令：

```bash
python -B - <<'PY'
import numpy as np
from pyqcd.operator._gradient_flow_gluon_ope import (
    _field_dictionary, _ordered_field, _straight_bilocal, flowed_gluon_ope,
)
from pyqcd.testing._gradient_flow_gluon_contract import _near_identity_su3
from pyqcd.tools import set_backend
set_backend('numpy')
g = _near_identity_su3(seed=81277)
z_dir = 1
data = flowed_gluon_ope(g, z_count=2, z_dir=z_dir)
fields = _field_dictionary(g, 'traceless')

def raw_ti(i):
    first = _ordered_field(fields, 3, i)
    plus = _straight_bilocal(g, first, first, z_dir, 2, +1)
    minus = _straight_bilocal(g, first, first, z_dir, 2, -1)
    return np.stack((plus, minus, plus + minus, plus - minus), axis=0)

actual = data[1, 0, :, 3]
print(np.max(np.abs(actual - raw_ti(0))))  # 0.0
print(np.max(np.abs(actual - raw_ti(2))))  # 0.21271369773077375
PY
```

输出表明当前 `ti` 精确等于 `i=0`，而参考循环指派要求 `i=2`。

建议的最小测试断言：构造空间各向异性、`z_dir=1` 的确定性 gauge，断言 `data[:, :, :, 3]` 与显式 `i=(z_dir+1)%3` 的 primitive 一致，`data[:, :, :, 4]` 与 `j=(z_dir+2)%3` 一致；同时断言 `ti+tj` 仍等于 `Mtiti`。

### G2：`compute_dtype` 是无效参数

`M_mu_lambda_nu_rho` 在 `pyqcd/renorm/_tmd.py:173-174` 设置 `compute_dtype`，但后续计算和返回都没有使用它。复现：

```bash
python -B - <<'PY'
import numpy as np
from pyqcd.renorm._tmd import M_mu_lambda_nu_rho
from pyqcd.testing import random_su3_gauge
from pyqcd.tools import set_backend
set_backend('numpy')
g = random_su3_gauge(L=2, seed=9123).astype(np.complex64)
for requested in (None, np.complex64, np.complex128, 'complex128', np.float32):
    out = M_mu_lambda_nu_rho(g, 3, 0, 3, 0, 1, 0, compute_dtype=requested)
    print(repr(requested), out.dtype)
PY
```

当前所有分支均输出 `complex64`，包括 `complex128` 和非法 `float32`。这不会改变当前管线默认路径，但会静默违背调用者选择，且可能在下游误判精度。

建议的最小测试断言：`complex64` gauge 配合 `compute_dtype=complex128` 时输出为 `complex128`；`float32`、布尔和字符串 dtype token 必须抛 `TypeError` 或 `ValueError`。

### G3：空间求和可以掩盖逐点材质虚部

`tmd_matrix_elements` 先计算 `sum(O, axis=(1,2,3))`，再对求和结果调用 `_checked_real_part`。若两个空间点分别为 `1+0.1i` 和 `1-0.1i`，求和后为零虚部并被接受；同一个逐点结果在 `spatial_sum=False` 下则被拒绝。复现：

```bash
python -B - <<'PY'
import numpy as np
from unittest.mock import patch
from pyqcd.renorm import _tmd
value = np.array([[[[1.0 + 0.1j, 1.0 - 0.1j]]]], dtype=np.complex128)
g = np.zeros((1, 1, 1, 2, 4, 3, 3), complex)
with patch.object(_tmd, '_tmd_clover_fields', return_value=(None, None, None)), \
     patch.object(_tmd, '_gluon_tmd_operator_from_fields', return_value=value):
    print(_tmd.tmd_matrix_elements(g, [0], [0], L=0)[0, 0])  # 2.0
    try:
        _tmd.tmd_matrix_elements(g, [0], [0], L=0, spatial_sum=False)
    except ValueError as exc:
        print(exc)
PY
```

正确 SU(3) 输入下局部 bilocal 的虚部应仅为舍入误差，因此求和不应当成为失败开放的校验通道。

建议的最小测试断言：在 `tmd_matrix_elements` 和 `tmd_matrix_elements_time` 中，对空间求和前先检查局部 `O` 的材质虚部；构造相反虚部的空间点时必须抛错，而 `1e-14` 级 complex128 舍入误差仍应通过。

## 已核验且未发现错误的项目

| 检查项 | 证据 |
|---|---|
| Wilson flow RK3 系数 | `pyqcd/renorm/_gradient_flow.py:126-136` 与 donghx `flowed_gluon_ope.cpp:195-227` 的三阶段系数逐项一致。 |
| RK3 收敛阶 | `tau=0.2` 时步长 `0.04 -> 0.02 -> 0.01` 的误差比约 `8.04`、`8.95`，符合三阶。 |
| 6-staple 方向与时间 roll | 显式逐格点 oracle 覆盖全部 4 个链接方向和时间方向，最大差 `8.95e-16`。 |
| 流生成元 | 随机 SU(3) 场上 `Z+Z†=0` 的误差为 `0.0`，`Tr Z` 最大 `4.44e-16`。 |
| 四叶 Clover 符号和迹 | 生产 `plaquette_clover` 与测试内独立四叶路径 oracle 在全部有序 Lorentz 对上通过；反序满足 `F_{νμ}=-F_{μν}`，残差 `0.0`。弱场迹归一测试 2/2 通过。 |
| dual tensor 的 `1/2` | 独立 Levi-Civita oracle 最大残差 `0.0`；`dual(F_{01})=F_{23}` 和 `dual(F_{31})=F_{02}` 残差均为 `0.0`。 |
| 直线 Wilson line 正负方向 | 对 `z_dir=0,1,2` 和 `direction=±1` 与显式逐点有向路径比较，最大差不超过 `1.89e-14`。 |
| staple 三段路径和端点 | 对 `z/b/L` 的正、零、负值组合与显式 `-L*z_dir -> b*b_dir -> (L+z)*z_dir -> x+z*z_dir+b*b_dir` oracle 比较，全部通过。 |
| 固定 `L` | 单点 `L=None` 为 `abs(z)`；批量 `L=None` 统一为 `max(abs(z_list))`。现有测试显示固定长度敏感性 `5.95`，批量与显式 `L=max(abs(z))` 差 `0.0`。 |
| 无迹色空间投影 | 两个端点均执行 `F -> F-Tr(F)I/Nc`；加入任意逐点单位阵污染的测试通过，complex64 保留在低层 bilocal。 |
| 基础迹与伴随归一 | 标准 `Tr(T^aT^b)=delta^{ab}/2` 下，`F^a W_adj^{ab}F^b = 2 Tr[FWFW†]`；独立生成元计算残差 `2.14e-14`，实现的两倍因子精确。 |
| 空间/时间求和 | 注入逐时间常量 `1,2,3` 后，`tmd_matrix_elements_time` 返回空间和 `[2,4,6]`，`tmd_matrix_elements` 再取时间均值 `4`。这与函数文档和 3pt 使用方式一致。 |
| `spatial_sum=False` | 返回 `(nz, nb, Nt, Nz, Ny, Nx)`，不会压缩为均值；现有契约测试通过。 |
| `z_dir/b_dir` 动态横向方向 | z_dir 与 b_dir 必须为不同非布尔空间整数；TMD 的 `F_ti/F_tj/F_ij` 随 z_dir 动态构造。组合通道对 `z_dir=0,1,2` 的轴重标记协变误差不超过 `3.33e-16`。G1 仅影响 schema-v2 原始方向标签。 |
| 复数输出 dtype | `tmd_matrix_elements*` 按公开契约返回实 `float64`；`spatial_sum=False` 也返回实值。schema-v2 OPE 保留 `complex128`。G2 是低层单通道 API 的独立 dtype 缺陷。 |

## 参考语义辨析

- donghx `flowed_gluon_ope.cpp:343-367` 将螺旋度 `combined` 保存为欧氏时间约定下的 `Mtiti+Mijij`，这是诊断/原始组合；PyQCD 在 `select_gradient_flow_gluon_component` 中显式提供物理目标 `(Mtiti-Mijij)_odd`，没有把诊断加号冒充物理定义。
- donghx 生产 Clover 同时保留 `legacy_untraced` 和 `traceless`。PyQCD schema-v2 保留两个投影；TMD bilocal 则按技能契约先逐点去迹，二者用途不同，不构成矛盾。
- donghx C++ 要求 `tau/epsilon` 为整数步，PyQCD 使用 `ceil(tau/eps)` 后令最后一步统一为 `tau/n_steps`。后者仍不超过 `eps` 并精确到达 `tau`，属于有意的接口扩展，不是物理偏差。
- `z_count=None` 在 PyQCD 中解析为该周期方向全长；C++ CLI 默认较小的固定值。该差异是调用参数语义，不是几何错误。

## 执行证据

以下只读测试均通过：

```bash
python -B -m pyqcd.testing._gradient_flow_gluon_contract       # 12/12 PASS
python -B -m pyqcd.testing._tmd_su3_geometry_contract          # 7/7 PASS
python -B -m pyqcd.testing._operator_staple_contract           # 6/6 PASS
python -B -m pyqcd.testing._flow_normalization                 # 2/2 PASS
python -B -m pyqcd.testing._ope_channel_contract               # 19/19 PASS
python -B -m pyqcd.testing._field_strength_cache_contract      # 25/25 PASS
python -B -m pyqcd.testing._renorm_numerical_guards_contract   # 26/26 PASS
```

未运行全量 `python -m pyqcd.testing`，也未执行真实 conf6250 流化。此次任务的独立小格点 oracle 已覆盖被审查的几何与代数路径；真实大体积和多 GPU 行为仍不属于本次证据范围。

## 剩余风险

### R1：大 `eps` 时 `su3_exp` 不再保持 SU(3)

`su3_exp` 在 `pyqcd/renorm/_gradient_flow.py:46-55` 使用四阶 Taylor，再在 `proj_su3` 中仅将行列式归一，并未恢复严格幺正性。`wilson_flow` 在 `pyqcd/renorm/_gradient_flow.py:161-166` 接受任意正 `eps`。复现：

```bash
python -B - <<'PY'
import numpy as np
from pyqcd.renorm._gradient_flow import wilson_flow_step
from pyqcd.testing import random_su3_gauge
from pyqcd.tools import set_backend
set_backend('numpy')
U = random_su3_gauge(L=2, seed=211)
for eps in (0.05, 0.5):
    V = np.asarray(wilson_flow_step(U, eps))
    err = np.max(np.abs(V @ V.conj().swapaxes(-1, -2) - np.eye(3)))
    print(eps, err)
PY
```

当前输出约为 `eps=0.05: 8.68e-10`、`eps=0.5: 2.54e-3`。生产 `eps=0.01` 和参考建议上限 `0.05` 下不构成实际偏差；风险是 API 没有拒绝更大步长。

建议的最小防护断言：若继续使用 Taylor 近似，则入口拒绝 `eps > 0.05`；若承诺任意正步长保持 SU(3)，则改用精确 Cayley-Hamilton 指数或可靠的幺正重投影，并测试随机 anti-Hermitian traceless `A` 在 `|A|=0.5` 时的 `expm` 误差与 `VV†` 误差。

### 其他未闭合项

- 未编译并逐位对跑 donghx C++ 程序，只做了源码公式比对和独立 Python oracle。
- 未在真实 conf6250 的 72x24^3 场上执行 `tau=3` 全链，因此大体积缓存、GPU/CPU 累加差异和性能未验证。
- Cupy/CUDA 未实测；NumPy 和已安装的 Torch 路径由现有契约测试覆盖。
- 当前证据上限是算符/几何代数与小型确定性数值核验，不是完整 TMD-PDF 物理方案闭合。

阻塞：无。
