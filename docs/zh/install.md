# 安装

```bash
pip install -U pyqcd
```

**强烈建议始终用 `pip install -U pyqcd` 安装最新版，并关注安装过程中的输出与任何提醒。**
包以 `pyqcd` 的名字发布在 PyPI 上；一次安装就得到整个工具库，以及本站记录的全部内容。

当前发行版：`0.5.18`。需要固定版本时：`pip install "pyqcd==0.5.18"`。

## 依赖

| 项 | 用于 |
|---|---|
| Python ≥ 3.9 | 必需 |
| `numpy`、`h5py`、`scipy` | 必需（自动安装） |
| `torch` | 可选：torch 后端（`pip install "pyqcd[torch]"`） |
| `cupy-cuda12x` | 可选：GPU 数组（`pip install "pyqcd[cuda]"`） |
| `matplotlib` | 可选：作图辅助（`pip install "pyqcd[plot]"`） |
| C++17 编译器、`bash`、`nm`、OpenMP、OpenBLAS 头文件/库 | 默认原生 KUPL 构建 |
| 含 `nvcc` 和兼容主机编译器的 CUDA 工具链 | 可选 CUDA 归档 |

`requires-python` 为 `>=3.9`。Python 3.8 已不再支持。

默认 KUPL 归档在本机编译，使用可移植的基线 ISA。只有在所有导入该安装的进程
都运行于同一代 CPU、且明确需要 `-march=native` 时，才设置
`PYQCD_EINSUM_NATIVE_ARCH=1`。

## 可选 extras

```bash
pip install "pyqcd[cuda]"     # adds cupy-cuda12x (CUDA 12.x)
pip install "pyqcd[torch]"    # adds torch (optional backend)
pip install "pyqcd[plot]"     # adds matplotlib (Wick/analysis plots)
```

## 验证安装

最快的检查是模块入口，它不需要写 import，直接报告实际装上了什么：

```bash
python -m pyqcd            # version, native backend, optional deps, subpackages
python -m pyqcd --version  # version only
python -m pyqcd --json     # machine-readable
```

`python -m pyqcd` 还会打印构建时配置/实际安装的原生后端，以及 CUDA 被跳过的原因；
因此 pip 丢弃成功构建日志后，这些决策仍可追溯。

```python
import pyqcd
import pyqcd.einsum as pe

print(pyqcd.__version__)
print(pe.build_info())
# {'backend': 'kupl', 'native_available': True,
#  'array_backend': 'numpy', 'native_library': '.../libpyqcdeinsum_kupl.so'}

info = pe.build_info()
assert pyqcd.__version__.count(".") == 2
assert info["backend"] in {"cuda", "kupl", "python"}
assert isinstance(info["native_library"], str)
```

`backend` 取值为 `cuda`、`kupl` 或 `python`（后者表示正在用纯 Python 规划）；
`array_backend` 是你的数组是什么——这是两个不同的问题，`backend='cuda'` 的安装里
numpy 输入依然跑在 CPU 库上。

功能自检：

```python
import numpy as np
import pyqcd.einsum as pe

a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)
assert np.allclose(pe.contract("ij,jk->ik", a, b), a @ b)
```

## GPU 安装检查

CUDA 从 sdist 现场构建，因此 `nvcc` 必须出现在**调用 `python -m pip` 的同一个
shell** 中。若它不存在，安装仍可使用可移植的 KUPL CPU 后端。进入 GPU 节点后，
先查询 PyQCD 实际加载了什么：

```bash
python -m pyqcd --gpu-probe
python -m pyqcd --gpu-probe --json
```

判断前先读 `cuda archive` 行。`kernel enabled` 只是环境开关；归档行才包含实际路径
以及 `ready`、`missing`、`load-failed`、`symbol-missing` 之一。CUDA 归档缺失或
不完整时，设置环境开关并不能修复：应加载 CUDA 工具链，再用同一解释器、禁用缓存
重新安装。即使机器没有 GPU，probe 也会检查归档，因此打包缺陷不会被设备缺席掩盖。

```bash
python -m pip install --force-reinstall --no-cache-dir "pyqcd==0.5.18"
```

需要明确安装纯 Python 规划器时：

```bash
PYQCD_SKIP_NATIVE_BUILD=1 python -m pip install --force-reinstall --no-cache-dir "pyqcd==0.5.18"
```

构建模式：`PYQCD_EINSUM_BACKEND=auto` 为缺省，`kupl` 只请求 CPU 原生，
`cuda` 尝试 KUPL+CUDA，`PYQCD_SKIP_NATIVE_BUILD=1` 安装不含 `.so` 的纯 Python
规划器。默认 KUPL 构建失败是硬错误；没有原生编译器时必须显式使用跳过开关。

## 安装包里有什么

`pip install pyqcd` 安装的是整个工具包：`pyqcd` 本身，加上 `lattice`、`gauge`、
`smear`、`operator`、`vertex`、`contraction`、`renorm`、`tools`、`analysis`、
`pipeline`、`parallel`，以及 `pyqcd.einsum`。

`pyqcd.testing` **不在包里**：测试套件、基准驱动与回归基线只留在源码仓库，所以在已安装
环境里执行 `python -m pyqcd.testing` 会报 `ModuleNotFoundError`。本站展示的一切都只需
已安装的包即可复现。

## 运行期环境变量

| 变量 | 作用 |
|---|---|
| `PYQCD_EINSUM_NATIVE` | `kupl` / `cuda` / `auto`（默认）；`auto` 优先 CUDA，失败回落 KUPL，再回落纯 Python 规划 |
| `PYQCD_EINSUM_SPECIALIZE=0` | 关闭方程形态特化 |
| `PYQCD_EINSUM_DISABLE_FUSED=1` | 关闭原生融合内核 |
| `PYQCD_EINSUM_KERNEL_KUPL=0` / `PYQCD_EINSUM_KERNEL_CUDA=0` | 关闭指定后端的专用 kernel |
| `PYQCD_EINSUM_KERNEL_FORCE` | 强制一条路线：`generic` / `dedicated` / `topology` / `backend`（单值，缺省 `generic`） |
| `PYQCD_EINSUM_NOTICE=0` | 关闭首次命中提示 |
| `PYQCD_EINSUM_LANG=zh\|en` | 提示语言 |
| `PYQCD_EINSUM_CUDA_LIB` | 覆盖诊断与拓扑执行使用的 CUDA 归档路径 |

## 后端与数组类型

| 输入的数组 | 实际执行的路径 |
|---|---|
| `numpy` | 装了原生 CPU 库就走 CPU 原生，否则纯 Python 规划 |
| `cupy` | 特化形态走 CUDA kernel |
| `torch` | torch 后端（`set_backend("torch")`），即由 PyQCD 派发到 `torch.einsum` |

## 线程

引擎在 import 时一次性配置 OpenMP 与 BLAS，因为"两个运行时都吃满逻辑核"是这里最常见的
性能事故来源（它们会互相超订）。

| 变量 | 作用 | 默认 |
|---|---|---|
| `OMP_NUM_THREADS` | OpenMP 工作线程——最高优先级 | 自动 |
| `OPENBLAS_NUM_THREADS` | BLAS（矩阵乘）线程——最高优先级 | 自动 |
| `PYQCD_EINSUM_OMP_THREADS` | 同上，第二优先级 | 自动 |
| `PYQCD_EINSUM_BLAS_THREADS` | 同上，第二优先级 | 自动 |
| `PYQCD_EINSUM_THREADS=off` | 完全关闭自动配置 | 开启 |

运行期查看或修改：

```python
import pyqcd.einsum as pe

pe.thread_info()                 # {'omp': 4, 'blas': 2, 'source': 'auto (...)', ...}
pe.configure_threads(4, 2)       # pin OpenMP=4, BLAS=2 explicitly
pe.configure_threads(blas=4)     # only change BLAS

info = pe.thread_info()
assert info["omp"] >= 1 and info["blas"] >= 1
assert "profile" in info
```

`python -m pyqcd` 也会打印当前生效值。

## 查询接口

```python
import pyqcd.einsum as pe

pe.build_info()        # backend / native_available / array_backend / native_library
pe.thread_info()       # effective omp / blas threads and where they came from
pe.specializations()   # registered equation-shape specialisations

info = pe.build_info()
assert {"backend", "native_available", "array_backend"} <= set(info)
assert len(pe.specializations()) >= 1
```

## 源码仓库

本项目在 GitLab 主开发，并同步到 Gitee（国内发布）与 GitHub（国外发布），三者历史一致：

* 国内开发 — https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD
* 国内发布 — https://gitee.com/zhangxin8069/PyQCD
* 国外发布 — https://github.com/zhangxin8069/PyQCD
