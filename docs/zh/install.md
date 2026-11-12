# 安装

包以 `pyqcd` 的名字发布在 PyPI 上（源码分发 sdist）。pip 会在本地构建，原生内核也在
这一步于本机编译。

```bash
pip install pyqcd
```

当前发行版：`0.3.8`。需要固定版本时：`pip install "pyqcd==0.3.8"`。

PyPI 上**有意不发平台 wheel**：原生库链接的是构建主机的 C++ 运行时，要产出可移植的
manylinux wheel 需要 `auditwheel` 修补。因此首次安装会编译一次（笔记本上约一分钟），
pip 会缓存编译好的 wheel，在相同解释器的其它环境里再装就很快。

## 依赖

| 项 | 用于 |
|---|---|
| Python ≥ 3.9 | 必需 |
| `numpy`、`h5py`、`scipy` | 必需（自动安装） |
| `bash` + C++17 编译器 | 原生 KUPL（CPU）内核 —— 基线构建 |
| `nvcc` | 可选的原生 CUDA（GPU）内核 |
| Cython | 可选的 CUDA 扩展；由 pip 构建环境提供，无需手动安装 |
| `torch` | 可选：torch 后端（`pip install "pyqcd[torch]"`） |
| `matplotlib` | 可选：作图辅助（`pip install "pyqcd[plot]"`） |

`requires-python` 为 `>=3.9`。本套文档背后的验证在 Python 3.12（完整测试套件）与
3.14（构建 + 导入 + `python -m pyqcd`）上执行；3.9–3.11、3.13 预期可用，但发布机器上
没有自动化实跑记录。Python 3.8 已不再支持：发行版改用 PEP 639 许可证元数据，它要求
`setuptools>=77`，而该版本已放弃 3.8。

Cython 不需要自己装：`pyproject.toml` 已在 `[build-system] requires` 中列出
`cython>=3.0`，所以 pip 为 `pip install pyqcd` 创建的隔离构建环境里一定有它；
只有当构建使用 `--no-build-isolation` 时，才需要手动 `pip install Cython`。

## 可选 extras

```bash
pip install "pyqcd[cuda]"     # adds cupy-cuda12x (CUDA 12.x)
pip install "pyqcd[torch]"    # adds torch (optional backend)
pip install "pyqcd[plot]"     # adds matplotlib (Wick/analysis plots)
```

extras 只拉取 Python 依赖；原生库仍在同一次 `pip install` 中、在有工具链时编译。

## 安装过程做了什么

1. **KUPL 总是构建**。CPU 原生库（`cpp/kupl`）是基线：用宿主 C++17 编译器编译，
   然后放入 `pyqcd/einsum/kupl/`。
2. **CUDA 是在此之上的额外尝试，不是替代**。默认 `auto` 模式下，只有当 `nvcc`、
   `cpp/cuda/einsum/src` 下的 CUDA 源码、以及 Cython 三者齐备时，才会再编译 CUDA
   库及其 Cython 扩展。Cython 由 pip 构建环境提供，所以实际起作用的门槛就是
   `nvcc`。KUPL 先编，因此 CUDA 工具链出问题绝不会连带丢掉 CPU 后端。
3. **KUPL 构建失败会让安装失败；CUDA 失败只警告**。KUPL 是原生基线：工具链缺失或
   `build-kupl.sh` 返回非零时，`pip install` 直接中止，并在信息里给出原因、编译器
   日志路径与两个显式退出开关。CUDA 构建失败只发 `RuntimeWarning`，安装继续，
   `pyqcd.einsum` 仍走 KUPL。
4. 按实际产出的库安装 `pyqcd`；`build_info()['backend']` 报告运行期真正可用的后端。

下列开关在 `pip install` 构建阶段读取：

| 变量 | 作用 |
|---|---|
| `PYQCD_EINSUM_BACKEND=kupl` | 只构建 KUPL |
| `PYQCD_EINSUM_BACKEND=cuda` | 构建 KUPL **和** CUDA —— 该取值表示“两个都编”，KUPL 仍是必需基线 |
| `PYQCD_EINSUM_BACKEND=auto`（默认） | 总是构建 KUPL；`nvcc`、CUDA 源码与 Cython 齐备时再加编 CUDA |
| `PYQCD_EINSUM_REQUIRE_NATIVE` | 默认 `1`：**KUPL** 的跳过或失败是硬错误；设为 `0` 才允许带着纯 Python 规划装完。对 CUDA 从不生效 |
| `PYQCD_EINSUM_SKIP_CYTHON=1` | 假装没有 Cython：`auto` 降级为只编 KUPL，`cuda` 仍编 CUDA 库但不编其 Cython 扩展 |
| `PYQCD_SKIP_NATIVE_BUILD=1` | 完全跳过原生编译，若已有现成库则直接复用；否则就是显式选择纯 Python 规划 |

## 安装包里有什么

`pip install pyqcd` 安装的是整个工具包：`pyqcd` 本身，加上 `lattice`、`gauge`、
`smear`、`operator`、`vertex`、`contraction`、`renorm`、`tools`、`analysis`、
`pipeline`、`parallel`，以及 `pyqcd.einsum` 和实际构建出的原生产物（KUPL 总会尝试，
CUDA 只在编译成功时才有）。

`pyqcd.testing` **不在包里**。`setup.py` 以
`find_packages(exclude=["pyqcd.testing", "pyqcd.testing.*"])` 发布，因此测试、基准
驱动与回归基线只留在源码仓库。在已安装环境里执行 `python -m pyqcd.testing` 会报
`ModuleNotFoundError: No module named 'pyqcd.testing'`；这类命令请在源码检出根目录
执行（见最后两节）。

## 验证安装

最快的检查是模块入口，它不需要写 import，直接报告实际装上了什么：

```bash
python -m pyqcd            # version, native backend, optional deps, subpackages
python -m pyqcd --version  # version only
python -m pyqcd --json     # machine-readable
```

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

`backend` 取值为 `cuda`、`kupl` 或 `python`（后者表示正在用纯 Python 规划）。
优先 CUDA：CUDA 扩展导入不了时回落到 KUPL，因此 CUDA 构建被跳过的机器仍然报
`kupl`。

功能自检：

```python
import numpy as np
import pyqcd.einsum as pe

a = np.arange(6.0).reshape(2, 3)
b = np.arange(12.0).reshape(3, 4)
assert np.allclose(pe.contract("ij,jk->ik", a, b), a @ b)
```

## 运行期环境变量

| 变量 | 作用 |
|---|---|
| `PYQCD_EINSUM_NATIVE` | `kupl` / `cuda` / `auto`（默认）；`auto` 优先 CUDA，失败回落 KUPL，再回落纯 Python 规划 |
| `PYQCD_EINSUM_SPECIALIZE=0` | 关闭方程形态特化 |
| `PYQCD_EINSUM_DISABLE_FUSED=1` | 关闭原生融合内核 |
| `PYQCD_EINSUM_KERNEL_KUPL=0` / `PYQCD_EINSUM_KERNEL_CUDA=0` | 关闭指定后端的专用 kernel |
| `PYQCD_EINSUM_ORACLE=installed` | 差分测试改用已安装的 `opt_einsum` |

`PYQCD_EINSUM_BACKEND`（见上）在构建阶段做同样的选择：构建期读的是
`PYQCD_EINSUM_BACKEND`，运行期读的是 `PYQCD_EINSUM_NATIVE`。

### CUDA 架构：自动探测，且始终嵌入 PTX

构建时的架构解析顺序：

| 优先级 | 来源 |
|---|---|
| 1 | `PYQCD_CUDA_ARCHITECTURES`（可多值，逗号/分号/空格分隔） |
| 2 | `CMAKE_CUDA_ARCHITECTURES` |
| 3 | `PYQCD_SM_ARCH`（旧变量名） |
| 4 | 自动探测：`nvidia-smi --query-gpu=compute_cap`，再退到 GPU 型号回退表 |
| 5 | 默认 `6.0 7.0`（无 GPU 的构建机，例如 CI 或集群登录节点） |

每个架构都以 `code=[sm_XX,compute_XX]` 编译——**cubin 与 PTX 并存**。带 PTX 意味着
没有对应 SASS 的新卡仍能运行：驱动会把 PTX JIT 成目标机器码。没有 PTX 时，
kernel 入口返回 `QCD_STATUS_INTERNAL`，引擎会静默退回 CPU 路径。

在 RTX 4060 Laptop（算力 8.9）上实测：

| 构建方式 | kernel 状态 | 结果 |
|---|---|---|
| 旧默认，仅 SASS（`sm_70`） | `3`（`QCD_STATUS_INTERNAL`） | 回落 CPU |
| 自动探测 `8.9`，cubin+PTX | `0`（`QCD_STATUS_OK`） | 走 GPU，rel 2.8e-14 |
| 显式 `7.0`，cubin+PTX | `0`（`QCD_STATUS_OK`） | 经 PTX JIT 走 GPU，rel 2.8e-14 |

因此**开箱即用**已经能在当前显卡上跑 GPU。只有在想要最快路径（为你的卡生成原生
SASS），或构建机没有 GPU、但你明确知道目标架构时，才需要设变量：

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader      # e.g. 8.9
PYQCD_CUDA_ARCHITECTURES=8.9 pip install "pyqcd[cuda]"
```

诊断已安装的包：

```bash
# 1) which architectures were compiled (visible during the build)
pip install -v "pyqcd[cuda]" 2>&1 | grep -E "^  arch:"

# 2) does the installed archive carry a CUDA fatbin at all (cubin + PTX)?
readelf -S <site-packages>/pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so | grep nv_fatbin

# 3) the decisive check — does the GPU kernel really run?
PYQCD_EINSUM_SPECIALIZE_VERBOSE=1 python -c "
import numpy as np, cupy as cp, pyqcd.einsum as pe
e = 'gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M'
r = np.random.default_rng(0)
o = [cp.asarray((r.standard_normal(s)+1j*r.standard_normal(s)).astype(np.complex128))
     for s in [(4,4,16,16)]*3 + [(4,4)]*2 + [(4,16,16,16)]*2 + [(4,4)]]
pe.contract(e, *o)
"
```

提示为 `[kernel: cuda_nucleon_2pt]` 表示真的跑了 GPU kernel，`[plan-cache]` 表示没跑。
注意：`strings | grep compute_` 只对"纯 SASS"产物有效；fatbin 布局下架构名不是明文
ASCII，请改用运行期提示判断。

### 首次安装为什么要一分钟——以及怎么看进度

`pip install pyqcd` 全部从源码构建，顺序如下（4 核笔记本实测）：

| 阶段 | 典型耗时 | 做什么 |
|---|---|---|
| pip 构建隔离 | 3–8 s | 建临时 venv 并安装 `setuptools`、`wheel`、`cython` |
| KUPL（CPU）库 | ~6 s | 一次 `g++ -O3 -fopenmp` 编译 6 个源文件 |
| CUDA 库 | ~20 s | 一次 `nvcc` 编译 10 个源文件（无 `nvcc` 时跳过） |
| Cython 扩展 | ~10 s | 把 CUDA 库接到 Python（无 CUDA 时跳过） |
| 打 wheel + 安装 | 2–5 s | 落位 `.so` 并安装 wheel |

合计约 **带 CUDA 30 s、不带 10–15 s**，另加 numpy/h5py/scipy 下载时间。

pip 默认吞掉构建输出，所以编译期间看起来像卡死。为此构建过程**直接把进度写到终端**
（`/dev/tty`，没有终端时退回 stderr），安装时你会看到：

```
[pyqcd build] compiling the native kupl library (this takes a while; pip will stay silent until it finishes)
[pyqcd build] KUPL (CPU): compiling 6 sources
[pyqcd build] KUPL (CPU): done in 6s
[pyqcd build] native kupl library built in 6.3s
[pyqcd build] compiling the native cuda library (this takes a while; pip will stay silent until it finishes)
[pyqcd build] CUDA: compiling 10 sources for arch 89 (nvcc, about 20 s)
[pyqcd build] CUDA: done in 19s
[pyqcd build] native cuda library built in 20.0s
```

想看完整编译器日志，或者确实怀疑卡住：

```bash
pip install -v "pyqcd[cuda]"     # full build backend log
echo $?                          # 0 means the install finished; the wheel is cached
pip cache list pyqcd             # the built wheel is reused by later installs
```

缩短安装的几种办法：

| 场景 | 做法 |
|---|---|
| 只需要 CPU 内核 | `PYQCD_EINSUM_BACKEND=kupl pip install pyqcd`（跳过 nvcc，省约 25 s） |
| 完全不想编译 | `PYQCD_SKIP_NATIVE_BUILD=1 pip install pyqcd` → 纯 Python 规划 |
| 多个环境 | 装一次，之后复用 pip 的 wheel 缓存（无需额外参数） |
| 离线集群 | 先做 wheelhouse，见「离线 / air-gapped 安装」一节 |

## 从源码仓库构建（开发用途）

```bash
bash install.sh kupl     # CPU (KUPL) backend
bash install.sh cuda     # CUDA backend + Cython extension
bash install.sh          # auto
source ./env.sh          # PYTHONPATH / LD_LIBRARY_PATH / backend selection
```

CUDA 架构按 `PYQCD_CUDA_ARCHITECTURES` → `CMAKE_CUDA_ARCHITECTURES` → `PYQCD_SM_ARCH`
的顺序解析；换新代显卡时务必显式指定：

```bash
PYQCD_CUDA_ARCHITECTURES=8.9 bash install.sh cuda
```

## 测试（源码仓库）

这些命令依赖 `pyqcd.testing`，而它不在 PyPI 包里，因此只能在源码检出根目录执行：

```bash
python -m pyqcd.testing                                  # integration suite

python -m pyqcd.testing.einsum.verify_native --backend kupl
python -m pyqcd.testing.einsum.verify_native --backend cuda
python -m pyqcd.testing.einsum.verify_compat             # differential vs opt_einsum

bash pyqcd/testing/einsum/simple/run-all.sh              # einsum sub-system suite
bash pyqcd/testing/simple/run-all.sh                     # per-function API parity suite
```

## 源码仓库

本项目在 GitLab 主开发，并同步到 Gitee（国内发布）与 GitHub（国外发布），三者历史一致：

* 国内开发 — https://gitlab.itp.ac.cn/ZhangXin8069/PyQCD
* 国内发布 — https://gitee.com/zhangxin8069/PyQCD
* 国外发布 — https://github.com/zhangxin8069/PyQCD

## 后端与数组类型

| 输入的数组 | 实际执行的路径 |
|---|---|
| `numpy` | 装了 KUPL 原生库就走 CPU 原生，否则纯 Python 规划 |
| `cupy` | 特化形态走 CUDA kernel；通用执行器目前可能报 `QCD_STATUS_INTERNAL` 并回落到 Python —— 只会发一次 `RuntimeWarning` 说明原因，结果始终正确 |
| `torch` | torch 后端（`set_backend("torch")`），即由 PyQCD 派发到 `torch.einsum` |

`pyqcd.einsum.build_info()` 同时给出 `backend`（哪个原生库可用）与
`array_backend`（你的数组是什么）；这是两个不同的问题——`backend='cuda'` 的构建，
numpy 输入依然跑在 CPU 库上。

## 性能开关

下面所有开关都从环境变量读取，可放在命令行、脚本或作业文件里。

### 线程（两个后端通用）

引擎在 import 时一次性配置 OpenMP 与 BLAS，因为"两个运行时都吃满逻辑核"是这里最常见的
性能事故来源（它们会互相超订）。

| 变量 | 作用 | 默认 |
|---|---|---|
| `OMP_NUM_THREADS` | OpenMP 工作线程——最高优先级 | 自动 |
| `OPENBLAS_NUM_THREADS` | BLAS（矩阵乘）线程——最高优先级 | 自动 |
| `PYQCD_EINSUM_OMP_THREADS` | 同上，第二优先级 | 自动 |
| `PYQCD_EINSUM_BLAS_THREADS` | 同上，第二优先级 | 自动 |
| `PYQCD_EINSUM_THREADS=off` | 完全关闭自动配置 | 开启 |

kernel 路径与 BLAS 路径想要**相反**的配置，因此引擎维护两档并按调用切换——命中特化形态
用 `kernel` 档，其余用 `generic` 档：

| 档位 | OpenMP | BLAS | 面向 |
|---|---|---|---|
| `kernel` | `min(物理核, 8)` | `clamp(物理核 // 2, 1, 4)` | 核子 2pt/3pt 专用 kernel |
| `generic` | `clamp(物理核 // 2, 1, 8)` | `clamp(物理核, 2, 4)` | GEMM 链、打包、置换 |

只有档位真正变化时才切换，且配置**作用到已加载的运行时**（`libgomp` 用
`omp_set_num_threads`，OpenBLAS 用 `openblas_set_num_threads` 或 numpy/scipy wheel 里的
`scipy_openblas_*` 符号名），不只是导出变量。

在 4 物理核 / 8 线程笔记本上实测（KUPL 后端）：

| 配置 | 通用链 800³ | 核子 2pt，nvec=32 |
|---|---|---|
| `PYQCD_EINSUM_THREADS=off`（全默认） | 103.8 ms | 25.7 ms |
| **默认（按调用切档）** | **15.0 ms** | **21.1 ms** |
| 钉死 `OMP=1` / `BLAS=4` | 15.9 ms | 57.5 ms |
| 钉死 `OMP=4` / `BLAS=8` | 55.7 ms | 21.6 ms |

按调用切档可以同时拿到两种手工调优的效果：同一进程里，链式收缩走 `generic` 档、
内核走 `kernel` 档。

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

### KUPL（CPU）后端

| 变量 | 作用 | 默认 |
|---|---|---|
| `PYQCD_EINSUM_NATIVE=kupl` | 强制使用 CPU 库 | auto |
| `PYQCD_EINSUM_LIB` | 指定要加载的 `libpyqcdeinsum_kupl.so` | 包内 |
| `PYQCD_EINSUM_CBLAS_LIB` | 指定 `dlopen` 的 BLAS（如 MKL、BLIS） | 自动探测 |
| `PYQCD_EINSUM_POOL=0` | 关闭执行层缓冲池 | 开启 |
| `PYQCD_EINSUM_CONTEXT_CACHE_LIMIT` | 缓存的原生上下文个数 | 内置 |
| `PYQCD_EINSUM_KERNEL_KUPL=0` | 跳过 KUPL 专用 kernel | 开启 |
| `PYQCD_EINSUM_SPECIALIZE=0` | 跳过整个专用路径（kernel + 计划缓存） | 开启 |
| `PYQCD_EINSUM_SPECIALIZE_VERBOSE=1` | **每次**调用都打印命中提示 | 仅首次 |
| `PYQCD_EINSUM_DISABLE_FUSED=1` | 保留规划，去掉融合路径 | 关闭 |

核子 2pt kernel 的诊断开关（不影响数值）：

| 变量 | 作用 |
|---|---|
| `PYQCD_KUPL_N2PT_MODE=steps\|fused` | 7 步参考路径 vs 融合实现 |
| `PYQCD_KUPL_N2PT_PROFILE=1` | 逐阶段耗时输出到 stderr |
| `PYQCD_KUPL_N2PT_BUDGET_MB` | 中间缓冲预算（默认 512 MB） |
| `PYQCD_KUPL_N2PT_CHUNK` | 强制 `a1` 分块大小 |
| `PYQCD_KUPL_N2PT_POISON=1` | 每次调用前把工作区填 NaN，自检"未写就读/跨调用残留" |

### CUDA 后端

| 变量 | 作用 | 默认 |
|---|---|---|
| `PYQCD_EINSUM_NATIVE=cuda` | 强制使用 GPU 库 | auto |
| `PYQCD_EINSUM_KERNEL_CUDA=0` | 跳过 CUDA 专用 kernel | 开启 |
| `PYQCD_CUDA_ARCHITECTURES` | **构建期**目标架构（SASS+PTX） | 自动探测 |

提醒：`PYQCD_EINSUM_NATIVE=cuda` 选的是**库**，不是数据。numpy 数组仍走 CPU 库；
GPU kernel 需要 `cupy` 数组。

### 构建期

| 变量 | 作用 |
|---|---|
| `PYQCD_EINSUM_BACKEND=kupl\|cuda\|auto` | 构建哪些库 |
| `PYQCD_SKIP_NATIVE_BUILD=1` | 完全跳过编译 |
| `PYQCD_EINSUM_REQUIRE_NATIVE=1` | 缺 KUPL 工具链时变成硬错误 |
| `PYQCD_EINSUM_SKIP_CYTHON=1` | 编 CUDA 库但不编 Cython 扩展 |
| `PYQCD_SM_ARCH` | CUDA 架构的旧变量名 |

### 查询接口

```python
import pyqcd.einsum as pe

pe.build_info()        # backend / native_available / array_backend / native_library
pe.thread_info()       # effective omp / blas threads and where they came from
pe.specializations()   # registered equation-shape specialisations

info = pe.build_info()
assert {"backend", "native_available", "array_backend"} <= set(info)
assert len(pe.specializations()) >= 1
```
