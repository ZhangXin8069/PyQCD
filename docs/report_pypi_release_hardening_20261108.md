# pyqcd 发布硬化结果报告（2026-11-08）

> 目标：让 PyPI 上的 `pyqcd` **足够规范、足够易用、有生产力价值**；判定口径一律是
> **从 PyPI 远程安装到全新 venv**，源码树里的 `import pyqcd` 不算证据。
> 约束：不修改核心物理/算法代码（接口级 debug 允许）。
> 过程逐轮记录见 `logs/pypi-hardening-20261108.txt`；逐项清单见
> `docs/pypi_release_audit_20261108.md`（A 规范性 / B 易用性 / C 生产力，R0–R21）。

## 1. 一句话结果

6 小时内发布 **9 个版本（0.1.3 → 0.1.11）**，修掉 **4 类发布级缺陷、6 类接口缺陷、
5 处文档失实**，新增 **5 套自动化门禁**；最终 `release_check` **17/17 通过**、
测试 **119 passed / 1 skipped / 0 failed**，打 tag **`dev34`** 并推送 gitlab 与 gitee。

## 2. 最终交付状态

| 项 | 值 |
|---|---|
| PyPI 包 | `pyqcd 0.1.11`，仅 sdist（`pyqcd-0.1.11.tar.gz`） |
| PyPI 页面 | https://pypi.org/project/pyqcd/0.1.11/ |
| 文档站 | https://py-qcd.readthedocs.io/en/latest/ 与 `/zh-cn/latest/`，均含 0.1.11 |
| 标签 | `dev34`（annotated，tag 对象 `cae654c0` → 提交 `42cda20`） |
| 代码托管 | gitlab.itp.ac.cn（国内开发）/ gitee.com（国内发布）/ github.com（国外发布） |
| 安装自检 | `python -m pyqcd` 一条命令给出后端、可选依赖、子包列表 |

## 3. 修掉的发布级缺陷（用户可见）

### 3.1 打包

| 缺陷 | 后果 | 修复版本 |
|---|---|---|
| `packages` 硬编码 5 个包 | 0.1.3 只发 `pyqcd` + `pyqcd.einsum`，`lattice/gauge/smear/operator/vertex/contraction/renorm/tools/analysis/pipeline/parallel` **全部缺失** | 0.1.4（`find_packages(exclude pyqcd.testing)`） |
| `SOURCES.txt` 含 `/home/zhangxin/...` | 开发者家目录随发布永久公开且重建不自愈 | 0.1.5（cythonize 相对路径） |
| `Typing :: Typed` 但无 `py.typed` | PEP 561 声明虚假，类型信息全丢 | 0.1.5 |
| `AGENTS.md` 进 sdist | 内部贡献者说明泄漏给终端用户 | 0.1.5 |
| 放宽 MANIFEST 误带 177 个测试文件 | 包体膨胀且暴露回归基线 | 0.1.5（`prune pyqcd/testing`） |
| `license` TOML 表（2027-02 硬截止） | 未来构建失败 | 0.1.5（PEP 639 + `setuptools>=77`） |

### 3.2 安装可用性

| 缺陷 | 后果 | 修复版本 |
|---|---|---|
| `[build-system]` 无 cython | pip 隔离环境必然 `import Cython` 失败，CUDA 装不上 | 0.1.5 |
| `PYQCD_EINSUM_BACKEND=cuda` 装出半成品 | CUDA `.so` 有、Cython 扩展无、KUPL 也没编 → 运行期 `backend='python'` 且每次告警 | 0.1.6（KUPL 先编 + CUDA 只警告） |
| 编译器存在但编不过 → pip 硬失败 | 用户在"有编译器但环境坏"的机器上完全装不上 | 0.1.6（先试编译 `int main(){return 0;}`，失败只跳过 CUDA） |
| `pyqcd[cuda]` 指向源码包 `cupy` | `pip install "pyqcd[cuda]"` 必然编译失败 | 0.1.8（改 `cupy-cuda12x`） |
| `cached_contract` 无条件 `from opt_einsum import ...` | 纯安装环境调用即崩（opt_einsum 不是依赖） | 0.1.5（回退自带引擎） |
| `h5py`/`scipy` 未声明 | 装完 `import pyqcd.tools` 直接 `ModuleNotFoundError` | 0.1.4 |

### 3.3 接口与易用性

| 缺陷 | 后果 | 修复版本 |
|---|---|---|
| `import pyqcd` 后 `pyqcd.einsum` → `AttributeError`，`dir()` 为空 | 新用户第一步就断 | 0.1.5（PEP 562 懒加载 + `__all__` + 模块 docstring） |
| `compute_dual_field_strength` 缺分量**静默返回 `None`** | 用户以为算出来了 | 0.1.7（`ValueError` 列出缺哪些分量） |
| 同上，传 ndarray → 无信息量 `AttributeError` | 不知如何构造入参 | 0.1.7（`TypeError` + 构造示范） |
| 缺 matplotlib/torch 时裸 `ModuleNotFoundError` | 不知道装哪个 extra | 0.1.5（`pip install "pyqcd[plot]"` / `"[torch]"`） |
| `build_info()` 只报 `backend` | numpy 输入其实走 CPU 库，性能归因错误 | 0.1.6（新增 `array_backend`、`native_available`） |
| 无自检入口 | 用户无法确认装上了什么 | 0.1.5（`python -m pyqcd`） |

### 3.4 文档失实

| 问题 | 修复版本 |
|---|---|
| 性能表用发行版里**不存在**的融合 kernel 数字（γ 夹心 8.33×/3.371×） | 0.1.8：只保留发布即可复现的三行，历史数字单独标注"不可从 PyPI 复现" |
| `usage.md` 两处 API 签名过期（`compute_dual_field_strength(U)`、`Bootstrap(..., n_samples=)`） | 0.1.5，并新增可执行门禁防复发 |
| `dynamic_contraction` 伪代码未标注 | 0.1.5 |
| E1 命中提示引用**已删除的** `PYQCD_EINSUM_KERNEL_MIN_ELEMENTS` | 0.1.10 |
| 未说明"为何没有 wheel / 首次安装要编译" | 0.1.9 |

## 4. 新增的自动化门禁

| 门禁 | 命令 | 基线 |
|---|---|---|
| 文档示例执行 | `python -m pyqcd.testing.docs_examples` | en/zh 各 16 块，0 失败 |
| 文档 API AST 校验 | `test_docs_api_contracts` | 35 个引用，0 失败 |
| 收缩边界矩阵 | `test_contract_edge_cases` | 8 组：全 dtype、ellipsis、标量、零尺寸、6 种 optimize、`out=`、错误类型 |
| 并发契约 | `test_thread_safety` | 多线程 matmul / 专用路径 / `shared_intermediates` 隔离 |
| 发布元数据契约 | `test_release_metadata_contracts` | 版本三处一致、`__all__`↔磁盘、MANIFEST 策略、`build_info` 字段 |
| 一键终检 | `python -m pyqcd.testing.release_check --with-tests` | **17/17** |

## 5. 最终验证证据（实跑）

```
release_check --with-tests
  versions agree        pyproject=0.1.11 pyqcd=0.1.11 einsum=0.1.11
  twine check           PASSED
  sdist contents        153 files; clean（无 testing/.so，含 py.typed）
  sdist absolute paths  0
  docs examples         ok=16 skipped=1 failed=0
  en/zh code blocks     identical
  sphinx -W (en/zh)     exit 0 / 0
  pypi latest version   0.1.11 == local
  pypi ships            sdist only
  pypi page content     all probes present
  rtd rebuilt (en/zh)   contains 0.1.11
  remote in sync        gitlab == gitee == local
  test suite            119 passed, 1 skipped, 0 failed
17/17 checks passed
```

## 6. 性能数据（发布即可复现）

`pyqcd==0.1.11`（`backend=kupl`、`array_backend=numpy`）对 `opt_einsum 3.4.0`（`dp`），
`complex128`，预热 3 次取 15 次中位数：

| 工作负载 | opt_einsum | pyqcd | 倍率 |
|---|---|---|---|
| 核子 2pt，nvec=16（生产契约） | 12.95 ms | 2.15 ms | **6.02×** |
| 核子 2pt，nvec=24 | 41.45 ms | 7.11 ms | **5.83×** |
| γ 夹心，T=8，nvec=64 | 0.169 ms | 0.058 ms | **2.91×** |

默认路径对 `PYQCD_EINSUM_SPECIALIZE=0`：nvec 8/16/24 分别快 **1.3× / 7.5× / 6.3×**。

双原生后端一致性：同一方程在 `PYQCD_EINSUM_NATIVE=kupl` 与 `=cuda` 下结果**逐位相同（0.00e+00）**。

## 7. 已知限制与未验证项（诚实清单）

| 项 | 现状 |
|---|---|
| Python 3.10 / 3.11 / 3.13 | 本机无解释器，**未自动验证**；文档已声明只实测 3.12（完整套件）与 3.14（构建+导入） |
| GPU 通用执行器（cupy 输入） | 可能报 `QCD_STATUS_INTERNAL` 并回落 Python，只警告一次，结果正确；加速只来自专用 kernel |
| 峰值内存 | 约为 `opt_einsum` 的 **1.4–2 倍**（1200²: 18→41 MB；2000²: 73→104 MB），已在文档标注 |
| `pyqcd.testing` | 不随包发布，基准/测试需源码仓库（文档已标注） |
| `pyqcd[torch]` | 端到端已验证（torch 2.15.0+cu132），未在无 GPU 机器复测 |
| 专用 kernel 覆盖 | 仅核子 2pt 与 3pt 主项；E4/E5/sush 其余形态只有计划缓存 |

## 8. 复现与后续维护

```bash
# 安装与自检
pip install pyqcd==0.1.11 && python -m pyqcd

# 一键终检（打 tag 前必跑）
python -m pyqcd.testing.release_check --with-tests

# 新一轮发布
#   1. 改代码/文档 → 2. 版本三处同步 → 3. 构建 sdist → 4. twine check
#   5. 干净 venv 安装自检 → 6. twine upload（仅 sdist）→ 7. 从 PyPI 复验
#   8. 更新台账与 skills → 9. gpush.sh 与 gpush.sh -t gitee.com
```

维护入口：`skills/pypi-release/SKILL.md`（发布全流程与 9 类踩坑）、
`skills/readthedocs/SKILL.md`（双语站与三道一致性门）。
