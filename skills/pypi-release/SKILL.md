---
name: pypi-release
description: |
  Use when bumping the `pyqcd` version, building or uploading the PyPI sdist,
  checking pypi.org CDN propagation, verifying a clean-venv install of the
  published package, or yanking / rolling back a release.
metadata:
  openclaw:
    emoji: 📦
---

# pypi-release — PyPI 发布运维

## 目的与边界

本技能负责 `pyqcd` 在 PyPI 上的发布闭环：三处版本号同步、sdist 构建与内容校验、
twine 上传、CDN 传播等待、干净 venv 安装自检、回滚策略与双远端推送。

它不判断物理正确性、不改算法、不写测试：发布资格由源码侧的
`python -m pyqcd.testing` 和相应 `pyqcd-*` 技能给出。文档站交给 `readthedocs`，
仓库命名与目录格式交给 `form`。

## 已实测的发布事实

| 项 | 值 |
|---|---|
| 分发名 | `pyqcd`（**不是** `pyqcd-einsum`） |
| 导入名 | `pyqcd` |
| 上传物 | 只有 sdist：`dist/pyqcd-<ver>.tar.gz` |
| 用户名 | `TWINE_USERNAME=__token__` |
| 凭据 | 环境变量 `PYPI_ZHANGXIN8069_API_KEY`（`source ~/env-key.sh` 后可用） |
| 项目页 | `https://pypi.org/project/pyqcd/` |
| 元数据 JSON | `https://pypi.org/pypi/pyqcd/<ver>/json` |
| 简单索引 | `https://pypi.org/simple/pyqcd/` |
| yank/管理 | `https://pypi.org/manage/project/pyqcd/release/<ver>/` |

## 发布策略

- **默认自动发版**：bump → 构建 → 检查 → 上传 → 复验一次走完，不逐次向用户确认。
- 必须先问用户：回滚、删包/删文件、改名（分发名或导入名）、降级。
- 同版本不可覆盖，任何修正都用新版本号。

## 最小工作流程

1. **同步版本号（三处必须一致）**：`pyproject.toml` 的 `[project] version`、
   `pyqcd/__init__.py` 的 `__version__`、`pyqcd/einsum/_version.py` 的 `__version__`。
   `pyqcd/einsum/_version.py` 的 `__native_version__` 是原生 ABI 版本，**不跟着改**。

   ```bash
   python - <<'PY'
   import pathlib, re, tomllib

   ver = tomllib.load(open("pyproject.toml", "rb"))["project"]["version"]
   top = re.search(r'^__version__ = "([^"]+)"',
                   pathlib.Path("pyqcd/__init__.py").read_text(), re.M)[1]
   ein = pathlib.Path("pyqcd/einsum/_version.py").read_text()
   einv = re.search(r'^__version__ = "([^"]+)"', ein, re.M)[1]
   native = re.search(r'^__native_version__ = "([^"]+)"', ein, re.M)[1]
   assert ver == top == einv, f"mismatch: {ver} {top} {einv}"
   print(f"version {ver} consistent in 3 places; __native_version__={native} (kept)")
   PY
   ```

2. **加载凭据**：只从环境读取，不 `echo`、不写文件、不进 git、不进命令历史。

   ```bash
   source ~/env-key.sh
   test -n "${PYPI_ZHANGXIN8069_API_KEY:-}" && echo "key loaded (value hidden)"
   ```

3. **打包前检查**：逐条核对下节四个检查项，任一失败先改源码再构建。

4. **构建 sdist 并做 twine 元数据检查**：

   ```bash
   rm -rf build dist/pyqcd-<ver>.tar.gz
   python -m build --sdist
   python -m twine check dist/*
   ```

   `twine check` 必须 `PASSED`；构建日志里应出现 `PyQCD einsum build backends:` 行。

5. **sdist 内容校验**（防静默漏包）：

   ```bash
   python - <<'PY'
   import tarfile, tomllib
   from setuptools import find_packages

   ver = tomllib.load(open("pyproject.toml", "rb"))["project"]["version"]
   expected = find_packages(include=["pyqcd", "pyqcd.*"],
                            exclude=["pyqcd.testing", "pyqcd.testing.*"])
   with tarfile.open(f"dist/pyqcd-{ver}.tar.gz") as tar:
       names = tar.getnames()
   missing = [p for p in expected if not any(
       n.startswith(f"pyqcd-{ver}/{p.replace('.', '/')}/") for n in names)]
   assert not missing, f"subpackages missing from sdist: {missing}"
   assert not any("pyqcd/testing/" in n for n in names), "pyqcd.testing leaked"
   assert any(n.endswith("README.zh-CN.md") for n in names), "README.zh-CN.md missing"
   print(f"sdist OK: {ver}, {len(expected)} packages, no pyqcd.testing")
   PY
   ```

   当前源码对应 16 个包；任何 `missing` 都说明漏包，必须重建。

6. **只上传 sdist**：

   ```bash
   TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_ZHANGXIN8069_API_KEY" \
     python -m twine upload --non-interactive --disable-progress-bar \
     dist/pyqcd-<ver>.tar.gz
   ```

   不传 wheel：wheel 标签是 `py3-none-any` 却含 Linux `.so`，不符合 PyPI 约定。
   只把 tar.gz 路径交给 twine，不要让 `dist/*` 把 wheel 带上去。

7. **等待 CDN 传播**：`https://pypi.org/pypi/pyqcd/<ver>/json` 往往先 200，而
   `https://pypi.org/simple/pyqcd/` 仍缺新版本，此时 `pip install pyqcd==<ver>`
   报 `No matching distribution found`。轮询 simple index 直到出现 sdist 再复验
   （实测约 1 分钟内）。

   ```bash
   python - <<'PY'
   import time, tomllib, urllib.request

   ver = tomllib.load(open("pyproject.toml", "rb"))["project"]["version"]
   url = "https://pypi.org/simple/pyqcd/"
   for attempt in range(60):
       body = urllib.request.urlopen(url, timeout=15).read().decode()
       if f"pyqcd-{ver}.tar.gz" in body:
           print(f"simple index serves {ver} after ~{attempt * 5}s")
           break
       time.sleep(5)
   else:
       raise SystemExit(f"simple index still lacks {ver} after 300s")
   PY
   ```

   若 `https://pypi.org/pypi/pyqcd/<ver>/json` 是 404，说明上传没成功或版本号不一致，
   不要继续等 CDN。

8. **干净 venv 安装自检**（必须新建 venv，不用当前开发环境）：

   ```bash
   VENV=$(mktemp -d)/venv
   python -m venv "$VENV"
   "$VENV/bin/pip" install -q "pyqcd==<ver>"
   cd /tmp && "$VENV/bin/python" - <<'PY'
   import importlib.util

   import numpy as np
   import pyqcd
   import pyqcd.einsum as pe

   assert "site-packages" in pyqcd.__file__, pyqcd.__file__
   missing = [n for n in ("analysis", "contraction", "gauge", "lattice", "operator",
                          "parallel", "pipeline", "renorm", "smear", "tools", "vertex")
              if importlib.util.find_spec(f"pyqcd.{n}") is None]
   assert not missing, f"subpackages missing from the install: {missing}"

   info = pe.build_info()
   assert info["backend"] in {"python", "kupl", "cuda"}, info
   a, b = np.arange(6.0).reshape(2, 3), np.arange(12.0).reshape(3, 4)
   assert np.allclose(pe.contract("ij,jk->ik", a, b), a @ b)
   _path, pinfo = pe.contract_path("ij,jk->ik", a, b)
   assert float(pinfo.opt_cost) == 48.0, pinfo.opt_cost
   n = len(pe.specializations())
   assert n >= 8, n
   print(f"backend={info['backend']} opt_cost={float(pinfo.opt_cost)} "
         f"specializations={n}")
   PY
   ```

   判据：`backend` 为 `python`/`kupl`/`cuda` 之一（缺工具链时 `python` 是**正确**的降级，
   不是失败）；`opt_cost==48.0`；`specializations()` 当前为 9（注册表增长时同步更新下限）；
   11 个子包全部随包安装。2026-11-08 实测（无 `CXX`/`nvcc` 的干净 venv）：
   `backend=python opt_cost=48.0 specializations=9`，退出码 0，安装只发 RuntimeWarning。

   两个实测注意点：
   - **必须在源码树外运行**（或断言 `__file__` 含 `site-packages`），否则 `import pyqcd`
     命中的是仓库目录，自检验证的是源码而不是安装结果。
   - 用 `importlib.util.find_spec` 判包完整性，不要 `import pyqcd.<sub>`：`pyqcd.tools`、
     `analysis`、`renorm` 等在模块级 import `h5py`/`scipy`，而 `pyproject.toml` 的
     `dependencies` 目前只有 `numpy`，裸 venv 里这些导入必然 `ModuleNotFoundError`。
     这是已知的依赖声明缺口，改动依赖属于需要先问用户的变更。

9. **推送双远端**：发布提交必须两边都推，否则 PyPI 侧有新版本而远端仓库缺提交。

   ```bash
   gpush.sh
   gpush.sh -t gitee.com
   ```

## 打包检查项（每条对应一次真实故障）

1. `[build-system].requires` 必须含 `cython`，否则 pip 隔离构建环境里 `import Cython`
   必然失败，CUDA 扩展无法构建（0.1.3 实测：`requires` 只有 `setuptools>=64`, `wheel`）。
   核对：`grep -n cython pyproject.toml`。
2. `packages` 必须用 `find_packages(include=["pyqcd", "pyqcd.*"],
   exclude=["pyqcd.testing", "pyqcd.testing.*"])`；硬编码子包列表会**静默漏发**
   （0.1.3 实测漏掉 `pyqcd.lattice`、`analysis`、`contraction`、`gauge`、`operator`、
   `parallel`、`pipeline`、`renorm`、`smear`、`tools`、`vertex` 共 11 个）。
   核对：`grep -n "find_packages" setup.py` + 第 5 步内容校验。
   现状记录：仓库里 `dist/pyqcd-0.1.3.tar.gz`（即已上传 PyPI 的那份）命中的就是这一条，
   11 个子包全缺；用修好的 `setup.py` 在隔离环境重建后为 16 个包、`twine check` PASSED。
   修补只能发新版本（如 0.1.4），0.1.3 不可覆盖。
3. `pyqcd.testing`（测试、基准、回归基线）不进包，文档里的基准/回归命令必须标注
   “需源码仓库”；包内代码不得 import 它。
4. 缺工具链时安装必须仍然成功：没有 `CXX`、没有 `nvcc`、没有 Cython 或构建脚本失败时，
   安装降级到纯 Python planner 并以 0 退出（0.1.3 前修过两次：`CXX` 不存在、
   `auto` 选中 CUDA 导致构建脚本成功而 Cython 扩展崩溃）。核对方式：在干净 venv 里用
   `CXX=/nonexistent/c++ NVCC=/nonexistent/nvcc "$VENV/bin/pip" install pyqcd==<ver>`
   必须 exit 0，且 `pe.build_info()["backend"] == "python"`（实测只发 RuntimeWarning）。

## 撤销与回滚

- 同版本不可覆盖（PyPI 拒绝重复文件名），修正只能发新版本。
- yank 只能在网页 `https://pypi.org/manage/project/pyqcd/release/<ver>/` 操作，
  `twine` 不支持 yank；yank 后 `pip install pyqcd` 不再选它，已锁定的
  `pyqcd==<ver>` 仍可装。
- 回滚、删包、改名、降级一律先问用户，不做“静默修好”。

## 常见错误与验收门

| 现象 | 处理 |
|---|---|
| `pip install pyqcd==<ver>` 报 `No matching distribution found`，但 JSON 已 200 | CDN 传播延迟，轮询 simple index 后复验，不要重传 |
| twine 报文件已存在 / `400 File already exists` | 同版本不可覆盖，bump 版本重发 |
| 隔离构建里 `ModuleNotFoundError: No module named 'Cython'` | `[build-system].requires` 缺 `cython` |
| 安装成功但 `import pyqcd.lattice` 失败 | `packages` 硬编码漏包，改用 `find_packages` 重建并重跑第 5 步 |
| 自检全绿但其实测的是源码树 | 在源码树外运行，并断言 `pyqcd.__file__` 含 `site-packages` |
| 裸 venv 里 `import pyqcd.tools` 报 `h5py`/`scipy` 缺失 | 0.1.4 起 `h5py`/`scipy` 已是硬依赖；若复发说明 `dependencies` 被回退 |
| 无 `CXX`/`nvcc` 的机器安装失败 | 工具链检查或降级路径被打断，安装必须仍以 0 退出 |
| `dist/` 里混入 wheel | 只上传 `dist/pyqcd-<ver>.tar.gz`；wheel 标签与 Linux `.so` 不符约定 |
| token 出现在日志/文件/git | 立即停手：只从 `~/env-key.sh` 读取并走 `TWINE_PASSWORD` 环境变量 |

验收门（全部满足才算发版完成）：三处版本号一致；`twine check` PASSED；sdist 含全部子包（当前 12 个顶层子包）且无
`pyqcd.testing`；simple index 已见新版本；干净 venv 自检通过；两个远端均已推送。

## 交接

- 发布内容正确性、原生后端行为：交给 `pyqcd-einsum` / `pyqcd-infra`；测试失败交给 `test`/`debug`。
- 文档站在 Read the Docs 侧的重建与双语校验：交给 `readthedocs`。
- 版本标签与提交格式：交给 `tag` / `form`。

交付：版本号 diff、构建与 `twine check` 日志、sdist 内容校验输出、simple index 轮询结果、
venv 自检输出、两个远端的推送结果，以及未验证项（例如无 CUDA 机器的原生路径）。

## 发布硬化循环（2026-11-08 起，长期任务）

长期迭代目标：让 PyPI 上的 `pyqcd` **足够规范、足够易用、有足够生产力价值**。
台账：`docs/pypi_release_audit_20261108.md`（A 规范性 / B 易用性 / C 生产力，逐项状态）。

### 判定口径（唯一）
只认 **从 PyPI 远程安装** 的结果：

```bash
python -m venv /tmp/v && /tmp/v/bin/pip install --no-cache-dir \
  --index-url https://pypi.org/simple "pyqcd==<ver>"
```

源码树里的 `import pyqcd`、本地 `dist/*.tar.gz` 都不构成证据；自检脚本必须打印
`pyqcd.__file__` 并断言包含 `site-packages`。

### 索引延迟的两个坑（实测）
1. `https://pypi.org/pypi/pyqcd/<ver>/json` 返回 200 **不代表** pip 能装：
   `https://pypi.org/simple/pyqcd/` 及其 CDN 边缘节点可能仍是旧列表，
   pip 会报 `No matching distribution found`。加 `--index-url https://pypi.org/simple`
   并轮询 simple index，等出现新版本再复验。
2. 国内镜像（如清华 `pypi.tuna.tsinghua.edu.cn`）滞后更久；不要在镜像上判"发版失败"。
3. `pip index versions pyqcd` 的最后一行是 pip 升级 notice，必须 `grep "Available versions"`。

### 自检首选入口
```bash
python -m pyqcd            # 版本 + 原生后端 + 可选依赖 + 子包列表
python -m pyqcd --json     # 机器可读，便于脚本断言
```
该入口由 `pyqcd/__main__.py` 提供，不导入任何重依赖，适合作为发布后第一条命令。

### 依赖分层策略（0.1.4 起）
| 类别 | 包 | 声明方式 |
|---|---|---|
| 必装 | `numpy`、`h5py`、`scipy` | `dependencies` |
| 可选后端/作图 | `torch`、`matplotlib`、`cupy`、`mpi4py` | `optional-dependencies`：`torch` / `plot` / `cuda` / `mpi` |

规则：**可选依赖一律函数内延迟导入**，顶层 `import pyqcd` 与任何子包的 import 都不得拉它们。
`matplotlib` 的约定是函数内先 `matplotlib.use("AGG")` 再 `import matplotlib.pyplot`
（`pyqcd/analysis/_plots.py` 是样板，`pyqcd/contraction/_wickplot.py` 曾违反、已修）。

### 类型信息一致性
`classifiers` 声明 `Typing :: Typed` 时，包内必须有 `pyqcd/py.typed` 且进 `package_data`；
否则判定为元数据不一致（PEP 561）。

### 每轮循环的最小动作
1. 读台账 → 2. PyPI 远程安装 → 3. 跑未通过项 → 4. 修（不碰核心算法）→ 5. bump + 构建 + `twine check`
→ 6. 上传 → 7. 重新远程安装复验 → 8. 追加台账一行 + 更新本技能 → 9. `gpush.sh` 与 `gpush.sh -t gitee.com`。

### 0.1.5 踩过的坑（逐条都是实测）
1. **`SOURCES.txt` 泄漏家目录**：`cythonize(build_dir=<绝对路径>)` 会把
   `/home/<user>/.../einsum.cpp` 写进 egg-info，且**重建不自愈**（setuptools 复用旧 egg-info）。
   修法：`build_dir` 与 `sources`/`depends` 全部传 HOME 相对路径；发版前必须
   `rm -rf build *.egg-info dist`。
2. **`py.typed` 三条规则都会漏**：`include-package-data = false` + 顶层 `package_data`
   未列 + `MANIFEST.in` 只写 `recursive-include pyqcd/einsum`。必须同时：
   `package_data={"pyqcd": ["py.typed"]}` 与 `include pyqcd/py.typed`。
3. **放宽 MANIFEST 会连测试一起发**：`recursive-include pyqcd *.py` 会把
   `pyqcd/testing/**` 全带上（实测 177 个成员），必须 `prune pyqcd/testing`。
4. **license 元数据**：`license = { text = "MIT" }` 与 license classifier 均已弃用；
   改成 PEP 639 的 `license = "MIT"` + `license-files = ["LICENSE"]`，
   并把 `[build-system].requires` 提到 `setuptools>=77`。
5. **可选依赖一定要延迟导入**：`cached_contract` 曾在函数顶部无条件
   `from opt_einsum import ...`（opt_einsum 不是依赖）→ 纯安装环境直接崩；
   正确做法是 `try: opt_einsum ... except ImportError: from ..einsum import ...`。
6. **顶层可发现性**：只加 `__all__` 不够（`dir(pyqcd)` 仍空、`pyqcd.einsum` 仍
   `AttributeError`），必须配合 PEP 562 的 `__getattr__`/`__dir__` 懒加载。
7. **安装不 hermetic**：PyPI 只有 sdist 时，编译器"存在但编不过"会让 pip 直接失败。
   构建脚本失败要能被捕获成警告；CUDA 扩展的编译前先做一次主机编译器试编译
   （`int main(){return 0;}`），失败就跳过扩展而不是让 pip 挂掉。
8. **文档代码块要能被机器执行**：把 `usage.md` 的 python 块按顺序 `exec` 一遍，
   立刻暴露 `compute_dual_field_strength(U)`（真实签名 `(F_dict, mu, nu)`）与
   `Bootstrap(data, n_samples=200)`（真实 `(data, Nconf_axes=0, ...)`）两处错误。
   约定：不可运行的片段必须显式写 `# pseudo-code`。
9. **构建产物里的绝对路径审计**：发版前检查
   `tar -xzOf <sdist> */pyqcd.egg-info/SOURCES.txt | grep -c '/home/'` 必须为 0。

### 构建告警的正确处理
`global-exclude`/`recursive-exclude` 在**没有任何文件匹配**时一定会打印
`warning: no previously-included files matching '*.log' found anywhere in distribution`。
这类告警是 setuptools 的固有行为，**不要为了消警告删掉 `*.so`/`*.log`/`*.pyc` 的排除规则**
（它们是防止产物泄漏的安全网）。真正要清零的是：

* `ProjectMetadataDeprecatedLicenseFormat`（`license` 表形式）→ 用 PEP 639 的
  `license = "MIT"` + `license-files`；
* `path must be relative`（`SOURCES.txt` 绝对路径）→ `cythonize` 用相对路径。

发版前的构建告警门：`python -m build --sdist 2>&1 | grep -iE "deprecat|path must be relative"`
必须为空。

### 文档示例门（新增，必跑）
```bash
python -m pyqcd.testing.docs_examples --keep-going       # 英文四页
python -m pyqcd.testing.docs_examples --lang zh --keep-going
```
它把 `docs/<lang>/*.md` 里的 ```` ```python ```` 块按页面顺序 `exec` 一遍；
不可运行的片段必须显式写 `# pseudo-code` 或 `# docs-skip`。
当前基线：en 16 块 / zh 16 块，0 失败。**改文档后必跑**，否则会出现
`compute_dual_field_strength(U)` 这类"签名早已改、文档还在旧写法"的事故。

### 性能 A/B 的三个陷阱（0.1.6 实测教训，务必先读）

1. **`specialize=False` ≠ 只关 kernel**。它关掉整个专用分支，**连计划缓存一起关**。
   用它当"无 kernel 基线"会把缓存收益算进 kernel，得出反向结论。
   正确的用户级对照就是它本身（因为文档就是这么告诉用户的），但要清楚它测的是什么。
2. **kernel 只对"生产契约形状"生效**。E1 的契约是
   3×(4,4,nvec,nvec) perambulator + 3×(4,4) γ + 2×(nmom,nvec,nvec,nvec) VVV。
   用 `(2,2,2,2)` 之类合成形状去测，永远不会命中 kernel，测到的只是计划缓存差异。
   用非契约形状得到的"kernel 慢 10.9×"是**测量假象**，据此加的规模门槛是错的。
3. **先确认结构体计数器**：`pyqcd.einsum._specialize.announce` 现在打印
   `[kernel: <name>]` 或 `[plan-cache]`，**据实反映真正执行了什么**。
   看到 `[plan-cache]` 就说明该次调用没走 kernel，一切性能结论都不成立。

最终实测（生产契约形状，30 次中位数，同机）：

| nvec | 默认 | `PYQCD_EINSUM_SPECIALIZE=0` |
|---|---|---|
| 8 | 0.85–0.97 ms | 1.21 ms |
| 16 | 2.11–2.22 ms | 16.1 ms |
| 24 | 7.16–7.24 ms | 44–49 ms |

即默认**从不变差**，因此不需要任何"小规模回退门槛"。

### `build_info()` 的字段语义
| 字段 | 含义 |
|---|---|
| `backend` | 可用的原生库：`cuda` / `kupl` / `python` |
| `native_available` | 是否真的有可用原生库（`python` 时为 `False`） |
| `array_backend` | 当前数组后端（`numpy` / `cupy` / `torch`）——决定数据在哪、走哪个 kernel |
| `kupl_mpi_build` / `mpi` | KUPL 库是否编入 MPI / 当前生效后端是否是它 |

注意：`backend=cuda` 但 `array_backend=numpy` 时，numpy 数据走的是 **KUPL CPU kernel**，
GPU kernel 只在 cupy/torch 张量上生效。汇报性能时必须同时给这两个字段。

### 公开函数的"静默 None"是发布级缺陷
`compute_dual_field_strength(F_dict, mu, nu)` 曾在缺分量时**静默返回 `None`**、
传 ndarray 时抛出无信息量的 `AttributeError: 'ndarray' object has no attribute 'get'`。
纯安装用户第一次调用就会卡住。约定：

* 参数类型不对 → `TypeError`，并在消息里写清"该传什么、怎么构造"；
* 输入合法但信息不足 → `ValueError`，列出缺哪些键；
* 绝不用 `None` 表示"没算出来"。

### 版本漂移要用测试锁死
`pyqcd/testing/_release_metadata_contract.py` 提供 5 条契约（版本三处一致、
`__all__` 与磁盘目录一致、子包可导入、MANIFEST 策略、`build_info` 字段）。
新增测试函数后**必须**同时：在 `pyqcd/testing/__init__.py` 加 `test_*` 包装，
并在 `pyqcd/testing/conftest.py` 的导入清单里登记——`conftest.main()` 只遍历
`globals()`，漏登记就永远不跑（本次实测 115→116 通过数即验证）。

### `auto` 后端随构建机变化，汇报必须带版本与后端
同一台机器连续两次 `pip install pyqcd` 可能得到 `backend=cuda` 与 `backend=kupl`
（取决于隔离环境里 nvcc/Cython/编译器是否可用）。因此每条性能或结论都要写成
"`pyqcd==<ver>` + `backend=<x>` + `array_backend=<y>`"三件套。

### 文档里的性能数字必须"发布即可复现"
开发构建上测出的数字不能直接写进用户文档：0.1.x 只导出核子 2pt 与 3pt kernel，
而早期文档写的 γ 夹心 8.33×/3.371× 来自**不在发行版里的**融合 kernel。
处理规则：

1. 面向用户的表只放**能在 `pip install` 的包上复现**的数字，并写明
   `pyqcd==<ver>` + `build_info()` 的 `backend`/`array_backend` + 对照版本 + 计时口径；
2. 开发构建的历史数字单独开一节，标题里明确"不可从 PyPI 复现"；
3. 每次发版前用下面脚本重跑一遍可复现表（在纯安装 venv 里）：
   `python /tmp/bench_pub.py`（E1 nvec=16/24、E5 T=8/nv=64，pyqcd vs opt_einsum dp）。

0.1.8 实测（backend=kupl, array_backend=numpy）：E1 nvec=16 → 6.02×，
nvec=24 → 5.83×，E5 → 2.91×。

### extras 必须真的能装
`cuda = ["cupy"]` 在 PyPI 上是**源码包**，`pip install "pyqcd[cuda]"` 会在构建阶段失败；
正确依赖是 `cupy-cuda12x`（CUDA 12.x 预编译轮子），CUDA 11 用户用 `cupy-cuda11x`。
发版检查（不需要真下载）：

```bash
for ex in plot mpi cuda torch; do
  python -m pip install --dry-run --index-url https://pypi.org/simple "pyqcd[$ex]" \
    | grep -oE "Would install .{0,80}"
done
```

### /tmp 占满会伪造出一堆"失败"
本轮 `cupy-cuda12x`/`torch` 的 dry-run 一开始全部无输出，真实原因是 `/tmp`（tmpfs 7.8G）
被历史 venv 撑到 100%，pip 报 `No space left on device` 却被 `grep` 吃掉。
规则：任何"解析/安装失败"先跑 `df -h /tmp`；循环里每轮结束清理
`/tmp/p1*`、`/tmp/audit*/venv*`、`/tmp/buildlog*` 等。

### 为什么本项目不发 wheel（决定 + 理由，别反复重提）
- `libpyqcdeinsum_kupl.so` 的 `ldd` 显示它依赖**构建主机**的 `libstdc++.so.6`
  与 `libgomp.so.1`（conda 路径），直接打 `linux_x86_64` 轮子在其他机器上会缺符号；
- 可移植的 manylinux 轮子需要 `auditwheel repair` + `patchelf`，本机没有；
- 发 `py3-none-any` 轮子更糟：pip 会优先选它，所有用户都退化成纯 Python。
- 结论：**只发 sdist**，并在文档里写明"首次安装会编译约一分钟、pip 会缓存结果"。

### extras 端到端验证（不能只看能否解析）
`pyqcd[plot]` 已在干净 venv 里实跑通过：

```python
from pyqcd.contraction import wick_contraction, plot_figure_wick
w = wick_contraction(sink_operators=[...11 项...], source_operators=[...11 项...],
                     Cpt='2pt', curr_operators=[], Pindex=['p'], Vindex=['v'], Gindex=['g'])
fig, ax = plot_figure_wick(w, diagram_index=0, Cpt='2pt')
fig.savefig('/tmp/wick_demo.png')      # 实测 107 KB，键数 12
```

`pyqcd[torch]` 因 torch 轮子 539 MB 未做端到端安装（只做了 `--dry-run` 解析），
台账里必须标为"未验证"，不要写成已验证。

### extras 端到端验证矩阵（0.1.9 实测）
| extra | 验证方式 | 结果 |
|---|---|---|
| `plot` | 干净 venv 装包 → `wick_contraction → plot_figure_wick` → `savefig` | 107 KB PNG，12 键 ✅ |
| `torch` | 干净 venv 装包 → `set_backend('torch')` → torch 张量 contract vs `a @ b`；`pyqcd.lattice.gamma(0)` | torch 2.15.0+cu132，全部一致 ✅ |
| `cuda` | `pip install --dry-run pyqcd[cuda]` | `Would install cuda-pathfinder, cupy-cuda12x` ✅ |
| `mpi` | `pip install --dry-run pyqcd[mpi]` | `Would install mpi4py` ✅ |

注意：`pyqcd[torch]` 会拉入 CUDA 版 torch 及其 nvidia 依赖（安装后 venv 约 6 GB）。
验证时把 venv 放在**真实磁盘**（如 `~/venvs/`）而不是 `/tmp`（tmpfs 7.8 G 容易被撑满）。
验证完记得删除，并把这条写进循环清理清单。

### PyPI 项目页怎么核对（HTML 有 bot 挑战）
`curl https://pypi.org/project/pyqcd/` 会返回 3 KB 的 "Client Challenge" 页面，
无法用于断言；但 **JSON API 里的 `info.description` 就是渲染源**，直接数探针即可：

```bash
curl -s https://pypi.org/pypi/pyqcd/json | python -c "
import json,sys; d=json.load(sys.stdin)['info']; t=d['description']
for p in ('## Repositories','gitlab.itp.ac.cn','gitee.com','py-qcd.readthedocs.io','python -m pyqcd','6.02'):
    print(p, t.count(p))
"
```

### "drop-in 替换"要按边界矩阵验收（0.1.9 实测 0 失败）
`pyqcd/testing/_contract_edge_cases_contract.py` 覆盖以下矩阵，全绿才算兼容：

| 维度 | 用例 |
|---|---|
| dtype | float32/float64/complex64/complex128/int64/bool |
| 语法 | 隐式输出 `ij,jk`、ellipsis `...ij,...jk->...ik`、重复标签 `ii->i`、迹 `ii->`、标量与 `,ij->ij`、恒等 `ij->ij`、转置、逐元素 |
| 形状 | 非连续视图（`base[::2, ::2]`）、零尺寸维 `(0,4)` |
| optimize | `False/True/'greedy'/'optimal'/'dp'/'auto'` |
| kwargs | `out=`（**注意 `out` 形状必须与真结果一致，写错会得到广播错误而非误判库 bug**） |
| 对象 | `contract_expression`、`shared_intermediates` 的 cache 字典 |
| 规划 | `contract_path(...).opt_cost` 与 `opt_einsum` 完全一致 |
| 错误 | 操作数个数不符 / 标签尺寸冲突 / 输出标签未知 → 必须 `ValueError`/`TypeError` |

**`dtype=` 与 `order=` 不是缺陷**：`opt_einsum` 3.4.0 自己也会抛
`TypeError: _tensordot() got an unexpected keyword argument 'dtype'`；
pyqcd 与之一致即为兼容（bug-for-bug）。若将来要支持，属于**扩展**而非修 bug，
必须在文档里明确写成"超出 opt_einsum 3.4.0 的行为"。

跑法：
```bash
python -m pyqcd.testing._contract_edge_cases_contract    # 或整包 python -m pyqcd.testing
```

### 错误路径与解释器开关也要验（0.1.9 实测）
错误消息必须与参照实现**逐字一致**，否则用户的报错搜索/经验迁移会失效。
6 类非法输入实测全部 `SAME`（操作数个数、标签尺寸冲突、输出标签未知、输出标签重复、
操作数过多、下标类型非法），`contract_path` 在非法输入下同样等价。

解释器开关矩阵（`python -O` / `-OO` / `-B` / 只读 site-packages）必须照常工作：

```bash
for f in "" "-O" "-OO" "-B"; do python $f -c "import pyqcd.einsum as pe; ..."; done
chmod -R a-w <site-packages>/pyqcd && python -B -c "import pyqcd"; chmod -R u+w ...
```

坑：`-OO` 会清空 `__doc__`，凡是用 `__doc__.splitlines()[0]` 当 `argparse` 描述的
工具在 `-OO` 下会 `AttributeError`（本轮 `pyqcd/testing/docs_examples.py` 就中招，已改字面量）。
**发布代码里不要用 `__doc__` 构造运行时必需的值。**
