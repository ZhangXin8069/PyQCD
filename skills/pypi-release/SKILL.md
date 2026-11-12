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
   assert n >= 5, n
   print(f"backend={info['backend']} opt_cost={float(pinfo.opt_cost)} "
         f"specializations={n}")
   PY
   ```

   判据：`backend` 为 `python`/`kupl`/`cuda` 之一（缺工具链时 `python` 是**正确**的降级，
   不是失败）；`opt_cost==48.0`；`specializations()` 当前为 **5**（0.3.5 起：删掉旧的
nucleon-2pt/3pt 专用 kernel 后只剩 5 条 `plan-cache` 形态，其余 8638 条走拓扑特化）；
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
4. 缺 KUPL 工具链时安装必须**失败**（0.3.3 起）：KUPL 是原生基线，缺库却 exit 0 等于
   悄悄发一个只会走纯 Python 规划的包——0.3.2 就是这样在只有 `g++` 的集群上翻车的
   （`setup.py` 的探针只认 `CXX`/`c++`，而 `make.sh` 实际用 `g++`）。核对方式：
   `CXX=/nonexistent/c++ "$VENV/bin/pip" install pyqcd==<ver>` 必须 **exit != 0**，且信息里
   含原因、编译器日志路径与 `PYQCD_SKIP_NATIVE_BUILD=1` / `PYQCD_EINSUM_REQUIRE_NATIVE=0`
   两个显式退出开关；`PYQCD_SKIP_NATIVE_BUILD=1` 时必须 exit 0 且
   `pe.build_info()["backend"] == "python"`。CUDA 缺 `nvcc` 仍只警告、exit 0。

## 撤销与回滚

- 同版本不可覆盖（PyPI 拒绝重复文件名），修正只能发新版本。
- yank 只能在网页 `https://pypi.org/manage/project/pyqcd/release/<ver>/` 操作，
  `twine` 不支持 yank；yank 后 `pip install pyqcd` 不再选它，已锁定的
  `pyqcd==<ver>` 仍可装。
- 回滚、删包、改名、降级一律先问用户，不做“静默修好”。

### 回滚已发布版本的两条路（0.2.6→0.2.7 实测）

| 手段 | 效果 | 何时用 |
|---|---|---|
| yank 坏版本 | 不删除文件；`pip install pyqcd` 跳过它，但 `pyqcd==<ver>` 仍可装 | 坏版本有安全/正确性问题且不想再发版 |
| **发新版本号重新发布稳定实现** | 覆盖式接管 `pip install pyqcd`（本次 0.2.7 = 0.2.4 的代码 + 0.2.7 的版本号） | 想让默认安装直接拿到回滚后的实现（推荐） |

回滚发版的完整口径（本次实测 0.2.7）：

1. `git revert` 掉出问题的提交，**不整体回退**（其后可能已有别的合法工作）；核对
   `git diff <上次稳定 tag> -- <受影响子包>` 只剩版本号一行，并 `grep -rn` 断言已删接口 0 处残留；
2. 三处版本号同步到新号（`pyproject.toml`、`pyqcd/__init__.py`、`pyqcd/einsum/_version.py`）
   ——代码回到稳定实现、版本号仍前进，这是唯一能让 pip 默认安装拿到回滚实现的方式；
3. 构建 → `twine check` → 上传 → 从 PyPI 远程安装复验（断言被删 API 不存在、核心结果正确）；
4. 台账写一条"回滚轮"（说清为什么废弃），另立 postmortem 文档记失败经验；
5. 格式错误的已推标签按用户授权撤回后重建，前驱用 `git describe --tags --abbrev=0 <tag>^` 取。

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

### 离线 / air-gapped 安装（HPC 集群常态）
sdist 必须能在**没有网络**时装出来。标准验证流程：

```bash
# 1) 联网机器上准备 wheelhouse（含构建依赖，必须 :all: 二进制）
pip download --only-binary :all: -d /tmp/wh "setuptools>=77" wheel "cython>=3.0" numpy h5py scipy
cp dist/pyqcd-<ver>.tar.gz /tmp/wh/

# 2) 断网安装
PIP_NO_INDEX=1 pip install --find-links /tmp/wh pyqcd==<ver>
```

0.1.9 实测：wheelhouse 7 个包 / 60 MB，`PIP_NO_INDEX=1` 下
`Successfully built pyqcd` + `Successfully installed ... pyqcd-0.1.9`，
装完 `backend=cuda`、contract 正确。**不要忘了 wheelhouse 里要有
`cython`**，否则隔离构建环境拿不到它，CUDA 扩展会被跳过。

### Python 版本声明要诚实
`requires-python = ">=3.10"` + classifiers 3.10–3.14，但发布机器只有 3.12 与 3.14
（3.14 无 pip，用 `setup.py build` + `PYTHONPATH` 验证）。因此文档里必须写清
"实测于 3.12（完整套件）与 3.14（构建+导入），3.10/3.11/3.13 预期可用但未自动验证"，
不要含糊写成"支持 3.10–3.14"。

### "静默漏发"守卫的受控实验（0.1.9 实测）
0.1.3 曾因 `packages=[...]` 硬编码丢掉 11 个子包。现在的配置是否还会漏？
受控实验（做完立刻删除）：

```bash
printf 'PROBE = 1\n' > pyqcd/_tmp_probe.py
mkdir -p pyqcd/_tmp_pkg && printf 'PROBE = 1\n' > pyqcd/_tmp_pkg/__init__.py
python -m build --sdist -o /tmp/probe_dist
tar -tzf /tmp/probe_dist/pyqcd-<ver>.tar.gz | grep -E '_tmp_probe|_tmp_pkg'
```

结果：**模块与子包都会自动进 sdist**（`recursive-include pyqcd *.py` +
`find_packages(include=["pyqcd","pyqcd.*"])`），且 `pyqcd._tmp_pkg` 自动进入 packages。
同时 `test_public_subpackages_exist` 会**立刻失败**并指出
`only on disk=['_tmp_pkg']` —— 即"新增子包忘记登记 `__all__`"会被守卫抓住。
这就是双保险：打包层自动包含 + 测试层强制登记。

### 构建可复现性（结论：内容可复现，字节不可复现）
- 同一工作区连续两次 `python -m build --sdist`：tarball sha256 **不同**，
  但解包后 `diff -r` **完全一致**；
- 加 `SOURCE_DATE_EPOCH=<固定值>`（setuptools 84.2.1）**仍然不同**（tar 成员时间戳等），
  不要为了"字节可复现"花时间。
- 因此：**用内容哈希（解包后逐文件 sha256）判断"是否同一份发布"**，
  不要用 tarball sha256 比较两次构建；PyPI 记录的 sha256 只用于校验那次上传本身。

### 双原生后端一致性（0.1.10 实测）
同一组方程在 `PYQCD_EINSUM_NATIVE=kupl` 与 `=cuda` 下（numpy 输入）：

| 方程 | kupl vs numpy | cuda vs numpy | kupl vs cuda |
|---|---|---|---|
| 核子 2pt（生产契约 nvec=6） | 3.28e-15 | 3.28e-15 | **0.00e+00** |
| `ij,jk->ik`（64×128） | 8.29e-16 | 8.29e-16 | **0.00e+00** |
| ellipsis | 2.91e-16 | 2.91e-16 | **0.00e+00** |
| E4 形态 | 0.00e+00 | 0.00e+00 | **0.00e+00** |

即两条路径**逐位相同**。注意：`PYQCD_EINSUM_NATIVE=cuda` 只是选库，
numpy 数组仍走 CPU kernel；设备端加速要靠 cupy 数组。

### cupy 数组的真实能力边界（必须写进文档）
- 特化形态（核子 2pt/3pt）→ CUDA kernel；形状不匹配 → `[plan-cache]`；
- 通用执行器在 cupy 输入上目前可能抛 `QCD_STATUS_INTERNAL` 并回落 Python，
  只发**一次** `RuntimeWarning`（`_warn_native_fallback_once` 机制），结果仍正确；
- 因此文档必须给出"数组类型 × 执行路径"矩阵，别让用户以为 cupy 一定更快。

### 提示文案会随重构过期
`_specialize.REGISTRY` 的 `note` 曾在移除规模门槛后仍写着
`PYQCD_EINSUM_KERNEL_MIN_ELEMENTS`（该环境变量已不存在），0.1.10 才修。
**改动任何 `note` 依赖的机制时，搜索全仓是否还有引用**：

```bash
grep -rn "REGISTRY\|KERNEL_MIN_ELEMENTS\|PYQCD_EINSUM_[A-Z_]*" pyqcd/einsum/_specialize.py
```

发版前用一条真实调用把 banner 打出来，确认文案与实际行为一致。

### 内存与 memory_limit（0.1.11 实测）
用 `resource.getrusage(RUSAGE_SELF).ru_maxrss` 在**独立进程**里测峰值
（同一进程里连续测会互相污染，必须一进程一次）：

| 形状（float64, dp） | opt_einsum | pyqcd |
|---|---|---|
| 1200²，2 操作数 | 18 MB | 41 MB |
| 1200²，3 操作数 | 30 MB | 52 MB |
| 1200²，4 操作数 | 29 MB | 52 MB |
| 2000²，3 操作数 | 73 MB | 104 MB |

结论：**峰值约为 opt_einsum 的 1.4–2 倍**（单步结束前多保留一份输出同尺寸中间量），
耗时不受影响。这条必须写进面向用户的文档——生产上按 1.5 倍预留内存。

`memory_limit` 语义与 opt_einsum 一致（`None` / `'max_input'` / 具体字节数四档实测
异常类型与成败完全相同）；`largest_intermediate`、`opt_cost`、`path` 也逐值相同。

一个已知的良性回落：`optimize='optimal'` + **很小的** `memory_limit` 时规划器会产出
arity=3 的步，而原生执行器只支持一元/二元步 → 打印一次
`native plan unavailable; using Python execution: step 0 arity must be one or two, got 3`
并回落 Python（结果正确）。这是设计好的 fail-closed 行为，不是 bug；
排查类似告警先看"规划器输出是否超出原生执行器的步型契约"。

### 并发安全（0.1.11 实测，已固化为门禁）
引擎有模块级状态（计划缓存、kernel 句柄、sharing 上下文），必须验证多线程/多进程：

| 场景 | 规模 | 结果 |
|---|---|---|
| 多线程 matmul | 8 线程 × 16 任务 × 25 次 | 0 错误 |
| 多线程专用路径（核子 2pt） | 6 线程 × 12 种子 × 10 次 | 0 错误，逐线程与 numpy 一致 |
| `shared_intermediates` 线程隔离 | 4 线程 | 每线程各自 cache（同尺寸 5 条目），无串扰 |
| 多进程 | 4 进程 × 8 任务 | 全部正确 |

门禁：`pyqcd/testing/_thread_safety_contract.py`（4 个用例，总耗时约 0.07 s），
已接入 `conftest`，`python -m pyqcd.testing` 118→**119 passed**。
写并发测试时注意：`count_cached_ops(cache)` **需要传 cache 参数**，
无参调用会 `TypeError`（本轮就写错过，别误判成库缺陷）。

### 终检一条命令（打 tag 前必跑）
```bash
python -m pyqcd.testing.release_check                 # 全量（含网络）
python -m pyqcd.testing.release_check --offline       # 断网/内网
python -m pyqcd.testing.release_check --with-tests    # 连测试套件一起
```
它按顺序验证 16 项并打印 `PASS/FAIL + 一行证据`，任一失败退出码非 0，可直接做 tag 门：

1. 版本三处一致（pyproject / `pyqcd.__version__` / `einsum._version`）+ 分发名 `pyqcd`
2. sdist 存在（缺则自动构建）、`twine check`、内容洁净（无 `pyqcd/testing`、无 `.so`、
   有 `py.typed`、无 `AGENTS.md`）、`SOURCES.txt` 无 `/home/` 绝对路径
3. 文档示例可执行、中英代码块结构一致（注释各自本地化，英文页无中文）、`sphinx -W` 双站 exit 0
4. PyPI：最新版号与本地一致、只发 sdist、项目页探针（Repositories / 文档链接 /
   `python -m pyqcd` / `6.02`）
5. Read the Docs：en 与 zh-cn 首页都已含当前版本号
6. git 远端：local HEAD == gitlab main == gitee main

0.1.11 实测：全量 **16/16 passed**，`--offline --with-tests` **10/10 passed**；
0.2.7 复测同为 **16/16** 与 **10/10**（套件 123 passed/1 skipped/0 failed）。
注意 `sphinx` 与 `twine` 必须在当前解释器可用（本机是 conda 环境）。

### 与 opt_einsum 3.4.0 的差异必须显式记录（0.1.11 实测）
"完全兼容"是承诺，任何偏差都要写进文档并定期复核：

| 行为 | opt_einsum 3.4.0 | pyqcd |
|---|---|---|
| `optimize=` 传可调用对象 | `TypeError: 'function' object is not iterable` | **支持**（返回 path 即用） |
| `PathInfo` 位置 | `opt_einsum.paths.PathInfo` | 包层级也导出（超集） |
| `contract_expression(*shapes)` | ok | ok（逐值一致的 path/cost） |
| `contract_expression(..., constants={...})` | 同样抛 `ValueError` | 同样抛（bug-for-bug） |
| 不同 shape 的路径缓存 | 正确重算 | 正确重算（cost 240 vs 252 一致） |

结论：pyqcd 是**超集**（多接受 callable、多导出 PathInfo），没有"更少"的能力。
新增差异时必须同时改 `docs/{en,zh}/usage.md` 的 "Differences from opt_einsum" 小节。

### 台账自身的腐化也要修
多轮迭代后，台账**顶部清单**最容易与底部迭代记录矛盾（本轮就是：A8/A10/B2–B5/
C1–C4 明明在后面几轮已验证，顶部还写着"待查"）。规则：

1. 每轮结束把顶部清单的对应行改成"✅ + 轮次号"，而不是只追加一条迭代记录；
2. 收尾用 `grep -c "待查\|待评\|待核\|待复核"` 断言归零；
3. 打 tag 前跑 `python -m pyqcd.testing.release_check --with-tests`，
   全绿（0.1.11 为 **17/17**）才允许打。

### 打 tag 的实操序列（本次 dev41）
```bash
# 1) 门槛前最后一次全量终检（必须全绿）
python -m pyqcd.testing.release_check --with-tests

# 2) 前驱标签必须用 git 取，不许按序号猜
git describe --tags --abbrev=0 dev<N>^

# 3) annotated tag，多行消息用 -F（消息格式见根 AGENTS.md）
git tag -a dev<N> -F /tmp/tagmsg.txt && git push origin dev<N>
git push git@gitee.com:zhangxin8069/PyQCD.git dev<N>

# 4) 三端校验（tag 对象哈希必须一致）
git rev-parse dev<N>
git ls-remote origin refs/tags/dev<N>
git ls-remote git@gitee.com:zhangxin8069/PyQCD.git refs/tags/dev<N>
```
注意：`git push origin dev<N>` 只推该 tag，不要 `--tags`（免得把历史 tag 一并推）。
**`follow` 的后继标签不是序号前驱**：`test<N>` 与 `dev<N>` 各自独立计数，`test12`
的打标时间晚于 `dev40`，所以 `dev41^` 的最近标签是 `test12` 而不是 `dev40`（本次写错过一次，
撤回重建）。一律用第 2 步的命令取前驱，不要按"当前 dev 序号"推断。

### CUDA 编译架构是"支持 CUDA"的真假分界线（0.1.13 实测）
`cpp/cuda/einsum/make-nv.sh` 的 `DEFAULT_ARCHITECTURES="6.0 7.0"`，且只嵌 SASS
（`-gencode=arch=compute_XX,code=sm_XX`，不含 PTX）。因此在比 Volta 新的卡上：

| 构建 | `.so` 内 | kernel 入口返回 | 现象 |
|---|---|---|---|
| 默认 | `compute_70` | `3`（`QCD_STATUS_INTERNAL`） | 静默回落 CPU/Python，**用户以为"支持 CUDA"其实没跑** |
| `PYQCD_CUDA_ARCHITECTURES=8.9` | `compute_89` | `0`（`QCD_STATUS_OK`） | 真跑 GPU（rel 2.8e-14） |

排查三步（不依赖日志）：
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader        # 本机算力
strings <site-packages>/pyqcd/einsum/cuda/libpyqcdeinsum_cuda.so | grep -oE "compute_[0-9]+" | sort -u
python -c "from pyqcd.einsum import _kernel_cuda as k; print(k.enabled())"
```
判据：`PYQCD_EINSUM_SPECIALIZE_VERBOSE=1` 下提示是 `[kernel: cuda_nucleon_2pt]`
（真跑）还是 `[plan-cache]`（没跑）。

`PYQCD_CUDA_ARCHITECTURES` 对 **`pip install` 同样生效**（sdist 现场编译），文档里
必须给出 `PYQCD_CUDA_ARCHITECTURES=8.9 pip install "pyqcd[cuda]"` 这种写法。
可选改进（未实施，需用户批准，属构建配置而非接口）：把默认改成
`code=[sm_XX,compute_XX]` 以嵌入 PTX，让新卡走驱动 JIT。

### CUDA 架构：自动探测 + 始终嵌 PTX（0.1.14 起）
`cpp/cuda/einsum/make-nv.sh` 现在的解析顺序：`PYQCD_CUDA_ARCHITECTURES` →
`CMAKE_CUDA_ARCHITECTURES` → `PYQCD_SM_ARCH` → **自动探测**
（`nvidia-smi compute_cap` → GPU 型号回退表，取自 PyQCU `env.sh`）→ 默认 `6.0 7.0`。
每个架构都编译为 `-gencode=arch=compute_XX,code=[sm_XX,compute_XX]`（cubin + PTX）。

关键收益（实测 RTX 4060 / sm_89）：

| 构建 | kernel status | 说明 |
|---|---|---|
| 旧默认 SASS-only `sm_70` | 3 | 新卡跑不了，静默回落 |
| 显式 `7.0` + PTX | **0** | 驱动 JIT PTX，**无 GPU 的登录节点也能编出可用的包** |
| 自动探测 `8.9` | **0** | 原生 SASS，最快 |

`pip install pyqcd==0.1.14`（不设任何变量）实测 `status=0`。

**诊断方式已变**：`strings <so> | grep compute_` 只对纯 SASS 产物有效；fatbin 布局下
架构名不是明文 ASCII。改用：
1. `pip install -v ... 2>&1 | grep -E "^  arch:"`（构建时打印来源与架构）
2. `readelf -S <so> | grep nv_fatbin`（是否有 fatbin）
3. `PYQCD_EINSUM_SPECIALIZE_VERBOSE=1` 看提示是 `[kernel: cuda_nucleon_2pt]` 还是 `[plan-cache]`

### 基准前先钉住线程（KUPL 实测，直接决定数字可信度）
KUPL 后端是 **OpenMP + dlopen 的 BLAS** 双层并行，两层同时开就会过度并行：

| OMP | OPENBLAS | 700×700 fp64 中位 |
|---|---|---|
| 1 | 1 | 11.04 ms |
| 4 | 1 | 10.80 ms |
| 1 | 4 | **7.25 ms** |
| 4 | 4 | 8.21 ms |
| 8 | 8 | 23.77 ms（退化 3 倍） |

基准脚本必须：①在**独立进程**里跑每种配置；②打印 `OMP_NUM_THREADS` /
`OPENBLAS_NUM_THREADS` / `backend` / `array_backend`；③默认建议
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=<n>`。
详细剖析见 `docs/report_kupl_acceleration_20261108.md`。

### "没接线"与"有门槛"是两件事（0.1.18 的教训）
我一度把 E4/E5 的融合 kernel 说成"未进发行版"——错。事实是：

* `cpp/cuda/einsum/src/e4_fused.cu` / `sandwich.cu` **参与编译**，`apply_contract.cu`
  里 `execute_e4_fused()` 会调用它们；
* 但 `kE4FusedMaxSpatialVolume = 1`：只有 `z*y*x <= 1` 才启用（L24 实测慢 33×，
  属于"主动关闭"而非"没实现"）；
* Python 层只包了 2pt / 3pt，所以提示恒为 `[plan-cache]`——**提示只反映 Python 分发，
  不反映 C++ 执行器内部是否融合**。

判定证据链的正确顺序：
1. `ls cpp/cuda/einsum/src`（源码是否在）；
2. `grep -n "execute_e4_fused\|qcd_cuda_sandwich" apply_contract.cu`（是否被调用）；
3. `grep -n "kE4FusedMaxSpatialVolume" ...`（是否有体积/收益门）；
4. 运行期 `PYQCD_EINSUM_TRACE=1` 看是否真的打出 `PYQCD_EINSUM_CUDA_E4 ... kernel=1`。

**不要**只用 `nm -D | grep qcd_cuda_`（版本脚本 `local: *` 会把内部 kernel 隐藏），
也不要只看 Python 提示。说"未进发行版"之前必须走完这四步。

### 幂等标记与自己的记录必须用同一套键（0.1.22 的真 bug）
`ensure_runtime()` 用 `{"gomp","blas"} - set(state["runtime_applied"])` 判断是否做完，
而 `_set_runtime_threads()` 追加的是 `"blas:<symbol>"` → 集合永远不相等 →
**每次 `contract()` 都重新调用 `openblas_set_num_threads` / `omp_set_num_threads`**。
代价：一次小收缩从 0.05 ms 变成 0.53 ms（E5 从 4.9× 掉到 0.25×）。

修法：运行时记录只写规范键（`"gomp"` / `"blas"`，符号名另存），并加"最多尝试 3 次"兜底。
判据：**数调用次数**，不要只看耗时——
```python
orig = _threads._set_runtime_threads
_threads._set_runtime_threads = lambda *a, **k: (count.__setitem__('n', count['n']+1), orig(*a, **k))[1]
for _ in range(10): pe.contract(...)
assert count['n'] == 1        # 10 次调用只应设置 1 次
```

另外：把自动值写回 `OMP_NUM_THREADS` 会让 `_pinned()` 误判为用户设置，从而永久禁用切档；
必须把导出值记在 `state["exported_env"]`，`_pinned()` 见到相同值要忽略。

### 猎 bug 的有效套路（0.1.23）
按"状态会被跨调用复用"这条线索查，命中率最高：

1. **缓存键**：同形状不同数据、同数据重复、换 dtype、换 out、换 optimize、换 backend；
   每条都要断言"结果与独立参考一致"，而不只是"不报错"。
2. **档位/开关的幂等**：数内部设置的**调用次数**（上个 bug 就是 10 次调用重设 10 次）。
3. **回落是否真的兜住**：monkeypatch 内部函数让它抛异常，验证结果仍正确且警告 ≤1。
   本轮真 bug 就在这里：只有"执行阶段"被包住，`_try_native_contract` 自身抛错会冒到用户。
4. **内存**：几百次调用后 `ru_maxrss` 增量有阈值。
5. **并发**：多线程交替两种形态，逐线程校验。

判"是缺陷"之前必须先跑参照实现：`out` dtype 截断、零尺寸标签冲突这些，
opt_einsum / numpy 的行为完全相同，属于"兼容"而不是 bug。

### 写不变量测试的三条纪律（0.1.24 排查踩坑）
1. **先确认数学对象是什么**：DeGrand–Rossi 的 6..15 号是 **乘积**（γ2γ3 等），不与单个 γ 反交换；
   16/17 是 `γ3γ1·(1±γ4)/2`，平方自然不是 ±1。真正满足 Clifford 的只有 1..4（外加 γ5 反交换）。
2. **先看签名与文档串再调用**：`SU2combine(states)` 收的是 `[(j1,m1),(j2,m2),...]`，
   不是两个矩阵；`SU2decompose(j_list, target)` 也不是它的逆。
3. **区分"理想公式"与"实现约束"**：`plan_parallel` 的目标式是 `N*a=n*b`，但 Y 会被
   任务数/CPU/内存封顶，此时等式不成立——正确的不变量是
   `N == n*Y`、`N*X >= m`、`gpu_needed <= b`，并把 `notes` 里的 cap 当作预期输出。

对应地，文档也要写清"这是未封顶目标 + 压缩规则"，否则用户会按公式反推得到错误结论。

### "性能差"先查是不是测到了旧安装（0.1.25 的真实案例）
用户报告 CPU `0.41×`、GPU sandwich `0.75×`，实际原因是 **notebook 从别的目录运行，
`import pyqcd` 命中了 base conda 里残留的 0.1.12**：该构建没有线程档（8×8 超订）
且 CUDA `.so` 只有 `compute_70` SASS（无 PTX）→ 在 sm_89 上 kernel 返回 status 3 静默回落。

三步定位（都在用户机器上可做）：
1. `print(pyqcd.__version__, pyqcd.__file__)` —— 先确认版本与路径；
2. `python -m pyqcd --gpu-probe` —— kernel 状态/命中提示/结论；
3. `strings <so> | grep compute_` 或直接看 kernel 状态码（3 = 架构不匹配）。

由此固化：**所有基准脚本开头必须打印版本与文件路径**；库里在"注册了 kernel 却没跑"
时 `RuntimeWarning` 立即告警（同形态同原因一次），否则静默回落会把一切伪装成"库很慢"。

### 诊断代码自身的两个坑（0.1.26）
1. **别在读之前取值**：`reason = module.LAST_FAILURE[...]` 写在了 `attempt()` **之前**，
   于是告警里报的是上一次调用残留的原因。凡是"记录失败原因"的模块级状态，
   必须在调用**之后**读取。
2. **诊断不能改全局**：`probe()` 为了看到提示而 `_specialize._announced.clear()`，
   会破坏调用方"每个形态只提示一次"的语义。诊断类函数要**保存/恢复**，
   或用局部开关。

另外：多个 kernel 共享一个 `LAST_FAILURE` 时，必须按形态过滤原因
（E1 → 2pt 标签，P1/P2 → 3pt 标签），否则会把 3pt 的失败原因报给 2pt。

### 顺序计时会把"持平"测成"2 倍差距"（0.1.28）
同一台机器上，顺序测量（先跑完全部 pyqcd 再跑全部 opt_einsum）得到 matmul `0.76×`，
**交错测量**（同一轮里相邻各跑一次，取每轮比值中位数）得到 `0.97×`。忙碌机器上顺序计时
漂移 20–40%，足以伪造一个 2 倍差距。基准脚本一律用交错协议。

### 两条"该绕开自己"的路径（0.1.28 实测）
1. **无求和标签的方程**（逐元素 / 转置 / 对角 / 迹 / 外积）：分步执行纯属簿记，
   直接 `backend.einsum` —— Hadamard 4000² 89 ms → 30 ms。
2. **KUPL 通用计划执行器**：2D 链与单矩阵乘上比 numpy 分步慢 6–10%（打包/置换无收益），
   但在 batch/ellipsis 形态上快 5–10×（numpy 会退化成逐 batch 循环）。
   因此按 `ndim>=3` 启用；CUDA 侧不能关——融合 kernel 就在它的执行器内部。

### 基准尺寸决定结论（0.1.29）
同一份代码，小尺寸与大尺寸的比值可以相差 5 倍以上：

| 形态 | 小尺寸 | 生产尺寸（单次 0.5–2.7 s） |
|---|---|---|
| cupy 批量 GEMM | 0.94× | **0.21×**（原生 batched GEMM 比 cuBLAS 慢 5×） |
| γ 夹心（融合 kernel） | 7–16× | **2.0×**（受工作区带宽限制） |
| 核子 2pt kernel | 2.3×（nvec=64） | **0.99×**（nvec=112，cuBLAS 主导） |

因此：**基准必须跑到生产规模**（单次 1–10 s、重复 3–5 次、交错协议），
否则会把"大尺寸的退化"藏起来，也会把"小尺寸的假优势"当成卖点。

同时记住这条经验：**GPU 上 cupy 已经是 cuBLAS 的薄封装，通用方程交回 cupy.einsum
通常比自研执行器更快**；只有存在融合 kernel 的形态才值得走原生执行器。

## 打包 · 安装 · 线程 · 性能的实测规则（0.1.13–0.1.29）

### 打包配置自查表（八条，每条一个命令）
| 项 | 权威位置 | 核对 → 期望 |
|---|---|---|
| 分发名 `pyqcd` | `pyproject.toml` 的 `[project] name` | `python -c "import tomllib;print(tomllib.load(open('pyproject.toml','rb'))['project']['name'])"` → `pyqcd` |
| 版本三处一致 | `pyproject.toml` / `pyqcd/__init__.py` / `pyqcd/einsum/_version.py` | 「最小工作流程」第 1 步 → `version <ver> consistent in 3 places; __native_version__=0.1.0 (kept)` |
| `__native_version__` 不动 | `pyqcd/einsum/_version.py` | 同上脚本单独打印 → 当前 `0.1.0`，只随原生 ABI 变更而改 |
| 打包清单 | `setup.py`：`find_packages(include=["pyqcd","pyqcd.*"], exclude=["pyqcd.testing","pyqcd.testing.*"])` | 命令 1 → `16 ['pyqcd']`（`pyqcd` + 15 个子包） |
| `py.typed` 三处 | ①文件 `pyqcd/py.typed` ②`package_data={"pyqcd": ["py.typed"]}` ③`MANIFEST.in` 的 `include pyqcd/py.typed`（另需 classifier `Typing :: Typed`） | 命令 2 第 1 行 → `1`；三处缺一即 PEP 561 失效（0.1.5 之前是三处同时缺） |
| 内部文件不入包 | `MANIFEST.in`：`prune pyqcd/testing` + `global-exclude AGENTS.md` | 命令 2 第 2、3 行 → `0`、`0` |
| license 与构建依赖 | PEP 639 `license = "MIT"` + `license-files = ["LICENSE"]`；`[build-system].requires` 含 `setuptools>=77` 与 `cython>=3.0` | `python -m build --sdist 2>&1 \| grep -ci deprecat` → `0`；`grep -c cython pyproject.toml` → `1` |
| `SOURCES.txt` 无绝对路径 | `setup.py` 的 `_rel()` 把 `cythonize` 的 `sources`/`depends`/`build_dir` 全转 HOME 相对路径 | 命令 2 第 4 行 → `0` |

```bash
# 1) 包清单：必须 16，且不含 pyqcd.testing（期望输出: 16 ['pyqcd']）
python -c "from setuptools import find_packages as f; p=f(include=['pyqcd','pyqcd.*'], exclude=['pyqcd.testing','pyqcd.testing.*']); print(len(p), [x for x in p if '.' not in x])"

# 2) sdist 内容四问（T = dist/pyqcd-<ver>.tar.gz）
T=dist/pyqcd-<ver>.tar.gz
tar -tzf "$T" | grep -c "pyqcd/py.typed"        # 1
tar -tzf "$T" | grep -c "pyqcd/testing/"        # 0
tar -tzf "$T" | grep -c "AGENTS.md"             # 0
tar -tzf "$T" | grep -cE "\.so$"                # 0
tar -xzOf "$T" */pyqcd.egg-info/SOURCES.txt | grep -c "/home/"   # 0
```
`grep -c` 无匹配时退出码是 1，脚本里写成 `... | grep -c X || true` 或直接读输出，别让 `set -e` 提前退出。
`SOURCES.txt` 的两步修法：①`cythonize` 的三个路径参数一律过 `_rel()`；②发版前
`rm -rf build *.egg-info dist` 再构建——setuptools 会复用旧 egg-info，**只改代码不清理不自愈**。

### 安装期三阶段的耗时与失败语义
| 阶段 | 触发 | 失败语义（必须如此） | 实测耗时 |
|---|---|---|---|
| pip 隔离构建环境 | 每次从 sdist 安装 | 硬错误（缺 `setuptools>=77`/`wheel`/`cython` 即失败） | 3–8 s |
| KUPL CPU 原生库 | **永远构建**：`PYQCD_EINSUM_BACKEND` 取 `auto`/`kupl`/`cuda` 都含它 | **默认硬错误**（缺工具链或 `build-kupl.sh` 非零即中止安装）；`PYQCD_SKIP_NATIVE_BUILD=1` 或 `PYQCD_EINSUM_REQUIRE_NATIVE=0` 才降级为警告 + 纯 Python planner | ~6 s |
| CUDA 原生库 | **机会性**：`nvcc` 在 PATH + `cpp/cuda/einsum/src/*.cu` 存在 + Cython 可 import | 永远只警告，KUPL 仍是可用原生后端；`=cuda` 也只保证"尝试" | ~20 s |
| Cython CUDA 扩展 | 仅当 CUDA 库建成 | CUDA 没建成时先从 `self.extensions` 里剔除再 `super().run()`，不报错 | ~10 s |

判据：`pip install -v pyqcd==<ver> 2>&1 | grep "build backends"` → `PyQCD einsum build backends: kupl, cuda`；只出现 `kupl` 说明机会性条件不满足（缺 nvcc / 缺 Cython / 缺源），不是缺陷。
主机编译器"存在但编不过"必须先试编译：CUDA 阶段用 `$CXX`（默认 `c++`）编 `int main(){ return 0; }`
（180 s 超时），失败只跳过 CUDA、**KUPL 照旧构建**。

KUPL 的编译器探测必须与 `cpp/kupl/einsum/make.sh` 同序：`CXX` → `g++` → `c++` → `mpicxx`，
并额外要求 `bash` 与 `nm`。2026-11-11 的实测事故：探针只查 `c++`，而集群 modulefile 常只给
`g++`，于是"工具链缺失"被误判 → 静默跳过 KUPL → 用户在 site-packages 里看不到
`libpyqcdeinsum_kupl.so`。判据：仅 `g++` 在 PATH 时也必须编出该 `.so`（357 KB 量级）。

### 构建进度必须写 `/dev/tty`（pip 会吞掉构建输出）
pip 捕获后端的 stdout，只在**失败时**回放；两分钟的编译在用户眼里就是卡死。三处都写：
`setup.py` 的 `_notify_progress()`、`cpp/kupl/einsum/make.sh` 与 `cpp/cuda/einsum/make-nv.sh` 的
`_pyqcd_notify()`（先试 `/dev/tty`，`OSError` 退回 stderr）。

```text
[pyqcd build] compiling the native kupl library (this takes a while; pip will stay silent until it finishes)
[pyqcd build] KUPL (CPU): compiling 12 sources
[pyqcd build] native kupl library built in 6.1s
[pyqcd build] CUDA: compiling 8 sources for arch 8.9 (nvcc, 约 20 s)
[pyqcd build] CUDA: done in 19s
```
判据：`pip install ... > /tmp/install.log 2>&1` 之后 `grep -c "pyqcd build" /tmp/install.log` ≥ 4
（无 tty 时降级到 stderr，所以重定向也能收到）。为 0 说明某条进度链被改回 `print`，安装会像卡死。

### CUDA 架构：先看构建时的 `arch:` 行，再看 kernel 状态码
架构来源优先级：`PYQCD_CUDA_ARCHITECTURES` → `CMAKE_CUDA_ARCHITECTURES` → `PYQCD_SM_ARCH` →
自动探测（`nvidia-smi --query-gpu=compute_cap`，无卡时按 GPU 型号回退表）→ 默认 `6.0 7.0`。
每个架构一律 `-gencode=arch=compute_XX,code=[sm_XX,compute_XX]`，即 **cubin + PTX 双嵌**。

| 构建 | sm_89 上 kernel 状态 | 结论 |
|---|---|---|
| 旧默认 SASS-only `compute_70` | `3`（`QCD_STATUS_INTERNAL`） | 静默回落 CPU，用户以为"支持 CUDA"其实没跑（0.1.12 就是这样） |
| 只指定 `7.0` 但带 PTX | `0` | 驱动 JIT，**无 GPU 的登录节点也能编出可用的包** |
| 自动探测 `8.9` | `0` | 原生 SASS 最快；`pip install` 不设变量即走这条（0.1.14 起） |

```bash
pip install -v pyqcd==<ver> 2>&1 | grep -E "^  arch:"
# 期望:   arch:     8.9 (source: auto-detected; cubin+PTX embedded)
SO=$(python -c "import os,pyqcd.einsum.cuda as m;print(os.path.dirname(m.__file__))")/libpyqcdeinsum_cuda.so
readelf -S "$SO" | grep -c nv_fatbin                     # ≥1
python -m pyqcd --gpu-probe | grep -E "kernel status|verdict"   # ok — ... / CUDA fast path is active
```
`strings <so> | grep compute_` 在 fatbin 布局下**无效**（架构名不是明文 ASCII），不要再拿它当判据。

### 线程档：kernel/generic 双档与不可省的幂等键
`pyqcd/einsum/_threads.py` 用两档，因为 kernel 路径与 BLAS 路径在小机器上要相反的配置：

| 档 | OMP | BLAS | 服务对象 |
|---|---|---|---|
| `kernel` | `min(cores, 8)` | `clamp(cores//2, 1, 4)` | 命中特化形态（核子 2pt/3pt 专用 kernel） |
| `generic` | `max(cores//2, 1)` | `min(max(cores, 2), 4)` | 普通 GEMM 链与打包 |

优先级（高者胜）：`OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` → `PYQCD_EINSUM_OMP_THREADS` /
`PYQCD_EINSUM_BLAS_THREADS` → auto（`cores = physical_cores() // mpi_ranks()`）→
`PYQCD_EINSUM_THREADS=off` 整体关闭。`contract()` 每次按形态选档，档位不变时不动运行时。

实测（4 物理核 / 8 线程，KUPL，0.1.14；GEMM 700² / 核子 2pt）：未设 27.3 / 2.1 ms，
`1/4` 4.3 / 3.7，`4/2` **5.8 / 1.1**（最优），`4/8` 27.5 / 2.7。
判据：`python -m pyqcd | grep threads` → `threads        : omp=4 blas=2 [auto kernel (4 physical cores, 1 rank); runtime: blas]`。

幂等标记必须与自身记录**同一套键**：`ensure_runtime()` 曾用 `{"gomp","blas"} - set(runtime_applied)`
判断完成，而 `_set_runtime_threads()` 追加的是 `"blas:<symbol>"` → 集合永不相等 → **每次 `contract()`
都重设一次线程**，小收缩 0.05 ms → 0.53 ms（10×），E5 从 4.9× 掉到 0.25×。判据是数调用次数：

```python
import numpy as np, pyqcd.einsum as pe, pyqcd.einsum._threads as t
a, b = (np.ones((16, 16)),) * 2
n, orig = {"c": 0}, t._set_runtime_threads
t._set_runtime_threads = lambda *a, **k: (n.__setitem__("c", n["c"] + 1), orig(*a, **k))[1]
for _ in range(10): pe.contract("ij,jk->ik", a, b)
print(n["c"])          # 期望 1；大于 1 就是幂等键又坏了
```
导出值必须记账：`apply()` 会把自动值写回 `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`（GOMP 在
dlopen **时**读它，运行时 API 对未加载的库无效），这些值必须同时记进 `state["exported_env"]`，
否则 `_pinned()` 把它当成用户显式设置，从此**永久禁用切档**。

### 性能方法论：交错测量 + 生产尺度
1. **顺序计时会伪造差距**：同一台机器顺序测（先跑完全部 pyqcd 再跑全部 opt_einsum）得 matmul `0.76×`，
   交错（interleaved）测（每轮相邻各跑一次、取每轮比值、报中位数）得 `0.97×`；忙碌机器上顺序漂移 20–40%，
   足以造出一个假 2 倍。协议：`for _ in range(3): t_pyqcd(); t_ref()`，脚本开头打印两实现的版本与 `__file__`。
2. **尺寸必须到生产量级**：单次 0.5–2.7 s、重复 3 次。小尺寸结论不可外推：

| 形态 | 小尺寸 | 生产尺寸 |
|---|---|---|
| cupy 批量 GEMM（16×1024²） | 0.94× | **0.21×** |
| γ 夹心（融合 kernel） | 7–16× | **2.0×**（8192×1024 工作区带宽受限） |
| 核子 2pt kernel | 2.3×（nvec=64） | **0.99×**（nvec=112，cuBLAS 主导） |

判据：任何"快 N 倍"必须同时给出 单次耗时（秒）、重复次数、交错协议、`OMP`/`OPENBLAS` 线程数、
`backend`/`array_backend`、`pyqcd.__version__`。

### 执行路径选择启发式（五条分支，按输入形态择一）
| 输入形态 | 走哪条路 | 实测（交错，生产尺度） |
|---|---|---|
| 无求和标签：逐元素 / 转置 / 对角 / 迹 / 单操作数 | `backend.einsum` 原生 | Hadamard 4000² 89→30 ms（0.33×→1.00×），生产 1.26× |
| numpy 数组 + batch 标签（两操作数共享且留在输出，或含 `...`） | KUPL 原生计划执行器 | batch 10.6×（生产 11.6×）、ellipsis 5.8×（7.1×） |
| numpy 无 batch（2D 链 / 单矩阵乘 / 双收缩） | numpy 分步（tensordot 阶梯） | chain 1.03×（0.98×）、matmul 0.97×（1.01×）、complex 1.03×（0.96×） |
| cupy + 融合形态 `ab,...bce,cd->...dae`（γ 夹心） | CUDA 原生执行器 | 2.0× |
| cupy 其它 | `cupy.einsum`（cuBLAS） | 16×1024²：0.21× → 0.98× |

强制开关：`PYQCD_EINSUM_KUPL_PLAN=1` 打开 CPU 计划执行器（默认只对 batch 形态启用）、
`PYQCD_EINSUM_NATIVE=kupl|cuda` 选原生库、`PYQCD_EINSUM_SPECIALIZE=0` 关专用分支（连计划缓存一起关）、
`PYQCD_EINSUM_TRACE=1` 看 C++ 执行器内部是否真融合。
判据：同一方程跑两次（一次 `PYQCD_EINSUM_KUPL_PLAN=1`、一次不设），比值落在 0.9–1.1 之外且形态属 batch，
说明分支判定错了；`abc,bcd->ad` 这种"三维但无 batch"的方程走原生会慢 1.5×。

### `python -m pyqcd --gpu-probe` 的判据
```text
GPU probe
  build_info     : backend=cuda native_available=True array_backend=numpy
  cupy           : 14.2.0
  device         : NVIDIA GeForce RTX 4060 Laptop GPU (compute 8.9)
  kernel enabled : True
  kernel status  : ok — kernel returned QCD_STATUS_OK
  hit hint       : [pyqcd.einsum] 专用路径: nucleon-2pt(E1) [kernel: cuda_nucleon_2pt] ...
  verdict        : CUDA fast path is active
```
`verdict` 是给人看的字符串（不要按枚举断言），实际出现过的：`CUDA fast path is active` /
`kernel works, but this operand set did not hit it (shape contract mismatch)` /
`CUDA fast path is NOT active — results fall back`、`no GPU available — ...`、`cupy missing — ...`、`GPU probe failed — see kernel_status`。

判据：**永远 exit 0 且一定有 `verdict` 行**——无 GPU、cupy 缺失、探测自身抛异常都走 `probe-failed`
分支给结论而不是抛异常；无 GPU 时 `kernel status` 为 `no-device` 并给
`unset CUDA_VISIBLE_DEVICES or run on a GPU node` 的提示。诊断不得改动调用方状态
（`probe()` 保存/恢复 `_specialize._announced`）。`python -m pyqcd --gpu-probe --json` 是机器可读版。

### 内核未运行必须立即告警（同形态同原因只报一次）
注册了特化形态却没跑成 kernel 时，库发 `RuntimeWarning`，原因串直接取自 kernel 的
`LAST_FAILURE["reason"]`，四种，各自对应一种处置：

| 原因串 | 含义 / 处置 |
|---|---|
| `kernel disabled (PYQCD_EINSUM_KERNEL_* = 0, or library missing)` | 开关关了：`PYQCD_EINSUM_KERNEL_CUDA` / `..._KUPL` 被设成 `0/false/off/no`，或 `.so` 缺失 |
| `operand shapes do not satisfy the kernel contract` | 形状不符：不是生产契约形状（合成小形状永远不命中） |
| `kernel entry point is missing from the native library` | 版本脚本（`local: *`）隐藏了符号，或装的是旧 `.so` |
| `kernel returned status 3 (wrong CUDA architecture or an internal error)` | 架构不匹配：按上一节重建，必须带 PTX |

去重键是 `(形态名, 原因串)`；原因必须**调用之后**读取、并按形态过滤（E1→2pt、P1/P2→3pt），
否则会把 3pt 的失败原因报给 2pt。回归用例：
`pyqcd/testing/_contract_edge_cases_contract.py::test_kernel_skip_warns_immediately`。
判据：对同一契约形态连续调用两次，`python -W always::RuntimeWarning -c "<契约调用>" 2>&1 | grep -c "did not run"` → `1`（不是 0，也不是 2）。

### 过期文案的检查法（删掉环境变量之后必跑）
删除或改名任何 `PYQCD_*` 开关后，提示文案里残留旧名字不会报错，只会误导：
`_specialize.REGISTRY` 的 note 曾在规模门槛删除后仍写着 `PYQCD_EINSUM_KERNEL_MIN_ELEMENTS`（0.1.10 才修）。

```bash
# 1) 盘点所有被"提到"的名字（提示文案、help、文档都算）
grep -rhoIE "\bPYQCD_[A-Z0-9_]+" pyqcd docs README.md | sort -u
# 2) 对可疑名字逐条找读点（读点长这样：os.environ.get(…)、getenv("…")、${VAR:-}、_ENV = "PYQCD_…"）
grep -rnI "PYQCD_EINSUM_KERNEL_MIN_ELEMENTS" pyqcd cpp setup.py | grep -v __pycache__
```
判据：第 2 步**必须为空**（该变量早已删除，实测只有 `docs/` 里的历史台账还提它，
那是记录不是文案）；反过来，若命中的行是 `note=`、`help=`、`print(`、`warnings.warn`，即为过期文案。
开关常经常量间接读取（`DISABLE_ENV = "PYQCD_EINSUM_SPECIALIZE"` + `os.environ.get(DISABLE_ENV)`），
所以别指望一条 `comm` 自动判定，要逐个指出读点。
最后用一条真实调用把 banner 打出来（`PYQCD_EINSUM_SPECIALIZE_VERBOSE=1 python -c "...contract(...)"`），banner 里出现的每个 `PYQCD_*` 都必须能指到读点。

### 踩坑清单（0.1.12–0.1.29，每条都有实测数字）
| 现象 | 真因 | 处置 / 判据 |
|---|---|---|
| 用户 notebook 里 CPU `0.41×`、GPU sandwich `0.75×` | `import pyqcd` 命中 base conda 里的 **0.1.12**：无线程档（8×8 超订）+ CUDA `.so` 只有 `compute_70` SASS → sm_89 上 status=3 | 先 `print(pyqcd.__version__, pyqcd.__file__)` + `--gpu-probe`；新 venv 复跑 0.92/8.38/0.90/8.36×，环境升到 0.1.26 后 sandwich 16.15× |
| `No matching distribution found` 但 JSON 已 200 | simple index 比 `/pypi/<ver>/json` 慢（CDN 边缘） | 轮询 simple index（实测约 1 分钟），不要重传 |
| 国内镜像装不到新版本 | 镜像同步滞后 | 复验一律 `--index-url https://pypi.org/simple`，别在镜像上判发版失败 |
| `pip index versions` 抓不到版本 | 最后一行是 pip 升级 notice | `\| grep "Available versions"` |
| `strings <so> \| grep compute_` 什么都没 | fatbin 里架构名不是明文 ASCII | 用 `pip -v` 的 `arch:` 行 / `readelf -S \| grep nv_fatbin` / 运行期 `[kernel: ...]` |
| `count_cached_ops()` 抛 `TypeError` | 该函数需要 cache 参数 | `count_cached_ops(cache)`，别误判成库缺陷 |
| `np.einsum(..., optimize='optimal')` 不返回 | 15 个操作数的全路径枚举组合爆炸 | 用 `timeout 120` 当判据；改 `'dp'`/`'greedy'` 或直接交给 `pyqcd.einsum` 默认规划 |
| `plan_parallel` 给的 `N` 不满足 `N*a=n*b` | 该式是**未封顶目标**，Y 会被任务数/CPU/内存压缩 | 正确不变量：`N == n*Y`、`N*X >= m`、`gpu_needed <= b`；压缩原因写在 `notes` |
| 解析/安装莫名失败 | `/tmp` tmpfs 被历史 venv 撑满 | 先 `df -h /tmp` |

### 发版检查单（一条命令 + 远程复验 + 双推 + tag）
```bash
# 1) 终检 16 项（--with-tests 连 123 passed 套件）：必须 16/16；离线口径 10/10
python -m pyqcd.testing.release_check --with-tests
# 2) 构建与元数据
rm -rf build *.egg-info dist && python -m build --sdist && python -m twine check dist/*
# 3) 只上传 tar.gz
TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_ZHANGXIN8069_API_KEY" \
  python -m twine upload --non-interactive --disable-progress-bar dist/pyqcd-<ver>.tar.gz
# 4) 发布后必须从 PyPI 远程安装复验（本地 sdist 不算证据）
python -m venv /tmp/rv && /tmp/rv/bin/pip install --no-cache-dir \
  --index-url https://pypi.org/simple "pyqcd==<ver>"
/tmp/rv/bin/python -c "import pyqcd; print(pyqcd.__version__, pyqcd.__file__)"   # 含 site-packages
/tmp/rv/bin/python -m pyqcd --gpu-probe
# 5) 双推
gpush.sh && gpush.sh -t gitee.com
# 6) tag：多行消息用 -F，别堆 -m
git tag -a dev<N> -F /tmp/tagmsg.txt && git push origin dev<N>
```
17 项覆盖：版本三处一致、sdist 构建 / `twine check` / 内容洁净 / 无绝对路径、文档示例可执行、
中英代码块结构一致且英文页注释为英文、`sphinx -W` 双站、PyPI 版本与项目页探针、RTD 双语言含当前版本、
gitlab/gitee 与本地 HEAD 同步、测试套件（`--offline --with-tests` 为 10 项）。
发布后 24 h 内发现的任何问题只能用新版本号修；远程复验输出的 `pyqcd.__file__` 必须含 `site-packages`。
