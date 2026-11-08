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
| 裸 venv 里 `import pyqcd.tools` 报 `h5py`/`scipy` 缺失 | 已知的依赖声明缺口（`dependencies` 只有 numpy）；包完整性门用 `find_spec`，不要改成补 import |
| 无 `CXX`/`nvcc` 的机器安装失败 | 工具链检查或降级路径被打断，安装必须仍以 0 退出 |
| `dist/` 里混入 wheel | 只上传 `dist/pyqcd-<ver>.tar.gz`；wheel 标签与 Linux `.so` 不符约定 |
| token 出现在日志/文件/git | 立即停手：只从 `~/env-key.sh` 读取并走 `TWINE_PASSWORD` 环境变量 |

验收门（全部满足才算发版完成）：三处版本号一致；`twine check` PASSED；sdist 含 16 个包且无
`pyqcd.testing`；simple index 已见新版本；干净 venv 自检通过；两个远端均已推送。

## 交接

- 发布内容正确性、原生后端行为：交给 `pyqcd-einsum` / `pyqcd-infra`；测试失败交给 `test`/`debug`。
- 文档站在 Read the Docs 侧的重建与双语校验：交给 `readthedocs`。
- 版本标签与提交格式：交给 `tag` / `form`。

交付：版本号 diff、构建与 `twine check` 日志、sdist 内容校验输出、simple index 轮询结果、
venv 自检输出、两个远端的推送结果，以及未验证项（例如无 CUDA 机器的原生路径）。
