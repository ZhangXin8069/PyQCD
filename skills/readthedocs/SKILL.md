---
name: readthedocs
description: |
  Use when editing or building the PyQCD Read the Docs sites in `docs/en/` and
  `docs/zh/`, adding a documentation page, keeping the bilingual fenced code
  blocks byte-identical, or fixing a failed Read the Docs build.
metadata:
  openclaw:
    emoji: 📚
---

# readthedocs — Read the Docs 双语文档站

## 目的与边界

本技能负责 PyQCD 两套 Read the Docs 站点的源文、构建门和发布联动：目录与配置、
页面集合、双语代码块一致性、`-W` 构建验收，以及推送后触发重建。

它不负责 PyPI 发版（交给 `pypi-release`）、不负责 LaTeX 报告与 PDF 版式
（交给 `pyqcd-docs`）、不定义物理结论。`docs/` 根目录下的研究报告和 `.tex`
不属于任何站点源文。

## 站点事实

| 项 | 英文 | 中文 |
|---|---|---|
| 源目录 | `docs/en/` | `docs/zh/` |
| 配置 | `docs/en/.readthedocs.yaml` → `docs/en/conf.py` | `docs/zh/.readthedocs.yaml` → `docs/zh/conf.py` |
| 语言 | `en` | `zh-cn` |
| 站点 | `https://py-qcd.readthedocs.io/en/latest/` | `https://py-qcd.readthedocs.io/zh-cn/latest/` |
| 依赖 | `docs/en/requirements.txt` | `docs/zh/requirements.txt` |

- 项目 slug 是 `py-qcd`，不是 `pyqcd`。
- 配置文件名**必须**是 `.readthedocs.yaml`（RTD 不认其他拼写），并且要被仓库根
  `.gitignore` 放行：根 `.gitignore` 用 `.*` 忽略隐藏文件，现由
  `!docs/en/.readthedocs.yaml`、`!docs/zh/.readthedocs.yaml` 显式豁免；
  未跟踪 = 线上不存在。
- `sphinx.configuration` 分别指向 `docs/en/conf.py`、`docs/zh/conf.py`；
  两份 `conf.py` 都不 `import pyqcd`，因此构建不依赖原生库或 CUDA。
- 不使用 `docs/` 根目录建站，不引用 `docs/` 下的研究报告。

核对命令：

```bash
git ls-files docs/en/.readthedocs.yaml docs/zh/.readthedocs.yaml
grep -n "docs/en/.readthedocs.yaml\|docs/zh/.readthedocs.yaml" .gitignore
```

## 页面集合与视角

- 每侧固定五个源文：`index.rst` 加 MyST 的 `overview.md`、`usage.md`、
  `performance.md`、`install.md`。
- 新增页必须**同时**加入两份 `index.rst` 的 toctree，否则线上无导航或 404。
- 视角规则：从 PyPI 包 `pyqcd` 出发，示例只依赖 `import pyqcd` 与
  `pip install pyqcd`，不出现仓库路径、`pyqcd/testing`、`build-*.sh`。
  确需源码仓库的段落必须显式标注“需源码仓库 / source checkout only”
  （`install.md` 的构建与测试段、`performance.md` 的基准段已如此）。

## 最小工作流程

1. 改 `docs/en/` 的页面；示例先按已安装包的口吻写。
2. 同步 `docs/zh/` 对应段落，**围栏代码块逐字节复制**，不重译代码。
3. 跑双语一致性门（下节脚本），必须 `0` mismatch。
4. 新增页时补两份 `index.rst` 的 toctree。
5. 两份 `-W` 构建都必须 exit 0。
6. `gpush.sh` 与 `gpush.sh -t gitee.com` 双推，Read the Docs 才会重建。
7. 复验 `https://py-qcd.readthedocs.io/en/latest/` 与 `/zh-cn/latest/` 的新页面。

## 双语一致性门

中英文的围栏代码块必须逐字节相同；正则 ```` ```[a-z]*\n(.*?)``` ```` + `re.S`：

```bash
python - <<'PY'
import pathlib, re, sys

BLOCK = re.compile(r"```[a-z]*\n(.*?)```", re.S)
PAGES = ("overview", "usage", "performance", "install")
bad = 0
for name in PAGES:
    en = BLOCK.findall((pathlib.Path("docs/en") / f"{name}.md").read_text(encoding="utf-8"))
    zh = BLOCK.findall((pathlib.Path("docs/zh") / f"{name}.md").read_text(encoding="utf-8"))
    same = sum(a == b for a, b in zip(en, zh))
    ok = len(en) == len(zh) and same == len(en)
    bad += 0 if ok else 1
    print(f"{name}: en={len(en)} zh={len(zh)} same={same} {'OK' if ok else 'MISMATCH'}")
print("pages with mismatch:", bad)
sys.exit(1 if bad else 0)
PY
```

2026-11-08 实测输出（当前基线，块数变化时只更新期望值，不放松判据）：

```text
overview: en=4 zh=4 same=4 OK
usage: en=11 zh=11 same=11 OK
performance: en=1 zh=1 same=1 OK
install: en=7 zh=7 same=7 OK
pages with mismatch: 0
```

## 构建门

```bash
python -m sphinx -b html -d /tmp/x/doctrees -W docs/en /tmp/x/html
python -m sphinx -b html -d /tmp/x/doctrees -W docs/zh /tmp/x/html-zh
```

`-W` 把告警变成错误，所以 exit 0 才可发布；两条都必须 `build succeeded.`。
中文构建若与英文共用 doctree 目录仍须 exit 0，但建议中文另用
`-d /tmp/x/doctrees-zh` 避免缓存串扰。2026-11-08 实测：两条均 exit 0。

新增 Sphinx 扩展或主题时，必须同时写进 `docs/en/requirements.txt` 和
`docs/zh/requirements.txt`——RTD 只安装各自 `requirements.txt` 里的依赖。

## 常见错误与验收门

| 现象 | 处理 |
|---|---|
| RTD 报找不到配置文件 | 文件名必须是 `.readthedocs.yaml`，且被 `.gitignore` 放行并已提交 |
| 英文站/中文站串页 | 各自只索引本站源目录；不要用 `docs/` 根做源目录 |
| 新增页 404 或不在导航 | 两份 `index.rst` 的 toctree 都补上 |
| 中英代码块内容不同 | 双语一致性门拦下，按英文块原样同步 |
| 文档写了仓库路径或 `pyqcd.testing` | 违反视角规则；基准/回归命令必须声明“需源码仓库” |
| 本地 `-W` 失败 | 先修告警（未引用文档、重复标签、缺 toctree 条目），不放宽 `-W` |
| 推送后站点未更新 | RTD 由远端触发，两个远端都要推 |
| 站上还是旧版安装说明 | README/安装说明改动需随新版本发布才进 PyPI 项目页 |

验收门：一致性脚本 exit 0、两份构建 exit 0、两份 `.readthedocs.yaml` 已跟踪、
新增页在两份 toctree、两个站点 URL 可访问且内容与源码一致。

## 与 PyPI 的联动

- PyPI 项目页描述来自根 `README.md`；`README.zh-CN.md` 进 sdist 但**不进**项目页。
- 因此 README 改动只有发新版本后才会体现在 PyPI 上；发版流程交给 `pypi-release`。
- 文档改动同样走 `gpush.sh` 和 `gpush.sh -t gitee.com`。

## 交接

- 发版、sdist 内容、venv 自检：`pypi-release`。
- 报告、PDF 与 LaTeX 版式：`pyqcd-docs`；仓库目录与命名：`form`。
- 站点示例涉及的 API 行为：交给对应 `pyqcd-*` 技能核实，不在文档里自造结论。
