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

### 两套站点的结构：`docs/en` + `docs/zh`，同一 slug `py-qcd`

```text
docs/en/                                     docs/zh/
├── .readthedocs.yaml                        ├── .readthedocs.yaml
├── conf.py           （language 默认 en）    ├── conf.py          （language = "zh_CN"）
├── index.rst         ← 唯一 RST，挂 toctree  ├── index.rst        ← 同一条 toctree
├── overview.md  usage.md                    ├── overview.md  usage.md
├── performance.md  install.md               ├── performance.md  install.md
└── requirements.txt                         └── requirements.txt
```

- 一个 RTD 项目 slug `py-qcd`，两个语言版本：`en` → `https://py-qcd.readthedocs.io/en/latest/`，
  `zh-cn` → `https://py-qcd.readthedocs.io/zh-cn/latest/`。三个写法别混：URL 语言码是 `zh-cn`（连字符），
  `docs/zh/conf.py` 里是 `language = "zh_CN"`（下划线），本地目录是 `docs/zh/`。
- 每侧只有一页 RST：`index.rst`（标题 + `pip install pyqcd` + toctree + “从哪里开始”）。四个
  MyST 页 `overview` / `usage` / `performance` / `install` 两侧同名。`conf.py` 里
  `extensions = ["myst_parser"]`、`source_suffix = {".rst": "restructuredtext", ".md": "markdown"}`
  让 `index.rst` 直接索引 Markdown；不要给每页再套 `.rst` 壳。
- `conf.py` 不 `import pyqcd`，`exclude_patterns = ["_build"]`；源目录就是 `docs/en/`，`docs/` 下的研究报告天然在站外。
- 新增一页的落地清单（缺一条 = 404 或漏检）：① `docs/en/<slug>.md` 与 `docs/zh/<slug>.md` 同名建好；
  ② 两份 `index.rst` 的 toctree **各加一行** `<slug>`；③ `pyqcd/testing/docs_examples.py` 的
  `PAGES` 常量加 `<slug>`；④ 三道一致性门全绿（见下节）。
- toctree 用脚本判，不靠肉眼：

```bash
python - <<'PY'
import pathlib, re

def toc(path):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    body = re.search(r"\.\. toctree::\n(?:[ \t]*:[^\n]+\n)*\n((?:[ \t]+\S+\n)+)", text).group(1)
    return [line.strip() for line in body.strip().splitlines()]

en, zh = toc("docs/en/index.rst"), toc("docs/zh/index.rst")
disk = {lang: {p.stem for p in pathlib.Path(f"docs/{lang}").glob("*.md")}
        for lang in ("en", "zh")}
print("en:", en); print("zh:", zh); print("disk:", disk)
assert en == zh, "两份 toctree 顺序/内容不一致"
assert set(en) == disk["en"] == disk["zh"], "有页面没进 toctree，或 toctree 指向不存在的页"
PY
```

判据：exit 0。2026-11-08 实测 `en == zh == ['overview', 'usage', 'performance', 'install']`，
与磁盘文件集合一致（顺序按导航比，集合按文件名比，不要用 `sorted()` 去比顺序）。

### `performance.md` 的章节骨架（数字页唯一合法顺序）

顺序固定、中英同步：**生产尺度表 → 小尺寸表 → 走哪条路径 → 增益来源 → 内存 → 正确性 →
复现脚本 → 局限**。

| # | 英文标题 | 中文标题 | 内容判据 |
|---|---|---|---|
| 1 | `Production-scale results` | `生产尺度结果` | 单次 0.4–2.7 s 的用例，CPU/GPU 分表，每行 `pyqcd` / `opt_einsum` / 倍率 |
| 2 | `Small-shape results` | `小尺寸结果` | 必须带“只用于定位、不要外推”的限制句 |
| 3 | `What runs where` | `什么情况走哪条路径` | 输入类型 → 执行路径 → 备注，含 `[kernel: …]` / `[plan-cache]` 说明 |
| 4 | `Where the gains come from` | `增益从哪来` | 编号列表，每条对应上表某一列 |
| 5 | `Memory` | `内存` | 峰值 RSS，独立进程测量；异常项（当前批量 8×512² 52 MB）必须点名 |
| 6 | `Correctness` | `正确性` | 三重验证 + 一段可执行自验证脚本 + 不可破坏的不变量 |
| 7 | `Reproduce` | `复现` | 与表一一对应的脚本，`CASES` 标签与表行同名 |
| 8 | 局限 | 局限 | 当前形式是生产表后的 “Two honest notes / 两点如实说明”：哪些倍率掉到持平、为什么 |

```bash
python - <<'PY'
import pathlib
WANT = {
    "en": ("Production-scale results", "Small-shape results", "What runs where",
           "Where the gains come from", "Memory", "Correctness", "Reproduce"),
    "zh": ("生产尺度结果", "小尺寸结果", "什么情况走哪条路径", "增益从哪来",
           "内存", "正确性", "复现"),
}
for lang, want in WANT.items():
    heads = [l[3:].strip() for l in pathlib.Path(f"docs/{lang}/performance.md")
             .read_text(encoding="utf-8").splitlines() if l.startswith("## ")]
    pos = [heads.index(w) if w in heads else -1 for w in want]
    print(lang, "ORDER OK" if pos == sorted(pos) and -1 not in pos else "BROKEN", pos)
PY
```

硬规则：表里每个数字必须由**同页脚本**产出，改尺寸就重跑并同时改表；复现不出来的数字只有两条出路——
删除，或标注“历史数据 / 不可复现 + 原测量条件”；禁止“约、大概、估计”；“持平”写成 `1.01×` / `0.98×`。

## 最小工作流程

1. 改 `docs/en/` 的页面；示例先按已安装包的口吻写。
2. 同步 `docs/zh/` 对应段落，**围栏代码块逐字节复制**，不重译代码。
3. 跑双语一致性门（下节脚本），必须 `0` mismatch。
4. 新增页时补两份 `index.rst` 的 toctree。
5. 两份 `-W` 构建都必须 exit 0。
6. `gpush.sh` 与 `gpush.sh -t gitee.com` 双推，Read the Docs 才会重建。
7. 复验 `https://py-qcd.readthedocs.io/en/latest/` 与 `/zh-cn/latest/` 的新页面。

### 三道一致性门：命令、判据、一次跑完

| # | 门 | 命令 | 通过判据 |
|---|---|---|---|
| ① | 中英围栏代码块逐字节相同 | “双语一致性门”脚本 | `pages with mismatch: 0`，exit 0 |
| ② | 双站 Sphinx 无告警构建 | `python -m sphinx -b html -d /tmp/x/doctrees -W docs/en /tmp/x/html`（中文把 `docs/en`→`docs/zh`、doctree 换 `-d /tmp/x/doctrees-zh`） | 两条都 `build succeeded.` 且 exit 0 |
| ③ | 文档示例可执行且自验证 | `python -m pyqcd.testing.docs_examples --keep-going`，再跑 `--lang zh --keep-going` | `failed=0`，且 `ok + skipped` = 页面 `python` 块总数（`skipped` 只能来自标了 `# pseudo-code` / `# docs-skip` 的块） |

一次跑完（`-euo pipefail` 保证任何一环失败即停；把输出贴进提交说明）：

```bash
set -euo pipefail
python -m sphinx -b html -d /tmp/x/doctrees    -W docs/en /tmp/x/html    | tail -2
python -m sphinx -b html -d /tmp/x/doctrees-zh -W docs/zh /tmp/x/html-zh | tail -2
python -m pyqcd.testing.docs_examples --keep-going | tail -3
python -m pyqcd.testing.docs_examples --lang zh --keep-going | tail -3
```

- 门③ 的耗时取决于 `performance.md` 的基准块是否被 `# docs-skip` 豁免：现状两块都标了（门禁跳过，整页秒级结束）；
  去掉标记后门禁会真的跑 4600²/16000² 收缩（单次 0.5–2.7 s × 3 次 × 两种实现），单页十分钟量级，且两个语言
  版本**必须串行**——并发会把 16000² 的两个 2.05 GB 数组同时压进内存，实测被内核 `Killed`（exit 137）。
- 块之间共享一个命名空间、按 `PAGES` 顺序执行（overview → usage → performance → install）。

2026-11-08 实测（`pyqcd 0.2.0`，Sphinx 9.1.1）：门① 输出见“双语一致性门”一节的基线块
（同一次运行，`pages with mismatch: 0`）；门③ 两侧输出一致：

```text
en/overview     blocks ok=3  skipped=0 -> OK
en/usage        blocks ok=11 skipped=1 -> OK
en/performance  blocks ok=1  skipped=2 -> OK
en/install      blocks ok=4  skipped=0 -> OK
total: ok=19 skipped=3 failed=0
```

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
usage: en=12 zh=12 same=12 OK
performance: en=4 zh=4 same=4 OK
install: en=14 zh=14 same=14 OK
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

## 文档示例可执行门

文档里的 Python 示例必须能跑，否则"文档即真相"就是假的。仓库自带校验器：

```bash
python -m pyqcd.testing.docs_examples --keep-going
python -m pyqcd.testing.docs_examples --lang zh --keep-going
```

规则：
1. 每个 ```` ```python ```` 块会在同一命名空间里按页面顺序执行；
2. 需要源码仓库或有意简写的片段，块内必须出现 `# pseudo-code` 或 `# docs-skip`；
3. 中英文的块**逐字节相同**（见"双语一致性门"），所以只需保证英文块可执行；
4. 当前基线 en 22 / zh 22 个 ```` ```python ```` 块（overview 3 / usage 12 / performance 3 /
   install 4），其中 19 执行 / 3 跳过（usage 1 + performance 2）/ 0 失败；新增页面要同步加入 `PAGES` 常量。

实测价值：0.1.5 之前 `usage.md` 有两处示例签名与实现不符
（`compute_dual_field_strength(U)` 实际是 `(F_dict, mu, nu)`；
`Bootstrap(data, n_samples=200)` 实际是 `(data, Nconf_axes=0, ...)`），
正是这个门把它们暴露出来的。

## 文档 API 一致性门（新增）

除了"代码块能跑"，还要保证**引用的符号真实存在**。仓库提供 AST 检查器：

```bash
python -m pyqcd.testing._docs_api_contract   # 或 python -m pyqcd.testing 里的 test_docs_api_contracts
```

它解析 `docs/en/*.md` 的每个 ```` ```python ```` 块，收集
`import pyqcd.x as y` 与 `from pyqcd.x import n`，再对块内所有 `y.attr` / 导入名做
`hasattr` 校验；`pyqcd.testing` 因不随 PyPI 发布而列入白名单。
当前基线：35 个引用、0 失败，并断言"至少 20 个引用"以防解析器退化成空转。

同时验证文档里的**命令**（bash 块不会被 Python 执行器覆盖）：

```bash
python -m pyqcd.testing.einsum.bench_native --help
python -m pyqcd.testing.einsum.verify_compat --help
python -m pyqcd.testing.einsum.verify_native --help
python -m pyqcd.testing.einsum.bench_native --data synthetic --profiles T1 \
  --backends cpu --iters 2 --warmup 1 --planner shared --smoke
```

注意 bench 会自己打印"阻断"（too_short / oracle 非冻结 / 代理数据），
这是设计如此，不是失败；文档里引用这些命令时应保留其产物路径说明。

### 文档示例必须"自验证"，不是"能跑就行"
只检查"代码块不报错"毫无价值：`print()` 出来的错数照样通过。规则（0.1.21 起强制）：

* 每个 ```` ```python ```` 块**必须含 `assert`**，验证自己的结果（对 numpy / torch 的
  独立参考、字段完整性、形状或不变量）；
* 基准脚本要**先验数再计时**，否则测的是错误的数字；
* 确实无法运行的片段必须写 `# pseudo-code` 或 `# docs-skip`；
* `docs_examples` 遇到无断言的块直接判失败，消息里给出修复方法。

补断言时踩过的三个坑（都是断言写错，不是库错）：
`count_cached_ops(cache)` 返回的是 **Counter**（用 `sum(ops.values())`）；
`plan_parallel()` 返回 **dict**（用 `{"N","Y","X","m"} <= set(plan)`）；
cupy 结果不能 `np.asarray`，要 `cp.asnumpy(...)`。

### “无断言即失败”的实现判据与三条真实回归

`pyqcd/testing/docs_examples.py::run_page` 的判据是硬编码的：块内既没有 `assert `、也没有
`# pseudo-code` / `# docs-skip` → 直接记一条失败
`no assertion — every example must verify its own result`，**该块根本不执行**。
所以“能跑通”不等于通过，`print()` 出来的错数也永远不会暴露。

断言必须对**独立参考**，不能对自己：数值用 `np.einsum(..., optimize="optimal")` 与
`torch.einsum(...)` 双参考 + `np.allclose(rtol=1e-10, atol=1e-12)`；结构用
`{"backend", "native_available", "array_backend", "mpi"} <= set(info)`；形状/不变量用
`shape == (2, 2, 2, 2, 3, 3)`、`np.all(np.isfinite(...))`；返回类型要贴合真实对象
（`count_cached_ops` → `Counter`，用 `sum(ops.values())`；`plan_parallel` → `dict`，
用 `{"N", "Y", "X", "m"} <= set(plan)`）。

三条真实回归（文档写了、实现没有；都靠这个门在 0.1.5 / 0.1.21 当场抓住）：

| 文档原来写的 | 真实签名 / 返回 | 改法（可直接照抄） |
|---|---|---|
| `compute_dual_field_strength(U)` | `(F_dict, mu, nu)`；`F` 由 `clover_field_strength(U, mu, nu)` 给出 | `F = {(mu, nu): G.clover_field_strength(U, mu, nu) for mu in range(4) for nu in range(4) if mu != nu}`，再 `assert O.compute_dual_field_strength(F, 0, 1).shape == (2, 2, 2, 2, 3, 3)` |
| `Bootstrap(data, n_samples=200)` | `Bootstrap(data, Nconf_axes=0, ...)`；没有 `n_samples`，返回 dict | `samples = A.Bootstrap(data, Nconf_axes=0)`；`assert {"data_mean", "data_err", "data_sample"} <= set(samples)`；`assert np.shape(samples["data_mean"])[0] == 5` |
| `count_cached_ops(cache)` 被写成无参调用 | 必须传 `cache`，返回 `Counter` | `with pe.shared_intermediates() as cache:` 里连做两次 `pe.contract(...)`，然后 `assert sum(pe.count_cached_ops(cache).values()) >= 1` |

教训：签名、参数名、返回类型的错误只有“真跑 + 真断言”会暴露；纯 `print()` 的页面看起来一切正常。

### 基准脚本规范（`performance.md` 的计时块必须全满足）

| # | 规则 | 判据 |
|---|---|---|
| 1 | 交错测量 | 同一轮里 `a = one(ours); b = one(theirs)`，不是“先测完 A 再测 B” |
| 2 | 单次 0.5–2.7 s | 用例尺寸按此选（当前生产表 0.44–2.67 s）；更短的运行测到的是计时分辨率与缓存预热 |
| 3 | 3 次重复取中位数 | `rounds=3`，时间与比值都过 `statistics.median(...)` |
| 4 | 先断言后计时 | 进 `compare()` 之前 `assert np.allclose(ours, np.einsum(..., optimize="optimal"))`；断言失败就不产生时间 |
| 5 | 打印被测构建 | 开头 `print("measuring pyqcd", pyqcd.__version__, pyqcd.__file__)`：`site-packages` 里的旧 `pyqcd` 会把所有数字变成“PyQCD 很慢”，缺这两行事后无法判断测的是哪份 |
| 6 | GPU 同步计时 | 每次调用前后 `cp.cuda.Device(0).synchronize()`；比较前 `cp.asnumpy(...)`，不要 `np.asarray(cupy_array)` |

```bash
python - <<'PY'
import pathlib, re
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
KEYS = ("assert ", "pyqcd.__version__", "pyqcd.__file__", "rounds=3", "statistics.median")
for path in ("docs/en/performance.md", "docs/zh/performance.md"):
    for i, block in enumerate(BLOCK.findall(pathlib.Path(path).read_text(encoding="utf-8"))):
        if "perf_counter" not in block:
            continue
        missing = [k for k in KEYS if k not in block]
        print(f"{path}#{i}: {'OK' if not missing else 'MISSING ' + ','.join(missing)}")
PY
```

判据：每个计时块都 `OK`。2026-11-08 实测：`#1`（CPU 生产尺度）`OK`；`#2`（GPU 生产尺度）报
`MISSING pyqcd.__file__` —— 该块只打印了版本与设备名，属于必须先补的缺口（改
`performance.md` 时加一行 `print(pyqcd.__file__)`），不是门的例外。

### RTD 构建与发布联动：配置放行 → 推送 → 重建探针

- 两份 `.readthedocs.yaml` 必须**被 `.gitignore` 放行且已提交**：根 `.gitignore` 用 `.*` 忽略隐藏文件，
  靠 `!docs/en/.readthedocs.yaml`、`!docs/zh/.readthedocs.yaml` 豁免；未跟踪 = 线上没配置 = 构建失败。

```bash
git ls-files docs/en/.readthedocs.yaml docs/zh/.readthedocs.yaml   # 两行都必须有输出
git check-ignore -v docs/en/.readthedocs.yaml docs/zh/.readthedocs.yaml; echo "exit=$? (1 且无输出 = 已放行)"
```

- 两份配置各自写 `sphinx.configuration: docs/<lang>/conf.py` 与 `python.install.requirements: docs/<lang>/requirements.txt`；
  RTD 的 “Configuration file” 要指到各自文件；`formats: [pdf, epub]` 会在 HTML 之外再跑 LaTeX/EPUB，PDF 失败也计构建失败。
- `conf.py` 的 `version`/`release` 只决定页脚与 `<title>`（`PyQCD <release> documentation`），与 PyPI 版本无自动关联：改版本时 `conf.py` 的 `release` 与 `install.md` 的发行号（en `Current release:` / zh `当前发行版：`）必须一起改，现在两处都是 `0.2.0`。
- 推送后重建有延迟（分钟级），且**不同页面可能停在不同版本**——按页探针，别看完首页就下结论：

```bash
for lang in en/latest zh-cn/latest; do
  curl -s "https://py-qcd.readthedocs.io/$lang/install.html" | grep -o '<title>[^<]*</title>'
  curl -s "https://py-qcd.readthedocs.io/$lang/performance.html" | grep -c -e 'Production-scale results' -e '生产尺度结果'
done
```

判据：探针里的版本号/章节标题必须与刚提交的源文一致；不一致 = 还没重建完（等 5–10 分钟再探），不是构建失败。
2026-11-08 实测：两站 `install.html` 的 `<title>` 已是 `0.1.29`，而 `performance.html` 仍是旧结构（导航 `Measure it yourself` / `自己动手测`，新版已改名 `Reproduce` / `复现`）——同站不同页可差一个版本；`README.zh-CN.md` 只进 sdist 不进 PyPI 项目页（项目页描述取自根 `README.md`），文档改动要随版本才生效。
