# PyPI 发布硬化台账（2026-11-08 起）

> **⚠️ 勘误（2026-11-19）**：本页记录的所有 **GPU 耗时 / GPU 倍率 / GPU 对照**均出自 0.5.9 之前的实现，那时计时**没有设备同步**（`cupy`/`torch` 异步下发，量到的是入队时间）、`--compare opt_einsum` 也只拿 numpy 副本当参照。这些数字**不得再作为证据引用**，详见 [`docs/ERRATA_gpu_timing_20261119.md`](./ERRATA_gpu_timing_20261119.md)。CPU 计时、数值结论与形状/dtype/命中结论不受影响。

> 判定口径：**一律以 `pip install pyqcd==<ver>` 从 PyPI 远程安装到全新 venv 的结果为准**，
> 本地 sdist、源码树里的 `import pyqcd` 都不算证据。
> 约束：不修改核心代码（接口级 debug 允许）。经验实时回写 `skills/pypi-release/` 与
> `skills/readthedocs/`。

## 验收清单

### A. 规范性（Packaging compliance）
| # | 检查项 | 判定方法 | 当前状态 |
|---|---|---|---|
| A1 | 包名/版本/摘要/许可证/Python 版本元数据正确 | PyPI JSON API 字段核对 | 0.1.4 ✅ |
| A2 | 只发布 sdist（wheel 标签对 `.so` 不成立） | PyPI JSON 的 `urls` 只有 `.tar.gz` | 0.1.4 ✅ |
| A3 | sdist 无 `pyqcd/testing`、无 `.so`、无绝对路径条目 | 解包 `tar -tzf` 审计 | 0.1.4 ✅ |
| A4 | 所有声明的子包都在 sdist 内 | `find_packages` 列表 vs tar 列表 | 0.1.4 ✅（12 个） |
| A5 | 依赖声明完整（无隐性 import 失败） | 全新 venv 里逐个 `import pyqcd.<sub>` | 0.1.4 ✅（12/12） |
| A6 | 无可选重依赖被硬编入（torch/matplotlib 按需） | `find_spec` 断言 + 子包 import | 0.1.4 ✅ |
| A7 | `twine check` 全绿；README 渲染为 Markdown | twine + PyPI `description_content_type` | 0.1.4 ✅ |
| A8 | PEP 561 类型信息 | `py.typed` 在包内 + `Typing :: Typed` 声明 | ✅ 0.1.5 起随包发布（R2/R14 复核） |
| A9 | 项目 URL 指向真实可达平台 | 三个托管地址 200 | 0.1.4 ✅ |
| A10 | 无 setuptools 弃用告警 | 构建日志 grep `deprecat` | ✅ PEP 639 修完（R4）；仅剩 `global-exclude` 无害告警（R4 记录） |

### B. 易用性（First-run usability）
| # | 检查项 | 判定方法 | 当前状态 |
|---|---|---|---|
| B1 | 安装即用：一条命令、无额外手动步骤 | 全新 venv `pip install pyqcd` | ✅ |
| B2 | `import pyqcd` 后的可发现性 | `dir(pyqcd)` + PEP 562 懒加载 | ✅ `__all__`/`__getattr__`/`__dir__` + `python -m pyqcd`（R1/R2） |
| B3 | 文档示例在**已安装包**上可直接跑通 | `python -m pyqcd.testing.docs_examples` | ✅ 16 块 0 失败（R2/R9）；API 引用 AST 校验 35 项 0 失败（R9） |
| B4 | 错误信息可行动 | 触发 ImportError 检查文案 | ✅ torch/plot 均给出 `pip install "pyqcd[...]"`（R2）；6 类非法输入与 opt_einsum 逐字一致（R12） |
| B5 | `pyqcd.einsum` 与 `opt_einsum` API 对等 | 差分套件 + 边界矩阵 | ✅ 49 项行为 MATCH；差异仅两处增量（callable planner、顶层 PathInfo，R19） |
| B6 | 基准/测试命令明确标注"需源码仓库" | 文档措辞 | ✅ |

### C. 生产力价值（Productivity）
| # | 检查项 | 判定方法 | 当前状态 |
|---|---|---|---|
| C1 | 无需源码即可完成一次真实收缩 | 纯安装环境跑 E1 方程 + 正确性 | ✅ rel 2.4e-15（R5/C 组）；可复现表 6.02×（R7） |
| C2 | 与 `opt_einsum` 可互换 | 双实现同脚本对照 | ✅ `opt_cost`/路径逐值一致（C 组 + R19） |
| C3 | 性能优势可在安装环境复现 | 发布包上 A/B 计时 | ✅ 默认对 `SPECIALIZE=0` 快 1.3×/7.5×/6.3×（R5），对 opt_einsum 6.02×/5.83×/2.91×（R7） |
| C4 | 物理组件可组合 | 端到端小例子 | ✅ clover→dual→γ→夹心 全链通（R14 文档示例实跑）；`compute_dual_field_strength` 错误已可行动（R6） |

## 迭代记录

| 轮次 | 时间 | 版本 | 发现 | 处置 |
|---|---|---|---|---|
| R0 | 11:45 | 0.1.4 | 建台账；A5/A6 已由 0.1.4 修复（h5py/scipy 声明、torch/matplotlib 延迟） | — |
| R1 | 12:05 | 0.1.4（待 0.1.5） | B2 可发现性差（`__init__` 无 docstring/`__all__`，注释里子包列表过期）；A8 `Typing :: Typed` 但无 `py.typed` | 已加模块 docstring + `__all__`（12 子包）+ `pyqcd/py.typed` + `package_data`；新增 `python -m pyqcd` 自检入口并写入中英文 install 文档 |
| R2 | 12:40 | 0.1.5 | A组 9 条（SOURCES 泄漏家目录、py.typed 漏打包、license 弃用、classifier 缺 3.13/3.14、plot 报错不可行动、AGENTS.md 进 sdist…）；B组 9 条（安装不 hermetic、usage 两处 API 签名错、torch/matplotlib 裸 ModuleNotFoundError、顶层零可发现性、混合精度无处方、native_library 指向不存在文件）；C组终止前发现 `cached_contract` 依赖未声明的 opt_einsum | 0.1.5 已修：cythonize 相对路径；`py.typed`+package_data+MANIFEST；PEP 639 `license`+`license-files`+setuptools>=77；3.13/3.14 classifier；`pyqcd[plot]`/`[torch]` 可行动报错；PEP 562 `__getattr__/__dir__`；`cached_contract` 回退自带引擎；`build_info().native_available`；主机编译器试编译失败只跳过 CUDA；wickplot 必需键校验；MANIFEST `prune pyqcd/testing`；usage.md 三处代码块改为真实签名并通过顺序执行 |
| R3 | 13:10 | 0.1.5 | 0.1.5 从 PyPI 远程复验通过：12/12 子包 import（无 torch/matplotlib）、`pyqcd.einsum` 懒加载、`cached_contract` 无 opt_einsum、`py.typed` 就位、torch/plot 报错可行动；Python 3.14 用 `setup.py build` + `PYTHONPATH` 实测 `import pyqcd` 与 `python -m pyqcd` 正常（该环境无 numpy，后端如实报 unknown） | 3.13/3.14 classifier 保留（3.14 已验证构建与导入）；README 补 `python -m pyqcd` |
| R4 | 13:25 | 0.1.5 | S1 复测通过：`CC=/bin/false CXX=/bin/false NVCC=/bin/false` 下 `pip install pyqcd==0.1.5` 退出码 0，`backend='python'`、`native_available=False`、12/12 子包与 contract 正常；A10 复检：license 弃用告警已消失、SOURCES 绝对路径 0；新增文档示例门 | 新增 `pyqcd/testing/docs_examples.py`（en/zh 各 16 块、0 失败）；MANIFEST 精简（`global-exclude` 的 4 条无害告警保留，已在技能中说明不要为消警告删安全排除） |
| R5 | 14:20 | 0.1.6 | C 组三条：①「专用路径小规模慢 10.9×」经复核为**测量假象**（`specialize=False` 连计划缓存一起关；且合成形状从不命中 kernel）；②E4/E5 在 REGISTRY 里标 `fused` 并打印不存在的 CUDA kernel → 改为 `plan-cache` 并据实打印 `[kernel: …]`/`[plan-cache]`；③`build_info()` 只报 `backend` 误导（numpy 输入其实走 KUPL） → 新增 `array_backend` | 先加再撤规模门（实测默认从不变差：nvec 8/16/24 上默认比 `SPECIALIZE=0` 快 1.3×/7.5×/6.3×）；0.1.6 已发布并从 PyPI 复验：12/12 子包、`[kernel: kupl_nucleon_2pt]` 标签正确、rel 3.5e-15/1.7e-14 |
| R6 | 15:05 | 0.1.7 | C 组遗留接口缺陷：`compute_dual_field_strength` 缺分量静默返回 `None`、传 ndarray 抛无信息量 `AttributeError`；缺元数据一致性回归测试 | 0.1.7：`TypeError`/`ValueError` 带处方；新增 `pyqcd/testing/_release_metadata_contract.py`（版本三处一致、`__all__`↔磁盘、子包可导入、MANIFEST 策略、`build_info` 字段）并接入 `conftest`，`python -m pyqcd.testing` 115→**116 passed**；PyPI 远程复验 12/12 子包、两条错误文案正确 |
| R7 | 16:05 | 0.1.8 | ①文档性能表不可复现（E4/E5 的 8.33×/3.371× 依赖发行版里没有的融合 kernel）；②`pyqcd[cuda]` 指向源码包 `cupy`，安装必然失败；③`/tmp` tmpfs 被历史 venv 撑满导致解析失败被误判 | 0.1.8：性能表改为"发布即可复现"三行（E1 6.02×/5.83×、E5 2.91×）+ 历史数字单独标注不可复现；`cuda` extra → `cupy-cuda12x`；清理 /tmp（100%→49%）并把 `df -h /tmp` 作为失败排查第一步；PyPI 复验三个 extras 全部 `Would install` 通过 `[作废]` |
| R8 | 16:55 | 0.1.9 | ①`pyqcd[plot]` 只验证过"能解析"，未验证可用；②文档未说明"为何无 wheel / 首次安装要编译"；③A9 五个 URL 复检 | 0.1.9：干净 venv 装 `pyqcd[plot]` 端到端跑通 `wick_contraction → plot_figure_wick`（107 KB PNG，12 键）；install 页（中英）补"无平台 wheel 的理由 + 首次安装约 1 分钟 + pip 缓存"；五个 URL 全部 200；`pyqcd[torch]` 因 539 MB 轮子仅做 dry-run，标注未验证 |
| R9 | 17:40 | 0.1.9（无新发行） | ①文档 API 引用此前只靠"代码块能跑"间接保障，伪代码块不被执行；②文档里的 bash 命令未被任何门禁覆盖；③RTD 站点是否已重建到 0.1.9 未知 | 新增 `pyqcd/testing/_docs_api_contract.py`（AST 解析，35 个引用 0 失败，含"至少 20 引用"防空转）并接入 `conftest`，`python -m pyqcd.testing` 116→**117 passed**；三条文档命令 `--help` + bench `--smoke` 实跑通过；RTD en/zh 已确认重建（含 6.02×、auditwheel、cupy-cuda12x、plan-cache）。本包内容未变，**不发新版本** `[作废]` |
| R10 | 18:35 | 0.1.9（无新发行） | ①`pyqcd[torch]` 仅做过 dry-run，未端到端；②PyPI 项目页无法用 HTML 抓取核对（bot 挑战） | torch extra 端到端通过：torch 2.15.0+cu132、`set_backend('torch')`、torch 张量 contract 与 `a @ b` 一致、numpy 输入仍可用、`gamma(0)` 返回 torch 复数张量、`python -m pyqcd` 正确列出缺失 extras；PyPI 页改用 JSON `info.description` 探针核对（Repositories/三平台/文档/python -m pyqcd/6.02 全部命中）。包内容未变，不发新版本 |
| R11 | 19:30 | 0.1.9（无新发行） | 对抗性复查：dtype/形状/kwargs 边界此前无门禁；`dtype=`/`order=` 报 TypeError 疑似缺陷 | 新增 `_contract_edge_cases_contract.py`（8 组：全 dtype、ellipsis/重复标签/标量/迹、非连续与零尺寸、6 种 optimize、`out=`、expression+shared_intermediates、opt_cost 与 opt_einsum 逐值一致、三类非法输入错误类型）；`python -m pyqcd.testing` 117→**118 passed**。经对照确认 `dtype=`/`order=` 在 **opt_einsum 3.4.0 上同样抛 TypeError**，属 bug-for-bug 兼容，不修 |
| R12 | 20:20 | 0.1.9（无新发行） | ①错误路径消息质量未与 opt_einsum 逐字比对；②`-O/-OO/-B`/只读安装目录未验；③`-OO` 下 `docs_examples` 崩 | 6 类非法输入与 `contract_path` 的异常消息**逐字一致（SAME）**；`none/-O/-OO/-B` 与只读 site-packages 下 contract/opt_cost 全部正常；修 `docs_examples.py` 的 `__doc__` 依赖（-OO 下 `AttributeError`），118 passed 保持 |
| R13 | 21:05 | 0.1.9（无新发行） | ①离线/air-gapped 安装未验证（HPC 常态）；②classifier 声称 3.10–3.14，但本机只有 3.12 与 3.14（且 3.14 无 pip） | 离线安装验证通过：wheelhouse 7 包/60 MB，`PIP_NO_INDEX=1` 下成功构建并安装 0.1.9，装后 `backend=cuda`、contract 正确；文档（中英 install）新增"实测于 3.12 完整套件 / 3.14 构建+导入，3.10/3.11/3.13 未自动验证"的诚实声明；代码块仍 8/8 一致，sphinx -W 双站 0 告警 |
| R14 | 21:50 | 0.1.9（无新发行） | ①0.1.3 曾静默漏发 11 个子包，现行配置是否有同类风险未知；②构建可复现性未测 | 受控探针实验：新增 `pyqcd/_tmp_probe.py` 与 `pyqcd/_tmp_pkg/` 均**自动进入 sdist**，且 `pyqcd._tmp_pkg` 自动进 packages；同时 `test_public_subpackages_exist` 立即报 "only on disk=['_tmp_pkg']"，证明"未登记子包"会被抓住；探针已清理、工作树干净。可复现性：两次 sdist tarball 哈希不同但**解包内容逐字节相同**；加 `SOURCE_DATE_EPOCH` 仍不同（setuptools 84.2.1），故约定用内容哈希判断，不用 tarball sha256 |
| R15 | 22:40 | **0.1.10** | ①双原生后端一致性（kupl vs cuda）未验证；②cupy 设备数组走通用执行器报 `QCD_STATUS_INTERNAL` 回落，文档未说明；③`_specialize` 的 E1 banner **仍在引用已删除的** `PYQCD_EINSUM_KERNEL_MIN_ELEMENTS` | 一致性：4 个方程在 kupl/cuda 两条库路径上**逐位相同（0.00e+00）**，对 numpy 为 3.3e-15/8.3e-16/2.9e-16/0；cupy 实测结果正确但通用执行器回落且只警告一次，已写入"数组类型 × 执行路径"矩阵（中英 install）；banner 文案修正并随 0.1.10 发布，远程复验确认不再出现该变量、rel=3.5e-15 |
| R16 | 23:25 | **0.1.11** | ①`memory_limit` 语义与峰值内存从未验证；②内存差异对生产 OOM 有关键影响但文档未提 | `memory_limit` 四档语义与 opt_einsum 完全一致，`opt_cost`/`largest_intermediate`/`path` 逐值相同；峰值实测 pyqcd 为 opt_einsum 的 **1.4–2×**（1200²:18→41 MB、2000²:73→104 MB），耗时不变；该结论写入 performance 页（中英）与 README 并随 0.1.11 发布（PyPI `info.description` 已含 "1.4–2× the RSS"）；另记录良性回落：`optimize='optimal'`+极小 `memory_limit` 产生 arity=3 步 → 原生只支持一/二元步 → 单次 warning 回落 Python |
| R17 | 24:05 | 0.1.11（无新发行） | 并发安全未验证（嵌入流水线时的常见前提） | 多线程 matmul（8×16×25）、多线程专用路径（6×12×10）、`shared_intermediates` 线程隔离（4 线程各自 cache）、多进程（4×8）全部 0 错误；新增 `_thread_safety_contract.py` 四用例（0.07 s）并接入 conftest，`python -m pyqcd.testing` 118→**119 passed**；记录 `count_cached_ops(cache)` 需传参的坑 |
| R18 | 24:50 | 0.1.11（无新发行） | 终检证据分散在多个命令里，打 tag 时容易漏项 | 新增 `pyqcd/testing/release_check.py`：一条命令验证 16 项（版本三处一致、sdist 存在/twine/内容洁净/无绝对路径、文档示例+中英块一致+sphinx -W 双站、PyPI 版本与页面探针、RTD 双语言含当前版本、gitlab/gitee 与本地 HEAD 同步），失败非 0 退出可直接做 tag 门。实测全量 **16/16 passed**，`--offline --with-tests` **10/10 passed** |
| R19 | 25:35 | 0.1.11（无新发行） | ①`contract_expression` 的 constants/形状语义未与 opt_einsum 对照；②自定义 `optimize` callable 与路径缓存失效未验；③"完全兼容"承诺缺少差异清单 | 实测：`contract_expression(*shapes)` 两边一致；`constants={...}` 两边抛**同样的** `ValueError`（bug-for-bug）；不同 shape 的路径与 cost（240 vs 252）一致；`optimize=` 传 callable 时 **pyqcd 支持而 opt_einsum 3.4.0 抛 TypeError**（超集）。已在 usage 页（中英）新增 "Differences from opt_einsum" 小节，仅列两处增量差异 |
| R20 | 25:20 | 0.1.11 | 台账顶部 A/B/C 清单仍留着 R0 时的"待查/待评"旧状态，与后面 19 轮的结论不一致 | 把 A8/A10/B2/B3/B4/B5/C1/C2/C3/C4 全部改写为已验证状态并标注对应轮次证据，`待查/待评/待核/待复核` 归零；`release_check --with-tests` 全量 **17/17 passed**（含 119 passed 测试套件） |
| R21 | 17:46 | 0.1.11 | 6 小时门槛到达，需终检并打 tag | 终检 **17/17 passed**（含 119 passed 测试套件）；工作树干净；打 annotated tag **`dev34`**（dev 系列 33 → 34）并推送到 gitlab 与 gitee，三端 tag 对象 `cae654c0` 一致；循环终止 |
| R22 | 26:10 | **0.1.13** | 用户问"PyPI 装的 pyqcd 是否支持 CUDA"→ 实测发现：CUDA 库**编译得出来**，但 `make-nv.sh` 默认架构 `6.0 7.0` 只嵌 SASS，在 RTX 4060（sm_89）上 kernel 入口返回 `QCD_STATUS_INTERNAL`，**静默回落 CPU**；文档只在源码构建一节提到架构变量 | 0.1.13：install 页（中英）新增"CUDA 架构必须匹配"专节（默认 7.0 vs 实测对照表、`PYQCD_CUDA_ARCHITECTURES=8.9 pip install` 写法、`strings`/`nvidia-smi` 诊断命令、`[plan-cache]` 判据），README 双版加一行提示；实测 8.9 下 kernel status=0、rel 2.8e-14；PyPI 0.1.13 远程复验通过且页面已含该提示 |
| R23 | 27:00 | **0.1.14** | 用户要求：①嵌入 PTX（`code=[sm_XX,compute_XX]`）；②加入类似 PyQCU `cpp/cuda/qcu/*` 的多环境支持（GPU 自动探测 + 回退表）；③文档里的 `strings` 诊断在新布局下失效 | `make-nv.sh` 改造：架构优先级 `PYQCD_CUDA_ARCHITECTURES`→`CMAKE_CUDA_ARCHITECTURES`→`PYQCD_SM_ARCH`→自动探测（`nvidia-smi compute_cap` + PyQCU 型号回退表）→默认 6.0/7.0；所有架构一律 cubin+PTX。实测：显式 `7.0`+PTX 在 sm_89 上 status=0（JIT 生效），自动探测 8.9 亦 status=0；**`pip install pyqcd==0.1.14` 不设变量即 status=0、rel 2.8e-14**。文档诊断三改为 `pip -v` 的 `arch:` 行 / `readelf nv_fatbin` / 运行期 `[kernel: …]` 提示 |
| R24 | 27:40 | **0.1.18** | 用户追问"为什么 E4/E5 融合 kernel 未进发行版"→ 核查发现**我此前的说法是错的**：`e4_fused.cu`/`sandwich.cu` 确实参与编译并被 `apply_contract.cu` 调用；真正的限制是 ①C++ 体积门 `kE4FusedMaxSpatialVolume=1`（L24 实测慢 33×，故对真实体积主动关闭）②Python 层只有 2pt/3pt 分发，提示恒为 `[plan-cache]` | 0.1.18：`_specialize` 的 E4/E5 文案改为"Python 层无专用 kernel（计划缓存），CUDA 执行器在体积门内会内部融合"；性能页"历史构建"段与连线报告同步更正。实测证据：`z=y=x=1` 时 trace 输出 `PYQCD_EINSUM_CUDA_E4 ... kernel=1`，rel 2.0e-16；`z=1,y=2,x=2` 时无 E4 行、走 GEMM，rel 2.6e-16 |
| R25 | 28:30 | **0.1.21** | 用户要求"文档每段示例都要做正确性验证"→ 审计发现 22 个 python 块里**只有 2 个含 assert**，其余只 print，无法证明正确 | 0.1.21：给 20 个块补断言（收缩类对 numpy/torch 参考；introspection 校验字段；物理类校验形状/不变量；基准脚本先验数再计时），`docs_examples` 新增**"无 assert 即失败"**门禁（`# pseudo-code` 可豁免）；新增「Correctness / 正确性」专节（三重验证 + 自行验证脚本 + 四条不变量）。实测 en 22 块 / zh 22 块全部 0 失败 |
| R26 | 29:40 | **0.1.22** | ①"历史（开发构建）"表全是不可复现数据；②通用 einsum 实例过少；③按调用切档存在 **perf bug**：`ensure_runtime` 的完成标志与 `runtime_applied` 键名不匹配 → **每次 contract 都重设一次线程**，小收缩被拖慢 10×（E5 从 4.9× 变 0.25×） | 0.1.22：删除历史表，改为**可复现**的 CPU/GPU 三条（nvec=100 CPU 9.59×、GPU 夹心 4.54×、GPU E4 1.08×）；新增「通用 einsum 逐例对照」8 个用例 + 自验证脚本（批量 8.9–9.5×、ellipsis 8.7×、外积 1.3–1.5×、Hadamard 0.33×）；修 `_threads` 幂等判断（`runtime_applied` 改规范键 + 最多重试 3 次），每 10 次调用重设次数 10→1，E5 恢复 4.88×/8.32×，E1 6.94×/6.24× |
| R27 | 30:30 | **0.1.23** | 用户要求继续排查 bug。四组针对性猎杀（缓存复用/out=/内存/切档竞态/原生回落）发现 **1 个真缺陷**：`_try_native_contract` 自身抛异常（规划/编译阶段）会直接冒到用户代码，违背"原生失败只回落"的硬不变量 | 0.1.23：`_core_contract` 对该调用加 try/except + 单次警告回落；新增回归用例 `test_native_failure_falls_back_without_raising`（模拟规划期抛错，断言结果正确、警告 ≤1、恢复后仍走原生）。其余 12 项"疑似"均经与 opt_einsum 对照确认一致（out dtype 截断同样只发 ComplexWarning、callable optimize 是 pyqcd 超集、零尺寸冲突 numpy 同样报 ValueError），未改动 |
| R28 | 31:20 | **0.1.24** | 继续排查：路径有效性、显式 path、广播、奇异方程、out 别名、PathInfo 字段、backend/use_blas/memory_limit、以及 700 个随机方程差分 | 全部通过：4 类路径回放可复现、PathInfo 五个字段与 opt_einsum 逐值一致、out 与输入共享内存结果正确、`dp/optimal` 代价一致、随机方程 400（含重复标签/混合 dtype）+ 300（ellipsis + path 回放）**0 失配**。3 处"失败"复测后确认与参照一致（列表输入两边都接受、`contract_expression` 少参两边都 ValueError、`memory_limit=0` 两边同错）。新增 `_random_equations_contract.py`（48+28 例，0.05 s）接入套件，121 passed |
| R29 | 32:10 | 0.1.24（无新发行） | 转向物理模块不变量：gamma 代数、SU(2) CG、Clover、Stout 幺正性、`plan_parallel` 公式 | 实质结论：**无新代码缺陷**。gamma 1..4 满足 `{γi,γj}=2δij`、γ5 与它们反交换、乘积 6..15 与文档定义逐元素一致、16/17 等于 `γ3γ1·(1±γ4)/2`；CG 完备性 Σ|c|²=1 且交换对称；Clover Hermitian/无迹/`F10=-F01`；Stout 涂抹保持 `UU†=1`（1e-10）。5 处"失败"全是我写错不变量（把乘积当 Clifford 生成元、SU2 入参格式、转置轴数、`N*Y` 应为 `N*X`、`N*a=n*b` 只在不封顶时成立）。**修正文档**：usage 页（中英）把 `N*a=n*b` 明确为"未封顶目标"，并说明 Y 会被任务数/CPU/内存压缩、压缩见 `notes` |
| R30 | 33:40 | **0.1.25** | 用户实测 CPU/GPU 都差（matmul 0.41×、sandwich 0.75×）。定位：notebook 从用户数据目录运行，导入的是 **base conda 里的 0.1.12**（`site-packages`），该构建 ①无线程档（CPU 8×8 超订 → 2.6×）②CUDA `.so` 只含 `compute_70` SASS（无 PTX）→ sm_89 上 kernel 返回 **status 3** 静默回落（sandwich 66 ms vs 正常 8 ms）。用 0.1.12 精确复现其数字（28.69/35.78/25.69/106.78 与 kernel status=3） | 0.1.25：①新增 `python -m pyqcd --gpu-probe`（build_info/cupy/设备/算力/kernel 状态/命中提示/结论）②**该跑 kernel 却没跑时立即 RuntimeWarning**（含原因：开关关闭/形状不符/状态非零），同形态同原因只报一次，含 `test_kernel_skip_warns_immediately` 回归 ③文档三个基准脚本均打印 `pyqcd.__version__` 与 `__file__`，并新增「动手之前」小节提示旧安装会伪造“PyQCD 很慢”。新 venv 复跑：0.92× / 8.38× / 0.90× / 8.36× `[作废]` |
| R31 | 34:30 | **0.1.26** | ①`_run_specialized_kernel` 读取失败原因**早于**调用（拿到上一次残留值）；②`probe()` 直接 `_announced.clear()` 破坏调用方提示状态 | 0.1.26：原因改为"失败后读取"并按形态过滤（E1→2pt、P1/P2→3pt，避免把 3pt 的原因报给 E1）；`probe()` 改为保存/恢复提示状态。用户授权后将 **base conda 环境由 0.1.12 升级到 0.1.26**，用其 notebook 原样复跑：CPU 0.90/7.23/0.79/4.59/0.81/0.35/1.11/1.08，CUDA nucleon 1.81×、sandwich **16.15×**、batched 0.94×，`--gpu-probe` 报 `CUDA fast path is active`（此前 0.41×/0.75× 的"差"全部由旧安装导致） |
| R32 | 36:10 | **0.1.28** | 用户要求"通用计算也不应比 opt_einsum 慢"。交错测量暴露真因：①逐元素/置换类方程走分步执行（Hadamard 0.33×）；②KUPL **通用计划执行器**在 2D 链/矩阵乘上比 numpy 分步慢 6–10%，但在 batch/ellipsis 上是 5–10× 优势 | 0.1.28：①新增快速路径——**无求和标签**的方程直接交 backend 原生 einsum（Hadamard 89→30 ms，0.33×→1.00×）；②KUPL 计划执行器改为**按 batch 结构启用**（任一操作数 ndim≥3 才走，`PYQCD_EINSUM_KUPL_PLAN=1` 可强制），CUDA 不受影响。base env 交错复测：batch **10.6×**、ellipsis **5.8×**、chain 1.03×、complex 1.03×、hadamard 1.00×、outer 1.00×、4-op 1.00×、matmul 0.97×；文档脚本改为交错协议并附回归用例 |
| R33 | 38:40 | **0.1.29** | 用户要求"单次 1–10 s、重复 3–5 次"的生产尺度基准。大尺寸暴露新缺陷：**cupy 批量 GEMM 16×1024² 只有 0.21×**（原生 batched GEMM 比 cuBLAS 慢 5 倍）；γ 夹心在 8192×1024 降到 1.6×（2.1 GB 工作区带宽限制）；核子 2pt nvec=112 降到 0.99×（cuBLAS 主导） | 0.1.29：cupy 输入且**无融合 kernel** 的方程在规划前直接交给 `cupy.einsum`（走 cuBLAS），融合形态仍走原生执行器；委托统一带 `optimize=True`。base env 生产尺度复验：CPU batched **11.6×**、ellipsis **7.1×**、Hadamard 1.26×、matmul 1.01×、chain 0.98×、complex 0.96×；GPU sandwich **2.0×**、batched 0.98×、nucleon 0.99×（后两者如实标注为持平/无余量）。文档两张表与基准脚本改为生产尺度（单次 0.4–2.7 s、3 次重复、交错协议） `[作废]` |
| R34 | 41:20 | **0.2.0** | 收尾审查：按用户要求只用小尺寸、低资源方式审 qcd-einsum；同时把文档改为生产尺度并清掉过时内容 | ①**真缺陷**：`cupy`/`torch` 的 `einsum` 没有 `out=` 参数，`out=` 直接抛 `TypeError`（`opt_einsum` 却支持）→ 在 `backends/dispatch._wrap_einsum` 里按同语义回填（算完写回，形状不符报 `ValueError`）；②测试健壮性：`test_setup_scenario_completes_within_bounded_wait` 写死 15 s，负载下必挂 → 改用模块默认 30 s；③文档：`performance.md` 整页重写（生产尺度 0.4–2.7 s × 3 次交错的 CPU/GPU 表、小尺寸表、"走哪条路径"表、增益来源、内存、正确性、复现脚本、局限），两处基准脚本标 `# docs-skip`；README 双语换生产尺度表并纠正"只有 2pt/3pt 有 kernel"的旧说法；版本号全面升 0.2.0 | 25 项语义边界 + 错误类型一致性全部通过；文档门禁 19 ok/3 skipped/0 failed；测试 **121 passed / 1 skipped / 0 failed**；base env 装 0.2.0 后 `cupy out=`、`torch out=`、批量、逐元素、`--gpu-probe`（CUDA fast path active）全部通过 `[作废]` |
| R35 | 45:10 | **0.2.1** | 用户指出"不应只有 8 个 case"，并要求解释历史上 10–48× 的来源 | ①case 扩到 **15 个结构类别**（两操作数实/复、逐元素、外积、全/部分约简、3/4 操作数链、hyper-edge、批量 GEMM、ellipsis、批×矩阵、大量小批量、两个 kernel 形态），CPU/GPU 用**同一批 case 与同一尺寸**；②解释：核子 2pt 的 18.6× 来自参考实现的 7 步路径与 89.9 M 元素（1.44 GB）中间体，批量 GEMM 的 11.9× 来自参考把 batch 轴逐批执行（2.99 s vs numpy.matmul 0.28 s），夹心的 7.8× 来自 537 MB 中间体；③**真缺陷修复**：规划器在等价代价上顺序不同，`ij,jk,ik->i` 实测 0.55× → `auto` 对 ≤8 算子改用 opt_einsum 的 dp 路径（不可用自动回退），现 1.05×；④文档删除"视图类"用例（`ij->ji`/`ii->i` 返回视图，微秒级无意义），换成有工作量的约简 | 15 例 CPU/GPU 全量实测（CPU 0.93–18.6×、GPU 0.96–4.04×，通用形态 0.93–1.15×）；测试 121 passed/1 skipped/0 failed；docs_examples 19 ok/2 skipped/0 failed；sphinx -W 双站 0 `[作废]` |
| R36 | 22:10 | **0.2.3** | 用户要求"最小改动、发布支持 Python 3.8+"。用 3.8 语法解析器扫 271 个模块（0 语法错误），再在真实 3.8.20 环境安装验证 | ①最小改动：5 个模块补 `from __future__ import annotations`（`__init__`/`_plan_py`/`blas`/`helpers`/`parser`）；`_plan_py.Step` 是**模块级运行时表达式** `tuple[...] | None`，3.8 import 即 `TypeError` → 移入 `TYPE_CHECKING`（仅类型检查期需要）；②构建元数据：PEP 639 的 `license = "MIT"` 需要 setuptools≥77（不支持 3.8），改为经典 `license = {text="MIT"}` 并下调 `setuptools>=61`；`requires-python = ">=3.8"` + classifiers 3.8/3.9；③**顺带修一个真缺陷**：`python -m pyqcd --gpu-probe` 在无 cupy 安装上 `kernel_status` 为 None → `TypeError`，改为打印 "n/a — no CUDA kernel in this installation" | 真实 3.8.20 环境：`native_available=True`（KUPL .so 在 3.8 上构建并加载）、8 类收缩与核子 2pt/γ 夹心全部正确、`--gpu-probe` 给出结论而非崩溃；3.12 全量测试 121 passed/1 skipped/0 failed，docs 门禁 19 ok/2 skipped/0 failed，sphinx -W 双站 0 |
| R37 | 21:40 | **0.2.4** | 继续查缺陷：核对"计划缓存"是否真的生效 | **发现计划缓存从未对通用方程生效**（门限写死 `specialized is not None`，只覆盖 9 个注册形态），而规划占小/中型图调用时间的 **31–84%**（8 算子链 0.86 ms 中 0.76 ms，解析仅 0.002 ms）——重复调用 8 算子链此前 0.856 ms、比 opt_einsum 的 0.29 ms 慢 3× | 新增 `_CONTRACT_PLAN_CACHE`（64→128 项 LRU，键=方程+形状+dtype+设备+优化器+memory_limit+use_blas），**≥3 算子**才启用（两算子建键比规划更贵，实测反而慢 23 µs）。效果：chain4 0.212→0.063 ms、hyper-edge 0.183→0.055 ms、8 算子 0.856→0.097 ms；对 opt_einsum 现为 1.14/1.41/3.07× 更快；换形状不复用错误计划，300 组随机差分（每组连续两次，命中缓存）0 失败 | 121 passed/1 skipped/0 failed；随机差分 300 组;docs 门禁与 sphinx -W 全绿 |
| R38 | 11:40 | **0.2.7** | direct 模式（跳过规划器、直接调 backend.einsum）经实测判定失败：两算子慢 7–44×、三算子链慢 3900×、四算子链 10 分钟未返回；γ 夹心这类"注册但无 Python kernel"的形态慢 30× | 全部回滚：revert direct 提交（`git diff dev39 -- pyqcd/einsum` 为空）、删除 `docs/{en,zh}/usage.md` 的该节与测试、撤回格式错误的 `dev41` 标签；**0.2.7 重新发布 0.2.4 的稳定实现**（PyPI 不允许覆盖已上传文件，故用新版本号恢复），`pip install pyqcd` 现在拿到的是无 direct API 的稳定版。失败经验与绕道提醒写入 `docs/report_direct_mode_postmortem_20261111.md` | 121→（同口径）123 passed/1 skipped/0 failed（123 含 dev39 之后新增的两个无关测试）；docs_examples 19 ok/2 skipped/0 failed；sphinx -W 双站 0；form-audit PASS（17 例外/0 未预期）；PyPI 0.2.7 复验：direct API 已移除、链式结果正确 |

## 2026-11-12 发布资格快照（0.3.9，历史）

当时结论：**不具备正式发布条件**。该快照保留为 0.3.9 的失败证据，当前状态见下一节。

| 门禁 | 当前证据 | 判定 |
|---|---|---|
| 版本可发布 | 本地三处版本均为 `0.3.9`；PyPI 最新版本也是 `0.3.9` | FAIL：同版本不可覆盖，必须 bump |
| Git 清洁度 | `git describe --dirty` 为 `dev56-dirty`，核心源码、文档、测试和日志均有未提交/未跟踪改动 | FAIL |
| PEP 561 | `Typing :: Typed` 已声明，`setup.py`/`MANIFEST.in` 也引用 `pyqcd/py.typed`，但文件当前不存在 | FAIL |
| 统一套件 | `117 passed / 2 skipped / 2 failed`，两项均因环境缺 `gvar` | FAIL |
| offline release_check | job `1648472`：`5/9 PASS`；缺 `build` 导致 sdist 未生成，Sphinx 中英 `-W` 均失败 | FAIL |
| 性能声明边界 | 对 PyQCD 内部 generic，扩展 43 条 CPU/GPU 均 42/43 不慢；但对 `opt_einsum(dp)`，CPU 仅 36/43 不慢，7 条较慢，其中 2 条约 2.4–2.5× | WARN：不得宣称对所有外部 planner 全面领先 |

发版前最小闭环：

1. 恢复并纳入 `pyqcd/py.typed`。
2. bump 到 PyPI 未占用的新版本。
3. 补齐 `gvar` 或明确收缩发布测试门禁后，统一套件全绿。
4. 安装发布工具链 `build`/`twine`/`sphinx`，`release_check --offline --with-tests` 通过。
5. 构建 sdist、twine check、内容洁净、绝对路径审计。
6. 提交并双远端推送后，从 PyPI 新版本做干净 venv 安装复验。

## 0.4.0 发布闭环（完成）

用户指定目标版本为 `0.4.0`；发布前 PyPI 最新为 `0.3.9`，不存在同版本覆盖。

| 门禁 | 0.4.0 候选状态 | 证据 |
|---|---|---|
| 版本三处同步 | PASS | `pyproject=pyqcd=einsum=0.4.0`，`__native_version__=0.1.0` 保持不动 |
| PEP 561 | PASS | `pyqcd/py.typed` 已恢复，sdist/Wheel/远程安装均包含该文件 |
| 发布依赖 | PASS | `build 1.6.1`、`twine 7.0.0`、`sphinx 9.0.4`、`gvar 13.1.10` 已安装到 `qcu` |
| 文档数据 | PASS | README/概览/性能/安装/特化附录已切换 0.4.0；CPU 正面声明限定为相对 PyQCD generic，保留 7 条 CPU 对 `opt_einsum(dp)` 退化 |
| CPU 发布门禁 | PASS | `release_check --offline`：`11/11`；完整测试由 GPU 作业承担 |
| GPU 发布门禁 | PASS | 统一套件 `120 passed / 1 skipped / 0 failed`；KUPL/CUDA 均为 16/16 |
| sdist / twine / 本地安装 | PASS | job `1648752`：隔离 sdist、`twine PASSED`、原生安装与纯 Python fallback 均通过 |
| PyPI 上传 / 远程安装 | PASS | 发布 `pyqcd==0.4.0`，远程安装 job `1648812` 实测 `backend=kupl` |
| 双远端 / dev tag | PASS | GitLab/Gitee `main=158f8e21`；annotated `dev57` 已推送两端 |
| 在线终检 | PASS | job `1648826`：`18/18`，含 PyPI、RTD en/zh 与双远端 |

性能发布口径固定为：

* 43 条 full-scene：CPU 对 PyQCD generic `42/43` 不慢，中位 `0.8612x`；
  GPU `42/43` 不慢，中位 `0.2760x`。 `[作废]`
* 对 `opt_einsum 3.4.0 dp`：CPU `36/43` 不慢，GPU `42/43` 不慢。
* CPU `seq=6585`：`22.445 s -> 10.310 s`。
* 8638 条计划全部可规划；资源门 CPU 放行 3833、GPU 放行 2864。
* 43 条 full-scene 最大相对误差：CPU `6.95e-15`，GPU `6.68e-15`。

### sdist 漏包修复

首次从最终 sdist 安装时，KUPL 编译报 `family_apply.h` 缺失。该文件实际存在于
源码与隔离副本，`MANIFEST.in` 显式 `include` 也已存在，但 setuptools 84.2.1
隔离构建在 `get_requires_for_build_sdist` 阶段仍未看到它；直接运行
`setup.py egg_info` 则能正确收录。为使发布不再依赖该版本相关行为，
`NativeSdist.make_release_tree()` 现在在 manifest 处理完成后强制复制
`cpp/kupl/einsum/include/family_apply.h`。

job `1648752` 复验：

* sdist `pyqcd-0.4.0.tar.gz`，1,056,739 bytes；
* SHA-256 `032a56302b50c8b1a500a0976a66615296bec6e34cbe2fec5d00ce7da29fa25e`；
* `twine check` PASSED，无绝对路径；
* clean venv 从 sdist 成功构建 KUPL wheel，`backend=kupl`；
* `PYQCD_SKIP_NATIVE_BUILD=1` clean venv 成功安装，`backend=python`。

### 发布结果

* PyPI：https://pypi.org/project/pyqcd/0.4.0/
* PyPI 文件：仅 `pyqcd-0.4.0.tar.gz`
* sdist SHA-256：`032a56302b50c8b1a500a0976a66615296bec6e34cbe2fec5d00ce7da29fa25e`
* 发布提交：`158f8e21953d33c9d1c787fed35b8b813db50129`
* 发布标签：`dev57`
* PyPI 远程安装：`0.4.0`，导入路径位于 venv `site-packages`，
  `native_available=True`、`backend=kupl`
* 历史标签链自检仍报告 5 个既有已推送标签注释问题；本任务未改写这些历史标签。

## 0.5.16 发布闭环（GPU 归档完整性）

### 故障现象

用户执行 `pip install pyqcd -U` 后，在 A100 节点运行
`python -m pyqcd --gpu-probe` 得到：

```text
kernel status : fallback — the topology entry did not run for any contract dtype:
complex128=fail(stage=native: NativeUnavailable:
qcd_cuda_topology_apply is not exported by the archive)
```

这不是“CUDA 不可用”的统一结论。0.5.15 的发布 sdist 可构建完整 CUDA 归档，
本机现存 0.5.15 安装也实际导出全部 10 个必需符号；故障来自安装链允许
**旧归档或半成品归档通过校验并进入 site-packages**：

1. `cpp/cuda/einsum/make-nv.sh` 当时只硬校验 `applyPlanQcd`。同一份归档缺少
   `qcd_cuda_topology_apply`、`qcd_cuda_dedicated_apply` 等入口时仍会被发布。
2. `setup.py::NativeBuildExt` 在非校验退出码且同名 `.so` 已存在时会复用旧库；
   `_stage_native_library` 仅按文件名复制，不验证运行期所需导出集。
3. `package_data` 无条件按文件名打包 `libpyqcdeinsum_cuda.so`，于是构建日志中的
   CUDA 警告不会阻止不完整归档进入 wheel；pip 成功后用户直到 probe 才发现问题。
4. 原 probe 只给出一句缺符号原因，没有报告实际归档路径、文件是否存在，以及
   `missing` / `load-failed` / `symbol-missing` 的差异，难以区分安装、加载和 ABI 问题。

### 0.5.16 修复

| 层 | 修复 |
|---|---|
| CUDA 构建 | `make-nv.sh` 硬校验冻结 C ABI、topology、dedicated 共 10 个导出；使用临时产物校验后原子发布；缺 `nm` 或校验失败以退出码 3 失败 |
| 安装暂存 | `setup.py` 在扩展编译前、原生文件暂存前和 wheel 收集前都验证完整导出集；失败 CUDA 归档不进入 `package_data`，也不复用为可用后端 |
| 构建可追溯性 | wheel 写入 `_build_config.json`，持久化请求/配置/实际安装后端与跳过原因；`python -m pyqcd` 可回放这些决定 |
| 运行诊断 | `--gpu-probe` 输出实际 `cuda archive` 路径、`present`、`exported` 和 `missing` / `load-failed` / `symbol-missing` / `ready`；即使无 CuPy 也检查归档 |
| 运行缓存 | 绑定缓存按 inode、大小与 mtime 刷新；同一路径归档被替换时要求重启，不继续声称旧 binding 有效 |
| 第三方安装 | 缺 `nvcc` 明确说明只安装 KUPL；纯 Python 安装必须显式 `PYQCD_SKIP_NATIVE_BUILD=1`；wheel 不再错误标记为 `py3-none-any`；KUPL 默认使用可移植基线 ISA |
| 发布门 | `release_check.py` 每次重建新鲜 sdist，真实构建 wheel，检查平台标签和 KUPL/CUDA 导出，扫描全部可读文本中的绝对路径，并运行中英文文档与双站 Sphinx |

### 验证证据（2026-11-21）

| 门禁 | 结果 |
|---|---|
| 安装态身份 | `pyqcd 0.5.16`、CuPy 13.2.0 均从测试 venv 的 `site-packages` 导入；使用 `-I`、`PYTHONNOUSERSITE=1`，且清空 PyQCD 覆盖变量 |
| GPU probe | `gpu024` / NVIDIA A100-PCIE-40GB / compute 8.0；`cuda archive.status=ready`、`present=true`、`exported=true`；四种 dtype 全部成功，`max|diff|=0.0`；结论 `CUDA fast path is active` |
| topology 实际执行 | `kernel_force.requested=topology`、`ran=cuda_topology`、`steps=1`；安装版本和包路径随 JSON 留存 |
| GPU 定向契约 | GPU probe 诊断 `18 passed / 0 skipped / 0 failed`；topology `27 passed / 0 skipped / 0 failed` |
| 全回归 | Slurm `4038294`，`gpu-debug` / `gpu024`：`COMPLETED 0:0`；`160 passed / 1 skipped / 0 failed` |
| 静态制品（final4） | `twine check --strict` PASSED；3,303,568 bytes；SHA-256 `810f4081f198126e1f45246c26c692b406790c64530f2aab2f3b9d260beb283f`；199 个条目，0 个 testing 文件、0 个 `.so`、0 个 `data/tmp`、0 个绝对路径 |

复查期间又修复了 CUDA binding 缓存的另一个边界：同一路径归档被成功加载后若被替换，
“需重启”状态现在按绝对路径持久化；A→B→A 也不会重新加载旧路径。相关诊断契约最终为
`15 passed / 4 skipped / 0 failed`（无 GPU 节点上的 4 项 CUDA 实测按环境跳过）。

最终候选为 `final9`：

| 项 | 值 |
|---|---|
| 文件 | `pyqcd-0.5.16.tar.gz` |
| 大小 | 3,303,626 bytes |
| SHA-256 | `8d8b68fdae2d8dd5d689e990a91291549cfc0c2f768b5fb29e3e70e4abd4f2c6` |
| 静态门 | `twine check --strict` PASSED；199 个条目；0 docs / testing / `.so` / `data/tmp` / `AGENTS.md` / 绝对路径 |
| 干净安装 | venv `include-system-site-packages=false`；`pip check` 无冲突；PyQCD 0.5.16 与 CuPy 13.6.0 均从该 venv 的 `site-packages` 导入 |
| 原生导出 | CUDA 10/10、KUPL 3/3；`_build_config.json` 记录 `kupl`/`cuda` 均 built |
| 安装三态 | CUDA extra：`backend=cuda`；KUPL-only：`backend=kupl` 且仅 1 个 KUPL `.so`；`PYQCD_SKIP_NATIVE_BUILD=1`：`backend=python`、0 个原生库；三者 `pip check` 均无冲突 |
| 最终 A100 | Slurm `4038362`，`gpu-debug` / `gpu024`，`COMPLETED 0:0`；`cuda archive.status=ready`、四类 dtype `max|diff|=0.0`、`CUDA fast path is active`；topology JSON 为 `requested=topology`、`ran=cuda_topology`；GPU probe 诊断 `18 passed`、topology `27 passed`、全回归 `160 passed / 1 skipped / 0 failed` |

十轮复查采用“完整门 + 定向门 + 最终制品门”的交叉口径：先前 `final7`/`final8` 的
构建证据只用于发现缓存边界和过期 `egg-info`；`final9` 是由干净元数据重建、无 `docs/`
泄漏的发布件。最终发布以本表 `final9` 哈希、安装三态、A100 和随后在线
`release_check` 为准。
