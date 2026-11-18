# 拓扑步表统一：口径、生成与交付的判据（2026-11-18）

面向**下一次动这两张表或这套生成器的人**。这里只写"怎么判、怎么做、怎么证明"，
不写流水账；时间线与原始命令输出见 `logs/einsum-topology-unification-20261118.txt`，
精炼约定见 `pyqcd/einsum/AGENTS.md` §10.5 与 `skills/pyqcd-einsum/SKILL.md` §[T3]。

| 项 | 值 |
|---|---|
| 范围 | `pyqcd/einsum/{_topology_gen_common,topology_steps_gen}.py`、`pyqcd/testing/{_topology_kernel_gen,_cuda_topology_steps_gen}.py`（deprecated 转发）、两张生成表 |
| 不覆盖 | 用户侧 `optimize` / `kernel_force` 语义（见 `einsum_framework_convergence_20261118.md`）、C++ 执行器实现 |
| 结论口径 | 只认"当前源码 + 当前产物实测"；缺报告时**不转述**，重新测 |

---

## 1. 三条不变量（先背下来）

1. **形状源只有一个**：`ROLE_SHAPES`
   （`peram [4,4,100,100]`、`gamma [4,4]`、`VDV [100,100]`、`VVV [100,100,100,100]`），
   由 `role_sizes()` 按角色绑定每个 label。`production`（生产异形轴）与 `uniform`
   （等长探针 dim=4）**已删除**，不是 deprecated。
2. **优化器在生成侧统一 `auto`**，用户侧可写集合仍是 7 个叶子、不含 `auto`。
   两个"auto"同名不同义，别混用（详见收敛文档 §10.3）。
3. **两套表不可互译**：KUPL `.inc` 是"打包 → cblas GEMM → 解包"的**字节码**
   （指令绑定 cblas 调用约定）；CUDA `.h` 是 `int8` **活表位置对**（只有顺序，
   由参数化执行器运行期展开 cuBLAS）。粒度不同，能复用的只有"语料枚举 +
   路径求解"上游，格式层必须各写各的。

---

## 2. 重生成：五个必须过的判据

### 2.1 换了口径就必须换 `sha256`

- **现象**：把默认形状源改成 `roles` 后再"重新生成"，产物 `sha256` 与旧表一字不差。
- **根因**：只改了 CLI 默认值/文档，而 `build_kupl(...)` 的缺省实参仍是
  `shape_source="production"`、CUDA 的仍是 `"uniform"` —— 默认路径没变，
  "重生成"就是原样重跑。
- **判据/做法**：先取旧值再比新值，**必须不同**；相同即视为"口径未生效"，
  不许交付。
  ```bash
  git show HEAD:cpp/cuda/einsum/src/topology_steps.h | sha256sum
  git show HEAD:cpp/kupl/einsum/src/topology_kernels.inc.gz | gzip -dc | sha256sum
  sha256sum cpp/cuda/einsum/src/topology_steps.h \
            cpp/kupl/einsum/src/topology_kernels.inc
  ```
- **教训**：*"改了默认值"不等于"口径生效"* —— 唯一可靠证据是产物字节变了。

### 2.2 同一输入两次生成必须同 `sha256`

- **现象**：表头"去重后 N 条不同步序"里的 N 每次不同（3358 / 3051 / …）。
- **根因**：生成过程不确定（`set`/`dict` 迭代序、并发、未排序的去重结果）。
  `layout()` 里 `sorted({...})` 是这条不变量的实现位置。
- **判据/做法**：连续生成两次（写到 `/tmp`，不要动仓库），两次 `sha256` 相同，
  且等于工作区表。若不同，**修生成器**，不要继续用产物。
- **教训**：*"数字每次都变"是生成器在报警，不是表的特性。*

### 2.3 `--verify` 必须全零 + 步序全等

```bash
python -B -m pyqcd.einsum.topology_steps_gen --target cuda --verify
# missing=0 extra=0 mismatch=0 orders 3477/3477 match=True covers_file=True
```

`--verify 2582` 只核对历史基础表时会正常报 `extra=6056 covers_file=False`，
那**不是**失败（扩表后的预期），判据只看 `missing` / `mismatches`。

### 2.4 静默兜底必须已经消失

- **现象**：忘传 `sizes` 时收缩序悄悄改用等长探针，表照样生成成功。
- **根因**：`sizes=None → 探针` 的兜底分支把"忘记绑定 roles"变成**看不见**的行为。
- **判据/做法**：`solve_path(equation)` 必须直接报错。
  ```bash
  python -B -c "from pyqcd.einsum._topology_gen_common import solve_path; \
                solve_path('cadb,menf,ac,em,bd,fn->')"
  # TypeError: solve_path() missing 1 required positional argument: 'sizes'
  ```
- **教训**：*缺参数要吵，不要猜* —— 静默兜底是生成器里最贵的一类"方便"。

### 2.5 `--help` 里的口径只能是现行的那一个

`--shape-source` 的 `choices` 必须只剩 `("roles",)`；帮助文本可以提"旧口径已删除"，
但**不能**留可选项。改完 docstring 一定重跑 `--help`：本模块用
`RawDescriptionHelpFormatter`，markdown（`**加粗**`、`::` 字面量、grid table）
会原样打出来。

---

## 3. 生成器的分层：共享上游 + 两个格式层

```
_topology_gen_common.py        # 语料枚举（equations）、路径求解（solve_path/solve_pairs）、
                               # 角色绑定（role_name/role_shape/role_sizes/ROLE_SHAPES）
        │
        ├── topology_steps_gen.py :: generate_kupl  → cpp/kupl/einsum/src/topology_kernels.inc(.gz)
        └── topology_steps_gen.py :: generate_cuda  → cpp/cuda/einsum/src/topology_steps.h(.gz)
```

- 共享层输出**与后端无关**的一步一记录：`pool`（CUDA 要的活表位置）+ `a_src/b_src/dst`
  （KUPL 要的槽位口径），两个后端各取所需。
- 格式层只做"编码 + 排版"，不重算路径。判据：**新增一个后端时不应改共享层**；
  若必须改，说明共享层的产物不中立。
- 旧入口 `pyqcd.testing._topology_kernel_gen` / `_cuda_topology_steps_gen` 现在是
  **转发薄包装**（保留了全部旧 import 面）。它们的 `__main__` 仍可用，所以对外文档里
  的旧命令不会立刻失效——但新代码请用 `python -m pyqcd.einsum.topology_steps_gen`。

---

## 4. 大文件交付：压缩件入库，构建时解压

| 项 | KUPL `.inc` | CUDA `.h` |
|---|---|---|
| 本体大小 | 13,485,987 B | 1,113,387 B |
| `.gz` | 1,658,036 B（8.13×） | 250,476 B（4.44×） |
| 入库形态 | 只入 `.gz`（`.inc` 被 gitignore） | `.h` **仍被跟踪**（历史遗留，见 §8） |

- **解压条件必须写对**：`本体缺失 或 .gz 更新`。
  写成"No. gz 存在就解压"会在本地刚生成表后把它覆盖回旧版本（一次无谓的
  "我明明重生成过，怎么又变回去了"）。
  ```bash
  [ ! -f "$STEP_TABLE" ] || { [ -f "$STEP_TABLE_GZ" ] && [ "$STEP_TABLE_GZ" -nt "$STEP_TABLE" ]; }
  ```
- **两侧构建脚本必须对称**：CUDA 侧早年没有 `gzip` 分支，一旦仓库只发布 `.h.gz`，
  干净检出直接构建失败；`CUDA_SYNC` 后已补齐（`make-nv.sh:222-235`）。
  新增"压缩件入库"时，先 `grep -n gzip cpp/*/einsum/make*.sh` 确认两边都在。
- **`.gz` 必须确定性**：`gzip.compress(data, compresslevel=9, mtime=0)`。默认头部带
  时间戳，会让被跟踪的 `.gz` 每次重生成都变脏（载荷其实没变）。

---

## 5. 图纸数据（`_wick_tasklist.py`）：生成物纪律

1. **生成物不可手改**。`_wick_tasklist.py` 的 `_BLOB` 每次生成整段重写，手改必被擦除。
   新增条目只能写 `_wick_extra.jsonl`（每行一个 JSON 对象），由
   `python -m pyqcd.einsum.wick_tasklist_gen` 合并，`seq` 从 `N_base+1` 顺延。
2. **整表改格式时的判据是"逐行不动点"**，不是"逐字节等于图纸"：
   `normalize(旧行) == 新行` 对**全部 8638 行**成立。也不要用
   `旧行.replace(" ", "") == 新行`——现行规范把 `. `/`：` 换成 `-`，不是纯删空格。
3. **`seq` 必须连续 1..N，不能由 T 号推**。图纸的「基础表」与「附录」各自从
   `T00001` 起编（720 个重复）；`line` 是图纸物理行号，因夹标题/空行而跳号。
4. **`--check` 是唯一的漂移判据**（按同一规则重算再比）；拿图纸原文逐字节对，
   在规范化之后必然不等，属于用错判据。

---

## 6. 提示语与语义：删功能必须同步改提示

- `kernel_force` 删掉 `auto` 后，`_specialize` 里"先试 dedicated、失败才试 topology"
  就成了**假话**（现在只按强制值试一个，失败即报错或回落）。
  改法：条目只写"命中了什么形态"，行尾说明按现场事实分三档拼
  （强制跑成 / 未强制 / 强制失败回落）。
- **不要保留"未做…的自动尝试"这类多余解释**：用户明确要求只陈述实际走了什么。
- **`[match: X]` 与 `[kernel: Y]` 必须分开读**：前者是命中的**形态族**，后者是
  **实际执行**条目（`LAST_INFO['path']`）。`[match: topology] [kernel: kupl_dedicated]`
  是正常结果，不是标签取错。
- 写这类提示前先确认底层语义还在：`kupl_topology_v2` 这条回退路线已被删除
  （未命中即 status 4），文档里若还写"两条路线"就是假话。

---

## 7. 同批踩过的通用核对项（与表无关，但每次重构都会遇到）

1. **删常量先 `grep` 引用方**：删 `_kernel_force.AUTO_LABEL` 后 `_api.py` 3 处引用立刻
   `AttributeError`，连锁 5 项契约失败。判据：`grep -rn '<NAME>' pyqcd/ pyqcd/testing/`
   为 0 才动手（含契约文件，它们也在 `pyqcd/testing/**` 里）。
2. **元组层级要数清**：`KernelUnavailableError((requested,), ...)` 比预期多包一层，
   下游 `", ".join()` 直接崩。传参前用 `type(x) is tuple and all(isinstance(i, str) …)`
   之类的显式判断，别靠"看起来像"。
3. **默认值 ≠ 显式**：CLI 把 `--kernel-force` 的默认值写成 `"generic"`，会让"未指定"
   被判成"显式指定 generic"，与配置表里正常命中的行冲突报错。
   **默认值必须用 `None` 表示"没给"**，需要"真的给过吗"时取一个显式布尔位。
4. **路径不展开 `~`**：`os.path.abspath("~/.cache")` 得到的是**字面 `~` 目录**，
   `--profile-dir` 静默失效。所有用户输入的路径先 `expanduser()` 再 `abspath()`。

---

## 8. 尚未收敛的边界（登记，不在本轮范围）

1. **表头 banner 仍写历史入口名**（`pyqcd/testing/_topology_kernel_gen.py` 等）。
   改动会让产物与已发布表差一行注释；已在 `render_kupl`/`render_cuda` 就地写明
   "别顺手改"。要统一改口，得单独一轮（banner + AGENTS/SKILL + `make.sh` 注释 +
   `.gitignore` 注释一起）。
2. **`.h` 与 `.h.gz` 的入库形态与裁定相反**：`.h` 仍被跟踪（`.gitignore` 是白名单，
   `!*.h` 放行），`.h.gz` 被 `*` 忽略。属仓库侧裁定，改之前先确认发布/构建流程。
3. **`equations()` 惰性 import 了 `pyqcd.testing`**：生成器是开发期工具，为不改语料
   解析口径暂时如此。彻底解耦要把语料解析迁进 `pyqcd/einsum/`（会动 3 个契约的 import）。
4. **文档里仍有旧入口名与旧口径数字**：见归档日志 §5 的清单；旧入口仍可用，
   所以不是错误，只是不够新。

---

## 9. 一次完整重生成的检查单

```bash
# 0) 备份旧表（供 sha256 对比）
cp cpp/cuda/einsum/src/topology_steps.h /tmp/old.h
gzip -dc cpp/kupl/einsum/src/topology_kernels.inc.gz > /tmp/old.inc

# 1) 重生成（默认 roles + auto）
python -B -m pyqcd.einsum.topology_steps_gen --target all

# 2) 判定：新旧 sha256 必须不同
sha256sum /tmp/old.h cpp/cuda/einsum/src/topology_steps.h
sha256sum /tmp/old.inc cpp/kupl/einsum/src/topology_kernels.inc

# 3) 判定：两次生成必须相同（写 /tmp，不碰仓库；--output 只能配单目标）
for i in 1 2; do
  python -B -m pyqcd.einsum.topology_steps_gen --target cuda --no-gzip --output /tmp/check_h_$i.h
  python -B -m pyqcd.einsum.topology_steps_gen --target kupl --no-gzip --output /tmp/check_inc_$i.inc
done
sha256sum /tmp/check_h_*.h /tmp/check_inc_*.inc

# 4) 判定：表自校验
python -B -m pyqcd.einsum.topology_steps_gen --target cuda --verify

# 5) 判定：`.gz` 载荷等于本体
gzip -dc cpp/kupl/einsum/src/topology_kernels.inc.gz | cmp - cpp/kupl/einsum/src/topology_kernels.inc
gzip -dc cpp/cuda/einsum/src/topology_steps.h.gz      | cmp - cpp/cuda/einsum/src/topology_steps.h

# 6) 构建两侧（解压逻辑必须都在）
bash cpp/kupl/einsum/make.sh && bash cpp/cuda/einsum/make-nv.sh
```

数值正确性另算：抽语料方程按 `ROLE_SHAPES` 造形状，与 `np.einsum` 比 `rel ≤ 1e-12`；
再从**旧表覆盖不到**的序号（>2582）抽样，证明扩表/换序真的生效。
