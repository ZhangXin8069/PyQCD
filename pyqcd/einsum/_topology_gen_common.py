"""拓扑步表生成器的共享层：语料枚举、路径求解与角色形状约定。

KUPL 与 CUDA 两张表的**输出格式**完全不同（``.inc`` 字节码步表 vs ``.h`` 活表
位置对），但它们吃同一份语料、用同一套路径求解。本模块收这两段；两个后端只剩
格式层，见 :mod:`pyqcd.einsum.topology_steps_gen`。

形状源**只有一个**：:data:`ROLE_SHAPES`（用户硬性约定，见下），默认
``shape_source="roles"``、由 :func:`role_sizes` 落地，**整个** 8638 条语料都走它。
两条历史口径 ``production``（生产异形轴）与 ``uniform``（等长探针）已 **删除**：
它们生成的表与 ``ROLE_SHAPES`` 口径不可互换，保留只会让"哪张表是哪个口径"变成
无法判断的历史包袱（两种口径生成的
表互不相同，逐字节不可互换；默认表的排序就是 ``ROLE_SHAPES`` 上的结果）。

优化器在**生成侧统一为** ``auto``（:data:`KUPL_OPTIMIZE` / :data:`CUDA_OPTIMIZE`，
用户裁定 2026-11-18）：``auto`` 与 ``greedy`` 不是一件事，实测 seq=646 上 greedy
4.08e8 MAC vs auto 1.08e8 MAC（26%），所以两张表都按 ``auto`` 定型。

**生成侧 ``auto`` / 用户侧 7 叶子：这是两条互不重叠的口径，不许混用。**

* 生成侧（本模块与 :mod:`pyqcd.einsum.topology_steps_gen`）写死 ``"auto"``——那是
  ``opt_einsum`` 风格的"按算子数选 optimal/branch/greedy"选择器，也是
  ``contract_path`` 的缺省值，步表要固化的是**最好**的序而不是最便宜求出的序。
* 用户侧 ``optimize`` 的可写取值仍是 **7 个叶子**
  （``greedy``/``optimal``/``dp``/``branch_1``/``branch_2``/``branch_all``/``none``，
  见 ``_profile_cache.OPTIMIZE_LEAVES``），**不含 ``auto``**：``auto`` 只作为
  ``True``/``None`` 的内部归一结果存在（``paths._IMPLICIT_DEFAULT_NAMES``），
  ``paths.get_path_fn("auto")`` 照旧抛 ``KeyError``，配置表里写 ``auto`` 仍是非法值。
  用户侧该写哪个叶子由 ``--tune`` 的实测结论决定，**不是** ``auto``。

形状源**只有一个**（``ROLE_SHAPES``），优化器与形状源是两件事：形状源决定各轴长度，
优化器决定在同一组长度上选哪条路径；两表都取自 ``ROLE_SHAPES`` 上的 ``auto``。

角色形状的**用户硬性约定**（:data:`ROLE_SHAPES`，默认且唯一形状源）::

    peram → [4,4,100,100]
    gamma → [4,4]            （``C*gamma_*`` 前缀同样按 gamma）
    VDV   → [100,100]
    VVV   → [100,100,100,100]

绑定规则见 :func:`role_sizes`：``gamma`` / ``VDV`` / ``VVV`` 的规范形状各轴同长，
故把指标段的**每个**标签钉到该长度（语料里它们的轴数是 2–3 / 2–3 / 3–4，硬卡规范
轴数会把附录形态判死）；``peram`` 四轴不等长，按**位置**绑定。角色名从
``_wick_tasklist`` 的 ``roles`` 取，同一 label 被钉成两个长度即报错。

生成器的两条**验收判据**（比"看着像对的"可靠得多，重生成后先跑这两条）：

1. **换了口径就必须换 ``sha256``**。只改默认值不算生效：早先 ``build_kupl`` 的缺省
   ``shape_source`` 仍是 ``production``、CUDA 的仍是 ``uniform``，于是"按 roles 重新
   生成"实际是原样重跑，产物 ``sha256`` 与旧表逐字节相同。判据是**新产物 ``sha256``
   与旧表必须不同**——相同即证明默认路径没换，此时不要交付。
2. **同一输入两次生成必须 ``sha256`` 相同**。若两次不同，先查 ``set``/``dict`` 迭代序
   与并发，而不是继续用产物。"去重步序数每次不同"就是这种不确定的表征，修的是生成器，
   不是表。

两套表**不可互译**：KUPL ``.inc`` 是"打包→cblas GEMM→解包"的字节码（指令逐条绑定
cblas 调用约定），CUDA ``.h`` 是 ``int8`` 活表位置对（只有顺序，由参数化执行器在
运行期展开 cuBLAS）。粒度不同，任何"从一张表推出另一张"的想法都不成立；能复用的
只有本模块的"语料枚举 + 路径求解"上游，格式层必须各写各的。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import contract_path

#: 编译表覆盖的方程上限 = **整个语料**（基础表 2582 + 附录 6056）。
MAX_KERNELS = 8638

#: 默认（也是唯一）形状源：按 :data:`ROLE_SHAPES` 绑定每个 label 的长度。
DEFAULT_SHAPE_SOURCE = "roles"

#: KUPL 步表的路径优化器。``"auto"`` 与 ``_kernel_kupl.try_dedicated`` 同口径。
KUPL_OPTIMIZE = "auto"
#: CUDA 步表的路径优化器。与 KUPL 统一为 ``"auto"``（用户裁定 2026-11-18）；
#: 生成侧写死 ``auto``，用户侧 ``optimize`` 仍是 7 叶子、不含 ``auto``（见模块文档）。
CUDA_OPTIMIZE = "auto"

#: 角色 → 规范形状（用户硬性约定，默认配置）。见模块文档的适用边界。
ROLE_SHAPES: dict[str, tuple[int, ...]] = {
    "peram": (4, 4, 100, 100),
    "gamma": (4, 4),
    "VDV": (100, 100),
    "VVV": (100, 100, 100, 100),
}

#: 角色名前缀，顺序即匹配优先级（``peram``/``gamma`` 先于张量族）。
_ROLE_PREFIXES = ("peram", "gamma", "VDV", "VVV")


def equations(include_momentum: bool = True) -> list[str]:
    """按文件顺序返回语料方程（去重前）。

    委托测试包的 ``_parse_equations``：那是本库语料解析的**唯一**实现，另写一份
    必然与 ``_wick_tasklist`` 和两张表漂移。惰性 import —— 生成器是开发期工具，
    正常 ``import pyqcd.einsum`` 不会把测试包拖进来。
    """
    from ..testing._topology_equations_contract import _parse_equations

    return _parse_equations(include_momentum=include_momentum)


def parse(equation: str) -> tuple[list[str], str]:
    """方程 → (操作数项列表, 输出项)。"""
    lhs, rhs = equation.split("->")
    return lhs.split(","), rhs


def solve_path(equation: str,
               # 必填、**不要**给它加缺省值：此处曾允许 sizes=None 并退回等长探针，
               # 结果是"忘记绑定 roles"变成看不见的行为（收缩序已经换了却没人报错）。
               sizes: Mapping[str, int],
               *,
               optimize: str = CUDA_OPTIMIZE,
               ) -> tuple[list[dict[str, Any]], list[str], str]:
    """镜像一次收缩，返回每步的两侧项、输出项与源槽位。

    ``sizes`` **必须**给出每个 label 的长度，收缩序就在该形状上定型
    （调用方传 :func:`role_sizes` 的结果，即 ``ROLE_SHAPES`` 口径）。
    早先还有一条"``sizes`` 缺省时退回等长探针"的分支，已随
    ``--shape-source uniform`` 一起**删除**：探针形状与生产形状的收缩序没有可比性，
    静默兜底只会让"忘记绑定 roles"变成看不见的行为。

    每步的键：

    * ``pool``：本步两个操作数在**当前池**里的位置（``contraction_list`` 的
      ``inds`` 原值，池每步删旧项、末尾追加新项）；CUDA 表要的就是这个；
    * ``a_src``/``b_src``/``dst``：KUPL 的槽位口径——非负数表示原始操作数下标，
      负数表示中间槽 ``-1-slot``，``dst=-1`` 表示最终输出缓冲区；
    * ``left``/``right``/``out``：该步等价 einsum 的三段项。

    ``optimize`` 缺省取 :data:`CUDA_OPTIMIZE`（两个后端生成侧统一为 ``"auto"``）。
    这只是**生成期**的取值，用户侧 ``optimize`` 的可写集合是 7 个叶子、
    不含 ``auto``（见模块文档）。
    """
    terms, out = parse(equation)
    try:
        shapes = [tuple(int(sizes[ch]) for ch in term) for term in terms]
    except KeyError as missing:
        raise AssertionError(
            f"{equation}: label {missing.args[0]!r} has no size; "
            f"the corpus roles must bind every label") from None
    _, info = contract_path(equation, *shapes, shapes=True, optimize=optimize)

    pool_terms = list(terms)
    pool_slots = list(range(len(terms)))
    next_slot = 0
    steps: list[dict[str, Any]] = []
    total = len(info.contraction_list)
    for number, (inds, _removed, einsum_str, _remaining, _blas) in enumerate(
            info.contraction_list):
        left, _arrow, result = einsum_str.partition("->")
        tmp = left.split(",")
        if len(inds) != 2:
            raise AssertionError(f"{equation}: unary planner step is unsupported")
        if [pool_terms[index] for index in inds] != tmp:
            raise AssertionError(f"{equation}: pool mirror disagrees with the planner step")
        a_slot, b_slot = pool_slots[inds[0]], pool_slots[inds[1]]
        for index in sorted(inds, reverse=True):
            del pool_terms[index]
            del pool_slots[index]
        final = number == total - 1
        dst = -1 if final else next_slot
        if not final:
            next_slot += 1
        steps.append({"pool": (int(inds[0]), int(inds[1])),
                      "left": tmp[0], "right": tmp[1], "out": result,
                      "a_src": a_slot, "b_src": b_slot, "dst": dst})
        pool_terms.append(result)
        # 中间槽在 KUPL 的步表里编码成 -1-slot：非负数一律表示操作数下标。
        pool_slots.append(-1 - dst)
    if pool_terms != [out]:
        raise AssertionError(f"{equation}: the path does not end on the equation output")
    return steps, terms, out


def solve_pairs(equation: str,
                sizes: Mapping[str, int],
                *,
                optimize: str = CUDA_OPTIMIZE,
                ) -> tuple[tuple[int, int], ...]:
    """CUDA 表口径：每步两个活表位置的无序对（升序）。

    与 :func:`solve_path` 的 ``pool`` 同源，只是收成 ``(min, max)``——C++ 侧
    ``kQcdTopoOrderN[]`` 就是这种对，去重后按字典序聚合。

    ``sizes`` 必填（``ROLE_SHAPES`` 口径）；等长探针分支已删除。
    """
    steps, terms, _out = solve_path(equation, sizes, optimize=optimize)
    if len(steps) != len(terms) - 1:
        raise AssertionError(f"{equation}: {len(steps)} steps for {len(terms)} operands")
    return tuple((min(step["pool"]), max(step["pool"])) for step in steps)


def role_name(role: str) -> str:
    """角色名归一：去空白与 ``C *`` 前缀（``C * gamma_BH`` → ``gamma``）。"""
    text = role.replace(" ", "")
    if text.startswith("C*"):
        text = text[2:]
    for prefix in _ROLE_PREFIXES:
        if text.startswith(prefix):
            return prefix
    raise ValueError(
        f"unknown role {role!r}; expected one of {', '.join(_ROLE_PREFIXES)} "
        f"(optionally prefixed with 'C * ')")


def role_shape(role: str) -> tuple[int, ...]:
    """角色的规范形状（:data:`ROLE_SHAPES`）。"""
    return ROLE_SHAPES[role_name(role)]


def role_sizes(terms: Sequence[str], roles: Sequence[str]) -> dict[str, int]:
    """按 :data:`ROLE_SHAPES` 位置绑定每个 label 的长度（默认且唯一形状源）。

    绑定次序与 ``bench_specialized._label_sizes`` 同构——**先小张量、peram 补余**，
    这样语料里 peram 与 ``gamma`` / ``VDV`` / ``VVV`` 大面积的共享标签
    （``ac`` 同时属于 ``gamma_M`` 与 ``peram_u``）才有解：

    1. ``gamma`` / ``VDV`` / ``VVV`` 的规范形状**各轴同长**，因此它们把指标段的
       **每个**标签钉到那个长度，标签个数由语料决定——实测 ``gamma`` 2–3 个、
       ``VDV`` 2–3 个、``VVV`` 3–4 个。硬卡"``VDV`` 必须 2 个、``VVV`` 必须 4 个"
       会把 8638 条里的 7880 条直接判死（附录形态就是 3 轴顶点）。
    2. ``peram`` 四轴不等长，只有在**全语料**都是 4 轴（实测 50102/50102）时才可
       按位置绑定 ``[4,4,100,100]``；它只补**尚未**被钉住的标签，已钉住的必须同值。
    3. 同一 label 被钉成两个不同长度、或 ``peram`` 的轴数不是 4（语料形态变了）
       即报错——不猜。

    返回的 ``sizes`` 供 ``contract_path(..., shapes=True)`` 定型收缩序：序只取决于
    各轴长**相对**大小，因此这张表就是步表排序的唯一依据。
    """
    if len(terms) != len(roles):
        raise ValueError(f"{len(roles)} roles for {len(terms)} operands")
    sizes: dict[str, int] = {}

    def bind(label: str, extent: int, role: str) -> None:
        known = sizes.setdefault(label, extent)
        if known != extent:
            raise ValueError(
                f"label {label!r} is {known} in one role but {extent} in {role}; "
                f"the roles disagree with ROLE_SHAPES")

    # 1. 小张量先钉：规范形状各轴同长，逐标签取其长度。
    for term, role in zip(terms, roles):
        if role.startswith("peram"):
            continue
        shape = role_shape(role)
        if len(set(shape)) != 1:
            raise ValueError(
                f"{role}: ROLE_SHAPES[{role_name(role)!r}] = {list(shape)} has axes "
                f"of differing extent, so labels cannot be bound per-axis")
        for label in term:
            bind(label, shape[0], role)

    # 2. peram 按位置：四轴不等长，不能被它覆盖小张量钉住的值（同值才合法）。
    for term, role in zip(terms, roles):
        if not role.startswith("peram"):
            continue
        shape = role_shape(role)
        if len(term) != len(shape):
            raise ValueError(
                f"{role}: operand {term!r} has {len(term)} axes but "
                f"ROLE_SHAPES[{role_name(role)!r}] is {len(shape)}-axis")
        for label, extent in zip(term, shape):
            bind(label, extent, role)
    return sizes
