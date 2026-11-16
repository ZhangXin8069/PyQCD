"""等价映射接口层的契约：`pyqcd.einsum._equivalence` + verify_specialized 的集成。

本层是**纯接口层**：只把方程改写成等价形式并给出张量调整建议，不改任何执行路径。
契约因此只断言五件事：
1. 三族改写（操作数位置 / 单操作数轴序 / 输出段轴序，**任意复合**）的**数学等价性**
   （用 numpy 真值对拍，不是断言实现细节）；
2. 报告的**保守性**（非法/无解输入不崩、不猜、已命中的不改写、最小改动优先）；
3. `verify_specialized` 的输出里这些字段真的出现，且复合变换的"次序"与"轴序"
   两行**同时**出现且自洽；
4. **开关**（`PYQCD_EINSUM_EQUIVALENCE`）：默认开启；关闭后 `verify_specialized`
   的输出与引入本功能之前**逐字符相同**（关闭即无此功能）；
5. **置换的方向**（第 6 节）：非自逆置换必须"转对方向对、转反方向必错"——判据
   靠**非对称形状**（每个标签长度不同）而非数值巧合；自逆置换显式标注为"结构上
   测不出方向"，不许冒充方向证据。同时断言**空白不是轴位**（`perm` 长度恒等于
   操作数秩）。

第 5 条是补的一课：在它之前，把"存原始 perm 而非逆置换"的错误版本塞进本文件，
10 条用例**全绿**——对称形状 + 自逆置换两道盲区叠加，方向错误在结构上不可见。
"""

from __future__ import annotations

import ast
import io
import contextlib
import json
import os
import time
import unittest
import warnings

import numpy as np

import pyqcd.einsum as pe
from pyqcd.einsum import _equivalence, _specialize, verify_specialized


# ------------------------------------------------------------ 0. 共用工具

#: 语料 seq=1：``cadb,menf,ac,em,bd,fn->``。复合用例都以它为"目标式"。
CORPUS_SEQ1 = "cadb,menf,ac,em,bd,fn->"


def _seq1_label_sizes():
    """seq=1 的真实生产形状（``_label_sizes`` + ``dims("distill")``）。"""
    from pyqcd.einsum import _wick_tasklist as tasklist
    from pyqcd.einsum._bench_scene import dims
    from pyqcd.einsum._dedicated_kernels import parse_equation
    from pyqcd.einsum.bench_specialized import _label_sizes

    task = tasklist.lookup(1)
    terms, _ = parse_equation(task.equation)
    return _label_sizes(terms, task.roles, dims("distill"))


def _apply_mapping(equation, result, sizes, seed=20261116):
    """按文档语义执行映射，返回 ``(direct, via, reference)``。

    ``operand_order`` 是"新第 i 位取源第几个"，再对每个操作数按
    ``operand_permutations[i]``（**新次序**索引）做 ``transpose``；输出若有
    ``output_permutation`` 就再转一次。三处顺序都按字段 docstring 来，不猜。
    """
    terms, _ = _equivalence._split(equation)
    shapes = [tuple(sizes[ch] for ch in term) for term in terms]
    rng = np.random.default_rng(seed)
    ops = [rng.standard_normal(s) + 1j * rng.standard_normal(s) for s in shapes]

    direct = pe.contract(equation, *ops, backend="numpy", optimize="greedy")
    reference = np.einsum(equation, *ops, optimize="greedy")

    order = result.operand_order or tuple(range(len(terms)))
    perms = result.operand_permutations or (None,) * len(terms)
    mapped = []
    for i in range(len(terms)):
        perm = perms[i] if i < len(perms) else None
        tensor = ops[order[i]]
        mapped.append(np.transpose(tensor, perm) if perm else tensor)

    via = pe.contract(result.equation, *mapped, backend="numpy", optimize="greedy")
    if result.output_permutation:
        via = np.transpose(via, result.output_permutation)
    return direct, via, reference


def _assert_equivalent(equation, result, sizes, tol=1e-12):
    direct, via, reference = _apply_mapping(equation, result, sizes)
    scale = max(float(np.max(np.abs(reference))), 1e-300)
    assert via.shape == reference.shape, f"{via.shape} vs {reference.shape}"
    assert float(np.max(np.abs(direct - reference))) <= tol * scale, \
        f"{equation}: 直接算就不是 numpy 真值"
    assert float(np.max(np.abs(via - direct))) <= tol * scale, \
        f"{equation}: 映射后数值不一致（kind={result.kind}）"


# ------------------------------------------------------------------ 1. 等价性

def test_axis_permutation_mapping_is_numerically_equivalent():
    """单操作数轴序置换：改方程 + 同步 transpose 张量 → 结果逐元素不变。

    这是用户实际报的场景：``cadb,menf,ac,em,bd,fn->``（语料命中）与
    ``acdb,...``（MISS）只差第 0 个操作数里 ``c``/``a`` 两轴的位置。
    """
    target = "acdb,menf,ac,em,bd,fn->"
    result = _equivalence.normalize_equation(target)
    assert result is not None, "接口层没找到等价写法"
    assert result.equation == "cadb,menf,ac,em,bd,fn->", result.equation
    assert result.operand_permutations[0] == (1, 0, 2, 3), result.operand_permutations

    # 用真实语料形状对拍
    from pyqcd.einsum.bench_specialized import _label_sizes
    from pyqcd.einsum._bench_scene import dims
    from pyqcd.einsum import _wick_tasklist as tasklist
    from pyqcd.einsum._dedicated_kernels import parse_equation

    terms, _ = parse_equation(target)
    table = dims("distill")
    shapes = None
    for seq in range(1, 40):
        task = tasklist.lookup(seq)
        if task is None:
            continue
        corpus_terms, _ = parse_equation(task.equation)
        sizes = _label_sizes(corpus_terms, task.roles, table)
        if all(ch in sizes for term in terms for ch in term):
            shapes = [tuple(sizes[ch] for ch in term) for term in terms]
            break
    assert shapes is not None, "找不到覆盖该方程的语料 roles"

    rng = np.random.default_rng(20261116)
    ops = [rng.standard_normal(s) + 1j * rng.standard_normal(s) for s in shapes]
    direct = pe.contract(target, *ops, backend="numpy", optimize="greedy")

    mapped = list(ops)
    mapped[0] = np.transpose(mapped[0], result.operand_permutations[0])
    via = pe.contract(result.equation, *mapped, backend="numpy", optimize="greedy")

    reference = np.einsum(target, *ops, optimize="greedy")
    scale = max(float(np.max(np.abs(reference))), 1e-300)
    assert float(np.max(np.abs(via - direct))) <= 1e-12 * scale, "映射后数值不一致"
    assert float(np.max(np.abs(direct - reference))) <= 1e-12 * scale, "直接算就不是真值"


def test_operand_order_mapping_is_numerically_equivalent():
    """操作数位置交换：改方程 + 同序重排张量 → 结果逐元素不变。"""
    target = "menf,cadb,ac,em,bd,fn->"
    result = _equivalence.normalize_equation(target)
    assert result is not None, "接口层没找到等价写法"
    assert result.equation == "cadb,menf,ac,em,bd,fn->", result.equation
    assert result.operand_order == (1, 0, 2, 3, 4, 5), result.operand_order

    rng = np.random.default_rng(7)
    sizes = {"c": 3, "a": 3, "d": 3, "b": 3, "m": 3, "e": 3, "n": 3, "f": 3}
    lhs = target.split("->")[0].split(",")
    ops = [rng.standard_normal(tuple(sizes[c] for c in t)) for t in lhs]
    direct = pe.contract(target, *ops, backend="numpy", optimize="greedy")

    reordered = [ops[i] for i in result.operand_order]
    via = pe.contract(result.equation, *reordered, backend="numpy", optimize="greedy")
    scale = max(abs(direct), 1e-300)
    assert float(np.max(np.abs(via - direct))) <= 1e-12 * scale, "映射后数值不一致"


def test_output_axis_permutation_mapping_is_numerically_equivalent():
    """输出段轴序改写：结果需对输出做 transpose。第三族变换，原先漏掉了。

    用户报的第二个场景：`...->GMHew`（MISS）与 `...->MGHew`（HIT）只差输出轴
    前两位。收缩完全相同，结果互为 transpose。

    断言三件事：①映射出 `output-axes` 且置换正确；②按置换转回后与直接算逐元素
    相同；③映射后的方程真的能命中语料。
    """
    target = ("cwdx,eufv,mknl,oapb,isjt,qgrh,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx"
              "->GMHew")
    corpus = target.replace("->GMHew", "->MGHew")
    assert _specialize.match(target) is None, "该写法本不应命中"
    assert _specialize.match(corpus) is not None, "语料式应当命中"

    result = _equivalence.normalize_equation(target)
    assert result is not None, "接口层没找到等价写法"
    assert result.equation == corpus, result.equation
    assert result.kind == "output-axes", result.kind
    assert result.output_permutation == (1, 0, 2, 3, 4), result.output_permutation

    # 数值等价：16 个操作数、每字母长 2 的最小形状
    letters = {ch for ch in target.split("->")[0] if ch.isalpha()}
    sizes = {ch: 2 for ch in letters}
    lhs = target.split("->")[0].split(",")
    rng = np.random.default_rng(20261116)
    ops = [rng.standard_normal(tuple(sizes[c] for c in term)) for term in lhs]

    direct = pe.contract(target, *ops, backend="numpy", optimize="greedy")
    via = pe.contract(result.equation, *ops, backend="numpy", optimize="greedy")
    restored = np.transpose(via, result.output_permutation)
    scale = max(float(np.max(np.abs(direct))), 1e-300)
    assert restored.shape == direct.shape, f"{restored.shape} vs {direct.shape}"
    assert float(np.max(np.abs(restored - direct))) <= 1e-10 * scale, "转回后不等价"


def test_output_axis_family_covers_pairwise_rotation_and_cycle():
    """输出轴序族必须覆盖三种常见形态，而不只是两两互换。

    用户实测：`->GMHew`（两两）、`->MGHwe`（整体轮转）、`->HMGew`（子段三循环
    `(2,0,1,3,4)`）、`->wMGHe`（另一种轮转）都要能映射到语料式。
    第一版只做两两互换 + 秩<=4 全排列，后两种全落空。
    """
    base = ("cwdx,eufv,mknl,oapb,isjt,qgrh,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx"
            "->MGHew")
    for permuted in ("GMHew", "MGHwe", "HMGew", "wMGHe"):
        equation = base.replace("->MGHew", "->" + permuted)
        result = _equivalence.normalize_equation(equation)
        assert result is not None, f"->{permuted} 找不到等价写法"
        assert result.equation == base, f"->{permuted} 映射到 {result.equation}"
        assert _specialize.match(result.equation) is not None


def test_identity_output_needs_no_permutation():
    """输出轴序与原式相同时不该产生 output_permutation。"""
    result = _equivalence.normalize_equation("cadb,menf,ac,em,bd,fn->")
    assert result is not None and result.kind == "identity", result
    assert result.output_permutation is None, result.output_permutation


def test_mapping_really_reaches_the_specialized_path():
    """映射的**目的**：等价写法能命中语料，直接写法不能。

    只断言 `_specialize.match` 的命中差异——这是本层存在的理由。
    """
    target = "acdb,menf,ac,em,bd,fn->"
    assert _specialize.match(target) is None, "该写法本不应命中"
    result = _equivalence.normalize_equation(target)
    assert result is not None
    assert _specialize.match(result.equation) is not None, "等价写法应当命中"


# ------------------------------------------------------------------ 2. 保守性

def test_invalid_or_unmappable_inputs_do_not_raise():
    """非法 / 无法映射的输入一律返回 None 或空列表，绝不抛异常。"""
    for bad in ["", "->", "abc", "a,->", ",", "a->b", None, 123,
                "a,b->c", "ij,jk", "ij->ijk", "a,,b->"]:
        _equivalence.normalize_equation(bad)          # 不抛即通过
        _equivalence.find_equivalents(bad, limit=3)


def test_already_matching_equation_is_not_rewritten():
    """本身已命中时必须返回 identity，不产生任何张量调整。"""
    hit = "cadb,menf,ac,em,bd,fn->"
    result = _equivalence.normalize_equation(hit)
    assert result is not None and result.kind == "identity", result
    assert not result.operand_permutations, result.operand_permutations
    assert result.operand_order is None, result.operand_order


def test_unmappable_equation_returns_none():
    """没有任何等价写法能命中时返回 None（不猜）。"""
    assert _equivalence.normalize_equation("qz,wx->qz") is None


def test_high_rank_enumeration_stays_bounded():
    """高秩方程不得因阶乘枚举而爆炸。"""
    import time
    start = time.perf_counter()
    candidates = _equivalence.find_equivalents("abcdefgh,ijklmnop->", limit=5)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"枚举耗时 {elapsed:.2f}s 过久"
    assert len(candidates) <= 5, len(candidates)


# ------------------------------------------------------------------ 3. 集成

def test_verify_specialized_reports_the_equivalence():
    """`verify_specialized` 的 MISS 报告里必须带出等价映射与张量调整。"""
    report = verify_specialized.query("acdb,menf,ac,em,bd,fn->")
    assert not report["matched"], "该写法本不应命中"
    equivalent = report.get("equivalence")
    assert equivalent is not None, report
    assert equivalent["rewritten"] is True, equivalent
    assert equivalent["equation"] == "cadb,menf,ac,em,bd,fn->", equivalent
    assert equivalent["kind"] == "operand-axes", equivalent
    assert equivalent["operand_permutations"][0] == [1, 0, 2, 3], equivalent

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        verify_specialized.main(["acdb,menf,ac,em,bd,fn->"])
    text = buffer.getvalue()
    for token in ("等价写法", "cadb,menf,ac,em,bd,fn->", "transpose(1, 0, 2, 3)"):
        assert token in text, f"输出缺少 {token!r}：{text}"


def test_verify_specialized_json_carries_the_equivalence():
    report = verify_specialized.query("acdb,menf,ac,em,bd,fn->")
    payload = json.dumps(report, ensure_ascii=False)
    assert "cadb,menf,ac,em,bd,fn->" in payload
    assert report["equivalence"]["operand_permutations"][0] == [1, 0, 2, 3]


# ------------------------------------------------------------- 4. 复合变换

#: 复合用例：位置次序与轴序**同时**变化，①②③ 任一单族都映射不到语料式。
#: 每条都是把 ``cadb,menf,ac,em,bd,fn->``（seq=1）做一次"交换两操作数 +
#: 把其中一个反转"，因此只有"位置 + 轴序"组合才能还原。
COMPOSITE_TARGETS = (
    "cadb,menf,ac,em,fn,db->",     # 换第 4/5 位，换进来的 db 反转成 bd
    "cadb,menf,ac,em,nf,bd->",     # 换第 4/5 位，换进来的 nf 反转成 fn
    "cadb,menf,fn,em,bd,ca->",     # 换第 2/5 位，换进来的 ca 反转成 ac
    "cadb,menf,nf,em,bd,ac->",     # 换第 2/5 位，换进来的 nf 反转成 fn
)


def test_composite_mapping_is_numerically_equivalent():
    """复合变换（位置次序 + 轴序）：真实生产形状对拍，逐元素不变。

    验收口径与其余族一致：``direct`` == ``np.einsum`` 真值，且按返回的
    ``operand_order`` 重排 + ``operand_permutations`` 转置后再用映射方程算出的
    结果与之相同（``rel <= 1e-12``）。**必须逐条断言 ``kind`` 真的是复合族**，
    否则这条用例会在实现退化回单族时静默放过。
    """
    sizes = _seq1_label_sizes()
    for target in COMPOSITE_TARGETS:
        assert _specialize.match(target) is None, f"{target} 本不应直接命中"
        result = _equivalence.normalize_equation(target)
        assert result is not None, f"{target} 找不到等价写法"
        assert result.equation == CORPUS_SEQ1, f"{target} 映射到 {result.equation}"
        assert result.kind == "operand-order+axes", (target, result.kind)
        assert result.operand_order is not None, (target, result.operand_order)
        assert any(result.operand_permutations), (target, result.operand_permutations)
        assert _specialize.match(result.equation) is not None
        _assert_equivalent(target, result, sizes)


def test_composite_mapping_covers_the_user_reported_pair_swap_plus_reversal():
    """用户实际例子：第 4/5 位互换且换进来的 ``db`` 必须反转成 ``bd``。

    这条单独钉住"位置 + 轴序"是**同一次映射**给出的（不是两次单独改写），
    且两者都出现在 ``operand_order`` / ``operand_permutations`` 里。
    """
    target = "cadb,menf,ac,em,fn,db->"
    result = _equivalence.normalize_equation(target)
    assert result is not None and result.equation == CORPUS_SEQ1, result
    assert result.kind == "operand-order+axes", result.kind
    assert result.operand_order == (0, 1, 2, 3, 5, 4), result.operand_order
    # perm 用**新次序**索引：新第 4 位拿到源 ``db``，需 transpose(1,0) 才是 ``bd``
    assert result.operand_permutations[4] == (1, 0), result.operand_permutations
    # 报告要同时说清"位置怎么变"和"轴怎么转"
    assert "操作数次序改为 [0, 1, 2, 3, 5, 4]" in result.description, result.description
    assert "transpose(1, 0)" in result.description, result.description


def test_operand_axes_plus_output_axes_composite_is_equivalent():
    """第二类复合：单操作数轴序 × 输出轴序同时变（输入侧与输出侧都动）。

    用 seq=1650（13 操作数、输出秩 2）的真实形状：把第 0 个操作数反转、同时把
    输出段 reversal，只有"轴序 + 输出轴序"组合能还原。结果必须先
    ``transpose(output_permutation)`` 才是原方程答案。
    """
    from pyqcd.einsum import _wick_tasklist as tasklist
    from pyqcd.einsum._bench_scene import dims
    from pyqcd.einsum._dedicated_kernels import parse_equation
    from pyqcd.einsum.bench_specialized import _label_sizes

    task = tasklist.lookup(1650)
    terms, out = parse_equation(task.equation)
    sizes = _label_sizes(terms, task.roles, dims("distill"))
    reverted = list(terms)
    reverted[0] = reverted[0][::-1]
    target = ",".join(reverted) + "->" + out[::-1]

    assert _specialize.match(target) is None, "该写法本不应命中"
    result = _equivalence.normalize_equation(target)
    assert result is not None, "接口层没找到等价写法"
    assert result.equation == task.equation, result.equation
    assert result.kind == "operand-axes+output-axes", result.kind
    assert result.operand_permutations[0] is not None, result.operand_permutations
    assert result.output_permutation is not None, result.output_permutation
    _assert_equivalent(target, result, sizes)


def test_composite_enumeration_is_bounded():
    """16 操作数方程不得阶乘爆炸：耗时受控、返回条数受 ``limit`` 约束。

    口径变更（2026-11-16）：本用例原先断言 ``_candidate_equations`` /
    ``_MAX_CANDIDATES`` / ``_MAX_COMPOSITE_CANDIDATES``——那是"枚举候选方程"实现
    的内部上界。实现改成"结构不变量 + 构造式同构"后枚举不再存在，这些符号也没有
    了。契约改为断言**公开入口**的有界性，要守住的仍是原来那条要求：语料最大元数
    16，全排列 × 每操作数全轴排列是天文数字，绝不能出现在调用路径上。
    """
    import time

    cases = (
        # 真实 16 操作数语料方程（每操作数秩 4，输出秩 5），输出轴序需改写
        "cwdx,eufv,mknl,oapb,isjt,qgrh,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->GMHew",
        # 病态：16 操作数、每操作数秩 8
        ",".join("abcdefgh" for _ in range(16)) + "->",
    )
    for equation in cases:
        assert len(_parse(equation)[0]) == 16, "契约用例必须是 16 操作数"
        start = time.perf_counter()
        rows = _equivalence.find_equivalents(equation, limit=8)
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, f"枚举耗时 {elapsed:.2f}s 超出 10 s 上界"
        assert len(rows) <= 8, len(rows)
        for row in rows:
            _assert_structure(equation, row, f"find_equivalents({equation[:24]}…)")
        # 有界不等于"没产出"：真实 16 操作数方程必须真的给出内容
        if "cwdx" in equation:
            assert rows, "16 操作数方程一条结果都没产出（并非有界性，而是功能退化）"


def test_composite_enumeration_prefers_the_smallest_change():
    """最小改动优先：只差一个轴对的方程，返回的映射不得动其它操作数。"""
    # 只差第 0 个操作数的 c/a 两位 → 必须只动第 0 位，且必须是单轴序族
    result = _equivalence.normalize_equation("acdb,menf,ac,em,bd,fn->")
    assert result is not None and result.kind == "operand-axes", result
    assert result.operand_order is None, result.operand_order
    touched = [i for i, perm in enumerate(result.operand_permutations)
               if perm is not None]
    assert touched == [0], touched

    # 只换两个操作数位置 → 必须只给 order，不给任何轴序
    result = _equivalence.normalize_equation("menf,cadb,ac,em,bd,fn->")
    assert result is not None and result.kind == "operand-order", result
    assert not result.operand_permutations, result.operand_permutations

    # 复合族内部也必须挑"动得最少"的那条：两两互换 + 一次反转，
    # 而不是"三次位置轮转 + 两次反转"之类的等价长链。
    result = _equivalence.normalize_equation("cadb,menf,ac,em,fn,db->")
    assert result is not None, "找不到映射"
    assert result.operand_order is not None, result.operand_order
    moved = [i for i, j in enumerate(result.operand_order) if i != j]
    assert len(moved) == 2, f"位置改动不止两个：{result.operand_order}"
    perms = [p for p in (result.operand_permutations or ()) if p is not None]
    assert len(perms) == 1, f"轴序改动不止一个操作数：{result.operand_permutations}"


def test_composite_report_shows_both_ordering_and_permutation_lines():
    """复合变换的 MISS 报告里 `张量次序` 与 `张量调整` 必须同时出现且自洽。"""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        verify_specialized.main(["cadb,menf,ac,em,fn,db->"])
    text = buffer.getvalue()
    assert "  张量次序  : 按 [0, 1, 2, 3, 5, 4] 重排操作数" in text, text
    assert "  张量调整  : -, -, -, -, transpose(1, 0), -" in text, text
    assert "  变换类型  : operand-order+axes" in text, text
    # 两行的索引口径必须一致：次序说第 4 位来自源第 5 位，调整说第 4 位要转
    order_line = [line for line in text.splitlines() if "张量次序" in line][0]
    perm_line = [line for line in text.splitlines() if "张量调整" in line][0]
    assert "5, 4" in order_line and perm_line.count("transpose") == 1, text


# ------------------------------------------------------------------ 5. 开关

@contextlib.contextmanager
def _env(**values):
    """临时设置/删除环境变量，退出时**必定**恢复（不污染其它用例）。"""
    saved = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


#: 引入等价映射**之前**的 MISS 报告原文（`git show fd26357e:...` 实跑抓取）。
LEGACY_MISS_REPORT = (
    "方程        : acdb,menf,ac,em,bd,fn->\n"
    "归一化      : acdb,menf,ac,em,bd,fn->\n"
    "拓扑签名    : ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1), (5, 4), (3, 2), (7, 6), ())\n"
    "元数/槽位   : 6 个操作数 / 16 个指标槽（输出秩 0）\n"
    "摘要        : 91104304ce1d8e05ab572591\n"
    "特化开关    : 开启（PYQCD_EINSUM_SPECIALIZE）\n"
    "结果        : MISS\n"
    "  原因      : 既不在 wick_12cases 语料里，也没有形状特化路径，"
    "走通用分步路径（无特化内核）\n"
)


def test_equivalence_is_enabled_by_default():
    """不设环境变量时开关为 True，映射正常工作。"""
    with _env(PYQCD_EINSUM_EQUIVALENCE=None):
        assert _equivalence.enabled() is True
        result = _equivalence.normalize_equation("acdb,menf,ac,em,bd,fn->")
        assert result is not None, "默认应当开启"
        assert result.equation == CORPUS_SEQ1, result.equation


def test_disabling_the_switch_turns_every_entry_point_into_a_noop():
    """``PYQCD_EINSUM_EQUIVALENCE=0``：**产出映射的入口**全部变成空操作。

    开关只拦"给映射"的两个入口（``normalize_equation`` / ``find_equivalents``）；
    ``explain`` 是纯报告，**不**读开关——两个 CLI 的 ``--help`` 都写明
    ``--equivalence true`` 压过环境变量，而它们只经 ``explain`` 取报告（实测：
    ``test.py`` 不做环境变量翻转，若 ``explain`` 也拦开关，``--equivalence true``
    在环境变量为 0 时永远拿不到映射）。关闭时 ``explain`` 如实报告，并额外标注
    ``disabled=True`` 与 :data:`_equivalence.DISABLED_REASON`。
    """
    for flag in ("0", "false", "off", "no", "FALSE"):
        with _env(PYQCD_EINSUM_EQUIVALENCE=flag):
            assert _equivalence.enabled() is False, flag
            assert _equivalence.normalize_equation(
                "acdb,menf,ac,em,bd,fn->") is None, flag
            assert _equivalence.find_equivalents(
                "acdb,menf,ac,em,bd,fn->") == [], flag
            explained = _equivalence.explain("acdb,menf,ac,em,bd,fn->")
            assert explained.get("disabled") is True, explained
            assert explained.get("disabled_reason") == _equivalence.DISABLED_REASON, \
                explained
            assert explained["rewritten"] is True, explained
    # 恢复后必须立刻重新生效（开关在**调用时**读，不在 import 时求值）
    with _env(PYQCD_EINSUM_EQUIVALENCE=None):
        assert _equivalence.enabled() is True


def test_disabled_verify_specialized_output_matches_the_pre_change_original():
    """关闭后 MISS 报告与**引入本功能之前**逐字符相同——最强的一条证据。

    期望原文取自 ``git show fd26357e:pyqcd/einsum/verify_specialized.py``
    在本机实跑的字节流，直接嵌在 :data:`LEGACY_MISS_REPORT` 里。
    """
    buffer = io.StringIO()
    with _env(PYQCD_EINSUM_EQUIVALENCE="0"):
        with contextlib.redirect_stdout(buffer):
            code = verify_specialized.main(["acdb,menf,ac,em,bd,fn->"])
        report = verify_specialized.query("acdb,menf,ac,em,bd,fn->")
    text = buffer.getvalue()
    assert code == 1, code
    assert text == LEGACY_MISS_REPORT, (
        f"关闭后的输出与改动前不一致：\n--- got ---\n{text}"
        f"--- expected ---\n{LEGACY_MISS_REPORT}")
    assert "equivalence" not in report, report
    assert "等价写法" not in text, text


# ------------------------------------- 6. 置换方向（非对称形状）与空白鲁棒性

#: 用户用例 1（基线）：精确命中语料（seq=8628）。用例 2~4 由它派生。
USER_EQUATION = (
    "cwdx,eufv,mknl,oapb,isjt,qgrh,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHew")
#: 用例 2：只把输出段改成 ``MGHwe``（末两轴互换）。
USER_CASE_2 = (
    "cwdx,eufv,mknl,oapb,isjt,qgrh,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->MGHwe")
#: 用例 3：第 15 个操作数 ``tvx``→``txv``（轴序）+ 输出轴序 ``MGHwe``。
USER_CASE_3 = (
    "cwdx,eufv,mknl,oapb,isjt,qgrh,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,txv->MGHwe")
#: 用例 4：再叠加 ``mo,ac`` 换位（操作数位置）+ 轴序 + 输出轴序，三重复合。
USER_CASE_4 = (
    "cwdx,eufv,mknl,oapb,isjt,qgrh,mo,ac,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,txv->MGHwe")

#: 输出段的**非自逆**置换。``transpose(result, perm)`` 就是调用方的还原方向；
#: 三循环 ``(2,0,1,3,4)`` 与五循环 ``(4,0,1,2,3)`` 的逆都与自身不同，所以"方向用反"
#: 必然改变形状——两两互换做不到（见 :data:`SELF_INVERSE_OUTPUT_PERMS`）。
NON_SELF_INVERSE_OUTPUT_PERMS = (("HMGew", (2, 0, 1, 3, 4)),
                                 ("wMGHe", (4, 0, 1, 2, 3)))

#: 输出段的**自逆**置换（两两互换）：``perm == inverse(perm)``，正反方向同形，
#: 结构上不可能验证方向。这里只断言数值等价，并**显式标注**该局限，避免它们
#: 被当成"方向也验证过了"。
SELF_INVERSE_OUTPUT_PERMS = ("GMHew", "MGHwe")

#: seq=1 的 8 个标签长度全不同（2..9）：任何轴序错误都改变形状。
CORPUS_SEQ1_SIZES = {"c": 2, "a": 3, "d": 4, "b": 5, "m": 6, "e": 7, "n": 8, "f": 9}

#: 空白变体 → 规范化写法。第一条是审查代理给出的**最小复现**。
WHITESPACE_VARIANTS = (
    ("逗号后一个空格（审查最小复现）", "acdb, menf, ac, em, bd, fn->",
     "acdb,menf,ac,em,bd,fn->"),
    ("行首空白", "   acdb,menf,ac,em,bd,fn->", "acdb,menf,ac,em,bd,fn->"),
    ("行尾空白", "acdb,menf,ac,em,bd,fn->   ", "acdb,menf,ac,em,bd,fn->"),
    ("多个连续空格", "acdb,   menf,  ac,em,bd,fn->", "acdb,menf,ac,em,bd,fn->"),
    ("逗号前后都有空格", "acdb , menf , ac , em , bd , fn ->",
     "acdb,menf,ac,em,bd,fn->"),
    ("术语内插空格", "ac db,menf,ac,em,bd,fn->", "acdb,menf,ac,em,bd,fn->"),
    ("制表符与换行", "acdb,\tmenf,\nac,em,bd,fn->", "acdb,menf,ac,em,bd,fn->"),
    ("操作数换位 + 空格", "menf, cadb, ac, em, bd, fn->",
     "menf,cadb,ac,em,bd,fn->"),
)


def _parse(equation):
    """契约自己的方程解析——**不用**被测实现的 ``_split``。

    "用被测对象验证被测对象"会让解析层的缺陷同时污染两边：空白到底算不算轴位，
    正是本节要裁决的语义，所以解析必须独立于实现。返回 ``(术语列表, 输出段)``。
    """
    text = "".join(str(equation).split())
    assert "->" in text, f"契约用例必须是完整方程：{equation!r}"
    lhs, _, rhs = text.partition("->")
    return lhs.split(","), rhs


def _invert(perm):
    """逆置换：``new[j] = old[perm[j]]`` 的逆满足 ``inverse[perm[j]] = j``。"""
    inverse = [0] * len(perm)
    for index, source in enumerate(perm):
        inverse[source] = index
    return tuple(inverse)


def _user_label_sizes():
    """用户 16 操作数方程的标签长度：**输出轴两两不同**（M2 G3 H4 e5 w6），其余 2。

    为什么必须非对称：所有轴同长时轴序错误既不改变形状也无法一眼看出，只留下
    "数值恰好不同"这条弱线索；长度互不相同后，方向用反直接表现为 shape 不符。
    只有 5 个输出轴需要区分（方向判据落在输出段），其余取 2 以控制规模。
    """
    sizes = {ch: 2 for ch in USER_EQUATION.split("->")[0] if ch.isalpha()}
    sizes.update({"M": 2, "G": 3, "H": 4, "e": 5, "w": 6})
    return sizes


def _run_mapping(equation, result, sizes, *, invert=False, reference_equation=None):
    """按字段语义施加映射，返回 ``(numpy 真值, 映射后结果)``。

    ``invert=True`` 把 ``operand_order`` 与每个置换都换成逆——这是"方向用反了"的
    精确模拟，必须算错，否则说明该用例根本区分不了方向。
    """
    terms, _ = _parse(equation)
    rng = np.random.default_rng(20261116)
    shapes = [tuple(sizes[ch] for ch in term) for term in terms]
    ops = [rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
           for shape in shapes]

    count = len(terms)
    order = (tuple(range(count)) if result.operand_order is None
             else tuple(result.operand_order))
    perms = tuple(result.operand_permutations or ())
    perms = (None,) * count if not perms else perms + (None,) * (count - len(perms))
    out_perm = result.output_permutation
    if invert:
        order = _invert(order)
        perms = tuple(None if perm is None else _invert(perm) for perm in perms)
        out_perm = None if out_perm is None else _invert(out_perm)

    tensors = []
    for index in range(count):
        tensor, perm = ops[order[index]], perms[index]
        tensors.append(tensor if perm is None else np.transpose(tensor, perm))
    # 逆方向本来就会因形状/标签长度不符触发内核回退（RuntimeWarning），那正是判据
    # 本身；别把它刷到测试输出里。正确方向**不**屏蔽任何警告。
    with warnings.catch_warnings():
        if invert:
            warnings.simplefilter("ignore")
        via = pe.contract(result.equation, *tensors, backend="numpy",
                          optimize="greedy")
    if out_perm is not None:
        via = np.transpose(via, out_perm)
    reference = np.einsum(reference_equation or equation, *ops, optimize="greedy")
    return reference, via


def _assert_same(reference, via, label, tol=1e-12):
    """逐元素比对（相对真值绝对值上界），不一致即失败。"""
    scale = max(float(np.max(np.abs(reference))), 1e-300)
    assert via.shape == reference.shape, f"{label}: shape {via.shape} != {reference.shape}"
    delta = float(np.max(np.abs(via - reference)))
    assert delta <= tol * scale, f"{label}: relerr {delta / scale:.3e} > {tol:.0e}"


def _wrong_direction_verdict(equation, result, sizes, reference):
    """把每个置换换成逆之后算一遍：必须崩、形状不符、或数值明显不同。

    返回判定依据（``raise:<异常>`` / ``shape`` / ``value``），供用例说明
    "是哪一层抓住了它"。
    """
    try:
        _, wrong = _run_mapping(equation, result, sizes, invert=True)
    except Exception as exc:                     # noqa: BLE001 - 崩掉同样算被抓到
        return f"raise:{type(exc).__name__}"
    if wrong.shape != reference.shape:
        return "shape"
    scale = max(float(np.max(np.abs(reference))), 1e-300)
    delta = float(np.max(np.abs(wrong - reference)))
    assert delta > 1e3 * 1e-12 * scale, (
        f"{equation}: 逆置换与正确结果只差 {delta / scale:.3e}——该用例区分不了方向")
    return "value"


def _structure_problems(equation, row):
    """字段结构自洽性检查，返回问题列表（空 = 合法）。

    抓的正是缺陷 2：若拆分不去空白、重排又按字符下标走，带空格的术语会被算出
    **比秩长**的 ``perm``（例：2 维操作数给 ``(0, 2, 1)``）。调用方按它转置会
    ``ValueError``；忽略它直接照候选方程算则静默错（实测 relerr 0.53）。
    """
    terms, out = _parse(equation)
    count = len(terms)
    problems = []
    order = (row.get("operand_order") if isinstance(row, dict)
             else getattr(row, "operand_order", None))
    order = tuple(range(count)) if order is None else tuple(order)
    if sorted(order) != list(range(count)):
        problems.append(f"操作数次序不是排列：{order}")
        return problems
    perms = (row.get("operand_permutations") if isinstance(row, dict)
             else getattr(row, "operand_permutations", None)) or ()
    if perms and len(perms) != count:
        problems.append(f"perm 个数 {len(perms)} != 操作数个数 {count}")
    for index, perm in enumerate(perms):
        if perm is None:
            continue
        rank = len(terms[order[index]])
        if len(perm) != rank:
            problems.append(
                f"第 {index} 个操作数秩 {rank}，perm {tuple(perm)} 长度却是 {len(perm)}")
        elif sorted(perm) != list(range(rank)):
            problems.append(
                f"第 {index} 个操作数 perm {tuple(perm)} 不是 0..{rank - 1} 的排列")
    out_perm = (row.get("output_permutation") if isinstance(row, dict)
                else getattr(row, "output_permutation", None))
    if out_perm and sorted(out_perm) != list(range(len(out))):
        problems.append(f"输出 perm {tuple(out_perm)} 不是 0..{len(out) - 1} 的排列")
    return problems


def _assert_structure(equation, row, label):
    problems = _structure_problems(equation, row)
    assert not problems, f"{label}: 畸形映射 {problems}"


def _check_solved(equation, result, label):
    """已给映射的用例统一验收：结构合法 + 命中语料 + 数值还原 relerr <= 1e-12。"""
    assert result.kind != "identity", f"{label}: 该写法本不该直接命中"
    assert _specialize.match(result.equation) is not None, (
        f"{label}: 映射到 {result.equation}，但它并不命中语料")
    _assert_structure(equation, result, label)
    reference, via = _run_mapping(equation, result, _user_label_sizes())
    _assert_same(reference, via, label, tol=1e-12)


def test_direction_output_axes_is_discriminated_both_ways():
    """**方向契约**：输出轴序的三循环 / 五循环，转对方向对、转反方向必错。

    抓的是什么：``output_permutation`` 曾存的是**原始 perm** 而非逆置换
    （`d8354097`），而当时的契约 10 条全绿——因为它把所有标签设成长度 2（对称
    形状）并用 ``->GMHew`` 这种两两互换（自逆）置换，正反方向完全同形。

    这里两道盲区都堵死：输出轴长度 2/3/4/5/6 互不相同，且只挑 ``(2,0,1,3,4)``
    与 ``(4,0,1,2,3)`` 这两个非自逆置换。逆置换会得到 ``(3,4,2,5,6)`` 这类对不上的
    形状，因此"存原始 perm"的版本在这里必然失败。
    """
    sizes = _user_label_sizes()
    output_sizes = [sizes[label] for label in "MGHew"]
    assert len(set(output_sizes)) == len(output_sizes), (
        f"前提失效：输出轴长度必须两两不同，当前 {output_sizes}")
    for suffix, expected in NON_SELF_INVERSE_OUTPUT_PERMS:
        equation = USER_EQUATION.replace("->MGHew", "->" + suffix)
        assert _specialize.match(equation) is None, f"->{suffix} 本不该直接命中"
        result = _equivalence.normalize_equation(equation)
        assert result is not None, f"->{suffix} 找不到等价写法"
        assert _specialize.match(result.equation) is not None, (
            f"->{suffix} 映射到 {result.equation}，但它并不命中语料")
        perm = result.output_permutation
        assert perm is not None, f"->{suffix}: 输出轴序变了却没给 output_permutation"
        assert tuple(perm) == expected, f"->{suffix}: {tuple(perm)} != {expected}"
        assert _invert(perm) != tuple(perm), (
            f"->{suffix}: {perm} 是自逆置换，无法区分方向")
        reference, via = _run_mapping(equation, result, sizes)
        _assert_same(reference, via, f"->{suffix} 正确方向")
        verdict = _wrong_direction_verdict(equation, result, sizes, reference)
        assert verdict == "shape", (
            f"->{suffix}: 逆置换本应因形状不符被抓到，实际判定 {verdict}")
        problems = _structure_problems(equation, result)
        assert not problems, f"->{suffix}: {problems}"


def test_self_inverse_output_permutations_are_marked_as_non_direction_tests():
    """显式标注：``->GMHew`` / ``->MGHwe`` 的 perm **自逆**，正反同形。

    审查代理指出旧契约恰是拿这类"两两互换"当方向覆盖，实际上它们在结构上测不出
    方向。本用例把它们降级为"仅数值等价"，并用 ``array_equal`` 证明正反方向逐位
    相同——这才是"不可区分"的证据。哪天它们不再自逆（实现换了写法），这条会失败，
    提醒把该写法移进方向契约，而不是继续挂在这里冒充方向验证。
    """
    sizes = _user_label_sizes()
    for suffix in SELF_INVERSE_OUTPUT_PERMS:
        equation = USER_EQUATION.replace("->MGHew", "->" + suffix)
        result = _equivalence.normalize_equation(equation)
        assert result is not None, f"->{suffix} 找不到等价写法"
        perm = result.output_permutation
        assert perm is not None, f"->{suffix}: 输出轴序变了却没给 output_permutation"
        assert tuple(perm) == _invert(perm), (
            f"->{suffix}: 期望自逆置换；若已不再自逆，请移入 "
            "test_direction_output_axes_is_discriminated_both_ways")
        reference, via = _run_mapping(equation, result, sizes)
        _assert_same(reference, via, f"->{suffix} 数值等价")
        _, flipped = _run_mapping(equation, result, sizes, invert=True)
        assert np.array_equal(flipped, via), (
            f"->{suffix}: 自逆置换的正反方向应当逐位相同（这正说明它测不出方向）")


def test_direction_operand_axes_three_cycles_are_discriminated():
    """**方向契约**：单操作数轴序的三/四循环，8 个标签长度 2..9 全不同。

    旧契约的轴序用例形状对称、置换也多为自逆互换；这里四个置换
    ``(2,0,1,3)`` / ``(1,2,0,3)``（互逆的三循环）与 ``(1,2,3,0)`` / ``(3,0,1,2)``
    （互逆的四循环）都**非自逆**，且 ``cadb`` 四轴长度 2/3/4/5 互不相同：映射方向
    用反要么形状不符，要么在 ``einsum`` 里因标签长度冲突直接报错。

    注：``(2,3,0,1)`` 看着像循环，其实是两个不相交对换（``0↔2``、``1↔3``）的复合、
    自逆，正反同形，**不能**用作方向证据——本条用例的前提断言会把这类误选挡下来。
    """
    corpus_term = "cadb"
    for applied in ((2, 0, 1, 3), (1, 2, 0, 3), (1, 2, 3, 0), (3, 0, 1, 2)):
        permuted = "".join(corpus_term[index] for index in applied)
        equation = permuted + ",menf,ac,em,bd,fn->"
        assert _invert(applied) != tuple(applied), f"前提失效：{applied} 自逆"
        assert _specialize.match(equation) is None, f"{equation} 本不该直接命中"
        result = _equivalence.normalize_equation(equation)
        assert result is not None, f"{equation} 找不到等价写法"
        assert result.equation == CORPUS_SEQ1, f"{equation} 映射到 {result.equation}"
        perm = (result.operand_permutations or (None,))[0]
        assert perm is not None, f"{equation}: 轴序没还原，却给了映射"
        assert tuple(perm) == _invert(applied), (
            f"{equation}: perm {tuple(perm)} != 逆置换 {_invert(applied)}")
        reference, via = _run_mapping(equation, result, CORPUS_SEQ1_SIZES)
        _assert_same(reference, via, equation)
        verdict = _wrong_direction_verdict(equation, result, CORPUS_SEQ1_SIZES, reference)
        assert verdict == "shape" or verdict.startswith("raise"), verdict
    # 前提自检：长度全不同，否则"形状就能抓到方向"不成立
    assert len(set(CORPUS_SEQ1_SIZES.values())) == len(CORPUS_SEQ1_SIZES), \
        CORPUS_SEQ1_SIZES


def test_direction_operand_order_three_cycle_is_discriminated():
    """**方向契约**：操作数位置的**三循环**（不是两两互换），错方向必被抓到。

    ``operand_order`` 也是置换，两两互换（自逆）同样测不出方向；这里用
    ``(2,0,1,3,4,5)`` 的循环移位，逆置换成 ``(1,2,0,3,4,5)`` 会把 4 维操作数放进
    2 维槽位——非对称形状下一眼就崩。
    """
    terms = CORPUS_SEQ1.split("->")[0].split(",")
    applied = (2, 0, 1, 3, 4, 5)
    equation = ",".join(terms[index] for index in applied) + "->"
    assert _invert(applied) != tuple(applied), f"前提失效：{applied} 自逆"
    assert _specialize.match(equation) is None, f"{equation} 本不该直接命中"
    result = _equivalence.normalize_equation(equation)
    assert result is not None, f"{equation} 找不到等价写法"
    assert result.equation == CORPUS_SEQ1, f"{equation} 映射到 {result.equation}"
    assert result.operand_order is not None, f"{equation}: 位置变了却没给 operand_order"
    assert tuple(result.operand_order) == _invert(applied), (
        f"{equation}: {tuple(result.operand_order)} != {_invert(applied)}")
    reference, via = _run_mapping(equation, result, CORPUS_SEQ1_SIZES)
    _assert_same(reference, via, equation)
    verdict = _wrong_direction_verdict(equation, result, CORPUS_SEQ1_SIZES, reference)
    assert verdict == "shape" or verdict.startswith("raise"), verdict


def test_whitespace_never_becomes_an_axis_position():
    """**缺陷 2 的正面断言**：空白不是轴位，``perm`` 长度必须等于操作数秩。

    旧实现 ``_split`` 不去空白、``_permuted`` 按字符下标重排，于是
    ``'acdb, menf, ac, em, bd, fn->'`` 里带前导空格的术语算出比秩长 1 的 perm
    （2 维操作数给 ``(0, 2, 1)``）。后果二选一：按 perm 转置 → ``ValueError``；
    忽略 perm 直接用候选方程 → relerr 0.53 静默错。逐族统计 ``operand-axes``
    2352/3543 条中招，且坏 perm 会直接印在 ``verify_specialized`` 的"张量调整"行里。

    断言的是**正确行为**（不存在畸形 perm），因此无论谁修、怎么修都能通过。
    """
    for name, variant, _canonical in WHITESPACE_VARIANTS:
        result = _equivalence.normalize_equation(variant)
        assert result is not None, f"{name}: {variant!r} 找不到等价写法"
        _assert_structure(variant, result, name)
        for row in _equivalence.find_equivalents(variant, limit=10):
            _assert_structure(variant, row, f"{name} / find_equivalents")
    # 坏 perm 的具体形态（2 维操作数给长度 3 的 perm）必须不存在
    victim = "acdb, menf, ac, em, bd, fn->"
    result = _equivalence.normalize_equation(victim)
    ranks = {index: len(term) for index, term in enumerate(_parse(victim)[0])}
    for index, perm in enumerate(result.operand_permutations or ()):
        assert perm is None or len(perm) == ranks[index], (
            f"第 {index} 个操作数秩 {ranks[index]} 却给了 perm {tuple(perm)}")


def test_whitespace_variants_produce_the_same_mapping_as_the_canonical_form():
    """空白变体与规范化写法必须给出**同一条**映射，且数值逐位一致。

    "同一条"覆盖四个字段：``equation``（按解析后的术语比较）、``operand_order``、
    ``operand_permutations``、``output_permutation``。空白本来就不该改变语义，
    所以这条同时挡住了"空格挤进轴序"与"空格挤进位置"两种走样。
    """
    for name, variant, canonical in WHITESPACE_VARIANTS:
        got = _equivalence.normalize_equation(variant)
        want = _equivalence.normalize_equation(canonical)
        assert got is not None and want is not None, f"{name}: 有一侧找不到映射"
        assert _parse(got.equation) == _parse(want.equation), (
            f"{name}: {got.equation} != {want.equation}")
        assert tuple(got.operand_order or ()) == tuple(want.operand_order or ()), name
        assert tuple(got.operand_permutations or ()) == tuple(
            want.operand_permutations or ()), name
        assert got.output_permutation == want.output_permutation, name
        reference, via = _run_mapping(variant, got, CORPUS_SEQ1_SIZES,
                                        reference_equation=canonical)
        _assert_same(reference, via, name)


def test_user_case_1_baseline_hits_the_corpus_entry():
    """用户用例 1（基线）：精确命中语料，映射是 identity，不动任何张量。"""
    assert _specialize.match(USER_EQUATION) is not None, "基线式应当命中语料"
    result = _equivalence.normalize_equation(USER_EQUATION)
    assert result is not None and result.kind == "identity", result
    assert not result.operand_permutations, result.operand_permutations
    assert result.operand_order is None and result.output_permutation is None, result


def test_user_case_2_output_axis_order_is_solved_and_numerically_exact():
    """用户用例 2（仅输出轴序 ``->MGHwe``）：有解，还原 relerr <= 1e-12。

    该用例的 ``output_permutation`` 是末两轴的互换（自逆），所以它验证的是
    "解对了"，**不是**"方向唯一"——方向由 ``test_direction_output_axes_...`` 负责。
    """
    result = _equivalence.normalize_equation(USER_CASE_2)
    assert result is not None, "用户用例 2 本应有解"
    _check_solved(USER_CASE_2, result, "用例 2")


def test_user_case_3_axis_order_plus_output_is_solved_and_numerically_exact():
    """用户用例 3（轴序 ``txv`` + 输出轴序）：有解，还原 relerr <= 1e-12。"""
    result = _equivalence.normalize_equation(USER_CASE_3)
    assert result is not None, "用户用例 3 本应有解"
    _check_solved(USER_CASE_3, result, "用例 3")


def test_user_case_4_order_plus_axes_plus_output_is_solved_or_explained():
    """用户用例 4（位置 + 轴序 + 输出轴序三重复合）：曾有解不了的说法。

    契约**不假装它必过**：有解就要求数值还原 relerr <= 1e-12（并命中语料）；
    无解则要求 ``explain()`` 给出明确原因，并标记 SKIP（"等待规范形式重写"），
    绝不把"无解"当成通过。
    """
    result = _equivalence.normalize_equation(USER_CASE_4)
    if result is not None:
        _check_solved(USER_CASE_4, result, "用例 4")
        return
    report = _equivalence.explain(USER_CASE_4)
    reason = report.get("reason")
    assert isinstance(reason, str) and reason.strip(), (
        f"无解时必须给出明确原因，实得 {report!r}")
    raise unittest.SkipTest(f"用例 4 当前无解（等待规范形式重写）：{reason}")


def test_sixteen_operand_normalize_is_not_factorial():
    """16 操作数方程的 ``normalize_equation`` 必须毫秒级，不许阶乘爆炸。

    为什么不用紧上界：本机实测 ~1 ms，但共享机负载会让单次测量抖动数倍。
    这里取 5 次的**最小值**（最小值对负载尖峰最鲁棒）再比 200 ms 的宽松上界
    （约 200 倍余量）。写死 ``assert elapsed < 1e-3`` 会变成负载敏感的 flaky
    用例——这正是本仓库其它计时用例吃过的亏。第一次调用含语料索引构建，
    所以先预热一次再计时。
    """
    import time

    equation = USER_CASE_2                     # MISS 路径才真正做映射
    _equivalence.normalize_equation(equation)  # 预热：建语料索引
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        _equivalence.normalize_equation(equation)
        timings.append(time.perf_counter() - start)
    best = min(timings)
    assert best < 0.2, (
        f"16 操作数方程 normalize_equation 最快 {best * 1e3:.1f} ms，超出 200 ms 上界")


# ------------------- 7. 结构键不变性 / 零枚举 / 旧覆盖缺口 / 独立性 / 性能
#
# 这一节钉住"结构不变量 + 构造式同构"这套实现本身该有的性质（不是数值行为）：
#   * 等价 ⟹ 同键（否则会漏判），真不等价 ⟹ 不同键（否则桶内白跑且掩盖误判）；
#   * 默认路径零枚举；
#   * 旧实现的三个已实测缺口（`orders[:96]` 截断、三重复合、空白当轴位）闭掉；
#   * 依赖面收敛、查询与建索引的耗时有上界。

#: 任务给的三重复合（与 `pyqcd.testing.einsum.verify_equivalence.USER16_LAYERED`
#: 同一形态）：位置 4<->5 + 新位置 5 的 ``isjt``->``istj`` + 输出 ``MGHew``->``GMHew``。
THREE_LAYER = (
    "cwdx,eufv,mknl,oapb,qgrh,istj,ac,mo,Ggi,Hkq,su,bd,fnp,Mhj,Mlr,tvx->GMHew")


def _variants(equation):
    """从一条方程派生等价写法：改名 / 换位 / 轴序 / 输出轴序 / 三重复合。"""
    terms, out = _parse(equation)
    order = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rows = []
    for shift in (1, 7, 25):
        table = {}
        for label in sorted({c for term in terms for c in term}):
            table[label] = order[(order.index(label) + shift) % len(order)]
        rows.append(",".join("".join(table[c] for c in term) for term in terms)
                    + "->" + "".join(table[c] for c in out))
    for left, right in ((0, 1), (0, len(terms) - 1), (len(terms) - 2, 1)):
        swapped = list(terms)
        swapped[left], swapped[right] = swapped[right], swapped[left]
        rows.append(",".join(swapped) + "->" + out)
    rows.append(",".join([terms[0][::-1]] + list(terms[1:])) + "->" + out)
    if out:
        rows.append(",".join(terms) + "->" + out[::-1])
        rows.append(",".join(terms) + "->" + out[1:] + out[:1])
    composite = list(terms)
    composite[0], composite[-1] = composite[-1][::-1], composite[0][::-1]
    if out:
        rows.append(",".join(composite) + "->" + out[1:] + out[:1])
    return rows


def _shape_key(equation):
    parsed = _equivalence._split(equation)
    assert parsed is not None, equation
    return _equivalence._Shape(*parsed).key


def test_structural_key_is_invariant_under_every_equivalence_transformation():
    """**无漏判的实现级证据**：桶键对改名 / 换位 / 轴序 / 输出轴序 / 复合全不变。

    这条是本层重写的核心：只要"等价 ⟹ 同键"，查询就一定能落进候选桶；桶内再用
    构造式同构确认。旧的枚举式实现没有这层保证，只能靠"上界内枚举到"来覆盖，
    于是出现 16 操作数 ``swap(10,11)`` 无解这类缺口。
    """
    for equation in (CORPUS_SEQ1, USER_EQUATION,
                     "ambn,codp,egfh,ce,mo,Mbdf,Nnph->MNag"):
        base = _shape_key(equation)
        for variant in _variants(equation):
            assert _shape_key(variant) == base, (
                f"{equation!r} 的变形 {variant!r} 落到了别的桶")


def test_bucket_key_separates_contractions_that_are_not_equivalent():
    """真不等价的两条方程**不能**同键（否则误判风险只能靠桶内偶然不命中兜住）。"""
    for left, right in (
            ("ij,jk->ik", "ij,jk->i"),
            ("ii->i", "ij->i"),
            (CORPUS_SEQ1, "cadb,menf,ac,em,bd,fm->"),
    ):
        assert _shape_key(left) != _shape_key(right), f"{left} 与 {right} 同键"


def test_truly_non_equivalent_equations_are_reported_as_no_mapping():
    """真正不同的收缩必须返回 ``None``，绝不能给映射（误判比漏判危险得多）。"""
    # 不同收缩：`ij,jk->i` 与 `ij,jk->ik` 的求和范围不同
    assert _equivalence.normalize_equation("ij,jk->i") is None
    # 对角 vs 一般：`ii->i` 与 `ij->i`
    diagonal = _shape_key("ii->i")
    probe = lambda candidate: (                          # noqa: E731
        _equivalence._split(candidate) is not None
        and _shape_key(candidate) == diagonal)
    assert _equivalence.normalize_equation("ij->i", probe=probe) is None
    # 语料形态改一个标签，结构就不再同构
    assert _equivalence.normalize_equation("cadb,menf,ac,em,bd,fm->") is None


def test_three_layer_composite_is_solved_with_the_minimal_fields():
    """任务给的三重复合必须**有解**（不是 SKIP），且三个字段各自只动必须动的。

    旧实现这里直接 MISS（三族叠加不在同一次枚举里）；新实现必须命中，并且桶内
    排序要挑出**动得最少**的那条语料条目（否则会给出四个位置的重排）。
    """
    assert _specialize.match(THREE_LAYER) is None, "该写法本不应直接命中"
    result = _equivalence.normalize_equation(THREE_LAYER)
    assert result is not None, "三重复合无解"
    assert result.equation == USER_EQUATION, result.equation
    assert result.kind == "operand-order+axes+output-axes", result.kind
    assert tuple(result.operand_order) == (0, 1, 2, 3, 5, 4) + tuple(range(6, 16)), \
        result.operand_order
    assert [index for index, perm in enumerate(result.operand_permutations)
            if perm is not None] == [4], result.operand_permutations
    assert tuple(result.operand_permutations[4]) == (0, 1, 3, 2), \
        result.operand_permutations[4]
    assert tuple(result.output_permutation) == (1, 0, 2, 3, 4), \
        result.output_permutation
    _assert_structure(THREE_LAYER, result, "三重复合")
    assert _specialize.match(result.equation) is not None
    reference, via = _run_mapping(THREE_LAYER, result, _user_label_sizes())
    _assert_same(reference, via, "三重复合", tol=1e-12)


def test_old_coverage_gap_pair_swaps_on_sixteen_operands_are_closed():
    """16 操作数的**全部**两两互换都要可映射。

    旧实现按族枚举并用 ``orders[:96]`` 截断，``C(16,2)=120`` 装不下，实测
    ``swap(10,11)`` 与 ``swap(14,15)`` 直接无解——截断式上界本身就是覆盖缺口的来源。
    """
    terms, _ = _parse(USER_EQUATION)
    sizes = _user_label_sizes()
    for left, right in ((0, 1), (4, 5), (10, 11), (14, 15)):
        swapped = list(terms)
        swapped[left], swapped[right] = swapped[right], swapped[left]
        equation = ",".join(swapped) + "->MGHew"
        assert _specialize.match(equation) is None, f"{left}<->{right} 本不该直接命中"
        result = _equivalence.normalize_equation(equation)
        assert result is not None, f"swap({left},{right}) 无解"
        assert result.kind == "operand-order", result.kind
        moved = [index for index, source in enumerate(result.operand_order or ())
                 if index != source]
        assert moved == [left, right], (left, right, result.operand_order)
        assert not result.operand_permutations, result.operand_permutations
        _assert_structure(equation, result, f"swap({left},{right})")
        reference, via = _run_mapping(equation, result, sizes)
        _assert_same(reference, via, f"swap({left},{right})")


def test_renamed_and_permuted_equation_still_maps():
    """字母**全换** + 换位 + 轴序同时发生也要命中。

    这条是"按字母排序的规范形式"方案的试金石：那种写法在改名后 100%（实测
    8638/8638）与语料键不同，会整族漏判；本实现的结构键对字母不变。
    """
    order = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    terms, out = _parse(USER_EQUATION)
    table = {}
    for label in sorted({c for term in terms for c in term}):
        table[label] = order[(order.index(label) + 9) % len(order)]
    renamed = ["".join(table[c] for c in term) for term in terms]
    permuted = list(renamed)
    permuted[3], permuted[7] = permuted[7], permuted[3]
    permuted[0] = permuted[0][::-1]
    equation = ",".join(permuted) + "->" + "".join(table[c] for c in out)
    assert _specialize.match(equation) is None, "该写法本不应直接命中"
    result = _equivalence.normalize_equation(equation)
    assert result is not None, "改名 + 换位 + 轴序被漏判"
    assert result.kind != "identity", result.kind
    _assert_structure(equation, result, "改名+换位+轴序")
    assert _specialize.match(result.equation) is not None
    sizes = {table[label]: value for label, value in _user_label_sizes().items()}
    reference, via = _run_mapping(equation, result, sizes)
    _assert_same(reference, via, "改名+换位+轴序")


def test_default_path_never_enumerates_candidate_equations():
    """默认路径（``probe is None``）不得走候选枚举——那正是被重写掉的做法。"""
    saved = _equivalence._probe_candidates

    def forbidden(*args, **kwargs):
        raise AssertionError("默认路径调用了候选枚举 _probe_candidates")

    _equivalence._probe_candidates = forbidden
    try:
        for equation in ("acdb,menf,ac,em,bd,fn->", THREE_LAYER,
                         "cadb,menf,ac,em,fn,db->"):
            assert _equivalence.normalize_equation(equation) is not None, equation
            assert _equivalence.explain(equation)["rewritten"] is True, equation
            _equivalence.find_equivalents(equation, limit=4)
        _equivalence.normalize_equation("qz,wx->qz")       # 无解也不许回头枚举
    finally:
        _equivalence._probe_candidates = saved
    assert not hasattr(_equivalence, "_candidate_equations"), \
        "旧的候选枚举入口还在"


def test_index_build_and_lookup_stay_inside_their_budgets():
    """语料索引构建 < 2 s（一次性），16 操作数查询中位 < 5 ms。"""
    saved_index = _equivalence._INDEX
    saved_shapes = dict(_equivalence._SHAPES)
    try:
        _equivalence._INDEX = None
        _equivalence._SHAPES.clear()
        start = time.perf_counter()
        index = _equivalence._index()
        build = time.perf_counter() - start
        assert _equivalence._index() is index, "第二次调用必须直接命中缓存"
    finally:
        _equivalence._INDEX = saved_index
        _equivalence._SHAPES.clear()
        _equivalence._SHAPES.update(saved_shapes)
        if _equivalence._INDEX is None:
            _equivalence._index()
    assert build < 2.0, f"索引构建 {build:.2f} s 超出 2 s 上界"

    from pyqcd.einsum import _wick_tasklist as tasklist
    assert 1 < len(index) < len(tasklist.TASKS), (
        f"桶数 {len(index)} 不像是结构分桶（语料 {len(tasklist.TASKS)} 条）")

    _equivalence.normalize_equation(THREE_LAYER)          # 预热
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        result = _equivalence.normalize_equation(THREE_LAYER)
        timings.append(time.perf_counter() - start)
    timings.sort()
    assert result is not None, "三重复合无解"
    assert timings[len(timings) // 2] < 0.005, (
        f"16 操作数查询中位 {timings[len(timings) // 2] * 1e3:.2f} ms 超出 5 ms")


def test_layer_imports_stay_inside_the_declared_set():
    """新实现只许 import 标准库 + 语料清单 / 特化匹配器，不得碰执行层。"""
    with open(_equivalence.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    allowed_absolute = {"__future__", "os", "collections", "dataclasses",
                        "itertools", "typing"}
    # ``_cache`` 是**只依赖标准库**的缓存原语层（没有执行层依赖），2026-11-16 起
    # 允许本层直接 import；执行层（``_api`` / ``_kernel_*`` / ``_shape_paths``）
    # 仍然一律不放行。
    allowed_relative = {"_cache", "_specialize", "_wick_tasklist"}
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed_absolute:
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if (node.module or "").split(".")[0] not in allowed_absolute:
                    offenders.append(node.module or "")
            else:
                for alias in node.names:
                    if alias.name not in allowed_relative:
                        offenders.append("." * node.level + alias.name)
    assert not offenders, f"等价映射层 import 了执行层模块：{sorted(set(offenders))}"


# ---------------------- 8. 条目选择：零代价（不转置）优先于"字母巧合"
#
# 同一结构桶里常有多条**结构同构**的语料写法（语料本身就是同构重排的集合，最大桶
# 384 条），挑哪条只影响"调用方要动什么"。`_alignment` 的第三分量 `shared` 只数
# **逐位置的字面字母交集**，会把"被重排位置上的字母巧合"当成"更接近"：于是一个只
# 换了两个操作数位置的查询可能挑到一条"还得再转置两个操作数"的写法。而
# `transpose` 返回的是**视图**、通常非 C 连续，`_kernel_kupl.try_dedicated` 的第一
# 道门就拒非连续操作数——同桶明明有零代价写法，却把查询推出了特化路径。
#
# 这一节钉住修复后的口径，两个方向都要：
#   * 有零代价写法时必须挑它（`operand_permutations` 空 + `output_permutation` 空）；
#   * 没有零代价写法时必须**一字不改**地退回旧选择，且新扫描的候选上界显式有界
#     （最大桶 384 条，绝不允许"无界扫完整个桶"）。

#: 主 agent 实测样例：语料 seq=3 把前两个操作数交换。桶内 4 条同构写法中
#: seq=5 ``aobp,cmdn,…`` 的 align=(7, 7, 20)（字母巧合更高）但要转置第 4/6 个
#: 操作数，seq=3 ``ambn,codp,…`` 的 align=(7, 7, 16) 且零置换——应选后者。
ZERO_COST_QUERY = "codp,ambn,egfh,ce,mo,bdf,nph->ag"
#: 期望选中的语料写法（= seq=3，只换两个操作数位置，不转任何轴）。
ZERO_COST_EXPECTED = "ambn,codp,egfh,ce,mo,bdf,nph->ag"


def _ordered_bucket(equation):
    """按实现口径取出同桶条目：``_alignment`` 降序（同值按 seq 升序）。

    契约自己复算这个次序，是为了能同时问"新口径挑了哪条"与"旧口径会挑哪条"
    ——只断言最终结果的话，"偏好被撤掉、恰好又挑中同一条"会静默通过。
    """
    terms, out = _parse(equation)
    query = _equivalence._Shape(terms, out)
    entries = _equivalence._index().get(query.key, ())
    if len(entries) > 1:
        entries = sorted(entries, key=lambda item: (
            tuple(-value for value in _equivalence._alignment(terms, out, item[1])),
            item[0]))
    return terms, out, query, entries


def test_zero_cost_mapping_is_preferred_over_the_axis_transposing_neighbour():
    """**零置换优先**：同桶里不转任何轴的写法必须压过"字母更巧但要转置"的那条。

    样例来自主 agent 的实测：修复前选中 seq=5（``operand_permutations`` 里第 4 个
    操作数要 ``transpose(1, 0)``、第 6 个要 ``transpose(1, 0, 2)``），修复后必须选
    seq=3，``operand_permutations`` 为空、``output_permutation`` 为 ``None``，只保留
    免费的 ``operand_order``（换个实参次序，张量本身不动、仍是 C 连续）。

    断言五件事：①挑中的是哪条；②三个置换字段都零代价；③它真的命中语料；④**旧口径
    确实要转置**（否则这条用例在偏好被撤掉时会假绿）；⑤数值仍逐元素等价 rel<=1e-12。
    """
    terms, out, query, entries = _ordered_bucket(ZERO_COST_QUERY)
    assert len(entries) > 1, "契约前提：该样例必须落在多条同构写法的桶里"
    previous = _equivalence._first_mapping(entries, terms, out, query)
    assert previous is not None, "旧口径应当也能给出映射（否则无从比较偏好）"
    assert any(perm is not None for perm in previous.operand_permutations), (
        f"契约前提失效：旧口径挑的 {previous.equation} 本就是零代价的，"
        "这条用例区分不了偏好")

    result = _equivalence.normalize_equation(ZERO_COST_QUERY)
    assert result is not None, "接口层没找到等价写法"
    assert result.equation == ZERO_COST_EXPECTED, result.equation
    assert not any(perm is not None for perm in result.operand_permutations), \
        result.operand_permutations
    assert result.output_permutation is None, result.output_permutation
    assert tuple(result.operand_order) == (1, 0, 2, 3, 4, 5, 6), result.operand_order
    assert result.kind == "operand-order", result.kind
    assert _specialize.match(result.equation) is not None, "等价写法应当命中语料"
    _assert_structure(ZERO_COST_QUERY, result, "零置换优先")

    # 数值验收：标签长度 2/3/4 交错，轴序用反会改变形状或数值
    labels = sorted({ch for term in terms for ch in term} | set(out))
    sizes = {label: 2 + index % 3 for index, label in enumerate(labels)}
    reference, via = _run_mapping(ZERO_COST_QUERY, result, sizes)
    _assert_same(reference, via, "零置换优先", tol=1e-12)


def test_zero_cost_preference_degrades_to_the_previous_selection():
    """桶内**没有**零代价映射时，必须一字不改地退回旧选择（不是挑一条更差的）。

    ``THREE_LAYER`` 要求第 4 个操作数换轴序、输出段换轴序，桶里因此不存在零代价
    候选；此时返回的四个字段必须与"不设偏好"的 :func:`_equivalence._first_mapping`
    完全相同——退化路径不能引入任何新的取舍。
    """
    terms, out, query, entries = _ordered_bucket(THREE_LAYER)
    previous = _equivalence._first_mapping(entries, terms, out, query)
    result = _equivalence.normalize_equation(THREE_LAYER)
    assert previous is not None and result is not None, "两边都应当给出映射"
    assert (result.equation, result.operand_order, result.operand_permutations,
            result.output_permutation, result.kind) == (
        previous.equation, previous.operand_order, previous.operand_permutations,
        previous.output_permutation, previous.kind), result
    # 这条退化路径必须**真的**有代价，否则"退化"没有意义
    assert (any(perm is not None for perm in result.operand_permutations)
            or result.output_permutation is not None), result
    _assert_structure(THREE_LAYER, result, "退化路径")
    reference, via = _run_mapping(THREE_LAYER, result, _user_label_sizes())
    _assert_same(reference, via, "退化路径", tol=1e-12)


def test_zero_cost_scan_is_bounded_below_the_largest_bucket():
    """新扫描的候选上界必须显式命名且**远小于**最大桶——绝不"无界扫完整个桶"。

    这是唯一能执行的界保障：上界是模块级常量，最大桶 384 条由语料索引现算。
    """
    index = _equivalence._index()
    largest = max(len(bucket) for bucket in index.values())
    assert largest > 16, f"最大桶只有 {largest} 条，该上界的取值依据需重估"
    assert 1 < _equivalence._MAX_ZERO_COST_SCAN < largest, (
        _equivalence._MAX_ZERO_COST_SCAN, largest)


TESTS = (
    test_axis_permutation_mapping_is_numerically_equivalent,
    test_operand_order_mapping_is_numerically_equivalent,
    test_output_axis_permutation_mapping_is_numerically_equivalent,
    test_output_axis_family_covers_pairwise_rotation_and_cycle,
    test_identity_output_needs_no_permutation,
    test_mapping_really_reaches_the_specialized_path,
    test_invalid_or_unmappable_inputs_do_not_raise,
    test_already_matching_equation_is_not_rewritten,
    test_unmappable_equation_returns_none,
    test_high_rank_enumeration_stays_bounded,
    test_verify_specialized_reports_the_equivalence,
    test_verify_specialized_json_carries_the_equivalence,
    test_composite_mapping_is_numerically_equivalent,
    test_composite_mapping_covers_the_user_reported_pair_swap_plus_reversal,
    test_operand_axes_plus_output_axes_composite_is_equivalent,
    test_composite_enumeration_is_bounded,
    test_composite_enumeration_prefers_the_smallest_change,
    test_composite_report_shows_both_ordering_and_permutation_lines,
    test_equivalence_is_enabled_by_default,
    test_disabling_the_switch_turns_every_entry_point_into_a_noop,
    test_disabled_verify_specialized_output_matches_the_pre_change_original,
    # 第 6 节：方向判别、空白鲁棒性、用户用例、性能
    test_direction_output_axes_is_discriminated_both_ways,
    test_self_inverse_output_permutations_are_marked_as_non_direction_tests,
    test_direction_operand_axes_three_cycles_are_discriminated,
    test_direction_operand_order_three_cycle_is_discriminated,
    test_whitespace_never_becomes_an_axis_position,
    test_whitespace_variants_produce_the_same_mapping_as_the_canonical_form,
    test_user_case_1_baseline_hits_the_corpus_entry,
    test_user_case_2_output_axis_order_is_solved_and_numerically_exact,
    test_user_case_3_axis_order_plus_output_is_solved_and_numerically_exact,
    test_user_case_4_order_plus_axes_plus_output_is_solved_or_explained,
    test_sixteen_operand_normalize_is_not_factorial,
    # 第 7 节：结构键不变性、零枚举、旧覆盖缺口、依赖面、性能预算
    test_structural_key_is_invariant_under_every_equivalence_transformation,
    test_bucket_key_separates_contractions_that_are_not_equivalent,
    test_truly_non_equivalent_equations_are_reported_as_no_mapping,
    test_three_layer_composite_is_solved_with_the_minimal_fields,
    test_old_coverage_gap_pair_swaps_on_sixteen_operands_are_closed,
    test_renamed_and_permuted_equation_still_maps,
    test_default_path_never_enumerates_candidate_equations,
    test_index_build_and_lookup_stay_inside_their_budgets,
    test_layer_imports_stay_inside_the_declared_set,
    # 第 8 节：条目选择（零代价优先 / 退化到旧行为 / 上界）
    test_zero_cost_mapping_is_preferred_over_the_axis_transposing_neighbour,
    test_zero_cost_preference_degrades_to_the_previous_selection,
    test_zero_cost_scan_is_bounded_below_the_largest_bucket,
)


def main() -> int:
    import unittest

    passed = skipped = failed = 0
    for test in TESTS:
        try:
            test()
        except unittest.SkipTest as exc:
            skipped += 1
            print(f"SKIP {test.__name__}: {exc}", flush=True)
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}", flush=True)
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    print(f"{passed} passed, {skipped} skipped, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
