"""统一拓扑步表生成器：一次生成两个后端的表。

用法：

    python -m pyqcd.einsum.topology_steps_gen                 # 两个后端都生成（缺省）
    python -m pyqcd.einsum.topology_steps_gen --target kupl   # 只写 .inc + .inc.gz
    python -m pyqcd.einsum.topology_steps_gen --target cuda   # 只写 .h + .h.gz
    python -m pyqcd.einsum.topology_steps_gen --verify 2582   # 只核对 CUDA 表的历史前缀

语料枚举、路径求解与角色形状约定在 pyqcd.einsum._topology_gen_common；本模块只剩
两个后端的输出格式层（写 .inc / 写 .h）。

shape 的**默认且唯一**形状源（用户硬性约定，落在 _topology_gen_common.ROLE_SHAPES）：

    peram → [4,4,100,100]
    gamma → [4,4]            （C * gamma_* 前缀同样按 gamma）
    VDV   → [100,100]
    VVV   → [100,100,100,100]

    输出形状由上述信息自动推出。

两个后端都要显式传 --shape-source 才会离开它：默认值就只有 roles。

    kupl → 产物 cpp/kupl/einsum/src/topology_kernels.inc
    cuda → 产物 cpp/cuda/einsum/src/topology_steps.h

优化器在**生成侧统一为 auto**（``_topology_gen_common.KUPL_OPTIMIZE`` /
``CUDA_OPTIMIZE``，用户裁定 2026-11-18）：两张表的收缩序都在 ``ROLE_SHAPES`` 上由
``contract_path(optimize="auto")`` 定型。

**生成侧 auto / 用户侧 7 叶子：两条互不重叠的口径。** 用户侧 ``optimize`` 的可写
取值仍是 **7 个叶子**（``greedy``/``optimal``/``dp``/``branch_1``/``branch_2``/
``branch_all``/``none``，见 ``_profile_cache.OPTIMIZE_LEAVES``），**不含 ``auto``**
——``auto`` 只作为 ``True``/``None`` 的内部归一结果存在
（``paths._IMPLICIT_DEFAULT_NAMES``），``paths.get_path_fn("auto")`` 抛 ``KeyError``，
配置表里写 ``auto`` 属非法值。用户侧该写哪个叶子由 ``--tune`` 的实测结论决定。
本模块是**生成侧**工具，不提供也不消费用户侧的 optimize 配置。

两条历史形状口径已**删除**（不是标 DEPRECATED）：``production``（生产异形轴，取自
``_label_sizes`` 的 ``distill`` 档）与 ``uniform``（等长探针 dim=4）。它们与
``ROLE_SHAPES`` 不是同一套长度，同一方程会得到**不同收缩序**，所以按旧口径生成的表
与本表**不可比、也已全部作废**——不要试图"混用"或"迁移"旧表，唯一正确的动作是重生成。

**重生成后的验收顺序**（缺一条就不要交付）：

1. 新产物的 ``sha256`` 与旧表**必须不同**——相同说明默认路径没变（曾出现过
   "改了默认值但 ``build_kupl`` 仍按 ``production``、CUDA 仍按 ``uniform``"，
   于是"重生成"其实是原样重跑）；
2. 连续生成两次，两次 ``sha256`` 必须相同（不确定性先查 ``set``/``dict`` 迭代序）；
3. ``--target cuda --verify`` → ``missing=0 extra=0 mismatch=0`` 且
   ``orders N/N match=True covers_file=True``；
4. 两表条目数都是 ``MAX_KERNELS``（8638），表头自述与 ``--shape-source`` 一致。

每张表都另外产出一份 .gz（--no-gzip 可关）。表本体是生成物，仓库里发布的是压缩件，
构建侧负责解压（KUPL 与 CUDA 两侧的脚本都必须有这段逻辑，缺一侧会让"只发布 .gz"
的干净检出直接构建失败；条件一律是"本体缺失 **或** .gz 更新"，否则会覆盖本地刚生成
的表）。

表头 banner 逐字保留历史入口名（pyqcd/testing/_topology_kernel_gen.py /
_cuda_topology_steps_gen.py）：旧入口现在是转发本模块的薄包装。**别顺手改这两行
banner**——它会让产物与已发布表差一行注释，而"改动可解释"是这类生成物的底线。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import pathlib
import re
import sys
import time
from typing import Any, Sequence

from ._specialize import canonical_signature as structural_signature
from ._topology_gen_common import (CUDA_OPTIMIZE, DEFAULT_SHAPE_SOURCE,
                                   KUPL_OPTIMIZE, MAX_KERNELS, equations, parse,
                                   role_sizes, solve_pairs,
                                   solve_path)

ROOT = pathlib.Path(__file__).resolve().parents[2]
KUPL_TARGET = "cpp/kupl/einsum/src/topology_kernels.inc"
CUDA_TARGET = ROOT / "cpp/cuda/einsum/src/topology_steps.h"

#: 基础表条目数（seq 1..2582）。仅用于 CUDA 侧 ``--verify`` 的历史前缀核对。
BASE_ENTRIES = 2582

MAX_RANK = 15

_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_ENTRY_RE = re.compile(
    r"\{0x([0-9a-f]{16})ULL, 0x([0-9a-f]{16})ULL, (\d+), (\d+), 0\}")
_ORDER_RE = re.compile(
    r"static const int8_t kQcdTopoOrder(\d+)\[\] = \{([^}]*)\};")
_ORDERS_RE = re.compile(r"\{kQcdTopoOrder(\d+), (\d+)\},")


# ════════════════════════════════════════════════════════════════════════════
# 共享：形状源选择
# ════════════════════════════════════════════════════════════════════════════
def _sizes_for(equation: str, roles: Sequence[str],
               shape_source: str) -> dict[str, int] | None:
    """形状源 → ``solve_path`` 的 ``sizes``。

    ``roles``（:data:`ROLE_SHAPES`）是**唯一**形状源。早先的 ``production``
    （生产异形轴）与 ``uniform``（等长探针）两条口径已随用户裁定**彻底删除**。
    """
    if shape_source == DEFAULT_SHAPE_SOURCE:
        return role_sizes(parse(equation)[0], roles)
    raise ValueError(f"unknown shape source {shape_source!r}")


def _task_roles(seq: int, equation: str, tasks: dict[int, Any]) -> tuple[str, ...]:
    task = tasks.get(seq)
    if task is None or task.equation != equation:
        raise AssertionError(
            f"seq={seq}: tasklist does not agree with the parsed corpus")
    return task.roles


def _step_sizes(equation: str, seq: int, tasks: dict[int, Any],
                shape_source: str) -> dict[str, int]:
    """本条的 ``sizes``（必须非空：探针口径已删除）。"""
    return _sizes_for(equation, _task_roles(seq, equation, tasks), shape_source)


# ════════════════════════════════════════════════════════════════════════════
# KUPL：v3 字节码步表 → topology_kernels.inc
# ════════════════════════════════════════════════════════════════════════════
def canonical_signature(terms: list[str], out: str) -> str:
    """与 ``_native_plan`` 的 symbol_order 同口径的规范签名（base-36 文本）。

    指标按"全式首次出现顺序"重新编号（先输入项按操作数顺序，后输出），于是字母
    改名等价式得到同一签名；操作数顺序不同则签名不同（保守不命中，回落通用路径）。
    """
    order: list[str] = []
    for term in list(terms) + [out]:
        for ch in term:
            if ch not in order:
                order.append(ch)
    index = {ch: pos for pos, ch in enumerate(order)}
    if len(order) > len(_B36):
        raise AssertionError(f"too many labels ({len(order)}) for a base-36 signature")

    def encode(term: str) -> str:
        return "".join(_B36[index[ch]] for ch in term)

    return ",".join(encode(term) for term in terms) + "->" + encode(out)


def fnv1a(text: str) -> int:
    value = _FNV_OFFSET
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return value


def plan_steps(equation: str, sizes: dict[str, int] | None = None
               ) -> tuple[list[dict], list[str], str]:
    """KUPL 口径的收缩步序列（``optimize="auto"``，与 ``try_dedicated`` 同口径）。

    ``sizes`` 来自 ``ROLE_SHAPES`` 且**必填**（探针与异形轴口径均已删除）。
    """
    return solve_path(equation, sizes, optimize=KUPL_OPTIMIZE)


def decompose(left: str, right: str, out: str,
              sizes: dict[str, int] | None = None) -> dict:
    """把二元 einsum 步拆成 GEMM 的 batch/M/K/N 分组与轴序。

    ``sizes`` 给出每个 label 的真实长度时，按**元素数**决定组内轴序（见下），与
    ``_kernel_kupl._dedicated_step`` 同口径；缺省则用保守的固定 ``reverse`` 顺序。
    """
    sl, sr, so = set(left), set(right), set(out)
    batch = [ch for ch in left if ch in sr and ch in so]
    inner = [ch for ch in left if ch in sr and ch not in so]
    rows = [ch for ch in left if ch not in sr]
    cols = [ch for ch in right if ch not in sl]
    where = f"{left},{right}->{out}"
    if not set(rows) <= so or not set(cols) <= so:
        raise AssertionError(f"a one-sided free label is not in the output: {where}")
    if set(batch) | set(inner) | set(rows) != sl or set(batch) | set(inner) | set(cols) != sr:
        raise AssertionError(f"label accounting failed for {where}")
    if set(batch) | set(rows) | set(cols) != so:
        raise AssertionError(f"output labels not explained by the inputs: {where}")
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise AssertionError(f"repeated label inside one operand: {where}")

    # 组内次序自由，但对**打包代价**有决定性影响：某一侧的最终轴序若恰好等于它
    # 的源轴序，那次 pack 就退化成恒等排列，执行器直接跳过拷贝。
    #
    # batch 与 inner 是两侧共用的组，只能选一个顺序；固定 ``reverse`` 只照顾右
    # 张量。实测 seq=3 第 2 步 ``nph,egfh->npegf``：左侧 100 元素、右侧 4,000,000
    # 元素，固定序要求 unpack 整个 4M 中间体；改按元素数取大张量原序后 ``c_axes``
    # 恒等（零拷贝），该步 42.6 ms → 1.6 ms。传 ``sizes`` 时走数据驱动。
    if sizes is not None:
        # 与 ``_kernel_kupl._dedicated_step`` 逐行同口径：共用组取**元素数更大
        # 那一侧**的原序（大张量零拷贝），rows/cols 各自按源序**升序**。升序是
        # 关键——只有升序才让 ``rows`` 与 ``left``、``cols`` 与 ``right`` 的
        # 相对次序一致，从而 ``c_axes`` 有机会退化成恒等（``[0,1,2,...]``）。
        def element_count(labels: list[str]) -> int:
            total = 1
            for label in labels:
                total *= int(sizes[label])
            return total

        shared = set(batch) | set(inner)
        left_shared = [ch for ch in left if ch in shared]
        right_shared = [ch for ch in right if ch in shared]
        ordered = right_shared if element_count(right) > element_count(left) else left_shared
        batch = [ch for ch in ordered if ch in batch]
        inner = [ch for ch in ordered if ch in inner]
        rows.sort(key=left.index)
        cols.sort(key=right.index)
    else:
        batch.sort(key=left.index, reverse=True)
        rows.sort(key=left.index, reverse=True)
        inner.sort(key=left.index, reverse=True)
        cols.sort(key=right.index, reverse=True)

    a_order = batch + rows + inner
    b_order = batch + inner + cols
    c_order = batch + rows + cols
    spec = {
        "rank_a": len(a_order),
        "rank_b": len(b_order),
        "rank_c": len(out),
        "a_axes": [left.index(ch) for ch in a_order],
        "b_axes": [right.index(ch) for ch in b_order],
        "c_axes": [c_order.index(ch) for ch in out],
        "batch": len(batch),
        "inner": len(inner),
        "rows": len(rows),
        "cols": len(cols),
    }
    for key in ("rank_a", "rank_b", "rank_c", "batch", "inner"):
        if not 0 <= spec[key] <= MAX_RANK:
            raise AssertionError(f"{where}: {key}={spec[key]} out of range")
    for key in ("a_axes", "b_axes", "c_axes"):
        if any(axis >= MAX_RANK for axis in spec[key]):
            raise AssertionError(f"{where}: axis index out of range in {key}")
    return spec


def encode_steps(equation: str, sizes: dict[str, int] | None = None) -> bytes:
    """编码一条方程的 KUPL 步表（v3）。

    v3 = **label 块 + 逐步描述 + 每步 c_labels**。v2 只存轴序（4 bit/轴），轴长靠
    ``dim`` 幂次反推，因此只能服务"所有轴长相等"的超立方体；生产 scene 的轴长是
    ``{4,5,40,50}`` 混杂，v2 直接拒绝。v3 把 label 带上，运行期用 ``sizes[label]``
    取真实长度——中间体长度**无法**由轴序拼接推出，这是 v3 必须存在的理由。

    段首::

        uint8 arity
        逐操作数: uint8 rank + rank×uint8 label_id（canonical 编号）

    每步 6 字节头（``a_src``/``b_src``/``dst``/ranks/batch+inner）+ 轴序 nibble 流，
    紧随其后追加 ``uint8 c_labels[rank_c]``。中间体经 unpack 后按 ``step["out"]``
    的轴序存放，所以后续步骤按位置引用它时拿到的正是这张表。
    """
    steps, terms, out = plan_steps(equation, sizes)
    if not steps or steps[-1]["out"] != out:
        raise AssertionError(f"{equation}: last step does not produce the output")

    order: list[str] = []
    for term in list(terms) + [out]:
        for ch in term:
            if ch not in order:
                order.append(ch)
    if len(order) > 255:
        raise AssertionError(f"{equation}: {len(order)} labels exceed uint8")
    labels = {ch: index for index, ch in enumerate(order)}

    stream = bytearray()
    stream.append(len(terms))
    for term in terms:
        stream.append(len(term))
        stream.extend(labels[ch] for ch in term)

    for step in steps:
        spec = decompose(step["left"], step["right"], step["out"], sizes)
        for key in ("a_src", "b_src", "dst"):
            if not -128 <= step[key] <= 127:
                raise AssertionError(f"{equation}: slot {key}={step[key]} does not fit int8")
        stream.append(step["a_src"] & 0xFF)
        stream.append(step["b_src"] & 0xFF)
        stream.append(step["dst"] & 0xFF)
        stream.append((spec["rank_a"] << 4) | spec["rank_b"])
        stream.append(spec["rank_c"])
        stream.append((spec["batch"] << 4) | spec["inner"])
        axes = spec["a_axes"] + spec["b_axes"] + spec["c_axes"]
        for index in range(0, len(axes), 2):
            high = axes[index] & 0x0F
            low = axes[index + 1] & 0x0F if index + 1 < len(axes) else 0
            stream.append((high << 4) | low)
        stream.extend(labels[ch] for ch in step["out"])
    return bytes(stream)


def _step_count(blob: bytes) -> int:
    """步数：按 v3 格式走一遍（跳过 label 块、nibble 流与 c_labels），用于记录校验。"""
    if not blob:
        raise AssertionError("empty step table")
    arity = blob[0]
    if arity <= 0:
        raise AssertionError(f"invalid arity {arity}")
    position = 1
    for _ in range(arity):
        if position >= len(blob):
            raise AssertionError("truncated label block")
        rank = blob[position]
        position += 1 + rank
        if position > len(blob):
            raise AssertionError("truncated label block")
    count = 0
    while position < len(blob):
        if position + 6 > len(blob):
            raise AssertionError("truncated step header")
        rank_a = blob[position + 3] >> 4
        rank_b = blob[position + 3] & 0x0F
        rank_c = blob[position + 4]
        position += 6 + (rank_a + rank_b + rank_c + 1) // 2 + rank_c
        if position > len(blob):
            raise AssertionError("truncated step body")
        count += 1
    if position != len(blob):
        raise AssertionError("step table does not divide evenly")
    return count


def _octal_literal(data: bytes) -> list[str]:
    """把字节流发成 C++ 字符串字面量：八进制转义 3 位固定宽度，无歧义。"""
    lines: list[str] = []
    width = 64
    for start in range(0, len(data), width):
        chunk = data[start:start + width]
        lines.append('    "' + "".join("\\%03o" % byte for byte in chunk) + '"')
    if not lines:
        lines.append('    ""')
    return lines


def collect_kupl(limit: int | None = None,
                 *, shape_source: str = DEFAULT_SHAPE_SOURCE
                 ) -> tuple[list[dict], int]:
    """按语料顺序生成 KUPL 表项；返回 (条目, 步表总字节数)。

    形状源固定 ``roles``（:data:`ROLE_SHAPES`），没有第二分支可选。

    ``records`` 排序与 offset 由 :func:`render_kupl` 的调用方负责（渲染前按 hash
    升序、逐条累加 offset）。
    """
    from . import _wick_tasklist as tasklist

    corpus = equations()
    if limit is not None:
        corpus = corpus[:limit]
    if len(corpus) > MAX_KERNELS:
        raise AssertionError(f"{len(corpus)} equations exceed MAX_KERNELS {MAX_KERNELS}")

    tasks: dict[int, Any] = tasklist.by_seq()
    if len(tasks) < len(corpus):
        raise AssertionError(
            f"corpus roles unavailable: {len(tasks)} tasks for "
            f"{len(corpus)} equations")

    entries: list[dict] = []
    seen_hash: dict[int, str] = {}
    for seq, equation in enumerate(corpus, start=1):
        terms, out = parse(equation)
        sizes = _step_sizes(equation, seq, tasks, shape_source)
        blob = encode_steps(equation, sizes)
        signature = canonical_signature(terms, out)
        digest = fnv1a(signature)
        if digest in seen_hash:
            raise AssertionError(f"hash collision: {signature!r} vs {seen_hash[digest]!r}")
        seen_hash[digest] = signature
        entries.append({
            "equation": equation,
            "signature": signature,
            "hash": digest,
            "blob": blob,
            "arity": len(terms),
            "out_rank": len(out),
            "steps": _step_count(blob),
        })
    return entries, sum(len(entry["blob"]) for entry in entries)


def render_kupl(records: list[dict], kernels: list[dict], total_code: int,
                shape_source: str = DEFAULT_SHAPE_SOURCE) -> str:
    """``records`` 按 hash 升序（二分查找用），``kernels`` 与语料同序（索引即语料序号）。"""
    out: list[str] = []
    # banner 逐字保留历史入口名（旧入口现已是本模块的薄包装）：改名会让产物与已发布
    # 表差一行注释。"旧入口 deprecated"写在模块 docstring 里，不要在这里顺手改。
    out.append("/* 由 pyqcd/testing/_topology_kernel_gen.py 生成，请勿手工编辑。")
    out.append(f" * 覆盖 refer/sush/wick_12cases.md 的 {len(records)} 条收缩路径；每条由")
    out.append(f" * pyqcd.einsum.contract_path(optimize='{KUPL_OPTIMIZE}') 求出的"
               "「打包 → cblas GEMM")
    out.append(" * → 解包」步序列描述，执行器见 src/topology_gemm.h。")
    out.append(f" * 形状源 ROLE_SHAPES（--shape-source {shape_source}）；"
               "roles = ROLE_SHAPES 角色形状。")
    out.append(f" * 步表共 {total_code} 字节：每条拓扑的 (code_offset, code_bytes) 定位一段。 */")
    out.append("")
    out.append('#include "topology_gemm.h"')
    out.append("")
    out.append("struct TopologyOperands { const float* const* data; };")
    out.append("")
    out.append("namespace pyqcd_einsum {")
    out.append("namespace topology_table {")
    out.append("")
    out.append(f"static const int kTopoRecordCount = {len(records)};")
    out.append("")
    out.append("/* 步表字节流：每条拓扑的 (code_offset, code_bytes) 定位到一段。 */")
    out.append("static const char kTopoCode[] =")
    for entry in records:
        out.extend(_octal_literal(entry["blob"]))
    out.append("    ;")
    out.append("")
    out.append("/* 记录按 hash 升序，供二分查找；equations 顺序由 kTopoRecordOfEquation 恢复。 */")
    out.append("static const pyqcd_einsum::topology_gemm::Record kTopoRecords[] = {")
    for entry in records:
        out.append(
            '    {{0x{hash:016x}ULL, "{sig}", {offset}, {size}, {arity}, {out_rank}, '
            "{steps}, 3}},".format(hash=entry["hash"], sig=entry["signature"],
                                offset=entry["offset"], size=len(entry["blob"]),
                                arity=entry["arity"], out_rank=entry["out_rank"],
                                steps=entry["steps"]))
    out.append("};")
    out.append("")
    out.append("/* 与语料同序：第 i 条方程对应 kTopoRecords[kTopoRecordOfEquation[i]]。 */")
    out.append("static const uint16_t kTopoRecordOfEquation[] = {")
    for start in range(0, len(kernels), 16):
        chunk = kernels[start:start + 16]
        out.append("    " + " ".join(str(entry["record_index"]) + "," for entry in chunk))
    out.append("};")
    out.append("")
    out.append("[[maybe_unused]] static const pyqcd_einsum::topology_gemm::Record* topo_find(")
    out.append("    uint64_t hash) {")
    out.append("    int low = 0;")
    out.append("    int high = kTopoRecordCount - 1;")
    out.append("    while (low <= high) {")
    out.append("        const int middle = low + (high - low) / 2;")
    out.append("        const uint64_t key = kTopoRecords[middle].hash;")
    out.append("        if (key == hash) {")
    out.append("            return &kTopoRecords[middle];")
    out.append("        }")
    out.append("        if (key < hash) {")
    out.append("            low = middle + 1;")
    out.append("        } else {")
    out.append("            high = middle - 1;")
    out.append("        }")
    out.append("    }")
    out.append("    return nullptr;")
    out.append("}")
    out.append("")
    out.append("}  // namespace topology_table")
    out.append("}  // namespace pyqcd_einsum")
    out.append("")
    out.append("[[maybe_unused]] static int topo_run_f(int index, const float* const* data,")
    out.append("                                         int dim, void* out) {")
    out.append("    const pyqcd_einsum::topology_gemm::Record& record =")
    out.append("        pyqcd_einsum::topology_table::kTopoRecords[index];")
    out.append("    const uint8_t* code = reinterpret_cast<const uint8_t*>(")
    out.append("        pyqcd_einsum::topology_table::kTopoCode) + record.code_offset;")
    out.append("    return pyqcd_einsum::topology_gemm::run(")
    out.append("        record, code, reinterpret_cast<const void* const*>(data), out,")
    out.append("        pyqcd_einsum::topology_gemm::kFloat, dim);")
    out.append("}")
    out.append("")
    out.append("[[maybe_unused]] static int topo_run_d(int index, const void* const* data,")
    out.append("                                         int dim, void* out) {")
    out.append("    const pyqcd_einsum::topology_gemm::Record& record =")
    out.append("        pyqcd_einsum::topology_table::kTopoRecords[index];")
    out.append("    const uint8_t* code = reinterpret_cast<const uint8_t*>(")
    out.append("        pyqcd_einsum::topology_table::kTopoCode) + record.code_offset;")
    out.append("    return pyqcd_einsum::topology_gemm::run(")
    out.append("        record, code, data, out, pyqcd_einsum::topology_gemm::kDouble, dim);")
    out.append("}")
    out.append("")
    out.append("/* 每条拓扑一个具名入口：编译期常量表索引，函数体是一次参数化 GEMM 调用。 */")
    for index, entry in enumerate(kernels):
        out.append(
            "static int topo_{index:04d}(const TopologyOperands& in, int dim, void* out) {{ "
            "return topo_run_f({record}, in.data, dim, out); }}".format(
                index=index, record=entry["record_index"]))
    out.append("")
    out.append('struct TopologyKernel { const char* equation; '
               'int (*run)(const TopologyOperands&, int, void*); };')
    out.append("static const TopologyKernel kTopologyKernels[] = {")
    for index, entry in enumerate(kernels):
        out.append('    {{"{eq}", topo_{index:04d}}},'.format(eq=entry["equation"], index=index))
    out.append("};")
    out.append(f"static const int kTopologyKernelCount = {len(kernels)};")
    out.append("")
    return "\n".join(out)


def generate_kupl(target: pathlib.Path, *,
                  shape_source: str = DEFAULT_SHAPE_SOURCE,
                  limit: int | None = None,
                  compress: bool = True) -> dict[str, Any]:
    """生成 KUPL 表（``.inc`` + ``.inc.gz``）。"""
    entries, total_code = collect_kupl(limit, shape_source=shape_source)

    ordered = sorted(range(len(entries)), key=lambda index: entries[index]["hash"])
    sorted_entries = [entries[index] for index in ordered]
    offset = 0
    for record_index, entry in enumerate(sorted_entries):
        entry["offset"] = offset
        entry["record_index"] = record_index
        offset += len(entry["blob"])

    text = render_kupl(sorted_entries, entries, offset, shape_source)
    data = (text + "\n").encode("utf-8")
    target.write_bytes(data)
    report: dict[str, Any] = {"target": target, "bytes": len(data),
                              "entries": len(entries), "steps": total_code,
                              "gz": None}
    if compress:
        # ``mtime=0``（且不带文件名）是**确定性**的一部分：默认的 gzip 头会写时间戳，
        # 于是每次重生成都让这个被跟踪的 .gz 变成脏文件，而载荷其实一字未变。
        payload = gzip.compress(data, compresslevel=9, mtime=0)
        gz_path = pathlib.Path(str(target) + ".gz")
        gz_path.write_bytes(payload)
        report["gz"] = (gz_path, len(payload))
    return report


# ════════════════════════════════════════════════════════════════════════════
# CUDA：活表位置对 + (key, fnv) → topology_steps.h
# ════════════════════════════════════════════════════════════════════════════
def table_key(signature: tuple) -> int:
    """签名 → 表键：规范 payload 的 blake2b-128 摘要前 8 字节（大端）。"""
    payload = "|".join(",".join(map(str, term)) for term in signature)
    digest = hashlib.blake2b(payload.encode(), digest_size=16).digest()
    return int.from_bytes(digest[:8], "big")


def fnv_fingerprint(signature: tuple) -> int:
    """规范标签流的 FNV-1a：逐项 rank 字节 + 标签字节，``0xFF``，输出标签。"""
    state = _FNV_OFFSET
    for term in signature[:-1]:
        state = ((state ^ (len(term) & 0xFF)) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
        for label in term:
            state = ((state ^ (label & 0xFF)) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    state = ((state ^ 0xFF) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    for label in signature[-1]:
        state = ((state ^ (label & 0xFF)) * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return state


def step_pairs(equation: str, sizes: dict[str, int] | None = None
               ) -> tuple[tuple[int, int], ...]:
    """CUDA 口径的收缩序（:data:`CUDA_OPTIMIZE` = ``auto``；``sizes`` 必填，
    来自 ``ROLE_SHAPES``，没有探针兜底）。"""
    return solve_pairs(equation, sizes, optimize=CUDA_OPTIMIZE)


def collect_cuda(limit: int | None = None,
                 *, shape_source: str = DEFAULT_SHAPE_SOURCE
                 ) -> list[dict[str, Any]]:
    """按语料顺序生成 CUDA 表项；``limit`` 用于复现历史前缀（2582 条）。

    形状源固定 ``roles``（:data:`ROLE_SHAPES`），没有第二分支可选。
    """
    from . import _wick_tasklist as tasklist

    corpus = equations()
    if limit is None:
        if len(corpus) != MAX_KERNELS:
            raise AssertionError(
                f"corpus size {len(corpus)} != MAX_KERNELS {MAX_KERNELS}")
    tasks: dict[int, Any] = tasklist.by_seq()

    records: list[dict[str, Any]] = []
    seen: dict[int, int] = {}
    for seq, equation in enumerate(corpus[:limit], start=1):
        terms, output = parse(equation)
        signature = structural_signature(equation)
        if signature is None:
            raise AssertionError(f"seq={seq}: no canonical signature for {equation!r}")
        if len(signature[-1]) != len(output):
            raise AssertionError(f"seq={seq}: output signature disagrees")
        key = table_key(signature)
        previous = seen.get(key)
        if previous is not None:
            raise AssertionError(
                f"seq={seq}: key {key:#018x} collides with seq={previous}")
        seen[key] = seq
        arity = len(terms)
        if not 0 < arity <= 255:
            raise AssertionError(f"seq={seq}: arity {arity} does not fit uint8")
        sizes = _step_sizes(equation, seq, tasks, shape_source)
        records.append({
            "seq": seq,
            "equation": equation,
            "key": key,
            "fnv": fnv_fingerprint(signature),
            "arity": arity,
            "pairs": step_pairs(equation, sizes),
        })
    return records


def layout(records: Sequence[dict[str, Any]]) -> tuple[list[tuple], list[dict], int]:
    """去重步序（字典序，与历史文件同序）+ 按 key 升序的条目。

    ``sorted({...})`` 同时承担**确定性**：去重靠 ``set``，排序把迭代序固定下来，
    所以同一语料两次生成的步序表逐字节相同。若有一天两次生成的"去重步序数"不同，
    先查这里与并发，而不是怀疑表本身。
    """
    orders = sorted({record["pairs"] for record in records})
    if len(orders) > 65535:
        raise AssertionError(f"{len(orders)} distinct orders exceed uint16")
    index = {pairs: position for position, pairs in enumerate(orders)}
    entries = sorted(records, key=lambda record: record["key"])
    for entry in entries:
        entry["order"] = index[entry["pairs"]]
    return orders, list(entries), len(orders)


def render_cuda(records: Sequence[dict[str, Any]],
                shape_source: str = DEFAULT_SHAPE_SOURCE) -> str:
    orders, entries, order_count = layout(records)
    out: list[str] = []
    # banner 逐字保留历史入口名（旧入口现已是本模块的薄包装）：改名会让产物与已发布
    # 表差一行注释。"旧入口 deprecated"写在模块 docstring 里，不要在这里顺手改。
    out.append("/* 由 pyqcd/testing/_cuda_topology_steps_gen.py 生成，请勿手工编辑。")
    out.append(" *")
    out.append(" * 每条真实语料拓扑一条紧凑步表：收缩顺序（活表位置对）为编译期常量，")
    out.append(" * 由 apply_contract.cu 的参数化执行器展开成 pack + cuBLAS 步序列。")
    out.append(" * key = 拓扑签名 blake2b-128 摘要的前 8 字节（大端），按 key 升序排列；")
    out.append(" * fnv = 同一规范标签流的 64 位 FNV-1a 指纹，运行期由原生侧重算，")
    out.append(' * 用来把"表项属于这条结构"绑到实际下发的标签上（失配即 fail closed）。')
    out.append(" *")
    out.append(f" * 语料 {len(records)} 条，去重后 {order_count} 条不同步序；顺序取自")
    out.append(f" * 优化器 {CUDA_OPTIMIZE}（形状源 ROLE_SHAPES，--shape-source {shape_source}）；"
               "roles = ROLE_SHAPES 角色形状。 */")
    out.append("")
    out.append("#ifndef PYQCD_EINSUM_CUDA_TOPOLOGY_STEPS_H")
    out.append("#define PYQCD_EINSUM_CUDA_TOPOLOGY_STEPS_H")
    out.append("")
    out.append("#include <cstdint>")
    out.append("")
    out.append("struct qcd_topology_order {")
    out.append("    const int8_t* pairs;  /* 2*steps 个位置对 (left, right) */")
    out.append("    int32_t steps;")
    out.append("};")
    out.append("")
    for position, pairs in enumerate(orders):
        flat = ", ".join(str(value) for pair in pairs for value in pair)
        out.append(f"static const int8_t kQcdTopoOrder{position}[] = {{{flat}}};")
    out.append("")
    out.append("static const qcd_topology_order kQcdTopoOrders[] = {")
    for position, pairs in enumerate(orders):
        out.append(f"    {{kQcdTopoOrder{position}, {len(pairs)}}},")
    out.append("};")
    out.append("")
    out.append("struct qcd_topology_entry {")
    out.append("    uint64_t key;      /* 签名摘要前 8 字节（大端）；表按此升序 */")
    out.append("    uint64_t fnv;      /* 规范标签流的 FNV-1a 指纹 */")
    out.append("    uint16_t order;    /* kQcdTopoOrders 下标 */")
    out.append("    uint8_t arity;     /* 操作数个数 */")
    out.append("    uint8_t reserved;")
    out.append("};")
    out.append("")
    out.append("static const qcd_topology_entry kQcdTopoEntries[] = {")
    for entry in entries:
        out.append("    {{0x{key:016x}ULL, 0x{fnv:016x}ULL, {order}, {arity}, 0}},"
                   .format(**entry))
    out.append("};")
    out.append("")
    out.append(f"static const int32_t kQcdTopoEntryCount = {len(entries)};")
    out.append("")
    out.append("#endif /* PYQCD_EINSUM_CUDA_TOPOLOGY_STEPS_H */")
    return "\n".join(out) + "\n"


def parse_header(path: pathlib.Path) -> dict[str, Any]:
    """只读解析现有表：步序数组、``(order, steps)`` 索引与表项。"""
    text = path.read_text(encoding="utf-8")
    arrays = {
        int(position): tuple(int(value) for value in body.replace(" ", "").split(",")
                             if value)
        for position, body in _ORDER_RE.findall(text)}
    orders_section = text.split("static const qcd_topology_order kQcdTopoOrders[]")[-1]
    orders = [(int(position), int(steps))
              for position, steps in _ORDERS_RE.findall(orders_section)]
    entries = [(int(key, 16), int(fnv, 16), int(order), int(arity))
               for key, fnv, order, arity in _ENTRY_RE.findall(text)]
    pairs: list[tuple[tuple[int, int], ...]] = []
    for position, steps in orders:
        flat = arrays[position]
        if len(flat) != 2 * steps:
            raise AssertionError(
                f"order {position}: {len(flat)} values for steps={steps}")
        pairs.append(tuple(zip(flat[0::2], flat[1::2])))
    return {"pairs": pairs, "entries": entries}


def verify_cuda(records: Sequence[dict[str, Any]], path: pathlib.Path) -> dict[str, Any]:
    """逐条比对 ``key → (步序, fnv, arity)``；返回计数与不一致明细。

    ``records`` 是语料前 N 条的生成结果。N 小于全量时文件里必然多出表项——那是
    扩表后的正常状态，只作信息报告，不判失败；步序总表同理，只在两边覆盖同一批
    表项（数量相等）时才要求逐一相等。真正判失败的是：生成的 key 在文件里缺失，
    或 key 相同而 fnv/arity/步序不同。
    """
    header = parse_header(path)
    orders, entries, _count = layout(records)
    generated = {entry["key"]: entry for entry in entries}
    recorded = {key: (fnv, order, arity) for key, fnv, order, arity in header["entries"]}

    missing = sorted(set(generated) - set(recorded))
    extra = sorted(set(recorded) - set(generated))
    mismatches: list[str] = []
    for key, (recorded_fnv, order, arity) in recorded.items():
        entry = generated.get(key)
        if entry is None:
            continue
        if entry["fnv"] != recorded_fnv:
            mismatches.append(f"{key:#018x}: fnv {entry['fnv']:#018x} != {recorded_fnv:#018x}")
        elif entry["arity"] != arity:
            mismatches.append(f"{key:#018x}: arity {entry['arity']} != {arity}")
        elif entry["pairs"] != header["pairs"][order]:
            mismatches.append(f"{key:#018x}: steps {entry['pairs']} != "
                              f"{header['pairs'][order]}")
    covers_file = len(entries) == len(recorded)
    orders_match = orders == header["pairs"]
    if covers_file and not orders_match:
        mismatches.append(f"deduped order list differs "
                          f"({len(orders)} vs {len(header['pairs'])})")
    return {
        "compared": len(recorded) - len(extra),
        "missing": missing,
        "extra": extra,
        "mismatches": mismatches,
        "orders_match": orders_match,
        "covers_file": covers_file,
        "generated_orders": len(orders),
        "recorded_orders": len(header["pairs"]),
        "recorded_entries": len(recorded),
    }


def generate_cuda(target: pathlib.Path, *,
                  shape_source: str = DEFAULT_SHAPE_SOURCE,
                  limit: int | None = None,
                  compress: bool = True) -> dict[str, Any]:
    """生成 CUDA 表（``.h`` + ``.h.gz``）。"""
    records = collect_cuda(limit, shape_source=shape_source)
    text = render_cuda(records, shape_source)
    data = text.encode("utf-8")
    target.write_bytes(data)
    orders, entries, order_count = layout(records)
    report: dict[str, Any] = {"target": target, "bytes": len(data),
                              "entries": len(entries), "orders": order_count,
                              "pairs": sum(len(item) for item in orders),
                              "gz": None}
    if compress:
        # 同 ``generate_kupl``：``mtime=0`` 保证同一输入产出同一个 .gz 字节流。
        payload = gzip.compress(data, compresslevel=9, mtime=0)
        gz_path = pathlib.Path(str(target) + ".gz")
        gz_path.write_bytes(payload)
        report["gz"] = (gz_path, len(payload))
    return report


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════
def _report(kind: str, report: dict[str, Any], elapsed: float) -> None:
    gz = report["gz"]
    detail = " ".join(f"{key}={report[key]}" for key in ("entries", "steps", "orders", "pairs")
                      if key in report)
    line = (f"{kind}: wrote {report['target']} ({report['bytes']} bytes, {detail}) "
            f"in {elapsed:.2f}s")
    if gz is not None:
        path, size = gz
        line += f"; {path.name}={size} bytes ({report['bytes'] / size:.1f}x)"
    print(line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=("kupl", "cuda", "all"), default=None,
                        help="生成哪个后端（默认 all；给了 --verify 时默认 cuda）")
    parser.add_argument("--output", default=None,
                        help="单目标的输出路径（默认写仓库内的表；.gz 与本体同目录）")
    parser.add_argument("--limit", type=int, default=None,
                        help=f"只取前 N 条语料（默认全量 {MAX_KERNELS}）")
    parser.add_argument("--shape-source", choices=("roles",),
                        default=DEFAULT_SHAPE_SOURCE,
                        help="形状源：roles=ROLE_SHAPES 角色形状（唯一取值）；"
                             "production（生产异形轴）与 uniform（等长探针）两条旧口径已删除")
    parser.add_argument("--no-gzip", action="store_true", help="只写表本体，不产出 .gz")
    parser.add_argument("--verify", type=int, nargs="?", const=MAX_KERNELS,
                        default=None, metavar="N",
                        help=f"只核对现有 CUDA 表的前 N 条（默认全量 {MAX_KERNELS}；"
                             f"传 {BASE_ENTRIES} 可只核对历史基础表），不写盘")
    args = parser.parse_args(argv)

    target = args.target or ("cuda" if args.verify is not None else "all")
    if args.output and target == "all":
        parser.error("--output 只用于单目标，请同时给 --target kupl|cuda")
    if args.verify is not None and target != "cuda":
        parser.error("--verify 只核对 CUDA 表，请用 --target cuda（或不给 --target）")

    if args.verify is not None:
        path = pathlib.Path(args.output) if args.output else CUDA_TARGET
        records = collect_cuda(args.verify, shape_source=args.shape_source)
        report = verify_cuda(records, path)
        print(f"compared {report['compared']} entries against {path}: "
              f"missing={len(report['missing'])} extra={len(report['extra'])} "
              f"mismatch={len(report['mismatches'])} "
              f"orders {report['generated_orders']}/{report['recorded_orders']} "
              f"match={report['orders_match']} "
              f"covers_file={report['covers_file']}")
        for item in report["mismatches"][:20]:
            print(f"  {item}")
        for key in report["missing"][:20]:
            print(f"  missing key {key:#018x}")
        if args.verify < MAX_KERNELS:
            print(f"  ({len(report['extra'])} extra entries are expected: this check "
                  f"covers only the first {args.verify} corpus rows)")
        else:
            for key in report["extra"][:20]:
                print(f"  extra key {key:#018x}")
        drifted = bool(report["missing"] or report["mismatches"])
        if args.verify >= MAX_KERNELS:
            drifted = drifted or bool(report["extra"])
        return 0 if not drifted else 1

    wanted = ("kupl", "cuda") if target == "all" else (target,)
    for kind in wanted:
        started = time.perf_counter()
        shape_source = args.shape_source
        output = pathlib.Path(args.output) if args.output else (
            ROOT / KUPL_TARGET if kind == "kupl" else CUDA_TARGET)
        if kind == "kupl":
            report = generate_kupl(output, shape_source=shape_source,
                                   limit=args.limit, compress=not args.no_gzip)
        else:
            report = generate_cuda(output, shape_source=shape_source,
                                   limit=args.limit, compress=not args.no_gzip)
        _report(kind, report, time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
