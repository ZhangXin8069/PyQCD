"""生成 per-拓扑特化内核（GEMM 分步形式）到 ``cpp/kupl/einsum/src/topology_kernels.inc``。

设计（与旧的"k 层嵌套循环"实现完全不同）：

1. 生成期用 ``pyqcd.einsum.contract_path(..., optimize="greedy")`` 求每条方程的
   **收缩步序列**，把每一步固化为「打包(pack) → cblas GEMM → 解包(unpack)」的
   静态描述；
2. 所有轴的步长只由 ``dim`` 的幂次决定（每条轴长度恒为 dim），因此描述里只存
   *轴的次序*（4 bit/轴），不存 stride，运行期零解析；
3. 2582 条方程 × 约 14 步 ≈ 3.6 万步，逐步展开成源码会超过 10 MB，所以采用
   **步表 + 参数化 GEMM 执行器**（``cpp/kupl/einsum/src/topology_gemm.h``）：
   每条拓扑仍有一个具名入口 ``topo_NNNN``（编译期常量表索引），函数体只有一次
   调用。取舍见 ``skills/pyqcd-einsum/references/native-implementation.md``。

步表字节格式（每条拓扑一段 ``uint8`` 流）::

    [0] int8  a_src     >=0 操作数下标；<0 表示中间槽 -1-slot
    [1] int8  b_src
    [2] int8  dst       >=0 中间槽；-1 表示最终输出缓冲区
    [3] uint8 (rank_a << 4) | rank_b        （各 <= 15）
    [4] uint8 rank_c                         （<= 15）
    [5] uint8 (batch << 4) | inner           （各 <= 15，M/N 由秩复原）
    [6..] nibble 流：a_axes[rank_a] + b_axes[rank_b] + c_axes[rank_c]，每轴 4 bit

``a_axes[j]`` 是 GEMM 规范序 ``[batch..., M..., K...]`` 第 j 轴在 A 源张量里的
轴位置；``b_axes`` 对应 ``[batch..., K..., N...]``；``c_axes[p]`` 是目的张量第 p
轴在规范序 ``[batch..., M..., N...]`` 里的下标。

重新生成::

    python -m pyqcd.testing._topology_kernel_gen
"""

from __future__ import annotations

import pathlib

from ..einsum import contract_path
from ._topology_equations_contract import _parse_equations

MAX_KERNELS = 2582
MAX_RANK = 15
# 生成期探针尺寸：**仅在没有生产尺寸可用时**才退回等长探针。
#
# 历史教训（2026-11-14 实测）：早期无条件用 ``PROBE_DIM = 2`` 生成收缩序，于是
# 步序列由 dim=2 的超立方体决定。等长时全部轴长相同，收缩序的代价与轴长无关，
# 看不出问题；一到生产异形轴（``Ns=4``/``Nev=100``/``Mom``/``Lt``）就暴露：
# 同样 13 步可能多做 12 倍的乘加，巨中间体如 ``M=15625 K=1600 N=25``（而
# dedicated 的同规模步是 ``625×625×400``）。生产方程上实测 topology 路径
# 246.8 ms vs dedicated 4.7 ms。
#
# 正解是用语料自带的 ``roles`` 反推每个 label 的真实尺寸
# （``pyqcd.einsum.bench_specialized._label_sizes``），再在该尺寸下求路径。
PROBE_DIM = 2
#: 生成期路径优化器。``"auto"`` 与 ``_kernel_kupl.try_dedicated`` 同口径
#: （它传 ``"auto"``，见 ``_dedicated_plan`` 的说明）：两者在部分式子上差 2.9×
#: FLOPs，实测 seq=646 上 greedy 4.08e8 MAC vs auto 1.08e8 MAC（26%）。
#: 用 ``"greedy"`` 生成会白白让步表多做 4 倍的乘加。
_OPTIMIZE = "auto"
TARGET = "cpp/kupl/einsum/src/topology_kernels.inc"

_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3


def parse(equation: str) -> tuple[list[str], str]:
    lhs, rhs = equation.split("->")
    return lhs.split(","), rhs


def canonical_signature(terms: list[str], out: str) -> str:
    """与 ``pyqcd.einsum._native_plan`` 的 symbol_order 同口径的规范签名。

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
    """镜像一次 greedy 收缩，返回每步的两侧项、输出项与源槽位。

    ``sizes`` 给出每个 label 的**真实生产长度**时，收缩序就在该异形轴上定型；
    缺省则退回 ``PROBE_DIM`` 等长探针（仅用于没有 roles 的临时调用）。两者的
    步序列**结构**相同（操作数个数与最后一步都产出整条方程的输出），差别只在
    二元步的结合次序——那正是决定 GEMM 形状与中间体大小的东西。
    """
    terms, out = parse(equation)
    if sizes is None:
        shapes = [(PROBE_DIM,) * len(term) for term in terms]
    else:
        try:
            shapes = [tuple(int(sizes[ch]) for ch in term) for term in terms]
        except KeyError as missing:
            raise AssertionError(
                f"{equation}: label {missing.args[0]!r} has no production size; "
                f"the corpus roles must bind every label") from None
    _, info = contract_path(equation, *shapes, shapes=True, optimize=_OPTIMIZE)

    pool_terms = list(terms)
    pool_slots = list(range(len(terms)))
    next_slot = 0
    steps: list[dict] = []
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
        steps.append({"left": tmp[0], "right": tmp[1], "out": result,
                      "a_src": a_slot, "b_src": b_slot, "dst": dst})
        pool_terms.append(result)
        # 中间槽在步表里编码成 -1-slot：非负数一律表示操作数下标，两者不能撞号。
        pool_slots.append(-1 - dst)
    if pool_terms != [out]:
        raise AssertionError(f"{equation}: the path does not end on the equation output")
    return steps, terms, out


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
    # batch 与 inner 是两侧共用的组，只能选一个顺序；这里用 reverse=True（源轴上
    # 更靠后的标签排在组内更靠后），使右侧大张量保持恒等。历史教训：早期按左操作
    # 数升序排序，令 1M 元素的右张量走非连续 pack，单步 7.94 ms（GEMM 仅 0.5 ms）。
    # ``pyqcd/einsum/_kernel_kupl._dedicated_step`` 用真实元素数选大张量原序，
    # 是同一原则的数据驱动版本；两者在等长时结果一致。
    #
    # 2026-11-15 补：固定 ``reverse`` 只照顾右张量，左张量明显更大时反而让**大**
    # 的一侧去付置换的账。实测 seq=3（``ambn,codp,egfh,ce,mo,bdf,nph->ag``）第 2 步
    # ``nph,egfh->npegf``：左侧 100 元素、右侧 4,000,000 元素，固定序给出
    # ``c_axes=[1,0,4,3,2]``（非恒等），要求 unpack 整个 4M 中间体；改按元素数取
    # 大张量原序后 ``c_axes=[0,1,2,3,4]``（恒等，零拷贝），该步 42.6 ms → 1.6 ms。
    # 传入 ``sizes`` 时走数据驱动，与 dedicated 的选法一致。
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
    """编码一条方程的步表（v3）。

    ``sizes`` 为该方程每个 label 的真实生产长度（见 :func:`plan_steps`）。给了
    它，收缩序就按生产异形轴定型；``None`` 退回等长探针。

    v3 = **label 块 + v2 逐步描述 + 每步 c_labels**。v2 只存轴序（4 bit/轴），
    轴长靠 ``dim`` 幂次反推，因此只能服务"所有轴长相等"的超立方体；生产 scene
    的轴长是 ``{4,5,40,50}`` 混杂，v2 直接拒绝。v3 把 label 带上，运行期用
    ``sizes[label]`` 取真实长度即可：中间体长度**无法**由轴序拼接推出（拼接结果
    与真实 label 长度不同），这是 v3 必须存在的理由。

    段首::

        uint8 arity
        逐操作数: uint8 rank + rank×uint8 label_id（canonical 编号，与
                  ``canonical_signature`` / symbol_order 同口径）

    每步沿用 v2 的 6 字节头 + 轴序 nibble 流，紧随其后追加::

        uint8 c_labels[rank_c]     该步输出张量**目标轴序**的 label id

    中间体经 unpack 后按 ``step["out"]`` 的轴序存放，所以后续步骤按位置引用它时
    拿到的正是这张表；记成 b 序会在下一步对不上（实测 step1 产出 ``aedf`` 的
    b 序是 ``['a','f','d','e']``，而 step3 引用时是 ``['a','e','f','d']``）。
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


def render(records: list[dict], kernels: list[dict], total_code: int) -> str:
    """``records`` 按 hash 升序（二分查找用），``kernels`` 与语料同序（索引即语料序号）。"""
    out: list[str] = []
    out.append("/* 由 pyqcd/testing/_topology_kernel_gen.py 生成，请勿手工编辑。")
    out.append(f" * 覆盖 refer/sush/wick_12cases.md 的 {len(records)} 条收缩路径；每条由")
    out.append(" * pyqcd.einsum.contract_path(optimize='greedy') 求出的「打包 → cblas GEMM")
    out.append(" * → 解包」步序列描述，执行器见 src/topology_gemm.h。")
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


def _production_sizes(equation: str, roles: tuple[str, ...]) -> dict[str, int]:
    """由语料 ``roles`` + 生产 dims 反推每个 label 的真实长度。

    直接复用 ``pyqcd.einsum.bench_specialized._label_sizes``：那是本库**唯一**
    的「角色 → 物理尺寸」权威实现（``gamma*``→``Ns``、基础表 ``VVV``/``VDV``→
    ``(Lt, Mom, Nev, ...)``、附录形态 ``VVV``/``VDV``→``(Mom, Nev, ...)``、peram
    剩余轴→``Nev``）。在这里另写一份必然与之漂移。

    生产 dims 取 ``distill`` 档（参考脚本 ``sink/source`` 端：``Nev=100``,
    ``Mom=1``）——那是语料里 peram/VVV/VDV 实际运行的那套尺寸。不取
    ``distill_curr``：两档的 label 集合一致，收缩序只取决于各轴长**相对**大小，
    ``Nev`` 从 100 变 400 是所有 peram 轴同比放大，不改变 greedy 的取舍。

    口径边界：本函数只服务编译表，而 ``MAX_KERNELS = 2582`` 恰好是**基础表**的
    全部（seq 1..2582，指标段纯小写）。附录形态（seq ≥ 2583，动量标签打头）不进
    编译表，所以 ``_label_sizes`` 对附录形态的修正（2026-11-15）**不改变**
    ``topology_kernels.inc`` 的字节（实测重新生成后 md5 一致）。
    """
    from ..einsum._bench_scene import dims as scene_dims
    from ..einsum.bench_specialized import _label_sizes

    terms, _out = parse(equation)
    return _label_sizes(terms, roles, scene_dims("distill"))


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[2]
    equations = _parse_equations()
    assert len(equations) <= MAX_KERNELS, len(equations)

    from ..einsum import _wick_tasklist as tasklist

    tasks = tasklist.by_seq()
    if len(tasks) < len(equations):
        raise AssertionError(
            f"corpus roles unavailable: {len(tasks)} tasks for {len(equations)} equations")

    entries: list[dict] = []
    seen_hash: dict[int, str] = {}
    for seq, equation in enumerate(equations, start=1):
        terms, out = parse(equation)
        task = tasks.get(seq)
        if task is None or task.equation != equation:
            raise AssertionError(
                f"seq={seq}: tasklist does not agree with the parsed corpus")
        sizes = _production_sizes(equation, task.roles)
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

    ordered = sorted(range(len(entries)), key=lambda index: entries[index]["hash"])
    sorted_entries = [entries[index] for index in ordered]
    offset = 0
    for record_index, entry in enumerate(sorted_entries):
        entry["offset"] = offset
        entry["record_index"] = record_index
        offset += len(entry["blob"])

    text = render(sorted_entries, entries, offset)
    target = root / TARGET
    target.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size} bytes, {len(entries)} kernels, "
          f"{offset} step-table bytes)")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
