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
# 生成期探针尺寸：步序列的结构与 dim 无关（所有轴等长），取 2 即可。
PROBE_DIM = 2
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


def plan_steps(equation: str) -> tuple[list[dict], list[str], str]:
    """镜像一次 greedy 收缩，返回每步的两侧项、输出项与源槽位。"""
    terms, out = parse(equation)
    shapes = [(PROBE_DIM,) * len(term) for term in terms]
    _, info = contract_path(equation, *shapes, shapes=True, optimize="greedy")

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


def decompose(left: str, right: str, out: str) -> dict:
    """把二元 einsum 步拆成 GEMM 的 batch/M/K/N 分组与轴序。"""
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


def encode_steps(equation: str) -> bytes:
    steps, _terms, out = plan_steps(equation)
    if not steps or steps[-1]["out"] != out:
        raise AssertionError(f"{equation}: last step does not produce the output")
    stream = bytearray()
    for step in steps:
        spec = decompose(step["left"], step["right"], step["out"])
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
            "{steps}}},".format(hash=entry["hash"], sig=entry["signature"],
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


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[2]
    equations = _parse_equations()
    assert len(equations) <= MAX_KERNELS, len(equations)

    entries: list[dict] = []
    seen_hash: dict[int, str] = {}
    for equation in equations:
        terms, out = parse(equation)
        blob = encode_steps(equation)
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
    """步数：按格式走一遍（跳过 nibble 流），用于记录校验。"""
    position = 0
    count = 0
    while position < len(blob):
        if position + 6 > len(blob):
            raise AssertionError("truncated step header")
        rank_a = blob[position + 3] >> 4
        rank_b = blob[position + 3] & 0x0F
        rank_c = blob[position + 4]
        position += 6 + (rank_a + rank_b + rank_c + 1) // 2
        count += 1
    if position != len(blob):
        raise AssertionError("step table does not divide evenly")
    return count


if __name__ == "__main__":
    raise SystemExit(main())
