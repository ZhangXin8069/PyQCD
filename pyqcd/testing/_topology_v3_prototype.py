"""步表 v3 格式推演（**不改生产代码**，只验证新格式能否自洽地服务异形轴）。

问题背景
--------
v2 步表只存轴序（4 bit/轴）与分组秩，轴长靠 ``dim^(rank-1-p)`` 反推，因此只能
服务"所有轴长都相等"的超立方体。生产 scene 档的轴长是 ``{4,5,40,50}`` 混杂，
于是 KUPL 拓扑入口在生产形状上恒 miss。

纸面推演已证明：**中间体长度无法由轴序拼接推出**。例（seq=2，scene 形状）::

    step3 产出 nbdf，按 batch+rows+cols 拼出的长度是 [4,5,4,5]
    step4 消费 nbdf 时 inner 段真实长度是 [5,40]     ← 40 来自 Lt，拼接里根本没有

原因是轴序只记录"第几轴"，不记录"哪个 label"，而不同 label 的长度不同。

v3 设计
-------
在每个拓扑的 code 段前追加**初始 label 布局**，并给每步追加**输出 label 序列**::

    段首（label 块）:
        [0] uint8 arity
        [1..] 逐操作数: uint8 rank, 然后 rank 个 uint8 label id（canonical 编号）

    每步仍以 v2 的 6 字节头 + 轴序 nibble 流开头，紧随其后追加:
        [..] uint8 c_labels[rank_c]

运行期只需要 ``sizes[label]``（来自 version-2 计划的 ``sizes``，与生成器
``canonical_signature`` 同口径）就能拿到每一步的逐轴真实长度：

    a_labels[j] = slot_labels[slot_a][a_axes[j]]      # 按轴序取
    a_len[j]    = sizes[a_labels[j]]
    B  = prod(a_len[0 : batch])
    M  = prod(a_len[batch : batch+rows])
    K  = prod(a_len[batch+rows : rank_a])
    N  = prod(b_len[batch+inner : rank_b])
    产出中间体: slot_labels[slot_c] = c_labels, slot_extents 由 c_labels 查 sizes

本模块用**真实 v2 步表字节**做基线，重新编码成 v3 并在纯 Python 里跑完整条收缩，
与 numpy 对照。验证通过后才改 C++ 与生成器。

用法::

    python -m pyqcd.testing._topology_v3_prototype
"""
from __future__ import annotations

import re
import pathlib
import unittest
from typing import Any

import numpy as np

from ._topology_kernel_gen import canonical_signature, parse, plan_steps, decompose
import pyqcd.einsum._wick_tasklist as tasklist
from pyqcd.einsum import _bench_scene
from pyqcd.einsum.bench_specialized import _label_sizes

INC = pathlib.Path(__file__).resolve().parents[2] / "cpp/kupl/einsum/src/topology_kernels.inc"
B36 = "0123456789abcdefghijklmnopqrstuvwxyz"
MAX_LABELS = 255


def _canonical_map(terms: list[str], out: str) -> dict[str, int]:
    """字母 → canonical label id（与 ``canonical_signature`` / symbol_order 同口径）。"""
    order: list[str] = []
    for term in list(terms) + [out]:
        for ch in term:
            if ch not in order:
                order.append(ch)
    return {ch: index for index, ch in enumerate(order)}


def encode_v3(equation: str) -> bytes:
    """把一条方程编码成 v3 字节流（label 块 + v2 逐步 + 每步 c_labels）。"""
    terms, out = parse(equation)
    labels = _canonical_map(terms, out)
    if len(labels) > MAX_LABELS:
        raise AssertionError(f"{equation}: {len(labels)} labels exceed uint8")

    stream = bytearray()
    # ---- label 块 ----
    stream.append(len(terms))
    for term in terms:
        stream.append(len(term))
        stream.extend(labels[ch] for ch in term)

    steps, _terms, _out = plan_steps(equation)
    for step in steps:
        spec = decompose(step["left"], step["right"], step["out"])
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
        # ---- v3：该步输出的 label 序列，按**该步输出张量的目标轴序**排列 ----
        # 中间体经 unpack 后按 step["out"] 的轴序存放，因此后续步骤按位置引用它时，
        # 拿到的就是这张 label 表；不能记成 b 序（两者仅在 c_axes 为恒等时相同）。
        stream.extend(labels[ch] for ch in step["out"])
    return bytes(stream)


def decode_v3(stream: bytes) -> dict[str, Any]:
    """解码 v3 字节流，返回 label 块与逐步描述。"""
    position = 0
    arity = stream[position]
    position += 1
    initial_labels = []
    for _ in range(arity):
        rank = stream[position]
        position += 1
        initial_labels.append(list(stream[position:position + rank]))
        position += rank

    steps = []
    while position < len(stream):
        head = position
        a_src = stream[position]; a_src = a_src - 256 if a_src >= 128 else a_src
        b_src = stream[position + 1]; b_src = b_src - 256 if b_src >= 128 else b_src
        dst = stream[position + 2]; dst = dst - 256 if dst >= 128 else dst
        rank_a = stream[position + 3] >> 4
        rank_b = stream[position + 3] & 0x0F
        rank_c = stream[position + 4]
        batch = stream[position + 5] >> 4
        inner = stream[position + 5] & 0x0F
        need = rank_a + rank_b + rank_c
        axes: list[int] = []
        position += 6
        while len(axes) < need:
            axes.append((stream[position] >> 4) & 0x0F)
            if len(axes) < need:
                axes.append(stream[position] & 0x0F)
            position += 1
        c_labels = list(stream[position:position + rank_c])
        position += rank_c
        steps.append(dict(
            offset=head, a_src=a_src, b_src=b_src, dst=dst, rank_a=rank_a,
            rank_b=rank_b, rank_c=rank_c, batch=batch, inner=inner,
            a_axes=axes[:rank_a], b_axes=axes[rank_a:rank_a + rank_b],
            c_axes=axes[rank_a + rank_b:], c_labels=c_labels))
    return dict(arity=arity, initial_labels=initial_labels, steps=steps)


def execute_v3(packed: dict[str, Any], operands: list[np.ndarray],
               sizes: dict[int, int]) -> np.ndarray:
    """在纯 Python 里按 v3 语义执行整条收缩（完全按 label 取长度）。"""
    arity = packed["arity"]
    labels = {index: list(packed["initial_labels"][index]) for index in range(arity)}
    data = {index: np.asarray(operands[index]) for index in range(arity)}
    for step in packed["steps"]:
        slot_a = step["a_src"] if step["a_src"] >= 0 else arity + (-1 - step["a_src"])
        slot_b = step["b_src"] if step["b_src"] >= 0 else arity + (-1 - step["b_src"])
        slot_c = arity + step["dst"] if step["dst"] >= 0 else None

        a_labels = [labels[slot_a][axis] for axis in step["a_axes"]]
        b_labels = [labels[slot_b][axis] for axis in step["b_axes"]]
        a_len = [sizes[item] for item in a_labels]
        b_len = [sizes[item] for item in b_labels]

        batch, inner = step["batch"], step["inner"]
        row_rank = step["rank_a"] - batch - inner
        col_rank = step["rank_b"] - batch - inner
        # batch / inner 两侧的 label 必须一致（顺序可以不同）
        assert sorted(a_labels[:batch]) == sorted(b_labels[:batch]), (
            "batch label mismatch", a_labels[:batch], b_labels[:batch])
        assert sorted(a_labels[batch + row_rank:]) == sorted(b_labels[batch:batch + inner]), (
            "inner label mismatch", a_labels[batch + row_rank:],
            b_labels[batch:batch + inner])

        # 打包成规范序 [batch, M, K] 与 [batch, K, N]
        left = np.transpose(np.asarray(data[slot_a]), step["a_axes"])
        right = np.transpose(np.asarray(data[slot_b]), step["b_axes"])
        count_batch = int(np.prod(a_len[:batch])) if batch else 1
        count_rows = int(np.prod(a_len[batch:batch + row_rank])) if row_rank else 1
        count_inner = int(np.prod(a_len[batch + row_rank:])) if inner else 1
        count_cols = int(np.prod(b_len[batch + inner:])) if col_rank else 1
        left = left.reshape(count_batch, count_rows, count_inner)
        right = right.reshape(count_batch, count_inner, count_cols)
        packed_out = np.matmul(left, right)  # [batch, M, N]

        # packed_out 的形状是 [batch, M, N]，即 b 序（batch + rows + cols）的分组。
        # c_axes[p] 是「目标第 p 轴在 c_order=batch+rows+cols 里的位置」，因此把
        # [batch,M,N] 摊平后按 c_axes 取轴即可得到目标轴序。
        batch_shape = tuple(a_len[:batch])
        row_shape = tuple(a_len[batch:batch + row_rank])
        col_shape = tuple(b_len[batch + inner:])
        grouped = packed_out.reshape(batch_shape + row_shape + col_shape)
        out_rank = step["rank_c"]
        if out_rank:
            unpacked = np.transpose(grouped, step["c_axes"])
        else:
            unpacked = grouped.reshape(())

        if slot_c is not None:
            labels[slot_c] = list(step["c_labels"])
            data[slot_c] = unpacked
        else:
            return unpacked
    raise AssertionError("v3 stream did not produce an output")


def _scene_shapes(equation: str, roles: tuple[str, ...],
                  table: dict[str, int]) -> list[tuple[int, ...]]:
    sizes = _label_sizes(equation.split("->")[0].split(","), roles, table)
    return [tuple(int(sizes[ch]) for ch in term)
            for term in equation.split("->")[0].split(",")]


def _label_sizes_map(equation: str, shapes: list[tuple[int, ...]]) -> dict[int, int]:
    terms, out = parse(equation)
    labels = _canonical_map(terms, out)
    result: dict[int, int] = {}
    for term, shape in zip(terms, shapes):
        for ch, extent in zip(term, shape):
            label = labels[ch]
            known = result.setdefault(label, int(extent))
            assert known == int(extent), (equation, ch, known, extent)
    return result


# ── 1. 编码/解码自洽 ────────────────────────────────────────────────────────

def test_v3_roundtrip_for_base_corpus():
    """v3 编码→解码必须自洽：label 块、逐步轴序、c_labels 全部可还原。"""
    checked = 0
    for seq in range(1, 2583, 97):
        task = tasklist.by_seq()[seq]
        stream = encode_v3(task.equation)
        packed = decode_v3(stream)
        terms, out = parse(task.equation)
        assert packed["arity"] == len(terms), (seq, packed["arity"])
        for index, term in enumerate(terms):
            assert len(packed["initial_labels"][index]) == len(term), (seq, index)
        assert packed["steps"], seq
        last = packed["steps"][-1]
        assert last["dst"] == -1, seq
        assert len(last["c_labels"]) == len(out), (seq, last["c_labels"], out)
        # 最终输出的 label 必须与方程输出按 canonical id 一一对应
        labels_map = _canonical_map(*parse(task.equation))
        assert last["c_labels"] == [labels_map[ch] for ch in out], (seq, last["c_labels"])

        checked += 1
    assert checked >= 20, checked


# ── 2. 等长超立方体：与 numpy 一致 ──────────────────────────────────────────

def test_v3_matches_numpy_on_uniform_shapes():
    """等长 ``dim`` 下，v3 执行结果必须与 numpy 一致（正确性基线）。"""
    for seq in (2, 500, 1000):
        task = tasklist.by_seq()[seq]
        terms = task.equation.split("->")[0].split(",")
        for dim in (2, 3):
            shapes = [(dim,) * len(term) for term in terms]
            rng = np.random.default_rng(seq + dim)
            operands = [np.ascontiguousarray(rng.standard_normal(shape))
                        for shape in shapes]
            sizes = _label_sizes_map(task.equation, shapes)
            got = execute_v3(decode_v3(encode_v3(task.equation)), operands, sizes)
            expected = np.einsum(task.equation, *operands, optimize="greedy")
            scale = float(np.max(np.abs(expected))) or 1.0
            error = float(np.max(np.abs(got - expected))) / scale
            assert error <= 1e-12, (seq, dim, error)


# ── 3. 异形轴：本次改动的目标 ───────────────────────────────────────────────

def test_v3_serves_production_heterogeneous_shapes():
    """生产 scene 形状（轴长混杂 {4,5,40,50}）下 v3 必须给出正确结果。

    这是 v2 做不到的能力：v2 只存轴序、轴长靠 ``dim`` 幂次反推，异形轴直接拒绝。
    """
    table = _bench_scene.dims("scene")
    for seq in (2, 500, 1000, 1700, 2400):
        task = tasklist.by_seq()[seq]
        shapes = _scene_shapes(task.equation, task.roles, table)
        distinct = sorted({extent for shape in shapes for extent in shape})
        assert len(distinct) > 1, (seq, distinct)

        rng = np.random.default_rng(seq)
        operands = [np.ascontiguousarray(rng.standard_normal(shape))
                    for shape in shapes]
        sizes = _label_sizes_map(task.equation, shapes)
        got = execute_v3(decode_v3(encode_v3(task.equation)), operands, sizes)
        expected = np.einsum(task.equation, *operands, optimize="greedy")
        scale = float(np.max(np.abs(expected))) or 1.0
        error = float(np.max(np.abs(got - expected))) / scale
        assert got.shape == expected.shape, (seq, got.shape, expected.shape)
        assert error <= 1e-12, (seq, distinct, error)


TESTS = (
    test_v3_roundtrip_for_base_corpus,
    test_v3_matches_numpy_on_uniform_shapes,
    test_v3_serves_production_heterogeneous_shapes,
)


def main() -> int:
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
        except Exception as exc:                          # noqa: BLE001 - 报告用
            failed += 1
            print(f"ERROR {test.__name__}: {type(exc).__name__}: {exc}", flush=True)
        else:
            passed += 1
            print(f"PASS {test.__name__}", flush=True)
    print(f"{passed} passed, {skipped} skipped, {failed} failed", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
