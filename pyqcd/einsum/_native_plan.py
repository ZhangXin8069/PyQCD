"""Encode the version-2 execution-plan stream consumed by the native runtime."""

from __future__ import annotations

import operator
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

EXECUTION_MAGIC = 0x51434532
EXECUTION_VERSION = 2

_HEADER_WORDS = 9
_MAX_RANK = 64
_MAX_INPUTS = 1 << 16
_MAX_STEPS = 1 << 20
_MAX_SYMBOLS = 1 << 20
_MAX_PLAN_BYTES = 1 << 30

_DTYPE_IDS = {
    ("c", 16): 0,  # complex128
    ("c", 8): 1,   # complex64
    ("f", 8): 2,   # float64
    ("f", 4): 3,   # float32
}
_DTYPES = {
    0: np.dtype(np.complex128),
    1: np.dtype(np.complex64),
    2: np.dtype(np.float64),
    3: np.dtype(np.float32),
}


@dataclass(frozen=True)
class ExecutionStepInfo:
    """One decoded pair or unary execution step."""

    positions: tuple[int, ...]
    inputs: tuple[tuple[int, ...], ...]
    output: tuple[int, ...]


@dataclass(frozen=True)
class ExecutionPlanInfo:
    """Structural view of a decoded version-2 execution plan."""

    dtype_id: int
    sizes: tuple[int, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    output_labels: tuple[int, ...]
    steps: tuple[ExecutionStepInfo, ...]


@dataclass
class _LiveOperand:
    term: str | None
    shape: tuple[int, ...]


def dtype_id(dtype: Any) -> int:
    """Return the native dtype id for a supported, native-endian dtype."""
    value = np.dtype(dtype)
    if not value.isnative:
        raise ValueError(f"non-native-endian dtype is not supported: {value}")
    try:
        return _DTYPE_IDS[(value.kind, value.itemsize)]
    except KeyError as exc:
        raise ValueError(f"unsupported native einsum dtype: {value}") from exc


def dtype_from_id(value: int) -> np.dtype:
    """Return the numpy dtype associated with a native dtype id."""
    try:
        native_id = operator.index(value)
    except TypeError as exc:
        raise TypeError("dtype id must be an integer") from exc
    try:
        return _DTYPES[native_id]
    except KeyError as exc:
        raise ValueError(f"unknown native einsum dtype id: {native_id}") from exc


def _append_i32(buffer: bytearray, value: int) -> None:
    number = operator.index(value)
    if not -(1 << 31) <= number < (1 << 31):
        raise OverflowError(f"int32 value out of range: {number}")
    buffer.extend(struct.pack("=i", number))


def _append_i64(buffer: bytearray, value: int) -> None:
    number = operator.index(value)
    if not -(1 << 63) <= number < (1 << 63):
        raise OverflowError(f"int64 value out of range: {number}")
    buffer.extend(struct.pack("=q", number))


def _align(buffer: bytearray, alignment: int = 8) -> None:
    padding = (-len(buffer)) % alignment
    if padding:
        buffer.extend(b"\x00" * padding)


def _operand_shape(value: Any) -> tuple[int, ...]:
    try:
        dimensions = value.shape
    except AttributeError:
        dimensions = np.shape(value)
    try:
        shape = tuple(operator.index(dimension) for dimension in dimensions)
    except TypeError as exc:
        raise ValueError(f"operand shape must contain integers: {dimensions!r}") from exc
    if len(shape) > _MAX_RANK:
        raise ValueError(f"operand rank {len(shape)} exceeds native limit {_MAX_RANK}")
    if any(dimension < 0 for dimension in shape):
        raise ValueError(f"operand shape contains a negative extent: {shape}")
    return shape


def _operand_dtype(value: Any) -> np.dtype:
    try:
        return np.dtype(value.dtype)
    except AttributeError:
        return np.dtype(np.asarray(value).dtype)


def _validate_term(term: str, shape: tuple[int, ...], sizes: dict[str, int]) -> None:
    if not isinstance(term, str):
        raise ValueError(f"contraction term must be a string, got {type(term).__name__}")
    if len(term) > _MAX_RANK:
        raise ValueError(f"term rank {len(term)} exceeds native limit {_MAX_RANK}")
    if len(term) != len(shape):
        raise ValueError(f"term {term!r} rank {len(term)} does not match shape {shape}")
    if len(set(term)) != len(term):
        raise ValueError(f"native execution does not support repeated axes in {term!r}")
    for label, extent in zip(term, shape):
        if label in ",->" or label.isspace():
            raise ValueError(f"invalid label {label!r} in term {term!r}")
        previous = sizes.setdefault(label, extent)
        if previous != extent:
            raise ValueError(
                f"inconsistent size for label {label!r}: {previous} versus {extent}"
            )


def _term_shape(term: str, sizes: dict[str, int]) -> tuple[int, ...]:
    try:
        return tuple(sizes[label] for label in term)
    except KeyError as exc:
        raise ValueError(f"label {exc.args[0]!r} was not assigned an input shape") from exc


def _assign_term(
    live: _LiveOperand,
    term: str,
    sizes: dict[str, int],
) -> None:
    if live.term is None:
        live.term = term
    elif live.term != term:
        raise ValueError(f"inconsistent live term: expected {live.term!r}, got {term!r}")
    _validate_term(term, live.shape, sizes)


def _parse_step_equation(equation: str, arity: int) -> tuple[tuple[str, ...], str]:
    if not isinstance(equation, str):
        raise ValueError("contraction equation must be a string")
    left, separator, output = equation.partition("->")
    if not separator or "->" in output:
        raise ValueError(f"contraction step must contain exactly one '->': {equation!r}")
    terms = tuple(left.split(","))
    if len(terms) != arity:
        raise ValueError(
            f"step equation has {len(terms)} terms for arity {arity}: {equation!r}"
        )
    if len(set(output)) != len(output):
        raise ValueError(f"duplicate output axes in step equation: {equation!r}")
    return terms, output


def _positions(value: Any, arity: int) -> tuple[int, ...]:
    try:
        positions = tuple(operator.index(position) for position in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("each contraction position must be an integer") from exc
    if len(positions) != arity:
        raise ValueError(f"step has {len(positions)} positions for arity {arity}")
    if len(set(positions)) != len(positions):
        raise ValueError(f"step contains duplicate operand positions: {positions}")
    if any(position < 0 for position in positions):
        raise ValueError(f"step contains a negative operand position: {positions}")
    return positions


def _simulate_plan(
    contraction_list: Sequence[Sequence[Any]],
    operands: Sequence[Any],
) -> tuple[
    tuple[_LiveOperand, ...],
    list[_LiveOperand],
    list[tuple[tuple[int, ...], tuple[str, ...], str]],
    dict[str, int],
]:
    input_shapes = tuple(_operand_shape(operand) for operand in operands)
    num_inputs = len(input_shapes)
    if num_inputs > _MAX_INPUTS:
        raise ValueError(f"input count {num_inputs} exceeds native limit {_MAX_INPUTS}")
    if len(contraction_list) > _MAX_STEPS:
        raise ValueError(f"step count exceeds native limit {_MAX_STEPS}")

    live = [_LiveOperand(None, shape) for shape in input_shapes]
    initial_live = tuple(live)
    sizes: dict[str, int] = {}
    steps: list[tuple[tuple[int, ...], tuple[str, ...], str]] = []

    for step_number, step in enumerate(contraction_list):
        try:
            raw_positions, raw_removed, equation, remaining, _ = step
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"step {step_number} must contain five contraction-list fields"
            ) from exc
        try:
            arity = len(raw_positions)
        except TypeError as exc:
            raise ValueError(f"step {step_number} positions are not iterable") from exc
        if arity not in (1, 2):
            raise ValueError(f"step {step_number} arity must be one or two, got {arity}")
        positions = _positions(raw_positions, arity)
        if any(position >= len(live) for position in positions):
            raise ValueError(
                f"step {step_number} position is outside live operand range {len(live)}"
            )
        terms, output = _parse_step_equation(equation, arity)

        popped = [live[position] for position in positions]
        for position in sorted(positions, reverse=True):
            live.pop(position)
        for operand, term in zip(popped, terms):
            _assign_term(operand, term, sizes)

        try:
            removed = set(raw_removed)
        except TypeError as exc:
            raise ValueError(f"step {step_number} removed labels are not iterable") from exc
        used_labels = set().union(*(set(term) for term in terms))
        absent_output = set(output) - used_labels
        if absent_output:
            raise ValueError(
                f"step {step_number} output contains labels absent from its inputs: "
                f"{sorted(absent_output)!r}"
            )
        expected_removed = used_labels - set(output)
        if removed != expected_removed:
            raise ValueError(
                f"step {step_number} removed-label set {removed!r} does not match "
                f"equation {expected_removed!r}"
            )

        output_shape = _term_shape(output, sizes)
        _validate_term(output, output_shape, sizes)
        live.append(_LiveOperand(output, output_shape))
        steps.append((positions, terms, output))

        if remaining is not None:
            if isinstance(remaining, (str, bytes)) or not isinstance(remaining, Sequence):
                raise ValueError(f"step {step_number} remaining operands are not a sequence")
            if len(remaining) != len(live):
                raise ValueError(
                    f"step {step_number} remaining count {len(remaining)} does not match "
                    f"live count {len(live)}"
                )
            for operand, term in zip(live, remaining):
                _assign_term(operand, term, sizes)

    if len(live) != 1:
        raise ValueError(f"incomplete contraction plan left {len(live)} live operands")
    if live[0].term is None:
        raise ValueError("final output term was not assigned by the plan")
    if any(operand.term is None for operand in initial_live):
        raise ValueError("one or more initial operand terms were never assigned")
    return initial_live, live, steps, sizes


def encode_execution_plan(
    contraction_list: Sequence[Sequence[Any]],
    operands: Sequence[Any],
) -> bytes:
    """Serialize a contraction list and its operands as execution-plan version 2."""
    operand_list = list(operands)
    if not operand_list:
        raise ValueError("at least one operand is required")
    if not contraction_list:
        raise ValueError("at least one contraction step is required")

    operand_shapes = tuple(_operand_shape(operand) for operand in operand_list)
    operand_dtypes = tuple(dtype_id(_operand_dtype(operand)) for operand in operand_list)
    if len(set(operand_dtypes)) != 1:
        raise ValueError(
            f"mixed precision is not supported: {operand_dtypes}; cast all "
            "operands to one dtype (for example numpy.complex128) to use the "
            "native path")
    native_dtype_id = operand_dtypes[0]

    initial_entries, live, steps, sizes = _simulate_plan(contraction_list, operand_list)
    final_term = live[0].term
    assert final_term is not None

    initial_terms = [operand.term for operand in initial_entries if operand.term is not None]
    if len(initial_terms) != len(operand_shapes):
        raise ValueError("failed to recover all initial operand terms")

    symbol_order: list[str] = []
    for term in initial_terms + [final_term]:
        for label in term:
            if label not in symbol_order:
                symbol_order.append(label)
    if len(symbol_order) > _MAX_SYMBOLS:
        raise ValueError(f"symbol count {len(symbol_order)} exceeds native limit {_MAX_SYMBOLS}")
    if set(sizes) != set(symbol_order):
        raise ValueError("symbol size table does not match the encoded terms")

    symbol_ids = {label: index for index, label in enumerate(symbol_order)}
    symbol_sizes = [sizes[label] for label in symbol_order]
    output_labels = [symbol_ids[label] for label in final_term]

    buffer = bytearray(
        struct.pack(
            "=9i",
            EXECUTION_MAGIC,
            EXECUTION_VERSION,
            0,
            len(operand_shapes),
            len(steps),
            len(symbol_sizes),
            len(output_labels),
            native_dtype_id,
            0,
        )
    )
    _align(buffer)
    for size in symbol_sizes:
        _append_i64(buffer, size)
    for shape in operand_shapes:
        _append_i32(buffer, len(shape))
        _align(buffer)
        for extent in shape:
            _append_i64(buffer, extent)
    for label in output_labels:
        _append_i32(buffer, label)

    for positions, terms, output in steps:
        _append_i32(buffer, len(positions))
        for position in positions:
            _append_i32(buffer, position)
        for term in terms:
            _append_i32(buffer, len(term))
            for label in term:
                _append_i32(buffer, symbol_ids[label])
        _append_i32(buffer, len(output))
        for label in output:
            _append_i32(buffer, symbol_ids[label])

    if len(buffer) > _MAX_PLAN_BYTES:
        raise ValueError(f"execution plan is too large: {len(buffer)} bytes")
    struct.pack_into("=i", buffer, 2 * struct.calcsize("=i"), len(buffer))
    return bytes(buffer)


class _Reader:
    def __init__(self, data: memoryview) -> None:
        self.data = data
        self.offset = 0

    def read_i32(self) -> int:
        end = self.offset + 4
        if end > len(self.data):
            raise ValueError("truncated execution plan")
        value = struct.unpack_from("=i", self.data, self.offset)[0]
        self.offset = end
        return value

    def read_i64(self) -> int:
        aligned = (self.offset + 7) & ~7
        end = aligned + 8
        if end > len(self.data):
            raise ValueError("truncated execution plan")
        value = struct.unpack_from("=q", self.data, aligned)[0]
        self.offset = end
        return value


def _byte_view(plan: Any) -> memoryview:
    view = memoryview(plan)
    if view.ndim != 1 or view.format != "B":
        view = view.cast("B")
    return view


def _read_labels(reader: _Reader, num_symbols: int) -> tuple[int, ...]:
    rank = reader.read_i32()
    if rank < 0 or rank > _MAX_RANK:
        raise ValueError(f"invalid label rank {rank}")
    labels = tuple(reader.read_i32() for _ in range(rank))
    if any(label < 0 or label >= num_symbols for label in labels):
        raise ValueError(f"label outside symbol table: {labels}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate axes in label vector: {labels}")
    return labels


def decode_execution_plan(plan: Any) -> ExecutionPlanInfo:
    """Decode and validate a version-2 execution-plan byte stream."""
    data = _byte_view(plan)
    if len(data) < _HEADER_WORDS * 4:
        raise ValueError("execution plan is shorter than its header")
    reader = _Reader(data)
    header = tuple(reader.read_i32() for _ in range(_HEADER_WORDS))
    (
        magic,
        version,
        total_bytes,
        num_inputs,
        num_steps,
        num_symbols,
        output_ndim,
        native_dtype_id,
        reserved,
    ) = header
    if magic != EXECUTION_MAGIC:
        raise ValueError(f"bad execution-plan magic: 0x{magic & 0xFFFFFFFF:08x}")
    if version != EXECUTION_VERSION:
        raise ValueError(f"unsupported execution-plan version: {version}")
    if total_bytes != len(data) or total_bytes > _MAX_PLAN_BYTES:
        raise ValueError(f"invalid execution-plan byte count: {total_bytes}")
    if not 0 <= num_inputs <= _MAX_INPUTS:
        raise ValueError(f"invalid input count: {num_inputs}")
    if not 0 <= num_steps <= _MAX_STEPS:
        raise ValueError(f"invalid step count: {num_steps}")
    if not 0 <= num_symbols <= _MAX_SYMBOLS:
        raise ValueError(f"invalid symbol count: {num_symbols}")
    if not 0 <= output_ndim <= _MAX_RANK:
        raise ValueError(f"invalid output rank: {output_ndim}")
    dtype_from_id(native_dtype_id)
    if reserved != 0:
        raise ValueError(f"reserved header field must be zero, got {reserved}")

    sizes = tuple(reader.read_i64() for _ in range(num_symbols))
    if any(size < 0 for size in sizes):
        raise ValueError("symbol size table contains a negative extent")

    input_shapes: list[tuple[int, ...]] = []
    for _ in range(num_inputs):
        rank = reader.read_i32()
        if rank < 0 or rank > _MAX_RANK:
            raise ValueError(f"invalid input rank: {rank}")
        shape = tuple(reader.read_i64() for _ in range(rank))
        if any(extent < 0 for extent in shape):
            raise ValueError(f"input shape contains a negative extent: {shape}")
        input_shapes.append(shape)

    output_labels = tuple(reader.read_i32() for _ in range(output_ndim))
    if any(label < 0 or label >= num_symbols for label in output_labels):
        raise ValueError(f"output label outside symbol table: {output_labels}")
    if len(set(output_labels)) != len(output_labels):
        raise ValueError(f"duplicate output labels: {output_labels}")

    steps: list[ExecutionStepInfo] = []
    for _ in range(num_steps):
        arity = reader.read_i32()
        if arity not in (1, 2):
            raise ValueError(f"execution step arity must be one or two, got {arity}")
        positions = tuple(reader.read_i32() for _ in range(arity))
        if any(position < 0 for position in positions):
            raise ValueError(f"negative execution-step position: {positions}")
        inputs = tuple(_read_labels(reader, num_symbols) for _ in range(arity))
        output = _read_labels(reader, num_symbols)
        steps.append(ExecutionStepInfo(positions, inputs, output))
    if reader.offset != len(data):
        raise ValueError(f"execution plan has {len(data) - reader.offset} trailing bytes")

    return ExecutionPlanInfo(
        native_dtype_id,
        sizes,
        tuple(input_shapes),
        output_labels,
        tuple(steps),
    )


def _self_check() -> None:
    if __package__:
        from ._api import contract_path
    else:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from pyqcd.einsum._api import contract_path

    rng = np.random.default_rng(20261106)
    cases = (
        ("ij,jk->ik", ((2, 3), (3, 4)), np.complex128),
        ("ab,bc,cd->ad", ((2, 3), (3, 4), (4, 5)), np.float64),
        ("bij,bjk->bik", ((2, 3, 4), (2, 4, 5)), np.complex64),
        ("i,j->ij", ((3,), (4,)), np.float32),
    )
    for equation, shapes, dtype in cases:
        operands = tuple(
            (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
            if np.dtype(dtype).kind == "c"
            else rng.normal(size=shape).astype(dtype)
            for shape in shapes
        )
        _, path_info = contract_path(equation, *operands, optimize="greedy")
        plan = encode_execution_plan(path_info.contraction_list, operands)
        decoded = decode_execution_plan(plan)
        expected = np.einsum(equation, *operands)
        header = struct.unpack_from("=9i", plan)
        if header[:2] != (EXECUTION_MAGIC, EXECUTION_VERSION):
            raise AssertionError(f"{equation}: invalid plan header")
        if header[2] != len(plan) or header[3] != len(operands):
            raise AssertionError(f"{equation}: inconsistent plan counts")
        output_shape = tuple(decoded.sizes[label] for label in decoded.output_labels)
        if output_shape != expected.shape:
            raise AssertionError(f"{equation}: {output_shape} != {expected.shape}")
        lhs, output = equation.split("->")
        symbols = []
        for label in lhs.replace(",", "") + output:
            if label not in symbols:
                symbols.append(label)
        expected_output_labels = tuple(symbols.index(label) for label in output)
        if decoded.output_labels != expected_output_labels:
            raise AssertionError(f"{equation}: encoded output labels disagree")
        if decoded.input_shapes != shapes:
            raise AssertionError(f"{equation}: encoded input shapes disagree")
        if decoded.dtype_id != dtype_id(expected.dtype):
            raise AssertionError(f"{equation}: encoded dtype disagrees")
        if len(plan) < _HEADER_WORDS * 4:
            raise AssertionError(f"{equation}: truncated plan")
        print(
            f"PASS {equation}: bytes={len(plan)} steps={len(decoded.steps)} "
            f"symbols={len(decoded.sizes)} output={output_shape} dtype={dtype.__name__}"
        )


__all__ = [
    "EXECUTION_MAGIC",
    "EXECUTION_VERSION",
    "ExecutionPlanInfo",
    "ExecutionStepInfo",
    "decode_execution_plan",
    "dtype_from_id",
    "dtype_id",
    "encode_execution_plan",
]


if __name__ == "__main__":
    _self_check()
