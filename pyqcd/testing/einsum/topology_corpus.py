"""Corpus reader, canonical topology signatures and operand shapes.

The corpus is the Markdown table set in ``refer/sush/wick_12cases.md``: eleven
hadron topologies listing every ``result_indx`` equation that lqcddb emits.
This module only reads that text; it never imports anything from ``refer/``.

Corpus layout: one ``## <case>. <title>`` section per topology, each holding a
``| 拓扑类 | 图号 | result_indx | result_name |`` table.  The trailing
``## 指标说明`` / ``## 动量指标与流分量指标`` sections repeat the same tables with
momentum and current-gamma labels and are therefore not part of the corpus.
"""

from __future__ import annotations

import os
import re
import string
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

ENV_CORPUS = 'PYQCD_WICK_CASES'
DEFAULT_CORPUS = Path(__file__).resolve().parents[3] / 'refer' / 'sush' / 'wick_12cases.md'
LABELS = string.ascii_lowercase + string.ascii_uppercase
DEFAULT_SIZES = (2, 3, 4)
DEFAULT_DTYPE = 'complex64'

_CASE_HEADING = re.compile(r'^##\s+(\d+)\.\s')
_ROW = re.compile(r'^\| (T\d+) \| (\d+) \| `([^`]+)` \| `([^`]+)` \|$')
_DECLARED = re.compile(r'^\| (\d+) \| (\d+) \| (\d+) \|$')
_APPENDIX_MARKER = '<!-- MOMENTUM-INDEX-SECTIONS -->'


@dataclass(frozen=True)
class CorpusRow:
    """One ``result_indx`` equation of the corpus."""

    case: int
    diagram: int
    topology: str
    equation: str
    tensors: tuple[str, ...]

    @property
    def signature(self) -> tuple[tuple[str, ...], str]:
        return canonical_signature(self.equation)


def corpus_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the corpus file, honouring ``PYQCD_WICK_CASES``."""
    if path is not None:
        candidate = Path(path)
    else:
        candidate = Path(os.environ.get(ENV_CORPUS) or DEFAULT_CORPUS)
    candidate = candidate.expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(
            f'Wick corpus not found: {candidate} (override with {ENV_CORPUS})')
    return candidate


def load_corpus(path: str | os.PathLike[str] | None = None) -> tuple[CorpusRow, ...]:
    """Parse the eleven case tables into ``CorpusRow`` records in file order."""
    rows: list[CorpusRow] = []
    case: int | None = None
    for line in corpus_path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith(_APPENDIX_MARKER):
            break
        heading = _CASE_HEADING.match(line)
        if heading:
            case = int(heading.group(1))
            continue
        if line.startswith('## '):
            case = None
            continue
        if case is None:
            continue
        match = _ROW.match(line)
        if match is None:
            continue
        tensors = tuple(part.strip() for part in match.group(4).split(','))
        rows.append(CorpusRow(
            case=case,
            diagram=int(match.group(2)),
            topology=match.group(1),
            equation=match.group(3).strip(),
            tensors=tensors,
        ))
    return tuple(rows)


def declared_summary(path: str | os.PathLike[str] | None = None) -> dict[int, tuple[int, int]]:
    """Read the ``## 汇总`` table as ``{case: (diagrams, topology classes)}``."""
    summary: dict[int, tuple[int, int]] = {}
    inside = False
    for line in corpus_path(path).read_text(encoding='utf-8').splitlines():
        if line.startswith('## '):
            inside = '汇总' in line
            continue
        if not inside:
            continue
        match = _DECLARED.match(line)
        if match:
            summary[int(match.group(1))] = (int(match.group(2)), int(match.group(3)))
    return summary


def parse_equation(equation: str) -> tuple[tuple[str, ...], str]:
    """Split ``'a,b->c'`` into ``(('a', 'b'), 'c')``."""
    if '->' not in equation:
        raise ValueError(f'equation without an explicit output: {equation!r}')
    left, right = equation.split('->', 1)
    inputs = tuple(part.strip() for part in left.split(','))
    if not inputs or any(not part for part in inputs):
        raise ValueError(f'empty operand in equation: {equation!r}')
    return inputs, right.strip()


def canonical_signature(equation: str) -> tuple[tuple[str, ...], str]:
    """Relabel the equation by first appearance, keeping operand order.

    The result is a hashable topology signature: any consistent renaming of the
    labels (including non-contiguous ones) maps to the same signature, while two
    equations with different contraction structure map to different signatures.
    """
    inputs, output = parse_equation(equation)
    mapping: dict[str, str] = {}

    def normalise(part: str) -> str:
        out: list[str] = []
        for label in part:
            if label not in mapping:
                if len(mapping) >= len(LABELS):
                    raise ValueError(f'too many distinct labels in {equation!r}')
                mapping[label] = LABELS[len(mapping)]
            out.append(mapping[label])
        return ''.join(out)

    return tuple(normalise(part) for part in inputs), normalise(output)


def relabel_equation(equation: str, rng: np.random.Generator) -> str:
    """Apply one random consistent renaming, drawn from all 52 letters.

    The pool is shuffled, so renamed labels are typically non-contiguous.
    """
    inputs, output = parse_equation(equation)
    order: list[str] = []
    for part in (*inputs, output):
        for label in part:
            if label not in order:
                order.append(label)
    pool = list(LABELS)
    rng.shuffle(pool)
    mapping = dict(zip(order, pool))
    return ','.join(''.join(mapping[label] for label in part) for part in inputs) + \
        '->' + ''.join(mapping[label] for label in output)


def operand_count(equation: str) -> int:
    return len(parse_equation(equation)[0])


def output_rank(equation: str) -> int:
    return len(parse_equation(equation)[1])


def equation_seed(equation: str, base: int = 0) -> int:
    """Stable per-equation RNG seed, independent of process and ``--jobs``."""
    return (zlib.crc32(equation.encode('utf-8')) ^ int(base)) & 0xFFFFFFFF


def parse_sizes(spec: str | int | Sequence[int] | None = None) -> tuple[int, ...]:
    """Normalise a ``--size`` specification into a cycling tuple of dimensions.

    ``None``/``''`` yields ``(2, 3, 4)``; ``'3'`` yields ``(3,)``; ``'2,3'``
    and ``'2 3'`` yield ``(2, 3)``.
    """
    if spec is None or spec == '':
        return DEFAULT_SIZES
    if isinstance(spec, (int, np.integer)):
        values = [int(spec)]
    elif isinstance(spec, str):
        values = [int(part) for part in spec.replace(',', ' ').split()]
    else:
        values = [int(part) for part in spec]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f'invalid size specification: {spec!r}')
    return tuple(values)


def operand_shapes(equation: str, sizes: Sequence[int] = DEFAULT_SIZES) -> tuple[tuple[int, ...], ...]:
    """Shape of every operand, assigning one dimension per distinct label.

    Labels are numbered by first appearance (operands left to right, then the
    output) and take ``sizes[i % len(sizes)]``.
    """
    inputs, output = parse_equation(equation)
    order: list[str] = []
    for part in (*inputs, output):
        for label in part:
            if label not in order:
                order.append(label)
    dims = {label: sizes[index % len(sizes)] for index, label in enumerate(order)}
    return tuple(tuple(dims[label] for label in part) for part in inputs)


def make_operands(
    equation: str,
    rng: np.random.Generator,
    dtype: str = DEFAULT_DTYPE,
    sizes: Sequence[int] = DEFAULT_SIZES,
) -> list[np.ndarray]:
    """Deterministically draw the operands of ``equation`` under ``rng``."""
    kind = np.dtype(dtype)
    operands: list[np.ndarray] = []
    for shape in operand_shapes(equation, sizes):
        if kind.kind == 'c':
            values = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        else:
            values = rng.standard_normal(shape)
        operands.append(np.ascontiguousarray(values.astype(kind)))
    return operands


def corpus_statistics(rows: Sequence[CorpusRow]) -> dict[str, object]:
    """Totals, per-case counts, operand/rank histograms and signature uniqueness."""
    signatures: dict[tuple[tuple[str, ...], str], list[CorpusRow]] = {}
    for row in rows:
        signatures.setdefault(canonical_signature(row.equation), []).append(row)
    duplicates = {
        signature: tuple((row.case, row.topology) for row in group)
        for signature, group in signatures.items() if len(group) > 1
    }
    return {
        'total': len(rows),
        'per_case': dict(sorted(Counter(row.case for row in rows).items())),
        'operand_histogram': dict(sorted(Counter(operand_count(row.equation) for row in rows).items())),
        'output_rank_histogram': dict(sorted(Counter(output_rank(row.equation) for row in rows).items())),
        'unique_signatures': len(signatures),
        'duplicate_signatures': duplicates,
    }


def corpus_problems(
    rows: Sequence[CorpusRow],
    path: str | os.PathLike[str] | None = None,
) -> tuple[str, ...]:
    """Cross-check parsed rows against the document's own summary table."""
    problems: list[str] = []
    declared = declared_summary(path)
    actual = Counter(row.case for row in rows)
    for case, (diagrams, topologies) in sorted(declared.items()):
        parsed = actual.get(case, 0)
        if parsed != diagrams:
            problems.append(f'case {case}: summary declares {diagrams} rows, parsed {parsed}')
        if topologies != diagrams:
            problems.append(
                f'case {case}: summary declares {topologies} topology classes for {diagrams} rows')
    missing = sorted(set(actual) - set(declared))
    if missing:
        problems.append(f'parsed cases absent from the summary table: {missing}')
    for case in sorted(actual):
        diagrams = [row.diagram for row in rows if row.case == case]
        if diagrams != list(range(len(diagrams))):
            problems.append(f'case {case}: diagram numbers are not 0..{len(diagrams) - 1}')
    return tuple(problems)


def select_rows(
    rows: Sequence[CorpusRow],
    cases: Sequence[int] | None = None,
    limit: int | None = None,
) -> tuple[CorpusRow, ...]:
    """Filter by case first, then keep the first ``limit`` rows in corpus order."""
    selected = [row for row in rows if cases is None or row.case in set(cases)]
    if limit is not None:
        selected = selected[:max(0, int(limit))]
    return tuple(selected)


def format_statistics(stats: dict[str, object], path: str | os.PathLike[str] | None = None) -> str:
    """Render corpus statistics as a compact text block."""

    def histogram(values: dict[int, int]) -> str:
        return ', '.join(f'{key}:{count}' for key, count in values.items())

    lines = [
        f'corpus: {corpus_path(path)}',
        f'rows: {stats["total"]}  unique signatures: {stats["unique_signatures"]}  '
        f'duplicates: {len(stats["duplicate_signatures"])}',
        f'per case: {histogram(stats["per_case"])}',
        f'operand histogram: {histogram(stats["operand_histogram"])}',
        f'output rank histogram: {histogram(stats["output_rank_histogram"])}',
    ]
    return '\n'.join(lines)
