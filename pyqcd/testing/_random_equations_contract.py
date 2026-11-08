"""Randomised differential test against ``numpy.einsum``.

Hand-written cases cover the situations we thought of; these cover the ones we did
not. Every generated equation is first checked against NumPy, then — for the ones
that plan successfully — the returned path is replayed through ``contract`` so an
invalid path cannot hide behind a correct-looking result.

The seeds are fixed so a failure is reproducible; the case counts are small enough
to keep the suite fast.
"""

from __future__ import annotations

import random

import numpy as np

import pyqcd.einsum as pe

LABELS = list("abcdef")
OPTIMIZERS = ("dp", "greedy", "optimal", False, "auto")


def _random_equation(rng: random.Random, n_ops: int, ellipsis: bool):
    labels = rng.sample(LABELS, rng.randint(2, 5))
    terms = []
    for _ in range(n_ops):
        count = rng.randint(1, min(3, len(labels)))
        term = rng.sample(labels, count)
        if ellipsis and rng.random() < 0.6:
            term = ["..."] + term
        terms.append("".join(term))
    used = [label for label in dict.fromkeys(
        "".join(terms).replace(".", ""))]
    rng.shuffle(used)
    output = used[:rng.randint(0, len(used))]
    if ellipsis and any("..." in term for term in terms):
        output = ["..."] + output
    return ",".join(terms) + "->" + "".join(output)


def _operands(rng: np.random.Generator, randomizer: random.Random, terms, batch=2):
    dims = {label: randomizer.choice([1, 2, 3])
            for label in set("".join(terms).replace(".", ""))}
    operands = []
    for term in terms:
        shape = ([batch] if "..." in term else []) + [
            dims[label] for label in term if label != "."]
        real = rng.standard_normal(shape)
        imag = rng.standard_normal(shape)
        operands.append(real + 0.5j * imag)
    return operands


def _run_case(seed: int, n_ops: int, ellipsis: bool) -> None:
    randomizer = random.Random(seed)
    rng = np.random.default_rng(seed)
    equation = _random_equation(randomizer, n_ops, ellipsis)
    terms = equation.split("->")[0].split(",")
    operands = _operands(rng, randomizer, terms)
    reference = np.einsum(equation, *operands, optimize="optimal")

    optimizer = randomizer.choice(OPTIMIZERS)
    result = np.asarray(pe.contract(equation, *operands, optimize=optimizer))
    assert result.shape == reference.shape, (equation, result.shape, reference.shape)
    assert np.allclose(result, reference, rtol=1e-6, atol=1e-8), equation

    # the planned path must reproduce the same numbers when handed back in
    path, _info = pe.contract_path(equation, *operands, optimize="dp")
    replay = np.asarray(pe.contract(equation, *operands, optimize=path))
    assert np.allclose(replay, reference, rtol=1e-6, atol=1e-8), (equation, path)


def test_random_equations_match_numpy() -> None:
    for seed in range(48):
        _run_case(seed, n_ops=1 + seed % 4, ellipsis=False)


def test_random_ellipsis_equations_match_numpy() -> None:
    for seed in range(100, 128):
        _run_case(seed, n_ops=1 + seed % 3, ellipsis=True)


def test_repeated_labels_and_scalars() -> None:
    rng = np.random.default_rng(7)
    a = rng.standard_normal((5, 5))
    assert np.allclose(pe.contract("ii->", a), np.trace(a))
    assert np.allclose(pe.contract("ii->i", a), np.diag(a))
    b = rng.standard_normal((5, 5))
    assert np.allclose(pe.contract("ij,ji->", a, b), np.trace(a @ b))
    scalar = np.array(3.0)
    assert np.allclose(pe.contract(",ij->ij", scalar, a), 3.0 * a)
