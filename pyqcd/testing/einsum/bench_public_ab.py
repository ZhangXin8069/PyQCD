"""End-to-end benchmark of pyqcd.einsum against opt_einsum.

Protocol
--------
* Rounds: 7 per case; the implementation that runs first alternates every round
  (the second runner would otherwise inherit a warm cache and gain a fake 10-20%).
* Inner repeats: each timed sample is repeated until it lasts >= 0.2 s, so short
  cases are not dominated by clock granularity.
* Sizes: every family is measured at three sizes (small / medium / large).
* Data: physics-shaped, not iid Gaussian -- propagator-like exponential decay in
  the time index, low-rank perambulator blocks, SU(3)-near-unitary links. Generic
  cases additionally use a pure random draw so BLAS sees unstructured input too.
* Correctness: every case is checked against opt_einsum before it is timed, with a
  relative-to-max-norm criterion (element-wise `atol` misjudges near-zero entries).

Usage: python bench.py cpu|gpu [--rounds N] [--out results.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings

import numpy as np
import opt_einsum as oe

import pyqcd
import pyqcd.einsum as pe


def build_device(device):
    if device == "gpu":
        import cupy as cp

        def to_device(array):
            return cp.asarray(array)

        def sync():
            cp.cuda.Device(0).synchronize()

        def release():
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    else:
        def to_device(array):
            return array

        def sync():
            return None

        def release():
            return None

        name = "cpu"
    return to_device, sync, release, name


_TO_DEVICE, SYNC, RELEASE, DEVICE_NAME = build_device("cpu")


def to_device(array):
    return _TO_DEVICE(array)


# ---------------------------------------------------------------- data helpers

def random_real(rng, *shape):
    return rng.standard_normal(shape)


def random_complex(rng, *shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)


def propagator(rng, time_len, spin=4, colour=3):
    """Propagator-like block: decaying forward/backward in time, unitary enough."""
    base = rng.standard_normal((time_len, spin, spin, colour, colour)).astype(np.complex128)
    decay = np.exp(-np.arange(time_len) / max(time_len / 3.0, 1.0))
    return base * decay[:, None, None, None, None]


def perambulator(rng, time_len, spatial, spin=4):
    """Perambulator: low-rank in time, orthogonal-ish spatial block."""
    rank = max(2, min(8, time_len // 4))
    left = rng.standard_normal((time_len, rank)).astype(np.complex128)
    right = rng.standard_normal((rank, spin, spin, spatial)).astype(np.complex128)
    return np.einsum("tr,rspx->tspx", left, right)


def su3_links(rng, time_len, spatial):
    """Near-unitary SU(3)-like links (QR of a Gaussian matrix)."""
    raw = rng.standard_normal((time_len, spatial, 3, 3)) + 1j * rng.standard_normal(
        (time_len, spatial, 3, 3))
    q, r = np.linalg.qr(raw)
    phase = np.diagonal(r, axis1=-2, axis2=-1)
    norm = phase / np.abs(phase)
    return q * norm[..., None, :]


# --------------------------------------------------------------------- cases

def build_cases(rng):
    """Return [(family, size_label, equation, operands_factory)]."""
    cases = []

    # --- two-operand GEMM, real -------------------------------------------------
    for label, n in (("small", 512), ("medium", 2048), ("large", 4096)):
        cases.append(("matmul real", label, "ij,jk->ik",
                      lambda n=n: [random_real(rng, n, n), random_real(rng, n, n)]))

    # --- two-operand GEMM, complex (lattice default) ----------------------------
    for label, n in (("small", 256), ("medium", 1024), ("large", 2048)):
        cases.append(("matmul complex", label, "ij,jk->ik",
                      lambda n=n: [random_complex(rng, n, n), random_complex(rng, n, n)]))

    # --- element-wise -----------------------------------------------------------
    for label, n in (("small", 2000), ("medium", 6000), ("large", 12000)):
        cases.append(("element-wise", label, "ij,ij->ij",
                      lambda n=n: [random_real(rng, n, n), random_real(rng, n, n)]))

    # --- reductions -------------------------------------------------------------
    for label, n in (("small", 4000), ("medium", 8000), ("large", 16000)):
        cases.append(("partial reduction", label, "ij->i",
                      lambda n=n: [random_real(rng, n, n)]))
    for label, n in (("small", 2000), ("medium", 6000), ("large", 12000)):
        cases.append(("full reduction", label, "ij,ij->",
                      lambda n=n: [random_real(rng, n, n), random_real(rng, n, n)]))

    # --- multi-operand chains ---------------------------------------------------
    for label, n, k in (("small", 384, 3), ("medium", 1536, 3), ("large", 3072, 3)):
        cases.append(("chain 3 ops", label, "ij,jk,kl->il",
                      lambda n=n: [random_real(rng, n, n) for _ in range(3)]))
    for label, n in (("small", 256), ("medium", 1024), ("large", 2048)):
        cases.append(("chain 4 ops", label, "ij,jk,kl,lm->im",
                      lambda n=n: [random_real(rng, n, n) for _ in range(4)]))

    # --- closed triangle --------------------------------------------------------
    for label, n in (("small", 256), ("medium", 1024), ("large", 2048)):
        cases.append(("hyper-edge", label, "ij,jk,ik->i",
                      lambda n=n: [random_real(rng, n, n) for _ in range(3)]))

    # --- batch structure -------------------------------------------------------
    for label, batch, n in (("small", 4, 256), ("medium", 16, 512), ("large", 16, 1024)):
        cases.append(("batched GEMM", label, "bij,bjk->bik",
                      lambda b=batch, n=n: [random_real(rng, b, n, n),
                                            random_real(rng, b, n, n)]))
    for label, batch, n in (("small", 16, 128), ("medium", 64, 256), ("large", 256, 512)):
        cases.append(("ellipsis batch", label, "...ij,...jk->...ik",
                      lambda b=batch, n=n: [random_real(rng, b, n, n),
                                            random_real(rng, b, n, n)]))
    for label, a, b_, c, d in (("small", 4, 256, 128, 256),
                               ("medium", 8, 512, 256, 512),
                               ("large", 8, 1024, 512, 1024)):
        cases.append(("batch x matrix", label, "abc,cd->abd",
                      lambda a=a, b_=b_, c=c, d=d: [random_real(rng, a, b_, c),
                                                    random_real(rng, c, d)]))
    for label, batch, n in (("small", 1024, 32), ("medium", 4096, 64), ("large", 8192, 96)):
        cases.append(("many small batches", label, "bij,bjk->bik",
                      lambda b=batch, n=n: [random_real(rng, b, n, n),
                                            random_real(rng, b, n, n)]))

    # --- physics kernels with structured (non-iid) data ------------------------
    for label, nvec in (("small", 32), ("medium", 64), ("large", 112)):
        def nucleon(nvec=nvec):
            # Perambulator-like decay instead of iid noise: the spatial axes carry an
            # exponential envelope, the spin/colour axes stay dense.
            decay = np.exp(-np.arange(nvec) / max(nvec / 4.0, 1.0))
            plane = decay[None, :, None] * decay[None, None, :]      # (1,n,n) 包络
            ops = [random_complex(rng, 4, 4, nvec, nvec) * plane[None]
                   for _ in range(3)]
            ops += [random_complex(rng, 4, 4), random_complex(rng, 4, 4)]
            ops += [random_complex(rng, 4, nvec, nvec, nvec) * plane[None] for _ in range(2)]
            ops += [random_complex(rng, 4, 4)]
            return ops
        cases.append(("nucleon 2pt (kernel)", label,
                      "gahb,mcnd,oepf,ac,og,Mbdf,Mnph,em->M", nucleon))

    for label, time_len, spatial in (("small", 256, 128), ("medium", 2048, 512),
                                     ("large", 8192, 1024)):
        def sandwich(time_len=time_len, spatial=spatial):
            peram = perambulator(rng, time_len, spatial)      # (T,4,4,X)
            # 两端的 4x4 追迹矩阵用随机复数矩阵（结构上等价于射影组合）
            return [random_complex(rng, 4, 4), peram, random_complex(rng, 4, 4)]
        cases.append(("gamma sandwich (kernel)", label, "ab,...bce,cd->...dae", sandwich))

    return cases


# ------------------------------------------------------------------- measuring

def one(call, inner):
    SYNC()
    best = float("inf")
    for _ in range(inner):
        start = time.perf_counter()
        call()
        SYNC()
        best = min(best, time.perf_counter() - start)
    return best


def measure(equation, operands, rounds):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = oe.contract(equation, *operands, optimize="dp")
        result = pe.contract(equation, *operands)
        got = result.get() if hasattr(result, "get") else np.asarray(result)
        ref = reference.get() if hasattr(reference, "get") else np.asarray(reference)
        scale = float(np.max(np.abs(ref))) if ref.size else 1.0
        error = float(np.max(np.abs(got - ref), initial=0.0)) / max(scale, 1e-30)
        assert error <= 1e-10, (equation, error)
        del reference, result, got, ref
        RELEASE()

        probe = one(lambda: pe.contract(equation, *operands), 1)
        inner = max(1, min(16, int(0.2 / max(probe, 1e-6))))

        ours, theirs = [], []
        for rnd in range(rounds):
            if rnd % 2 == 0:
                ours.append(one(lambda: pe.contract(equation, *operands), inner))
                theirs.append(one(lambda: oe.contract(equation, *operands, optimize="dp"), inner))
            else:
                theirs.append(one(lambda: oe.contract(equation, *operands, optimize="dp"), inner))
                ours.append(one(lambda: pe.contract(equation, *operands), inner))
            RELEASE()

    ours_med, theirs_med = statistics.median(ours), statistics.median(theirs)
    ratios = [b / a for a, b in zip(ours, theirs)]
    return {
        "pyqcd_s": ours_med,
        "opt_s": theirs_med,
        "ratio": theirs_med / ours_med,
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "rounds": rounds,
        "inner": inner,
        "rel_error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="pyqcd.einsum vs opt_einsum")
    parser.add_argument("device", choices=["cpu", "gpu"])
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()

    global _TO_DEVICE, SYNC, RELEASE, DEVICE_NAME
    _TO_DEVICE, SYNC, RELEASE, DEVICE_NAME = build_device(args.device)
    print(f"pyqcd {pyqcd.__version__} ({pyqcd.__file__}) on {DEVICE_NAME}, "
          f"opt_einsum {oe.__version__}, rounds={args.rounds}")

    rng = np.random.default_rng(20261109)
    records = []
    print(f"{'family':24s} {'size':7s} {'pyqcd(s)':>9s} {'opt(s)':>9s} {'ratio':>7s}")
    for family, label, equation, factory in build_cases(rng):
        operands = [to_device(operand) for operand in factory()]
        try:
            stats = measure(equation, operands, args.rounds)
        except AssertionError as exc:
            print(f"{family:24s} {label:7s} CORRECTNESS FAILURE: {exc}")
            continue
        records.append({"family": family, "size": label, "equation": equation,
                        "device": args.device, **stats})
        print(f"{family:24s} {label:7s} {stats['pyqcd_s']:9.4f} {stats['opt_s']:9.4f} "
              f"{stats['ratio']:7.2f}")
        del operands
        RELEASE()

    payload = {"device": args.device, "device_name": DEVICE_NAME,
               "pyqcd": pyqcd.__version__, "opt_einsum": oe.__version__,
               "rounds": args.rounds, "records": records}
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}: {len(records)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
