"""Review-only flow candidates. NOT GPU validated or wired into production.

The caller must provide the frozen v2 gpu_ope module as `gpu` and compare all
five OPE products with v2 before adopting either candidate. Flow precision,
RK order, epsilon, raw source, and OPE algebra must stay unchanged.
"""
from time import perf_counter


def flow_independent_batched(raw, tau, epsilon, gpu):
    """Preserve independent raw starts; one QUDA API call for all RK steps.

    This avoids the high-level LatticeGauge.wilsonFlow Python loop and its
    unconditional final dirac.energy() call. That returned energy is not used
    by the current producer. Independent plaquette/SU(3) checks remain needed.
    """
    steps = round(tau / epsilon)
    assert steps > 0 and abs(steps * epsilon - tau) < 1e-12
    gauge = gpu.prepare(raw.copy())
    with gauge.use() as dirac:
        try:
            dirac.wilsonFlow(steps, epsilon, 0.0, False,
                             compute_plaquette=False, compute_qcharge=False)
            dirac.saveSmearedGauge(gauge)
        finally:
            dirac.freeSmearedGauge()
    gpu.cp.cuda.get_current_stream().synchronize()
    return gauge


def iter_flow_checkpoints(raw, taus, epsilon, gpu):
    """Single raw-start trajectory; yield copies at requested RK step counts.

    This changes the currently frozen independent-per-tau execution contract.
    For the v2 grid, exactly 120 rather than 470 RK steps are performed. Only
    the internal *unfixed* gauge is continued. This has no gauge-fixing or
    best-of operation. Numerical equivalence must be checked at every tau.

    Generator must be consumed completely or closed, so resident QUDA fields
    are freed even if an OPE measurement raises. No result/metadata publication
    is implemented in this draft.
    """
    taus = list(taus)
    assert taus == sorted(set(taus)) and taus and taus[0] > 0
    counts = [round(tau / epsilon) for tau in taus]
    assert all(abs(n * epsilon - tau) < 1e-12 for n, tau in zip(counts, taus))
    holder = gpu.prepare(raw.copy())
    with holder.use() as dirac:
        previous = 0
        try:
            for tau, target in zip(taus, counts):
                before = perf_counter()
                dirac.wilsonFlow(target - previous, epsilon, previous * epsilon,
                                 previous != 0, compute_plaquette=False,
                                 compute_qcharge=False)
                # saveSmearedGauge leaves the resident flow state available to
                # restart=True; the returned host object is an independent copy.
                dirac.saveSmearedGauge(holder)
                gpu.cp.cuda.get_current_stream().synchronize()
                seconds = perf_counter() - before
                previous = target
                yield tau, holder.copy(), seconds
        finally:
            dirac.freeSmearedGauge()
