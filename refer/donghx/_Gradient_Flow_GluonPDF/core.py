"""Lattice-unit estimator, reproducible noise, and immutable small artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ENSEMBLE = "beta6.41_mu-0.2295_ms-0.2050_L32x96"
FLOW_SOURCE = Path("/public/home/donghx/Gradient_Flow/Blending_L32x96")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_npz(path, **payload):
    """Publish without replacing another run's artifact, even on a race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".kinetic-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        os.unlink(tmp)


def metadata_array(value):
    return np.array(json.dumps(value, sort_keys=True, allow_nan=False))


def gamma_dr():
    # Same DeGrand--Rossi matrices as flow_common_l32x96.gamma_matrices.
    g = np.zeros((4, 4, 4), complex)
    g[0, 0, 3] = g[0, 1, 2] = 1j
    g[0, 2, 1] = g[0, 3, 0] = -1j
    g[1, 0, 3] = g[1, 3, 0] = -1
    g[1, 1, 2] = g[1, 2, 1] = 1
    g[2, 0, 2] = g[2, 3, 1] = 1j
    g[2, 1, 3] = g[2, 2, 0] = -1j
    g[3, 0, 2] = g[3, 1, 3] = g[3, 2, 0] = g[3, 3, 1] = 1
    return g


def shift_fermion(field, mu, forward, xp=np):
    """field(t,z,y,x,spin,color); APBC only at the global temporal boundary."""
    axis = 3 - mu
    result = xp.roll(field, -1 if forward else 1, axis=axis)
    if mu == 3:
        edge = [slice(None)] * field.ndim
        edge[axis] = -1 if forward else 0
        result[tuple(edge)] *= -1
    return result


def hop(field, gauge, mu, forward, xp=np):
    """Periodic gauge(mu,t,z,y,x,color,color), with no embedded APBC phase."""
    shifted = shift_fermion(field, mu, forward, xp)
    link = gauge[mu]
    if not forward:
        link = xp.roll(link, 1, axis=3 - mu).swapaxes(-1, -2).conj()
    return xp.einsum("...ab,...sb->...sa", link, shifted)


def dslash_central(field, gauge, xp=np):
    if field.ndim != 6 or field.shape[-2:] != (4, 3):
        raise ValueError("Expected field(t,z,y,x,4,3)")
    if gauge.shape != (4, *field.shape[:4], 3, 3):
        raise ValueError("Gauge/fermion layout mismatch")
    out = xp.zeros_like(field)
    gamma = xp.asarray(gamma_dr())
    for mu in range(4):
        delta = (hop(field, gauge, mu, True, xp) - hop(field, gauge, mu, False, xp)) / 2
        out += xp.einsum("rs,...sc->...rc", gamma[mu], delta)
    return out


def laplace(field, gauge, xp=np):
    out = -8 * field.copy()
    for mu in range(4):
        out += hop(field, gauge, mu, True, xp) + hop(field, gauge, mu, False, xp)
    return out


def kinetic_sample(rho, phi, gauge, xp=np):
    """-2/V rho^dagger Dslash phi; no division by spin/color dimensions."""
    volume = int(np.prod(phi.shape[:4]))
    return -2.0 * xp.vdot(rho, dslash_central(phi, gauge, xp)) / volume


def z4_noise(shape, seed, conf, noise_id):
    # Full spin/color/spacetime support, E[eta eta^dagger]=I, not unit-norm noise.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(conf), int(noise_id)]))
    return np.asarray([1, 1j, -1, -1j], dtype=np.complex128)[rng.integers(0, 4, size=shape)]


def unit_gauge(shape):
    return np.broadcast_to(np.eye(3, dtype=complex), (4, *shape, 3, 3)).copy()


def free_wilson_kinetic(lattice_xyzt, mass, flow_times):
    """Exact finite-volume Wilson/free-flow trace; continuum Zchi need not be 1."""
    shape = tuple(reversed(lattice_xyzt))
    p = np.meshgrid(*[2 * np.pi * (np.arange(n) + (0.5 if i == 0 else 0)) / n
                     for i, n in enumerate(shape)], indexing="ij")
    s2 = sum(np.sin(k)**2 for k in p)
    w = mass + sum(1 - np.cos(k) for k in p)
    phat2 = 4 * sum(np.sin(k / 2)**2 for k in p)
    return np.array([-8 * 3 * np.mean(np.exp(-2 * t * phat2) * s2 / (w*w + s2))
                     for t in flow_times])


def free_noise_reference(eta, mass, flow_times):
    """Exact continuum-in-flow-time free lattice estimator for the SAME noise."""
    shape = eta.shape[:4]
    twist = np.exp(-1j * np.pi * np.arange(shape[0]) / shape[0]).reshape(-1, 1, 1, 1, 1, 1)
    ek = np.fft.fftn(eta * twist, axes=(0, 1, 2, 3), norm="ortho")
    p = np.meshgrid(*[2 * np.pi * (np.arange(n) + (0.5 if i == 0 else 0)) / n
                     for i, n in enumerate(shape)], indexing="ij")
    w = mass + sum(1 - np.cos(k) for k in p)
    den = w*w + sum(np.sin(k)**2 for k in p)
    gsin = sum(np.sin(p[3-mu])[..., None, None] * np.einsum("rs,...sc->...rc", g, ek)
               for mu, g in enumerate(gamma_dr()))
    qk = (w[..., None, None] * ek - 1j * gsin) / den[..., None, None]
    dqk = sum(1j * np.sin(p[3-mu])[..., None, None] * np.einsum("rs,...sc->...rc", g, qk)
                for mu, g in enumerate(gamma_dr()))
    phat2 = 4 * sum(np.sin(k / 2)**2 for k in p)
    return np.array([-2 * np.vdot(ek, np.exp(-2*t*phat2)[..., None, None] * dqk) / np.prod(shape)
                     for t in flow_times])


def validate_plan(plan):
    if plan.get("schema") != "quark_ringed_measurement_plan_v1":
        raise ValueError("Unsupported measurement plan")
    size = plan["lattice_xyzt"]
    if len(size) != 4 or any(type(n) is not int or n < 2 or n % 2 for n in size):
        raise ValueError("Use four even positive lattice extents")
    flow = np.asarray(plan["flow_times"], float)
    eps = float(plan["epsilon"])
    if flow.ndim != 1 or len(flow) == 0 or not np.isfinite(flow).all() or not np.isfinite(eps) or eps <= 0:
        raise ValueError("Invalid flow schedule")
    steps = np.rint(flow / eps).astype(int)
    if steps[0] < 1 or np.any(np.diff(steps) <= 0) or not np.allclose(flow / eps, steps, rtol=0, atol=1e-10):
        raise ValueError("Flow targets must be unique increasing integer steps")
    for name in ("mass", "csw", "stout_rho", "tol", "residual_max", "a_fm"):
        if not np.isfinite(plan[name]):
            raise ValueError(f"Nonfinite {name}")
    if plan["mass"] <= -4 or plan["tol"] <= 0 or plan["residual_max"] < plan["tol"] or plan["a_fm"] <= 0:
        raise ValueError("Invalid mass/tolerances/spacing")
    if plan["csw"] < 0 or plan["stout_rho"] < 0 or type(plan["stout_steps"]) is not int or plan["stout_steps"] < 0:
        raise ValueError("Invalid valence action")
    if plan["stout_dir_ignore"] != 4:
        raise ValueError("The matched valence action uses four-dimensional stout")
    if type(plan["maxiter"]) is not int or plan["maxiter"] <= 0:
        raise ValueError("Invalid maxiter")
    if plan["flow_gauge"] != "thin" or plan["fermion_boundary"] != "antiperiodic":
        raise ValueError("Only thin-link flow with APBC is implemented")
    if plan["mass_normalization"] != "QUDA_ASYMMETRIC_MASS_NORMALIZATION_full_MAT":
        raise ValueError("Unsupported propagator normalization")
    if not str(plan.get("action_provenance", "")).strip() or not str(plan.get("mass_treatment", "")).strip():
        raise ValueError("Missing action/mass provenance")
    if plan.get("free_field", False) and (plan["csw"] != 0 or plan["stout_steps"] != 0):
        raise ValueError("Free validation uses csw=0 and no stout")
    return steps
