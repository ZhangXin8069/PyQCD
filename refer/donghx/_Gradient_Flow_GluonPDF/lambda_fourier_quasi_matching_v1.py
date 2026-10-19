#!/usr/bin/env python3
"""Lambda-tail, Fourier and exploratory gluon quasi matching pipeline.

The input is the already GF-to-quasi converted, tau->0 extrapolation product
``unpolarized_tau0_selected.npz``.  Every operation is performed on the
existing 1000 bootstrap replicas; no second resampling is done.  The measured
points are kept unchanged and a *declared* damped-linear tail is appended in
lambda=z Pz only for the Fourier integral.  This is a finite-lambda model,
not a physical lambda->infinity extrapolation.  The quark-singlet mixing block
is intentionally omitted, as requested for this exploratory calculation.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from match_unpolarized_quasi_from_extrap_v1 import build_kernel_matrix

HERE = Path(__file__).resolve().parent
INPUT_DEFAULT = HERE / "unpolarized_relaxed_q1em6_chi10_minerr_v3" / "unpolarized_tau0_selected.npz"
OUT_DEFAULT = HERE / "unpolarized_relaxed_q1em6_chi10_minerr_v3" / "lambda_fourier_quasi_matching_v1"
CA = 3.0
HBARC = 0.1973269804


def fourier_matrix(z_a, pz_int, a_fm, xgrid):
    """Finite cosine transform, with the convention used in the note."""
    z_a = np.asarray(z_a, float)
    pz_gev = 2.0 * np.pi * float(pz_int) / (32.0 * float(a_fm) / HBARC)
    z_gev = z_a * float(a_fm) / HBARC
    # np.gradient supports the fine, uniformly spaced appended grid.
    w = np.gradient(z_gev)
    w[0] *= 0.5
    w[-1] *= 0.5
    lam = np.outer(xgrid, 2.0 * np.pi * float(pz_int) / 32.0 * z_a)
    return (pz_gev / np.pi) * np.cos(lam) * w[None, :]


def _damped_linear_tail(lam, values, lam_ext, tail_points=6, damping=0.15):
    """Fit the last points to A+B*lambda and damp the continuation.

    The measured values are not changed.  A negative or poorly determined
    slope is allowed; the positive ``damping`` factor prevents an unstable
    polynomial from growing beyond the measured lambda range.
    """
    n = len(lam)
    k = max(3, min(int(tail_points), n))
    xx = np.asarray(lam[-k:], float)
    yy = np.asarray(values[-k:], float)
    good = np.isfinite(xx) & np.isfinite(yy)
    if np.count_nonzero(good) < 3:
        return np.full_like(lam_ext, np.nan), np.array([np.nan, np.nan, np.nan]), "fit_failed"
    slope, intercept = np.polyfit(xx[good], yy[good], 1)
    d = np.asarray(lam_ext, float) - float(lam[-1])
    # C1 continuation at lambda_max; the exponential is a registered tail
    # regularizer, not an extra fit parameter.
    out = (float(values[-1]) + slope * d) * np.exp(-float(damping) * d)
    return out, np.array([intercept, slope, float(damping)]), "damped_linear"


def append_lambda_tail(z_a, h_boot, pz_int, a_fm, lambda_tail, dz_a,
                       tail_points=6, damping=0.15):
    """Append a common lambda tail separately for each bootstrap replica."""
    z_a = np.asarray(z_a, float)
    h_boot = np.asarray(h_boot, float)
    lam = 2.0 * np.pi * float(pz_int) / 32.0 * z_a
    lam_end = float(lam[-1] + lambda_tail)
    z_end = lam_end / (2.0 * np.pi * float(pz_int) / 32.0)
    z_grid = np.arange(z_a[0], z_end + 0.5 * dz_a, float(dz_a))
    # Ensure exact measured points and endpoint are present.
    z_grid = np.unique(np.r_[z_grid, z_a])
    z_grid.sort()
    h_ext = np.full((h_boot.shape[0], z_grid.size), np.nan)
    fit = np.full((h_boot.shape[0], 3), np.nan)
    status = np.empty(h_boot.shape[0], dtype="U24")
    measured = z_grid <= z_a[-1] + 1.e-10
    for b, vals in enumerate(h_boot):
        h_ext[b, measured] = np.interp(z_grid[measured], z_a, vals)
        tail_z = z_grid[~measured]
        tail_lam = 2.0 * np.pi * float(pz_int) / 32.0 * tail_z
        ext, coeff, tag = _damped_linear_tail(lam, vals, tail_lam, tail_points, damping)
        h_ext[b, ~measured] = ext
        fit[b] = coeff
        status[b] = tag
    return z_grid, h_ext, fit, status, measured, lam, lam_end


def mean_err(a, axis=0):
    return np.nanmean(a, axis=axis), np.nanstd(a, axis=axis, ddof=1)


def plot_tail(out, z, lam, h, zext, hext, pz, a_fm, fit):
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    m, e = mean_err(h, 0)
    me, ee = mean_err(hext, 0)
    ax.errorbar(lam, m, yerr=e, fmt="o", ms=3, ls="none", color="#2166ac", label="measured $z/a=0\ldots24$")
    tail = zext > z[-1] + 1.e-10
    lam_ext = 2*np.pi*pz/32*zext
    ax.errorbar(lam_ext[tail], me[tail], yerr=ee[tail], fmt="o", ms=2.3, ls="none", color="#b2182b", alpha=.75, label="damped-linear $\\lambda$ tail")
    ax.axvline(lam[-1], color=".35", lw=.8, ls="--")
    ax.set(xlabel=r"$\lambda=zP_z$", ylabel=r"$h(\lambda)$", title=rf"$n_z={pz}$, $a={a_fm:.4f}$ fm")
    ax.grid(alpha=.22); ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(out / f"lambda_tail_Pz{pz}.pdf"); fig.savefig(out / f"lambda_tail_Pz{pz}.png", dpi=180); plt.close(fig)


def plot_x(out, x, pabs, quasi, matched):
    fig, axs = plt.subplots(1, len(pabs), figsize=(5.0*len(pabs), 4.2), squeeze=False)
    for ip, p in enumerate(pabs):
        ax = axs[0, ip]
        q, qe = mean_err(quasi[ip], 0); m, me = mean_err(matched[ip], 0)
        good = np.abs(x) >= .02
        ax.errorbar(x[good], q[good], yerr=qe[good], fmt="o", ms=2.1, ls="none", alpha=.65, label="quasi")
        ax.errorbar(x[good], m[good], yerr=me[good], fmt="s", ms=2.0, ls="none", alpha=.72, label="matched gg")
        ax.axhline(0, color=".45", lw=.6); ax.grid(alpha=.2)
        ax.set(xlim=(-1, 1), xlabel="$x$", title=rf"$n_z={p}$")
        if ip == 0: ax.set_ylabel(r"$x\,g(x)$ (exploratory)")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Zero-flow gluon quasi matching: no singlet mixing")
    fig.tight_layout(); fig.savefig(out / "quasi_before_after_matching.pdf"); fig.savefig(out / "quasi_before_after_matching.png", dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=INPUT_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--operator", default="unpolarized_TU_minus_SU_even")
    ap.add_argument("--lambda-tail", type=float, default=8.0)
    ap.add_argument("--tail-points", type=int, default=6)
    ap.add_argument("--tail-damping", type=float, default=.15)
    ap.add_argument("--dz-a", type=float, default=.25)
    ap.add_argument("--x-min", type=float, default=-1.0); ap.add_argument("--x-max", type=float, default=1.0)
    ap.add_argument("--nx", type=int, default=401)
    ap.add_argument("--mu", type=float, default=2.0); ap.add_argument("--alpha-s", type=float, default=.25)
    ap.add_argument("--xi-max", type=float, default=30.0); ap.add_argument("--quadrature-order", type=int, default=48)
    ap.add_argument("--xkernel-min", type=float, default=.02)
    a = ap.parse_args()
    if a.lambda_tail <= 0 or a.dz_a <= 0 or a.tail_points < 3: ap.error("invalid tail parameters")
    a.out.mkdir(parents=True, exist_ok=True)
    with np.load(a.input, allow_pickle=False) as d:
        labels = np.asarray(d["operator_labels"]).astype(str) if "operator_labels" in d.files else None
        raw_selected = np.asarray(d["params_boot_selected"], float)
        if labels is not None and raw_selected.ndim == 5:
            op = int(np.flatnonzero(labels == a.operator)[0]); h = raw_selected[:, op, :, :, 0]
        else:
            # The selected product has no operator axis: (boot,p,z,intercept/slope).
            op = 0; h = raw_selected[..., 0]
        if h.ndim != 3: raise ValueError("expected params_boot_selected[...,0] with axes boot,p,z")
        z = np.asarray(d["z_values"], float); pabs = np.asarray(d["pabs_values"], int)
        if h.shape[1] != len(pabs): raise ValueError("pabs axis mismatch")
        # h was sliced above only for products carrying an operator axis.
        if h.shape[0] != int(d["nboot"]): raise ValueError("bootstrap count mismatch")
        a_fm = float(d["a_fm"]); nboot = h.shape[0]; nconf = int(d["nconf"])
    x = np.linspace(a.x_min, a.x_max, a.nx)
    quasi = np.full((len(pabs), nboot, len(x)), np.nan)
    matched = np.full_like(quasi, np.nan)
    zext_all = []; hext_all = []; fit_all = []; status_all = []; measured_all = []; tail_meta = []
    for ip, p in enumerate(pabs):
        zext, hext, fit, status, measured, lam, lam_end = append_lambda_tail(z, h[:, ip, :], int(p), a_fm, a.lambda_tail, a.dz_a, a.tail_points, a.tail_damping)
        F = fourier_matrix(zext, int(p), a_fm, x)
        q = hext @ F.T
        # The logarithm in C_ratio uses the physical parton momentum in GeV;
        # do not pass the integer lattice momentum here.
        pz_gev = 2.0 * np.pi * float(p) / (32.0 * a_fm / HBARC)
        B, mask, _ = build_kernel_matrix(x, pz_gev, a.mu, a.xkernel_min, a.xi_max, a.quadrature_order)
        pref = a.alpha_s * CA / (2*np.pi)
        m = q - pref * (B @ q.T).T
        m[:, ~mask] = np.nan
        quasi[ip] = q; matched[ip] = m
        zext_all.append(zext); hext_all.append(hext); fit_all.append(fit); status_all.append(status); measured_all.append(measured)
        tail_meta.append(dict(n_z=int(p), lambda_max_data=float(lam[-1]), lambda_max_extended=float(lam_end), zmax_extended=float(zext[-1]), tail_points=int(a.tail_points), damping=float(a.tail_damping), fit_success=int(np.sum(status == "damped_linear"))))
        plot_tail(a.out, z, lam, h[:, ip, :], zext, hext, int(p), a_fm, fit)
    plot_x(a.out, x, pabs, quasi, matched)
    qm, qe = mean_err(quasi, 1); mm, me = mean_err(matched, 1)
    meta = dict(schema="gluon_unpolarized_lambda_fourier_quasi_matching_v1", created_utc=datetime.now(timezone.utc).isoformat(), input=str(a.input.resolve()), output=str(a.out.resolve()), operator=a.operator, nboot=nboot, nconf=nconf, a_fm=a_fm, Lz=32, pabs_values=pabs.tolist(), pz_GeV=(2*np.pi*pabs/(32*a_fm/HBARC)).tolist(), z_data=z.tolist(), fourier="cosine, x*g_quasi=Pz/pi integral_0^infty dz cos(x Pz z) h(z,Pz)", lambda_definition="lambda=2*pi*n_z*(z/a)/32", tail_model="damped-linear continuation of last points; measured points unchanged", lambda_tail=float(a.lambda_tail), tail_points=int(a.tail_points), tail_damping=float(a.tail_damping), dz_a=float(a.dz_a), mu_GeV=float(a.mu), alpha_s=float(a.alpha_s), C_A=CA, matching_kernel="arXiv:2510.26425v2 Eq.(14), diagonal gg ratio kernel; Pz argument in GeV", quark_gluon_mixing_applied=False, gf_to_quasi_already_applied=True, second_bootstrap=False, status="exploratory diagnostic; no singlet mixing, continuum limit, target-mass or controlled finite-lambda systematic")
    def objseq(seq):
        out = np.empty(len(seq), dtype=object)
        out[:] = seq
        return out
    np.savez_compressed(a.out / "lambda_fourier_quasi_matching.npz", schema=meta["schema"], x=x, pabs_values=pabs, z_values=z, h_data_boot=h, z_extended=objseq(zext_all), h_extended_boot=objseq(hext_all), tail_fit_coeff=objseq(fit_all), tail_status=objseq(status_all), measured_mask=objseq(measured_all), quasi_boot=quasi, matched_boot=matched, quasi_mean=qm, quasi_error=qe, matched_mean=mm, matched_error=me, a_fm=a_fm, nboot=nboot, lambda_definition=meta["lambda_definition"], tail_model=meta["tail_model"], metadata_json=json.dumps(meta, ensure_ascii=False))
    (a.out / "manifest.json").write_text(json.dumps({**meta, "tail_momenta": tail_meta, "plots":[f"lambda_tail_Pz{int(p)}.pdf" for p in pabs] + ["quasi_before_after_matching.pdf"]}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({**meta, "tail_momenta": tail_meta}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
