#!/usr/bin/env python3
"""Exploratory NLO gluon quasi-to-light-cone matching for L32x96.

Input is the small-flow extrapolation product.  The GF-to-MSbar-quasi factor
has already been applied in that product and is deliberately *not* repeated.
This script applies the diagonal gg ratio-scheme LaMET kernel of
arXiv:2510.26425v2, with the quark-singlet mixing block set to zero as a
declared approximation.  The matching is performed replica by replica:

    x g_MS = x g_quasi - alpha_s C_A/(2 pi) K_ratio[x g_quasi]

The finite-z Fourier transform is a rectangular transform of the available
z/a=0..24 data.  This is therefore a controlled diagnostic, not a final PDF:
no lambda extrapolation, target-mass correction, continuum limit, or singlet
mixing is included.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import sici

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_DEFAULT = Path(__file__).resolve().parent / "bare_all_operators_matching_v1" / "flow_extrap_tau_ge1" / "matched_small_flow_extrap_tau_ge1.npz"
OUT_DEFAULT = Path(__file__).resolve().parent / "bare_all_operators_matching_v1" / "quasi_matching_tau_ge1"
CA = 3.0
HBARC = 0.1973269804


def gluon_ratio_kernel(xi, pz_parton, mu):
    """Real-emission part of Eq. (14) of arXiv:2510.26425v2."""
    xi = np.asarray(xi, dtype=float)
    pz = np.maximum(np.asarray(pz_parton, dtype=float), 1.0e-8)
    out = np.empty_like(xi)
    gt = xi > 1.0
    mid = (xi > 0.0) & (xi < 1.0)
    lt = xi < 0.0
    if np.any(gt):
        q = xi[gt]
        out[gt] = (2.0 * (1-q+q*q)**2/(1-q) * np.log(q/(q-1.0))
                   + (11-28*q+18*q*q-12*q**3)/(6*(1-q)))
    if np.any(mid):
        q = xi[mid]
        L = -np.log(mu*mu/(4*pz[mid]*pz[mid])) + np.log(q*(1-q))
        out[mid] = (2.0*(1-q+q*q)**2/(1-q)*L
                    - (15-56*q+102*q*q-96*q**3+48*q**4)/(6*(1-q)))
    if np.any(lt):
        q = xi[lt]
        out[lt] = (-2.0*(1-q+q*q)**2/(1-q)*np.log(q/(q-1.0))
                   - (11-28*q+18*q*q-12*q**3)/(6*(1-q)))
    if np.any(~(gt | mid | lt)):
        raise ValueError("kernel evaluated at xi=0 or xi=1")
    return out


def xi_quadrature(ximax=30.0, order=48):
    # Keep xi=0 and xi=1 out of the quadrature.  Splitting at 0,1 handles
    # the logarithmic endpoint structure of the published kernel.
    nodes, weights = [], []
    q, w = leggauss(order)
    bounds = [-ximax, -1.0, 0.0, 1.0, ximax]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        xx = 0.5*(hi-lo)*q + 0.5*(hi+lo)
        nodes.append(xx); weights.append(0.5*(hi-lo)*w)
    return np.concatenate(nodes), np.concatenate(weights)


def interp_row(grid, target):
    """Linear interpolation weights; out-of-grid targets are zero."""
    target = np.asarray(target, dtype=float)
    left = np.searchsorted(grid, target, side="right") - 1
    left = np.clip(left, 0, len(grid)-2)
    wr = (target-grid[left])/(grid[left+1]-grid[left])
    valid = (target >= grid[0]) & (target <= grid[-1])
    return left, 1.0-wr, wr, valid


def build_kernel_matrix(xgrid, pz, mu, xkernel_min=0.02, ximax=30.0, order=48):
    """Build B with (K f)(x)=B f(y), excluding alpha_s*CA/(2 pi)."""
    xi, wi = xi_quadrature(ximax, order)
    B = np.zeros((len(xgrid), len(xgrid)), dtype=float)
    computed = np.abs(xgrid) >= xkernel_min
    for ix, x in enumerate(xgrid):
        if not computed[ix]:
            continue
        y = x/xi
        li, wl, wr, valid = interp_row(xgrid, y)
        # Eq. (13) after xi substitution; this is the same plus-subtraction
        # convention used by Matching_norm.py.
        kval = np.zeros_like(xi)
        if np.any(valid):
            kval[valid] = gluon_ratio_kernel(xi[valid], np.abs(y[valid])*abs(pz), mu)
            coeff = wi[valid]*kval[valid]/np.abs(xi[valid])
            np.add.at(B[ix], li[valid], coeff*wl[valid])
            np.add.at(B[ix], li[valid]+1, coeff*wr[valid])
        fixed = gluon_ratio_kernel(xi, np.full_like(xi, abs(x)*abs(pz)), mu)
        sub = -float(np.sum(wi*fixed))
        li0, wl0, wr0, ok = interp_row(xgrid, np.array([x]))
        if not ok[0]:
            raise ValueError("x grid interpolation failure")
        B[ix, li0[0]] += sub*wl0[0]
        B[ix, li0[0]+1] += sub*wr0[0]
    reliable = (np.abs(xgrid) > 0.10) & (np.abs(xgrid) < 0.90)
    return B, computed, reliable


def fourier_matrix(z_a, pz_int, a_fm, xgrid, even=True):
    """Matrix taking h(z>=0) to x*g~(x), using the paper's convention."""
    zgev = np.asarray(z_a, float)*a_fm/HBARC
    nu = np.outer(xgrid, 2*np.pi*pz_int/32.0*np.asarray(z_a, float))
    # Pz/(2 pi) integral over [-zmax,zmax].  The even/odd extensions give
    # cosine/sine transforms; the helicity branch is not used in this script.
    dz = np.gradient(zgev)
    w = dz.copy(); w[0] *= .5; w[-1] *= .5
    pzgev = 2*np.pi*pz_int/(32.0*a_fm/HBARC)
    if even:
        return (pzgev/np.pi)*np.cos(nu)*w[None, :]
    return (pzgev/np.pi)*np.sin(nu)*w[None, :]


def finite_fourier(F, h):
    # F=(Nx,Nz), h=(Nboot,Nz) -> (Nboot,Nx), preserving NaN masks.
    good = np.isfinite(h)
    out = np.full((h.shape[0], F.shape[0]), np.nan)
    for b in range(h.shape[0]):
        if np.all(good[b]): out[b] = F @ h[b]
    return out


def mean_err(a):
    return np.nanmean(a, axis=0), np.nanstd(a, axis=0, ddof=1)


def plot_result(x, q, m, pz, label, out):
    qm, qe = mean_err(q); mm, me = mean_err(m)
    fig, ax = plt.subplots(figsize=(7.2,4.5))
    ax.axvspan(-.9,-.1,color='.8',alpha=.18); ax.axvspan(.1,.9,color='.8',alpha=.18)
    ax.errorbar(x, qm, yerr=qe, fmt='o', ms=2.4, linestyle='none', alpha=.65, label='quasi Fourier')
    ax.errorbar(x, mm, yerr=me, fmt='s', ms=2.2, linestyle='none', alpha=.75, label='NLO gg matched')
    ax.axvline(0,color='.4',lw=.6); ax.grid(alpha=.22)
    ax.set_xlim(-1,1); ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$xg(x)$ (diagnostic)')
    ax.set_title(f'{label}, $|P_z|={pz}$'); ax.legend(frameon=False,fontsize=8)
    fig.tight_layout(); fig.savefig(out,dpi=220); fig.savefig(out.with_suffix('.pdf')); plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=INPUT_DEFAULT)
    ap.add_argument('--out',type=Path,default=OUT_DEFAULT)
    ap.add_argument('--mu',type=float,default=2.0)
    ap.add_argument('--alpha-s',type=float,default=.25)
    ap.add_argument('--x-min',type=float,default=-1.0); ap.add_argument('--x-max',type=float,default=1.0)
    ap.add_argument('--nx',type=int,default=401); ap.add_argument('--xkernel-min',type=float,default=.02)
    ap.add_argument('--xi-max',type=float,default=30.0); ap.add_argument('--quadrature-order',type=int,default=48)
    ap.add_argument('--operator',default='unpolarized_TU_minus_SU_even')
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    with np.load(args.input,allow_pickle=False) as d:
        labels=np.asarray(d['operator_labels']).astype(str); op=int(np.flatnonzero(labels==args.operator)[0])
        z=np.asarray(d['z_values'],float); pabs=np.asarray(d['pabs_values'],int)
        boot=np.asarray(d['m0_boot'],float)[:,op] # boot,pz,z
        nboot=boot.shape[0]; a=float(d['a_fm'])
        x=np.linspace(args.x_min,args.x_max,args.nx)
        outboot=np.full((len(pabs),nboot,len(x)),np.nan); qboot=np.full_like(outboot,np.nan)
        stats=[]
        for ip,p in enumerate(pabs):
            h=boot[:,ip,:]
            F=fourier_matrix(z,p,a,x,even=True)
            q=finite_fourier(F,h)
            B,mask,reliable=build_kernel_matrix(x,p,args.mu,args.xkernel_min,args.xi_max,args.quadrature_order)
            pref=args.alpha_s*CA/(2*np.pi)
            # Strict one-loop inverse: use K on the uncorrected quasi sample.
            m=q.copy(); finite=np.isfinite(q).all(axis=1)
            if np.any(finite):
                m[finite] = q[finite] - pref*(B @ q[finite].T).T
            m[:,~mask]=np.nan
            qboot[ip]=q; outboot[ip]=m
            finite_frac=float(np.isfinite(m).mean())
            stats.append({'pz_integer':int(p),'pz_GeV':float(2*np.pi*p/(32*a/HBARC)),'finite_fraction':finite_frac,
                          'z_missing':z[~np.isfinite(h).all(axis=0)].astype(int).tolist()})
            plot_result(x,q,m,p,args.operator,args.out/f'{args.operator}_Pz{p}_quasi_matching.png')
        qm,qe=np.nanmean(qboot,axis=1),np.nanstd(qboot,axis=1,ddof=1)
        mm,me=np.nanmean(outboot,axis=1),np.nanstd(outboot,axis=1,ddof=1)
        np.savez_compressed(args.out/'unpolarized_quasi_matching.npz',schema='gluon_unpolarized_quasi_matching_ratio_nlo_v1',
          x=x,pabs_values=pabs,quasi_boot=qboot,matched_boot=outboot,quasi_mean=qm,quasi_error=qe,matched_mean=mm,matched_error=me,
          operator_label=args.operator,a_fm=a,mu_GeV=args.mu,alpha_s=args.alpha_s,C_A=CA,nboot=nboot,source=str(args.input),
          gf_to_quasi_already_applied=True,quark_gluon_mixing_applied=False,matching_kernel='arXiv:2510.26425v2 Eq.14 C_ratio',
          fourier='finite rectangular z/a=0..24; no lambda extrapolation',xkernel_min=args.xkernel_min,xi_max=args.xi_max,quadrature_order=args.quadrature_order)
        meta={'schema':'gluon_unpolarized_quasi_matching_ratio_nlo_v1','source':str(args.input),'operator':args.operator,
          'flow_extrapolation':'tau/a^2 >= 1.0 AIC-weighted linear M0','flow_to_quasi_already_applied':True,
          'kernel':'C_ratio of arXiv:2510.26425v2 Eq.(14)','alpha_s':args.alpha_s,'mu_GeV':args.mu,'C_A':CA,
          'mixing':'omitted by explicit user approximation','fourier':'rectangular finite-z transform; z/a=0..24',
          'nboot':nboot,'Ncfg':406,'p_z':stats,'warning':'diagnostic only; no lambda extrapolation, target-mass, continuum, or singlet mixing',
          'plots':[str(args.out/f'{args.operator}_Pz{p}_quasi_matching.pdf') for p in pabs]}
        (args.out/'manifest.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps(meta,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
