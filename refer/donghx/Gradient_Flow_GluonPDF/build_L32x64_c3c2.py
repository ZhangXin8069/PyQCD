#!/usr/bin/env python3
"""Build L32x64 flowed-gluon C3/C2 ratios and 500 bootstrap replicas.

The input products are the HYP-smeared OPE files from Ope_Gluon and the
positive-momentum two-point files in momsmear2x/y/z.  The script keeps the
three tensor combinations explicitly:

    titi          = ti + tj
    titi_minus_ij = ti + tj - 2 ij
    titi_plus_ij  = ti + tj + 2 ij

Here ``ij`` is one M_ij;ij component.  The factor two is part of the
Euclidean gluon-PDF definitions.  For each channel the disconnected C3 is a
configuration covariance, with source-time averaging performed after the
covariance.  The helicity numerator uses the full complex polarized C2; both
channels use the unpolarized (nopol) C2 in the ratio denominator.

The saved central products contain plus_z, minus_z, even_sum and
odd_difference.  Bootstrap replicas are saved for the physical projection
(even_sum for unpolarized and odd_difference for helicity), together with
their mean and standard deviations and the common index array.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np


AXES = ("x", "y", "z")
PABS_DEFAULT = (3, 4, 5)
TSEPS_DEFAULT = tuple(range(5, 11))
NTAU = 11
DZ = 15
NT = 64
MOM_PHASE = 2
FORMULAS = ("titi", "titi_minus_ijij", "titi_plus_ijij")
ORIENTATIONS = ("plus_z", "minus_z", "even_sum", "odd_difference")
CHANNELS = ("unpolarized", "helicity")

# (time, transverse i, transverse j) in the saved OPE convention.
COMPONENTS = {
    "x": ((3, 1), (3, 2), (1, 2)),
    "y": ((3, 2), (3, 0), (2, 0)),
    "z": ((3, 0), (3, 1), (0, 1)),
}
POL = {"x": "pol15", "y": "pol25", "z": "pol35"}


def sorted_numeric(values):
    return sorted(values, key=lambda s: int(s))


def confs_in(root: Path, axis: str):
    return {p.name for p in (root / f"{axis}dir").iterdir()
            if p.is_dir() and p.name.isdigit()}


def ope_path(root: Path, conf: str, axis: str, pair, dz: int,
             dual: bool, minus: bool):
    mu, nu = pair
    stem = "ops_minus_" if minus else "ops_"
    suffix = "_AS" if dual else ""
    return (root / f"{axis}dir" / conf /
            f"{stem}mu{mu}_nu{nu}_dz{dz}_conf{conf}{suffix}.npy")


def mom_components(axis: str, p: int):
    return {"x": (p, 0, 0), "y": (0, p, 0), "z": (0, 0, p)}[axis]


def twopt_path(root: Path, conf: str, axis: str, p: int, pol: str):
    px, py, pz = mom_components(axis, p)
    return (root / f"momsmear2{axis}" / conf /
            f"twopt_slice_pp_Px{px}Py{py}Pz{pz}_eginphase{MOM_PHASE}"
            f"_Cg5g4_{pol}_ss_conf{conf}.npy")


def exact_common_confs(ope_root: Path, twopt_root: Path, pabs, dz):
    sets = [confs_in(ope_root, a) for a in AXES]
    for axis in AXES:
        for pair in COMPONENTS[axis]:
            for dual in (False, True):
                for minus in ((False, True) if dual else (False,)):
                    expected_prefix = ("ops_minus_" if minus else "ops_")
                    expected_suffix = "_AS" if dual else ""
                    sets.append({p.parent.name for p in
                                 (ope_root / f"{axis}dir").glob(
                                     f"*/{expected_prefix}mu{pair[0]}_nu{pair[1]}"
                                     f"_dz{dz}_conf*.npy")
                                 if p.parent.name.isdigit() and p.name ==
                                 f"{expected_prefix}mu{pair[0]}_nu{pair[1]}_dz{dz}"
                                 f"_conf{p.parent.name}{expected_suffix}.npy"})
        for p in pabs:
            for pol in ("nopol", POL[axis]):
                sets.append({q.parent.name for q in
                             (twopt_root / f"momsmear2{axis}").glob(
                                 f"*/twopt_slice_pp_*_eginphase{MOM_PHASE}"
                                 f"_Cg5g4_{pol}_ss_conf*.npy")
                             if q.parent.name.isdigit() and
                             twopt_path(twopt_root, q.parent.name, axis, p,
                                        pol).exists()})
    common = set.intersection(*sets)
    return sorted_numeric(common)


def validate_inputs(confs, ope_root, twopt_root, pabs, tseps, dz):
    if not confs:
        raise RuntimeError("no complete configurations")
    for conf in confs:
        for axis in AXES:
            for pair in COMPONENTS[axis]:
                for dual in (False, True):
                    for minus in ((False, True) if dual else (False,)):
                        path = ope_path(ope_root, conf, axis, pair, dz,
                                        dual, minus)
                        if not path.exists():
                            raise FileNotFoundError(path)
            for p in pabs:
                for pol in ("nopol", POL[axis]):
                    path = twopt_path(twopt_root, conf, axis, p, pol)
                    if not path.exists():
                        raise FileNotFoundError(path)
    if min(tseps) < 0 or max(tseps) >= NT:
        raise ValueError(f"tseps must lie in [0,{NT})")
    if max(tseps) + 1 > NTAU:
        raise ValueError("the configured tau axis is too short")


def load_data(confs, ope_root, twopt_root, pabs, tseps, dz):
    nc, nd, nz, ntsep, npabs = len(confs), len(AXES), dz, len(tseps), len(pabs)
    # Channel, direction, orientation, component, conf, time, z.
    ope = np.empty((2, nd, 2, 3, nc, NT, nz), dtype=np.complex128)
    # Channel numerator and common nopol denominator:
    # channel, direction, conf, tsep, source-time, p.
    c2_num = np.empty((2, nd, nc, ntsep, NT, npabs), dtype=np.complex128)
    c2_den = np.empty((nd, nc, ntsep, NT, npabs), dtype=np.complex128)
    for ic, conf in enumerate(confs):
        if (ic + 1) % 25 == 0 or ic == 0:
            print(f"loading {ic + 1}/{nc}: {conf}", flush=True)
        for idir, axis in enumerate(AXES):
            pairs = COMPONENTS[axis]
            for ich, dual in enumerate((False, True)):
                for iori, minus in enumerate((False, True)):
                    if not dual and minus:
                        ope[ich, idir, iori, :, ic] = np.nan + 1j * np.nan
                        continue
                    vals = [np.load(ope_path(ope_root, conf, axis, pair,
                                              dz, dual, minus),
                                    allow_pickle=False)
                            for pair in pairs]
                    for val in vals:
                        if val.shape != (NT, dz):
                            raise ValueError(
                                f"unexpected OPE shape {val.shape} in {conf}")
                    ope[ich, idir, iori, :, ic] = vals
            for ip, p in enumerate(pabs):
                nopol = np.asarray(np.load(twopt_path(twopt_root, conf, axis,
                                                      p, "nopol"),
                                             allow_pickle=False),
                                   dtype=np.complex128)
                pol = np.asarray(np.load(twopt_path(twopt_root, conf, axis,
                                                    p, POL[axis]),
                                          allow_pickle=False),
                                 dtype=np.complex128)
                if nopol.shape != (NT, NT) or pol.shape != (NT, NT):
                    raise ValueError(f"unexpected C2 shape for {conf} {axis} {p}")
                for it, tsep in enumerate(tseps):
                    ts = np.arange(NT)
                    c2_den[idir, ic, it, :, ip] = nopol[(ts + tsep) % NT, ts]
                    c2_num[0, idir, ic, it, :, ip] = c2_den[idir, ic, it, :, ip]
                    c2_num[1, idir, ic, it, :, ip] = pol[(ts + tsep) % NT, ts]
    return ope, c2_num, c2_den


def source_summaries(ope, c2_num, c2_den, tseps):
    """Return per-configuration O, C and OC source averages.

    O has axes channel,direction,orientation,component,conf,tau,z;
    C has channel,direction,conf,tsep,p; OC has the same leading axes as C
    plus tau,z.  Source-time averaging occurs here, before bootstrap means.
    """
    nc = ope.shape[4]
    ntau = max(tseps) + 1
    # Keep source time explicit: the vacuum subtraction is formed at each
    # source before averaging over source times.
    osrc = np.empty((2, 3, 2, 3, nc, ntau, NT, DZ), dtype=np.complex128)
    source = np.arange(NT)
    for ich in range(2):
        for idir in range(3):
            for iori in range(2):
                raw = ope[ich, idir, iori]  # component,conf,time,z
                for itau in range(ntau):
                    osrc[ich, idir, iori, :, :, itau] = raw[
                        :, :, (source + itau) % NT, :
                    ]
    return osrc, c2_num, c2_den


def bootstrap_indices(nboot, nconf, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, nconf, size=(nboot, nconf), dtype=np.int32)


def weighted_bootstrap(arr, idx):
    # arr axis 0 is configurations; this avoids materializing idx-expanded data.
    return np.mean(arr[idx], axis=1)


def build_channel(ich, osrc, csrc, dsrc, idx, tseps):
    nb, nconf = idx.shape
    ntau, ntsep, npabs = max(tseps) + 1, len(tseps), csrc.shape[-1]
    # Include b=0 as the ordinary ensemble mean; b=1..nb are bootstrap rows.
    weights = np.zeros((nb + 1, nconf), dtype=np.float64)
    weights[0] = 1.0 / nconf
    for ib, row in enumerate(idx, 1):
        weights[ib] = np.bincount(row, minlength=nconf) / nconf
    ratio_all = np.full(
        (nb + 1, 3, 3, 4, npabs, DZ, ntsep, ntau),
        np.nan + 1j * np.nan, dtype=np.complex128,
    )
    for idir in range(3):
        # Both channels use nopol in the final ratio denominator.
        den_src = np.einsum("bc,ctsp->btsp", weights, dsrc[idir], optimize=True)
        den = den_src.mean(axis=2)  # b,tsep,p
        c = csrc[ich, idir]  # conf,tsep,source,p
        for iori in range(2):
            if ich == 0 and iori == 1:
                continue  # ordinary HYP OPE has only the + orientation
            o = osrc[ich, idir, iori]  # component,conf,tau,source,z
            # Per-source covariance.  The subtraction is recomputed for every
            # bootstrap row, then averaged over source time.
            own = np.einsum(
                "bc,kcasz,cqsp->bkaqszp", weights, o, c, optimize=True
            )
            om = np.einsum("bc,kcasz->bkasz", weights, o, optimize=True)
            cm = np.einsum("bc,cqsp->bqsp", weights, c, optimize=True)
            disconnected = np.einsum(
                "bkasz,bqsp->bkaqszp", om, cm, optimize=True
            )
            cov = (own - disconnected).mean(axis=4)  # b,k,tau,tsep,z,p
            cov = np.transpose(cov, (0, 1, 5, 4, 3, 2))
            # b,k,p,z,tsep,tau; denominator is b,p,tsep.
            rb = cov / den.transpose(0, 2, 1)[:, None, :, None, :, None]
            ratio_all[:, :, idir, iori] = np.stack(
                (rb[:, 0] + rb[:, 1],
                 rb[:, 0] + rb[:, 1] - 2 * rb[:, 2],
                 rb[:, 0] + rb[:, 1] + 2 * rb[:, 2]), axis=1,
            )
        if ich == 1:
            ratio_all[:, :, idir, 2] = (
                ratio_all[:, :, idir, 0] + ratio_all[:, :, idir, 1]
            )
            ratio_all[:, :, idir, 3] = (
                ratio_all[:, :, idir, 0] - ratio_all[:, :, idir, 1]
            )
        # Common tau storage is padded to max(tsep)+1.
        for it, tsep in enumerate(tseps):
            ratio_all[:, :, idir, :, :, :, it, tsep + 1:] = np.nan + 1j * np.nan
    return ratio_all[0], ratio_all[1:]


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ope-root", type=Path, required=True)
    ap.add_argument("--twopt-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--nconf", type=int, default=200)
    ap.add_argument("--nboot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1115)
    ap.add_argument("--channel", choices=("unpolarized", "helicity", "both"),
                    default="both",
                    help="run one channel for independent CPU jobs, or both")
    ap.add_argument("--pabs", default="3,4,5")
    ap.add_argument("--tseps", default="5,6,7,8,9,10")
    ap.add_argument("--dz", type=int, default=DZ)
    ap.add_argument(
        "--receipt-name",
        default="",
        help=(
            "Optional completion JSON filename.  A versioned name is useful "
            "when several momentum selections share one output directory."
        ),
    )
    ap.add_argument("--all-common", action="store_true",
                    help="use all complete configurations instead of first nconf")
    args = ap.parse_args()
    pabs = tuple(int(x) for x in args.pabs.split(",") if x.strip())
    tseps = tuple(int(x) for x in args.tseps.split(",") if x.strip())
    if args.dz != DZ:
        raise ValueError("this production layout is validated for dz=15")
    common = exact_common_confs(args.ope_root, args.twopt_root, pabs, args.dz)
    if not args.all_common:
        common = common[:args.nconf]
    elif args.nconf and len(common) > args.nconf:
        common = common[:args.nconf]
    validate_inputs(common, args.ope_root, args.twopt_root, pabs, tseps, args.dz)
    print(f"using {len(common)} configurations: {common[0]} ... {common[-1]}")
    ope, c2_num, c2_den = load_data(common, args.ope_root, args.twopt_root,
                                     pabs, tseps, args.dz)
    osrc, csrc, dsrc = source_summaries(ope, c2_num, c2_den, tseps)
    idx = bootstrap_indices(args.nboot, len(common), args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    selected_channels = CHANNELS if args.channel == "both" else (args.channel,)
    for channel in selected_channels:
        ich = CHANNELS.index(channel)
        central, boot = build_channel(ich, osrc, csrc, dsrc, idx, tseps)
        physical_orientation = 0 if channel == "unpolarized" else 3
        # Keep all central orientations.  Save physical bootstrap replicas and
        # mean/error; raw plus/minus bootstrap values are reconstructible from
        # rerunning with the same manifest and are intentionally not duplicated.
        phys = boot[:, :, :, physical_orientation]
        valid_tau_mask = np.zeros((len(tseps), max(tseps) + 1), dtype=bool)
        for it, tsep in enumerate(tseps):
            valid_tau_mask[it, :tsep + 1] = True
        path = args.output_dir / (
            f"c3c2_L32x64_{channel}_Nconf{len(common)}_Nboot{args.nboot}"
            f"_seed{args.seed}_P{'-'.join(map(str,pabs))}_tsep"
            f"{'-'.join(map(str,tseps))}.npz")
        np.savez_compressed(
            path,
            ratio_central=central,
            ratio_bootstrap=phys,
            ratio_bootstrap_mean=np.mean(phys, axis=0),
            ratio_bootstrap_error=np.std(phys, axis=0, ddof=1),
            bootstrap_indices=idx,
            confs=np.asarray(common),
            channel=np.asarray(channel),
            formula_labels=np.asarray(FORMULAS),
            direction_labels=np.asarray(AXES),
            orientation_labels=np.asarray(ORIENTATIONS),
            physical_orientation=np.asarray(ORIENTATIONS[physical_orientation]),
            pabs=np.asarray(pabs, dtype=np.int32),
            z=np.arange(args.dz, dtype=np.int32),
            tseps=np.asarray(tseps, dtype=np.int32),
            tau=np.arange(max(tseps) + 1, dtype=np.int32),
            valid_tau_mask=valid_tau_mask,
            axes_central=np.asarray(
                ["formula", "direction", "orientation", "pabs", "z", "tsep", "tau"]),
            axes_bootstrap=np.asarray(
                ["boot", "formula", "direction", "pabs", "z", "tsep", "tau"]),
            operator_definition=np.asarray(
                "titi=ti+tj; titi_minus_ijij=ti+tj-2*ij; "
                "titi_plus_ijij=ti+tj+2*ij"),
            c3_definition=np.asarray(
                "mean_tsrc(<O*C2>_cfg-<O>_cfg<C2>_cfg), then ratio"),
            denominator_definition=np.asarray(
                "nopol for both unpolarized and helicity"),
            helicity_numerator_definition=np.asarray(
                "full complex pol15/pol25/pol35 two-point function"),
            input_scheme=np.asarray("HYP 4D 10-times OPE"),
            nconf=np.asarray(len(common), dtype=np.int32),
            nboot=np.asarray(args.nboot, dtype=np.int32),
            bootstrap_seed=np.asarray(args.seed, dtype=np.int64),
        )
        files.append({"path": str(path), "sha256": sha256(path),
                      "ratio_bootstrap_shape": list(phys.shape),
                      "ratio_central_shape": list(central.shape)})
        print(f"saved {path}")
    receipt_name = args.receipt_name or (
        "completion.json" if args.channel == "both" else
        f"completion_{args.channel}.json"
    )
    receipt = args.output_dir / receipt_name
    receipt.write_text(json.dumps({
        "ensemble": "L32x64", "channel": args.channel,
        "nconf": len(common), "nboot": args.nboot,
        "seed": args.seed, "pabs": pabs, "tseps": tseps, "dz": args.dz,
        "confs": common, "files": files,
        "estimator": "configuration_covariance_source_average_ratio_v1",
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"saved {receipt}")


if __name__ == "__main__":
    main()
