#!/usr/bin/env python3
"""
Standalone version of calc_pion_3pt.py (pion only, no Arx7Tools import).

For every gauge config it computes the 3pt correlator

    C3(dz, t_sep, tau) = < (Og(dz, t, x, y) - <Og>) * (C2(tsink, tsrc) - <C2>) >

averaged over the spatial positions, tsrc, and gauge configs (one file per
config), and writes

    {BASE_DIR3}/{ens dir}/{momenta}/{tag_3pt_full}_{momenta}.conf{conf}.npy

Only the plus direction of the spatial separation is supported (dz_max >= 0),
and Psink == Psrc is assumed (zero momentum transfer).

Run:    python3 calc_pion_3pt_standalone.py
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
ENSEMBLE = "C32P29"
DIRECTIONS = ["x"]
OPERATOR_TYPE = "Ogmp"
DZ_MAX = None            # None -> use the ensemble value from ENSEMBLES
MOM_LIST = [4]
TAG_2PT = "Pi_2pt"
TAG_3PT = "Pi_3pt_aver_tsrc"
INTERPOLATOR_2PT = "G5Gt"
NUM_SMEAR_2PT = 1
PSMEAR = -3

OG_DIR = "/public/home/shenxm/data/Og"
BASE_DIR2 = "/public/home/shenxm/data/pion-2pt"
BASE_DIR3 = "/public/home/shenxm/data/pion-3pt"

ENSEMBLES = {
    "C32P29": dict(dir="beta6.20_mu-0.2770_ms-0.2400_L32x64", Nx=32, Nt=64,
                   conf_start=11000, conf_end=60300, conf_step=50,
                   conf_exception=[18350, 23050, 28400], dz_max=[17, 0],
                   Og_additional_factor=1),
    "F32P30": dict(dir="beta6.41_mu-0.2295_ms-0.2050_L32x96", Nx=32, Nt=96,
                   conf_start=1000, conf_end=41000, conf_step=50,
                   conf_exception=[10850, 11200, 11250, 11300, 11350, 11400, 11450,
                                   11500, 11550, 11600, 11650, 11700, 11750, 11800,
                                   11850, 11900, 11950, 12000, 12050, 12100, 12150,
                                   12200, 40300, 40350],
                   dz_max=[23, -17], Og_additional_factor=-1),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def conf_list(ens):
    """All gauge configs of the ensemble (conf_start..conf_end, step conf_step)."""
    return [c for c in range(ens["conf_start"], ens["conf_end"] + 1, ens["conf_step"])
            if c not in ens["conf_exception"]]


def resolve_dz_max(ens):
    """Number of dz values, as used in the Og filenames (`all{dz_max}`).

    The plus direction is used, and the 0-based maximum separation is increased
    by one to obtain the array length (e.g. dz_max=17 -> 18).
    """
    dz_max = ens["dz_max"] if DZ_MAX is None else DZ_MAX
    if isinstance(dz_max, list):
        dz_max = dz_max[0]
    return dz_max + 1


def momentum_list(p, direction):
    P = [0, 0, 0]
    P[{"x": 0, "y": 1, "z": 2}[direction]] = p
    return P


def momenta_str(P):
    return f"Psink{P[0]}.{P[1]}.{P[2]}.Psrc{P[0]}.{P[1]}.{P[2]}"


def smear_tag(P):
    """Smeared-momentum tag along the direction of P, e.g. P=[4,0,0] -> '1smear-3.0.0'."""
    Psmear = [0, 0, 0]
    Psmear[max(range(3), key=lambda i: abs(P[i]))] = PSMEAR
    return f"{NUM_SMEAR_2PT}smear{Psmear[0]}.{Psmear[1]}.{Psmear[2]}"


def load_Og(ens, direction, ndz, conf=None):
    """Og for one config, or (conf=None) the ensemble average. Shape (dz, t, x1, x2)."""
    if conf is None:
        suffix = f"average.{ens['conf_start']}.{ens['conf_end']}"
    else:
        suffix = f"conf{conf}"
    fname = os.path.join(OG_DIR, ens["dir"],
                         f"{OPERATOR_TYPE}_d{direction}all{ndz}_{suffix}.npy")
    return np.load(fname)


def load_corr_2pt(ens, P):
    """C2(tsink, tsrc) for all configs. Returns (dict conf -> C2, ensemble average)."""
    momenta = momenta_str(P)
    tag = f"{TAG_2PT}_{INTERPOLATOR_2PT}_{smear_tag(P)}"
    path_dir = os.path.join(BASE_DIR2, ens["dir"], momenta)

    corr = {}
    for conf in conf_list(ens):
        fname = os.path.join(path_dir, f"{tag}_{momenta}.conf{conf}.npy")
        if not os.path.isfile(fname):
            print(f"Warning: {fname} not found, skip conf {conf}")
            continue
        corr[conf] = np.load(fname)

    aver = np.zeros((ens["Nt"], ens["Nt"]), dtype=complex)
    for c in corr:
        aver += corr[c]
    aver /= len(corr)
    return corr, aver


def calc_corr_3pt_conf(corr_2pt, corr_2pt_aver, Og, Og_aver, Nt, Og_additional_factor):
    """
    C3 for one config. Indices of the inputs/outputs:
        corr_2pt, corr_2pt_aver: (tsink, tsrc)
        Og, Og_aver:             (dz, tcur, x1, x2)
        return:                  (dz, t_sep, tau)
    """
    fm1 = corr_2pt - corr_2pt_aver          # (tsink, tsrc)
    # Psink == Psrc, i.e. zero momentum transfer: sum over x1, x2
    fm2 = np.sum(Og - Og_aver, axis=(2, 3))  # (dz, tcur)

    # C3(dz, tsink - tsrc, toper - tsrc), averaging over tsrc
    corr = np.zeros((fm2.shape[0], Nt, Nt), dtype=complex)
    for tsrc in range(Nt):
        block = fm1[:, tsrc][None, :, None] * fm2[:, None, :]  # (dz, tsink, toper)
        corr += np.roll(block, (-tsrc, -tsrc), axis=(1, 2))
    corr /= Nt
    return corr * Og_additional_factor


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ens = ENSEMBLES[ENSEMBLE]
    ndz = resolve_dz_max(ens)
    Nt = ens["Nt"]
    assert abs(ndz) <= ens["Nx"]

    for p in MOM_LIST:
        for direction in DIRECTIONS:
            P = momentum_list(p, direction)
            momenta = momenta_str(P)
            print(f"Info: P={P}, direction={direction}, {momenta}")

            Og_aver = load_Og(ens, direction, ndz)
            corr_2pt, corr_2pt_aver = load_corr_2pt(ens, P)
            print(f"Info: {len(corr_2pt)} configurations with 2pt data")

            tag_3pt_full = (f"{TAG_3PT}_{INTERPOLATOR_2PT}_{OPERATOR_TYPE}_d{direction}{ndz}"
                            f"_{smear_tag(P)}")
            out_dir = os.path.join(BASE_DIR3, ens["dir"], momenta)
            os.makedirs(out_dir, exist_ok=True)

            start, end, step = ens["conf_start"], ens["conf_end"], ens["conf_step"]
            for conf in range(start, end + 1, step):
                if conf in ens["conf_exception"]:
                    continue
                if conf not in corr_2pt:
                    print(f"Warning: no 2pt data for conf {conf}, skip")
                    continue

                res = calc_corr_3pt_conf(corr_2pt[conf], corr_2pt_aver,
                                         load_Og(ens, direction, ndz, conf), Og_aver,
                                         Nt, ens["Og_additional_factor"])
                np.save(os.path.join(out_dir, f"{tag_3pt_full}_{momenta}.conf{conf}.npy"), res)

                if conf % (step * 20) == 0:
                    print(f"Info: {conf}/{end} completed")


if __name__ == "__main__":
    main()
