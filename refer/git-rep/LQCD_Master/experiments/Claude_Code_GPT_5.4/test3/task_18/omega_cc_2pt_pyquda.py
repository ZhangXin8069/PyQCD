from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

try:
    from pyquda_utils import core, io, gamma
except Exception as exc:
    print(f"ERROR: failed to import PyQUDA helpers: {exc}", file=sys.stderr)
    raise SystemExit(1)

ENSEMBLE_NAME = "C24P29"
BETA = 6.20
CFG_NUMBER = 10000
GAUGE_PATH = "/public/share/weiwang/clqcd/beta6.20_mu-0.2770_ms-0.2400_L24x72/Configurations/Original/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_10000.lime"
LATTICE = [24, 24, 24, 72]
T_BOUNDARY = -1
ANISOTROPY = 1.0
XI_0 = 1.0
LIGHT_QUARK_MASS = -0.2770
STRANGE_QUARK_MASS = -0.2356
CHARM_QUARK_MASS = 0.4159
CLOVER_COEFF = 1.160920226
MULTIGRID = [[6, 6, 6, 3], [4, 4, 4, 6]]
PROCESS_GRID = [1, 1, 1, 4]
MPI_NUM = 4
SOURCE_POSITION = [0, 0, 0, 0]
TOL = 1.0e-12
MAXITER = 1000
MRHS = 4
OUTPUT_NAME = "omega_cc_2pt_C24P29_cfg_10000.txt"


def ensure_inverter_converged(dirac, label):
    inv_param = getattr(dirac, "invert_param", None)
    if inv_param is None:
        inv_param = getattr(dirac, "inv_param", None)
    iterations = None if inv_param is None else getattr(inv_param, "iter", None)
    true_res = None if inv_param is None else getattr(inv_param, "true_res", None)
    if true_res is None and inv_param is not None:
        true_res = getattr(inv_param, "true_res_hq", None)
    if iterations is not None and iterations >= MAXITER:
        raise RuntimeError(
            f"Inverter reached maxiter without clear convergence for {label}: iter={iterations}, maxiter={MAXITER}"
        )
    if true_res is not None and (not np.isfinite(true_res) or true_res > 1.0e-8):
        raise RuntimeError(f"Inverter residual is too large for {label}: true_res={true_res}")


def build_dirac(latt_info, mass):
    return core.getDirac(
        latt_info=latt_info,
        mass=mass,
        tol=TOL,
        maxiter=MAXITER,
        xi_0=XI_0,
        clover_coeff_t=CLOVER_COEFF,
        clover_coeff_r=CLOVER_COEFF,
        multigrid=MULTIGRID,
    )


def invert_flavor(gauge, latt_info, mass, label):
    dirac = build_dirac(latt_info, mass)
    dirac.loadGauge(gauge)
    propagator = core.invert(dirac, source_type="point", t_srce=SOURCE_POSITION, mrhs=MRHS)
    ensure_inverter_converged(dirac, label)
    data = propagator.lexico()
    if not np.all(np.isfinite(data)):
        raise RuntimeError(f"Propagator contains non-finite values for {label}")
    return data, dirac


def gamma5_transform(prop, g5):
    return np.einsum("ab,tzyxbcij,cd->tzyxadij", g5, np.conjugate(prop), g5, optimize=True)


def connected_meson_local(prop_a, prop_b):
    return np.einsum("tzyxabij,tzyxabij->tzyx", np.conjugate(prop_a), prop_b, optimize=True)


def vector_meson_local(prop_a, prop_b, gamma_mu, g5):
    backward = gamma5_transform(prop_a, g5)
    tmp = np.einsum("ab,tzyxbcij->tzyxacij", gamma_mu, prop_b, optimize=True)
    tmp = np.einsum("tzyxabij,bc->tzyxacij", tmp, gamma_mu, optimize=True)
    return np.einsum("tzyxabij,tzyxbaji->tzyx", backward, tmp, optimize=True)


def gather_timeseries(local_field):
    if not np.all(np.isfinite(local_field)):
        raise RuntimeError("Correlator field contains non-finite values")
    gathered = core.gatherLattice2(local_field, [0, 1, 2, 3], reduce_op="sum", root=0)
    if core.getMPIRank() != 0:
        return None
    if gathered is None:
        raise RuntimeError("Failed to gather correlator on rank 0")
    correlator = gathered.sum(axis=(1, 2, 3))
    if correlator.shape[0] != LATTICE[3]:
        raise RuntimeError(
            f"Unexpected correlator length: got {correlator.shape[0]}, expected {LATTICE[3]}"
        )
    if not np.all(np.isfinite(correlator)):
        raise RuntimeError("Final correlator contains non-finite values")
    return correlator


def make_epsilon():
    epsilon = np.zeros((3, 3, 3), dtype=np.complex128)
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
    epsilon[2, 1, 0] = epsilon[1, 0, 2] = epsilon[0, 2, 1] = -1.0
    return epsilon


def baryon_cg5_local(prop_a, prop_b, prop_c, epsilon, Cg5, P_plus):
    term1 = np.einsum(
        "ijk,lmn,ab,cd,ef,gc,tzyxaeil,tzyxbfjm,tzyxdgkn->tzyx",
        epsilon,
        epsilon,
        Cg5,
        P_plus,
        Cg5,
        P_plus,
        prop_a,
        prop_b,
        prop_c,
        optimize=True,
    )
    return term1


def baryon_cg5_exchange_local(prop_a, prop_b, prop_c, epsilon, Cg5, P_plus):
    return np.einsum(
        "ijk,lmn,ab,cd,ef,gc,tzyxagin,tzyxbfjm,tzyxdekl->tzyx",
        epsilon,
        epsilon,
        Cg5,
        P_plus,
        Cg5,
        P_plus,
        prop_a,
        prop_b,
        prop_c,
        optimize=True,
    )


def baryon_cg1_local(prop_a, prop_b, prop_c, epsilon, Cg1, P_plus, G5):
    total = np.zeros(prop_a.shape[:4], dtype=np.complex128)
    for gamma_mu in (Gx, Gy, Gz, Gt):
        for gamma_nu in (Gx, Gy, Gz, Gt):
            total += np.einsum(
                "ijk,lmn,ab,cd,ef,gc,tzyxaeil,tzyxbfjm,tzyxdgkn->tzyx",
                epsilon,
                epsilon,
                Cg1,
                P_plus @ gamma_mu @ G5,
                gamma_nu @ Cmat,
                gamma_nu @ G5 @ P_plus,
                prop_a,
                prop_b,
                prop_c,
                optimize=True,
            )
    return total


if __name__ == "__main__":
    diracs = []
    try:
        script_dir = Path(__file__).resolve().parent
        run_dir = script_dir / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        resource_path = run_dir / ".cache" / "quda"
        resource_path.mkdir(parents=True, exist_ok=True)

        if not os.path.exists(GAUGE_PATH):
            raise FileNotFoundError(f"Gauge configuration file not found: {GAUGE_PATH}")

        core.init(grid_size=PROCESS_GRID, latt_size=LATTICE, resource_path=str(resource_path))

        if core.getMPISize() != MPI_NUM:
            raise RuntimeError(
                f"MPI size mismatch: ensemble {ENSEMBLE_NAME} expects {MPI_NUM} ranks, got {core.getMPISize()}"
            )

        latt_info = core.LatticeInfo(LATTICE, t_boundary=T_BOUNDARY, anisotropy=ANISOTROPY)
        gauge = io.readChromaQIOGauge(GAUGE_PATH)
        gauge.stoutSmear(1, 0.125, 4)

        G0 = np.asarray(gamma.gamma(0), dtype=np.complex128)
        Gx = np.asarray(gamma.gamma(1), dtype=np.complex128)
        Gy = np.asarray(gamma.gamma(2), dtype=np.complex128)
        Gz = np.asarray(gamma.gamma(4), dtype=np.complex128)
        Gt = np.asarray(gamma.gamma(8), dtype=np.complex128)
        G5 = np.asarray(gamma.gamma(15), dtype=np.complex128)
        Cmat = Gy @ Gt
        Cg5 = Cmat @ G5
        Cg1 = Cmat @ Gx
        P_plus = (G0 + Gt) * 0.5
        epsilon = make_epsilon()

        strange_prop, strange_dirac = invert_flavor(gauge, latt_info, STRANGE_QUARK_MASS, "strange")
        diracs.append(strange_dirac)
        charm_prop, charm_dirac = invert_flavor(gauge, latt_info, CHARM_QUARK_MASS, "charm")
        diracs.append(charm_dirac)

        local_field = baryon_cg5_local(strange_prop, charm_prop, charm_prop, epsilon, Cg5, P_plus)
        correlator = gather_timeseries(local_field)

        if core.getMPIRank() == 0 and correlator is not None:
            output = np.column_stack((correlator.real, correlator.imag))
            output_path = run_dir / OUTPUT_NAME
            np.savetxt(output_path, output, fmt="%.16e")
    except Exception as exc:
        try:
            rank = core.getMPIRank()
        except Exception:
            rank = 0
        if rank == 0:
            print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        for dirac in reversed(diracs):
            destroy = getattr(dirac, "destroy", None)
            if callable(destroy):
                destroy()
