from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

try:
    from pyquda_utils import core, io, gamma
    from pyquda_plugins import pycontract
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
STOUT_SMEAR_STEPS = 1
STOUT_SMEAR_RHO = 0.125
STOUT_SMEAR_DIR_IGNORE = 4
TOL = 1.0e-12
MAXITER = 1000
MRHS = 4
OUTPUT_NAME = "pion_2pt_C24P29_cfg_10000.txt"


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
    propagator_lexico = propagator.lexico()
    if not np.all(np.isfinite(propagator_lexico)):
        raise RuntimeError(f"Propagator contains non-finite values for {label}")
    return propagator, dirac


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

        pycontract.init()

        latt_info = core.LatticeInfo(LATTICE, t_boundary=T_BOUNDARY, anisotropy=ANISOTROPY)
        gauge = io.readChromaQIOGauge(GAUGE_PATH)
        gauge.stoutSmear(STOUT_SMEAR_STEPS, STOUT_SMEAR_RHO, STOUT_SMEAR_DIR_IGNORE)

        gamma_x = gamma.Gamma(1)
        gamma_y = gamma.Gamma(2)
        gamma_z = gamma.Gamma(4)
        gamma_t = gamma.Gamma(8)
        gamma_5 = gamma.Gamma(15)
        identity = gamma.Gamma(0)
        C = gamma_y @ gamma_t
        Cg5 = C @ gamma_5
        Cg1 = C @ gamma_x
        Pp = (identity + gamma_t) / 2

        light_prop, light_dirac = invert_flavor(gauge, latt_info, LIGHT_QUARK_MASS, "light")
        diracs.append(light_dirac)

        local_field = pycontract.mesonTwoPoint(light_prop, light_prop, gamma_5, gamma_5).lexico()
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
