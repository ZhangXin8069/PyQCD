#!/usr/bin/env python3
"""Single-rank PyQUDA vacuum trace. Run as python -m quark_ringed_kinetic_v1.measure."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .core import (atomic_npz, canonical_hash, digest, free_noise_reference,
                   free_wilson_kinetic, kinetic_sample, metadata_array, unit_gauge,
                   validate_plan, z4_noise)


def input_identity(plan, conf, hash_gauge=True):
    if plan.get("free_field", False):
        return {"kind": "unit_links"}
    path = Path(plan["gauge_path_template"].format(conf=conf)).resolve(strict=True)
    expected = int(np.prod(plan["lattice_xyzt"])) * 4 * 3 * 3 * 2 * 8
    if path.stat().st_size != expected:
        raise ValueError(f"Raw ILDG payload size mismatch: {path}")
    return {"kind": "ildg_binary_big_endian_float64", "path": str(path),
            "bytes": expected, "sha256": digest(path) if hash_gauge else "not_computed_dry_run"}


def check_shard(path, identity_hash, nflow, noise_id=None):
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["metadata_json"]))
        if meta["run_identity_sha256"] != identity_hash:
            raise ValueError(f"Refusing resume with changed run identity: {path}")
        if noise_id is not None and meta.get("noise_id") != noise_id:
            raise ValueError(f"Noise label mismatch: {path}")
        values = np.array(d["kinetic_a4_noise"])
        if values.shape != (nflow,) or not np.isfinite(values).all():
            raise ValueError(f"Incomplete noise shard: {path}")
        return {k: np.array(d[k]) for k in d.files}


class QudaMeasurement:
    """Keep solve gauge and flow gauge lifetimes separate; no resident-gauge nesting."""
    def __init__(self, plan, gauge_identity, cache):
        import cupy as cp
        private_dir = os.environ.get('RINGED_QUDA_LIB_DIR')
        if private_dir:
            # Installed PyQUDA has DT_RPATH, which precedes LD_LIBRARY_PATH.
            # Load the explicit SONAME first, before importing its extension.
            import ctypes
            self.private_library = ctypes.CDLL(str(Path(private_dir).resolve(strict=True)/'libquda.so'),
                                               mode=ctypes.RTLD_GLOBAL)
        import pyquda
        from pyquda.field import LatticeInfo, LatticeGauge, LatticeFermion, MultiLatticeFermion
        from pyquda.dirac import GaugeDirac
        from pyquda.dirac.clover_wilson import CloverWilsonDirac
        from pyquda.dirac.wilson import WilsonDirac
        from pyquda import enum_quda as enums
        from pyquda.quda import performGFlowQuda

        if pyquda.getMPISize() != 1:
            raise ValueError("This full-volume stencil supports exactly one MPI rank/GPU")
        cache.mkdir(parents=True, exist_ok=True)
        pyquda.init(grid_size=[1, 1, 1, 1], latt_size=plan["lattice_xyzt"],
                    backend="cupy", resource_path=str(cache))
        libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                            if '/libquda.so' in line and line.split()[-1].startswith('/')})
        if len(libraries) != 1:
            raise RuntimeError(f"Cannot identify one loaded libquda: {libraries}")
        if private_dir and Path(libraries[0]).resolve() != (Path(private_dir)/'libquda.so').resolve():
            raise RuntimeError('Requested private QUDA was not loaded')
        self.runtime = {"pyquda": str(getattr(pyquda, "__version__", "unknown")),
                        "cupy": cp.__version__, "pyquda_module": str(Path(pyquda.__file__).resolve()),
                        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                        "libquda_path": libraries[0], "libquda_sha256": digest(libraries[0])}
        print(json.dumps({"runtime": self.runtime}), flush=True)
        self.cp, self.plan, self.enums = cp, plan, enums
        self.latt = latt = LatticeInfo(plan["lattice_xyzt"], t_boundary=-1)
        self.Gauge, self.Fermion, self.Multi = LatticeGauge, LatticeFermion, MultiLatticeFermion
        self.GaugeDirac, self.Clover, self.Wilson = GaugeDirac, CloverWilsonDirac, WilsonDirac
        self.perform = performGFlowQuda
        shape = tuple(reversed(plan["lattice_xyzt"]))
        if plan.get("free_field", False):
            lex = unit_gauge(shape)
        else:
            raw = np.memmap(gauge_identity["path"], mode="r", dtype=">f8", shape=(*shape, 4, 3, 3, 2))
            lex = np.ascontiguousarray((raw[..., 0] + 1j * raw[..., 1]).transpose(4, 0, 1, 2, 3, 5, 6))
        self.thin = LatticeGauge(latt, 4, cp.asarray(latt.evenodd(lex, multi=True)))
        del lex
        # The historical *valence* action uses one stout step, not flowed links.
        self.valence = self.thin.copy()
        if plan["stout_steps"]:
            self.valence.stoutSmear(plan["stout_steps"], plan["stout_rho"], plan["stout_dir_ignore"])
        self.flow_input = self.thin.copy()
        self.flow_input.setAntiPeriodicT()

    def sample(self, noise, direct=False):
        cp, p, latt, e = self.cp, self.plan, self.latt, self.enums
        # Previous flow temporaries are dead, but CuPy's idle pool can otherwise
        # starve the next independent QUDA multigrid setup on a 40 GB device.
        cp.get_default_memory_pool().free_all_blocks()
        eta = self.Fermion(latt, cp.asarray(latt.evenodd(noise, multi=False)))
        solver_kind = p.get('solver', 'bicgstab')
        multigrid = p['multigrid_blocks'] if solver_kind == 'gcr_mg' else None
        if p["csw"]:
            solver = self.Clover(latt, p["mass"], p["tol"], p["maxiter"], p["csw"], 1.0, multigrid)
        else:
            solver = self.Wilson(latt, p["mass"], p["tol"], p["maxiter"], multigrid)
        if solver_kind == 'cg_normop':
            solver.invert_param.inv_type = e.QudaInverterType.QUDA_CG_INVERTER
            solver.invert_param.solve_type = e.QudaSolveType.QUDA_NORMOP_PC_SOLVE
        elif solver_kind == 'gcr_mg':
            solver.multigrid.param.setup_use_mma = [0]*e.QUDA_MAX_MG_LEVEL
            solver.multigrid.param.dslash_use_mma = [0]*e.QUDA_MAX_MG_LEVEL
            if 'multigrid_setup_maxiter' in p:
                solver.multigrid.param.setup_maxiter = [int(p['multigrid_setup_maxiter'])]*e.QUDA_MAX_MG_LEVEL
                solver.multigrid.param.verbosity = [e.QudaVerbosity.QUDA_SUMMARIZE]*e.QUDA_MAX_MG_LEVEL
        elif solver_kind != 'bicgstab':
            raise ValueError(f'Unsupported declared solver: {solver_kind}')
        # Both choices return the full D_mass inverse, checked below, not DdagD inverse.
        if solver.invert_param.solution_type != e.QudaSolutionType.QUDA_MAT_SOLUTION:
            raise RuntimeError('Expected full MAT solution')
        if solver.invert_param.mass_normalization != e.QudaMassNormalization.QUDA_ASYMMETRIC_MASS_NORMALIZATION:
            raise RuntimeError("Unexpected solver normalization; refusing hidden 2*kappa conversion")
        solver.invert_param.preserve_source = e.QudaPreserveSource.QUDA_PRESERVE_SOURCE_YES
        started = time.perf_counter()
        with solver.useGauge(self.valence):
            q = solver.invert(eta)
            residual = float((cp.linalg.norm(solver.mat(q).data - eta.data) / cp.linalg.norm(eta.data)).get())
            iterations = int(solver.invert_param.iter)
        if not np.isfinite(residual) or residual > p["residual_max"]:
            raise RuntimeError(f"True full-MAT residual {residual} exceeds {p['residual_max']}")
        solve_seconds = time.perf_counter() - started
        source = self.Multi(latt, 2)
        source.data[0] = eta.data
        source.data[1] = q.data
        del eta, q, solver
        work = [self.Multi(latt, 2), self.Multi(latt, 2)]
        gd = self.GaugeDirac(latt)
        gd.smear_param.smear_type = e.QudaGaugeSmearType.QUDA_GAUGE_SMEAR_WILSON_FLOW
        gd.smear_param.epsilon = p["epsilon"]
        gd.obs_param.compute_plaquette = e.QudaBoolean.QUDA_BOOLEAN_TRUE
        gd.obs_param.compute_qcharge = e.QudaBoolean.QUDA_BOOLEAN_FALSE
        values, gauge_errors = [], []
        previous, current = 0, source
        gd.loadGauge(self.flow_input)
        try:
            for i, target in enumerate(validate_plan(p)):
                restart = i > 0 and not direct
                segment = int(target - previous) if restart else int(target)
                gd.smear_param.n_steps = segment
                gd.smear_param.t0 = previous * p["epsilon"] if restart else 0.0
                gd.smear_param.restart = (e.QudaBoolean.QUDA_BOOLEAN_TRUE if restart
                                          else e.QudaBoolean.QUDA_BOOLEAN_FALSE)
                gd.smear_param.meas_interval = segment + 1
                out = work[i % 2]
                self.perform(out.data_ptrs, (current if restart else source).data_ptrs,
                             gd.invert_param, gd.smear_param, gd.obs_param, 2)
                saved = self.Gauge(latt, 4)
                gd.saveSmearedGauge(saved)
                saved.setAntiPeriodicT()  # derivative applies APBC itself, exactly once
                gauge = latt.lexico(saved.data, multi=True, backend="cupy")
                fields = latt.lexico(out.data, multi=True, backend="cupy")
                if not bool(cp.isfinite(gauge).all().get()) or not bool(cp.isfinite(fields).all().get()):
                    raise RuntimeError(f"Nonfinite flowed gauge/fermion at checkpoint {p['flow_times'][i]}")
                values.append(complex(kinetic_sample(fields[0], fields[1], gauge, cp).get()))
                unit_err, det_err = 0.0, 0.0
                for mu in range(4):
                    u = gauge[mu]
                    unit_err = max(unit_err, float(cp.max(cp.abs(u @ u.swapaxes(-1, -2).conj() - cp.eye(3))).get()))
                    det_err = max(det_err, float(cp.max(cp.abs(cp.linalg.det(u) - 1)).get()))
                gauge_errors.append([unit_err, det_err])
                print(json.dumps({"checkpoint": p["flow_times"][i], "kinetic_real": values[-1].real,
                                  "kinetic_imag": values[-1].imag, "unitarity_max": unit_err,
                                  "determinant_max": det_err,
                                  "temporal_last_link_trace": str(complex(cp.trace(gauge[3, -1, 0, 0, 0]).get()))}), flush=True)
                if max(unit_err, det_err) > 1e-8 or not np.isfinite(values[-1]):
                    raise RuntimeError(f"Flowed gauge SU(3) or kinetic finite check failed: {unit_err=}, {det_err=}, E={values[-1]}")
                previous, current = int(target), out
                del saved, gauge, fields
        finally:
            gd.freeSmearedGauge()
            gd.freeGauge()
        return dict(kinetic_a4_noise=np.asarray(values), true_residual=np.array(residual),
                    iterations=np.array(iterations), solve_seconds=np.array(solve_seconds),
                    solver_kind=np.array(solver_kind),
                    gauge_unitarity_determinant_errors=np.asarray(gauge_errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--conf", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=9182026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--direct", action="store_true", help="Independent flow per target for restart comparisons")
    parser.add_argument("--dry-run", action="store_true", help="CPU preflight; no GPU imports or writes")
    args = parser.parse_args()
    with args.plan.open() as f:
        plan = json.load(f)
    steps = validate_plan(plan)
    if args.noise_count < 2 or args.seed < 0 or args.conf < 0:
        parser.error("Need at least two noise samples and nonnegative seed/configuration")
    for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS"):
        if int(os.environ.get(key, "1")) != 1:
            parser.error("Use a single-rank GPU job; parallelize separate configurations")
    gauge = input_identity(plan, args.conf, hash_gauge=not args.dry_run)
    vol = int(np.prod(plan["lattice_xyzt"]))
    print(json.dumps({"conf": args.conf, "steps": steps.tolist(), "gauge": gauge,
          "estimated_peak_GiB_not_measured": vol * 16 * (24*12 + 10*36 + 72) / 2**30,
          "status": "planned_unvalidated_vacuum_trace", "MPI_size": 1}, indent=2), flush=True)
    if args.dry_run:
        return
    identity = {"plan": plan, "conf": args.conf, "gauge": gauge, "seed": args.seed,
                "noise_count": args.noise_count, "direct": args.direct,
                "code_sha256": {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("core.py"))}}
    ihash = canonical_hash(identity)
    out = args.output.resolve()
    if out.exists() and not args.resume:
        raise FileExistsError("Output exists: use matching --resume or a new directory")
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".measurement.lock"
    # A crash leaves a visible lock; inspect the old job before removing it.
    with lock.open("x") as f:
        f.write(json.dumps({"pid": os.getpid(), "identity": ihash}) + "\n")
    try:
        backend = None
        records = []
        for inoise in range(args.noise_count):
            shard = out / f"noise_{inoise:05d}.npz"
            if shard.exists():
                records.append(check_shard(shard, ihash, len(steps), inoise))
                continue
            if backend is None:
                backend = QudaMeasurement(plan, gauge, out / "quda_cache")
            noise = z4_noise((*reversed(plan["lattice_xyzt"]), 4, 3), args.seed, args.conf, inoise)
            values = backend.sample(noise, args.direct)
            if plan.get("free_field", False):
                reference = free_noise_reference(noise, plan["mass"], plan["flow_times"])
                values["free_same_noise_reference"] = reference
                rel = float(np.max(np.abs(values["kinetic_a4_noise"] - reference) / np.maximum(np.abs(reference), 1e-14)))
                values["free_same_noise_max_relative_error"] = np.array(rel)
                if rel > 5e-4:
                    raise RuntimeError(f"Free-field same-noise comparison failed: {rel}")
            meta = {"schema": "quark_ringed_noise_v1", "status": "measured_unvalidated",
                    "run_identity_sha256": ihash, "noise_id": inoise, "run_identity": identity,
                    "runtime": backend.runtime}
            atomic_npz(shard, metadata_json=metadata_array(meta), **values)
            records.append(check_shard(shard, ihash, len(steps), inoise))
            print(f"noise {inoise + 1}/{args.noise_count}: residual={float(values['true_residual']):.3e}", flush=True)
        final = out / "kinetic_conf.npz"
        runtimes = [json.loads(str(r["metadata_json"]))["runtime"] for r in records]
        if any(r != runtimes[0] for r in runtimes):
            raise ValueError("Resume changed the backend runtime; start a distinct run")
        data = np.stack([r["kinetic_a4_noise"] for r in records], axis=1)
        if not final.exists():
            meta = {"schema": "quark_ringed_configuration_raw_v1", "status": "measured_unvalidated",
                    "run_identity": identity, "run_identity_sha256": ihash,
                    "axes_kinetic": ["flow", "noise"], "trace_space": "full_spin_color_spacetime",
                    "temporal_truncation": "none", "noise": "full_volume_Z4_E_eta_eta_dagger_equals_I",
                    "fermion_normalization": "D_mass_inverse_no_extra_2kappa",
                    "derivative_discretization": "4D_symmetric_covariant_nearest_neighbor_aD",
                    "flavor_count": 1, "massless_prescription_validated": False,
                    "runtime": runtimes[0], "command": sys.argv,
                    "source_shards": [digest(out / f"noise_{i:05d}.npz") for i in range(args.noise_count)]}
            extra = {}
            if plan.get("free_field", False):
                extra["free_exact_trace"] = free_wilson_kinetic(plan["lattice_xyzt"], plan["mass"], plan["flow_times"])
            atomic_npz(final, metadata_json=metadata_array(meta), flow_times=plan["flow_times"],
                       configuration=np.array(args.conf), kinetic_a4_noise=data,
                       kinetic_a4_config=data.mean(axis=1), noise_sem_real=data.real.std(axis=1, ddof=1)/np.sqrt(args.noise_count),
                       true_residual=np.array([r["true_residual"] for r in records]), **extra)
        else:
            with np.load(final, allow_pickle=False) as d:
                if json.loads(str(d["metadata_json"]))["run_identity_sha256"] != ihash:
                    raise ValueError("Existing configuration product has wrong identity")
                if not np.array_equal(d["kinetic_a4_noise"], data):
                    raise ValueError("Existing configuration product differs from noise shards")
        print(str(final), flush=True)
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
