#!/usr/bin/env python3
"""Compute-node regression: independent nonzero NumPy OPE oracle and LIME I/O."""
import copy
import itertools
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ope_components import (ROOT, PRIMITIVES, atomic_json, export_product, sha256,
                            validate_product, validate_raw)


def write_lime(path, gauge):
    pairs = np.stack((gauge.real, gauge.imag), axis=-1).astype(">f8")
    payload = pairs.tobytes()
    header = struct.pack(">IHHQ128s", 0x456789AB, 1, 3, len(payload),
                         b"ildg-binary-data".ljust(128, b"\0"))
    path.write_bytes(header + payload + b"\0" * (-len(payload) % 8))
    return len(header), len(payload)


def fixture(root, gauge, tau, zcount, label):
    path = root / f"{label}.lime"
    offset, payload = write_lime(path, gauge)
    nt, nz, ny, nx = gauge.shape[:4]
    stat = path.stat()
    req = dict(conf_id="1", gauge_path=str(path), input_size=stat.st_size,
               input_mtime_ns=stat.st_mtime_ns, lime_payload_offset=offset,
               lime_payload_bytes=payload, lattice_xyzt=[nx, ny, nz, nt],
               a_fm=0.0897, tau=tau, epsilon=0.01, z_dir=2, z_count=zcount)
    base = root / label / "raw"
    subprocess.run([str(ROOT / "build/flowed_gluon_ope_components"), "--input", str(path),
                    "--output", str(base), "--conf-id", "1", "--nx", str(nx),
                    "--ny", str(ny), "--nz", str(nz), "--nt", str(nt),
                    "--tau", str(tau), "--epsilon", "0.01", "--z-count", str(zcount),
                    "--z-dir", "2"], check=True)
    return base, req


def oracle(gauge, zcount):
    """Float64 color-matrix oracle, independent of C++ storage and exporter."""
    dims = gauge.shape[:4]
    xyz_axis = (3, 2, 1, 0)

    def step(site, mu, sign):
        other = list(site)
        axis = xyz_axis[mu]
        other[axis] = (other[axis] + sign) % dims[axis]
        return tuple(other)

    def path_matrix(site, directions):
        w = np.eye(3, dtype=complex)
        for mu, sign in directions:
            if sign > 0:
                w = w @ gauge[site + (mu,)]
                site = step(site, mu, 1)
            else:
                site = step(site, mu, -1)
                w = w @ gauge[site + (mu,)].conj().T
        return w, site

    fields = np.zeros(dims + (4, 4, 3, 3), dtype=complex)
    for site in np.ndindex(dims):
        for mu, nu in itertools.combinations(range(4), 2):
            m, n = (mu, 1), (nu, 1)
            mm, nn = (mu, -1), (nu, -1)
            q = sum(path_matrix(site, p)[0] for p in ((m, n, mm, nn), (n, mm, nn, m),
                                                     (mm, nn, m, n), (nn, m, n, mm)))
            f = -1j * (q - q.conj().T) / 8
            fields[site + (mu, nu)] = f
            fields[site + (nu, mu)] = -f
    out = {name: np.zeros((2, 4, zcount, dims[0]), dtype=complex) for name in PRIMITIVES}
    # epsilon_xyzt=+1: dualF_tx=-F_yz; dualF_ty=+F_xz; dualF_xy=+F_zt.
    dual_pairs = ((-1, 1, 2), (+1, 0, 2), (+1, 2, 3))
    pairs = ((3, 0), (3, 1), (0, 1))
    for projection in range(2):
        def project(a):
            return a - np.trace(a) * np.eye(3) / 3 if projection else a
        for orientation, sign in enumerate((+1, -1)):
            for z in range(zcount):
                for site in np.ndindex(dims):
                    w, endpoint = path_matrix(site, [(2, sign)] * z)
                    us, hs = [], []
                    for (mu, nu), (dsign, r, s) in zip(pairs, dual_pairs):
                        a = project(fields[endpoint + (mu, nu)])
                        b = project(fields[site + (mu, nu)])
                        dual = project(dsign * fields[site + (r, s)])
                        us.append(np.trace(a @ w.conj().T @ b @ w))
                        hs.append(np.trace(a @ w.conj().T @ dual @ w))
                    index = (projection, orientation, z, site[0])
                    out["unpolarized/TU.npy"][index] += us[0] + us[1]
                    out["unpolarized/SU.npy"][index] += 2 * us[2]
                    out["helicity/TH.npy"][index] += hs[0] + hs[1]
                    out["helicity/SH.npy"][index] += 2 * hs[2]
    for a in out.values():
        a[:, 2] = a[:, 0] + a[:, 1]
        a[:, 3] = a[:, 0] - a[:, 1]
    return out


def must_reject(callback):
    try:
        callback()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError("invalid artifact/identity was accepted")


def main():
    with tempfile.TemporaryDirectory(prefix="gf64_ope_test_", dir="/tmp") as temp:
        root = Path(temp)
        unit = np.broadcast_to(np.eye(3), (2, 2, 2, 2, 4, 3, 3)).copy()
        base, req = fixture(root, unit, 0.02, 2, "unit")
        data, _ = validate_raw(base, req)
        assert np.max(np.abs(data)) < 1e-12
        export_product(base, req, root / "unit_product")
        validate_product(root / "unit_product", req)

        rng = np.random.default_rng(20261009)
        a = rng.normal(size=(4, 3, 3, 3, 4, 3, 3)) + 1j * rng.normal(size=(4, 3, 3, 3, 4, 3, 3))
        h = 0.2 * (a + a.conj().swapaxes(-1, -2))
        h -= np.trace(h, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
        eigenvalues, eigenvectors = np.linalg.eigh(h)
        gauge = (eigenvectors * np.exp(1j * eigenvalues)[..., None, :]) @ eigenvectors.conj().swapaxes(-1, -2)
        base, req = fixture(root, gauge, 0.0, 3, "nonzero")
        validate_raw(base, req)
        destination = root / "nonzero_product"
        export_product(base, req, destination)
        validate_product(destination, req)
        reference = oracle(gauge, 3)
        errors = {}
        for name, want in reference.items():
            got = np.load(destination / name)
            errors[name] = float(np.max(np.abs(got - want)))
            np.testing.assert_allclose(got, want, rtol=3e-5, atol=3e-6)
        th = reference["helicity/TH.npy"][:, 3]
        sh = reference["helicity/SH.npy"][:, 3]
        assert np.max(np.abs(sh[:, 1:])) > 1e-3  # discriminate minus from legacy plus
        np.testing.assert_allclose(np.load(destination / "helicity/H_TminusS_odd.npy"), th - sh,
                                   rtol=3e-5, atol=3e-6)
        for changed in (dict(req, conf_id="2"), dict(req, tau=0.8), dict(req, z_dir=1),
                        dict(req, lattice_xyzt=[3, 3, 3, 5])):
            must_reject(lambda: validate_raw(base, changed))
            must_reject(lambda: validate_product(destination, changed))
        (destination / "helicity/SH.npy").write_bytes(b"corrupt")
        must_reject(lambda: validate_product(destination, req))
        assert (destination / "unpolarized/TU.npy").is_file()
        evidence = dict(passed=True, oracle="independent float64 Clover/bilocal and explicit dual identities",
                        max_abs_error=errors, tested_lattice_xyzt=[3, 3, 3, 4],
                        identity_flow_steps=2, nonzero_fixture_tau=0,
                        guards=["wrong conf", "wrong tau", "wrong zdir", "wrong lattice", "corrupt file"],
                        target="(TH-SH)_odd distinct from legacy (TH+SH)_odd")
        atomic_json(ROOT / "validation/unit_oracle_tests.json", evidence)
        print("COMPONENT_ORACLE_TEST_OK " + json.dumps(evidence), flush=True)


if __name__ == "__main__":
    main()
