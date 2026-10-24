"""Cross-check PyQCD against the archived donghx gradient-flow gluon OPE.

The reference implementation is a straight Wilson-line gluon OPE.  It does not
contain the transverse staple used by the PyQCD TMD extension.  This harness
therefore compares the shared flow and OPE layers, plus the shared downstream
matching primitives, and records the TMD collinear reduction separately.

All generated artifacts stay under ``pyqcd/testing/tmd/artifacts``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parents[3]
REFERENCE_DIR = ROOT / "refer/donghx/_Gradient_Flow_GluonPDF"
REFERENCE_CPP = REFERENCE_DIR / "flowed_gluon_ope.cpp"
REFERENCE_VALIDATOR = REFERENCE_DIR / "validate_output_v5.py"
ARTIFACT_ROOT = Path(__file__).resolve().parent / "artifacts"

DEFAULT_SHAPE = (8, 4, 4, 4)  # (Nt,Nz,Ny,Nx) -> 4x4x4x8 physical lattice
DEFAULT_TAUS = (0.0, 0.1, 0.5)
DEFAULT_EPS = 0.01
DEFAULT_SEED = 20261024
CONF_ID = "cross-reference-4x4x4x8"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_lime(path: Path, gauge: np.ndarray) -> None:
    pairs = np.stack((gauge.real, gauge.imag), axis=-1).astype(">f8")
    payload = pairs.tobytes(order="C")
    header = struct.pack(
        ">IHHQ128s",
        0x456789AB,
        1,
        3,
        len(payload),
        b"ildg-binary-data".ljust(128, b"\0"),
    )
    path.write_bytes(header + payload + b"\0" * (-len(payload) % 8))


def _near_identity_su3(shape: tuple[int, int, int, int], *,
                       seed: int, strength: float = 0.12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gauge = np.empty(shape + (4, 3, 3), dtype=np.complex64)
    for index in np.ndindex(*shape, 4):
        matrix = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        generator = matrix - matrix.conj().T
        generator -= np.trace(generator) * np.eye(3) / 3.0
        gauge[index] = expm(strength * generator)
    return gauge


def _compile_reference(source: Path, output: Path) -> dict:
    command = [
        "g++", "-O3", "-std=c++17", "-ffast-math", "-Wall", "-Wextra",
        str(source), "-o", str(output),
    ]
    subprocess.run(command, check=True)
    version = subprocess.run(
        ["g++", "--version"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout.splitlines()[0]
    return {"command": command, "compiler": version, "binary": str(output)}


def _write_flow_driver(path: Path, source: Path) -> None:
    escaped = str(source).replace("\\", "\\\\").replace('"', '\\"')
    path.write_text(
        f'''#define main donghx_archived_cli_main
#include "{escaped}"
#undef main

static void write_flowed_links(const fs::path &path,
                               const std::vector<Mat3> &links) {{
  fs::path tmp = path;
  tmp += ".tmp";
  std::ofstream stream(tmp, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open flow output");
  for (const Mat3 &matrix : links) {{
    for (int component = 0; component < 9; ++component) {{
      const float real = matrix.v[component].real();
      const float imag = matrix.v[component].imag();
      stream.write(reinterpret_cast<const char *>(&real), sizeof(float));
      stream.write(reinterpret_cast<const char *>(&imag), sizeof(float));
    }}
  }}
  stream.close();
  if (!stream) throw std::runtime_error("cannot write flow output");
  fs::rename(tmp, path);
}}

int main(int argc, char **argv) {{
  try {{
    std::string input, output;
    int nx = 4, ny = 4, nz = 4, nt = 8;
    double tau = 0.0, epsilon = 0.01;
    for (int i = 1; i < argc; ++i) {{
      const std::string key = argv[i];
      auto value = [&]() {{
        if (++i >= argc) throw std::runtime_error("missing value for " + key);
        return std::string(argv[i]);
      }};
      if (key == "--input") input = value();
      else if (key == "--output") output = value();
      else if (key == "--nx") nx = std::stoi(value());
      else if (key == "--ny") ny = std::stoi(value());
      else if (key == "--nz") nz = std::stoi(value());
      else if (key == "--nt") nt = std::stoi(value());
      else if (key == "--tau") tau = std::stod(value());
      else if (key == "--epsilon") epsilon = std::stod(value());
      else throw std::runtime_error("unknown argument " + key);
    }}
    if (input.empty() || output.empty() || tau < 0.0 || epsilon <= 0.0)
      throw std::runtime_error("invalid arguments");
    const long long steps = std::llround(tau / epsilon);
    if (std::abs(tau - steps * epsilon) > 5e-12)
      throw std::runtime_error("tau/epsilon must be an integer");
    Lattice lattice(nx, ny, nz, nt);
    LimePayload payload;
    auto links = read_ildg(input, lattice, payload);
    wilson_flow(links, lattice, static_cast<int>(steps), static_cast<float>(epsilon));
    write_flowed_links(output, links);
    return 0;
  }} catch (const std::exception &error) {{
    std::cerr << "ERROR: " << error.what() << std::endl;
    return 2;
  }}
}}
''',
        encoding="utf-8",
    )


def _load_donghx_module(name: str, filename: str):
    if str(REFERENCE_DIR) not in sys.path:
        sys.path.insert(0, str(REFERENCE_DIR))
    spec = importlib.util.spec_from_file_location(name, REFERENCE_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_reference_ope(binary: Path, input_path: Path, base: Path, *,
                       shape: tuple[int, int, int, int], tau: float,
                       epsilon: float) -> dict:
    nt, nz, ny, nx = shape
    command = [
        str(binary), "--input", str(input_path), "--output", str(base),
        "--conf-id", CONF_ID, "--nx", str(nx), "--ny", str(ny),
        "--nz", str(nz), "--nt", str(nt), "--tau", repr(tau),
        "--epsilon", repr(epsilon), "--z-count", str(nz), "--z-dir", "2",
    ]
    process = subprocess.run(
        command, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    validate = subprocess.run(
        [
            sys.executable, str(REFERENCE_VALIDATOR), str(base),
            "--conf-id", CONF_ID, "--z-count", str(nz), "--nt", str(nt),
            "--tau", repr(tau), "--epsilon", repr(epsilon),
        ],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return {
        "run_output": process.stdout.strip(),
        "validation_output": validate.stdout.strip(),
        "data": np.load(str(base) + ".npy", allow_pickle=False),
    }


def _run_reference_flow(driver: Path, input_path: Path, output_path: Path, *,
                        shape: tuple[int, int, int, int], tau: float,
                        epsilon: float) -> np.ndarray:
    nt, nz, ny, nx = shape
    subprocess.run(
        [
            str(driver), "--input", str(input_path), "--output", str(output_path),
            "--nx", str(nx), "--ny", str(ny), "--nz", str(nz), "--nt", str(nt),
            "--tau", repr(tau), "--epsilon", repr(epsilon),
        ],
        check=True,
    )
    raw = np.fromfile(output_path, dtype=np.float32)
    expected = int(np.prod(shape)) * 4 * 3 * 3 * 2
    if raw.size != expected:
        raise ValueError(f"flow output has {raw.size} floats, expected {expected}")
    pairs = raw.reshape(shape + (4, 3, 3, 2))
    return (pairs[..., 0] + 1j * pairs[..., 1]).astype(np.complex64)


def _difference(reference: np.ndarray, candidate: np.ndarray) -> dict:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        return {
            "shape_reference": list(reference.shape),
            "shape_candidate": list(candidate.shape),
            "shape_match": False,
        }
    difference = candidate - reference
    reference_norm = float(np.linalg.norm(reference.ravel()))
    difference_norm = float(np.linalg.norm(difference.ravel()))
    scale = float(np.max(np.abs(reference))) if reference.size else 0.0
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    return {
        "shape_match": True,
        "max_abs": max_abs,
        "rms_abs": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
        "relative_l2": difference_norm / max(reference_norm, 1e-300),
        "normalized_max": max_abs / max(scale, 1e-300),
        "reference_max_abs": scale,
        "finite": bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
    }


def _group_differences(reference: np.ndarray, candidate: np.ndarray) -> dict:
    output = {}
    for projection_index, projection in enumerate(
            ("legacy_untraced", "traceless")):
        for channel_index, channel in enumerate(("unpolarized", "helicity")):
            key = f"{projection}/{channel}"
            output[key] = _difference(
                reference[projection_index, channel_index],
                candidate[projection_index, channel_index],
            )
    return output


def _compare_ratio_convention(donghx_ratio) -> dict:
    rng = np.random.default_rng(20261026)
    operator = rng.normal(size=(6, 3, 5)) + 1j * rng.normal(size=(6, 3, 5))
    correlator = (
        3.0 + rng.normal(size=(6, 5, 2))
        + 1j * (0.2 + 0.05 * rng.normal(size=(6, 5, 2)))
    )
    from pyqcd.analysis._gradient_flow_ratio import covariance_ratio_estimator

    reference = donghx_ratio.covariance_ratio_estimator(
        operator, correlator, True, denominator_correlator=correlator)
    candidate = covariance_ratio_estimator(
        operator, correlator, True, denominator_correlator=correlator)
    reference_ratio = reference[0] / reference[1].real[None, :]
    candidate_ratio = candidate[0] / candidate[1][None, :]
    return {
        "central_c3": _difference(reference[0], candidate[0]),
        "central_c2": _difference(reference[1], candidate[1]),
        "reported_ratio": _difference(reference_ratio, candidate_ratio),
        "jackknife_error_real": _difference(reference[4], candidate[4]),
        "jackknife_error_imag": _difference(reference[5], candidate[5]),
        "expected_convention": (
            "donghx divides by Re(C2_nopol); PyQCD schema-v2 ratio keeps "
            "complex C2 denominator"
        ),
    }


def _compare_flow_to_quasi(donghx_flow_to_quasi, py_match_one,
                           reference_ope: np.ndarray,
                           candidate_ope: np.ndarray, tau: float) -> dict:
    reference, reference_metadata = donghx_flow_to_quasi.match_one(
        reference_ope, tau, 0.0775, 2.0, 0.25, 3.0)
    candidate, candidate_metadata = py_match_one(
        reference_ope, tau, 0.0775, 2.0, 0.25, 3.0,
        z0_policy="legacy_endpoint",
        helicity_mode="legacy_plus_combined",
    )
    candidate_native, _ = py_match_one(
        candidate_ope, tau, 0.0775, 2.0, 0.25, 3.0,
        z0_policy="legacy_endpoint",
        helicity_mode="legacy_plus_combined",
    )
    return {
        "difference": _difference(reference, candidate),
        "native_pipeline_difference": _difference(reference, candidate_native),
        "reference_status": reference_metadata,
        "candidate_status": candidate_metadata,
    }


def _compare_matching_kernel(donghx_matching) -> dict:
    from pyqcd.renorm._matching import _matching_kernels

    xi = np.array([-3.0, -1.5, -0.5, 0.2, 0.5, 0.8, 1.5, 2.0, 3.0])
    y = np.array([0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0, 3.0])
    pz_gev = 2.0
    mu = 2.0
    zs_fm = 0.3
    hbar = 0.1973269804
    lambda_s = zs_fm * pz_gev / hbar
    reference = donghx_matching.kernel(xi, np.abs(y) * pz_gev, mu, zs_fm)
    candidate = _matching_kernels(xi, y, pz_gev, mu, lambda_s)
    return {
        "difference": _difference(np.asarray(reference), np.asarray(candidate)),
        "xi": xi.tolist(),
        "y": y.tolist(),
        "pz_gev": pz_gev,
        "mu_gev": mu,
        "zs_fm": zs_fm,
    }


def _compare_fourier(donghx_fourier, h_z: np.ndarray) -> dict:
    from pyqcd.renorm._ensembles import fm_to_GeV
    from pyqcd.renorm._tmdextract import ioffe_time_distribution

    z_a = np.arange(h_z.size, dtype=float)
    a_fm = 0.0775
    pz_integer = 1
    x_grid = np.linspace(-0.9, 0.9, 41)
    original_hbar = donghx_fourier.HBARC
    results = {}
    try:
        donghx_fourier.HBARC = fm_to_GeV
        pz_gev = 2.0 * np.pi * pz_integer / (32.0 * a_fm / fm_to_GeV)
        reference = donghx_fourier.fourier_matrix(
            z_a, pz_integer, a_fm, x_grid, even=True) @ h_z
        _, candidate = ioffe_time_distribution(
            h_z, z_a * a_fm, pz_gev, x_grid=x_grid, n_pts=len(z_a))
        results["same_hbar"] = _difference(reference, candidate)

        donghx_fourier.HBARC = original_hbar
        pz_default = 2.0 * np.pi * pz_integer / (32.0 * a_fm / original_hbar)
        reference_default = donghx_fourier.fourier_matrix(
            z_a, pz_integer, a_fm, x_grid, even=True) @ h_z
        _, candidate_default = ioffe_time_distribution(
            h_z, z_a * a_fm, pz_default, x_grid=x_grid, n_pts=len(z_a))
        results["default_constants"] = _difference(
            reference_default, candidate_default)
        results["pyqcd_hbar"] = fm_to_GeV
        results["donghx_hbar"] = original_hbar
    finally:
        donghx_fourier.HBARC = original_hbar
    return results


def _tmd_collinear_reduction(gauge: np.ndarray, tau: float, epsilon: float,
                             shape: tuple[int, int, int, int]) -> dict:
    from pyqcd.operator import flowed_gluon_ope
    from pyqcd.renorm._gradient_flow import wilson_flow
    from pyqcd.renorm._tmd import tmd_matrix_elements_time

    _, nz, _, _ = shape

    def reduction(dtype):
        flowed = wilson_flow(gauge.astype(dtype), tau=tau, eps=epsilon)
        straight = flowed_gluon_ope(flowed, z_count=nz, z_dir=2)
        tmd = tmd_matrix_elements_time(
            flowed, list(range(nz)), [0], z_dir=2, b_dir=0, L=0,
            color_normalization="fundamental_trace")
        straight_unpolarized = straight[1, 0, 0, 0, :, :].real
        return flowed, tmd, _difference(
            straight_unpolarized, tmd[:, 0, :])

    _, tmd64, difference64 = reduction(np.complex64)
    flowed128, _, difference128 = reduction(np.complex128)
    finite_tmd = tmd_matrix_elements_time(
        flowed128, list(range(nz)), [0, 1, 2, 3], z_dir=2, b_dir=0, L=3,
        color_normalization="fundamental_trace")
    return {
        "b0_L0_vs_straight_complex64": difference64,
        "b0_L0_vs_straight_complex128": difference128,
        "b0_L0_tmd_complex64_shape": list(tmd64.shape),
        "finite_b_shape": list(finite_tmd.shape),
        "finite_b_finite": bool(np.isfinite(finite_tmd).all()),
        "finite_b_max_abs": float(np.max(np.abs(finite_tmd))),
        "finite_b_has_no_donghx_reference": True,
    }


def _render_report(result: dict) -> str:
    last_tau_key = f"{result['tau_values'][-1]:.3f}"
    flow_to_quasi = result["flow_to_quasi"]
    flow_to_quasi_key = list(flow_to_quasi)[-1] if flow_to_quasi else None
    fourier_tau_key = result["fourier_source_tau"]
    flow_to_quasi_text = (
        "not tested (no positive tau)"
        if flow_to_quasi_key is None else
        f"`{flow_to_quasi[flow_to_quasi_key]['difference']['max_abs']:.3e}`"
    )
    native_flow_to_quasi_text = (
        "not tested (no positive tau)"
        if flow_to_quasi_key is None else
        f"`{flow_to_quasi[flow_to_quasi_key]['native_pipeline_difference']['max_abs']:.3e}`"
    )
    lines = [
        "# Donghx cross-reference: 4x4x4x8",
        "",
        f"- Lattice `(Lx,Ly,Lz,Nt)=(4,4,4,8)`, array shape `{DEFAULT_SHAPE}`",
        f"- tau values: `{result['tau_values']}`; epsilon: `{result['epsilon']}`",
        f"- Input SHA-256: `{result['input_sha256']}`",
        "",
        "## Flow and straight OPE",
        "",
        "| tau | flow rel-L2 | OPE c64 rel-L2 | OPE c128 rel-L2 |",
        "|---:|---:|---:|---:|",
    ]
    for tau in result["tau_values"]:
        key = f"{tau:.3f}"
        flow = result["flow_comparisons"][key]["complex64"]["relative_l2"]
        ope64 = result["ope_comparisons"][key]["pyqcd_complex64"]["relative_l2"]
        ope128 = result["ope_comparisons"][key]["pyqcd_complex128"]["relative_l2"]
        lines.append(f"| {tau:g} | {flow:.3e} | {ope64:.3e} | {ope128:.3e} |")
    lines.extend([
        "",
        "## Shared post-processing",
        "",
        f"- flow-to-quasi max difference: {flow_to_quasi_text}",
        f"- native OPE-to-quasi max difference: {native_flow_to_quasi_text}",
        f"- gg matching-kernel max difference: "
        f"`{result['matching_kernel']['difference']['max_abs']:.3e}`",
        f"- Fourier same-hbar max difference: "
        f"`{result['fourier']['same_hbar']['max_abs']:.3e}` "
        f"(source tau `{fourier_tau_key}`)",
        f"- Fourier default-constant max difference: "
        f"`{result['fourier']['default_constants']['max_abs']:.3e}`",
        f"- ratio central-C3 max difference: "
        f"`{result['ratio_convention']['central_c3']['max_abs']:.3e}`",
        f"- ratio denominator-convention max difference: "
        f"`{result['ratio_convention']['reported_ratio']['max_abs']:.3e}`",
        "",
        "## TMD boundary",
        "",
        f"- PyQCD `b=0,L=0` collinear reduction, complex64 max difference: "
        f"`{result['tmd'][last_tau_key]['b0_L0_vs_straight_complex64']['max_abs']:.3e}`",
        f"- PyQCD `b=0,L=0` collinear reduction, complex128 max difference: "
        f"`{result['tmd'][last_tau_key]['b0_L0_vs_straight_complex128']['max_abs']:.3e}`",
        "- Finite-`b` TMD is computed by PyQCD but has no donghx counterpart "
        "in the archived reference.",
        "",
        f"Overall: **{result['overall_status']}**",
    ])
    return "\n".join(lines) + "\n"


def run(out_dir: Path, *, shape: tuple[int, int, int, int] = DEFAULT_SHAPE,
        taus: tuple[float, ...] = DEFAULT_TAUS,
        epsilon: float = DEFAULT_EPS, seed: int = DEFAULT_SEED) -> dict:
    from pyqcd.operator import gradient_flow_gluon_ope, validate_gradient_flow_gluon_ope
    from pyqcd.renorm._gradient_flow import wilson_flow
    from pyqcd.renorm._gradient_flow_quasi import match_one as py_match_one
    from pyqcd.tools import set_backend

    if len(shape) != 4 or any(int(value) <= 0 for value in shape):
        raise ValueError("shape must be four positive integers")
    if not taus or any(not np.isfinite(tau) or tau < 0.0 for tau in taus):
        raise ValueError("taus must be finite and non-negative")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    set_backend("numpy")
    out_dir.mkdir(parents=True, exist_ok=False)
    build_dir = out_dir / "build"
    build_dir.mkdir()
    input_path = out_dir / "gauge_4x4x4x8.lime"
    gauge = _near_identity_su3(shape, seed=seed)
    _write_lime(input_path, gauge)

    reference_binary = build_dir / "flowed_gluon_ope"
    compile_result = _compile_reference(REFERENCE_CPP, reference_binary)
    self_test = subprocess.run(
        [str(reference_binary), "--self-test"], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )

    driver_source = build_dir / "flow_driver.cpp"
    driver_binary = build_dir / "flow_driver"
    _write_flow_driver(driver_source, REFERENCE_CPP)
    _compile_reference(driver_source, driver_binary)

    donghx_flow_to_quasi = _load_donghx_module(
        "donghx_flow_to_quasi", "gluon_flow_to_quasi.py")
    donghx_matching = _load_donghx_module(
        "donghx_match_v2", "match_unpolarized_quasi_from_extrap_v2.py")
    donghx_fourier = _load_donghx_module(
        "donghx_fourier", "match_unpolarized_quasi_from_extrap_v1.py")
    donghx_ratio = _load_donghx_module(
        "donghx_ratio", "calc_ratio_matched.py")

    arrays = {"gauge": gauge}
    result = {
        "schema": "pyqcd.donghx_cross_reference.v1",
        "created": datetime.now().astimezone().isoformat(),
        "lattice": {"lx": shape[3], "ly": shape[2], "lz": shape[1], "nt": shape[0]},
        "array_shape": list(shape),
        "seed": seed,
        "tau_values": list(taus),
        "epsilon": epsilon,
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "reference_source": str(REFERENCE_CPP),
        "reference_source_sha256": _sha256_file(REFERENCE_CPP),
        "compile": compile_result,
        "self_test": self_test.stdout.strip(),
        "ope_comparisons": {},
        "flow_comparisons": {},
        "flow_to_quasi": {},
        "tmd": {},
    }

    required_checks = []
    check_details = []

    def add_check(name: str, passed: bool, observed: float,
                  threshold: float) -> None:
        passed = bool(passed)
        required_checks.append(passed)
        check_details.append({
            "name": name,
            "passed": passed,
            "observed": float(observed),
            "threshold": float(threshold),
        })
    for tau in taus:
        key = f"{tau:.3f}"
        cpp_base = out_dir / f"donghx_ope_tau{key}"
        reference_run = _run_reference_ope(
            reference_binary, input_path, cpp_base, shape=shape,
            tau=tau, epsilon=epsilon)
        cpp_ope = reference_run["data"]

        cpp_flow = _run_reference_flow(
            driver_binary, input_path, out_dir / f"donghx_flow_tau{key}.raw",
            shape=shape, tau=tau, epsilon=epsilon)
        py_flow64 = wilson_flow(gauge, tau=tau, eps=epsilon)
        py_flow128 = wilson_flow(gauge.astype(np.complex128), tau=tau, eps=epsilon)
        result["flow_comparisons"][key] = {
            "complex64": _difference(cpp_flow, py_flow64),
            "complex128": _difference(cpp_flow, py_flow128),
        }

        py_ope64, metadata64 = gradient_flow_gluon_ope(
            gauge, tau=tau, epsilon=epsilon, z_count=shape[1], z_dir=2,
            conf_id=CONF_ID)
        validate_gradient_flow_gluon_ope(
            py_ope64, metadata64, conf_id=CONF_ID, tau=tau, epsilon=epsilon)
        py_ope128, metadata128 = gradient_flow_gluon_ope(
            gauge.astype(np.complex128), tau=tau, epsilon=epsilon,
            z_count=shape[1], z_dir=2, conf_id=CONF_ID)
        validate_gradient_flow_gluon_ope(
            py_ope128, metadata128, conf_id=CONF_ID, tau=tau, epsilon=epsilon)
        result["ope_comparisons"][key] = {
            "donghx_validation": reference_run["validation_output"],
            "donghx_shape": list(cpp_ope.shape),
            "pyqcd_shape": list(py_ope64.shape),
            "pyqcd_complex64": _difference(cpp_ope, py_ope64),
            "pyqcd_complex128": _difference(cpp_ope, py_ope128),
            "groups_complex64": _group_differences(cpp_ope, py_ope64),
        }
        arrays[f"cpp_ope_{key}"] = cpp_ope
        arrays[f"py_ope_c64_{key}"] = py_ope64
        arrays[f"py_ope_c128_{key}"] = py_ope128
        arrays[f"cpp_flow_{key}"] = cpp_flow
        arrays[f"py_flow_c64_{key}"] = py_flow64
        arrays[f"py_flow_c128_{key}"] = py_flow128

        flow64_relative = result["flow_comparisons"][key]["complex64"]["relative_l2"]
        ope64_relative = result["ope_comparisons"][key]["pyqcd_complex64"]["relative_l2"]
        ope128_relative = result["ope_comparisons"][key]["pyqcd_complex128"]["relative_l2"]
        add_check(f"flow_complex64_relative_l2_tau{key}", flow64_relative < 1.0e-5,
                  flow64_relative, 1.0e-5)
        add_check(f"ope_complex64_relative_l2_tau{key}", ope64_relative < 2.0e-5,
                  ope64_relative, 2.0e-5)
        add_check(f"ope_complex128_relative_l2_tau{key}", ope128_relative < 2.0e-5,
                  ope128_relative, 2.0e-5)

        if tau > 0.0:
            result["flow_to_quasi"][key] = _compare_flow_to_quasi(
                donghx_flow_to_quasi, py_match_one, cpp_ope, py_ope64, tau)
            observed = result["flow_to_quasi"][key]["difference"]["max_abs"]
            add_check(f"flow_to_quasi_max_abs_tau{key}", observed < 1.0e-12,
                      observed, 1.0e-12)
            native_relative = result["flow_to_quasi"][key][
                "native_pipeline_difference"]["relative_l2"]
            add_check(
                f"native_flow_to_quasi_relative_l2_tau{key}",
                native_relative < 1.0e-5, native_relative, 1.0e-5)

        result["tmd"][key] = _tmd_collinear_reduction(
            gauge, tau, epsilon, shape)
        tmd64 = result["tmd"][key]["b0_L0_vs_straight_complex64"]["max_abs"]
        tmd128 = result["tmd"][key]["b0_L0_vs_straight_complex128"]["max_abs"]
        add_check(f"tmd_collinear_complex64_max_abs_tau{key}", tmd64 < 1.0e-5,
                  tmd64, 1.0e-5)
        add_check(f"tmd_collinear_complex128_max_abs_tau{key}", tmd128 < 1.0e-12,
                  tmd128, 1.0e-12)

    result["matching_kernel"] = _compare_matching_kernel(donghx_matching)
    matching_difference = result["matching_kernel"]["difference"]["max_abs"]
    add_check("matching_kernel_max_abs", matching_difference < 1.0e-12,
              matching_difference, 1.0e-12)

    positive_keys = [f"{tau:.3f}" for tau in taus if tau > 0.0]
    fourier_source_tau = positive_keys[0] if positive_keys else f"{taus[0]:.3f}"
    result["fourier_source_tau"] = fourier_source_tau
    h_z = arrays[f"cpp_ope_{fourier_source_tau}"][1, 0, 0, 0, :, 0].real
    result["fourier"] = _compare_fourier(donghx_fourier, h_z)
    fourier_difference = result["fourier"]["same_hbar"]["max_abs"]
    add_check("fourier_same_hbar_max_abs", fourier_difference < 1.0e-12,
              fourier_difference, 1.0e-12)

    result["ratio_convention"] = _compare_ratio_convention(donghx_ratio)
    ratio_c3_difference = result["ratio_convention"]["central_c3"]["max_abs"]
    add_check("ratio_central_c3_max_abs", ratio_c3_difference < 1.0e-12,
              ratio_c3_difference, 1.0e-12)
    result["ratio_convention"]["conventions_differ_observably"] = bool(
        result["ratio_convention"]["reported_ratio"]["max_abs"] > 1.0e-8)
    ratio_difference = result["ratio_convention"]["reported_ratio"]["max_abs"]
    add_check("ratio_conventions_observably_different", ratio_difference > 1.0e-8,
              ratio_difference, 1.0e-8)

    result["checks"] = {
        "required_count": len(required_checks),
        "passed_count": int(sum(bool(value) for value in required_checks)),
        "all_required_passed": bool(all(required_checks)),
        "details": check_details,
    }
    result["overall_status"] = (
        "pass" if result["checks"]["all_required_passed"] else "fail")
    result["scope"] = {
        "compared": [
            "native donghx build and self-test",
            "4D Wilson-flow links",
            "straight Wilson-line gluon OPE",
            "GF-to-MSbar endpoint conversion",
            "gg matching kernel",
            "unpolarized cosine Fourier",
            "disconnected-ratio denominator convention",
            "PyQCD TMD collinear b=0,L=0 reduction",
        ],
        "not_comparable": [
            "finite-b transverse staple OPE: no donghx implementation",
            "nucleon gluon C3/C2 from real configurations: no common ensemble/C2 product",
            "soft/rapidity and complete TMD matching: absent from the donghx TMD scope",
        ],
    }

    np.savez_compressed(out_dir / "arrays.npz", **arrays)
    (out_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(_render_report(result), encoding="utf-8")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="default: pyqcd/testing/tmd/artifacts/donghx_cross_reference_<timestamp>")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--taus", default=",".join(str(value) for value in DEFAULT_TAUS),
        help="comma-separated non-negative flow times")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    taus = tuple(float(value) for value in args.taus.split(",") if value)
    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ARTIFACT_ROOT / f"donghx_cross_reference_{stamp}"
    result = run(out_dir, taus=taus, epsilon=args.epsilon, seed=args.seed)
    print(json.dumps({
        "status": result["overall_status"],
        "out_dir": str(out_dir.resolve()),
        "checks": result["checks"],
        "tau_values": result["tau_values"],
        "epsilon": result["epsilon"],
        "input_sha256": result["input_sha256"],
    }, indent=2))
    return 0 if result["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
