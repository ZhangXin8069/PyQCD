#!/usr/bin/env python3
"""
Create a PDF of momentum-shell coefficient diagnostics for LapH eigenvectors.

Input arrays use the axis order

    (Nev, Nz, Ny, Nx, Nc)

or, for several time slices/samples,

    (Nsamp, Nev, Nz, Ny, Nx, Nc).

For every selected integer-momentum shell P2 = nz^2 + ny^2 + nx^2, the PDF
contains one page with two panels:

  1. a coefficient-count histogram with a Gaussian MLE overlay and its
     centre (mu) and width (sigma);
  2. a scatter plot of coefficient versus eigenvector index, with horizontal
     lines at mu and mu +/- sigma.

Each dot is the color norm of one exact momentum direction:

    A_n(k) = sqrt(sum_c |v_tilde_n^c(k)|^2),

where

    v_tilde_n^c(k) = 1/sqrt(V) sum_x exp(-2*pi*i*k.x/L) v_n^c(x).

All exact momentum directions satisfying n^2=P2 are included. If a 6-D input
contains several samples, all samples are included in the distribution and
scatter plot.

Examples
--------
    python eigenvector_momentum_shells.py eigvecs.npy -o momentum_scatter.pdf

    python eigenvector_momentum_shells.py eigvecs.npz --key eigvecs \\
        --p2 0 1 4 9 16 -o momentum_scatter.pdf

    python eigenvector_momentum_shells.py eigvecs.npy \\
        --p2 0,1,4,9,16,25,36,49 --bins 60 -o momentum_scatter.pdf

The Fourier spectrum of gauge-covariant eigenvectors is gauge dependent. It
is a useful fixed-gauge diagnostic, but is not a gauge-invariant observable by
itself.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_P2 = (
    0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14,
    # 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 29,
    # 30, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 43,
    # 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57,
    # 59, 61, 62, 65, 66, 68, 70, 72, 73, 75, 76, 77,
    # 81, 86, 88, 97, 108
)


def _load_array(path: Path, key: str | None) -> np.ndarray:
    """Load a NPY/NPZ array without accepting pickle objects."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        # Memory-map ordinary NPY files, so a block rather than the entire
        # eigenvector basis is transformed at once.
        return np.load(path, mmap_mode="r", allow_pickle=False)

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key is None:
                if len(archive.files) != 1:
                    choices = ", ".join(archive.files)
                    raise ValueError(
                        f"{path} contains {len(archive.files)} arrays ({choices}); "
                        "select the eigenvector array with --key."
                    )
                key = archive.files[0]
            if key not in archive.files:
                choices = ", ".join(archive.files)
                raise KeyError(f"Key {key!r} is not in {path}; available: {choices}")
            return np.asarray(archive[key])

    raise ValueError(f"Unsupported input format {path.suffix!r}; use .npy or .npz.")


def load_samples(paths: Sequence[Path], key: str | None) -> list[np.ndarray]:
    """
    Return 5-D samples, each with canonical axis order (Nev,Z,Y,X,C).

    One 6-D input is interpreted as (Nsamp,Nev,Z,Y,X,C). Several 5-D input
    files are interpreted as several samples.
    """
    arrays = [_load_array(path, key) for path in paths]
    if len(arrays) == 1 and arrays[0].ndim == 6:
        samples = [arrays[0][sample] for sample in range(arrays[0].shape[0])]
    else:
        if any(array.ndim != 5 for array in arrays):
            found = ", ".join(str(array.shape) for array in arrays)
            raise ValueError(
                "Use either one 6-D array (Nsamp,Nev,Z,Y,X,C) or one/more "
                f"5-D arrays (Nev,Z,Y,X,C). Found: {found}"
            )
        samples = arrays

    if not samples:
        raise ValueError("No eigenvector samples were found.")
    reference = samples[0].shape
    if len(reference) != 5:
        raise ValueError(f"Expected 5-D samples, received shape {reference}.")
    if reference[-1] < 1:
        raise ValueError("The color axis C must contain at least one component.")
    if min(reference[1:4]) < 3:
        raise ValueError("Each spatial axis must contain at least three sites.")
    if reference[1] != reference[2] or reference[2] != reference[3]:
        raise ValueError(
            "The requested integer P2 shells assume a cubic spatial lattice "
            "(Nz = Ny = Nx)."
        )
    for sample_index, sample in enumerate(samples):
        if sample.shape != reference:
            raise ValueError(
                f"All samples must have the same shape. Sample 0: {reference}; "
                f"sample {sample_index}: {sample.shape}."
            )
    return samples


def parse_p2(tokens: Sequence[str]) -> tuple[int, ...]:
    """Accept --p2 0 1 4 or --p2 0,1,4 syntax."""
    result: list[int] = []
    for token in tokens:
        for item in token.split(","):
            item = item.strip()
            if not item:
                continue
            value = int(item)
            if value < 0:
                raise ValueError("P2 values must be non-negative integers.")
            if value not in result:
                result.append(value)
    if not result:
        raise ValueError("At least one P2 value is required.")
    return tuple(result)


def canonical_modes(length: int, cutoff: int) -> range:
    """
    Return non-aliased signed FFT modes with |n| <= cutoff.

    For even L, the Nyquist mode is represented only as -L/2. This avoids
    double-counting +L/2 and -L/2, which map to the same FFT coefficient.
    """
    half = length // 2
    lower = -half
    upper = half if length % 2 else half - 1
    return range(max(lower, -cutoff), min(upper, cutoff) + 1)


def shell_momenta(
    p2_values: Iterable[int],
    spatial_extent: int,
) -> dict[int, np.ndarray]:
    """
    Enumerate all non-aliased integer momenta with n^2=P2.

    The finite-lattice canonical momentum range is enforced. Therefore high
    P2 shells near the Brillouin-zone edge contain only genuinely distinct
    Fourier modes.
    """
    requested = tuple(p2_values)
    cutoff = int(np.ceil(np.sqrt(max(requested))))
    modes = canonical_modes(spatial_extent, cutoff)
    result: dict[int, list[tuple[int, int, int]]] = {
        value: [] for value in requested
    }
    for nz in modes:
        for ny in modes:
            for nx in modes:
                norm2 = nz * nz + ny * ny + nx * nx
                if norm2 in result:
                    result[norm2].append((nz, ny, nx))

    missing = [value for value, momenta in result.items() if not momenta]
    if missing:
        joined = ", ".join(str(value) for value in missing)
        raise ValueError(
            f"No non-aliased lattice momentum belongs to P2={joined} for "
            f"L={spatial_extent}."
        )
    return {
        value: np.asarray(sorted(momenta), dtype=np.int64)
        for value, momenta in result.items()
    }


def selected_fourier_coefficients(
    samples: Sequence[np.ndarray],
    momenta: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """
    Extract selected spatial Fourier coefficients using an orthonormal DFT.

    The result has shape (Nsamp,Nev,Nmom,Nc). The transform uses
    exp(-2*pi*i*k.x/L)/sqrt(V).
    """
    if block_size < 1:
        raise ValueError("--block-size must be at least one.")

    nev, nz, ny, nx, nc = samples[0].shape
    coefficients = np.empty(
        (len(samples), nev, len(momenta), nc), dtype=np.complex128
    )
    fft_indices = [
        (int(nz_mode % nz), int(ny_mode % ny), int(nx_mode % nx))
        for nz_mode, ny_mode, nx_mode in momenta
    ]
    for sample_index, sample in enumerate(samples):
        for first in range(0, nev, block_size):
            last = min(first + block_size, nev)
            transformed = np.fft.fftn(
                np.asarray(sample[first:last]), axes=(1, 2, 3), norm="ortho"
            )
            for momentum_index, (iz, iy, ix) in enumerate(fft_indices):
                coefficients[sample_index, first:last, momentum_index, :] = (
                    transformed[:, iz, iy, ix, :]
                )
    return coefficients


def split_shells(
    coefficients: np.ndarray,
    momenta_by_p2: dict[int, np.ndarray],
    p2_values: Sequence[int],
) -> dict[int, dict[str, np.ndarray]]:
    """Return color-norm coefficients grouped by the requested P2 values."""
    result: dict[int, dict[str, np.ndarray]] = {}
    offset = 0
    for p2 in p2_values:
        momenta = momenta_by_p2[p2]
        count = len(momenta)
        complex_values = coefficients[:, :, offset : offset + count, :]
        amplitudes = np.sqrt(np.sum(np.abs(complex_values) ** 2, axis=-1))
        result[p2] = {"momenta": momenta, "amplitudes": amplitudes}
        offset += count
    return result


def gaussian_parameters(values: np.ndarray) -> tuple[float, float]:
    """Normal maximum-likelihood estimates: centre mu and width sigma."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite Fourier coefficients are available.")
    return float(np.mean(finite)), float(np.std(finite, ddof=0))


def _histogram_edges(values: np.ndarray, bins: int) -> np.ndarray:
    """Build finite, uniformly spaced histogram edges even for sigma=0."""
    lower = float(np.min(values))
    upper = float(np.max(values))
    if np.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 1.0e-12)
        lower -= padding
        upper += padding
    return np.linspace(lower, upper, bins + 1)


def draw_page(
    p2: int,
    shell: dict[str, np.ndarray],
    output: PdfPages,
    bins: int,
) -> None:
    """Add one P2 page with a distribution histogram and a scatter plot."""
    amplitudes = shell["amplitudes"]  # (Nsamp, Nev, exact momentum degeneracy)
    values = amplitudes[np.isfinite(amplitudes)]
    mu, sigma = gaussian_parameters(values)
    _, nev, degeneracy = amplitudes.shape
    x_scatter = np.broadcast_to(
        np.arange(nev)[None, :, None], amplitudes.shape
    ).ravel()
    y_scatter = amplitudes.ravel()

    fig, (hist_ax, scatter_ax) = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.suptitle(
        rf"Eigenvector Fourier coefficients: $P^2=\mathbf{{n}}^2={p2}$"
        f" ({degeneracy} exact momentum directions)",
        fontsize=14,
        y=0.97,
    )

    edges = _histogram_edges(values, bins)
    hist_ax.hist(
        values,
        bins=edges,
        color="#5B8DB8",
        edgecolor="white",
        linewidth=0.35,
        alpha=0.90,
        label="coefficient count",
    )
    if sigma > 0.0:
        x_curve = np.linspace(edges[0], edges[-1], 500)
        bin_width = edges[1] - edges[0]
        gaussian_count = (
            values.size
            * bin_width
            * np.exp(-0.5 * ((x_curve - mu) / sigma) ** 2)
            / (sigma * np.sqrt(2.0 * np.pi))
        )
        hist_ax.plot(
            x_curve,
            gaussian_count,
            color="#B22222",
            linewidth=1.7,
            label="Gaussian MLE",
        )
    parameter_text = (
        "Gaussian fit\n"
        rf"$\mu={mu:.5e}$" "\n"
        rf"$\sigma={sigma:.5e}$" "\n"
        rf"$N_{{\rm coeff}}={values.size}$"
    )
    hist_ax.text(
        0.97,
        0.96,
        parameter_text,
        transform=hist_ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92},
    )
    hist_ax.set_xlabel(r"coefficient $A_n(\mathbf{k})$")
    hist_ax.set_ylabel("number of eigenvector coefficients")
    hist_ax.set_title("Coefficient distribution")
    hist_ax.grid(alpha=0.22)
    hist_ax.legend(loc="best", fontsize=9)

    scatter_ax.scatter(
        x_scatter,
        y_scatter,
        s=7,
        alpha=0.27,
        linewidths=0,
        color="#2166AC",
        rasterized=True,
        label="all exact directions / samples",
    )
    scatter_ax.axhline(mu, color="black", linewidth=1.45, label=rf"$\mu={mu:.5e}$")
    if sigma > 0.0:
        scatter_ax.axhline(
            mu + sigma,
            color="#B22222",
            linewidth=1.20,
            linestyle="--",
            label=rf"$\mu+\sigma={mu + sigma:.5e}$",
        )
        scatter_ax.axhline(
            mu - sigma,
            color="#B22222",
            linewidth=1.20,
            linestyle="--",
            label=rf"$\mu-\sigma={mu - sigma:.5e}$",
        )
    scatter_ax.set_xlabel("eigenvector index")
    scatter_ax.set_ylabel(r"coefficient $A_n(\mathbf{k})$")
    scatter_ax.set_title("Coefficient scatter")
    scatter_ax.grid(alpha=0.22)
    scatter_ax.legend(loc="best", fontsize=8)

    fig.text(
        0.5,
        0.035,
        "Each point is a color-norm coefficient from one exact momentum "
        r"direction with $\mathbf{n}^2=P^2$.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.10, 0.98, 0.93))
    output.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def create_pdf(
    paths: Sequence[Path],
    output_path: Path,
    p2_values: Sequence[int],
    key: str | None,
    block_size: int,
    bins: int,
    N: int,
) -> None:
    """Run the Fourier projection and write one PDF page for each P2 shell."""
    if bins < 2:
        raise ValueError("--bins must be at least two.")
    samples = [load_samples(paths, key)[0][:N]]
    spatial_extent = samples[0].shape[1]
    momenta_by_p2 = shell_momenta(p2_values, spatial_extent)
    all_momenta = np.concatenate(
        [momenta_by_p2[p2] for p2 in p2_values], axis=0
    )
    coefficients = selected_fourier_coefficients(samples, all_momenta, block_size)
    shells = split_shells(coefficients, momenta_by_p2, p2_values)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "Eigenvector momentum-shell coefficient diagnostics"
        metadata["Subject"] = (
            "Gaussian coefficient distributions and eigenvector-index scatter plots"
        )
        metadata["Keywords"] = "LapH, distillation, eigenvector, Fourier, momentum"
        for p2 in p2_values:
            draw_page(p2, shells[p2], pdf, bins)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a PDF with one coefficient histogram/scatter page per "
            "integer P2 momentum shell."
        )
    )
    parser.add_argument(
        "input",
        nargs="+",
        type=Path,
        help=(
            "One 5-D/6-D .npy/.npz array, or several 5-D sample files. "
            "Unquoted shell globs are expanded by the shell."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("momentum_coefficient_scatter.pdf"),
        help="Output PDF filename.",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Array key for an NPZ file containing more than one array.",
    )
    parser.add_argument(
        "--p2",
        nargs="+",
        default=[str(value) for value in DEFAULT_P2],
        help=(
            "Integer P2 shell values, as space-separated or comma-separated "
            f"numbers (default: {','.join(str(value) for value in DEFAULT_P2)})."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of bins in each coefficient distribution (default: 50).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Number of eigenvectors transformed per FFT block (default: 16).",
    )
    
    parser.add_argument(
        "--vector",
        type=int,
        default=500,
        help="Number of eigenvectors transformed per FFT block (default: 16).",
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p2_values = parse_p2(args.p2)
    create_pdf(
        paths=args.input,
        output_path=args.output,
        p2_values=p2_values,
        key=args.key,
        block_size=args.block_size,
        bins=args.bins,
        N = args.vector
    )
    print(f"PDF created: {args.output}")
    print(f"P2 pages: {', '.join(str(value) for value in p2_values)}")


if __name__ == "__main__":
    main()
