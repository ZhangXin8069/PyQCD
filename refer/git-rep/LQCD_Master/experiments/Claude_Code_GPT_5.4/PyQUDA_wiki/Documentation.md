- [Module `pyquda`](#module-pyquda)
  - [Initialize MPI environment and QUDA](#initialize-mpi-environment-and-quda)
  - [Get the MPI and backend configuration](#get-the-mpi-and-backend-configuration)
  - [Print information, warnings, and errors with only one process by logger](#print-information-warnings-and-errors-with-only-one-process-by-logger)
  - [Use QUDA's functions directly or get QUDA's parameter struct](#use-qudas-functions-directly-or-get-qudas-parameter-struct)
  - [Use QUDA's enum](#use-qudas-enum)
- [Module `pyquda.field`](#module-pyqudafield)
  - [Defining a lattice with class `LatticeInfo`](#defining-a-lattice-with-class-latticeinfo)
  - [Working with lattice fields](#working-with-lattice-fields)
  - [Data format in the lattice field](#data-format-in-the-lattice-field)
  - [HDF5 I/O](#hdf5-io)
  - [Read/write fields with I/O functions](#readwrite-fields-with-io-functions)
  - [Pure gauge functions bound to `LatticeGauge`](#pure-gauge-functions-bound-to-latticegauge)
- [Module `pyquda_utils.core`](#module-pyquda_utilscore)
  - [Initialization and re-exported symbols](#initialization-and-re-exported-symbols)
  - [Build Dirac operators](#build-dirac-operators)
  - [Build sources](#build-sources)
  - [Apply source phases](#apply-source-phases)
  - [Invert sources and propagators](#invert-sources-and-propagators)
  - [MPI gather-scatter helpers](#mpi-gather-scatter-helpers)
  - [Deprecated compatibility layer](#deprecated-compatibility-layer)
- [Module `pyquda_plugins.pycontract`](#module-pyquda_pluginspycontract)
  - [Initialize the plugin](#initialize-the-plugin)
  - [Meson and baryon contractions](#meson-and-baryon-contractions)

## Module `pyquda`

This module handles MPI configuration, backend selection, device initialization, and the QUDA library setup.

### Initialize MPI environment and QUDA
The `init()` function initializes the QUDA library to perform lattice QCD calculations on GPUs. You should set the grid to let QUDA know how to split the lattice. The following code defines a grid with size `Gx, Gy, Gz, Gt = 1, 1, 1, 2`, which means we use 2 GPUs to split the lattice into 2 parts in the t direction.
```python
from pyquda import init

init([1, 1, 1, 2])
```
`mpiexec -n 2` is needed to run the code like this. The number of MPI processes should be equal to the grid volume. The default grid is `[1, 1, 1, 1]`, and you can suppress it when you are using only 1 GPU to perform the calculation.

The `init()` function accepts many keyword arguments to configure the QUDA runtime:
```python
init(
    grid_size=[1, 1, 1, 2],
    latt_size=[4, 4, 4, 8],      # set the default lattice
    backend="cupy",                # "cupy", "torch", "numpy", or "dpnp"
    resource_path=".cache",        # QUDA tuning cache directory
    enable_mps=False,              # enable CUDA MPS
    enable_gdr=False,              # enable GPUDirect RDMA
)
```

**CAUTION: Initialization should be performed before any operation in PyQUDA.**

### Get the MPI and backend configuration
There are functions to get the MPI and backend configuration:
- `getMPIComm()` — returns the MPI communicator
- `getMPISize()` — returns the total number of MPI processes
- `getMPIRank()` — returns the rank of the current process
- `getGridSize()` — returns the grid size `[Gx, Gy, Gz, Gt]`
- `getGridCoord()` — returns the grid coordinate of the current process
- `getGridRanks()` — returns the mapping from grid coordinate to MPI rank
- `getArrayBackend()` — returns the backend name (`"cupy"`, `"torch"`, `"numpy"`, `"dpnp"`)
- `getArrayBackendTarget()` — returns the backend target (`"cuda"`, `"hip"`, `"cpu"`, etc.)
- `getArrayDevice()` — returns the current device index

### Print information, warnings, and errors with only one process by logger
`pyquda` provides a logger to make sure output or error will only be printed once. The logging level is the same as in `logging` package.
```python
from pyquda import getLogger

logger = getLogger()
logger.debug("This is DEBUG")
logger.info("This is INFO")
logger.warning("This is WARNING", RuntimeWarning)
logger.error("This is ERROR", RuntimeError)
logger.critical("This is CRITICAL", RuntimeError)
```

### Use QUDA's functions directly or get QUDA's parameter struct

QUDA's functions and parameter struct are stored in submodule `pyquda.pyquda`, which is the main file to wrap up the QUDA functions. You **should not** call functions from this module directly unless you know what you are doing. QUDA parameter structs are also defined here, and it's not recommended to make such a struct by yourself.
```python
from pyquda import QudaInvertParam

inv_param = QudaInvertParam()
```

### Use QUDA's enum

QUDA's enums are stored in submodule `pyquda.enum_quda`, which is basically a translation of QUDA's `enum_quda.h`.
```python
from pyquda.enum_quda import QudaInverterType

inv_param.inv_type = QudaInverterType.QUDA_CG_INVERTER
```

## Module `pyquda.field`

This module defines the classes to store process-specific data on GPUs. Many pure gauge functions are bound to the `LatticeGauge` class for convenience.

### Defining a lattice with class `LatticeInfo`

This class handles the information to construct a lattice with multiple processes. It defines the size of the lattice and the grid to split the lattice. It also handles the extra information to define the lattice (the t-boundary and the anisotropy).

For example, the code below defines a lattice with the global size `Lx, Ly, Lz, Lt = 4, 4, 4, 8`. The lattice is anti-periodic on the t-boundary and isotropic.
```python
from pyquda.field import LatticeInfo

latt_info = LatticeInfo([4, 4, 4, 8], t_boundary=-1, anisotropy=1.0)
```

A default lattice can be set during `init()`:
```python
from pyquda import init

init([1, 1, 1, 2], latt_size=[4, 4, 4, 8])
```

### Working with lattice fields

Once we get a `LatticeInfo` instance, we can create lattice field objects. The `LatticeInfo` instance is saved in the `latt_info` attribute.
```python
from pyquda.field import LatticeGauge, LatticePropagator

gauge = LatticeGauge(latt_info)
propagator = LatticePropagator(latt_info)
```

Data stored in fields can be accessed via the `data` attribute. Use `getHost()` to materialize a host-side copy, or `copy()` to deep-copy the entire field object.
```python
gauge_data = gauge.data
gauge_host_copy = gauge.getHost()
gauge_copy = gauge.copy()
```

`location` identifies where the data currently resides. `toDevice()` and `toHost()` transfer data between host and device memory.
```python
print(gauge.location)  # cupy
gauge.toHost()
print(gauge.location)  # numpy
gauge.toDevice()
print(gauge.location)  # cupy
```

All data is stored in even-odd preconditioned format. `lexico()` returns a `numpy.ndarray` copy of the data without even-odd preconditioning.
```python
print(gauge.data.shape)        # (Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc)
print(gauge.lexico().shape)    # (Nd, Lt, Lz, Ly, Lx, Nc, Nc)
print(propagator.data.shape)   # (2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc)
print(propagator.lexico().shape)  # (Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc)
```

For `LatticeGauge`, you can also access individual directions and use `even`/`odd` parity components:
```python
gauge_x = gauge[0]        # LatticeLink for the x-direction
gauge_even = gauge.even   # even sites
gauge_odd = gauge.odd     # odd sites
```

### Data format in the lattice field

Data in the `data` attribute can be a `numpy.ndarray`, `cupy.ndarray`, or `torch.Tensor` depending on the backend. The dtype is `complex128` in most field types and `float64` in momentum and clover fields. All Dirac indices use the DeGrand-Rossi basis.

The shapes of `data` in different fields:
- `LatticeGauge.data` — `(Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc)`, row-column order, complex
- `LatticeMom.data` — `(Nd, 2, Lt, Lz, Ly, Lx // 2, 10)`, lower triangle (6) + diagonal (3) + reserved (1), real
- `LatticeClover.data` — `(2, Lt, Lz, Ly, Lx // 2, 2, ((Ns // 2) * Nc) ** 2)`, upper-left + lower-right block, diagonal (6) + lower triangle (30), real
- `LatticeRotation.data` — `(1, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc)`, complex
- `LatticeFermion.data` — `(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)`, complex
- `LatticePropagator.data` — `(2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc)`, sink-source order, complex
- `LatticeStaggeredFermion.data` — `(2, Lt, Lz, Ly, Lx // 2, Nc)`, complex
- `LatticeStaggeredPropagator.data` — `(2, Lt, Lz, Ly, Lx // 2, Nc, Nc)`, sink-source order, complex

### HDF5 I/O

Lattice fields support HDF5 I/O via `h5py` (requires `h5py` to be installed):
```python
gauge.saveH5("gauge.h5")
gauge_loaded = LatticeGauge.loadH5("gauge.h5")

# Append to existing HDF5 file (e.g., accumulating configurations)
gauge.appendH5("trajectory.h5")

# Save in single precision for smaller file size
gauge.saveH5("gauge_fp32.h5", use_fp32=True)
```

### Read/write fields with I/O functions

The I/O functions are defined in the `pyquda_utils.io` module (`from pyquda_utils import io`).

For example, we can read a gauge configuration generated by Chroma in `.lime` format:
```python
from pyquda_utils import io

gauge = io.readChromaQIOGauge("weak_field.lime")
```

The full list of supported I/O operations:

**Chroma QIO (LIME)** — read only (write is not supported)
- `readChromaQIOGauge(filename, checksum=True, reunitarize_sigma=1e-6)`
- `readChromaQIOPropagator(filename, checksum=True)`
- `readChromaQIOStaggeredPropagator(filename, checksum=True)`

**ILDG** — read only
- `readILDGGauge(filename, checksum=True, reunitarize_sigma=1e-6)`
- `readILDGBinGauge(filename, dtype, latt_size)`

**MILC** — read/write gauge, read propagator
- `readMILCGauge(filename, checksum=True, reunitarize_sigma=1e-6)`
- `writeMILCGauge(filename, gauge)`
- `readMILCQIOPropagator(filename)`
- `readMILCQIOStaggeredPropagator(filename)`

**NERSC** — read/write
- `readNERSCGauge(filename, checksum=True, plaquette=True, link_trace=True, reunitarize_sigma=1e-6)`
- `writeNERSCGauge(filename, gauge, use_fp32=False, ensemble_id="PyQUDA", ensemble_label="", sequence_number=0)`

**OpenQCD** — read/write
- `readOpenQCDGauge(filename, plaquette=True)`
- `writeOpenQCDGauge(filename, gauge)`

**KYU** — read/write
- `readKYUGauge(filename, latt_size)`
- `writeKYUGauge(filename, gauge)`
- `readKYUPropagator(filename, latt_size)`
- `writeKYUPropagator(filename, propagator)`

**XQCD** — read/write
- `readXQCDPropagator(filename, latt_size)`
- `readXQCDStaggeredPropagator(filename, latt_size)`
- `writeXQCDPropagator(filename, propagator)`
- `writeXQCDStaggeredPropagator(filename, propagator)`

**NPY (NumPy)** — read/write
- `readNPYGauge(filename)`
- `writeNPYGauge(filename, gauge)`
- `readNPYPropagator(filename)`
- `writeNPYPropagator(filename, propagator)`

**Gamma basis conversion**
- `rotateToDiracPauli(propagator)` — convert from DeGrand-Rossi to Dirac-Pauli basis
- `rotateToDeGrandRossi(propagator)` — convert from Dirac-Pauli to DeGrand-Rossi basis

### Pure gauge functions bound to `LatticeGauge`

The following methods are bound to the `LatticeGauge` class:

**Covariant operations:**

- `covDev(x, covdev_mu)`
  - Applies the covariant derivative on `x` in direction `covdev_mu`. `x` should be `LatticeFermion`. 0/1/2/3 represent +x/+y/+z/+t and 4/5/6/7 represent -x/-y/-z/-t. The covariant derivative is defined as $\psi'(x)=U_\mu(x)\psi(x+\hat{\mu})$.

- `laplace(x, laplace3D)`
  - Applies the Laplacian operator on `x`, and `laplace3D` takes 3 or 4 to apply Laplacian on spatial or all directions. `x` should be `LatticeStaggeredFermion`. The Laplacian operator is defined as $\psi'(x)=\frac{1}{N_\mathrm{Lap}}\sum_\mu\psi(x)-\dfrac{1}{2}\left[U_\mu(x)\psi(x+\hat{\mu})+U_\mu^\dagger(x-\hat{\mu})\psi(x-\hat{\mu})\right]$

- `wuppertalSmear(x, n_steps, alpha)`
  - Applies Wuppertal (Gaussian) smearing to `x` (`LatticeFermion` or `LatticeStaggeredFermion`). `n_steps` is the number of smearing steps and `alpha` is the smearing strength.

**Gauge field utilities:**

- `staggeredPhase(applied)`
  - Applies or removes the staggered phase from the gauge field. `applied` is a boolean indicating the current state. The convention is controlled by `LatticeGauge.pure_gauge.gauge_param.staggered_phase_type`, which defaults to the MILC convention.

- `projectSU3(tol)`
  - Projects the gauge field onto SU(3). `tol` is the tolerance for deviation from SU(3). `2e-15` (10× fp64 epsilon) is a good choice.

- `setAntiPeriodicT()`
  - Applies anti-periodic boundary conditions in the temporal direction.

- `setAnisotropy()`
  - Applies the anisotropy factor to temporal links.

**Path and loop operations:**

- `path(paths)`
  - `paths` is a list of integer direction sequences defining gauge-transported paths.

- `loop(loops, coeff)`
  - `loops` is a list of length 4: `[loops_x, loops_y, loops_z, loops_t]`. All `loops_*` should have the same shape. `coeff` is a list of coefficients.

- `loopTrace(loops)`
  - Returns the traces of all loops as a `numpy.ndarray` of dtype `complex128`. Each element is $\sum_x\mathrm{Tr}\,W(x)$.

**Gauge smearing:**

- `apeSmear(n_steps, alpha, dir_ignore)`
  - APE smearing. `alpha` is the smearing strength. `dir_ignore` is the direction to exclude (-1 for none, 3 for spatial-only).

- `apeSmearChroma(n_steps, factor, dir_ignore)`
  - A variant of `apeSmear()` with Chroma's convention for the `factor` parameter.

- `stoutSmear(n_steps, rho, dir_ignore)`
  - Stout smearing. `rho` is the smearing strength.

- `hypSmear(n_steps, alpha1, alpha2, alpha3, dir_ignore)`
  - HYP smearing. `alpha1/alpha2/alpha3` are the smearing strengths on levels 3/2/1.

**Gradient flow:**

- `wilsonFlow(n_steps, epsilon)`
  - Wilson flow. Returns the energy (all, spatial, temporal) for each step.

- `wilsonFlowScale(max_steps, epsilon)`
  - Returns $t_0$ and $w_0$ with up to `max_steps` Wilson flow steps.

- `wilsonFlowChroma(n_steps, time)`
  - Wilson flow using total `time` instead of per-step `epsilon`.

- `symanzikFlow(n_steps, epsilon)`
  - Symanzik flow. Returns the energy (all, spatial, temporal) for each step.

- `symanzikFlowScale(max_steps, epsilon)`
  - Returns $t_0$ and $w_0$ with up to `max_steps` Symanzik flow steps.

- `symanzikFlowChroma(n_steps, time)`
  - Symanzik flow using total `time` instead of per-step `epsilon`.

**Observables:**

- `plaquette()` — returns (all, spatial, temporal) plaquette
- `polyakovLoop()` — returns (real, imaginary) Polyakov loop
- `energy()` — returns (all, spatial, temporal) energy density
- `qcharge()` — returns the topological charge
- `qchargeDensity()` — returns the topological charge density array

**Random generation:**

- `gauss(seed, sigma)`
  - Fills the gauge field with random SU(3) matrices: $U = \exp(\sigma H)$, where $H$ is Gaussian-distributed su(3). `sigma=0` gives free field, `sigma=1` gives maximum disorder.

**Gauge fixing:**

- `fixingOVR(gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta)`
  - Gauge fixing with over-relaxation (supports multi-GPU).
    - `gauge_dir`: 3 for Coulomb, 4 for Landau
    - `relax_boost`: typical value 1.5 or 1.7
    - `stopWtheta`: 0 for MILC criterion, 1 for theta

- `fixingFFT(gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta)`
  - Gauge fixing with FFT (single GPU only).
    - `gauge_dir`: 3 for Coulomb, 4 for Landau
    - `alpha`: typical value 0.08
    - `autotune`: 1 to auto-tune alpha

## Module `pyquda_utils.core`

High-level helpers live in `pyquda_utils.core`, with source construction in `pyquda_utils.source`. Import with `from pyquda_utils import core, source`.

### Initialization and re-exported symbols

`pyquda_utils.core` re-exports the most common runtime and field symbols, so many workflows can stay inside one namespace:
```python
from pyquda_utils import core

core.init(resource_path=".cache/quda")
latt_info = core.LatticeInfo([4, 4, 4, 8], t_boundary=-1)
logger = core.getLogger()
```

The following groups are re-exported from `pyquda` / `pyquda.field`:
- Runtime helpers: `init`, `getMPIComm`, `getMPISize`, `getMPIRank`, `getGridSize`, `getGridCoord`, `getGridMap`, `getArrayBackend`, `getArrayDevice`, `getLogger`, `setLoggerLevel`
- Common field constants: `Ns`, `Nc`, `Nd`, `X`, `Y`, `Z`, `T`
- Common field classes: `LatticeInfo`, `LatticeGauge`, `LatticePropagator`, `LatticeStaggeredPropagator`, `LatticeFermion`, `LatticeStaggeredFermion`, and their `Multi*` variants

### Build Dirac operators

The recommended constructors now take a `LatticeInfo` directly. This replaces the older `getDslash()` / `getStaggeredDslash()` style that accepted a local lattice size.

```python
from pyquda_utils import core

latt_info = core.LatticeInfo([4, 4, 4, 8], t_boundary=-1, anisotropy=1.0)

wilson = core.getWilson(latt_info, mass=0.1, tol=1e-12, maxiter=1000)
clover = core.getClover(
    latt_info,
    mass=0.1,
    tol=1e-12,
    maxiter=1000,
    xi_0=1.0,
    clover_csw_t=1.2,
    clover_csw_r=1.2,
)
hisq = core.getHISQ(latt_info, mass=0.01, tol=1e-12, maxiter=1000, naik_epsilon=0.0)
```

Available constructors:
- `getDirac(latt_info, mass, tol, maxiter, xi_0=1.0, clover_coeff_t=0.0, clover_coeff_r=1.0, multigrid=None)`
- `getStaggeredDirac(latt_info, mass, tol, maxiter, tadpole_coeff=1.0, naik_epsilon=0.0)`
- `getWilson(latt_info, mass, tol, maxiter, multigrid=None)`
- `getClover(latt_info, mass, tol, maxiter, xi_0=1.0, clover_csw_t=..., clover_csw_r=..., multigrid=None)`
- `getStaggered(latt_info, mass, tol, maxiter, tadpole_coeff=1.0)`
- `getHISQ(latt_info, mass, tol, maxiter, naik_epsilon=0.0, multigrid=None)`

`multigrid` accepts either a geometric block-size list such as `[[2, 2, 2, 2], [4, 4, 4, 4]]` or a `Multigrid` object.

### Build sources

Source construction helpers live in `pyquda_utils.source`:
```python
from pyquda_utils import source

point = source.source(latt_info, "point", [0, 0, 0, 0], spin=0, color=0)
wall = source.source(latt_info, "wall", 4, spin=0, color=0)
volume = source.source(latt_info, "volume", None, spin=0, color=0)

prop_point = source.source12(latt_info, "point", [0, 0, 0, 0])
stag_point = source.source3(latt_info, "point", [0, 0, 0, 0])
```

Current source types for `source.source()`:
- `"point"`: `t_srce=[x, y, z, t]`
- `"wall"`: `t_srce=t`
- `"volume"`: `t_srce=None`
- `"momentum"`: wall or volume source multiplied by `source_phase`
- `"colorvector"`: wall or volume color-vector source from `source_phase`

Convenience constructors:
- `multiFermion()` / `multiStaggeredFermion()` build multi-RHS fermion collections
- `propagator()` / `staggeredPropagator()` build propagator-shaped sources
- `gaussianSmear()` applies Wuppertal smearing to fermions, propagators, and `Multi*` fields
- `sequential()`, `sequential12()`, and `sequential3()` keep only one sink time-slice

Passing a raw lattice-size list to `source()` / `source12()` / `source3()` is still accepted for compatibility, but it is deprecated. Prefer passing `LatticeInfo`.

### Apply source phases

`pyquda_utils.phase` defines reusable phase fields:
```python
from pyquda_utils.phase import MomentumPhase, GridPhase
from pyquda_utils import source

momentum_phase = MomentumPhase(latt_info).getPhase([1, 2, 3])
grid_phase = GridPhase(latt_info, stride=[2, 2, 2, 2]).getPhase([1, 1, 1, 1])

momentum_source = source.source12(latt_info, "wall", 4, source_phase=momentum_phase)
grid_source = source.source12(latt_info, "volume", None, source_phase=momentum_phase * grid_phase)
```

### Invert sources and propagators

The `core.invert*()` family is now the main high-level inversion interface:

```python
from pyquda_utils import core, source

dirac = core.getWilson(latt_info, mass=0.1, tol=1e-12, maxiter=1000)
dirac.loadGauge(gauge)

propag = core.invert(dirac, source_type="point", t_srce=[0, 0, 0, 0], mrhs=4)
seq_source = source.sequential12(propag, t_srce=4)
seq_propag = core.invertPropagator(dirac, seq_source, mrhs=4)
```

Available helpers:
- `invert(dirac, source_type, t_srce, source_phase=None, mrhs=1, restart=0) -> LatticePropagator`
- `invertEigenvector(dirac, t_srce, source_propag, mrhs=1, restart=0) -> MultiLatticeFermion`
- `invertSequential(dirac, source_propag, t_srce, mrhs=1, restart=0) -> LatticePropagator`
- `invertPropagator(dirac, source_propag, mrhs=1, restart=0) -> LatticePropagator`
- `invertStaggered(dirac, source_type, t_srce, source_phase=None, mrhs=1, restart=0) -> LatticeStaggeredPropagator`
- `invertStaggeredSequential(dirac, source_propag, t_srce, mrhs=1, restart=0) -> LatticeStaggeredPropagator`
- `invertStaggeredPropagator(dirac, source_propag, mrhs=1, restart=0) -> LatticeStaggeredPropagator`

`mrhs` controls multi-right-hand-side batching, and `restart` applies recursive residual correction through `invertMultiSrcRestart()`.

### MPI gather-scatter helpers

`pyquda_utils.core` also contains helpers for collecting lattice-shaped NumPy arrays over MPI:
- `gatherLattice2(data, tzyx, reduce_op="sum", root=0)`
- `scatterLattice(data_all, tzyx, root=0)`
- `gatherScatterLattice(data, tzyx, reduce_op="sum", root=0)`
- `gatherLattice(data, axes, reduce_op="sum", root=0)`

`gatherLattice2()` / `gatherScatterLattice()` support `sum`, `mean`, `prod`, `max`, and `min`.

### Deprecated compatibility layer

The following names are still re-exported from `pyquda_utils.core` for backward compatibility, but new code should avoid them:

| Deprecated name | Use instead |
| --- | --- |
| `getDslash()` | `getDirac()`, `getWilson()`, or `getClover()` |
| `getStaggeredDslash()` | `getStaggeredDirac()`, `getStaggered()`, or `getHISQ()` |
| `invert12()` | `invert()` or `invertPropagator()` |
| `smear()` | `LatticeGauge.stoutSmear()` |
| `smear4()` | `LatticeGauge.hypSmear()` or `LatticeGauge.stoutSmear()` |
| `cb2()` | `evenodd()` |

## Module `pyquda_plugins.pycontract`

The `pycontract` plugin ships a high-level Python wrapper around the generated Cython bindings. The canonical usage is reflected in [`tests/test_pycontract.py`](https://github.com/CLQCD/PyQUDA/blob/master/tests/test_pycontract.py).

### Initialize the plugin

Initialize PyQUDA first, then initialize the plugin:
```python
from pyquda_utils import core
from pyquda_plugins import pycontract

core.init(resource_path=".cache/quda")
pycontract.init()
```

Unlike the low-level generated binding, `pycontract.init()` does not take a device id. It reads the current device from the initialized PyQUDA runtime.

### Meson and baryon contractions

Typical usage:
```python
from pyquda_utils import core, gamma, io
from pyquda_plugins import pycontract

propag = io.readQIOPropagator("pt_prop_0")
propag.toDevice()

gamma_2 = gamma.Gamma(2)
gamma_4 = gamma.Gamma(8)
gamma_5 = gamma.Gamma(15)
C = gamma_2 @ gamma_4
CG_A = C @ gamma_4 @ gamma_5
CG_B = C @ gamma_5
Pp = (gamma.Gamma(0) + gamma_4) / 2

meson = pycontract.mesonTwoPoint(propag, propag, CG_A, CG_B)
all_sink = pycontract.mesonAllSinkTwoPoint(propag, propag, CG_B)
baryon = pycontract.baryonTwoPoint(
    propag,
    propag,
    propag,
    pycontract.BaryonContractType.IK_JL_NM,
    CG_A,
    CG_B,
    Pp,
)
```

Current high-level wrapper functions:
- `mesonTwoPoint(propag_a, propag_b, gamma_ab, gamma_dc) -> LatticeComplex`
- `mesonAllSinkTwoPoint(propag_a, propag_b, gamma_dc) -> MultiLatticeComplex`
- `mesonAllSourceTwoPoint(propag_a, propag_b, gamma_ab) -> MultiLatticeComplex`
- `baryonDiquark(propag_i, propag_j, gamma_ij, gamma_kl) -> LatticePropagator`
- `baryonTwoPoint(propag_i, propag_j, propag_n, contract_type, gamma_ij, gamma_kl, gamma_mn) -> LatticeComplex`
- `baryonGeneralTwoPoint(propag_i, propag_j, propag_n, contract_type, gamma_ij, gamma_kl, project_mn) -> LatticeComplex`
- `baryonSequentialTwoPoint(propag_i, propag_j, propag_n, contract_type, sequential_type, gamma_ij, gamma_kl, gamma_mn) -> LatticePropagator`
- `baryonTwoPoint_v2(propag_i, propag_j, propag_m, contract_type, gamma_ij, gamma_kl, gamma_mn) -> LatticeComplex`

`gamma_mn` can be either a `Gamma` or a `Projector`. The wrapper expands `Projector` values into the corresponding linear combination of `Gamma` contractions automatically.
