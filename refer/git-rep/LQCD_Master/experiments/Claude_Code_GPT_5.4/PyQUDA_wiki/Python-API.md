# Python API Reference

> This page lists the public Python API of the core runtime (`pyquda`, `pyquda_comm`) and the most common high-level helper module (`pyquda_utils.core`).
> For extended usage examples, see [Documentation](Documentation).

---

## Table of Contents

- [pyquda_comm](#pyquda_comm)
  - [Grid & Device Initialization](#grid--device-initialization)
  - [State Query](#state-query)
  - [Grid Utilities](#grid-utilities)
  - [MPI I/O](#mpi-io)
  - [File Header Utilities](#file-header-utilities)
- [pyquda_comm.field](#pyquda_commfield)
  - [Data Layout Functions](#data-layout-functions)
  - [LatticeInfo & LexicoInfo](#latticeinfo--lexicoinfo)
  - [BaseField (Common Interface)](#basefield-common-interface)
  - [Field Class Hierarchy](#field-class-hierarchy)
  - [LatticeGauge](#latticegauge)
  - [LatticeRotation](#latticerotation)
  - [LatticeMom](#latticemom)
  - [LatticePropagator & LatticeStaggeredPropagator](#latticepropagator--latticestaggeredpropagator)
  - [Other Field Types](#other-field-types)
- [pyquda](#pyquda-1)
  - [Initialization](#initialization)
  - [QUDA State](#quda-state)
  - [Missing Legacy Entries](#missing-legacy-entries)
- [pyquda_utils.core](#pyquda_utilscore)
  - [Re-exported Symbols](#re-exported-symbols)
  - [Dirac Constructors](#dirac-constructors)
  - [Inversion Helpers](#inversion-helpers)
  - [MPI Lattice Helpers](#mpi-lattice-helpers)
  - [Deprecated Aliases](#deprecated-aliases)

---

## `pyquda_comm`

### Grid & Device Initialization

```python
def initGrid(
    grid_map: GridMapType = "default",
    grid_size: Optional[Sequence[int]] = None,
    latt_size: Optional[Sequence[int]] = None,
    evenodd: bool = True,
)
```
Initialize the MPI grid topology. `GridMapType` can be `"default"`, `"t_major"`, `"x_major"`, `"minimize"`, `"shared"`, or `"dist_graph"`.

```python
def initDevice(
    backend: BackendType = "numpy",
    backend_target: BackendTargetType = "cuda",
    device: int = -1,
    enable_mps: bool = False,
)
```
Initialize the compute backend and GPU device. `BackendType` can be `"numpy"`, `"cupy"`, `"torch"`, or `"dpnp"`.

### State Query

| Function | Return Type | Description |
|---|---|---|
| `isGridInitialized()` | `bool` | Check whether the MPI grid has been initialized |
| `isDeviceInitialized()` | `bool` | Check whether the compute device has been initialized |
| `getLogger()` | `_MPILogger` | Get the MPI-aware logger |
| `setLoggerLevel(level)` | `None` | `level`: `"debug"`, `"info"`, `"warning"`, `"error"`, `"critical"` |
| `getMPIComm()` | `MPI.Intracomm` | Return the MPI communicator |
| `getMPISize()` | `int` | Return the total number of MPI processes |
| `getMPIRank()` | `int` | Return the current MPI rank |
| `getGridMap()` | `GridMapType` | Return the current grid mapping type |
| `getGridSize()` | `List[int]` | Return the grid dimensions |
| `getGridCoord()` | `List[int]` | Return the grid coordinate of the current rank |
| `getGridRanks()` | `List[int]` | Return the rank-to-default-coordinate mapping table |
| `getArrayBackend()` | `BackendType` | Return the array backend (`"numpy"`, `"cupy"`, `"torch"`, `"dpnp"`) |
| `getArrayBackendTarget()` | `BackendTargetType` | Return the backend target device type |
| `getArrayDevice()` | `int` | Return the current GPU device index |

### Grid Utilities

```python
def getRankFromCoord(grid_coord: List[int]) -> int
def getCoordFromRank(mpi_rank: int) -> List[int]
def getNeighbourRank() -> List[int]
def getSublatticeSize(latt_size: Sequence[int], force_even: bool = True) -> List[int]
def getDefaultGrid(mpi_size: int, shared_size: int, latt_size: Sequence[int], evenodd: bool = True) -> Tuple
```

### MPI I/O

```python
def readMPIFile(filename: str, dtype: DTypeLike, offset: int, shape: Sequence[int], axes: Sequence[int]) -> NDArray
def readMPIFileInChunks(filename: str, dtype: DTypeLike, offset: int, count: int, shape: Sequence[int], axes: Sequence[int]) -> Generator[Tuple[int, NDArray], None, None]
def writeMPIFile(filename: str, dtype: DTypeLike, offset: int, shape: Sequence[int], axes: Sequence[int], buf: NDArray)
def writeMPIFileInChunks(filename: str, dtype: DTypeLike, offset: int, count: int, shape: Sequence[int], axes: Sequence[int], buf: NDArray) -> Generator
```

```python
def openReadHeader(filename: str) -> ContextManager[_FileWithOffset]
def openWriteHeader(filename: str, root: int = 0) -> ContextManager[_FileWithOffset]
```

### File Header Utilities

```python
def read_array_header(filename: str) -> Tuple[Tuple[int, ...], str, int]
def write_array_header(filename: str, shape: Tuple[int, ...], dtype: str) -> int
```
Read/write `.npy` file headers (for parallel MPI I/O).

---

## `pyquda_comm.field`

### Data Layout Functions

```python
def lexico(data: NDArray, axes: List[int], dtype=None) -> NDArray
```
Convert even-odd preconditioning data to lexicographic order. `axes` should have 5 elements `[parity, t, z, y, x]`.

```python
def evenodd(data: NDArray, axes: List[int], dtype=None) -> NDArray
```
Convert lexicographic data to even-odd preconditioning order. `axes` should have 4 elements `[t, z, y, x]`.

```python
def cb2(data: NDArray, axes: List[int], dtype=None) -> NDArray
```
**Deprecated**. Alias for `evenodd()`.

### LatticeInfo & LexicoInfo

```python
class LatticeInfo(BaseInfo):
    def __init__(
        self,
        latt_size: List[int],
        t_boundary: Literal[1, -1] = 1,
        anisotropy: float = 1.0,
        Ns: int = 4,
        Nc: int = 3,
    )
```

Key attributes:
- `grid_size`, `grid_coord` – MPI grid dimensions and current rank's coordinate
- `global_size`, `global_volume` – full lattice dimensions and volume
- `size`, `volume` – sublattice dimensions and volume
- `Lx`, `Ly`, `Lz`, `Lt` – sublattice extents
- `GLx`, `GLy`, `GLz`, `GLt` – global lattice extents
- `t_boundary` – temporal boundary condition (`1` or `-1`)
- `anisotropy` – anisotropy parameter $\xi$
- `Nd`, `Ns`, `Nc` – number of dimensions, spin, color

Key methods:
- `lexico(data, multi, backend="numpy")` – convert even-odd data to lexicographic order
- `evenodd(data, multi, backend="numpy")` – convert lexicographic data to even-odd order
- `coordinate(mu=None)` – get even-odd coordinate arrays

```python
class LexicoInfo(BaseInfo):
    def __init__(self, latt_size: Sequence[int], Ns: int = 4, Nc: int = 3)
```
For lexicographic-order lattice fields.

### BaseField (Common Interface)

All lattice field classes inherit from `BaseField`, which provides:

**Properties:**
| Property | Type | Description |
|---|---|---|
| `latt_info` | `BaseInfo` | Associated lattice info |
| `location` | `BackendType` | Current storage location (`"numpy"`, `"cupy"`, `"torch"`) |
| `data` | array | Field data (settable; auto-copies between host/device) |
| `data_ptr` | `NDArray` | Flattened data view |
| `data_ptrs` | `NDArray` | Flattened data view grouped by L5 |
| `data_void_ptr` | `Pointer` | C pointer to data |

**Methods:**
```python
# Persistence
@classmethod
def load(cls, filename: str) -> Self                            # Load from .npy file
def save(self, filename: str, *, use_fp32: bool = False)        # Save to .npy file

# HDF5 I/O
@classmethod
def loadH5(cls, filename: str, label, *, check=True) -> Self    # Load from HDF5
def saveH5(self, filename: str, label, *, annotation="", check=True, use_fp32=False)
def appendH5(self, filename: str, label, *, annotation="", check=True, use_fp32=False)
def updateH5(self, filename: str, label, *, annotation="", check=True, use_fp32=False)

# Data manipulation
def lexico(self, force_numpy: bool = True) -> NDArray           # Convert to lexicographic order
def copy(self) -> Self                                          # Deep copy
def toDevice(self)                                              # Move data to GPU
def toHost(self)                                                # Move data to CPU
def getHost(self) -> NDArray                                    # Get CPU copy
def norm2(self, all_reduce=True) -> float                       # L2 norm squared

# Arithmetic: +, -, *, /, unary -, +=, -=, *=, /=
# Indexing: __getitem__, __setitem__ (by global coordinates)
```

`FullField` (even-odd fields) adds:
```python
@property
def even -> Field       # Access even-site component
@property
def odd -> Field        # Access odd-site component
def shift(self, n: int, mu: int) -> Self    # Shift by n sites along direction mu (with MPI)
```

`MultiField` (L5-extended fields) adds:
```python
def __getitem__(self, key: int) -> Field    # Access single component
def __setitem__(self, key, value: Field)    # Set single component
```

### Field Class Hierarchy

| Class | Base | Shape (per site) | Description |
|---|---|---|---|
| `LatticeInt` | `FullField` | `1` | Integer field |
| `LatticeReal` | `FullField` | `1` | Real field |
| `LatticeComplex` | `FullField` | `1` | Complex field |
| `LatticeLink` | `FullField` | `Nc × Nc` | Single SU(3) link field (unit matrix default) |
| `LatticeGauge` | `MultiField[LatticeLink]` | `Nd × Nc × Nc` | Full gauge field |
| `LatticeRotation` | `MultiField[LatticeLink]` | `1 × Nc × Nc` | Rotation matrix field (L5=1) |
| `LatticeMom` | `MultiField[FullField]` | `Nd × Nc × Nc` | Conjugate momentum (for HMC) |
| `LatticeClover` | `FullField` | clover | Clover term field |
| `HalfLatticeFermion` | `ParityField` | `Ns × Nc` | Single-parity fermion |
| `LatticeFermion` | `FullField[HalfLatticeFermion]` | `Ns × Nc` | Full fermion field |
| `MultiLatticeFermion` | `MultiField[LatticeFermion]` | `L5 × Ns × Nc` | Multiple fermion fields |
| `LatticePropagator` | `FullField` | `Ns × Nc × Ns × Nc` | Full propagator |
| `HalfLatticeStaggeredFermion` | `ParityField` | `Nc` | Single-parity staggered fermion |
| `LatticeStaggeredFermion` | `FullField[HalfLatticeStaggeredFermion]` | `Nc` | Full staggered fermion |
| `MultiLatticeStaggeredFermion` | `MultiField[LatticeStaggeredFermion]` | `L5 × Nc` | Multiple staggered fermions |
| `LatticeStaggeredPropagator` | `FullField` | `Nc × Nc` | Staggered propagator |

Each type also has a `Multi*` variant backed by `MultiField`.

### LatticeGauge

In addition to `BaseField` methods, `LatticeGauge` provides gauge-specific operations:

**Initialization & setup:**
```python
gauge = LatticeGauge(latt_info)             # Default: 4 unit SU(3) links per site
gauge.setAntiPeriodicT()                     # Apply anti-periodic temporal boundary
gauge.setAnisotropy(anisotropy: float)       # Set anisotropy factor
```

**Covariant operations:**
```python
gauge.covDev(x: LatticeFermion, covdev_mu: int) -> LatticeFermion
```
Covariant derivative: $\psi'(x)=U_\mu(x)\psi(x+\hat{\mu})$. Directions: 0–3 = +x/+y/+z/+t, 4–7 = −x/−y/−z/−t.

```python
gauge.laplace(x: LatticeStaggeredFermion, laplace3D: int) -> LatticeStaggeredFermion
```
Laplacian: $\psi'(x)=\frac{1}{N}\sum_\mu\psi(x)-\frac{1}{2}[U_\mu(x)\psi(x+\hat\mu)+U_\mu^\dagger(x-\hat\mu)\psi(x-\hat\mu)]$. `laplace3D`: 3 (spatial) or 4 (all).

```python
gauge.wuppertalSmear(x: LatticeFermion | LatticeStaggeredFermion, n_steps: int, alpha: float)
```
Wuppertal (Gaussian) smearing on a fermion field.

**Gauge path & Wilson loop:**
```python
gauge.path(paths: List[List[int]]) -> LatticeGauge
gauge.loop(loops: List[List[List[int]]], coeff: List[float])
gauge.loopTrace(loops: List[List[int]]) -> NDArray
```

**Smearing:**
```python
gauge.apeSmear(n_steps, alpha, dir_ignore, compute_plaquette=False, compute_qcharge=False)
gauge.apeSmearChroma(n_steps, factor, dir_ignore, compute_plaquette=False, compute_qcharge=False)
gauge.stoutSmear(n_steps, rho, dir_ignore, compute_plaquette=False, compute_qcharge=False)
gauge.hypSmear(n_steps, alpha1, alpha2, alpha3, dir_ignore, compute_plaquette=False, compute_qcharge=False)
```

**Gradient flow:**
```python
gauge.wilsonFlow(n_steps, epsilon, compute_plaquette=False, compute_qcharge=True) -> List
gauge.wilsonFlowScale(max_steps, epsilon, ...) -> Tuple[float, float]   # Returns (t0, w0)
gauge.symanzikFlow(n_steps, epsilon, compute_plaquette=False, compute_qcharge=True) -> List
gauge.symanzikFlowScale(max_steps, epsilon, ...) -> Tuple[float, float]
gauge.wilsonFlowChroma(n_steps, time, ...)   # Chroma convention (total time instead of epsilon)
gauge.symanzikFlowChroma(n_steps, time, ...)
```

**Staggered phase & projection:**
```python
gauge.staggeredPhase(applied: bool)    # Apply (True) / remove (False) staggered phase
gauge.projectSU3(tol: float)           # Project onto SU(3)
```

**Observables:**
```python
gauge.plaquette()           # Returns (all, spatial, temporal)
gauge.polyakovLoop()        # Returns (real, imag)
gauge.energy()              # Returns (all, spatial, temporal)
gauge.qcharge() -> float    # Topological charge
gauge.qchargeDensity() -> NDArray   # Shape: (2, Lt, Lz, Ly, Lx // 2)
```

**Random & gauge fixing:**
```python
gauge.gauss(seed: int, sigma: float)
gauge.fixingOVR(gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta)
gauge.fixingFFT(gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta)
```

Parameters for gauge fixing:
- `gauge_dir`: 3 for Coulomb, 4 for Landau
- `Nsteps`: maximum number of steps
- `verbose_interval`: print info every N steps
- `relax_boost` / `alpha`: method-specific parameter (1.5–1.7 for OVR, 0.08 for FFT)
- `tolerance`: convergence tolerance (0 = run all steps)
- `reunit_interval`: reunitarize every N steps (OVR only)
- `autotune`: 1 to enable alpha autotune (FFT only)
- `stopWtheta`: 0 for MILC criterion, 1 for theta value

**HDF5 I/O:**
```python
LatticeGauge.loadH5(filename, *, check=True) -> LatticeGauge    # labels: ["X","Y","Z","T"]
gauge.saveH5(filename, *, annotation="", check=True, use_fp32=False)
gauge.appendH5(filename, *, annotation="", check=True, use_fp32=False)
gauge.updateH5(filename, *, annotation="", check=True)
```

**Context manager:**
```python
with gauge.use():
    # gauge is loaded into QUDA within this block
    ...
```

### LatticeRotation

Rotation matrix field (L5=1, Nc×Nc per site). Useful for gauge fixing transforms.

```python
rotation = LatticeRotation(latt_info)

LatticeRotation.loadH5(filename, *, check=True) -> LatticeRotation   # label: "R"
rotation.saveH5(filename, *, annotation="", check=True)
rotation.appendH5(filename, *, annotation="", check=True, use_fp32=False)
rotation.updateH5(filename, *, annotation="", check=True)

rotation.pack(x: LatticeFermion)     # Pack rotation matrix columns into fermion field
rotation.unpack(x: LatticeFermion)   # Unpack fermion field into rotation matrix columns
```

### LatticeMom

Conjugate momentum field (for HMC). Nd × Nc × Nc per site, traceless anti-Hermitian.

```python
mom = LatticeMom(latt_info)

mom.gauss(seed: int, sigma: float)    # Fill with Gaussian random momenta

LatticeMom.loadH5(filename, *, check=True) -> LatticeMom
mom.saveH5(filename, *, annotation="", check=True, use_fp32=False)
mom.appendH5(filename, *, annotation="", check=True, use_fp32=False)
mom.updateH5(filename, *, annotation="", check=True)
```

### LatticePropagator & LatticeStaggeredPropagator

```python
propag = LatticePropagator(latt_info)
propag.setFermion(fermion: LatticeFermion, spin: int, color: int)
propag.getFermion(spin: int, color: int) -> LatticeFermion
```

```python
propag = LatticeStaggeredPropagator(latt_info)
propag.setFermion(fermion: LatticeStaggeredFermion, color: int)
propag.getFermion(color: int) -> LatticeStaggeredFermion
```

### Other Field Types

| Class | Constructor | Notes |
|---|---|---|
| `LatticeInt(latt_info)` | | Integer field on even-odd lattice |
| `LatticeReal(latt_info)` | | Real field on even-odd lattice |
| `LatticeComplex(latt_info)` | | Complex field on even-odd lattice |
| `LatticeLink(latt_info)` | | Single SU(3) link (defaults to identity) |
| `LatticeClover(latt_info)` | | Clover term field |
| `HalfLatticeFermion(latt_info)` | | Single-parity fermion |
| `LatticeFermion(latt_info)` | | Full even-odd fermion |
| `MultiLatticeFermion(latt_info, L5)` | | L5 fermion fields |
| `HalfLatticeStaggeredFermion(latt_info)` | | Single-parity staggered fermion |
| `LatticeStaggeredFermion(latt_info)` | | Full even-odd staggered fermion |
| `MultiLatticeStaggeredFermion(latt_info, L5)` | | L5 staggered fermion fields |

All `Multi*` classes support indexing (`field[i]`) and iteration.

---

## `pyquda`

### Initialization

`pyquda` re-exports all `pyquda_comm` functions and additionally provides:

```python
def init(
    grid_size: Optional[List[int]] = None,
    latt_size: Optional[List[int]] = None,
    grid_map: GridMapType = "default",
    backend: BackendType = "cupy",
    backend_target: BackendTargetType = ...,
    init_quda: bool = True,
    use_malloc_quda: bool = False,
    *,
    resource_path: str = "",
    rank_verbosity: List[int] = [0],
    enable_mps: bool = False,
    enable_gdr: bool = False,
    enable_gdr_blacklist: List[int] = [],
    enable_p2p: Literal[-3,-2,-1,0,1,2,3,4,5,6,7] = 3,
    enable_p2p_max_access_rank: int = 0x7FFFFFFF,
    enable_zero_copy: bool = False,
    enable_nvshmem: bool = True,
    allow_jit: bool = False,
    reorder_location: Literal["GPU","CPU"] = "GPU",
    enable_tuning: bool = True,
    enable_tuning_shared: bool = True,
    tune_version_check: bool = True,
    tuning_rank: int = 0,
    profile_output_base: str = "",
    enable_target_profile: List[int] = [],
    do_not_profile: bool = False,
    enable_trace: Literal[0,1,2] = 0,
    enable_force_monitor: bool = False,
    enable_monitor: bool = False,
    enable_monitor_period: int = 1000,
    enable_device_memory_pool: bool = True,
    enable_pinned_memory_pool: bool = True,
    enable_managed_memory: bool = False,
    enable_managed_prefetch: bool = False,
    deterministic_reduce: bool = False,
    device_reset: bool = False,
    max_multi_rhs: int = 0,
)
```
**Main entry point**: initializes MPI grid + compute device + QUDA library. Keyword arguments correspond to QUDA environment variables. See [Environment](Environment) for details.

```python
def isQUDAInitialized() -> bool
```
Check whether QUDA has been initialized.

### QUDA State

`pyquda` also re-exports the full `pyquda_comm` initialization and query surface:
- `initGrid()`, `initDevice()`
- `isGridInitialized()`, `isDeviceInitialized()`, `isQUDAInitialized()`
- `getMPIComm()`, `getMPISize()`, `getMPIRank()`
- `getGridMap()`, `getGridSize()`, `getGridCoord()`, `getGridRanks()`
- `getArrayBackend()`, `getArrayBackendTarget()`, `getArrayDevice()`
- `getLogger()`, `setLoggerLevel()`

### Missing Legacy Entries

Older wiki revisions mentioned `setDefaultLattice()` / `getDefaultLattice()`. These functions are not present in the current `pyquda` package and should be considered removed from the public API.

---

## `pyquda_utils.core`

### Re-exported Symbols

`pyquda_utils.core` is the recommended high-level entry point for most application code:

```python
from pyquda_utils import core

core.init(resource_path=".cache/quda")
latt_info = core.LatticeInfo([4, 4, 4, 8])
```

It re-exports:
- Runtime helpers from `pyquda`: `init`, `getMPIComm`, `getMPISize`, `getMPIRank`, `getGridSize`, `getGridCoord`, `getGridMap`, `getArrayBackend`, `getArrayDevice`, `getLogger`, `setLoggerLevel`
- Common field constants: `Ns`, `Nc`, `Nd`, `X`, `Y`, `Z`, `T`
- Common field classes: `LatticeInfo`, `LatticeGauge`, `LatticePropagator`, `LatticeStaggeredPropagator`, `LatticeFermion`, `LatticeStaggeredFermion`, `LatticeComplex`, and their `Multi*` variants
- Layout helpers: `lexico()`, `evenodd()`

`LaplaceLatticeInfo` is an alias for `LatticeInfo`.

### Dirac Constructors

```python
def getDirac(
    latt_info: LatticeInfo,
    mass: float,
    tol: float,
    maxiter: int,
    xi_0: float = 1.0,
    clover_coeff_t: float = 0.0,
    clover_coeff_r: float = 1.0,
    multigrid: Union[List[List[int]], Multigrid, None] = None,
)
```
Return `WilsonDirac` or `CloverWilsonDirac` depending on the clover coefficients.

```python
def getStaggeredDirac(
    latt_info: LatticeInfo,
    mass: float,
    tol: float,
    maxiter: int,
    tadpole_coeff: float = 1.0,
    naik_epsilon: float = 0.0,
)
```
Return a `HISQDirac`.

```python
def getWilson(...)
def getClover(...)
def getStaggered(...)
def getHISQ(...)
```
Specialized convenience constructors for the corresponding Dirac operator classes.

### Inversion Helpers

```python
def invert(
    dirac: FermionDirac,
    source_type: Literal["point", "wall", "volume", "momentum", "colorvector"],
    t_srce: Union[List[int], int, None],
    source_phase=None,
    mrhs: int = 1,
    restart: int = 0,
) -> LatticePropagator
```

```python
def invertEigenvector(
    dirac: FermionDirac,
    t_srce: int,
    source_propag: LatticeStaggeredFermion,
    mrhs: int = 1,
    restart: int = 0,
) -> MultiLatticeFermion
```

```python
def invertSequential(
    dirac: FermionDirac,
    source_propag: LatticePropagator,
    t_srce: int,
    mrhs: int = 1,
    restart: int = 0,
) -> LatticePropagator
```

```python
def invertPropagator(
    dirac: FermionDirac,
    source_propag: LatticePropagator,
    mrhs: int = 1,
    restart: int = 0,
) -> LatticePropagator
```

```python
def invertStaggered(
    dirac: StaggeredFermionDirac,
    source_type: Literal["point", "wall", "volume", "momentum", "colorvector"],
    t_srce: Union[List[int], int, None],
    source_phase=None,
    mrhs: int = 1,
    restart: int = 0,
) -> LatticeStaggeredPropagator
```

```python
def invertStaggeredSequential(
    dirac: StaggeredFermionDirac,
    source_propag: LatticeStaggeredPropagator,
    t_srce: int,
    mrhs: int = 1,
    restart: int = 0,
) -> LatticeStaggeredPropagator
```

```python
def invertStaggeredPropagator(
    dirac: StaggeredFermionDirac,
    source_propag: LatticeStaggeredPropagator,
    mrhs: int = 1,
    restart: int = 0,
) -> LatticeStaggeredPropagator
```

Notes:
- `mrhs` batches right-hand sides through `invertMultiSrcRestart()`
- `restart` applies recursive residual correction
- `source_type` now includes `"momentum"` and `"colorvector"`

### MPI Lattice Helpers

```python
def gatherLattice2(
    data: numpy.ndarray,
    tzyx: List[int],
    reduce_op: Literal["sum", "mean", "prod", "max", "min"] = "sum",
    root: int = 0,
) -> Optional[numpy.ndarray]
```

```python
def scatterLattice(data_all: Optional[numpy.ndarray], tzyx: List[int], root: int = 0) -> numpy.ndarray
def gatherScatterLattice(data: numpy.ndarray, tzyx: List[int], reduce_op="sum", root: int = 0)
def gatherLattice(data: numpy.ndarray, axes: List[int], reduce_op: Literal["sum", "mean"] = "sum", root: int = 0)
```

These helpers gather sublattice-shaped NumPy arrays across MPI ranks and optionally reduce over selected lattice directions.

### Deprecated Aliases

The following names remain importable from `pyquda_utils.core`, but emit deprecation warnings and should be avoided in new code:
- `cb2()` -> `evenodd()`
- `smear()` / `smear4()` -> gauge-field smearing methods on `LatticeGauge`
- `invert12()` -> `invert()` / `invertPropagator()`
- `getDslash()` -> `getDirac()` / `getWilson()` / `getClover()`
- `getStaggeredDslash()` -> `getStaggeredDirac()` / `getStaggered()` / `getHISQ()`
