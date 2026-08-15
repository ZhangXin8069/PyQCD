---
name: pyquda-tool
description: >
  PyQUDA tool usage skill. Generates Python code that calls PyQUDA to solve
  quark propagators on lattice gauge configurations. Covers: configuration
  loading, quark parameter setup
  (Wilson/clover action, mass or kappa, clover coefficient, link smearing),
  multigrid solver configuration, source construction (point, Gaussian
  smearing with APE/HYP/stout links), propagator inversion, and residual
  verification. Reads ensemble parameters from ensemble_registry.yaml.
  Trigger on: "compute propagators", "solve propagator", "run inversions",
  "call PyQUDA", "solve Dirac equation", or when
  lqcd-physics-correlator has produced a propagator requirements list.
  For pure-gauge observables (Wilson loops, Polyakov loops, etc.)
  use the pyquda-gauge skill.
---

# PyQUDA Tool Usage

## Purpose

Translate a propagator specification into executable PyQUDA code
that reads gauge configurations and produces propagator or correlator data files.
This skill generates a **data production script only** — a single script that
computes and saves per-configuration results. It does not produce analysis code.

## Prerequisites

- PyQUDA installed with working QUDA backend and GPU access
- Gauge configurations accessible at paths specified in ensemble registry
- Python 3.8+, numpy, (optional) cupy for GPU arrays, (optional) h5py for output
- MPI environment to run PyQUDA with multiple GPUs

## Code style and execution model

**Flat, self-contained scripts**: Generated PyQUDA code should be a single-file, one-off script with all physical parameters (masses, clover coefficients, lattice dimensions, file paths, source positions, etc.) hardcoded as plain variables at the top of the file. Do not factor the code into many small functions or classes. The script should read top-to-bottom so that a collaborator can immediately see and verify every physical parameter. This is standard practice in lattice QCD: computation scripts are shared and cross-checked by multiple people, not maintained as reusable software. Command-line arguments (argparse) should be reserved for parameters that distinguish independent parallel jobs — typically the **configuration ID** (so the same script can be submitted as multiple jobs for different configs). Physical parameters that define the calculation itself must remain hardcoded in the file.

**Always run with MPI**: PyQUDA scripts must be launched via MPI, e.g. `mpirun -np 4 python script.py` or `srun -n 4 python script.py` on SLURM clusters. Even single-GPU runs use `mpirun -np 1`. The number of MPI ranks must equal the product of grid_size dimensions, with one GPU per MPI rank by default. Always include a comment at the top of the generated script showing the expected launch command, e.g. `# Run: mpirun -np 4 python this_script.py`.

**Heavy computation — no interactive monitoring**: PyQUDA propagator inversions are computationally intensive and typically run for minutes to hours per configuration. Do not add progress bars, interactive prompts, or suggest frequent status checking. QUDA prints solver iteration counts and residuals to stdout automatically — the user monitors progress via stdout or log redirection. Submit the script and let it run to completion.

## Before generating code (REQUIRED)

Do NOT generate any computation code until the following are resolved. Present the information gathered in step 1 to the user, then ask the questions in step 2 and wait for answers.

**Step 1 — Physics derivation**: The propagator requirements must be known before writing code. Use `lqcd-physics-correlator` reasoning to derive: what interpolating operators are needed, what the Wick contraction looks like, and which propagators (quark flavors, source$\rightarrow$ sink structure) are required. Present this derivation to the user.

**Step 2 — Ask the user** to confirm or specify the following source configuration, as the optimal choice depends on the target observable and computational budget:

- **Source type**: point, wall, smeared (Gaussian/Wuppertal), or volume
- **Smearing**: whether to apply Gaussian smearing to the source, and if so, the smearing parameters (radius `rho`, number of steps `n_steps`) and which gauge field to use for smearing
- **Source positions**: spatial position(s) `[x0, y0, z0]` and time slices `t_src`. Using multiple source time slices (e.g., `t_src = 0, T//4, T//2, 3*T//4`) on each configuration significantly improves the signal-to-noise ratio by multiplying the effective statistics

Momentum projection is **not** a user choice — it is determined by the correlator definition. If the target observable requires momentum $\vec{p}$, the source must be projected to that momentum accordingly.

**Step 3 — Generate code** only after the user has confirmed the source configuration.

## Workflow

### Step 0: Common conventions

PyQUDA is a Python wrapper of the QUDA library, which provides GPU-accelerated operations for lattice QCD. PyQUDA uses NumPy arrays to handle lattice fields, and uses CuPy/PyTorch/DPNP arrays when GPU accelerated linear algebra is needed. PyQUDA is designed to run on cluster environment which have a job scheduler like SLURM or PBS, thus it usually runs in an MPI environment. There are some key concepts to understand when using the package:


#### MPI

In PyQUDA, MPI is used to run the code on multiple GPUs across different nodes in a cluster. Each MPI rank corresponds to a separate process that can run on a different GPU, and they communicate with each other during the computation. The number of MPI ranks should match the total number of GPUs being used for the job. PyQUDA might use a custom MPI communicator instead of the COMM_WORLD, and defines the API to access the communicator.

```python
from pyquda_utils import core

comm = core.getMPIComm()
rank = core.getMPIRank()
size = core.getMPISize()
```
 ⚠️ `pyquda_utils.core` does NOT have an `allreduce()` function.
 Do NOT write `core.allreduce(...)` — it will crash with AttributeError.

#### Grid

Grid in PyQUDA indicates how to partition the lattice across multiple MPI ranks, which is very close to the concept of a Cartesion communicator. For example, a grid size of `[2, 2, 1, 1]` means that the lattice will be partitioned into 4 sublattices in the x and y dimensions, while the z and t dimensions are not partitioned. The product of the grid dimensions must equal the total number of MPI ranks used to run the job. If one dimension is partitioned, there will be communication between MPI ranks in that dimension during the solver iterations. If the grid size is not specified, but the targeting lattice size is provided during the initialization, PyQUDA will automatically generate a grid size, trying to minimize the communication between different MPI ranks. PyQUDA defines the API to access the grid information.

```python
from pyquda_utils import core

grid_size = core.getGridSize()
grid_coord = core.getGridCoord()
```

#### Device

Device in PyQUDA refers to the local GPU device ID that each MPI rank will use. By default, PyQUDA will assign GPU devices based on the local rank of each MPI process. For example, if you have 4 GPUs and 4 MPI ranks, each rank will be assigned to a different GPU (rank 0 to GPU 0, rank 1 to GPU 1, etc.). Sometimes, a cluster will offer a binding script to bind each MPI rank to a specific NUMA node, GPU, and NIC, and environment variables such as `CUDA_VISIBLE_DEVICES` are usually set by the script. You will have to set `enable_mps=True` during the initialization in this situation to allow all ranks in one node can use the same GPU ID 0, although they are actually using different devices. PyQUDA defines the API to access the current backend (the pacakge to manage GPU arrays) and device information.

```python
from pyquda_utils import core

backend = core.getArrayBackend()
backend_target = core.getArrayBackendTarget()
device = core.getArrayDevice()
```

#### Lattice information

`LatticeInfo` class in PyQUDA is used to store the lattice information, including the lattice dimensions, boundary conditions, and anisotropy. Assuiming the global lattice size is `[GLx, GLy, GLz, GLt]`, the grid size is `[Gx, Gy, Gz, Gt]`, and the local lattice size for each MPI rank is `[Lx, Ly, Lz, Lt]`, where `Lx = GLx // Gx`, `Ly = GLy // Gy`, `Lz = GLz // Gz`, and `Lt = GLt // Gt`. The `LatticeInfo` object is used to initialize the Dirac operator and to create lattice fields, ensuring that all operations are consistent with the lattice geometry and partitioning.

#### Lattice fields

Lattice fields are objects in PyQUDA to handle the data of fields used in LQCD. For example, the gauge field is hold by a `LatticeGauge` object, and the quark propagator is hold by a `LatticePropagator` object. The `data` attribute of these objects is a `numpy.ndarray` (or `cupy.ndarray`/`torch.Tensor`/`dpnp.ndarray` array if using GPU acceleration) that contains the actual field data. The layout of the field data is `[2, Lt, Lz, Ly, Lx // 2]`, which is the even-odd preconditioned layout. The first dimension of size 2 corresponds to the parity (even/odd) of the lattice sites, and the last dimension of size `Lx // 2` corresponds to the half-lattice size in the x direction due to the even-odd preconditioning. Note the order of dimensions is "xyzt" in most cases in PyQUDA, except for the data layout of a field (which is "tzyx"). If a filed has both source and sink spin/color indices, the order will always be `[snk, src]`. For example, the shape of a `LatticePropagator.data` will be `[2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc]`, and the meaning of each dimension is `[parity, t, z, y, x//2, spin_snk, spin_src, color_snk, color_src]`. The `LatticeGauge.data` will have a shape of `[4, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc]`, where the additional dimension of size 4 corresponds to the four directions of the gauge links.

#### Array location

PyQUDA can handle arrays in different locations (CPU or GPU) with different backends (NumPy, CuPy, PyTorch, DPNP). The `backend` parameter in the initialization determines which array library is used for handling lattice fields by default. If `backend="cupy"` is set, PyQUDA will create a CuPy array for the field data when creating a `LatticeGauge` or `LatticePropagator` object. But remember if a `LatticeGauge` or `LatticePropagator` is created by loading from disk, the field data will always be created as a NumPy array on CPU, and you will have to use the `toDevice()` to transfer the data to GPU memory. PyQUDA provides a consistent API to support array linear algebra operations on GPU with backends. You can check `pyquda_comm.array` module for the supported operations. Using the `"numpy"` backend will keep all arrays on CPU, saving GPU memory but without acceleration. PyQUDA defines the API to transfer arrays between CPU and GPU.

```python
from pyquda_comm import array

# Define the GPU array backend
backend = "cupy" # or "torch", "dpnp", "numpy"

# Transfer a CPU (NumPy) array to GPU (CuPy/PyTorch/DPNP)
gpu_array = array.arrayAsArray(cpu_array, backend=backend)

# Transfer a GPU (CuPy/PyTorch/DPNP) array to CPU (NumPy)
cpu_array = array.arrayAsNumpy(gpu_array, backend=backend)

# Transfer a GPU (CuPy/PyTorch/DPNP) array to CPU (NumPy), and ensure the data is contiguous in memory
cpu_array = array.arrayAsNumpyCopy(gpu_array, backend=backend)
```

#### Gamma matrices

The basis for the gamma matrices provided by `pyquda_utils.gamma` module is the DeGrand-Rossi basis. PyQUDA uses a bit-field encoding for gamma matrices, where `gamma.gamma(1)` corresponds to $\gamma_{1}$, `gamma.gamma(2)` to $\gamma_{2}$, `gamma.gamma(4)` to $\gamma_{3}$, and `gamma.gamma(8)` to $\gamma_{4}$. Products of gamma matrices can be represented using bitwise OR. For example, `gamma.gamma(15)` corresponds to $\gamma_{1}\gamma_{2}\gamma_{3}\gamma_{4}$ = $\gamma_{5}$.



### Step 1: Read ensemble metadata

Read the ensemble registry (YAML/JSON) to obtain:
- Configuration file paths and format (ILDG, QIO, milc, ...)
- Lattice dimensions (Lx, Ly, Lz, Lt)
- Gauge action parameters (beta, tadpole factor if applicable)

### Step 2: Initialize PyQUDA and then get the lattice information

```python
from pyquda_utils import core

grid_size = [Gx, Gy, Gz, Gt] # grid partitioning
latt_size = [Lx, Ly, Lz, Lt] # lattice dimensions
core.init(grid_size, latt_size, backend="cupy", resource_path="/path/to/quda/tunecache")
latt_info = core.LatticeInfo(latt_size, t_boundary=-1, anisotropy=1.0)
```
Here we initialize the PyQUDA context with the MPI, and set the lattice partitioning `grid_size`, and then create a `LatticeInfo` object with the lattice dimensions `latt_size`. One must `init` before call the `LatticeInfo`.  The `latt_size` in `core.init` will be ignored if `grid_size` is specified, and the MPI size must be equal to the product of `grid_size`. PyQUDA will automatically generate a `grid_size` if only `latt_size` is provided, trying to minimize the communication between different MPI ranks. The `t_boundary=-1` indicates anti-periodic boundary conditions in time, and `anisotropy=1.0` indicates no anisotropy (use `xi_0 / nu` if using an anisotropic lattice, where `xi_0` is the gauge anisotropy and `nu` is the input light speed). The `backend` parameter specifies which Array API we should use to handle lattice fields. `"cupy"` backend could be helpful for GPU acceleration in the "contraction to correlator" step. The `resource_path` is where PyQUDA saves and loads QUDA **autotuning cache**. On the first run with a given lattice size, grid partitioning, and solver configuration, QUDA automatically benchmarks many GPU kernel launch parameters to find the optimal ones. This autotuning can take significant extra time (minutes to tens of minutes) on the first run, but subsequent runs with the same parameters load the cached settings and start computing immediately. Always set `resource_path` to a persistent directory (e.g., `"./"` or a shared project directory) so the tune cache is preserved across runs. If the cache was generated by a different QUDA version, delete it and retune or use another directory to avoid incompatible launch parameters.

### Step 3: Load gauge configuration

```python
from pyquda_utils import io

gauge = io.readChromaQIOGauge("/path/to/gauge/configuration")
```
Here we load the gauge configuration from disk using the appropriate read function based on the file format. The `readChromaQIOGauge` function is used for Chroma generated QIO formatted files. Make sure to replace the path with the actual location of your gauge configuration. You can read the PyQUDA source code to determine which read function to use for other formats (e.g. `readMILCGauge` for MILC format).

### Step 4: Configure quark and solver parameters

#### Wilson fermion:
```python
from pyquda_utils import core

dirac = core.getWilson(latt_info, mass, tol, maxiter, multigrid)
```
Here we create a Wilson Dirac operator with the specified mass, solver tolerance, maximum iterations, and multigrid settings. `multigrid` should be the parameter to determine the aggregation size of every level. If `multigrid` is `None`, BiCGStab is used. These parameters should be set according to the requirements of your calculation. For example, for light quarks, you might need a smaller mass and a looser solver tolerance, while for heavy quarks, you can use a larger mass and a tighter tolerance to ensure the accuracy in timeslices far from the source. The multigrid is usually very helpful for light quark propagators, because of the critical slowing down issue of LQCD.

#### Clover fermion:
```python
from pyquda_utils import core

dirac = core.getClover(latt_info, mass, tol, maxiter, xi_0, csw_t, csw_r, multigrid)
```
Here we create a Clover Dirac operator with the specified mass, solver tolerance, maximum iterations, gauge anisotropy `xi_0`, clover coefficients `csw_t` and `csw_r`, and multigrid settings. Note that `csw_t` and `csw_r` are the clover coefficients for the temporal and spatial components, respectively. If your lattice is isotropic, you can set `csw_t = csw_r = csw`. `xi_0` is the gauge anisotropy,  please clarify it from the fermion anisotropy `xi = xi_0 / nu`.

#### Load all required fields into GPU memory:
```python
from pyquda_utils import core

gauge.stoutSmear(1, rho, 4)
with dirac.useGauge(gauge):
    # Do something
    ...
```
Here we first apply 4-dimentional stout smearing of parameter `rho` to the gauge field for 1 time. This might be necessary to calculate the propagator if you found quarks.smearing in the ensemble registry. You can read the PyQUDA source code to determine the appropriate smearing algorithm and parameters. Then we use the `useGauge` context manager to ensure that the gauge field and all auxiliary fields are loaded into QUDA. Inside this context, you can solve quark propagators defined by the Dirac operator. If mutliple context managers are nested, the innermost one will take effect. `useGauge` is not a free operation, it will trigger the data transfer between CPU and GPU if the gauge field is not already on GPU, and it will also trigger the reorder operion to convert the gauge field data into the layout required by QUDA. So it is better to put all the operations that require the gauge field inside the same `useGauge` context to avoid unnecessary data transfer and reordering.

### Step 5: Construct source and solve propagator

#### Directly solve propagator from a simple (point, wall, volume) source:
```python
from pyquda_utils import core, phase_v2

phase = phase_v2.MomentumPhase(latt_info).getPhase([kx, ky, kz], [x0, y0, z0])

with dirac.useGauge(gauge):
  propag_pt = core.invert(dirac, "point", [x0, y0, z0, t0], phase.data)
  propag_wl = core.invert(dirac, "wall", t0, phase.data)
  propag_vl = core.invert(dirac, "volume", None, phase.data)
```
Here we call the `invert` function to solve for the propagator using different source types. For a point source, we specify the position `[x0, y0, z0, t0]`. For a wall source, we specify the time slice `t0`. For a volume source, no additional parameters are needed. Here we also apply a momentum phase to the source, which is necessary for computing momentum wall source propagators. If no `phase` is given, the values for all three types will default to 1.

#### Solve propagator from an existing propagator:
```python
from pyquda_utils import core, source

with dirac.useGauge(gauge):
  source_pt = source.propagator(latt_info, "point", [x0, y0, z0, t0])
  source_sh = source.gaussianSmear(source_pt, gauge, rho, n_steps)
  propag_sh = core.invertPropagator(dirac, source_sh)
```
Here we first create a point source propagator using the `source.propagator` function. Then we apply Gaussian smearing to this source using the `source.gaussianSmear` function, which takes the original source, the gauge field, the smearing radius in the momentum space `rho`, and the number of smearing steps `n_steps`. Note the gauge we used here for the gaussian smearing might be different from the one used for the Dirac operator, depending on the requirements. Finally, we solve for the smeared propagator using the `core.invertPropagator` function.

#### Solve sequential propagator from an existing propagator on a specific time slice:
```python
from pyquda_utils import core

with dirac.useGauge(gauge):
  propag_sq = core.invertSequential(dirac, propag_sh, t_seq)
```
Here we use the `core.invertSequential` function to solve for a sequential propagator from the smeared propagator. This is useful for three-point correlator calculations where we need to insert an operator at a specific time slice. The `t_seq` parameter specifies the time slice where the sequential source is defined.

#### Source shift (before inversion)

When the Wilson line is on the source side, shift the source field before
inversion. Sink-side shifts are covered in the **Covariant displacement**
section below.

```python
from pyquda_utils import core, X, Y, Z, T

with gauge_shift.use() as dirac_shift:
    for spin in range(4):
        for color in range(3):
            tmp = src.getFermion(spin, color)
            tmp = dirac_shift.covDev(tmp, direction)  # e.g. -Z for -z shift
            src.setFermion(tmp, spin, color)
```
Gauge mixing issues are covered in **Common mistakes** below.

#### Multiple source times

Structure the main computation as a loop over source times when multiple sources are used:

```python
for t_src in t_srcs:
    propag = core.invert(dirac, "point", [x0, y0, z0, t_src])
    # ... contract to correlator and save with t_src label
```

### Step 6: Save propagator (optional)
This is not always necessary, but you can save the propagator to disk in a format of your choice (e.g. HDF5, NumPy binary) for later analysis.

```python
propag_sh.save("propag_sh.npy", use_fp32=False)
propag_sh.saveH5("propag_sh.h5", tag, annotation=annotation, check=True, use_fp32=False)
```
Here we save the smeared propagator in both NumPy binary format and HDF5 format. The `save` method saves the propagator as a `.npy` file, while the `saveH5` method saves it as an `.h5` file with additional metadata such as a tag, annotation, and a check for data integrity. The `use_fp32` parameter determines whether to save the data in single precision (float32) or double precision (float64).

### Step 7: Contraction to correlator (optional)

#### 7a — Einsum generation via `generate_einsum` tool (REQUIRED)

ALL einsum strings for correlator contractions MUST be produced by calling
the `generate_einsum` tool. A single call is sufficient for each correlator —
do not call the tool multiple times for the same correlator type.
Never write einsum strings by hand.

Correlator types supported by the generate_einsum tool:

| type | required params | tool returns | integration into main.py |
|------|----------------|--------------|--------------------------|
| `meson_2pt` | **tool_args fields**:<br>`antiquark`, `quark` — meson flavor content<br>`gamma_snk`, `gamma_src` — gamma matrices (default gamma5). `gamma_src` gets Dirac adjoint automatically. | `code`: a `contract()` call string | Paste `result["code"]` at step 5. Variables needed: `prop_l.data` (or `prop_s`/`prop_c`). |
| `baryon_2pt` | **tool_args fields**:<br>`quark_a`, `quark_b`, `quark_c` — baryon quark flavors<br>`projector` — P_plus or P_minus<br>`diquark_gamma`, `c_gamma` — gamma matrices (default Cg5, I4) | `code`: one or more `contract()` call strings, one per Wick topology. `topologies`: dict with per-topology einsum and sign. |Paste all `contract()` calls from `result["code"]`, replacing variable names (`prop_l`/`prop_s`/`prop_c`) as needed. Sum over topologies. Finally `core.gatherLattice(..., [dt, -1, -1, -1])`. Here, dt refers to which dimension of the data the time dimension corresponds to. One call is enough — if the returned code looks correct, paste it and proceed. |
| `multi_hadron_2pt` | `specs`, `out_name` | `sink_path`, `sink_file` | Use `exec(open(tool_result["sink_file"]).read())` at step 5. |
| `meson_3pt` | **tool_args fields**:<br>`src_antiquark`, `src_quark` — source meson flavors (initial state)<br>`sink_antiquark`, `snk_quark` — sink meson flavors (final state)<br>`current_quark`, `current_antiquark` — **⚠ direction**: current = anti(current_antiquark) · Γ · q(current_quark). This is OPPOSITE to the task's q̄·Γ·q convention. E.g. for c→s transition: `current_quark="c"`, `current_antiquark="s"`.<br>`gamma_snk`, `gamma_src`, `gamma_cur` — gamma names. Accepts gamma1-5, gamma_x/y/z/t, gx/gy/gz/gt.<br>`tseq` — time separation | `code`: complete sink block + seq source + contraction | Paste `result["code"]` at step 5. Adapt propagator variable names (`prop_l`/`prop_s`/`prop_c`) to match the script. Adapt `dirac_{seq_flavor}` to the Dirac operator name used in the script. |
| `baryon_3pt` | **tool_args fields**:<br>`src_a`, `src_b`, `src_c` — source baryon quark flavors<br>`snk_a`, `snk_b`, `snk_c` — sink baryon quark flavors<br>`current_quark`, `current_antiquark` — current flavors<br>`current_gamma` — gamma matrix<br>`projector` — P_plus or P_minus<br>`diquark_gamma_snk`, `diquark_gamma_src` — diquark gamma (default Cg5)<br>`c_gamma_snk`, `c_gamma_src` — c-quark gamma (default I4)<br>`tseq` — time separation | `code`: complete sink block + seq source + contraction | Paste `result["code"]` at step 5. Adapt propagator variable names. One call is enough — the tool returns everything needed. |
---

#### 7b — Contraction example (meson two-point, zero momentum)

**Pion (gamma5 interpolator, meson_2pt):**
```python
result = tool_call(meson_operator(antiquark,quark,gamma),meson_operator(antiquark,quark,gamma))
# The first variable is the operator at the sink while the second variable is the operator at the source
# result["code"] = 'contract("wtzyxCBba, wtzyxCBba -> t", prop_l.conj(), prop_l)'
C_t_local = contract(einsum_str, prop_l.data.conj(), prop_l.data)
C_t = core.gatherLattice(C_t_local.get(), [0, -1, -1, -1])

if core.getMPIRank() == 0:
    # Save or process C_t
```

**Rho (vector, generate_einsum type: meson_2pt, meson=rho):**
```python
einsum_str = "wtzyx, wtzyxjiba, jk, wtzyxklba, li -> t"  # from generate_einsum tool
C_t_local = 0
for gi in [gamma.gamma(1), gamma.gamma(2), gamma.gamma(3)]:
    C_t_local += contract(einsum_str, phase.data, prop_l.data.conj(),
                          gamma5 @ gi, prop_l.data, gamma4 @ gi.conj().T @ gamma4 @ gamma5)
C_t_local /= 3
C_t = core.gatherLattice(C_t_local.get(), [0, -1, -1, -1])

if core.getMPIRank() == 0:
    # Save or process C_t
```

**Baryon 2pt (octet, Cγ₅ diquark, generate_einsum type: baryon_2pt):**
Call the tool once with `generate_einsum(baryon_operator(quark_1,quark_2,quark_3,diquark_gamma,gamma),baryon_operator(quark_1,quark_2,quark_3,diquark_gamma,gamma))`.
The tool returns `code` containing one or more `contract()` calls, one per
Wick topology. Before those calls, add the required gamma/einsum definitions:
```python
# Gamma matrix and tensor definitions (GPU tensors)
I4 = cp.eye(4, dtype=cp.complex128)
G5 = cp.asarray(gamma.gamma(15), dtype=cp.complex128)
eps = cp.zeros((3, 3, 3), dtype=cp.complex128)
eps[0,1,2] = eps[1,2,0] = eps[2,0,1] = 1.0
eps[0,2,1] = eps[2,1,0] = eps[1,0,2] = -1.0
Cmat = cp.asarray(gamma.gamma(2) @ gamma.gamma(8), dtype=cp.complex128)
Cg5 = Cmat @ G5
Tmat = cp.asarray((I4 + gamma.gamma(8)) * 0.5, dtype=cp.complex128)  # P_plus
```
Then paste the `contract()` calls from `result["code"]` and sum them. For example:
```python
# Sum over Wick topologies (from generate_einsum tool)
C_t_local = 0
C_t_local += contract("wtzyxCBba, wtzyxCBba -> t", ...)  # topo 0
C_t_local += contract("wtzyxCBba, wtzyxCBba -> t", ...)  # topo 1
C_t = core.gatherLattice(C_t_local.get(), [0, -1, -1, -1])

```
A single call to `generate_einsum` provides all topologies.

**General gamma:**
Call `generate_einsum(meson_operator(antiquark,quark,gamma),meson_operator(antiquark,quark,gamma))`.
Use the returned `gamma_expressions.jk` and `gamma_expressions.li` as
the gamma matrix arguments in the 5-input contract call.
the source because of the conjugation in the source operator;
`lqcd-physics-correlator` explains that sign and phase convention. Finally, we
gather the lattice data from all MPI ranks to
obtain the full correlator in the root rank as a function of time. The
second argument of `gatherLattice` specifies the dimensions to gather in
`tzyx` order, where `0` indicates that we want to gather the data in the
$t$ dimension, and `-1` means performing reduction in all three spatial
dimensions. The array passed to `gatherLattice` should be a NumPy array, and
`array.arrayAsNumpy(field, backend=backend)` is used to transfer the data from GPU to CPU if necessary.

⚠️ CRITICAL: `array.arrayAsNumpy()` REQUIRES the `backend=` argument (e.g., `backend="cupy"`).
Calling `array.arrayAsNumpy(field)` without `backend=` will raise a runtime error in recent PyQUDA.
Always use `array.arrayAsNumpy(field, backend=backend)` or `array.arrayAsNumpy(field, backend="cupy")`.

### Step 8: Save output

Use txt output only.

```python
# Minimal local 2pt example (no extra indices)
if core.getMPIRank() == 0:
    C_t_root = np.asarray(C_t, dtype=np.complex128).reshape(-1)
    t = np.arange(C_t_root.shape[0], dtype=np.int32)
    out = np.column_stack((t, C_t_root.real, C_t_root.imag))
    np.savetxt(out_path, out, fmt=["%d", "%.16e", "%.16e"])
```

Make sure only the root rank (rank 0) writes the output file to avoid conflicts. The output file should use a `.txt` suffix, but the number of columns is observable-dependent.

- local 2pt: `t, Re, Im`
- nonlocal 2pt (Wilson-line/displacement): `z_sep, t, Re, Im` (or equivalent ordering)
- 3pt: `t_seq`, insertion time, `Re`, `Im`

Always include all discrete labels needed to uniquely identify each row (e.g. shift count, source-sink separation, momentum index) before the real/imag columns. The second argument of `gatherLattice` specifies the dimensions to gather in `tzyx` order, where `0` indicates that we want to gather the data in the $t$ dimension, and `-1` means performing reduction in all three spatial dimensions.


### three-point correlation function (sequential source)

This section is **only relevant when the task is about three-point correlation functions** (e.g., $\Lambda \to p$ transition). 


**Einsum generation: see Step 7a for the generate_einsum dispatch table.** All
correlator types (meson 2pt, baryon 2pt, 3pt sink block, propagator trace)
have a dedicated generate_einsum function listed there. Call it at generation time,
then paste the returned einsum strings directly as string literals into the
runtime script.

**Step 1** — Call generate_einsum exactly once (generation-time only, do NOT import at runtime):

Call `generate_einsum(type="baryon_3pt")` or `generate_einsum(type="meson_3pt")`
with the required parameters. The tool returns `code` containing a complete
sink block + sequential source setup + final contraction. A single call is
sufficient — paste `result["code"]` into the script at step 5.

The final script must NOT contain any generate_einsum imports.


**Step 2** — Build the sequential source pipeline (einsum returned from Step 1):

```python
# Sink block (einsum from generate_einsum, variable names from result['code'])
B = core.LatticePropagator(latt_info)
B.data = (
    + 1 * contract('<einsum_string>',
        phase, eps, eps, Cg5, Cg5, Tmat, prop_l.data, prop_l.data)
    - 1 * contract('<einsum_string>',
        phase, eps, eps, Cg5, Cg5, Tmat, prop_l.data, prop_l.data)
)

# First dagger: gamma5 @ B^dag @ gamma5
B.data = contract("AB, wtzyxCBji, CD -> wtzyxADij", G5, B.data.conj(), G5)

# Sequential source and solve
src_seq = source.sequential12(B, t_seq)
with dirac_l.useGauge(gauge_stout):
    prop_seq = core.invertPropagator(dirac_l, src_seq)

# Second dagger
prop_seq_dag = core.LatticePropagator(latt_info)
prop_seq_dag.data = contract("AB, wtzyxCBji, CD -> wtzyxADij", G5, prop_seq.data.conj(), G5)
```

**Step 3** — Final contraction and MPI gather:

```python
# Sequential source 3pt final contraction
three_pt_site = contract(
    "wtzyxijba, jk, wtzyxkiab -> wtzyx",
    prop_seq_dag.data, Gamma_cur, prop_current.data,
)
C3_t_local = contract("wtzyx -> t", three_pt_site)
C3_t = core.gatherLattice(C3_t_local.get(), [0, -1, -1, -1])
```
gatherLattice returns None on non-root ranks. Always guard save operations with `if core.getMPIRank() == 0:`.

```python
#three_pt_local = contract("wtzyx, wtzyx -> t", mom_phase_current, three_pt_site)
#C3_t = core.gatherLattice(
#    array.arrayAsNumpy(three_pt_local, backend="cupy"), [0, -1, -1, -1]
#)
```

**Step 5** — Save the result 

```python
t_list = list(range(1, t_seq))

if core.getMPIRank() == 0:
    C3_window = np.asarray(C3_t[t_list], dtype=np.complex128).reshape(-1)
    t_ins = np.asarray(t_list, dtype=np.int32)
    tseq_col = np.full(t_ins.shape, t_seq, dtype=np.int32)
    out = np.column_stack((tseq_col, t_ins, C3_window.real, C3_window.imag))
    np.savetxt(out_path, out, fmt=["%d", "%d", "%.16e", "%.16e"])
```

t_list defines the insertion-time window to save. The output file should use a `.txt` suffix and contain at least four columns for this case: `t_seq`, insertion time, real part, and imaginary part of the three-point correlator.


## Common issues

- **GPU out of memory**: Use more GPUs (increase MPI size), use `backend="numpy"` when initializing PyQUDA.
- **Solver not converging**: Check gauge configuration integrity, try restarting with tighter intermediate tolerance

## Covariant displacement (Wilson lines / nonlocal operators)

For nonlocal lattice operators (e.g. $\bar{q}(x) \Gamma W(x, x+z) q(x+z)$),
the propagator at the shifted sink must be connected back to the source via a
covariant displacement (Wilson line). This is implemented using `covDev()` on
the raw (unsmeared) gauge field.

### Gauge field management

Always maintain **two copies** of the gauge field:

```python
gauge_raw = io.readChromaQIOGauge(cfg_path)
gauge_raw.toDevice()

gauge_stout = gauge_raw.copy()          # copy BEFORE smearing
gauge_stout.stoutSmear(n_step, rho, ndim)  # for Dirac inversion
```

- `gauge_raw` — **raw links**, used only for covariant displacement
- `gauge_stout` — **smeared links**, used for propagator inversion
- Always `copy()` before smearing; `stoutSmear` modifies in place

### Gauge context: Dirac inversion vs covariant displacement

The Dirac operator and covDev use DIFFERENT gauge fields. Their `use()` contexts
are mutually exclusive — QUDA can only hold one gauge field at a time.

```python
# Step 1: Inversion on smeared gauge
with dirac_l.useGauge(gauge_stout):          # ← opens stout context
    prop_l = core.invertPropagator(dirac_l, src)
# → stout context CLOSED after the block

# Step 2: Covariant displacement on raw gauge
with gauge_raw.use() as dirac_shift:          # ← opens raw context
    ...covDev calls...
# → raw context CLOSED after the block
```

### Covariant displacement loop

Shift a propagator's sink by `z_sep` steps in direction `Z` (or `X`, `Y`, `T`):

```python
from pyquda_utils.core import X, Y, Z, T

with gauge_raw.use() as dirac_shift:
    prop_shift = prop_l.copy()
    for _ in range(z_sep):
        for spin in range(4):
            for color in range(3):
                tmp = prop_shift.getFermion(spin, color)
                tmp = dirac_shift.covDev(tmp, direction)  # X / Y / Z / T
                prop_shift.setFermion(tmp, spin, color)
```

Key rules:
- The `gauge_raw.use()` context manager provides `dirac_shift`, which
  contains the Wilson-line building operators.
- `covDev` acts on a **single spin-color component** (one fermion field).
  You must loop over all 4 spins × 3 colors and update each.
- Always start from `prop_shift = prop_l.copy()` — a fresh copy of the
  original propagator for each new displacement distance.
- One `covDev` call shifts by **one lattice step**; repeat `z_sep` times.

### Contraction and MPI gather (OUTSIDE the use() context)

After the covDev loop, perform contraction and MPI gather **after** the
`gauge_raw.use()` block. gatherLattice involves MPI communication that
can deadlock if QUDA still holds a gauge context.

```python
# ═══════════════════════════════════════════════════════════════
# IMPORTANT: C_loc shape and parity handling
# ═══════════════════════════════════════════════════════════════
# The einsum "wtzyxjiba, wtzyxjiba -> t" contracts parity (w) together
# with spin, color, and spatial indices. The result is 1D per zsep:
#   shape (Lt_local,)  — ONLY the time dimension remains.
#
# Store results in a 2D array, NOT 3D:
#   ✅ C_loc = cp.zeros((zmax + 1, Lt_local))        ← correct
#   ❌ C_loc = cp.zeros((zmax + 1, 2, Lt_local))     ← WRONG! parity already contracted
#
# After the use() block, add parity back via reshape for gatherLattice.
# ═══════════════════════════════════════════════════════════════

C_loc = cp.zeros((zmax + 1, latt_info.Lt), dtype=cp.complex128)  # 2D, NOT 3D

# covDev in use() context:
with gauge_raw.use() as dirac_shift:
    for zsep in range(zmax + 1):
        prop_shift = prop_l.copy()
        for _ in range(zsep):
            for spin in range(4):
                for color in range(3):
                    tmp = prop_shift.getFermion(spin, color)
                    tmp = dirac_shift.covDev(tmp, Z)
                    prop_shift.setFermion(tmp, spin, color)
        # Get the einsum by calling the `generate_einsum` tool:
        #   generate_einsum(type="meson_2pt")
        # → returns {"to_t": "wtzyxjiba, wtzyxjiba -> t", ...}
        # Hardcode the returned einsum string here. DO NOT write by hand.
        # ⚠️ CRITICAL: Use "-> t" (directly reduce to time-dimension)
        # Do NOT use "-> wtzyx" (keep spatial info).
        einsum_str = "wtzyxjiba, wtzyxjiba -> t"  # ← from generate_einsum tool, MUST be -> t
        # Result is 1D (Lt_local,) — parity already contracted in → t
        C_loc[zsep] = contract(
            einsum_str,
            prop_l.data.conj(),  # S^dagger(Original)
            prop_shift.data,             # S (shifted)
        )

# MPI communication AFTER use() is closed.
C_full = np.zeros((zmax + 1, latt_size[3]), dtype=np.complex128)
for zsep in range(zmax + 1):
    t_field_global = core.gatherLattice(C_loc[zsep].get(), [0, -1, -1, -1])
    if core.getMPIRank() == 0:
        C_full[zsep, :] = t_field_global
```

For nonlocal 2pt (Wilson line) call `generate_einsum` with `type="meson_2pt"` to
produce this einsum — never write a Tr[S†S] contraction by hand.

### Full workflow summary

```
1. Invert propagator:  with dirac_l.useGauge(gauge_stout): prop_l = ...
2. Open raw gauge:      with gauge_raw.use() as dirac_shift:
3.   for z = 0 ... z_max:
4.     prop_shift = prop_l.copy()
5.     for _ in range(z): covDev(prop_shift, direction)
6.     C_loc[z] = Tr[prop_shift * prop_l^dag]      ← contraction inside use()
7. Close raw gauge                                  ← use() block ends
8. gatherLattice(C_loc)                             ← MPI outside use()
```

### Common mistakes

| Mistake | Fix |
|---------|------|
| Forgetting `copy()` before smearing | Always `gauge_raw.copy()` before `stoutSmear` |
| Reusing `prop_shift` without fresh copy for each z | `prop_shift = prop_l.copy()` inside the z-loop |
| `covDev` on full propagator without spin/color loop | Loop over `spin in range(4)`, `color in range(3)` |
| Using smeared gauge for displacement | Use `gauge_raw`, NOT `gauge_stout` |
| Forgetting `gauge_raw.use()` context | `with gauge_raw.use() as dirac_shift:` must wrap the covDev loop |
| **gatherLattice inside use() context** | **Move gatherLattice AFTER the use() block — MPI + QUDA gauge context = deadlock** |
| **Inversion context overlapping with covDev context** | **Close Dirac useGauge() before opening gauge_raw.use()** |
| **C_loc with parity dimension (zmax+1, 2, Lt)** | **C_loc should be 2D (zmax+1, Lt_local). The `-> t` einsum contracts parity. Add it back via reshape before gatherLattice.** |
