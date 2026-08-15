## Physics Objective

Compute the nonlocal two-point correlation function of the η_c meson (charm-anticharm pseudoscalar, J^PC = 0^-+) with spatial quark-antiquark separation z = 0 to 10 along the +z direction. This is the fundamental building block for extracting quasi-distribution amplitudes (quasi-DAs) from lattice QCD.

## Key Revisions (addressing peer-review critique)

### 1. Correlator type changed to `nonlocal_meson_2pt`

The original plan used `meson_2pt`, which routes to `generate_einsum(type="meson_2pt")` and produces `Tr[S† S]` with only a gamma-matrix insertion — no slot for the Wilson line. The revised plan uses the explicit type `nonlocal_meson_2pt` and specifies the full contraction formula in the measurement notes:

```
C(z;t) = einsum('ab, bcxy, cayx ->', W, S_c_z, S_c_dag_0)
```

where `W` is the Nc×Nc Wilson line acting on the left color index of `S_c(z,t)`, and the spin trace contracts `S_c(z,t)` against `S_c†(0,t)`.

### 2. Explicit gauge-copy procedure for dual-gauge strategy

The original plan did not specify how to preserve unsmeared links while also using stout-smeared links for the inverter. In PyQUDA, `gauge.stoutSmear()` modifies the gauge field in-place. The revised plan mandates:
- `gauge_original = gauge.deep_copy()` **before** any smearing
- `gauge.stoutSmear(n_steps=1, rho=0.125, ndim=4)` — modifies active gauge in-place for the Dirac operator only
- `gauge_original` is kept on GPU and used exclusively for Wilson line construction

### 3. Color/spin index separation and explicit contraction

The Wilson line `W` is a pure color matrix (Nc×Nc), while propagators carry both color (Nc×Nc) and spin (4×4) indices. The revised plan defines the contraction precisely:
- `W^{ab}` contracts with the left (sink) color index of `S_c^{bc}_{αβ}(z,t)`
- The spin indices are then traced against `S_c†^{ca}_{βα}(0,t)`
- The einsum `'ab, bcxy, cayx ->'` captures this correctly: `a,b,c` are color, `x,y` are spin

### 4. Wilson line construction algorithm specified

For each t and z, the straight Wilson line is built from the original unsmeared U_z links:
```
W(0,z;t) = ∏_{k=0}^{z-1} U_z(x=0, y=0, z=k, t)   (left-to-right path ordering)
```
For z=0, W is the 3×3 identity and the correlator reduces to the standard local η_c two-point function `Tr[S_c† S_c]`.

## Known Risks (noted but not mitigated — follow user specification)

- **Gauge-background mismatch**: The propagator `S_c` was computed in a stout-smeared gauge field, while the Wilson line is built from original unsmeared links. The product `W·S_c` is not gauge-covariant without gauge fixing. This affects the renormalization and matching of the quasi-DA. The user explicitly requested this dual-gauge setup.
- **Single configuration, single point source**: Using only cfg 10000 with one point source provides no statistical sampling. The result will be dominated by a single gauge-field fluctuation with no error estimate.

## Technical Details (unchanged from original)

- **Ensemble**: C24P29, 24³×72 lattice, a ≈ 0.105 fm, N_f = 2+1 Clover
- **Charm mass**: a m_c = 0.4159, clover coefficient c_sw = 1.160920226
- **Solver**: CG with two-level multigrid preconditioner, tol=1e-12, maxiter=20000
- **MPI**: 4 ranks in time direction (process grid [1,1,1,4])
- **Output**: raw complex numbers to `etac_nonlocal_2pt_z0-10.txt`, no header