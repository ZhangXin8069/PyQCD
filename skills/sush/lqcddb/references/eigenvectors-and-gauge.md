# Eigenvectors, gauge links, smearing, and momentum phases

## Eigenvector validation

For a matrix whose columns are eigenvectors, calculate

`G = V.conj().T @ V` and `residual = max(abs(G - I))`.

Use an explicitly justified tolerance and report both diagonal normalization and off-diagonal orthogonality. Do not rely solely on `vector_creator.check()` or `vertex_creator.check()` in the current source: their off-diagonal comparison can miss negative or complex overlaps.

Compression validation must also check:

- input group lengths and divisibility;
- zero norms and linearly dependent vectors;
- output rank and orthogonality;
- stochastic-vector distribution and unit-modulus roots for a claimed `Z_N` ensemble.

The current V4 string `Z_N` is not a usable literal parameter, and its hard-coded higher phases are not a general normalized roots-of-unity construction.

## Coordinates and phases

- Fix the spatial storage order, such as `(z,y,x,color,eigen)` versus `(x,y,z,...)`, before generating phases.
- Verify `exp(+i p.x)` versus `exp(-i p.x)` from the Fourier convention.
- Test zero momentum, a single-axis unit momentum, and a conjugate momentum pair.
- Momentum-list generation can contain duplicates; validate and deduplicate shells before averaging.

## Gauge links and smearing

Establish a direction table mapping `mu` to the array axis. Check both forward and backward staples on an identity field and on a small random SU(3) field.

For every smearing step verify:

- identity and `rho=0` are fixed points;
- all results are finite;
- `U U^dagger = I` within tolerance;
- `det(U) = 1` within tolerance;
- only intended spatial/time directions are updated.

Do not use the current Stout-smearing routine for production without this independent validation. Its documented rank and internal transpose differ, the zero-generator branch can divide by zero, and the present spatial roll-axis mapping requires review.
