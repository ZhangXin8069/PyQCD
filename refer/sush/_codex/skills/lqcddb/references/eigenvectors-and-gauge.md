# Eigenvectors, Vertices, Gauge Links, and Momentum Phases

## Eigenvector contract

The common current eigenvector layout is `(Nev,Nz,Ny,Nx,Nc)`. Flatten it as $E=\text{eigvecs.reshape}(N_{ev},-1)$ and compute
$$G=EE^\dagger,\qquad r=\max|G-I|.$$
If an external program stores columns, use $V^\dagger V$. Report layout, tolerance, diagonal normalization, and off-diagonal orthogonality.

Compression and noise workflows must also check group lengths, divisibility, retained rank, zero norms, linear dependence, finite output, random distributions, and unit-modulus roots. The current `random_type="Z_N"` path is not a general normalized-root-of-unity construction; `create_omega_accelerate(exact=0)` rejects its default value.

## Coordinates and phases

Vertex construction uses eigenvectors `(Nev,z,y,x,color)`, momentum triplets `(p_z,p_y,p_x)`, and
$$e^{-2\pi i(p_z z+p_y y+p_x x)/N_x}.$$
`phase_exp_VdV` returns color-replicated phases with shape `(Nx,Ny,Nz,Nc)`, while `phase_exp_VVV` returns a three-dimensional spatial phase. Check zero momentum, unit momentum in each coordinate direction, and conjugate momentum pairs, and deduplicate momentum shells before averaging.

For every VDV/VVV, state whether it is $V^\dagger O V$, its conjugate, or a displaced version with gauge links. In a hybrid workflow, multiply the declared $\Omega$ weight into VDV exactly once; for VVV, check every leg permutation and parity.

## Gauge links and Stout

The full gauge-link layout is `(Nd,Nt,Nz,Ny,Nx,Nc,Nc)` with directions `(X,Y,Z,T)`; a time-slice vertex is usually `(Nd,Nz,Ny,Nx,Nc,Nc)`. Build a correspondence table for directions, link axes, and spatial-roll axes before use. At every Stout step, check the identity field and $\rho=0$ invariants, finiteness, unitarity, determinant, and forward/backward path staples; the identity-field branch of the current implementation may still produce non-finite values.
