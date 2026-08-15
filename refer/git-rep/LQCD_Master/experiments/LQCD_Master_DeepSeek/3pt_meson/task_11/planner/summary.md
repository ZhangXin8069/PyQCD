## Physics Objective

Compute the three-point correlation function for the **K- to pi-** transition mediated by the **s to d vector current** J_x = bar{d} gamma_x s. This is a prototype for K_{ell 3} semileptonic decay form-factor calculations on the C24P29 ensemble (24^3 x 72, a ~ 0.105 fm, clover-Wilson fermions).

## Operators and Correlator Structure

| Component | Operator | Quark Content |
|-----------|----------|---------------|
| Source (K- creation) | O_K^-^dag = -bar{s} gamma_5 u | anti-u s |
| Sink (pi- annihilation) | O_pi^- = bar{u} gamma_5 d | anti-u d |
| Current insertion | J_x = bar{d} gamma_x s | s to d vector |

All interpolating operators use gamma_5 in the DeGrand-Rossi Euclidean Dirac basis.

## Wick Contraction and Sequential-Source Strategy

Only one connected diagram contributes (no disconnected pieces for this flavour-non-singlet transition). Applying gamma_5-hermiticity and the cyclic property of the trace, the 3pt correlator reduces to:

C_3(t_f, tau, 0) = sum_{x,z} Tr[ S_l^dag(x;0) S_l(x;z) gamma_x S_s(z;0) ]

with x = (x_vec, t_f), z = (z_vec, tau). The sequential-source method factorises this as:

1. **Forward solves**: compute light propagator prop_l (S_l) and strange propagator prop_s (S_s) from a point source at [0,0,0,0].
2. **Sequential source**: at t_f = 8, construct eta^seq = gamma_5 B^dag gamma_5 (two-dagger convention) from prop_l combined with the gamma_5 sink structure, then solve D_l G^seq = eta^seq to obtain prop_l_seq.
3. **Contraction**: C_3(tau) = sum_z Tr[ G^seq(z) gamma_x S_s(z) ] for each current time tau = 1,...,7.

No 2pt function is computed, as requested.

## Numerical Details

- **Ensemble**: C24P29, single configuration 10000 (LIME format)
- **Gauge smearing**: 1-step stout (rho = 0.125, 4-dim) on all Dirac inversions
- **Source**: point source at [0,0,0,0], zero momentum
- **Sink**: t_seq = 8, point sink
- **Solvers**: multigrid-preconditioned CG; light mass tolerance 1e-12, strange tolerance 1e-10, max 20000 iterations
- **MPI**: 4 ranks in process grid [1,1,1,4]
- **Output**: raw C_3(tau) values (7 lines, tau = 1..7) to K_to_pi_3pt_result.txt, no headers

## Reasonable Assumptions Made

- **Solver tolerances**: conservative 1e-12 (light) and 1e-10 (strange), standard for clover-Wilson fermions at these quark masses
- **Max iterations**: 20000 for multigrid-preconditioned CG
- **Stout ndim**: 4 (all directions), the standard choice for gauge-link smearing in propagator inversions
- **Current insertion range**: all integer tau from 1 to t_f-1 = 7
- **Single configuration**: cfg_num contains only '10000'; the plan computes one 3pt correlator for that configuration
- **Contraction sign**: deferred to the generate_einsum tool, which derives the correct sign from Wick-contraction topology