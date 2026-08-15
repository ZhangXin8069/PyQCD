## Physics Objective

Compute the three-point correlation function for the semileptonic decay $B^- \to \pi^-$ via the $(b \to d)$ vector current $J_x = \bar{d}\,\gamma_x\,b$. This prototype targets zero momentum transfer, relevant for the $B\to\pi$ form factor $f_+(q^2)$ entering $|V_{ub}|$.

## Operators and Contraction

- **Source** ($t=0$, $\vec{p}=\vec{0}$):  $B^-$ created by $\mathcal{O}_{B^-}^\dagger = \bar{b}\,\gamma_5\,u$ at point $[0,0,0,0]$.
- **Sink** ($t=t_{\rm seq}=8$, $\vec{p}=\vec{0}$):  $\pi^-$ annihilated by $\mathcal{O}_{\pi^-} = \bar{u}\,\gamma_5\,d$.
- **Current** ($\tau$):  $J_x = \bar{d}\,\gamma_x\,b$.

The Wick contraction yields a single closed fermion loop with an overall minus sign. Under isospin symmetry ($S_u=S_d=S_l$) and $\gamma_5$-hermiticity, the correlator reduces to

$$C_3(\tau) = -\sum_{\vec{x},\vec{z}} \operatorname{Tr}\big[S_l(x,z)\,\gamma_x\,S_b(z,0)\,\gamma_5\,S_l(x,0)\,\gamma_5\big],$$

with $x=(\vec{x},t_f)$ and $z=(\vec{z},\tau)$.

## Propagators

Three inversions on stout-smeared links ($n_{\rm steps}=1$, $\rho=0.125$, $n_{\rm dim}=4$):

| Propagator | Quark | Source | Solver |
|------------|-------|--------|--------|
| `prop_l_fwd` | light | point $[0,0,0,0]$ | multigrid, tol $10^{-12}$ |
| `prop_b_fwd` | bottom | point $[0,0,0,0]$ | CG, tol $10^{-10}$, max 20000 iter; **convergence checked, config flagged on failure** |
| `prop_l_seq` | light (sequential) | spatial sum at $t=8$, constructed from $\text{prop\_l\_fwd}^\dagger$ | multigrid, tol $10^{-12}$ |

## Sequential Source (Revised)

The sequential source spans **all spatial points** at the sink time slice (not a single point). It is built from the conjugate transpose of `prop_l_fwd` via the two-dagger convention:

$$\eta^{\rm seq}(\vec{x}) = \gamma_5\,B^\dagger(\vec{x})\,\gamma_5, \qquad B(\vec{x}) = S_l(\vec{x},t_f;0,0)$$

After solving $D_l\,G_l^{\rm seq} = \eta^{\rm seq}$, the three-point function becomes

$$C_3(\tau) = -\sum_{\vec{z}} \operatorname{Tr}\big[G_l^{\rm seq}(\vec{z},\tau;0,0)\,\gamma_x\,S_b(\vec{z},\tau;0,0)\big].$$

## Key Revisions from Peer Review

1. **Sequential source spatial extent**: Changed from a single spatial point to a full spatial sum at $t=8$ with zero-momentum projection.
2. **Dagger convention**: Explicitly added `use_dagger: true` and documented the two-dagger steps to ensure the conjugate transpose is used, not the raw forward propagator.
3. **Bottom-quark convergence guard**: Added `convergence_check: true` and `abort_on_failure: true` so that CG failures are flagged rather than silently producing corrupted 3pt data.
4. **Sink $\gamma_5$ handling**: Clarified that the sink interpolator's $\gamma_5$ enters the sequential source via the standard two-dagger construction and cancels in the final trace.
5. **Code-generator interface**: Noted that `generate_einsum(type="meson_3pt")` is required; provided a fallback manual trace expression if `meson_3pt` support is not yet available.

## Output

Raw three-point correlator values as plain text (`.txt`), one value per $\tau$, with no header or metadata.