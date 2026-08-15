## Revised plan summary

This revision keeps the original physics target unchanged: a **charged-pion nonlocal two-point correlator** with a **wall source at `t=31`**, **body-diagonal source momenta**, and a **bilocal sink operator** connected by a recursively defined averaged diagonal Wilson-line displacement up to `z=12`.

### What is preserved
- The task remains a **standard hadronic correlator workflow**.
- The pion operator content is unchanged:
  - source gamma: `gamma.gamma(7) @ gamma.gamma(15)`
  - sink gamma: `gamma.gamma(15) @ gamma.gamma(7)`
- The five required wall-source light propagators are retained with momenta:
  - `0`, `+1`, `+2`, `-1`, `-2` along `(1,1,1)`
- The four physical channel pairings are retained:
  1. `(0,0)`
  2. `(+1,-1)`
  3. `(+1,-2)`
  4. `(+2,-1)`
- The output remains a **single `.npy` dictionary** with complex data of shape **`(4, 2, 13, 1, 1, 72)`**.

### Key fixes made
1. **Gauge-field usage is now explicit and executable**
   - The ensemble block is preserved exactly as required by the fixed configuration.
   - At the same time, the runtime plan now explicitly states that the actual computation must load the **Coulomb-gauge-fixed `.scidac` file** from the provided path.
   - The same Coulomb-gauge-fixed links are used:
     - **before stout smearing** for the custom `covDev` Wilson-line transport,
     - and as the base field for the **stout-smeared inversion links**.

2. **Solver setup is tightened to the requested task values**
   - The propagator solver entries now consistently encode:
     - mass `-0.2770`
     - clover `1 / 0.951479**3 = 1.160920226`
     - tolerance `1e-6`
     - maxiter `1000`
   - The plan explicitly states that the **task-specific multigrid block is `[[4,4,4,4]]`** for these inversions.

3. **Measurement graph now reflects the real channel assembly**
   - The structured measurement block no longer implies a single trivial `p=0` correlator only.
   - It now enumerates the four requested physical channels so the executor has a complete dependency graph for channel assembly.

4. **Contraction is defined physically, not by an ad hoc conjugation rule**
   - The revised plan avoids prescribing a naive manual complex-conjugation implementation for the anti-line.
   - Instead, it requires the **correlator-einsum generator** to build the correct **Hermitian-conjugate spin-color structure** for the nonlocal meson 2pt with the specified source/sink gamma matrices and one custom-displaced line.

5. **The nonlocal observable contract is made explicit**
   - The observable is promoted from an underspecified "other" description to an explicit **custom nonlocal meson-2pt contract**:
     - wall-source light propagators,
     - one line optionally displaced by recursive diagonal `covDev` transport,
     - source/sink gamma matrices fixed,
     - spatial sum reduced to time.

6. **Directional ambiguity in the diagonal shift is removed**
   - The plan no longer leaves negative directions as abstract `(4,5,6)` labels.
   - It now requires mapping to the library’s actual **`+x,+y,+z,-x,-y,-z` direction constants** before calling `covDev`, which makes the implementation unambiguous.

### Wilson-line construction used in the revised plan
For each diagonal step, the transported field is built as the average over the 6 shortest axis orderings:
- `(x,y,z)`, `(x,z,y)`, `(y,x,z)`, `(y,z,x)`, `(z,x,y)`, `(z,y,x)`

The same rule is applied for the negative body diagonal using `-x,-y,-z`. The displaced fields are then built **recursively**:
- `z=0`: unshifted field
- `z>0`: apply one averaged diagonal step to the already averaged field at `z-1`

This preserves the legacy convention and avoids replacing it by either a standard straight-link displacement or a full enumeration of all shortest paths of length `3z`.

### Final result
The revised plan is more executable while preserving the original objective. It now cleanly specifies:
- the runtime gauge input,
- the inversion links versus transport links,
- the five propagators,
- the four channel pairings and sink phases,
- the recursive diagonal transport rule,
- the correct contraction-generation path,
- and the final output format and axis ordering.