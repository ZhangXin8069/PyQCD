## Revised plan summary

This revision keeps the original objective and user choices intact, but makes the setup physically clearer and executable.

The task is now stated explicitly as a **single-configuration debug/smoke-test** for an **ssc baryon two-point correlator**, rather than a production spectroscopy result. That fixes the earlier overclaiming problem from using only one source on one configuration.

On the physics side, the operator is kept exactly as requested:
\[
\mathcal O(x)=\epsilon^{abc}(s^{Ta} C\gamma_x s^b)c^c,
\]
with the positive-parity projector
\[
T_{\rm mat}=(I+\gamma_t)/2.
\]
The plan now states clearly that this is a **user-fixed vector-diquark component** and that positive parity alone does **not** separately enforce a pure spin-1/2 projection against possible overlap with the broader vector-diquark channel. That addresses the main channel-identification criticism without changing the requested operator.

The measurement chain is now closed: the baryon correlator explicitly depends on **both** required propagators, `prop_s` and `prop_c`, with flavor structure `[s, s, c]`, diquark gamma `Cg1`, spectator gamma `I`, and zero-momentum projection. The previous incomplete measurement block referred only to the strange propagator; that is corrected.

The contraction prescription is also fixed: instead of hard-coding topology signs for the identical-`ss` case, the plan now requires a **validated baryon 2pt contraction/einsum generation step**. This is the safe and collaboration-style choice for identical-flavor baryons.

For inversions, the user’s stout-link request is preserved exactly: all propagators are solved on a **single stout-smeared gauge copy** with parameters `(n_steps=1, rho=0.125, ndim=4)`. The strange and charm masses and clover coefficient remain consistent with the fixed ensemble block. At the same time, unsupported claims about multigrid use for charm are removed; the plan now says not to assume charm multigrid unless the runtime stack explicitly validates it.

The saved observable is now better specified for downstream use: the output is the **trace-projected complex correlator** after zero-momentum spatial summation, stored as plain text columns
` t  Re[C(t)]  Im[C(t)] `
with **no header** and no extra text, exactly as requested. The plan also states that no extra `1/V` normalization is applied unless the executor explicitly documents such a convention.

Overall, the revised plan preserves the requested operator, source, smearing, projector, and output format, while fixing the missing propagator linkage, removing unjustified sign assumptions, and clarifying the physical interpretation of the chosen `Cγ_x` baryon channel.