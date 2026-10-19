# Contraction Construction Reference

Use this reference to derive, review, and run lqcddb contractions. Use `api-quick-reference.md` for current signatures.

## Fixed conventions

Before generating code, record the Euclidean gamma basis and the definitions of `gamma_5` and `C`; the field order at the sink, current, and source; whether source-operator tokens are already $\mathcal O^\dagger$; the Fourier sign; spatial storage order `(z,y,x,color)`; gauge directions `(X,Y,Z,T)`; temporal boundary and number of wraparound fermion lines; and any per-configuration light-flavor equality assumption.

When `source_operator` is not provided, use the declared physical initial state with
$$
\mathcal O^\dagger=c^*H_n^\dagger\cdots H_1^\dagger
$$
and call `conjugate_operator`, preserving the complete returned list and leading coefficient. A user-provided source operator takes priority and is passed unchanged after validation. A transition process must not mistake the final sink operator for the initial state.

Use a supported meson or standard baryon for every `|...|` factor; expand component-dependent gamma matrices before conjugation. Check the syntax and flavor conservation of every token first.

## Wick diagrams and tensors

For each diagram, record `result_indx`, `result_sign`, peram pairs, V type/time, gamma names, and free output axes. Counts must balance flavor by flavor; preserve physical coefficients and fermion signs. Do not exchange VDV ports; exchange VVV ports only after proving antisymmetry and permutation parity.

`PeramRegistry` keys are `(flavor,(t_quark,t_antiquark))`, `VRegistry` keys are `(V_name,time_label)`, and `GammaRegistry` keys are token names. Record shape, axes, owner/local index, conjugation/transpose, backend, and dtype for every item. Write an axis table before writing einsum; `validate_plan` can prove only that names exist, not dimensions or physical direction.

## Equivalent diagrams and boundaries

The defaults are `use_equivalence=False` and `ignore_dis=False`. Before enabling equivalent diagrams, prove that gamma-transpose signs, VVV permutations, topology, time direction, and output axes are identical. The antiperiodic sign is determined by the actual number of fermion lines crossing the temporal boundary and is applied exactly once. Preserve the complex correlator unless a projection proves that its real or imaginary part may be taken.
