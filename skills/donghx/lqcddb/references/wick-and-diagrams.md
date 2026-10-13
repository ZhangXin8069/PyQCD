# Wick expansion and diagram equivalence

## Conventions first

- Declare the quark-field ordering and whether the source operator has already been conjugated.
- Use `conjugate_operator` for the source and include its returned coefficient in the total diagram coefficient.
- Confirm flavor creation/annihilation counts before interpreting output.
- The current `wick()` defaults for `P`, `V`, and `G` are empty strings. Supply explicit prefixes when source/sink index separation is required; do not assume automatic upper-case prefixes from README prose.
- Validate a pion two-point sign and a baryon two-point permutation pattern against a hand Wick expansion before scaling up.

## Diagram inspection

For each term, record:

- numerical coefficient and fermion sign;
- each perambulator's flavor and source/sink time;
- gamma/charge-conjugation factors and transposes;
- vertex identity, momentum, displacement, and time;
- free output indices and final tensor shape;
- connected components in the quark-line graph.

Reject unsupported operator tokens instead of allowing them to disappear silently.

## Equivalence is opt-in

The current `identify_equivalent_diagrams()` normalization primarily compares perambulator index structure. It does not establish equality of all gamma tensors, vertices, time labels, or physical operator identities. Therefore:

- default `use_equivalence=False` for production calculations;
- never group diagrams from distinct operator groups merely because their einsum strings match;
- require identical tensor names and identities, exact time labels or a proved translation, and the same connected/disconnected topology;
- for `gamma_mu`, `gamma5 gamma_mu`, `sigma_mu_nu`, or other parameterized structures, derive the component-wise transpose sign instead of accepting a generic `Gamma^T = Gamma` fallback;
- compare grouped and ungrouped results on a small deterministic tensor set before enabling the optimization.

`ignore_dis` is a time-label heuristic in the current dynamic implementation, not a graph-theoretic connectedness proof. Classify disconnected diagrams from the contraction graph.

## Production handoff

After validating automatic Wick output, freeze a reviewed contraction plan with explicit tensor mappings and documented signs. Automatic generation is a derivation aid, not the final proof of a production correlator.
