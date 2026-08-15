This revised plan keeps the original calculation target and fixed run settings, but makes the physics scope explicit and internally consistent.

The task remains a standard hadronic three-point calculation for the `Lambda -> Lambda` matrix element of the local strange vector current `\bar{s} \gamma_x s`, with the same local Lambda interpolator at source and sink, point source at `[0,0,0,0]`, zero momentum, `tseq = 8`, and stout-smeared links `(1, 0.125, 4)` for all valence inversions. The ensemble block is kept consistent with the provided fixed configuration, including the valence strange mass `m_s = -0.2356`.

The main correction is that the observable is now treated as the full flavor-diagonal three-point function, not just the connected strange-line diagram. The plan therefore separates and computes:
- the connected sequential-source contribution with the current on the strange valence line,
- the disconnected strange-loop contribution estimated from stochastic strange volume-source inversions,
- and the final full correlator as the sum of those two pieces.

The previous overstatement about the projector is also fixed. The plan keeps the requested projector exactly as `Tmat = (I + gamma_t)/2`, but no longer claims that this setup extracts a meaningful nonzero forward strange-vector matrix element by itself. With `p_i = p_f = 0`, `q = 0`, and an unpolarized positive-parity projector, the spatial current `gamma_x` is symmetry-suppressed in the rest frame, so the result should be interpreted as a diagnostic three-point correlator expected to be consistent with zero within noise after averaging, rather than as a charge-like observable.

Finally, the output remains a plain text file in the run directory with no header. To preserve the full task content without adding extra files, the text layout now includes an integer contribution label so connected, disconnected, and full results can all be stored in the same headerless file for each insertion time.