This revised plan keeps the task as a pure-gauge freeform measurement and makes the Wilson-loop workflow more precise and executable.

The physical observable is the rectangular Wilson loop with spatial extent \(R=3\) and temporal extent \(T=2\), computed separately in the XT, YT, and ZT planes and then averaged with equal weight:
\[
W_{\mathrm{avg}}(3,2)=\frac{W_{XT}(3,2)+W_{YT}(3,2)+W_{ZT}(3,2)}{3}.
\]
For each plane, the path is explicitly fixed as
- XT: \(+X,+X,+X,+T,+T,-X,-X,-X,-T,-T\)
- YT: \(+Y,+Y,+Y,+T,+T,-Y,-Y,-Y,-T,-T\)
- ZT: \(+Z,+Z,+Z,+T,+T,-Z,-Z,-Z,-T,-T\)

The main corrections relative to the previous version are:
- the gauge-only nature is now stated more cleanly, with fermion-related ensemble entries explicitly marked as non-operative provenance rather than inputs to the measurement logic;
- the reduction chain is now unambiguous: extract per-site color matrices, compute \(\mathrm{ReTr}\), reshape the real field back to PyQUDA’s required local even-odd lattice layout, then apply `core.gatherLattice` in MPI mode;
- execution handling now includes both the preferred recorded process grid `[1,1,1,4]` and an explicit single-rank fallback `[1,1,1,1]`;
- the output location is now defined concretely as the current working directory, i.e. the run directory;
- the text output format is fixed to a single numeric line in deterministic scientific notation;
- a minimal runtime validation step is added by checking that the three plane values and the final average are finite and that the normalized Wilson loop lies in the sensible range `[-1,1]`.

The ensemble block is preserved exactly as required by the fixed configuration. At the same time, the plan makes clear that the Wilson-loop calculation itself uses only the gauge links and lattice geometry, with no propagators, solvers, or inversions.

The final deliverable is one txt file written by rank 0 only:
`wilson_loop_R3_T2_avg_cfg10000.txt`
containing exactly one formatted scalar value and nothing else.