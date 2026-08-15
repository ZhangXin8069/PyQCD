## Revised plan summary

This revision keeps the task in **freeform gauge-only mode** and tightens it into an executable Wilson-loop measurement plan. The physics target is now stated unambiguously as a **per-configuration** measurement on **cfg 10000**, not an ensemble expectation value.

### What is computed
The script measures the rectangular Wilson loop
\(W(R=3,T=1)\)
averaged over the three spacetime orientations:
- XT
- YT
- ZT

using only the gauge links from the loaded configuration.

### What was fixed
The revised plan removes gauge-irrelevant fermion workflow content from the executable logic: no propagators, no Dirac solves, no inversions, no hadron operators. The ensemble block is preserved exactly as required for configuration consistency, but the freeform execution path now uses only the gauge-field information relevant to the Wilson-loop task.

### Numerical/execution details
The plan now specifies:
- exact loop paths for XT, YT, ZT at \(R=3, T=1\);
- the required PyQUDA 4-group packing for `gauge.loop()`;
- extraction of per-site real traces from each active plane;
- the exact local field shape for `gatherLattice` with process grid `[1,1,1,4]`, namely `(2, 18, 24, 24, 12)`;
- a runtime check that the launched MPI size matches the requested process grid;
- normalization in one unambiguous formula,
  \[
  W_{\mathrm{avg}} = \frac{\sum_x [\mathrm{ReTr}_{XT}(x)+\mathrm{ReTr}_{YT}(x)+\mathrm{ReTr}_{ZT}(x)]}{3 V N_c},
  \]
  with \(N_c\) taken from the loaded gauge object rather than hard-coded.

### Output behavior
The output contract is now explicit and executable:
- resolve the **run directory** as the current working directory;
- write the result to `./wilson_loop_R3_T1.txt`;
- **only rank 0** writes;
- the file contains **exactly one numeric line** with no header, label, or metadata.

Overall, the new plan is physically cleaner, numerically safer, and closer to a production-ready pure-gauge Wilson-loop script.