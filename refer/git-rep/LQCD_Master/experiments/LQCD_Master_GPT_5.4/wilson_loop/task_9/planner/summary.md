## Revised plan summary

This revision keeps the task in freeform gauge-only mode, but tightens the physics scope and execution details.

### What is being computed
The output is explicitly defined as a **per-configuration Wilson-loop estimator** on **cfg 10000**, not an ensemble-averaged physics result. The measured quantity is the bare rectangular loop
\(W(R=3,T=3)\), averaged equally over the **XT**, **YT**, and **ZT** planes:
\[
W_{\mathrm{avg}} = \frac{W_{XT}+W_{YT}+W_{ZT}}{3}.
\]
This avoids the previous ambiguity between a single-configuration number and a full observable.

### Physics scope corrected
The plan now states clearly that this is a **pure-gauge Wilson-loop measurement only**. It does **not** attempt static-potential extraction, effective-potential analysis, or any fermionic calculation. The ensemble block is preserved exactly as required, but fermionic entries there are marked as unused metadata for this task.

### Executable reduction chain tightened
The main implementation risk in the old plan was the vague host-field reshaping step. The revised plan closes that gap by fixing the local gather shape for the stated geometry and MPI grid:
- global lattice: \(24^3 \times 72\)
- process grid: \([1,1,1,4]\)
- local time extent: \(72/4 = 18\)
- local even-odd scalar field shape for `gatherLattice`: **(2, 18, 24, 24, 12)**

For each active plane result from `gauge.loop()`:
1. transfer to host,
2. flatten only site dimensions while keeping the final color indices,
3. compute per-site \(\mathrm{ReTr}\),
4. reshape to `(2,18,24,24,12)`,
5. gather to rank 0,
6. divide exactly once by \(V N_c\), with \(V = 24\cdot24\cdot24\cdot72\) and \(N_c=3\).

This makes the normalization and MPI reduction explicit and checkable.

### 4-group padding clarified
The PyQUDA `gauge.loop()` packing is now stated unambiguously:
- `groups = [[XT], [YT], [ZT], [XT]]`
- `weights = [1, 1, 1, 0]`

The fourth slot is **dummy padding only** and must be ignored completely. Only result indices **0, 1, 2** enter the physics average.

### Minimal validation added
Before averaging, the plan now requires computing **XT**, **YT**, and **ZT** separately and checking that each is a finite real scalar. If any one fails, no txt file should be written. This is a useful guard against silent path-packing or indexing mistakes.

### Output contract closed
The output behavior is now fully specified:
- filename: `wilson_loop_R3_T3_avg_cfg10000.txt`
- location: current run directory
- writer: **rank 0 only**
- content: **one plain decimal floating-point number only**
- trailing newline: allowed
- overwrite existing file: yes
- normal success mode: no informational stdout text
- non-root ranks: no file writing

This matches the user's strict “no header or extra text” requirement.

### Bottom line
The revised plan is still simple, but now it is more scientifically precise and more executable: it is clearly a **single-configuration gauge-only measurement**, the `gauge.loop()` handling is explicit, the dummy slot cannot be accidentally double-counted, and the file output behavior is fully pinned down.