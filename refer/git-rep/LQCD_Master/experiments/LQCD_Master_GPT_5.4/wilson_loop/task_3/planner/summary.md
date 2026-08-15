## Revised plan summary

This revision keeps the task in the correct **freeform pure-gauge** mode and removes the misleading fermion-workflow logic from the executable part of the plan. The computation is now stated explicitly as a **gauge-link-only Wilson-loop measurement** on **configuration 10000** of the **C24P29** ensemble.

The observable is defined more precisely than before: for each of the three spatial-temporal planes **XT, YT, ZT**, the code must measure the **translationally averaged** rectangular Wilson loop
\[
W_{\mu T}(2,1)=\frac{1}{N_c V}\sum_x \operatorname{ReTr} U_{\mu T}(x;2,1),
\]
using the full per-site loop field returned by `gauge.loop()`. The final reported result is the arithmetic mean
\[
W_{\mathrm{avg}}=\frac{W_{XT}+W_{YT}+W_{ZT}}{3}.
\]
This makes clear that the estimator is a lattice-site average over all loop origins, not a single loop evaluation.

The MPI contract is also tightened. For each plane separately, the script must compute local per-site `ReTr`, perform the **global lattice reduction first**, and only then divide once by **`total_sites * Nc`** on rank 0. This avoids ambiguous or incorrect normalization in distributed runs.

The PyQUDA 4-group requirement is now handled safely: the three physical paths occupy result slots 0, 1, and 2, while slot 3 is a zero-weight padding group used only to satisfy the API. The revised plan explicitly requires that **only the first three returned groups** enter the final average.

Output behavior is now operationally deterministic. The “run directory” is defined as the **current working directory at job launch**, and the file name is fixed to **`wilson_loop_R2_T1_avg.txt`**. Only **rank 0** may write, and the file must contain **exactly one numeric scalar** with no header or extra text.

Finally, the plan states clearly that this is a **per-configuration utility measurement**. It produces a valid Wilson-loop number for the requested configuration, but not by itself an ensemble expectation value or uncertainty estimate.