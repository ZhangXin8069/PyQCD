## Physics classification
This task is a **pure-gauge measurement** of a **Wilson loop**, so the correct category is **gauge-only / static-potential-type observable**, not spectroscopy or matrix elements. It does **not** use the standard hadrons → propagators → correlators chain, so `task_mode: freeform` is the physically correct choice.

## Core observable
The target quantity is the rectangular Wilson loop
\[
W(R=1,T=4)
\]
measured in the three spatial-temporal planes:
- XT
- YT
- ZT

and then averaged:
\[
W_{\rm avg} = \frac{W_{XT}+W_{YT}+W_{ZT}}{3}.
\]

This is a gauge-link observable built directly from closed products of SU(3) links. There are:
- no quark propagators,
- no Dirac solves,
- no source/sink construction,
- no Wick contractions,
- no renormalization step required for this raw measurement.

## Numerical strategy
I set up the plan as a direct PyQUDA gauge-field workflow:
1. initialize PyQUDA on the provided lattice and MPI grid,
2. load the gauge configuration `10000` from the supplied ensemble path,
3. construct the three rectangular loop paths for `(R,T)=(1,4)` in XT, YT, ZT,
4. call `gauge.loop()` using the required 4-group PyQUDA packing convention,
5. extract per-site real traces,
6. MPI-gather and normalize by `total_sites * Nc`,
7. average the three plane results,
8. save only the final averaged scalar to a plain `.txt` file in the run directory.

## Reasonable completions made
Because the user did not specify some low-level implementation choices, I completed them conservatively and in the standard way:
- **No link smearing**: I left the gauge links unsmeared, since the request was simply to measure the Wilson loop on the loaded configuration.
- **Single configuration only**: I restricted the run to cfg `10000`, exactly as provided.
- **Output file naming**: I chose a concrete plain-text filename `wilson_loop_R1_T4_avg_cfg10000.txt` in the run directory.
- **Root-only output**: only MPI rank 0 writes the file, which is the standard safe cluster behavior.
- **No metadata/header**: I enforced a raw scalar-only text output because the user explicitly requested no extra text.

## Requirement satisfaction check
- Correctly identified as **pure-gauge**: yes.
- Used **freeform** rather than misleading standard hadron sections: yes.
- Included the provided **ensemble** block directly with the given values: yes.
- Produced a physically executable measurement plan: yes.
- Avoided placeholders and left a runnable default strategy: yes.
- No citations included because no web pages were parsed: yes.