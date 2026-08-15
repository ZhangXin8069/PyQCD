## Physics Objective

Compute the proton two-point correlation function $C_p(\vec{p}=\vec{0}; t, 0)$ on a single $24^3 \times 72$ Clover-Wilson gauge configuration (ensemble C24P29, $a \approx 0.1052\,\text{fm}$) to enable subsequent ground-state mass extraction.

## Core Strategy

- **Interpolating operator**: Standard proton operator $\mathcal{O}_p = \epsilon^{abc} (u^{Ta} C\gamma_5 d^b) u^c$ with the $C\gamma_5$ (`Cg5`) diquark structure.
- **Parity projection**: Apply $T_{\text{mat}} = \frac{I + \gamma_4}{2}$ to isolate the positive-parity ground state.
- **Propagator**: Single light-quark propagator (`prop_l`) from a point source at $[0,0,0,0]$, reused for all three quark lines via isospin/flavor symmetry ($\text{prop}_u = \text{prop}_d = \text{prop}_l$).
- **Sink**: Point sink (no smearing), giving a point-point (PP) correlator.
- **Gauge field**: Stout-smeared links ($\rho=0.125$, $n_{\text{steps}}=1$, $n_{\text{dim}}=4$) used in the Dirac inversion to suppress UV fluctuations.
- **Solver**: Multigrid-accelerated CG with two coarse levels $[6,6,6,3]$ and $[4,4,4,6]$, clover coefficient $c_{\text{sw}}=1.160920226$, light quark mass parameter $-0.277$, tolerance $10^{-12}$.

## Wick Contraction

The two-point function involves two contraction topologies (direct and exchange) arising from pairing the sink $u$ quarks with the source $\bar{u}$ quarks. After color-index relabeling on the exchange term, $\epsilon$-antisymmetry flips the sign, and both terms combine with a **positive** sign. Applying $\gamma_5$-hermiticity and flavor symmetry yields:

$$C_p(\vec{p}; t,0) = \sum_{\vec{x},\vec{y}} e^{-i\vec{p}\cdot(\vec{x}-\vec{y})} \epsilon^{abc} \epsilon^{a'b'c'} (C\gamma_5)_{\alpha\beta} (C\gamma_5)_{\alpha'\beta'} P^+_{\gamma'\gamma} \Big[ S_{l,\alpha\alpha'}^{aa'} S_{l,\beta\beta'}^{bb'} S_{l,\gamma\gamma'}^{cc'} + S_{l,\alpha\gamma'}^{aa'} S_{l,\beta\beta'}^{bb'} S_{l,\gamma\alpha'}^{cc'} \Big]$$

With a point source at $\vec{x}_0=\vec{0}$, $t=0$, the estimator reduces to a single spatial sum at the sink:

$$C_p(\vec{0}; t,0) \approx \sum_{\vec{x}} \epsilon^{abc} \epsilon^{a'b'c'} (C\gamma_5)_{\alpha\beta} (C\gamma_5)_{\alpha'\beta'} P^+_{\gamma'\gamma} \Big[ S_{l,\text{pt},\alpha\alpha'}^{aa'}(\vec{x},t) S_{l,\text{pt},\beta\beta'}^{bb'}(\vec{x},t) S_{l,\text{pt},\gamma\gamma'}^{cc'}(\vec{x},t) + S_{l,\text{pt},\alpha\gamma'}^{aa'}(\vec{x},t) S_{l,\text{pt},\beta\beta'}^{bb'}(\vec{x},t) S_{l,\text{pt},\gamma\alpha'}^{cc'}(\vec{x},t) \Big]$$

**Sign verification**: The derivation in `proton_mass.md` Step 3c.2 yields no leading minus sign (both terms positive). However, Step 4 of the same reference inconsistently carries a minus sign. The `generate_einsum(type="baryon_2pt")` tool is the authoritative source; the executed contraction must match its output. As a cross-check, the positive-parity proton correlator at zero momentum is expected to be positive-definite at all $t$.

## Known Limitations

- **Point-source SNR**: Baryon correlators from point sources suffer rapid signal-to-noise degradation; the ground-state signal is typically lost beyond $t \approx 6$–$8$. Gaussian/Wuppertal source smearing is essential for production-quality proton spectroscopy (per the `Smeared source essential` rule in the LQCD-physics skill). This run will execute correctly but yields a correlator of limited quantitative utility.
- **Single configuration**: Running on only cfg 10000 provides a single noisy estimate with no statistical error. For quantitative results, use $O(100)$–$O(1000)$ configurations with multiple source positions per configuration.

## Output

Raw correlator $C(t)$ for $t=0,\dots,71$ written as whitespace-separated values to `proton_2pt.txt` in the run directory, without headers or metadata.