## Physics Objective

Compute the two-point correlation function of the $\Lambda$ baryon on a single gauge configuration (cfg 10000) to extract its positive-parity ground-state signal. The correlator is defined as

$$C_\Lambda(\vec{p}=\vec{0}; t, 0) = \mathrm{Tr}\left[ T \, \langle \mathcal{O}_\Lambda(\vec{0}, t) \, \bar{\mathcal{O}}_\Lambda(\vec{0}, 0) \rangle \right], \qquad T = \frac{I + \gamma_4}{2},$$

with the standard interpolating operator

$$\mathcal{O}_\Lambda = \epsilon^{abc} (u^{Ta} C\gamma_5 d^b) s^c.$$

## Wick Contraction (Single Term)

Because the $\Lambda$ contains three **distinct** quark flavours ($u$, $d$, $s$) there is exactly one contraction term — no exchange contribution. The $u$ and $d$ fields, though degenerate in mass on this ensemble ($S_u = S_d = S_l$), are distinct field labels and cannot contract with each other's antiquarks. The contraction yields:

$$C_\Lambda(\vec{0}; t, 0) \approx \sum_{\vec{x}} \epsilon^{abc} \epsilon^{a'b'c'} (C\gamma_5)_{\alpha\beta} (C\gamma_5)_{\alpha'\beta'} T_{\gamma'\gamma} \, S_{l,\alpha\alpha'}^{aa'}(\vec{x},t) \, S_{l,\beta\beta'}^{bb'}(\vec{x},t) \, S_{s,\gamma\gamma'}^{cc'}(\vec{x},t).$$

## Strategy

- **Source**: Single point source at $[0,0,0,0]$ (origin). Both light and strange propagators share this source. No Gaussian/Wuppertal smearing is applied to the quark source.
- **Gauge smearing**: One iteration of stout smearing ($\rho=0.125$, 4-dimensional) is applied to the gauge links **before** building the Wilson-clover Dirac operator.
- **Solver**: Wilson-clover CG inverter (tolerance $10^{-12}$, max 2000 iterations) using ensemble parameters $m_l = -0.277$, $m_s = -0.2356$, $c_{\mathrm{SW}} = 1.160920226$.
- **Projector**: $T = (I + \gamma_4)/2$ isolates the forward-propagating positive-parity channel. The anti-periodic temporal boundary conditions of the three-quark state imply a backward-propagating negative-parity partner; the projector suppresses it.
- **Contraction**: Delegated to the `generate_einsum(type="baryon_2pt")` tool, which produces the correct spin-colour einsum with the proper sign.
- **Output**: Time-slice correlator $C_\Lambda(t)$ for $t=0,\dots,T-1$ saved as a single-column ASCII file (no header) in the run directory.

## Technical Details

- **Lattice**: $24^3 \times 72$, $a = 0.1052$ fm, anti-periodic temporal BC for fermions.
- **Gamma basis**: DeGrand-Rossi (PyQUDA default), with $\gamma_t = \gamma_4$ diagonal and $C = \gamma_2\gamma_4$ real symmetric.
- **Single configuration**: cfg 10000 only; scaling to higher statistics is straightforward by looping over additional configurations.
- **Propagator count**: Two copies of `prop_l` (re-used, same inversion) plus one `prop_s`.