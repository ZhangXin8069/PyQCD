# Theory and matching contract

## Flowed operator

Use synchronized four-dimensional flow,

\[
\partial_t B_\mu=D_\nu G_{\nu\mu},\qquad
\partial_t\chi=\Delta[B(t)]\chi,
\]

with kernel `K(t)` and the same flowed links in the straight Wilson line:

\[
O_\Gamma^{GF}(z,t)=\bar\chi(t,0)\Gamma W_{B(t)}(0,z)\chi(t,z).
\]

In distillation, do not form the dense spacetime kernel. Apply it to all-time
basis vectors and build

\[
J(t_J;\tau_a,\tau_b)=[K V_{\tau_a}]^\dagger
O[B(t),t_J][K V_{\tau_b}].
\]

The current requires both source-time sums. A timeslice-diagonal construction
is 3D flow, not complete 4D flow.

## Ringed fermions

The published SFTE coefficient assumes

\[
\mathring\chi=Z_\chi^{1/2}\chi,
\qquad
Z_\chi(t)=\frac{-2N_c}{(4\pi)^2t^2
\langle\bar\chi\overleftrightarrow{\slashed D}\chi\rangle_{m=0}}.
\]

There are two endpoints, hence one overall `Z_chi` in the bilocal. It can be
applied after the bare C3/C2 fit if it is an ensemble/flow-time scalar. This
reuses existing C3 and requires no operator-flow inversion. Measuring the
vacuum trace may nevertheless need stochastic estimators or additional
inversions unless the existing all-to-all representation is demonstrably
adequate.

## Corrected one-loop SFTE

Use N. Brambilla and X.-P. Wang, JHEP 06 (2024) 210,
arXiv:2312.05032. It corrects/supersedes the affected one-loop result of
C. Monahan, PRD 97 (2018) 054507, erratum PRD 110 (2024) 059902.

For the ringed bilocal,

\[
O_\Gamma^{GF,R}(z,t)=C_\psi(t,\mu)e^{\delta m(t)|z|}
O_\Gamma^{MSbar}(z,\mu)+O(t),
\]

\[
C_\psi=1-\frac{\alpha_s C_F}{4\pi}
\left[3\ln(2\mu^2t e^{\gamma_E})+2+\ln432\right]+O(\alpha_s^2),
\]

\[
\delta m(t)=-\frac{\alpha_s C_F}{4\pi}
\frac{\sqrt{2\pi}}{\sqrt t}+O(\alpha_s^2).
\]

Thus

\[
h_\Gamma^{MSbar}=C_\psi^{-1}e^{-\delta m|z|}
Z_\chi h_\Gamma^{GF}+O(t).
\]

This asymptotic coefficient is independent of `Gamma` at this order, but the
later LaMET kernels remain channel dependent. The natural scale is

\[
\mu_t=e^{-\gamma_E/2}/\sqrt{2t}.
\]

Run/evolve to a common scale and propagate scale/order uncertainty.

## Window and order of limits

Require a tested region rather than merely positive flow time:

\[
a\ll r_F=\sqrt{8t}\ll |z|,\qquad
\sqrt t P_z\ll1,\qquad t\Lambda_{QCD}^2\ll1.
\]

Fit residual dependence using declared terms such as `a^2/t`, `t/z^2`,
`t Pz^2`, and `t Lambda^2`. Take the continuum limit at fixed physical `t`,
`z`, and `Pz`, or use a justified joint fit. Do not copy one dimensionless
flow-time list between ensembles with different lattice spacings.

For the L32x96 pilot at `t/a^2=0.5`, `r_F/a=2`; `z/a<=2` is outside the simple
asymptotic separation condition. `sqrt(t)Pz` is approximately 0.42, 0.56, and
0.69 for momentum integers 3, 4, and 5, so the largest momenta are marginal.

## Linear term and LaMET

The exponential contains the flowed line-mass term. Choose either the direct
one-loop prescription or a consistently derived nonperturbative hybrid/ratio
scheme. Never subtract the same power contribution twice.

After SFTE, apply the ordinary vector/unpolarized, axial/helicity, or
tensor/transversity LaMET kernel. For unpolarized and helicity singlets,
include disconnected terms and quark--gluon mixing. Transversity has no
leading-twist gluon mixing, but its implementation sign still needs external
confirmation in this workflow.

## 3D flow boundary

Three-dimensional flow damps spatial momenta and regulates the one-loop
straight-line self energy, but leaves arbitrarily large Euclidean energy
undamped. It does not inherit the standard 4D all-order finiteness theorem,
ringed normalization, or published matching coefficient. A 3D result needs a
separate renormalization theorem and matching calculation.
