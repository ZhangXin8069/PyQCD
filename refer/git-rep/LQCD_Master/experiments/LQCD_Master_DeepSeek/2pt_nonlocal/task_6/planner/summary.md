## Physics Objective

Compute the nonlocal two‑point correlation function of the $D^+$ meson ($c\bar{d}$, pseudoscalar channel) on a single gauge configuration (cfg 10000) from the C24P29 ensemble ($24^3\!\times\!72$, $a=0.1052$ fm, clover fermions $N_f=2+1+1$).

The nonlocal sink operator displaces the **charm quark** field by $z$ lattice units along $+z$, connected by a straight Wilson line:

$$\mathcal{O}_{D^+}(\vec{x},t;z) = \bar{d}(\vec{x},t)\,\gamma_5\,W(\vec{x},t;\vec{x}+z\hat{e}_z,t)\,c(\vec{x}+z\hat{e}_z,t)$$

## Corrected Contraction Formula

After the full Wick contraction and $\gamma_5$-hermiticity simplification, **both $\gamma_5$ matrices cancel** and the correct two‑point function is:

$$C(z,t) = \sum_{\vec{x}} \mathrm{Tr}\!\big[ S_l^\dagger(\vec{x},t) \; W_{\text{orig}}(\vec{x},\vec{x}+z\hat{e}_z,t) \; S_c(\vec{x}+z\hat{e}_z,t) \big]$$

where $S_l^\dagger$ is the **plain Hermitian conjugate** (not the $\gamma_5$-dagger).  For $z=0$ this correctly reduces to the standard local $D^+$ correlator $\sum_{\vec{x}} \mathrm{Tr}[S_l^\dagger S_c]$.

The previous plan's formula $\mathrm{Tr}[S_l^\dagger\,\gamma_5\,W\,S_c\,\gamma_5]$ evaluates to $\mathrm{Tr}[S_l\,W\,S_c]$ (via $\gamma_5 S_l^\dagger\gamma_5=S_l$), which differs from the correct $\mathrm{Tr}[S_l^\dagger\,W\,S_c]$ and would produce wrong physics at all $z$.

## Strategy

1. **Propagator inversions** – Two point‑source propagators from the origin $[0,0,0,0]$ using **stout‑smeared** gauge links ($\rho=0.125$, $n_{\text{step}}=1$, $n_{\text{dim}}=4$) and the multigrid solver:
   - `prop_l`: light (down) quark, mass $-0.277$
   - `prop_c`: charm quark, mass $0.4159$
   - If multigrid fails for the heavy charm quark, **fall back to a plain CG solver** with the same tolerance and maxiter.

2. **Original gauge field preservation** – The original unsmeared gauge links are loaded and **kept as a separate copy before stout smearing is applied**, so both fields are available during the contraction.

3. **Wilson line construction** – For each $z=0,\dots,10$ and each spatial point $\vec{x}$ at timeslice $t$, the product of original $U_z$ links is built sequentially:
   $$W(\vec{x},\vec{x}+z\hat{e}_z) = U_z^{\text{orig}}(\vec{x})\,U_z^{\text{orig}}(\vec{x}+\hat{e}_z)\,\cdots\,U_z^{\text{orig}}(\vec{x}+(z-1)\hat{e}_z)$$
   For $z=0$, $W=I$.

4. **Periodic boundary conditions** – Both the charm‑propagator index $S_c(\vec{x}+z\hat{e}_z,t)$ and the Wilson‑line stepping wrap modulo $L_z=24$ at the $z=23\leftrightarrow 0$ boundary.

5. **Contraction** – Evaluated in the pseudoscalar channel using the corrected trace above.  Real and imaginary parts are saved for every $z$ and $t$.

6. **Output** – Plain text file `dplus_nonlocal_2pt_cfg10000.txt` with columns `z  t  Re[C]  Im[C]`, no header.