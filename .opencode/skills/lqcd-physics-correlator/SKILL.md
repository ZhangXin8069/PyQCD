---
name: lqcd-physics-correlator
description: >
  Lattice QCD correlator reasoning skill. Derives the chain from a physics
  observable to a computable correlator: interpolating operator
  construction (mesons, baryons), correlator definition (two-point,
  three-point), Wick contraction with $\gamma_{5}$-hermiticity and flavor
  symmetry, propagator requirements, and einsum expressions for
  contractions. Hands spectral decomposition off to lqcd-physics-spectrum. 
  Trigger on: hadron masses, form factors, local two-point correlation function, 
  nonlocal two-point correlation function, matrix elements, operator construction,
  Wick contraction, disconnected diagrams, three-point correlation function,
  weak current insertion, sequential source,
  or "what correlators/propagators do I need".
  For pure-gauge observables (Wilson loops, Polyakov loops, static potential)
  use the pyquda-gauge skill instead.
---

# LQCD Physics Reasoning

## Purpose

> **Notation note**: In the code, the light quark propagator is called `prop_l` and the strange propagator `prop_s`. The LaTeX $S_l$ and $S_s$ in the formulas below correspond to `prop_l` and `prop_s` respectively.

Given a physics observable such as hadron mass and decay constant that can be determined by local two-point correlation function (2pt), quasi-distribution amplitudes from nonlocal two-point correlation function, form factor that can be determined by three-point correlation function (3pt), ... ,
derive the complete chain:

  **Observable $\rightarrow$  Operator(s) $\rightarrow$  Correlator(s) $\rightarrow$  Wick contraction $\rightarrow$  Propagator(s) $\rightarrow$  Einsum(s)**

so that downstream tools (PyQUDA) know exactly what to compute.

For three-point correlation function, this skill additionally provides the full sequential-source implementation chain from
operator definition to executable contraction code, including the two-dagger
convention, propagator requirements, and final contraction forms.

This skill stops once the correlator expression, propagator list, and
contraction/einsum structure are fixed.

**IMPORTANT**: The contraction/einsum structure is for REASONING only. Never include it in the planner's output — the executor will regenerate it via the generate_einsum tool. For Euclidean-time dependence,
backward-state structure, and fit templates, hand off to
`lqcd-physics-spectrum`.


## Core workflow for Correlators

### Step 1: Identify the interpolating operator(s)

For a target hadron with quantum numbers $J^{PC}$ and flavor content, write the interpolating operator. Use Dirac bilinears for mesons, and appropriate diquark-quark structures for baryons.

For example, the simplest interpolating operator for a $\pi^+$ meson is usually written as $\mathcal{O}_{\pi^+} = \bar{d}^a \gamma_5 u^a$, and the simplest operator for a proton is $\mathcal{O}_p = \epsilon^{abc} (u^a C\gamma_5 d^b) u^c$. The simplest local operators are usually sufficient for ground state mass extraction, but interpolating operators can be constructed with gamma matrices and gauge covariant derivatives to access different quantum numbers, excited states, and observables related to hadron structure in general. For example, if we want to compute pion quasi-distribution amplitudes, we should use non-local operators with quark fields separated by a Wilson line, e.g. $\mathcal{O}_{\pi^+}(x;z) = \bar{d}^a(0) \gamma_5 W(0,z) u^a(z)$. If we want to compute a matrix element of an axial vector current inserted on the $u$ quark, we should insert the current operator $J_\mu=\bar{u}\gamma_5\gamma_\mu u$ between the source and sink operators in the three-point function.

**Gamma matrices convention**: Use the DeGrand-Rossi basis as the Euclidean Dirac basis, which is the default gamma basis in PyQUDA convention, can be defined in terms of the Pauli matrices $\sigma_i$ as:


$$
\gamma_1 = \begin{pmatrix} 0 & i \sigma_1 \nonumber \\ -i \sigma_1 & 0 \end{pmatrix},\quad
\gamma_2 = \begin{pmatrix} 0 & -i \sigma_2 \\ i \sigma_2 & 0 \end{pmatrix},\quad
\gamma_3 = \begin{pmatrix} 0 & i \sigma_3 \\ -i \sigma_3 & 0 \end{pmatrix},\quad
\gamma_4 = \begin{pmatrix} 0 & I_{2 \times 2}\\ I_{2 \times 2} & 0 \end{pmatrix}, \\
\gamma_5=\gamma_1\gamma_2\gamma_3\gamma_4=\begin{pmatrix} I_{2 \times 2} & 0 \\ 0 & -I_{2 \times 2} \end{pmatrix},\quad\gamma_0=I_{4 \times 4}=\begin{pmatrix} I_{2 \times 2} & 0 \\ 0 & I_{2 \times 2} \end{pmatrix},\quad C = \gamma_2\gamma_4 = \begin{pmatrix} -i\sigma_2 & 0\\ 0 & i\sigma_2 \end{pmatrix}$$


Clearly we have $\gamma_\mu^\dagger = \gamma_\mu$ for $\mu = 1,2,3,4,5$ in the DeGrand-Rossi basis. We have gamma anticommutation relations $\{\gamma_\mu, \gamma_\nu\} = 2\delta_{\mu\nu}$. Only $\gamma_1$ and $\gamma_3$ will generate a negative sign after the transpose. The auxiliary $\gamma_0$ is defined as the identity matrix, and the charge conjugation matrix $C=\gamma_2\gamma_4$.

### Step 2: Write the correlator

The basic observable in lattice QCD is the correlator of interpolating operators. For a mass extraction, this is a 2pt. For a matrix element or form factor, this is a 3pt with an appropriate current insertion.

**Two-point function** for a hadron is typically

$$C_2(\vec{p};t_f,t_i) = \langle \mathcal{O}(\vec{p},t_f) \mathcal{O}^\dagger(\vec{p},t_i) \rangle$$

where $\vec{p}$ is the momentum, and the interpolating operators are projected onto this momentum by Fourier transformation $\mathcal{O}(\vec{p},t) = \sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \mathcal{O}(\vec{x},t)$. By utilizing the time translation invariance of the vacuum, we can shift the time and average over equivalent time slices to enhance the signal. The operator for initial and final states can be the same or different depending on the observable (e.g. for a form factor you might have different operators at source and sink).

**Three-point function** for a matrix element or form factor:

$$C_3(\vec{q}; t_f,t_i,\tau) = \langle \mathcal{O}_\text{snk}(\vec{p}_f,t_f) J(\vec{q},\tau) \mathcal{O}^\dagger_\text{src}(\vec{p}_i,t_i) \rangle$$

with an appropriate current insertion $J$. Here, $t_f$ is the sink time, $\tau$ is the current insertion time, and $t_i$ is the source time. (We can shift the time slices to average over equivalent time slices to enhance the signal.) The transfer momentum is calculated by $\vec{q} = \vec{p}_i - \vec{p}_f$.

**For the specific $\Lambda \rightarrow p$ case**, the three-point function with sequential source is:

$$C_3(\vec{p}_f, \vec{p}_i; t_f, t_{seq}, 0) T = \sum_{\vec{x}, \vec{y}} e^{-i\vec{p}_f \cdot \vec{x}} e^{i\vec{q} \cdot \vec{z}} \big\langle \mathcal{O}_{p, D}(\vec{x}, t_f) \; J(\vec{y}, \tau) \; \bar{\mathcal{O}}_{\Lambda, I}(\vec{0}, t_i) \big\rangle T_{ID}$$
- `reference/Lambda_proton_formfactor.md` — theory equations and notation
- Keep notation consistent with the equations in `reference/Lambda_proton_formfactor.md`.
- The output must be a single flat script with all 3pt steps implemented.

### Step 3: Wick contraction and propagator determination

Expand the correlator by contracting all quark-antiquark pairs into
propagators $\text{prop}_f(x, y)$. Apply:

- **$\gamma_{5}$-hermiticity**: $\text{prop}_f(x, y) = \gamma_5 S_f^\dagger(y, x) \gamma_5$
  $\rightarrow$  converts backward propagators into forward ones (saves inversions)
- **Flavor symmetry**: for degenerate u/d quarks, $\text{prop}_u = S_d = S_l$
  $\rightarrow$  reduces number of distinct propagators needed
- **Charge conjugation / isospin**: may relate different diagram topologies

### Step 4: Determine propagator

A typical 2-point correlator $C_\pi(\vec{p}; t,0) = \sum_{\vec{x},\vec{y}} e^{-i \vec{p} \cdot (\vec{x} - \vec{y})} \text{Tr}[ S_l^\dagger(\vec{x},t; \vec{y},0) S_l(\vec{x},t; \vec{y},0) ]$ requires summing over both source $\vec{y}$ and sink $\vec{x}$, which is impossible in the real computation. Instead, we usually use point source propagator or wall source propator to estimate the correlator. Sometimes we can also use volume source propagator with stochastic estimation, but this is less common for two-point functions. Gaussian smearing can be applied to all types of sources to enhance ground state overlap, and APE/HYP smearing can be applied to the gauge links used in the source construction to further improve the signal. The choice of source type and smearing parameters depends on the specific observable and the desired balance between computational cost and statistical precision.

**Point source**: We can use a point source at a fixed spatial location and time slice (e.g., $\vec{y} = \vec{y}_0$ at $t=t_0$), set the phase $e^{i\vec{p}\cdot\vec{y}_0}$ at this point, and compute the propagator from this source point to all spatial points at all time slices, namely $\text{prop}_{l,\text{point}(\vec{p},\vec{y}_0,t_0)}(\vec{x},t)\equiv e^{i\vec{p}\cdot\vec{y}_0}S_l(\vec{x},t;\vec{y}_0,t_0)$. This gives us an estimate of the correlator: $C_\pi(\vec{p}; t,0) \approx \sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \text{Tr}[ S_{l,\text{point}(\vec{-p}_2,\vec{y}_0,t_0)}^{\dagger}(\vec{x},t) S_{l,\text{point}(\vec{p}_1,\vec{y}_0,t_0)}(\vec{x},t) ]$, where $\vec{p}=\vec{p}_1+\vec{p}_2$. The extra negative sign on $\vec{p}_2$ comes from the conjugate transpose of the propagator. This is how to use point source propagators to calculate the correlator. Note the momentum phase here is only a complex factor, we can just ignore it and set it to 1 without affecting any physical results. Then the momentum index $\vec{0}$ can be eliminated and the correlator estimated can be written as $C_\pi(\vec{p}; t,0) \approx \sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \text{Tr}[ S_{l,\text{point}(\vec{y}_0,t_0)}^{\dagger}(\vec{x},t) S_{l,\text{point}(\vec{y}_0,t_0)}(\vec{x},t) ]$. For better statistics, we can also use multiple point sources at different spatial locations and time slices, and average the resulting correlators.

**Wall source**: We can also use a wall source that spans the entire spatial volume at a fixed time slice (e.g., $t=t_0$), set the phase $e^{i\vec{p}\cdot\vec{x}}$ for each spatial point, and compute the propagator from this source to all spatial points at all time slices, namely $\text{prop}_{l,\text{wall}(\vec{p},t_0)}(\vec{x},t)\equiv\sum_{y}e^{i\vec{p}\cdot\vec{y}}S_l(\vec{x},t;\vec{y},t_0)$. This gives us an estimate of the correlator: $C_\pi(\vec{p}; t,0) \approx \sum_{\vec{x}} e^{-i \vec{p} \cdot \vec{x}} \text{Tr}[ S_{l,\text{wall}(-\vec{p}_2,t_0)}^{\dagger}(\vec{x},t) S_{l,\text{wall}(\vec{p}_1,t_0)}(\vec{x},t) ]$. Here we apply the similar momentum splitting strategy just like the point source case. But we cannot ignore the phase in the wall source, so we really need to calculate both $\text{prop}_{l,\text{wall}(-\vec{p}_2,t_0)}$ and $\text{prop}_{l,\text{wall}(\vec{p}_1,t_0)}$. Assuming that we only have a momentum in the $z$ direction, and we have $\vec{p}_i=(0,0,p_z)$, and the best strategy to split the momentum is basically equally splitting, i.e.
1. If $p_z=1$, choose $\vec{p}_1=(0,0,1)$, $\vec{p}_2=(0,0,0)$;
2. If $p_z=2$, choose $\vec{p}_1=(0,0,1)$, $\vec{p}_2=(0,0,1)$;
3. If $p_z=3$, choose $\vec{p}_1=(0,0,2)$, $\vec{p}_2=(0,0,1)$;
4. If $p_z=4$, choose $\vec{p}_1=(0,0,2)$, $\vec{p}_2=(0,0,2)$;

and so on. Finally, for better statistics, we can also use multiple wall sources at different time slices, and average the resulting correlators.

**Volume source**: Generally we do not use volume sources for two-point functions, but they can be used for all-to-all propagator with stochastic estimation. A volume source is defined as $\text{prop}_{l,\text{volume}(\vec{p},E)}(\vec{x},t)\equiv\sum_{\vec{y},\tau}e^{i(\vec{p}\cdot\vec{y}+E\tau)}S(\vec{x},t;\vec{y},\tau)$, which have the 4-momentum phase at each spatial point and time slice.

**Shifted propagator**: When using a non-local operator like $\mathcal{O}_{\pi^+}(x;z) = \bar{d}^a(x) \gamma_5 W(x,z) u^a(z)$, we will see the final propagator need to be shifted by applying the corresponding Wilson line like $\text{prop}_{u,W(\vec{z},t)}(\vec{x},t;\vec{y},0)\equiv W(\vec{z},t;\vec{x},t)S_u(\vec{z},t;\vec{y},0)$. This is because the quark field in the operator is located at $\vec{z}$ instead of $\vec{x}$, and the Wilson line connects the two points to make the operator gauge invariant. The same applies to current and baryon operators with non-local structures.

**Sequential propagator (for three-point functions)**: Three-point functions need a current insertion at an intermediate time, so a sequential propagator is required. Given a forward propagator $\text{prop}_{\text{fwd}}(x;0)$ from the source, one constructs a sequential source at the sink time and solves an additional inversion to obtain $\text{prop}_{\text{seq}}$. The three-point correlator is then obtained by contracting $\text{prop}_{\text{seq}}$ with the current and the forward strange propagator:

$$C_3(p_f,q; t_f,\tau,0) = \sum_z e^{iq\cdot z}\,\text{Tr}\,[\,S_{\text{seq}}(z,0)\,J\,S_s(z,0)\,]$$

Each sink momentum and gamma structure needs one additional inversion, which is the main cost for three point correlator calculations.

Output a list of propagators specifying:
- Quark flavor / mass parameter
- Source type (point, smeared-point with Gaussian/Wuppertal parameters)
- Source position(s) (time slice, number of sources per configuration)
- Whether APE/HYP smeared links are used for the source construction
- Sink treatment (point, smeared, or both $\rightarrow$  for SS/SP correlator matrix)



## Worked examples

See files in `reference/` directory for step-by-step demonstrations of the Wick contraction and propagator determination workflow. The filename indicates the target observable, e.g. `pion_mass.md` for the pion mass extraction example, `rho_mass.md` for the rho meson mass extraction example, and `proton_mass.md` for the proton mass extraction example.  `Lambda_proton_formfactor.md` for the Lambda to proton three point correlator example. Each example follows the same workflow outlined above, with detailed explanations of each step and the resulting expressions for the correlator, propagators needed, and einsum structure. The examples cover a range of observables and hadron types to illustrate the generality of the workflow.

## Interface to code generation

The einsum strings derived in the reference examples are for illustration.
Production code should obtain them from the code generation toolchain in
`tools.correlator_einsum`, which maps operator definitions
to fully specified einsum strings with correct spin/color labels and sign.

Input–output contract:

| Input | Format | Example |
|-------|--------|--------|
| Source operator | Hadron name + flavor dict + diquark gamma | `BaryonOp('proton', {a:u, b:d, c:u}, 'Cg5')` |
| Sink operator | Same format as source | `BaryonOp('proton', ...)` |
| Current | Flavor pair + gamma structure | `Current('s', 'u', 'gamma1')` |

| Output | Description |
|--------|-------------|
| Einsum string | Ready to hardcode into `opt_einsum.contract()` |
| Sign per topology | Physical sign convention for the sink block |
| Propagator type | Simple `Tr[S\u2020S]`, meson, baryon 2pt, or sequential 3pt |


*Note to Planner: after deriving the operator structure and propagator
requirements in this skill, route to `pyquda-tool` which will invoke the
generate_einsum tool. Do NOT attempt to produce einsum strings directly from the
physics derivation.*

*Note: For pure-gauge observables (Wilson loops, Polyakov loops, static
potential), see the **pyquda-gauge** skill instead. These involve no quark
propagators, no Wick contractions, and no fermion inversions.*


## Decision rules for source strategy

| Situation | Recommendation |
|----------------------------------|------------------------------------|
| Quick first look / debugging | Point source, 1 per config |
| Production meson spectroscopy | Smeared source, 4 sources/config |
| Baryon spectroscopy | Smeared source essential |
| Disconnected diagrams needed | Stochastic volume sources (Z_2/Z_4) |
| Form factor / 3pt function | Sequential source or stochastic |

**Multiple source times**: Computing propagators from multiple source time slices per configuration (e.g., `t_src = 0, T/4, T/2, 3T/4`) multiplies the effective statistics and improves the signal-to-noise ratio, especially for baryons and excited states. Each source time yields an independent correlator measurement after shifting to `t_src = 0`. The "4 sources/config" in the table above refers to 4 different source time positions. When determining propagator requirements, **ask the user** to confirm the source type, smearing, positions, and number of source times, as the optimal choice depends on the target observable and available computational budget.

## Common pitfalls

1. **Forgetting disconnected diagrams**: Flavor-singlet mesons ($\eta$, $\eta$', $\sigma$)
   have disconnected quark-loop contributions. These are computationally
   expensive and require different techniques (stochastic estimation).
   For flavor non-singlet mesons ($\pi$+, K+, $\rho$+), there are no disconnected
   diagrams.

2. **Wrong sign convention**: The overall sign of C(t) depends on the
   operator normalization and the number of fermion anticommutations in the
   Wick contraction. Always verify the sign is consistent with the expected
   large-t behavior of the channel. If you need the explicit fit template,
   hand off to `lqcd-physics-spectrum`.

3. **Periodic vs anti-periodic BC**: Fermions use anti-periodic temporal
   boundary conditions, but the **composite** state's BC depends on the
   number of quarks:
   Mesons:  (-1)^2 = +1 $\rightarrow$  C(t) ∝ e^{-mt} + e^{-m(T-t)}  (cosh-like)
   Baryons: (-1)^3 = -1 $\rightarrow$  backward state has opposite parity
   See `lqcd-physics-spectrum` for the explicit fit-function templates.
