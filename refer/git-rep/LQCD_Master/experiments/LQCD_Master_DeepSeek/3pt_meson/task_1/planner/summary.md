# D⁰ → K⁻ Three-Point Function — Revised Plan

## Physics objective (unchanged)
Compute the three-point correlation function for D⁰ → K⁻ via the (c→s) vector current \(\bar{s}\gamma_x c\).  Raw input for the \(D\to K\) vector form factor at \(q^2=0\).

## Corrections applied (from peer review)

### 1. Sequential-source formula  —  **fixed**

**Old (wrong):**  \(\eta_s^{\text{seq}}(x) = \gamma_5 S_u(x,8;0)\)

**New (correct):**  \(\eta_s^{\text{seq}}(x) = \gamma_5 S_u^{\dagger}(x,8;0)\)

This follows the two-dagger convention from the `Lambda_proton_formfactor` reference.  The hermitian conjugate on \(S_u\) is essential; omitting it produces a correlator that does not correspond to the physical Wick contraction.

### 2. Sequential-source spatial extent  —  **fixed**

The original plan specified `source_position: [0,0,0,8]`, which a code generator could interpret as a single-point source.  The sequential source must span **all spatial points** on time slice \(t=8\) (full-timeslice / wall-like) to capture the complete sink-meson wave function.  The revised plan marks this as `source_position: [all, all, all, 8]` and explicitly states “full timeslice at t=8” in the formula field.

### 3. Overall sign convention  —  **made explicit**

The connected Wick contraction contains one fermion loop, contributing an overall minus sign.  The revised plan states this explicitly in both the task description and the measurement block (`sign_convention` field), so the code-generation tool (`generate_einsum`) can be checked for consistency.

## Propagator summary (3 required)

| ID | Flavour | Type | Source |
|----|---------|------|--------|
| `prop_l` | light (u) | Forward, point | \([0,0,0,0]\) |
| `prop_c` | charm (c) | Forward, point | \([0,0,0,0]\) |
| `prop_s_seq` | strange (s) | Sequential | Full timeslice at \(t=8\), built from \(\gamma_5 S_u^{\dagger}\) |

## Contraction (schematic)

1. Solve forward propagators \(S_u(x;0)\) and \(S_c(x;0)\) from the point source.
2. At \(t_f=8\), construct \(\eta^{\text{seq}}(x) = \gamma_5 S_u^{\dagger}(x,8;0)\) at every spatial point \(x\).
3. Solve \(D_s G_s^{\text{seq}} = \eta^{\text{seq}}\).
4. Contract:  \(C_3(\tau) = -\sum_{\vec z} \operatorname{Tr}[G_s^{\text{seq}}(\vec z,\tau)\,\gamma_x\,S_c(\vec z,\tau;0)]\).

The minus sign is from the single fermion loop.

## Risks / caveats (non-blocking)

- **Single point source** on one configuration → signal-to-noise may be poor for a charm-light three-point function.
- **No two-point functions** are computed; normalisation into a form factor requires external 2pt data.
- **Stout smearing** (\(\rho=0.125\), 1 step) is a light-quark choice; the charm propagator may be over-smeared, potentially suppressing the D⁰ signal.  Kept as specified by the user.

## Output (unchanged)
Plain-text `.txt` file in the run directory, no headers or metadata, containing \(C_3(\tau)\) for all current-insertion times \(\tau\in[0,8]\).