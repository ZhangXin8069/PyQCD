Core physics goal: compute a standard hadron spectroscopy observable, namely the zero-momentum baryon two-point function for the Xi- octet baryon with flavor content dss, and project onto the positive-parity channel.

Strategy:
- This is a standard correlator task, so `task_mode` is `standard`.
- The Xi- interpolating operator is taken as
  \( \mathcal O_{\Xi^-} = \epsilon^{abc}(d^{Ta} C\gamma_5 s^b)s^c \),
  exactly matching the user’s requested `Cg5` ds diquark.
- The correlator is a baryon 2pt at zero momentum, with the parity projector
  \(Tmat=(I+\gamma_t)/2\), interpreted in Euclidean conventions as \(\gamma_t=\gamma_4\).
- Because this is a baryon two-point function, the numerically standard setup is a point source propagator from a fixed source point and a spatial sum at the sink for momentum projection.

Required propagators:
- One light propagator from source [0,0,0,0] with mass -0.277.
- One strange propagator from the same source with mass -0.2356.
- The strange propagator is reused for both strange quark legs in the Xi- contraction.

Gauge-link treatment and inversion details:
- All propagators are inverted on stout-smeared gauge links, using the user-specified stout parameters `(1, 0.125, 4)`, interpreted in PyQUDA-style order as `n_steps=1`, `rho=0.125`, `ndim=4`.
- I used the ensemble clover coefficient `1.160920226` directly.
- Since no solver stopping criteria were provided, I completed the plan conservatively with a production-quality residual tolerance `1e-12` and `maxiter=10000`.

Measurement details:
- The observable is classified as `baryon_2pt`.
- The plan keeps the correlator at zero momentum and positive parity only, which is exactly what the user asked for.
- The contraction should be generated through the baryon-2pt einsum/codegen path rather than manually hardcoding an index contraction, which is the safer and more extensible execution route.

Output details:
- The output is specified as plain text in the run directory with no header and no extra text.
- Accordingly, `output.format` is set to `other` and `metadata` is `false`.
- I explicitly recorded that the final file should contain only the correlator values, one per time slice.

Reasonable completions made:
- Interpreted `gamma_t` as Euclidean `gamma_4`.
- Interpreted stout parameters `(1, 0.125, 4)` as `(n_steps, rho, ndim)`.
- Added standard zero-momentum sink projection by spatial summation.
- Added conservative solver controls (`tolerance`, `maxiter`) because the user did not specify them.
- Added the mandatory `ensemble:` block exactly from the provided ensemble facts, without altering the supplied values.