This revised plan keeps the original objective and execution style, but makes the physics target and measurement definition explicit.

It computes the **connected** zero-momentum J/psi two-point correlator on ensemble **C24P29** from a **single charm point source at [0,0,0,0]**, using the requested **stout-smeared links** with the unambiguous mapping
\((n_{\mathrm{steps}}, \rho, n_{\mathrm{dim}}) = (1, 0.125, 4)\).
The stout-smeared links are specified to be used consistently throughout the clover Dirac operator, not only in the hopping term.

The vector structure is now encoded operationally rather than only by name:
for each spatial polarization \(i \in \{x,y,z\}\), the plan measures
\[
C_i(t)=\sum_{\vec{x}} \mathrm{Tr}\left[S_c^\dagger(\vec{x},t;0)\,(\gamma_5\gamma_i)\,S_c(\vec{x},t;0)\,(\gamma_i\gamma_5)\right],
\]
and then forms the requested average
\[
C(t)=\frac{1}{3}\left(C_x(t)+C_y(t)+C_z(t)\right).
\]
This guarantees that the requested \(\gamma_x,\gamma_y,\gamma_z\) local vector channels are the actual measured objects, with the proper zero-momentum sink sum.

The plan also avoids overclaiming: since no disconnected charm-annihilation contribution is included, the observable is labeled as the **connected J/psi correlator**, which is the standard practical choice for such a setup but not the full flavor-diagonal charmonium correlator.

Finally, the output is made operationally precise: the final file is a plain **txt** file in the run directory, **no header**, **one row per timeslice**, with columns
`time  Re C(t)  Im C(t)`.
Minimal validation is included: inspect the three polarization channels separately before averaging, verify the imaginary part is numerically small, confirm the spatial sum is applied, and record the achieved solver residual.