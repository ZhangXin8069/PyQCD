This revised plan targets the **connected eta_s proxy** correlator, not the full flavor-singlet \(\bar s \gamma_5 s\) channel with disconnected loops. The physics object is stated explicitly as the zero-momentum connected pseudoscalar strange correlator
\[
C(t)=\sum_{\vec x}\operatorname{Tr}\big[S_s^\dagger(\vec x,t;0)\,S_s(\vec x,t;0)\big],
\]
which is the standard connected contraction obtained from the interpolating operator \(\bar s\gamma_5 s\) using \(\gamma_5\)-hermiticity.

The plan keeps the requested setup unchanged where it was well defined:
- point source at **[0,0,0,0]**,
- stout smearing with **n_steps=1, rho=0.125, ndim=4**,
- plain-text output in the run directory.

It also closes the main ambiguities identified in review:
- the channel is now labeled correctly as a **flavor-diagonal connected strange pseudoscalar (eta_s proxy)**;
- disconnected strange-loop contributions are explicitly excluded;
- the measurement is specified as a **single-source per-configuration correlator**, with only the spatial sink sum used for zero-momentum projection;
- stout smearing is specified operationally as creating a **smeared copy of the gauge field** and using that copy in the strange-quark Dirac inversion, avoiding double-smearing ambiguity;
- the output contract is closed: one file per configuration, named `eta_s_connected_2pt_cfg_<cfg>.txt`, containing exactly **72 rows**, **real part only**, and **no header, time index, or comments**.

The ensemble section was kept consistent with the supplied C24P29 metadata, including the strange mass and clover coefficient already tied to that ensemble. This makes the plan executable for that specific ensemble, while also making clear in the solver note that those action parameters are ensemble-dependent.

Scientifically, this is a valid **per-configuration correlator production** plan. It is not, by itself, a statistically credible eta_s spectroscopy campaign, since one source on one configuration is only a minimal measurement setup.