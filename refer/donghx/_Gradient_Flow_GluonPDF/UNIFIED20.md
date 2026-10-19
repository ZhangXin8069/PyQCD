# 20-conf common projected C2/C3 view

Run `python build_unified_20.py` to refresh
`unified_tf0p5_pilot_z/thin_tf0p50000000_eps0p01000000_tw4_Nv400_dz21/Nev_200/CONF/`.
The files there are symlinks to existing results, not rewritten arrays. The
machine-readable source map is `unified_tf0p5_pilot_z/manifest.json`.

The common range is `tsep=6..10`. The new ten configurations also have
`tsep=5`, but those files are excluded from this common view because most of
the original ten do not have it. Keep configuration identifiers as listed in
the manifest; the second group is not evenly spaced.

The original and spin-reduced contraction methods both save the same projected
format: C2 `(pol,mom,t_source,t_sink)` and C3
`(sign,pol,mom,z,current_m,current_o,t_source,t_current)`. The external baryon
`4 -> 2` parity reduction occurred inside the new contraction. Neither saved
format contains uncontracted external baryon spin indices. The C3 `4 x 4`
axes are open current-spin indices needed for channel projection, so they
must remain four-dimensional. The differing `flowed_peram_storage` metadata
(`nvme` versus `device`) describes runtime storage, not different operators.

This view preserves original bytes and provenance. New files passed the
production finite-value/shape validator. This index checks the representative
archive schema for every configuration and the presence and size of all 100
common files; it does not rescan the old correlator payloads numerically.
