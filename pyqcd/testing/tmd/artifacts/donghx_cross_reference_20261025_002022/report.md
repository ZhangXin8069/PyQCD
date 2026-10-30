# Donghx cross-reference: 4x4x4x8

- Lattice `(Lx,Ly,Lz,Nt)=(4,4,4,8)`, array shape `(8, 4, 4, 4)`
- tau values: `[0.0, 0.1, 0.5]`; epsilon: `0.01`
- Input SHA-256: `5d7c80e2c0063db215f804b860096dd215de959f09ecacd03b7f26849c140f9f`

## Flow and straight OPE

| tau | flow rel-L2 | OPE c64 rel-L2 | OPE c128 rel-L2 |
|---:|---:|---:|---:|
| 0 | 0.000e+00 | 6.784e-08 | 2.954e-08 |
| 0.1 | 2.447e-07 | 2.763e-07 | 1.424e-07 |
| 0.5 | 5.540e-07 | 8.942e-07 | 3.336e-06 |

## Shared post-processing

- flow-to-quasi max difference: `0.000e+00`
- native OPE-to-quasi max difference: `1.946e-06`
- gg matching-kernel max difference: `0.000e+00`
- Fourier same-hbar max difference: `1.388e-17` (source tau `0.100`)
- Fourier default-constant max difference: `1.869e-04`
- ratio central-C3 max difference: `0.000e+00`
- ratio denominator-convention max difference: `6.770e-03`

## TMD boundary

- PyQCD `b=0,L=0` collinear reduction, complex64 max difference: `5.728e-08`
- PyQCD `b=0,L=0` collinear reduction, complex128 max difference: `1.110e-16`
- Finite-`b` TMD is computed by PyQCD but has no donghx counterpart in the archived reference.

Overall: **pass**
