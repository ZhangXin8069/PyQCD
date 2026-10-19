# Pre-generation Checklist

1. Confirm that the script preserves `M0`–`M8` and the selected asset's top-level order.
2. Derive peram, V, and gamma keys from the dynamic manifest and compare each with the actual registrations.
3. Confirm that global-time ownership comes from `get_mpi_tlist`, including the selected `Scatter/TScatter` path and wraparound time.
4. Check Registry dimensions, axes, conjugation, dtype, and backend against the einsum fields.
5. Check operator flavor conservation, coefficients, gamma order, temporal-wrap sign, and GEVP positive definiteness when applicable.
6. Confirm stage timers and total-runtime output on the root process.
7. Record GPU and full-ensemble paths that were not run; do not add test files to the generated program.
8. Confirm that the same-named `.sh` is delivered and compare it line by line using the submission-script whitelist.
