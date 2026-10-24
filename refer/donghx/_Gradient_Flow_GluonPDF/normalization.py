"""Paired, same-P, same-current local normalization. No fitting or Zchi model."""
import numpy as np


def local_normalize(matrix, replicas):
    """Input axes (component,p,z), (component,boot,p,z); even-real first.

    Divide both components by the *same* even-real local amplitude. Keep every
    bootstrap replica, including sign changes; zero/nonfinite divisors yield
    NaN, not clipping, abs(), replacement, or a reduced-replica error.
    """
    matrix = np.asarray(matrix, dtype=float)
    replicas = np.asarray(replicas, dtype=float)
    if (matrix.ndim != 3 or matrix.shape[0] != 2 or matrix.shape[-1] < 1
            or replicas.ndim != 4 or replicas.shape[0] != 2
            or replicas.shape[2:] != matrix.shape[1:] or replicas.shape[1] < 2):
        raise ValueError('expected (2,p,z) and (2,Nboot>=2,p,z)')
    denom, denom_boot = matrix[0, :, 0], replicas[0, :, :, 0]
    valid_d = np.isfinite(denom) & (denom != 0)
    valid_db = np.isfinite(denom_boot) & (denom_boot != 0)
    central = np.full_like(matrix, np.nan)
    boot = np.full_like(replicas, np.nan)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        np.divide(matrix, denom[None, :, None], out=central,
                  where=np.isfinite(matrix) & valid_d[None, :, None])
        np.divide(replicas, denom_boot[None, :, :, None], out=boot,
                  where=np.isfinite(replicas) & valid_db[None, :, :, None])
    central[~np.isfinite(central)] = np.nan
    boot[~np.isfinite(boot)] = np.nan
    available = np.isfinite(central) & np.isfinite(boot).all(axis=1)
    # Never nanstd: an absent replica cannot silently become a smaller sample.
    with np.errstate(invalid='ignore', over='ignore'):
        statistical_error = np.std(boot, axis=1, ddof=1)
    statistical_error[~available] = np.nan
    return dict(H_local=central, H_local_boot=boot, statistical_error=statistical_error,
                selected_available=available, denominator=denom.copy(),
                denominator_boot=denom_boot.copy(), denominator_valid=valid_d,
                denominator_valid_boot=valid_db)
