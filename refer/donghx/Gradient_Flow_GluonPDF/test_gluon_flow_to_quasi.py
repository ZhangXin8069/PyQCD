import math
import numpy as np
from gluon_flow_to_quasi import coefficients, match_one

def test_component_factors_and_reconstruction():
    a = np.zeros((1, 2, 4, 6, 3, 1), complex)
    a[0, 0, 0, 1, :, 0] = 2
    a[0, 0, 0, 2, :, 0] = 1
    out, c = match_one(a, 0.5, 0.0775, 2.0, 0.25)
    assert np.allclose(out[..., 0, :, :], out[..., 1, :, :] - out[..., 2, :, :])
    expected = (2.0 * c["component_factors_unpolarized"]) * np.exp(
        -c["delta_m_GeV"] * np.arange(3) * 0.0775 * 5.067730716
    )
    assert np.allclose(out[0, 0, 0, 1, :, 0], expected)
    assert np.allclose(out[0, 1, 0, 0, :, 0], 0.0)
    a[0, 1, 0, 1, :, 0] = 2
    a[0, 1, 0, 2, :, 0] = 1
    out, _ = match_one(a, 0.5, 0.0775, 2.0, 0.25)
    expected_h = (3.0 * c["component_factors_helicity"]) * np.exp(
        -c["delta_m_GeV"] * np.arange(3) * 0.0775 * 5.067730716
    )
    assert np.allclose(out[0, 1, 0, 0, :, 0], expected_h)

def test_coefficient_is_finite():
    vals = coefficients(0.5, 0.0775, 2.0, 0.25)
    assert all(math.isfinite(float(x)) for x in vals)
