"""Memory-bounded exact sigma_a sum for one sigma_b; NumPy/CuPy agnostic."""


def one_right_time_h(vertices, sequential, ib, contract):
    # Never gather/reshape the complete J² or retain H for all sigma_b.
    return contract('tSLnp,tambn->SLpamb', vertices[:, ib], sequential)[None]
