"""Pure tensor scheduling helpers; no changes to spin/weight/Wick identities."""


def c2_momentum_batched(p, vs, vr, g, q, contract, reduced, reference, xp):
    """Retain every momentum and full-spin oracle, with a singleton M axis."""
    result, gold = [], []
    for im in range(vs.shape[0]):
        result.append(reduced(p, vs[im:im+1], vr[im:im+1], g, q, contract))
        gold.append(reference(p, vs[im:im+1], vr[im:im+1], g, q, contract))
    return xp.concatenate(result, axis=1), xp.concatenate(gold, axis=1)


def selected_h_streamed(vertices, sequential, indices, contract, xp):
    """Sum sigma_a independently for each sigma_b, without copying full J².

    Input J [sigma_a,sigma_b,S,L,n,p], A [sigma_a,a,m,b,n].
    Only the predeclared finite window is selected; no further truncation.
    """
    shape = (len(indices),) + vertices.shape[2:4] + (
        vertices.shape[5], sequential.shape[1], sequential.shape[2], sequential.shape[3])
    result = xp.empty(shape, dtype=xp.result_type(vertices, sequential))
    aa = sequential[indices]
    for target, ib in enumerate(indices):
        jj = vertices[indices, ib][:, None]
        result[target:target+1] = contract('tuSLnp,tambn->uSLpamb', jj, aa)
        del jj
    return result
