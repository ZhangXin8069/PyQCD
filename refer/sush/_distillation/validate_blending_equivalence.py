"""Compare full lqcddb Omega4, independent tiles, and the reference A1--A5.

Run in this directory: python -B validate_blending_equivalence.py
Only synthetic tensors are used; the production script is never executed.
"""

import ast
from pathlib import Path
import time

import numpy as np
from opt_einsum import contract

from lqcddb.eigvectors.blending import BlendingWeights

EQUATION = 'fnph,manb,gehf,codp,bd,Mfn,Mph,ac,MGem,MGog->MG'
REFERENCE = (Path(__file__).resolve().parent.parent / 'pion_form_factor' /
             'contraction.block.numpy.compensation.all.py')


def reference_parts_code():
    """Extract the real reference's arithmetic, through its A5 assignment.

    Substitute only the hardcoded list of 150 non-noise indices by a supplied
    list at the small test size.  Leave all weight calls, einsums, tensor slices
    and diagonal-correction coefficients unchanged.
    """
    tree = ast.parse(REFERENCE.read_text())
    section = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.If) and ast.unparse(node.test) == 'Nev_curr == Nev_max')
    body = []
    for statement in section.body:
        if isinstance(statement, ast.Assign):
            names = [target.id for target in statement.targets if isinstance(target, ast.Name)]
            if 'Nev_compensation_indx' in names:
                continue
        body.append(statement)
        if isinstance(statement, ast.Assign) and 'A5' in names:
            break
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    return compile(module, str(REFERENCE), 'exec')


def compare(seed, noise, *, rank_count=1):
    from lqcddb import set_backend, get_backend
    from reference_lqcddb import reference
    vertex_creator = reference.vertex_creator
    set_backend('numpy')
    backend = get_backend()
    parameters = dict(exact=4, N_eigen=[12], N_sum=[8], N_extract=[4], noise=noise)
    weights = BlendingWeights(Nx=3, **parameters)
    eigen = vertex_creator(Nx=3)
    n = weights.nvec
    n_a = n - noise
    nsrc, spin, nmom, ngamma = 4, 2, 2, 2
    rng = np.random.default_rng(seed)

    def random(shape):
        return rng.normal(size=shape) + 1j*rng.normal(size=shape)

    u = random((spin, spin, n, nsrc))
    sep = random((spin, spin, n, n))
    down = random((spin, spin, nsrc, n))
    source = random((nsrc, nsrc))
    vdv = random((nmom, n, n))
    sink = random((nmom, n, n))
    gamma5 = random((spin, spin))
    current_gamma = random((nmom, ngamma, 2, spin, spin))
    g0, g1 = current_gamma[:, :, 0], current_gamma[:, :, 1]

    # Crucially, form the FULL weight with the actual imported lqcddb method.
    omega4 = eigen.create_omega_accelerate(dim=4, **parameters)
    dense = contract(EQUATION, omega4, u, sep, down, source, vdv, sink,
                     gamma5, g0, g1, optimize='greedy')
    generated = weights.block(dim=4)

    blocked = np.zeros_like(dense)
    # This simulates summing disjoint work from multiple ranks, without MPI.
    for rank in range(rank_count):
        for (f, ni, p, h), weight in weights.iter_blocks(
                dim=4, block_size=(3, 4, 5, 6), rank=rank, size=rank_count):
            blocked += contract(EQUATION, weight, u[:, :, ni, :], sep[:, :, h, f],
                                down[:, :, :, p], source, vdv[:, f, ni], sink[:, p, h],
                                gamma5, g0, g1, optimize='greedy')

    def reference_contract(equation, *operands, **kwargs):
        # Contraction optimization changes the evaluation path, not the tensor
        # equation. The reference's optimizer list is specific to its cache.
        return contract(equation, *operands, optimize='greedy')

    remaining = 3**3*3 - parameters['exact'] - sum(parameters['N_eigen'])
    namespace = dict(
        backend=backend, eigen=eigen, exact_eigen=parameters['exact'],
        N_eigen=parameters['N_eigen'], N_sum=parameters['N_sum'],
        N_extract=parameters['N_extract'], N_part1=n_a, N_part2=noise,
        Nev_curr=n, Nev_max=n, Nev_src=nsrc, rank=0, size=1,
        Nev_compensation_indx=[[i] for i in range(n_a)],
        peram_u_compensation=u, peram_u_sep=sep, peram_d_2pt=down,
        source=source, VdV_curr=vdv, sink_3pt=sink,
        gamma=lambda _: gamma5, mu_nu_gamma=current_gamma,
        cached_contract=reference_contract,
        get_mpi_data=lambda data, **_: data, time=time, print=lambda *_, **__: None,
        _weight=[(remaining-r)/(noise-r) for r in range(4)],
    )
    exec(reference_parts_code(), namespace)
    partial = sum(namespace[f'A{i}'] for i in range(1, 6))

    # Check EACH A1--A5 against its disjoint sector of the full lqcddb weight.
    # A1 has 0 noise indices; ...; A5 has 4 noise indices.
    prefix = contract('manb,gehf,codp,bd,ac,MGem,MGog->MGnfph',
                      u, sep, down, source, gamma5, g0, g1, optimize='greedy')
    integrand = np.einsum('MGnfph,Mfn,Mph->MGfnph', prefix, vdv, sink)
    del prefix
    coordinates = np.ogrid[(slice(0, n),)*4]
    noise_count = np.zeros((n,)*4, dtype=np.uint8)
    for coordinate in coordinates:
        noise_count += coordinate >= n_a
    sector_errors = []
    for count in range(5):
        sector = np.sum(integrand * np.where(noise_count == count, omega4, 0),
                        axis=(2, 3, 4, 5))
        actual = namespace[f'A{count+1}']
        np.testing.assert_allclose(actual, sector, rtol=5e-12, atol=1e-6)
        sector_errors.append(float(np.max(np.abs(actual-sector))))

    np.testing.assert_allclose(generated, omega4, rtol=3e-15)
    np.testing.assert_allclose(blocked, dense, rtol=5e-12, atol=1e-6)
    np.testing.assert_allclose(partial, dense, rtol=5e-12, atol=1e-6)
    scale = np.max(np.abs(dense))
    return dict(seed=seed, noise=noise, nvec=n,
                weight_max_abs=float(np.max(np.abs(generated-omega4))),
                block_max_abs=float(np.max(np.abs(blocked-dense))),
                block_scaled_error=float(np.max(np.abs(blocked-dense))/scale),
                reference_max_abs=float(np.max(np.abs(partial-dense))),
                reference_scaled_error=float(np.max(np.abs(partial-dense))/scale),
                sector_max_abs=sector_errors)


def main():
    import lqcddb
    print('Imported lqcddb:', lqcddb.__file__)
    print('Manual decomposition:', REFERENCE)
    for seed, noise, ranks in [(8261, 4, 1), (137, 5, 3), (945, 4, 5)]:
        result = compare(seed, noise, rank_count=ranks)
        print(f"seed={seed} noise={noise} nvec={result['nvec']} simulated_ranks={ranks}")
        print(f"  weight max absolute error: {result['weight_max_abs']:.6e}")
        print(f"  tiled vs dense: abs={result['block_max_abs']:.6e} "
              f"scaled={result['block_scaled_error']:.6e}")
        print(f"  manual A1--A5 vs dense: abs={result['reference_max_abs']:.6e} "
              f"scaled={result['reference_scaled_error']:.6e}")
        print('  A1--A5 sector max absolute errors:',
              ', '.join(f'{error:.6e}' for error in result['sector_max_abs']))
    print('PASS: all fourth-order weights and contractions agree within floating-point tolerance.')


if __name__ == '__main__':
    main()
