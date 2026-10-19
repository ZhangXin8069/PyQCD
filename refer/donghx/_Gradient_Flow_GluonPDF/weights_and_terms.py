"""Two explicit current-vertex weights; frozen proton Wick specs are reused.

No basis, perambulator, source/sink VVV or production file is reweighted.
The diagnostic retains each ordered (sigma_a,sigma_b) block separately.
"""
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'reference'))
from spin2_contractions import (SPECS, parity_basis, reduced_projectors,
                                external_perambulators)

COUNTS = (100,) + (10,) * 10 + (200,)
DIMENSIONS = (100, 50, 50, 100, 100, 200, 200, 350, 350, 550, 550, 95704)
SECTOR_NAMES = ('exact100', 'compressed50a', 'compressed50b',
                'compressed100a', 'compressed100b', 'compressed200a',
                'compressed200b', 'compressed350a', 'compressed350b',
                'compressed550a', 'compressed550b', 'high95704')


def weight_tables(counts=COUNTS, dimensions=DIMENSIONS):
    """Return first-order vector, same-frame second order, independent product."""
    if len(counts) != len(dimensions) or any(n < 2 or n > d for n, d in zip(counts, dimensions)):
        raise ValueError('sector lengths/counts must be consistent, 2 <= n <= d')
    labels = np.repeat(np.arange(len(counts)), counts)
    first = np.repeat(np.asarray(dimensions, float) / counts, counts)
    product = first[:, None] * first[None, :]
    second = product.copy()
    start = 0
    for n, d in zip(counts, dimensions):
        block = second[start:start+n, start:start+n]
        block[:] = d * (d-1) / (n * (n-1))
        np.fill_diagonal(block, d/n)
        start += n
    return first, second, product, labels


def table_for_times(sigma_a, sigma_b, second, product):
    return second if sigma_a == sigma_b else product


def iter_reduced_right(p, b, g, q, vs, vr, contract, xp):
    """One right factor at a time; b has a singleton original-time axis."""
    basis = parity_basis(xp)
    pl, pr, pb = external_perambulators(p, basis, contract)
    br = contract('uokpl,ks->uospl', b, basis)
    q2 = reduced_projectors(q, basis, contract)
    arrays = dict(p=p, b=b, pl=pl, pr=pr, pb=pb, br=br, g=g, q=q2, vs=vs, vr=vr)
    for spec in SPECS:
        name, flavor, coefficient, equation, keys, final, hkey = spec
        right = contract(equation, *(arrays[key] for key in keys))
        yield spec, right, q2, basis
        del right


def signed_block_term(h, spec, right, q2, basis, contract):
    """h: (sigma_a,u=1,S,L,p,a,m,b); output (sigma_a,S,P,M,L,m,o).

    X is an explicit batch index, NOT a summation. Each term includes its
    original fermionic sign. Current spin m/o stays full four dimensional.
    """
    _, _, coefficient, _, _, final, hkey = spec
    if hkey == 'hl':
        hh = contract('XuSLpamb,ar->XuSLprmb', h, basis.conj())
    else:
        hh = h
    inputs, output = final.split('->')
    equation = 'X' + inputs + '->X' + output
    args = (hh, right, q2) if 'Psr' in final else (hh, right)
    value = coefficient * contract(equation, *args)
    # X,P,S,M,L,m,o -> X,S,P,M,L,m,o
    return value.transpose(0, 2, 1, 3, 4, 5, 6)


def current_block_h(vertices, sequential, contract):
    """Keep sigma_a as a free batch axis, insert singleton sigma_b axis."""
    return contract('XSLnp,Xambn->XSLpamb', vertices, sequential)[:, None]


def summed_flavors(signed_terms):
    """signed_terms: (scheme,sigma_a,sigma_b,wick,S,P,M,L,m,o)."""
    block_terms = signed_terms.sum(axis=(1, 2))
    return np.stack((block_terms[:, :4].sum(axis=1),
                     block_terms[:, 4:].sum(axis=1)), axis=1)
