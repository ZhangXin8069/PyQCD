## Physics Objective

Compute the rectangular Wilson loop $W(R=1,T=2)$ averaged over the XT, YT, and ZT planes using unsmeared gauge links. This is a pure-gauge measurement requiring no quark propagators or fermion inversions. The task uses the C24P29 ensemble ($24^3 \times 72$, $a \approx 0.1052$ fm) with a single configuration (cfg 10000).

## Measurement Strategy

1. **No smearing**: Original unsmeared gauge links are used directly, as explicitly requested.
2. **Multi-plane averaging**: $W(R=1,T=2)$ is computed independently on XT, YT, and ZT planes. The per-site real traces from the three planes are averaged point-by-point before the global spatial average, effectively tripling statistics.
3. **Single-configuration**: Only cfg 10000 is processed; the output is one scalar value.
4. **Plain-text output**: The result is written as a bare floating-point number with no header or metadata.

## Path Construction

For each plane ($\mu\nu$ = XT, YT, ZT), the path is constructed as:
- $R=1$ step forward along $\mu$
- $T=2$ steps forward along $\nu$
- $R=1$ step backward along $\mu$
- $T=2$ steps backward along $\nu$

In PyQUDA direction constants: `[μ] + [ν,ν] + [-μ] + [-ν,-ν]`.

## Normalization

$$W = \frac{1}{N_{\text{sites}} \cdot N_c} \sum_{\text{sites}} \text{Re}\,\text{Tr}\,U_{\text{path}}(x)$$

with $N_c = 3$ and $N_{\text{sites}} = 24 \times 24 \times 24 \times 72 = 995\,328$.

## Technical Implementation

- **Library**: PyQUDA (`gauge.loop()`, `getHost()`, `gatherLattice()`)
- **MPI**: 4 ranks along t-direction, process grid [1,1,1,4]
- **Loop API**: `gauge.loop()` requires exactly 4 groups; three active planes use weights [1,1,1,0]
- **MPI reduction**: `gatherLattice(field, [-1,-1,-1,-1])` sums across ranks; rank 0 performs final division and file I/O
- **Output file**: `wl_R1_T2_cfg10000.txt`