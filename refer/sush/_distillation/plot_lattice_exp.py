"""
Plot y = e^{-ipx} on a discrete lattice to show why smaller p → smoother.

On the lattice, x = n (integer sites), and the complex exponential becomes:
    y(n) = e^{-ipn} = cos(pn) - i sin(pn)

Smoothness = how much y changes between adjacent sites:
    |y(n+1) - y(n)| = 2|sin(p/2)|

So p → 0 gives the flattest (most constant) behavior.
"""
import numpy as np
import matplotlib.pyplot as plt
# Lattice: 32 sites
N = 32
n = np.arange(N)

# Choose momenta to compare: p=0, small p, medium p, large p (Nyquist)
p_values = [0.0, 0.2, np.pi / 4, np.pi / 2, np.pi]

fig, axes = plt.subplots(
    nrows=len(p_values), ncols=2,
    figsize=(14, 10),
    sharex=True, sharey="row",
    gridspec_kw={"width_ratios": [1, 1], "hspace": 0.35, "wspace": 0.25},
)

for i, p in enumerate(p_values):
    y = np.exp(-1j * p * n)

    # ── Real part ──
    ax = axes[i, 0]
    ax.plot(n, y.real, "o-", ms=7, lw=1.8, color="#d55e00", markeredgewidth=0)
    ax.axhline(0, color="gray", lw=0.6, zorder=0)
    ax.set_ylabel(r"$\mathrm{Re}[e^{-ipn}]$" if i == 0 else "", fontsize=12)

    # ── Imag part ──
    ax = axes[i, 1]
    ax.plot(n, y.imag, "o-", ms=7, lw=1.8, color="#0072b2", markeredgewidth=0)
    ax.axhline(0, color="gray", lw=0.6, zorder=0)
    ax.set_ylabel(r"$\mathrm{Im}[e^{-ipn}]$" if i == 0 else "", fontsize=12)

    # Annotate p value
    label = r"$p=0$" if p == 0 else rf"$p={p:.2f}$" if p < 1 else r"$p=\pi/2$" if abs(p - np.pi / 2) < 0.01 else r"$p=\pi$ (Nyquist)"
    axes[i, 0].annotate(
        label, xy=(0.02, 0.9), xycoords="axes fraction",
        fontsize=13, fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85),
    )

    # Show |Δy| between adjacent sites as a smoothness indicator
    delta = np.abs(y[1:] - y[:-1])
    smoothness = np.mean(delta)
    axes[i, 1].annotate(
        rf"$\langle|\Delta y|\rangle = {smoothness:.3f}$",
        xy=(0.98, 0.1), xycoords="axes fraction",
        fontsize=10, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, ec="gray", lw=0.6),
    )

    # Tick every 4 sites
    axes[i, 0].set_xticks(n[::4])
    axes[i, 1].set_xticks(n[::4])

axes[-1, 0].set_xlabel(r"$n$ (lattice site)", fontsize=12)
axes[-1, 1].set_xlabel(r"$n$ (lattice site)", fontsize=12)

fig.suptitle(
    r"$y(n) = e^{-ipn}$ on a discrete lattice — smaller $p$ → smoother",
    fontsize=15, fontweight="bold", y=1.01,
)

plt.tight_layout()
out = "/public/home/sush/distillation/analyse/lattice_exp.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
plt.close()
