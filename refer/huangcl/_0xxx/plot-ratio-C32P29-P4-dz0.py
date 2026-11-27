"""
Reproduce the C3/C2 ratio plot (C32P29, P=(0,0,4), dz=0) from
analyze-pion-3pt-C32P29-G5Gt-Ogmp-1smear.ipynb, without Arx7Tools.

Output: ratio-C32P29-P4-dz0.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# ---- ensemble / operator parameters (C32P29, hard-coded) ----
Nt = 64
conf_start, conf_end, conf_step = 11000, 60300, 50
conf_exception = {18350, 23050, 28400}
ens_dir = "beta6.20_mu-0.2770_ms-0.2400_L32x64"
momenta = "Psink0.0.4.Psrc0.0.4"

dir_3pt = f"/public/home/shenxm/data/pion-3pt/{ens_dir}/{momenta}"
dir_2pt = f"/public/home/shenxm/data/pion-2pt/{ens_dir}/{momenta}"
tag_3pt = f"Pi_3pt_aver_tsrc_G5Gt_Ogmp_dz18_1smear0.0.-3_aver_xyz_{momenta}"
tag_2pt = f"Pi_2pt_G5Gt_1smear0.0.-3_aver_xyz_{momenta}"


def symmetrize_tau(arr):
    """Average 3pt correlator under tau <-> tsep - tau (only tau <= tsep <= Nt/2)."""
    res = np.zeros_like(arr)
    nt = arr.shape[2]
    for tsep in range(nt // 2 + 1):
        for tau in range(tsep + 1):
            res[:, tsep, tau] = (arr[:, tsep, tau] + arr[:, tsep, tsep - tau]) / 2.0
    return res


def read_file(path):
    return np.load(path)


# ---- configurations with both 2pt and 3pt data available ----
confs = [c for c in range(conf_start, conf_end + 1, conf_step)
         if c not in conf_exception
         and os.path.exists(f"{dir_3pt}/{tag_3pt}.conf{c}.npy")
         and os.path.exists(f"{dir_2pt}/{tag_2pt}.conf{c}.npy")]
N = len(confs)
print(f"# cfg = {N}")


def corr_of_t(mat):
    """2pt correlator averaged over tsrc as a function of t_sep = (tsink - tsrc) mod Nt."""
    res = np.zeros(Nt, dtype=complex)
    for t_sink in range(Nt):
        for t_source in range(Nt):
            res[(t_sink - t_source) % Nt] += mat[t_sink, t_source] / Nt
    return np.real(res)


# ---- ensemble averages ----
corr_3pt = np.zeros((18, Nt, Nt), dtype=complex)
dcorr_of_t = np.zeros((N, Nt))
for i, conf in enumerate(confs):
    corr_3pt += symmetrize_tau(read_file(f"{dir_3pt}/{tag_3pt}.conf{conf}.npy"))
    dcorr_of_t[i] = corr_of_t(read_file(f"{dir_2pt}/{tag_2pt}.conf{conf}.npy"))
corr_3pt /= N
corr_of_t_ens = dcorr_of_t.mean(axis=0)

# ---- ratio C3/C2 and jackknife (bias-corrected value + sdev) ----
ratio = corr_3pt / corr_of_t_ens[np.newaxis, :, np.newaxis]
ratios = np.zeros((N,) + corr_3pt.shape, dtype=complex)
for i, conf in enumerate(confs):
    c3 = (corr_3pt * N - symmetrize_tau(read_file(f"{dir_3pt}/{tag_3pt}.conf{conf}.npy"))) / (N - 1)
    c2 = (corr_of_t_ens * N - dcorr_of_t[i]) / (N - 1)
    ratios[i] = c3 / c2[np.newaxis, :, np.newaxis]

sdev = np.sqrt(np.sum(np.abs(ratios - ratio) ** 2, axis=0) * (N - 1) / N)
bias = -(N - 1) * (ratios.mean(axis=0) - ratio)
ratio += bias

# ---- plot  ----
dz = 0
fig = plt.figure(figsize=(9, 6), dpi=200)
ax = fig.add_subplot(111)
ax.set(ylim=[0.09, 0.79])

dx = 0.0
for tsep in [6, 8, 10, 11, 12]:
    tau_min, shift, yfactor = 2, -0.05, -1.0
    if tau_min > tsep / 2:
        continue
    taus = np.arange(tau_min, tsep - tau_min + 1)
    x = taus - tsep / 2.0
    y = np.real(ratio[dz, tsep, taus]) * yfactor
    yerr = sdev[dz, tsep, taus]
    ax.errorbar(x + dx, y, yerr=yerr, capsize=5, fmt='o',
                label=r"$t_{s}$" + f"={tsep}", markersize=6)
    dx += shift

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ystep = (ymax - ymin) * 0.07
xtext = xmin + (xmax - xmin) * 0.03
ax.text(xtext, ymax - ystep * 1.0, f"C32P29," + r"$n_{cfg}$" + f"={N}, a=0.1053fm")
ax.text(xtext, ymax - ystep * 2.0, "P=(0, 0, 4)" + r", $\Delta z$=0")

# gamma / averaging labels (as notebook addtext)
for k, s in enumerate([r"$\Gamma=\gamma^5\gamma^t$", r"aver. over xyz, 1 peram. smeared"]):
    ypos = 0.7 + 0.07 * k
    ax.text(0, ymin * ypos + ymax * (1 - ypos), s)

ax.legend()
ax.tick_params(axis='y', which='both', left=True, right=True,
               labelright=False, direction='in')
ax.tick_params(axis='x', which='both', top=True, direction='in')
ax.grid(axis='y')
ax.set_xlabel(r"$\tau-t_s/2$", fontsize=13)
ax.set_ylabel(r"$\frac{{\rm Re}[C_{\rm 3 pt}(t_s, \tau, \Delta z)]}{C_{\rm 2pt}(t_s)}$",
              fontsize=13)

fig.savefig("ratio-C32P29-P4-dz0.png", bbox_inches="tight")
print("saved ratio-C32P29-P4-dz0.png")
