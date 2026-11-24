#!/public/home/huangcl/.venv3_10/bin/python
"""导出指定 tsep 和 tau 的 3ptdisc 与 selfratio 样本数据."""

from pathlib import Path

import numpy as np

DATA_DIR = Path(
    "/public/home/huangcl/07_pion_gluon_pdf/test01_check_selfratio/"
    "01_data/c32p29/P4")
INPUT_PATH = DATA_DIR / "3ptdisc.npy"
EXCLUDED_PATH = DATA_DIR / "excluded_sample_ids.npy"
OUTPUT_PATH = DATA_DIR / "ex_data.txt"
TAU_BY_TSEP = {
    10: [0, 1, 2, 8, 9, 10],
    11: [0, 1, 2, 3, 8, 9, 10, 11],
}

# para: sample, dt, dtau, z
corr3_disc = np.load(INPUT_PATH, mmap_mode="r")
# para: dt, dtau, z
excluded_ids = np.load(EXCLUDED_PATH)
lines = ["tsep\ttau\tsample_id\t3ptdisc_z0_real\t3ptdisc_z1_real\tselfratio_real"]

for tsep, tau_list in TAU_BY_TSEP.items():
    for tau in tau_list:
        sample_id = excluded_ids[tsep, tau, 1]
        value_z0 = corr3_disc[sample_id, tsep, tau, 0].real
        value_z1 = corr3_disc[sample_id, tsep, tau, 1].real
        selfratio = np.real(
            corr3_disc[sample_id, tsep, tau, 1]
            / corr3_disc[sample_id, tsep, tau, 0])
        lines.append(
            f"{tsep}\t{tau}\t{sample_id}\t{value_z0:.3g}"
            f"\t{value_z1:.3g}\t{selfratio:.3g}")

OUTPUT_PATH.write_text("\n".join(lines) + "\n")
print(f"saved data: {OUTPUT_PATH}")
