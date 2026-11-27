#!/public/home/huangcl/.venv3_10/bin/python
# 功能: 逐条检查 NP 的 36 条收缩路径, 比较 pyqcd 和 opt_einsum 的收缩速度及数值结果.
# 输入: contract_str.py 中生成的 NP_indexes, 以及固定的 gamma 和 vvv 指标.
# 输出: 终端打印每条路径的耗时和比较结果, 最后输出全部路径的检查汇总.

from compare_precision import check_precision
from compare_contract import check_path
from contract_str import NP_indexes


tail = ",np,vx,wu,om,bdf,hjl,gik,ace->rqzy"

# opt_total = 0.0
# pyqcd_total = 0.0

# for path_id, index in enumerate(NP_indexes):
#     print(f"path id = {path_id}")
#     opt_median, pyqcd_median = check_path(index + tail)
#     opt_total += opt_median
#     pyqcd_total += pyqcd_median

# # check_path("nmba,rqfe,vuhg,pwdi,xyjk,zolc,np,vx,wu,om,bdf,hjl,gik,ace->rqzy")

# print(f"total opt_einsum median time: {opt_total:.2f} s")
# print(f"total pyqcd median time: {pyqcd_total:.2f} s")
# print(f"total speedup (opt_einsum / pyqcd): {opt_total / pyqcd_total:.2f}x")
# print("job done")


for path_id, index in enumerate(NP_indexes):
    print(f"path id = {path_id}")
    check_precision(index + tail)

print("job done")
