from lqcddb import run_wick_analysis

sink_ope = [[2, "|", "s", "u", "gamma_7", "d", "|"], [1, "|", "d", "u", "gamma_7", "s", "|"], [-1, "|", "u", "d", "gamma_7", "s", "|"]]

source_ope = [["|", "u^d", "gamma_7", "d^d", "u^d", "|", "|", "s^d", "gamma_5", "u", "|"], ["|", "u^d", "gamma_7", "d^d", "d^d", "|", "|", "s^d", "gamma_5", "d", "|"]]


run_wick_analysis(
    operator_groups = [(i, j)for i in sink_ope for j in source_ope],
    Cpt = "2pt",
    use_equivalence = True,
    ignore_dis = False,
    plot = "./"
)

# sink_ope = [[2, "|", "s", "u", "gamma_7", "d", "|"], [1, "|", "d", "u", "gamma_7", "s", "|"], [-1, "|", "u", "d", "gamma_7", "s", "|"]]

# source_ope = [[2, "|", "u^d", "gamma_7", "d^d", "s^d", "|"], [1, "|", "u^d", "gamma_7", "s^d", "d^d", "|"], [-1, "|", "d^d", "gamma_7", "s^d", "u^d", "|"]]


# run_wick_analysis(
#     operator_groups = [(i, j)for i in sink_ope for j in source_ope],
#     Cpt = "2pt",
#     use_equivalence = True
# )