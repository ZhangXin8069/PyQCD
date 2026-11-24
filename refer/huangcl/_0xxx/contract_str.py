# O = N1^T (Cgi) N2, i = 1, 2, 3, 5, 计算 OO^\dag, 处理 peram 的 3 个 u 和 3 个 d 夸克, 共36种收缩方式
# N1 = p, N2 = n
import itertools


#  处理同种夸克交换时产生的负号, 即 Levi-Civita 符号
def perm_sign(perm):
    # i, j, k = perm; inv = (i > j) + (i > k) + (j > k)
    inv = sum(perm[i] > perm[j] for i in range(3) for j in range(i+1, 3))
    return 1 if inv % 2 == 0 else -1


# s 是控制 spin 收缩的 key, e 是控制 eig_idx 收缩的 key
u_snk = {
    0: {"s": "n", "e": "b"},
    1: {"s": "r", "e": "f"},    # p 的第三个夸克
    2: {"s": "v", "e": "h"}
}

d_snk = {
    0: {"s": "p", "e": "d"},
    1: {"s": "x", "e": "j"},
    2: {"s": "z", "e": "l"}     # n 的第三个夸克
}

u_src = {
    0: {"s": "m", "e": "a"},
    1: {"s": "q", "e": "e"},    # p 的第三个夸克
    2: {"s": "u", "e": "g"}
}

d_src = {
    0: {"s": "o", "e": "c"},
    1: {"s": "w", "e": "i"},
    2: {"s": "y", "e": "k"}     # n 的第三个夸克
}

NP_indexes = []
NP_coeff = []

# 固定 sink, 交换 source 的夸克进行收缩
# 枚举 source 端 3 个 u 和 3 个 d 的全部置换组合
for u_perm in itertools.permutations(range(3)):
    for d_perm in itertools.permutations(range(3)):
        index = ",".join([
            # 禁止 autopep8 格式化以下文本
            # fmt: off
            u_snk[0]["s"]+u_src[u_perm[0]]["s"]+u_snk[0]["e"]+u_src[u_perm[0]]["e"],
            u_snk[1]["s"]+u_src[u_perm[1]]["s"]+u_snk[1]["e"]+u_src[u_perm[1]]["e"],
            u_snk[2]["s"]+u_src[u_perm[2]]["s"]+u_snk[2]["e"]+u_src[u_perm[2]]["e"],
            d_snk[0]["s"]+d_src[d_perm[0]]["s"]+d_snk[0]["e"]+d_src[d_perm[0]]["e"],
            d_snk[1]["s"]+d_src[d_perm[1]]["s"]+d_snk[1]["e"]+d_src[d_perm[1]]["e"],
            d_snk[2]["s"]+d_src[d_perm[2]]["s"]+d_snk[2]["e"]+d_src[d_perm[2]]["e"],
            # fmt: on
        ])

        coeff = perm_sign(u_perm) * perm_sign(d_perm)
        NP_indexes.append(index)
        NP_coeff.append(coeff)
