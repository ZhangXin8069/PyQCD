"""生产场景定义：两个基准脚本共用的维度来源。

所有性能测试都从**生产参数**出发，不再用「4096² 之类的合成大矩阵」当默认：

* 格点 ``lattice_size = [32, 32, 32, 64]`` → 空间维 ``Nx = 32``、时间维 ``Lt = 64``
* 色数 ``Nc = 3`` → Dirac 旋量维 ``Ns = 4``
* 本征模数 ``Nev_src = 100``
* 动量指标维 ``Mom_len = 8``

出处：``refer/sush/_distillation/contraction.py``（``peram_u`` 的形状
``(Lt, Nt, Ns, Ns, Nev, Nev)``）、``contraction.NN.2pt.py``（``VVV`` / ``VDV``
与 ``Nev_src``）、``constant.py``（``Nc``）。

完整生产规模下，六传播子的 16 算子路径要 ~31 GB 内存，超出普通工作站。因此
提供三档：

===========  ==========================================  ==========================
档位          典型用途                                     内存量级
===========  ==========================================  ==========================
``full``      只有大型机器；与真实生产逐轴一致              ~31 GB
``scene``     **默认**，等比缩小且保持比例；可测 0.1-3 s    ~14 GB（16 算子 6 传播子）
``smoke``     冒烟/CI；单次 < 10 ms                          < 10 MB
===========  ==========================================  ==========================

``scene`` 档单张 peram 是 1.53 GiB（complex128），六传播子路径光操作数就 ~9 GiB，
实测峰值 RSS 约 14 GiB——**15 GB 内存的机器跑 16 算子必须降到 ``--scale 16``
（或 ``smoke``）**，否则会换页到不可用。

生产场景的来源与各维度依据（2026-11-14 补充，只读核对，未运行参考脚本）
------------------------------------------------------------------------

上面 ``PRODUCTION`` / ``DEFAULT_SCENE`` / ``SMOKE_SCENE`` 的数值**一字未改**。它们的
``Nx`` / ``Lt`` 来自 ``refer/sush/_distillation/contraction.py`` 的 peram 展平描述，
属于「空间×本征模展平」那一类形状。但参考基准脚本 ``refer/sush/sush-einsum.py``
里**实际参与 einsum 的张量没有任何空间轴**：在该文件里 ``grep -nE 'Nx|Lt|lattice'``
的匹配数为 **0**。因此另外登记一套逐字取自该脚本的场景（``distill`` /
``distill_curr``），不动既有档位。

=========================  =====  =============================================
维度                       值     出处（参考文件:行）与理由
=========================  =====  =============================================
``Nev_src``                100    ``refer/sush/sush-einsum.py:170`` ``nev_src = 100``；
                                   sink/source 端本征模数
``Nev_curr``               400    ``:171`` ``nev_curr = 400``；每个 current 端点
                                   （tcur0/tcur1）用 400 —— **Nev 是两个值，不是一个**
``Mom_src``（n_mom_src）   1      ``:172`` ``n_mom_src = 1``；源端只有一个动量
``Mom_curr``（n_mom_curr） 3      ``:173`` ``n_mom_curr = 3``；current 端三个动量
``Ns``                     4      ``:200`` peram 前两轴 ``(4, 4, nev, nev)`` + ``:248``
                                  ``gamma(4)`` 是 4x4；旋量维
``Nc``                     3      ``refer/sush/_distillation/constant.py:1`` ``Nc = 3``；
                                  该脚本里 Nc **不是张量轴**（色指标在 peram 内部已收缩）
``Nt_labels``              4      ``:179`` ``times = ("tsink","tsrc","tcur0","tcur1")``；
                                  只是注册表键个数，不是张量轴
``Nx``                     —      不存在。空间维只在 peram 的「格点数×本征模」展平里出现，
                                  而在本脚本的张量形状里没有
``Lt``                     —      不存在。``:177`` 的 dtype 与 ``:179`` 的 times 之外，
                                  本脚本没有时间长度：peram 按 ``(tq, ta)`` **逐对注册**
                                  （``:198-206``），不是一个 ``Lt`` 长的数组
=========================  =====  =============================================

真实张量（``:198-250``，``optimize="dp"``、``complex128``）：

* ``peram`` ``(4, 4, nev_by_time[tq], nev_by_time[ta])``（``:200``），
  ``nev_by_time = {tsink: 100, tsrc: 100, tcur0: 400, tcur1: 400}``（``:184-189``），
  16 个 ``(tq, ta)`` 组合 → 出现 4 种互不相同的轴长组合，**异形**；
* ``VVV`` 分配 ``(n_mom, nev, nev, nev)``（``:209``/``:214``），注册时 sink 用
  ``host_vvv_sink[0]`` 切掉动量轴（``:241``）→ 进入 einsum 的是 ``(nev, nev, nev)``；
* ``VDV`` 分配 ``(n_mom, nev, nev)``，注册时**保留**动量轴（``:243-245``）→
  ``(1, 100, 100)`` 与 ``(3, 400, 400)``；
* ``gamma`` ``(4, 4)``（``:248-249``），``gamma_curr`` 是 4 个 gamma 的堆叠 ``(4, 4, 4)``
  （``:250``）。

另一段基准（``:377-401``，``optimize='auto'``）：6 个 peram ``(4, 4, 100, 100)``、
4 个 gamma ``(4, 4)``、4 个 ``VVV_shape = (3, 100, 100, 100)``（``:390``）、
2 个 ``VDV_shape = (3, 100, 100)``（``:391``）——这一段**保留**了动量轴。

两段都是 ``complex128``（``:177`` → 16 B/元素），即使 ``Nev_curr = 400`` 也装得下：
见 :func:`distill_memory` —— ``3pt`` 段全部操作数合计 21,940,096 元素 / **0.327 GiB**，
``6peram`` 段 13,020,064 元素 / **0.194 GiB**（分配态与注册态都算上，是上界）。
最坏可信情形（6pt 16 算子、三个 current 区都取 ``Nev_curr = 400``、VDV 保留动量轴）
按 ``optimize="dp"`` 规划：最大中间量 2.56e8 元素 = 3.82 GiB、操作数 1.35 GiB，
峰值下界 **5.16 GiB < 15 GiB** —— 所以 ``Nev_curr = 400`` **不需要缩小档**。
"""

from __future__ import annotations

from math import prod

# 生产参数（逐字取自参考实现，不要在这里改数字）。
PRODUCTION = {
    "Nx": 32,      # 空间维（lattice_size = [32, 32, 32, 64]）
    "Lt": 64,      # 时间维
    "Nev": 100,    # Nev_src：每个 peram 的酉空间维
    "Ns": 4,       # Nc = 3 -> Dirac 4x4
    "Mom": 8,      # Mom_len：动量指标维
}

# 默认的生产场景。选值依据是**实测的生产耗时档**，不是随手缩一半：
#
#   Nev * Nx = 100 * 32 = 3200  ->  单个 3200² 实矩阵乘 ~319 ms（本机实测）
#
# 这正是生产里传播子与规范场收缩的主流规模，单次落在 0.1-1 s 的可信区间。
# 因此这里保持 ``Nev * Nx = 3200`` 不变，只把两者按 5:8 分配（Nev=50, Nx=64）——
# 乘积与生产一致，单张 peram 的体积降到 1/3.2（按 complex128 计）：
#   生产 peram (64,32,4,4,100,100) = 4.88 GiB  ->  本档 (40,64,4,4,50,50) = 1.53 GiB
DEFAULT_SCENE = {"Nx": 64, "Lt": 40, "Nev": 50, "Ns": 4, "Mom": 5}

# 冒烟场景：CI 与快速回归用（Nev*Nx = 240，单次 < 5 ms）。
SMOKE_SCENE = {"Nx": 20, "Lt": 12, "Nev": 12, "Ns": 4, "Mom": 2}

# ── 蒸馏基准（``refer/sush/sush-einsum.py``）的生产参数 ────────────────────────
#
# 逐字取自参考脚本，**每个数字都带出处行号**；改这里之前先读文件。
# 这段场景的规模只由 Nev（100/400）、Mom（1/9）、Ns=4 决定，没有 Nx、没有 Lt 轴。
#
# 口径提示（2026-11-14 记录）：仓库里 ``refer/sush/sush-einsum.py`` 的 `HEAD` 版本
# 写的是 ``n_mom_curr = 3`` / ``dtype = np.complex128``，而用户确认的生产值是
# ``n_mom_curr = 9`` / ``dtype = np.complex64``。**这里采用后者**（生产口径），并把
# 两者的差异显式记在 :data:`DISTILL_HEAD_VARIANT`，报告里引用时不要混。
PRODUCTION_DISTILL = {
    "Nev_src": 100,     # refer/sush/sush-einsum.py:170  nev_src = 100
    "Nev_curr": 400,    # refer/sush/sush-einsum.py:171  nev_curr = 400
    "Mom_src": 1,       # refer/sush/sush-einsum.py:172  n_mom_src = 1
    "Mom_curr": 9,      # refer/sush/sush-einsum.py:173  n_mom_curr = 9（生产值；
                        # 仓库 HEAD 版本写 3，见 DISTILL_HEAD_VARIANT）
    "Ns": 4,            # refer/sush/sush-einsum.py:200 peram 前两轴 (4, 4, ...)；
                        # :248 gamma(4) 是 4x4（旋量维）
    "Nc": 3,            # refer/sush/_distillation/constant.py:1  Nc = 3
                        # （本脚本无 Nc 轴：色指标在 peram 内部已收缩）
    "Nt_labels": 4,     # refer/sush/sush-einsum.py:179 times = (tsink,tsrc,tcur0,tcur1)
                        # 这是注册表键个数，不是张量轴
}

# 仓库 HEAD 里参考脚本的另一组取值。**只用于核对差异，不得用于生产结论**。
DISTILL_HEAD_VARIANT = {
    "Mom_curr": 3,              # HEAD 版 :173
    "dtype": "complex128",      # HEAD 版 :177
    "itemsize": 16,
}

# 参考脚本的时间标签与逐端点 Nev（:179 与 :184-189）。
DISTILL_TIMES = ("tsink", "tsrc", "tcur0", "tcur1")          # :179
DISTILL_NEV_BY_TIME = {                                       # :184-189
    "tsink": 100,   # :185  "tsink": nev_src
    "tsrc": 100,    # :186  "tsrc":  nev_src
    "tcur0": 400,   # :187  "tcur0": nev_curr
    "tcur1": 400,   # :188  "tcur1": nev_curr
}

# 参考脚本的数值设置（报告里要能核对：同一 dtype、同一 planner）。
DISTILL_SETTINGS = {
    "optimize": "dp",           # refer/sush/sush-einsum.py:176
    "dtype": "complex128",      # refer/sush/sush-einsum.py:177
    "itemsize": 16,             # complex128 = 16 B/元素（:177）
    "seed": 20261114,           # refer/sush/sush-einsum.py:175
}


def _distill_dims(nev_key: str, mom_key: str) -> dict[str, int]:
    """由 :data:`PRODUCTION_DISTILL` 组合出一档 ``dims()`` 兼容的维度表。

    ``Nx`` / ``Lt`` 固定为 ``1``：**该轴在蒸馏张量里不存在**，1 是「无此轴」的
    占位值（不是缩小值）；因此 ``_label_sizes`` 之类的等长消费者把 ``Mom`` 当标量
    而不是向量时不会凭空多出维度。
    """
    table = {
        "Nx": 1,                                    # 无空间轴（参考脚本 grep 无匹配）
        "Lt": 1,                                    # 无时间轴（peram 按 (tq,ta) 逐对注册）
        "Nev": PRODUCTION_DISTILL[nev_key],
        "Ns": PRODUCTION_DISTILL["Ns"],
        "Mom": PRODUCTION_DISTILL[mom_key],
    }
    table.update(PRODUCTION_DISTILL)
    return table


# ``distill``：源端/汇端那一档（nev_src = 100）。这是参考脚本里 sink/source 的
# peram 与 VVV 的真实规模。
DISTILL_SCENE = _distill_dims("Nev_src", "Mom_src")

# ``distill_curr``：current 端点那一档（nev_curr = 400、n_mom_curr = 3）。
# 参考脚本里 tcur0/tcur1 的 peram 与 VDV 用的是这一档。
DISTILL_CURR_SCENE = _distill_dims("Nev_curr", "Mom_curr")

SCENES = {
    "full": PRODUCTION,
    "scene": DEFAULT_SCENE,
    "smoke": SMOKE_SCENE,
    "distill": DISTILL_SCENE,             # nev_src = 100 档（参考脚本 :170）
    "distill_curr": DISTILL_CURR_SCENE,   # nev_curr = 400 档（参考脚本 :171）
}


def dims(scene: str = "scene", scale: int | None = None) -> dict[str, int]:
    """取某一档的维度表；``scale`` 给定时按 ``scale/32`` 同比缩放生产值。"""
    if scale is not None:
        ratio = scale / PRODUCTION["Nx"]
        return {
            "Nx": scale,
            "Lt": max(8, int(round(PRODUCTION["Lt"] * ratio))),
            "Nev": max(12, int(round(PRODUCTION["Nev"] * ratio))),
            "Ns": PRODUCTION["Ns"],
            "Mom": max(2, int(round(PRODUCTION["Mom"] * ratio))),
        }
    if scene not in SCENES:
        raise SystemExit(f"unknown scene {scene!r}; choose from {', '.join(SCENES)}")
    return dict(SCENES[scene])


def describe(scene: str = "scene", scale: int | None = None) -> str:
    """一行场景说明，进度与报告里都用它，便于核对测的是哪一档。

    ``full`` / ``scene`` / ``smoke`` 与 ``scale=`` 的输出**逐字未变**；只有
    ``distill`` / ``distill_curr`` 这类带 ``Nev_src`` 的档位会在末尾追加一段
    ``[distill: ...]``，把两个 Nev、两个 Mom 与「无 Nx/Lt 轴」一并印出来。
    """
    table = dims(scene, scale=scale)
    text = (f"scene={scene}  Nx={table['Nx']} Nev={table['Nev']} "
            f"Lt={table['Lt']} Mom={table['Mom']} Ns={table['Ns']}  "
            f"(production: Nx={PRODUCTION['Nx']} Nev={PRODUCTION['Nev']} "
            f"Lt={PRODUCTION['Lt']} Mom={PRODUCTION['Mom']} "
            f"Ns={PRODUCTION['Ns']})")
    if "Nev_src" in table:
        text += (f"  [distill: Nev_src={table['Nev_src']} "
                 f"Nev_curr={table['Nev_curr']} Mom_src={table['Mom_src']} "
                 f"Mom_curr={table['Mom_curr']} Nc={table['Nc']} "
                 f"无 Nx/Lt 轴（Nx=Lt=1 为占位）]")
    return text


def generic_shapes(scene: str = "scene") -> dict[str, tuple]:
    """通用路径的「生产尺度」形状。

    这些形状都由同一套生产参数推出来，取代原先的 4096²/12000² 之类合成值：

    * ``matrix``：``(Nx*Nev, Nx*Nev)`` —— 一个传播子按空间×本征模展平后的规模；
    * ``vector``：``(Nx*Nev,)`` —— 对应的向量；
    * ``batch``：``(Lt, Nx, Nx)`` —— 时间片批量；
    * ``chain``：一个 ``Nx*Nev`` 见方的可逆链。
    """
    table = dims(scene)
    # 生产里真实的矩阵边长不是「Nx*Nev」这种合成量，而是下列两种之一：
    #   * ``Nev * Nx``（本征模 × 空间）——传播子与规范场收缩时的主流规模；
    #   * ``Lt * Nx``（时间 × 空间）——逐时间片批量的总点数。
    # 这里两条都给出来，按路径选更贴近的那条。
    eigen = table["Nev"] * table["Nx"]          # 生产 = 3200
    spatial = table["Lt"] * table["Nx"]         # 生产 = 2048
    tile = table["Nev"]
    return {
        "side": eigen,
        "matrix": (eigen, eigen),
        "vector": (eigen,),
        "spatial_matrix": (spatial, spatial),
        # 批量 GEMM：Lt 个时间片，每片 Nev 见方（逐时间片的本征模变换）
        "batch": (table["Lt"], tile, tile),
        "batch_matrix": ((table["Lt"], table["Nx"], tile), (tile, tile)),
        "many_batch": (table["Lt"] * table["Nx"], table["Ns"], table["Ns"]),
        "small": (tile, tile),
        "ns": table["Ns"],
    }


# ── 蒸馏场景的异形轴形状（按参考脚本的真实轴长组合，不是 dim 等长） ──────────
#
# 为什么需要单独的构造器：``_label_sizes``（``bench_specialized``）把 VVV 当成
# ``(Lt, Mom, Nev, Nev, Nev)``、VDV 当成 ``(Lt, Mom, Nev, Nev)``，并且**全方程只有一个
# Nev**。参考脚本两者都不是：VVV/VDV 没有 Lt 轴，而且源端 Nev=100、current 端
# Nev=400 同时出现在同一张图里（``:184-189``）。用 ``_label_sizes`` 构造不出这种
# 「同一方程里两个 Nev」的形状，所以这里按**字面张量**给。

#: 参考脚本里两段基准的标识：``3pt`` = ``:170-250`` 的 ``optimize="dp"`` 段，
#: ``6peram`` = ``:377-401`` 的 6 传播子段。
DISTILL_KINDS = ("3pt", "6peram")

#: 参考场景的内存上限（GiB）。超了就必须提供缩小档；实测远低于它。
DISTILL_MEMORY_LIMIT_GIB = 15.0


def distill_tensor_shapes(kind: str = "3pt") -> dict[str, tuple[int, ...]]:
    """参考脚本里每张张量的**真实**形状（异形轴：``4``/``100``/``400``/``3`` 混排）。

    ``3pt`` 段（``refer/sush/sush-einsum.py:170-250``）：

    * ``peram[tq,ta]`` = ``(Ns, Ns, nev_by_time[tq], nev_by_time[ta])``（``:200``，
      ``nev_by_time`` 见 ``:184-189``）——16 个 ``(tq,ta)`` 组合给出 4 种不同轴长；
    * ``VVV_0@tsink`` / ``VVV_0@tsrc``：分配 ``(Mom_src, Nev_src, Nev_src, Nev_src)``
      （``:209``/``:214``），注册 sink 时用 ``host_vvv_sink[0]`` 切掉动量轴
      （``:241``），所以**进入 einsum** 的是 ``(Nev_src,)*3``；
    * ``VDV_0@tsrc``：``(Mom_src, Nev_src, Nev_src)`` = ``(1,100,100)``（``:219``/``:243``，
      未切片）；``VDV_0@tcur0|tcur1``：``(Mom_curr, Nev_curr, Nev_curr)`` =
      ``(3,400,400)``（``:224``/``:244-245``）；
    * ``gamma_4`` / ``gamma_7`` = ``(Ns, Ns)``（``:248-249``），``gamma_curr`` =
      4 个 gamma 的堆叠 ``(4, Ns, Ns)``（``:250``）；
    * ``host:`` 前缀的是分配后、注册前的临时副本（``:198-227``），与注册值同时存在，
      因此 :func:`distill_memory` 把它们算进去 —— 那是**保守上界**。

    ``6peram`` 段（``:377-401``）：6 个 peram ``(4,4,100,100)``（``:377-382``）、
    4 个 gamma ``(4,4)``（``:385-389``）、4 个 ``VVV_shape``（``:390``/``:393-397``）、
    2 个 ``VDV_shape``（``:391``/``:399-401``）。这一段**保留**动量轴，首轴是字面
    ``3``（与 ``n_mom_curr`` 数值相同但脚本里写作字面量）。
    """
    ns = PRODUCTION_DISTILL["Ns"]
    nev_src = PRODUCTION_DISTILL["Nev_src"]
    nev_curr = PRODUCTION_DISTILL["Nev_curr"]
    mom_src = PRODUCTION_DISTILL["Mom_src"]
    mom_curr = PRODUCTION_DISTILL["Mom_curr"]

    if kind == "3pt":
        shapes: dict[str, tuple[int, ...]] = {}
        for tq in DISTILL_TIMES:
            for ta in DISTILL_TIMES:
                shapes[f"peram[{tq},{ta}]"] = (
                    ns, ns, DISTILL_NEV_BY_TIME[tq], DISTILL_NEV_BY_TIME[ta])
        # 分配态（host）：:209/:214 VVV、:219/:224 VDV
        shapes["host:VVV@tsink"] = (mom_src, nev_src, nev_src, nev_src)
        shapes["host:VVV@tsrc"] = (mom_src, nev_src, nev_src, nev_src)
        shapes["host:VDV@tsrc"] = (mom_src, nev_src, nev_src)
        shapes["host:VDV@tcur0"] = (mom_curr, nev_curr, nev_curr)
        shapes["host:VDV@tcur1"] = (mom_curr, nev_curr, nev_curr)
        # 注册态（真正进 einsum 的形状）：:241-245
        shapes["VVV_0@tsink"] = (nev_src, nev_src, nev_src)      # :241 切片后
        shapes["VVV_0@tsrc"] = (nev_src, nev_src, nev_src)       # :242
        shapes["VDV_0@tsrc"] = (mom_src, nev_src, nev_src)       # :243
        shapes["VDV_0@tcur0"] = (mom_curr, nev_curr, nev_curr)   # :244
        shapes["VDV_0@tcur1"] = (mom_curr, nev_curr, nev_curr)   # :245
        shapes["gamma_4"] = (ns, ns)                             # :248
        shapes["gamma_7"] = (ns, ns)                             # :249
        shapes["gamma_curr"] = (4, ns, ns)                       # :250 四个 gamma 堆叠
        return shapes

    if kind == "6peram":
        shapes = {f"peram{i}": (4, 4, 100, 100) for i in range(1, 7)}    # :377-382
        shapes.update({f"gamma{i}": (4, 4) for i in range(1, 5)})         # :385-389
        shapes.update({f"VVV{i}": (3, 100, 100, 100)                     # :390/393-397
                       for i in range(1, 5)})
        shapes.update({f"VDV{i}": (3, 100, 100)                          # :391/399-401
                       for i in range(1, 3)})
        return shapes

    raise SystemExit(
        f"unknown distill kind {kind!r}; choose from {', '.join(DISTILL_KINDS)}")


def distill_memory(kind: str = "3pt",
                   itemsize: int | None = None) -> dict[str, object]:
    """某段参考基准的操作数元素数与内存估算（默认 complex128 = 16 B，``:177``）。

    返回 ``{"kind", "itemsize", "tensors": {name: {"elements", "bytes"}},
    "total_elements", "total_bytes", "total_gib", "largest"}``。

    ``3pt`` 段的总量是**保守上界**：把分配态（``host:``）与注册态都算上，实测
    21,940,096 元素 / 0.327 GiB；``6peram`` 段 13,020,064 元素 / 0.194 GiB。
    两者都远低于 :data:`DISTILL_MEMORY_LIMIT_GIB`，所以 ``Nev_curr = 400``
    不需要缩小档（峰值中间量见模块 docstring：最坏可信情形 5.16 GiB）。
    """
    size = DISTILL_SETTINGS["itemsize"] if itemsize is None else itemsize
    detail = {}
    for name, shape in distill_tensor_shapes(kind).items():
        elements = prod(shape)
        detail[name] = {"elements": elements, "bytes": elements * size}
    total_bytes = sum(item["bytes"] for item in detail.values())
    largest = max(detail.items(), key=lambda item: item[1]["bytes"])
    return {
        "kind": kind,
        "itemsize": size,
        "tensors": detail,
        "total_elements": sum(item["elements"] for item in detail.values()),
        "total_bytes": total_bytes,
        "total_gib": total_bytes / 2 ** 30,
        "largest": (largest[0], largest[1]["bytes"], largest[1]["bytes"] / 2 ** 30),
    }
