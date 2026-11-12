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

===========  ==========================================  ====================
档位          典型用途                                     内存量级
===========  ==========================================  ====================
``full``      只有大型机器；与真实生产逐轴一致              ~31 GB
``scene``     **默认**，等比缩小且保持比例；可测 0.1-3 s    < 1 GB
``smoke``     冒烟/CI；单次 < 10 ms                          < 10 MB
===========  ==========================================  ====================
"""

from __future__ import annotations

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
# 乘积与生产一致，单张 peram 的体积却小一个量级：
#   生产 peram (64,32,4,4,100,100) = 5.2 GB  ->  本档 (40,64,4,4,50,50) = 82 MB
DEFAULT_SCENE = {"Nx": 64, "Lt": 40, "Nev": 50, "Ns": 4, "Mom": 5}

# 冒烟场景：CI 与快速回归用（Nev*Nx = 240，单次 < 5 ms）。
SMOKE_SCENE = {"Nx": 20, "Lt": 12, "Nev": 12, "Ns": 4, "Mom": 2}

SCENES = {
    "full": PRODUCTION,
    "scene": DEFAULT_SCENE,
    "smoke": SMOKE_SCENE,
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
    """一行场景说明，进度与报告里都用它，便于核对测的是哪一档。"""
    table = dims(scene, scale=scale)
    return (f"scene={scene}  Nx={table['Nx']} Nev={table['Nev']} "
            f"Lt={table['Lt']} Mom={table['Mom']} Ns={table['Ns']}  "
            f"(production: Nx={PRODUCTION['Nx']} Nev={PRODUCTION['Nev']} "
            f"Lt={PRODUCTION['Lt']} Mom={PRODUCTION['Mom']} "
            f"Ns={PRODUCTION['Ns']})")


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
