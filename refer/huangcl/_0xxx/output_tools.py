"""数据保存用的工具包.

按文件格式提供 save_xxx 小函数, 每个函数都会自动创建父目录并打印保存信息.
当前支持保存的文件类型:
    图片: .png, .pdf, .svg 等 matplotlib 支持的格式, 用 save_fig.
    文本: .txt, 支持字符串和 PrettyTable 对象, 用 save_txt.
    单数组: .npy, 用 save_npy.
    多数组: .npz, 用 save_npz.
"""

from pathlib import Path

import numpy as np


def save_fig(path, figure, name=None, **kwargs) -> None:
    """
    保存 matplotlib 图片, 自动创建父目录并打印保存信息.

    参数:
    -----
    path: 保存路径 (含文件名和后缀), 后缀决定图片格式, 如 .png, .pdf, .svg.
    figure: matplotlib Figure 对象.

    可选项:
    name: 打印用的变量名, 默认打印 fig.
    kwargs: 原样传给 figure.savefig 的参数, 如 dpi, transparent.
            默认添加 bbox_inches="tight", 调用方传入时以调用方为准.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("bbox_inches", "tight")
    figure.savefig(path, **kwargs)
    print(f"saved {name or 'fig'}: {path}")


def save_txt(path, text, name=None) -> None:
    """
    保存文本, 自动创建父目录并打印保存信息.

    参数:
    -----
    path: 保存路径, 后缀为 .txt.
    text: 待写入的文本内容, 可为字符串或 PrettyTable 等可转字符串的对象.

    可选项:
    name: 打印用的变量名, 默认打印 txt.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text))
    print(f"saved {name or 'txt'}: {path}")


def save_npy(path, array, name=None) -> None:
    """
    保存单个 numpy 数组, 自动创建父目录并打印保存信息.

    参数:
    -----
    path: 保存路径, 后缀为 .npy.
    array: 待保存数组.

    可选项:
    name: 打印用的变量名, 默认打印 npy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    print(f"saved {name or 'npy'}: {path} {array.shape}")


def save_npz(path, name=None, **arrays) -> None:
    """
    保存多个 numpy 数组, 自动创建父目录并打印保存信息.

    参数:
    -----
    path: 保存路径, 后缀为 .npz.
    arrays: 关键字形式的待保存数组集合.

    可选项:
    name: 打印用的变量名, 默认打印 npz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    print(f"saved {name or 'npz'}: {path}")
