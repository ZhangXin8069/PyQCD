from argparse import Namespace
"""工具层：GPU/CPU 后端切换、缓存 einsum、数组切片、I/O 读取。"""
from ._backend import set_backend, get_backend, get_backend_name
from ._base import cached_contract, clear_cache, ArraySlicer, levi_civita_tensor, creat_mom_list
from ._io import (
    readin_eigvecs, readin_eigvecs_gpu, readin_peram, readin_peram_time_slice,
    safe_save, check_dir_path,
)

__all__ = [
    "set_backend", "get_backend", "get_backend_name",
    "cached_contract", "clear_cache", "ArraySlicer", "levi_civita_tensor", "creat_mom_list",
    "readin_eigvecs", "readin_eigvecs_gpu", "readin_peram", "readin_peram_time_slice",
    "safe_save", "check_dir_path",
]

Namespace.__module__ = "pyqcd.tools"
