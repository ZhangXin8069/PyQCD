#!/usr/bin/env python3
"""原地裁掉共享库的绝对 RUNPATH 条目（不改库大小、不动其它动态段）。

为什么需要它：conda 的编译器 wrapper 会无条件给产物注入
``-Wl,-rpath,<prefix>/lib``，连一个空 ``.so`` 都带。带绝对 RUNPATH 的
``libpyqcdeinsum_cuda.so`` 会在进程里拉进**第二份 libcublas**（构建机的 12.4 与
宿主 cupy 自带的那份并存）；实测症状是调过本库之后 cupy 的 complex64 GEMM 恒报
``CUBLAS_STATUS_NOT_SUPPORTED`` —— 两份 cuBLAS 抢同一份进程级状态，不是内核算错。
``LDFLAGS``/``LD_RUN_PATH`` 屏蔽与 ``-Xlinker -rpath`` 覆盖都拦不住 wrapper，
因此在链接后做这一步。

用法::

    python3 strip_rpath.py <library.so> [--keep <dir> ...]

``--keep`` 指定的目录会原样保留在 RUNPATH 里（用于 ``$ORIGIN`` 之类的相对项或
确实需要的运行时搜索路径）；不传 ``--keep`` 时把所有条目清空。

退出码：``0`` 处理完成（无论是否真的裁过）；``2`` 参数/读写失败。
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

_DT_NULL = 0
_DT_STRTAB = 5
_DT_RPATH = 15
_DT_RUNPATH = 29
_PT_LOAD = 1
_PT_DYNAMIC = 2

_ELF_MAGIC = b"\x7fELF"


def _dynamic_entries(blob: bytes) -> tuple[list[tuple[int, int]], int, int, int]:
    """返回 (dynamic 条目, 其文件偏移, 条目大小, 字节序标志)。"""
    if blob[:4] != _ELF_MAGIC:
        raise ValueError("not an ELF file")
    is64 = blob[4] == 2
    little = blob[5] == 1
    endian = "<" if little else ">"
    if is64:
        e_phoff = struct.unpack_from(endian + "Q", blob, 0x20)[0]
        e_phentsize = struct.unpack_from(endian + "H", blob, 0x36)[0]
        e_phnum = struct.unpack_from(endian + "H", blob, 0x38)[0]
        ph_format = endian + "IIQQQQQQ"
    else:
        e_phoff = struct.unpack_from(endian + "I", blob, 0x1C)[0]
        e_phentsize = struct.unpack_from(endian + "H", blob, 0x2A)[0]
        e_phnum = struct.unpack_from(endian + "H", blob, 0x2C)[0]
        ph_format = endian + "IIIIIIII"

    loads: list[tuple[int, int, int]] = []      # (vaddr, offset, filesz)
    dynamic: tuple[int, int] | None = None
    for index in range(e_phnum):
        base = e_phoff + index * e_phentsize
        fields = struct.unpack_from(ph_format, blob, base)
        if is64:
            p_type, _flags, p_offset, p_vaddr, _paddr, p_filesz = fields[:6]
        else:
            p_type, p_offset, p_vaddr, _paddr, p_filesz = fields[:5]
        if p_type == _PT_LOAD:
            loads.append((p_vaddr, p_offset, p_filesz))
        elif p_type == _PT_DYNAMIC:
            dynamic = (p_offset, p_filesz)
    if dynamic is None:
        raise ValueError("no PT_DYNAMIC segment")

    entry_size = 16 if is64 else 8
    pack = endian + ("qQ" if is64 else "iI")
    entries: list[tuple[int, int]] = []
    offset, size = dynamic
    for position in range(offset, offset + size, entry_size):
        tag, value = struct.unpack_from(pack, blob, position)
        entries.append((tag, value))
        if tag == _DT_NULL:
            break
    return entries, offset, entry_size, 1 if little else 0


def _vaddr_to_offset(loads: list[tuple[int, int, int]], address: int) -> int | None:
    for vaddr, offset, filesz in loads:
        if vaddr <= address < vaddr + filesz:
            return offset + (address - vaddr)
    return None


def strip_rpath(path: Path, keep: tuple[str, ...] = ()) -> bool:
    """把 RUNPATH/RPATH 条目裁到 ``keep``，返回是否改动过。

    默认清空（``keep=()``）；``keep`` 里的条目按原顺序保留。新串写回**同一偏移**
    并补 NUL 填充，因此不会移动任何其它段。
    """
    blob = bytearray(path.read_bytes())
    entries, _offset, _entry_size, _little = _dynamic_entries(bytes(blob))
    strtab = None
    targets: list[tuple[int, int]] = []
    for tag, value in entries:
        if tag == _DT_STRTAB:
            strtab = value
        elif tag in (_DT_RPATH, _DT_RUNPATH):
            targets.append((tag, value))
    if strtab is None or not targets:
        return False

    loads: list[tuple[int, int, int]] = []
    is64 = blob[4] == 2
    little = blob[5] == 1
    endian = "<" if little else ">"
    if is64:
        e_phoff = struct.unpack_from(endian + "Q", blob, 0x20)[0]
        e_phentsize = struct.unpack_from(endian + "H", blob, 0x36)[0]
        e_phnum = struct.unpack_from(endian + "H", blob, 0x38)[0]
        ph_format = endian + "IIQQQQQQ"
    else:
        e_phoff = struct.unpack_from(endian + "I", blob, 0x1C)[0]
        e_phentsize = struct.unpack_from(endian + "H", blob, 0x2A)[0]
        e_phnum = struct.unpack_from(endian + "H", blob, 0x2C)[0]
        ph_format = endian + "IIIIIIII"
    for index in range(e_phnum):
        fields = struct.unpack_from(ph_format, blob, e_phoff + index * e_phentsize)
        if is64:
            p_type, _flags, p_offset, p_vaddr, _paddr, p_filesz = fields[:6]
        else:
            p_type, p_offset, p_vaddr, _paddr, p_filesz = fields[:5]
        if p_type == _PT_LOAD:
            loads.append((p_vaddr, p_offset, p_filesz))

    changed = False
    for _tag, value in targets:
        offset = _vaddr_to_offset(loads, strtab + value)
        if offset is None:
            continue
        end = blob.index(b"\0", offset)
        current = bytes(blob[offset:end]).decode("utf-8", "replace")
        kept = [item for item in current.split(":") if item and item in keep]
        replacement = ":".join(kept).encode()
        if replacement == current.encode():
            continue
        if len(replacement) > end - offset:
            raise ValueError(f"replacement longer than the original string: {current!r}")
        blob[offset:end] = replacement + b"\0" * (end - offset - len(replacement))
        changed = True
    if changed:
        path.write_bytes(bytes(blob))
    return changed


def main(argv: list[str]) -> int:
    keep: list[str] = []
    names: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--keep":
            index += 1
            if index >= len(argv):
                print("strip_rpath: --keep needs a value", file=sys.stderr)
                return 2
            keep.append(argv[index])
        else:
            names.append(item)
        index += 1
    if not names:
        print("usage: strip_rpath.py <library.so> [--keep <dir> ...]", file=sys.stderr)
        return 2
    failures = 0
    for name in names:
        path = Path(name)
        try:
            if not path.is_file():
                print(f"strip_rpath: missing {path}", file=sys.stderr)
                failures += 1
                continue
            changed = strip_rpath(path, tuple(keep))
        except Exception as error:                       # noqa: BLE001 - 诊断用
            print(f"strip_rpath: {path}: {type(error).__name__}: {error}",
                  file=sys.stderr)
            failures += 1
            continue
        print(f"strip_rpath: {path} {'stripped' if changed else 'unchanged'}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
