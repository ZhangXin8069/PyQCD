"""
Binary Data I/O Readers
=======================

Functions for reading lattice QCD binary data files:
- ``readin_eigvecs``: Read Laplacian eigenvectors (binary float64 format)
- ``readin_peram``: Read perambulator (propagator) data (binary float64 format)
- ``safe_save``: Save numpy arrays with fallback paths
- ``save_tensor_h5`` / ``load_tensor_h5``: h5py-based tensor persistence
  (the canonical read/write tool; accepts numpy, cupy and torch arrays)

Adapted from lqcddb io/write_date.py.

Binary Format Conventions
-------------------------
- All data is stored as float64 (double precision) in big-endian byte order
- Complex numbers are stored as interleaved [real, imag] pairs
- Eigenvector file: (Nev, Nx³, Nc, 2) float64 → (Nev, Nx³, Nc) complex128
- Perambulator file: per time source, 4 Dirac source index files,
  concatenated then reshaped as (4, Nt, Nev, 4, Nev, 2) → (Nt, 4, 4, Nev, Nev)
"""

import gzip

import numpy as np
import os
from typing import Optional

import h5py


def readin_eigvecs(file_path: str, Nx: int):
    """Read distillation Laplacian eigenvectors from binary file.

    File format: float64, shape (Nev, Nx³, 3, 2)
    where the last dimension stores [real, imag] pairs.

    Parameters
    ----------
    file_path : str
        Path to the eigenvector binary file, e.g.
        ``/path/to/eigvecs_t000_6250``.
    Nx : int
        Spatial lattice size (assumed isotropic: Nx = Ny = Nz).

    Returns
    -------
    ndarray, shape (Nev, Nx*Nx*Nx, 3), dtype complex128
        Eigenvectors reshaped into (eigenvector_index, spatial_site, color).
        ``Nev`` is auto-detected from the file size.
    """
    with open(file_path, 'rb') as f:
        eigvecs = np.fromfile(f, dtype='f8')

    # Auto-detect Nev from file size
    eigvecs_size = eigvecs.size
    Nev = int(eigvecs_size / (Nx * Nx * Nx * 3 * 2))

    # Reshape: (Nev, Nx³, 3, 2) → complex → (Nev, Nx³, 3)
    eigvecs = eigvecs.reshape(Nev, Nx * Nx * Nx, 3, 2)
    eigvecs = eigvecs[..., 0] + eigvecs[..., 1] * 1j

    return eigvecs


def readin_eigvecs_gpu(file_path: str, Nx: int, Nev_use: int = None):
    """Read eigenvectors directly to GPU memory.

    Returns the array in the active backend: cupy when the backend is
    'cupy', a torch tensor (on CUDA when available) when the backend is
    'torch', else a numpy array.
    Requires cupy to be installed and the backend to be set to 'cupy'.

    Parameters
    ----------
    file_path : str
        Path to eigenvector binary file.
    Nx : int
        Spatial lattice size.
    Nev_use : int, optional
        Number of eigenvectors to keep (truncate to first ``Nev_use``).
        If None, use all eigenvectors in the file.

    Returns
    -------
    ndarray, shape (Nev_use, Nx*Nx*Nx, 3), dtype complex128
        Eigenvectors on GPU.
    """
    from ._backend import get_backend_name, get_torch_device

    backend_name = get_backend_name()
    if backend_name == 'torch':
        return readin_eigvecs_tensor(
            file_path, Nx, Nev_use, backend='torch',
            device=get_torch_device())
    if backend_name == 'cupy':
        import cupy as cp

        with open(file_path, 'rb') as f:
            eigvecs = np.fromfile(f, dtype='f8')

        Nev_full = int(eigvecs.size / (Nx * Nx * Nx * 3 * 2))
        eigvecs = eigvecs.reshape(Nev_full, Nx * Nx * Nx, 3, 2)
        eigvecs = eigvecs[..., 0] + eigvecs[..., 1] * 1j

        if Nev_use is not None and Nev_use < Nev_full:
            eigvecs = eigvecs[:Nev_use]

        return cp.asarray(eigvecs)

    return readin_eigvecs(file_path, Nx)[:Nev_use] if Nev_use is not None \
        else readin_eigvecs(file_path, Nx)


def readin_peram(peram_dir: str, conf_id: str, Nt: int,
                 Nev1: Optional[int] = None):
    """Read distillation perambulator from binary files.

    Reads all 4 Dirac source index files per time source and assembles
    the full perambulator array.

    File naming convention:
        ``perams.{conf_id}.{d_source}.{t_source}``

    Parameters
    ----------
    peram_dir : str
        Directory containing perambulator binary files.
    conf_id : str
        Configuration ID (e.g., ``'6250'``).
    Nt : int
        Number of time slices.
    Nev1 : int, optional
        Truncate eigenvector indices to first ``Nev1``. If None, use all.

    Returns
    -------
    ndarray, shape (Nt, Nt, 4, 4, Nev1, Nev1), dtype complex128
        Full perambulator array. Axes:
        - axis 0: t_source (source time)
        - axis 1: t_sink (sink time)
        - axis 2: d_sink (Dirac sink index, 0..3)
        - axis 3: d_source (Dirac source index, 0..3)
        - axis 4: ev_sink (eigenvector sink index)
        - axis 5: ev_source (eigenvector source index)
    """
    # First, determine Nev from the first file
    test_file = f"{peram_dir}/perams.{conf_id}.0.0"
    with open(test_file, 'rb') as f:
        test_data = np.fromfile(f, dtype='f8')

    # Single d_source file: Nt * Nev * 4 * Nev * 2 = 8 * Nt * Nev²
    Nev_full = int(np.sqrt(test_data.size / (8 * Nt)))

    if Nev1 is None:
        Nev1 = Nev_full

    # Allocate output array: (t_source, t_sink, d_sink, d_source, ev_sink, ev_source)
    peram_cpu_all = np.zeros((Nt, Nt, 4, 4, Nev1, Nev1), dtype=complex)

    for t_source in range(Nt):
        # Read d_source=0
        with open(f"{peram_dir}/perams.{conf_id}.0.{t_source}", 'rb') as f:
            peram = np.fromfile(f, dtype='f8')

        # Read and append d_source=1,2,3
        for d_source in range(1, 4):
            with open(f"{peram_dir}/perams.{conf_id}.{d_source}.{t_source}", 'rb') as f:
                temp = np.fromfile(f, dtype='f8')
                peram = np.append(peram, temp)

        # Reshape: (d_source=4, t_sink=Nt, ev_source=Nev, d_sink=4, ev_sink=Nev, complex=2)
        peram = peram.reshape(4, Nt, Nev_full, 4, Nev_full, 2)

        # Transpose: (t_sink, d_sink, d_source, ev_sink, ev_source, complex)
        peram = peram.transpose(1, 3, 0, 4, 2, 5)

        # Convert to complex
        peram = peram[..., 0] + peram[..., 1] * 1j

        # Truncate to Nev1 and store
        peram_cpu_all[t_source] = peram[:, :, :, :Nev1, :Nev1]

    return peram_cpu_all


def readin_peram_time_slice(peram_dir: str, conf_id: str, t_source: int,
                             Nt: int, Nev1: Optional[int] = None):
    """Read perambulator for a single time source slice.

    More memory-efficient than ``readin_peram`` when only one time source
    is needed at a time (avoids allocating the full Nt×Nt array).

    Parameters
    ----------
    peram_dir : str
        Directory containing perambulator files.
    conf_id : str
        Configuration ID.
    t_source : int
        Source time slice to read.
    Nt : int
        Total number of time slices (used to auto-detect Nev).
    Nev1 : int, optional
        Truncate to this many eigenvectors.

    Returns
    -------
    ndarray, shape (Nt, 4, 4, Nev1, Nev1), dtype complex128
        Perambulator for the given time source.
        Axes: (t_sink, d_sink, d_source, ev_sink, ev_source).
    """
    # Auto-detect Nev
    test_file = f"{peram_dir}/perams.{conf_id}.0.{t_source}"
    with open(test_file, 'rb') as f:
        test_data = np.fromfile(f, dtype='f8')
    Nev_full = int(np.sqrt(test_data.size / (8 * Nt)))

    if Nev1 is None:
        Nev1 = Nev_full

    # Read all 4 d_source files
    with open(test_file, 'rb') as f:
        peram = np.fromfile(f, dtype='f8')
    for d_source in range(1, 4):
        with open(f"{peram_dir}/perams.{conf_id}.{d_source}.{t_source}", 'rb') as f:
            temp = np.fromfile(f, dtype='f8')
            peram = np.append(peram, temp)

    # Reshape and transpose
    peram = peram.reshape(4, Nt, Nev_full, 4, Nev_full, 2)
    peram = peram.transpose(1, 3, 0, 4, 2, 5)  # (t_sink, d_sink, d_source, ev_sink, ev_source, complex)
    peram = peram[..., 0] + peram[..., 1] * 1j

    return peram[:, :, :, :Nev1, :Nev1]


def safe_save(file: str, arr, allow_pickle: bool = True,
              fix_imports: bool = True, fallback_dirs: list = None):
    """Save array as .npy file with automatic fallback on failure.

    Tries the primary path first; on OSError falls back to
    ``fallback_dirs`` (if provided), then to a timestamped subdirectory
    of the current working directory.

    Handles cupy arrays by calling ``.get()`` before saving.

    Parameters
    ----------
    file : str
        Target file path. ``.npy`` extension is auto-appended if missing.
    arr : ndarray
        Array to save (numpy or cupy).
    allow_pickle : bool
        Passed to ``numpy.save``.
    fix_imports : bool
        Passed to ``numpy.save``.
    fallback_dirs : list of str, optional
        Additional directories to try before auto-fallback.

    Returns
    -------
    str
        The actual path where the file was saved.

    Raises
    ------
    OSError
        If all save locations fail.
    """
    file = str(file)
    if not file.endswith('.npy'):
        file = file + '.npy'

    basename = os.path.basename(file)

    def _try_save(dir_path):
        candidate = os.path.join(dir_path, basename)
        try:
            os.makedirs(dir_path, exist_ok=True)
            _arr = arr
            if hasattr(_arr, 'get') and callable(_arr.get):
                _arr = _arr.get()
            np.save(candidate, _arr, allow_pickle=allow_pickle,
                    fix_imports=fix_imports)
            return candidate
        except OSError as e:
            print(f"safe_save: failed to save to '{candidate}': {e}")
            return None

    # 1. Try primary path
    primary_dir = os.path.dirname(file) or '.'
    result = _try_save(primary_dir)
    if result is not None:
        return result

    # 2. Try user-provided fallback dirs
    if fallback_dirs:
        for d in fallback_dirs:
            result = _try_save(d)
            if result is not None:
                print(f"safe_save: saved to fallback path '{result}'")
                return result

    # 3. Auto-fallback to cwd
    import time
    auto_dir = os.path.join(
        os.getcwd(), 'data',
        time.strftime('fallback_%Y%m%d_%H%M%S'))
    result = _try_save(auto_dir)
    if result is not None:
        print(f"safe_save: saved to auto fallback path '{result}'")
        return result

    raise OSError(f"safe_save: all save locations exhausted for '{basename}'")


def check_dir_path(save_path: str):
    """Create directory if it doesn't exist.

    Parameters
    ----------
    save_path : str
        Directory path to ensure exists.
    """
    import pathlib
    path = pathlib.Path(save_path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f'mkdir_save_path: {save_path}')


# ═══════════════════════════════════════════════════════════════════
# h5py tensor persistence (canonical read/write tool)
# ═══════════════════════════════════════════════════════════════════

def _to_numpy(arr):
    """Convert torch/cupy arrays to numpy (CPU) for h5py storage."""
    if hasattr(arr, 'detach'):          # torch.Tensor
        return arr.detach().cpu().numpy()
    if hasattr(arr, 'get'):             # cupy.ndarray
        return arr.get()
    return arr


def save_tensor_h5(arr, file_path: str, dataset: str = 'data',
                   verbose: bool = False):
    """Save an array/tensor (numpy, cupy or torch) to an HDF5 file via h5py.

    Each call opens its own file handle (with statement) so concurrent
    calls from multiple MPI ranks / threads are safe.

    Parameters
    ----------
    arr : ndarray or torch.Tensor
        Array to persist. numpy dtypes are stored verbatim; torch complex
        dtypes map to the matching numpy complex dtype.
    file_path : str
        Target HDF5 file path (``.h5`` appended if missing).
    dataset : str
        Dataset name inside the file (default ``'data'``).
    """
    file_path = str(file_path)
    if not file_path.endswith('.h5'):
        file_path = file_path + '.h5'
    os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
    arr_np = _to_numpy(arr)
    with h5py.File(file_path, 'w') as f:
        f.create_dataset(dataset, data=arr_np)
    if verbose:
        print(f"save_tensor_h5: {np.shape(arr)} {getattr(arr_np, 'dtype', '?')} "
              f"-> {file_path} (dataset='{dataset}')")


def load_tensor_h5(file_path: str, dataset: str = 'data',
                   backend='numpy', device=None, verbose: bool = False):
    """Load a tensor from an HDF5 file (h5py).

    Parameters
    ----------
    file_path : str
        HDF5 file path.
    dataset : str
        Dataset name (default ``'data'``).
    backend : str
        Return type: ``'numpy'`` (default) or ``'torch'`` (tensor on
        ``device``, default CPU).
    device : optional
        Target device for the torch backend.
    verbose : bool

    Returns
    -------
    ndarray or torch.Tensor
        The stored array.
    """
    file_path = str(file_path)
    if not file_path.endswith('.h5'):
        file_path = file_path + '.h5'
    with h5py.File(file_path, 'r') as f:
        arr = f[dataset][...]
    if backend == 'torch':
        import torch
        t = torch.from_numpy(arr)
        if device is not None:
            t = t.to(device)
        if verbose:
            print(f"load_tensor_h5: {arr.shape} {arr.dtype} "
                  f"<- {file_path} (torch, device={device})")
        return t
    if verbose:
        print(f"load_tensor_h5: {arr.shape} {arr.dtype} <- {file_path}")
    return arr


def readin_eigvecs_tensor(file_path: str, Nx: int, Nev_use=None,
                          backend='numpy', device=None):
    """Read Laplacian eigenvectors into the requested backend array.

    Like ``readin_eigvecs_gpu`` but backend-agnostic: returns a torch
    tensor (on ``device``) when backend='torch', a cupy array when
    backend='cupy', else a numpy array.
    """
    eigvecs = readin_eigvecs(file_path, Nx)
    if Nev_use is not None and Nev_use < eigvecs.shape[0]:
        eigvecs = eigvecs[:Nev_use]
    if backend == 'torch':
        import torch
        t = torch.from_numpy(eigvecs)
        if device is not None:
            t = t.to(device)
        return t
    if backend == 'cupy':
        import cupy as cp
        return cp.asarray(eigvecs)
    return eigvecs


# ═══════════════════════════════════════════════════════════════════
# L.Liu ASCII 关联函数格式读写（整合 donghx write_data_ascii + 读入端）
# ═══════════════════════════════════════════════════════════════════

def write_data_ascii(data, T, L, filename, is_complex=True, verbose=False):
    """L.Liu 格式 ASCII 写出（照抄 donghx input_output_4_cupy.write_data_ascii）。

    首行头 "nsamples T is_cx L 1"；每行 = 时间计数(i%T) + 实部 [+ 虚部]；
    filename 以 .gz 结尾时自动 gzip 压缩（原版 savetxt 的 .gz 分支语义）。

    Args:
        data: (nsamples, T, …) 或 (nsamples, T·k) 数组；一维按单样本处理。
        T: 时间长度。 L: 空间格点数。
        is_complex: 复数写出（实/虚两列）。
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[None, :]
    nsamples = data.shape[0]
    flat = data.reshape(nsamples * T, -1)
    counter = (np.arange(flat.shape[0]) % T)[:, None]
    head = f"{nsamples} {T} {int(is_complex)} {L} 1"
    _dir = os.path.dirname(filename)
    if _dir and not os.path.exists(_dir):
        os.makedirs(_dir)
    if os.path.isfile(filename):
        if verbose:
            print(filename + " already exists, overwriting...")
    fh = gzip.open(filename, "wb") if filename.endswith(".gz") \
        else open(filename, "wb")
    try:
        if is_complex:
            block = np.concatenate((counter, flat.real, flat.imag), axis=1)
            np.savetxt(fh, block, header=head, comments="",
                       fmt=["%i", "%.32e", "%.32e"])
        else:
            block = np.concatenate(
                (counter, np.real(flat)), axis=1)
            np.savetxt(fh, block, header=head, comments="",
                       fmt=["%i", "%.32e"])
    finally:
        fh.close()
    if verbose:
        print(f"saved {filename} ({nsamples} samples x T={T}, "
              f"{'complex' if is_complex else 'real'})")


def read_data_ascii(filename):
    """L.Liu 格式 ASCII 读入（与 write_data_ascii 配对的解析端）。

    Returns:
        (data, meta)：data 形状 (nsamples, T, k)——is_cx 时 k=2 为
        实/虚两列（复数组 (nsamples,T,k/2) 更直观，此处保持列结构并附
        meta['is_complex']）；meta 含 nsamples/T/is_complex/L/version。
        .gz 自动解压。
    """
    opener = gzip.open if filename.endswith(".gz") else open
    with opener(filename, "rt") as f:
        tokens = f.readline().split()
        nsamples, T, is_cx, L, _ver = map(int, tokens[:5])
        cols = [line.split() for line in f if line.strip()]
    arr = np.asarray(cols, dtype=float)
    if arr.shape[0] != nsamples * T:
        raise ValueError(f"行数 {arr.shape[0]} != nsamples·T="
                         f"{nsamples * T}")
    body = arr[:, 1:]
    data = body.reshape(nsamples, T, -1)
    return data, {'nsamples': nsamples, 'T': T,
                  'is_complex': bool(is_cx), 'L': L, 'version': _ver}


# ═══════════════════════════════════════════════════════════════════
# 预计算顶点积二进制 reader（整合 huangcl/98_tools input_output.py）
# ═══════════════════════════════════════════════════════════════════

def readin_vdv_all(vdv_dir: str, nev: int, nev1: int, Nt: int,
                   conf_id, Px: int = 0, Py: int = 0, Pz: int = 0):
    """读取 V†V 预计算顶点积二进制（照抄 input_output.readin_VdV_all）。

    文件 ``<dir>/VdaggerV.Px%dPy%dPz%d.conf%s``：f8 交错 [re,im]，
    (Nt, Nev, Nev, 2) → complex，截断到前 Nev1 模。
    """
    with open("%s/VdaggerV.Px%dPy%dPz%d.conf%s"
              % (vdv_dir, Px, Py, Pz, conf_id), "rb") as f:
        vdv = np.fromfile(f, dtype="f8")
    vdv = vdv.reshape(Nt, nev, nev, 2)
    vdv = vdv[..., 0] + vdv[..., 1] * 1j
    return np.array(vdv[:, 0:nev1, 0:nev1])


def readin_vvv_all(vvv_dir: str, nev1: int, Nt: int, conf_id,
                   Px: int = 0, Py: int = 0, Pz: int = 0):
    """读取逐时间片 VVV 三夸克顶点积二进制（照抄 readin_VVV_all）。

    文件 ``<dir>/VVV.t%03i.Px%iPy%iPz%i.conf%s``；每片的 Nev 由
    文件大小自探测（cbrt(size/2)），截断到 Nev1。
    """
    vvv = np.zeros((Nt, nev1, nev1, nev1), dtype=complex)
    for t in range(Nt):
        with open("%s/VVV.t%03i.Px%iPy%iPz%i.conf%s"
                  % (vvv_dir, t, Px, Py, Pz, conf_id), "rb") as f:
            temp = np.fromfile(f, dtype="f8")
        nev = int(round(np.cbrt(temp.size / 2)))
        temp = temp.reshape(nev, nev, nev, 2)
        temp = temp[..., 0] + temp[..., 1] * 1j
        vvv[t] = temp[0:nev1, 0:nev1, 0:nev1]
    return vvv


def readin_vvv(vvv_dir: str, nev: int, nev1: int, Nt: int, conf_id,
               Px: int = 0, Py: int = 0, Pz: int = 0):
    """读取整块 VVV 二进制（照抄 readin_VVV）。

    文件 ``<dir>/VVV.Px%iPy%iPz%i.conf%s``：(Nt, Nev, Nev, Nev, 2)
    → complex，截断到 Nev1。
    """
    with open("%s/VVV.Px%iPy%iPz%i.conf%s"
              % (vvv_dir, Px, Py, Pz, conf_id), "rb") as f:
        vvv = np.fromfile(f, dtype="f8")
    vvv = vvv.reshape(Nt, nev, nev, nev, 2)
    vvv = vvv[..., 0] + vvv[..., 1] * 1j
    return vvv[:, 0:nev1, 0:nev1, 0:nev1]
