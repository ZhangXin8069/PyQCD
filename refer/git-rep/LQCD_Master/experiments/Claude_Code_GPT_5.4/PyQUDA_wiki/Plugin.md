# Plugins for PyQUDA

PyQUDA can build Cython wrapper files from a C header file using the `pyquda_plugins` package. Currently supported plugins:

- **[pycontract](https://github.com/CLQCD/contract)** – Baryon contraction routines on GPU.
- **[pygwu](https://github.com/CLQCD/gwu-qcd)** – GWU overlap fermion inverter.

## Current `pycontract` wrapper API

Besides the generated low-level binding, the repository now ships a high-level Python wrapper at `pyquda_plugins.pycontract`. Its public API is the one used by [`tests/test_pycontract.py`](https://github.com/CLQCD/PyQUDA/blob/master/tests/test_pycontract.py):

```python
from pyquda_utils import core, gamma, io
from pyquda_plugins import pycontract

core.init(resource_path=".cache/quda")
pycontract.init()

propag = io.readQIOPropagator("pt_prop_0")
propag.toDevice()

gamma_2 = gamma.Gamma(2)
gamma_4 = gamma.Gamma(8)
gamma_5 = gamma.Gamma(15)
C = gamma_2 @ gamma_4
CG_A = C @ gamma_4 @ gamma_5
CG_B = C @ gamma_5
Pp = (gamma.Gamma(0) + gamma_4) / 2

meson = pycontract.mesonTwoPoint(propag, propag, CG_A, CG_B)
all_sink = pycontract.mesonAllSinkTwoPoint(propag, propag, CG_B)
baryon = pycontract.baryonTwoPoint(
    propag,
    propag,
    propag,
    pycontract.BaryonContractType.IK_JL_NM,
    CG_A,
    CG_B,
    Pp,
)
```

High-level functions currently exposed by `pyquda_plugins.pycontract`:
- `init()`
- `mesonTwoPoint()`
- `mesonAllSinkTwoPoint()`
- `mesonAllSourceTwoPoint()`
- `baryonDiquark()`
- `baryonTwoPoint()`
- `baryonGeneralTwoPoint()`
- `baryonSequentialTwoPoint()`
- `baryonTwoPoint_v2()`

The wrapper also exports `BaryonContractType` and `BaryonSequentialType`. For baryon contractions, `gamma_mn` may be either `pyquda_utils.gamma.Gamma` or `pyquda_utils.gamma.Projector`.

## Example

We use the [contract](https://github.com/CLQCD/contract) plugin as an example.

Consider that we want to expose some functions declared in [`contract.h`](https://github.com/CLQCD/contract/blob/master/include/contract.h), and the definitions are compiled into a shared library `libcontract.so`.

First, we need to compile the shared library.
```bash
git clone https://github.com/CLQCD/contract
cd contract
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(pwd)/install -DGPU_ARCH=60
cmake --build . -j8 && cmake --install .
cd ../..
```

And we want to expose functions declared in `contract.h`:
```cpp
#pragma once

#include "contract_define.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  IK_JL_MN,
  IK_JN_ML,
  IL_JK_MN,
  IL_JN_MK,
  IN_JK_ML,
  IN_JL_MK,
} BaryonContractType;

void init(int device);
void baryon_two_point(void *correl, void *propag_i, void *propag_j, void *propag_m, BaryonContractType contract_type,
                      unsigned long volume, int gamma_ij, int gamma_kl, int gamma_mn);
void proton(void *correl, void *propag_i, void *propag_j, void *propag_m, int contract_type, unsigned long volume,
            int gamma_ij, int gamma_kl, int gamma_mn);

#ifdef __cplusplus
}
#endif
```

After installing PyQUDA-Utils by `pip install pyquda-utils`, you can use the following command to build the plugin to wrap `libcontract.so`:
```bash
cd contract/build
python -m pyquda_plugins -i contract.h -l contract -I $(pwd)/install/include -L $(pwd)/install/lib
```
`pyquda_plugins` will build Cython source files `contract.pxd`, `_pycontract.pyx` along with the stub file `_pycontract.pyi` and then compile them into `pyquda_plugins.pycontract._pycontract` module. Arguments of C basic type will be transferred from corresponding Python objects as input, and pointers will be converted from `numpy.ndarray`, `cupy.ndarray` or `torch.Tensor`.

`contract.pxd`:
```python
cdef extern from "contract.h":
    ctypedef enum BaryonContractType:
        pass

    void init(int device)
    void baryon_two_point(void *correl, void *propag_i, void *propag_j, void *propag_m, BaryonContractType contract_type, unsigned long volume, int gamma_ij, int gamma_kl, int gamma_mn)
    void proton(void *correl, void *propag_i, void *propag_j, void *propag_m, int contract_type, unsigned long volume, int gamma_ij, int gamma_kl, int gamma_mn)
```

`_pycontract.pyx`:
```python
from enum import IntEnum
from libcpp cimport bool
from numpy cimport ndarray
from pyquda_comm.pointer cimport Pointer, _NDArray
cimport contract

class BaryonContractType(IntEnum):
    IK_JL_MN = 0
    IK_JN_ML = 1
    IL_JK_MN = 2
    IL_JN_MK = 3
    IN_JK_ML = 4
    IN_JL_MK = 5


def init(int device):
    contract.init(device)

def baryon_two_point(correl, propag_i, propag_j, propag_m, contract.BaryonContractType contract_type, unsigned long volume, int gamma_ij, int gamma_kl, int gamma_mn):
    _correl = _NDArray(correl, 1)
    _propag_i = _NDArray(propag_i, 1)
    _propag_j = _NDArray(propag_j, 1)
    _propag_m = _NDArray(propag_m, 1)
    contract.baryon_two_point(<void *>_correl.ptr, <void *>_propag_i.ptr, <void *>_propag_j.ptr, <void *>_propag_m.ptr, contract_type, volume, gamma_ij, gamma_kl, gamma_mn)

def proton(correl, propag_i, propag_j, propag_m, int contract_type, unsigned long volume, int gamma_ij, int gamma_kl, int gamma_mn):
    _correl = _NDArray(correl, 1)
    _propag_i = _NDArray(propag_i, 1)
    _propag_j = _NDArray(propag_j, 1)
    _propag_m = _NDArray(propag_m, 1)
    contract.proton(<void *>_correl.ptr, <void *>_propag_i.ptr, <void *>_propag_j.ptr, <void *>_propag_m.ptr, contract_type, volume, gamma_ij, gamma_kl, gamma_mn)
```

`_pycontract.pyi`:
```python
from enum import IntEnum
from numpy.typing import NDArray

class BaryonContractType(IntEnum):
    IK_JL_MN = 0
    IK_JN_ML = 1
    IL_JK_MN = 2
    IL_JN_MK = 3
    IN_JK_ML = 4
    IN_JL_MK = 5

def init(device: int) -> None: ...
def baryon_two_point(correl: NDArray, propag_i: NDArray, propag_j: NDArray, propag_m: NDArray, contract_type: BaryonContractType, volume: int, gamma_ij: int, gamma_kl: int, gamma_mn: int) -> None: ...
def proton(correl: NDArray, propag_i: NDArray, propag_j: NDArray, propag_m: NDArray, contract_type: int, volume: int, gamma_ij: int, gamma_kl: int, gamma_mn: int) -> None: ...
```

Then you can use functions from `libcontract.so` in Python. In practice, we recommend adding a thin pure-Python wrapper for a more stable user-facing API. The current `pycontract` wrapper in this repository is such a layer.

## Limitations

### ABI

Because we compile the header in C mode, ensure that all the declarations are wrapped with
```cpp
#ifdef __cplusplus
extern "C" {
#endif

// function declarations

#ifdef __cplusplus
}
#endif
```

### Argument type

Because of Cython's [automatic type conversions](https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#automatic-type-conversions), we have some limitations on the argument type:

| C type                                                           | Python type                                         |
| ---------------------------------------------------------------- | --------------------------------------------------- |
| `[unsigned] int`, `[unsigned] long`, `[unsigned] long long`      | `int`                                               |
| `float`, `double`                                                | `float`                                             |
| `float _Complex`, `double _Complex`                              | `complex`                                           |
| `int *`, `int **`, `int ***`                                     | `numpy.ndarray[int32]`[^1]                          |
| `double *`, `double **`, `double ***`                            | `numpy.ndarray[float64]`[^1]                        |
| `double _Complex *`, `double _Complex **`, `double _Complex ***` | `numpy.ndarray[complex128]`[^1]                     |
| `const char *`, `const char []`                                  | `bytes`                                             |
| `void *`, `void **`, `void ***`                                  | `numpy.ndarray`, `cupy.ndarray`, `torch.Tensor`[^2] |

[^1]: Typed pointers can be converted only from `numpy.ndarray` with corresponding `ndim`
[^2]: Untyped pointers can be converted from `numpy.ndarray`, `cupy.ndarray` or `torch.Tensor` with corresponding `ndim`

- `struct` is unacceptable as the arguments for now. We are working on it.
- `union` cannot be the argument type

### Return type

Only basic arithmetic type are allowed besides void pointer. Returned void pointers are usually instance in C/C++.

| C type                                                      | Python type                       |
| ----------------------------------------------------------- | --------------------------------- |
| `[unsigned] int`, `[unsigned] long`, `[unsigned] long long` | `int`                             |
| `float`, `double`                                           | `float`                           |
| `float _Complex`, `double _Complex`                         | `complex`                         |
| `void *`                                                    | `pyquda_comm.pointer.Pointer`[^3] |

[^3]: `Pointer` handles the pointer returned from C/C++, remember to free it in C/C++

### Include

We use [pycparser](https://github.com/eliben/pycparser) to parse the header, however it uses `fake_libc_include` to parse standard library headers, which cannot be easily installed by `pip install pycparser`. So you cannot include any standard library header such as `stdlib.h` in the header.
