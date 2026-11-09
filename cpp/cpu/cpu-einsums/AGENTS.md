# cpp/cpu/cpu-einsums — Einsums 字符串 `einsum` 的 CPU C ABI 桥

本目录把 Einsums 的**运行时字符串 spec** `einsum`（`einsums::compute_graph::einsum`，
与 Python 侧 `einsums` 用的是同一个引擎）包装成一个冻结的四入口 C ABI，
产物 `libpyqceinsums.so`（soname 同名），供 Python（ctypes / Cython）直接加载。
ABI 形状是 `cpp/cuda/matx` 的**逐项镜像**，唯一区别：这里全部是 **CPU 主机指针**，
不是设备指针。

本机**没有** Einsums 头文件/库，所以本目录的第一级目标是「接口层正确 + 无 Einsums
也能构建 + 加载后能证明 ABI」：没有 Einsums 时照常产出 `libpyqceinsums.so`，
`pyqceinsums_einsum()` 一律返回 `UNAVAILABLE(4)` 并给出可操作原因。
**真机 Einsums 路径从未编译过**（见 §6）。

## 1. 文件与归属

| 文件 | 职责 |
|---|---|
| `include/cpu_einsums.h` | 冻结 C ABI v1：状态码/dtype/槽位枚举 + 四入口声明 |
| `src/cpu_einsums.cpp` | 唯一实现：fail-closed 校验 + 桩/真机两条路径的派发 |
| `python/pyqceinsums.h` | Cython/ctypes 侧的平坦 C 声明（与 `cpp/cuda/matx/python/pyqcmatx.h` 同形） |
| `make.sh` | `g++` 直编 + 头文件探测 + 版本脚本 + 原子软链发布 |
| `build/` | 产物、`build.log`、生成的 `cpu_einsums_exports.map`（不提交，`*.so` 与无扩展名文件被根 `.gitignore` 忽略） |
| `libpyqceinsums.so` | 指向 `build/libpyqceinsums.so` 的稳定软链（不提交） |

本目录**不碰** `pyqcd/**`、`refer/**`、`cpp/cuda/**`、`cpp/kupl/**`，也不改根
`.gitignore`/`.gitmodules`/`AGENTS.md`。

## 2. ABI 槽位表（v1）

```c
int  pyqceinsums_abi_version(void);                          // == 1
int  pyqceinsums_backend_enabled(void);                      // 真机=1，桩=0
int  pyqceinsums_einsum(const char* spec,
                        long long set_ptrs, long long params);
const char* pyqceinsums_last_error(void);                    // 线程局部
```

`set_ptrs` 是 `int64_t[PYQCEINSUMS_SET_SIZE=8]`：

| 下标 | 名 | 含义 |
|---|---|---|
| 0 | `PYQCEINSUMS_SET_OUTPUT` | 输出张量的**主机**指针 |
| 1 | `PYQCEINSUMS_SET_INPUTS` | 指向 `int64_t[nops]` 的操作数主机指针表 |
| 2 | `PYQCEINSUMS_SET_INPUT_SHAPES` | 指向 `int64_t[nops*rank]`，行主序，每个操作数 `rank` 个维度 |
| 3 | `PYQCEINSUMS_SET_OUTPUT_SHAPE` | 指向 `int64_t[out_ndim]` |

`params` 是 `int32_t[PYQCEINSUMS_PARAM_SIZE=8]`：

| 下标 | 名 | 取值 |
|---|---|---|
| 0 | `PYQCEINSUMS_PARAM_NUM_OPS` | 1..2 |
| 1 | `PYQCEINSUMS_PARAM_INPUT_RANK` | 0..4（**每个**操作数共用；二元即同 rank） |
| 2 | `PYQCEINSUMS_PARAM_DTYPE_ID` | 0=complex128, 1=complex64, 2=float64, 3=float32 |
| 3 | `PYQCEINSUMS_PARAM_OUT_NDIM` | 0..4 |

状态码：`OK=0`、`INVALID=1`、`MEMORY=2`、`INTERNAL=3`、`UNAVAILABLE=4`。

`spec` 原样交给 Einsums，**不做格式翻译**：Einsums 写法是
`"out <- in1 ; in2"`（输出在前、分号分隔输入），等价于 numpy 的 `"in1,in2->out"`。
例如 `"ij <- ik ; kj"` == `ik,kj->ij`。ABI **不暴露** `alpha/beta` 或任何缩放参数，
完全交给 Einsums `einsum(spec, ...)` 的默认语义（unverified，见 §6 第 5 条）。

## 3. 校验顺序与 fail-closed

`pyqceinsums_einsum()` 先清空线程局部诊断，再按**固定顺序**校验；任一条不通过
立刻返回 `INVALID(1)`，绝不进入 Einsums（真机路径也绝不进入）：

1. `spec` 非空（`NULL` 或 `""` 都算空）；
2. `set_ptrs`/`params` 非零；
3. `nops == 2`（`PYQCEINSUMS_MAX_OPERANDS`；Einsums 的 `cg::einsum` 没有一元重载）；
4. `0 <= rank <= 4`（`PYQCEINSUMS_MAX_RANK`）；
5. `0 <= out_ndim <= 4`；
6. `dtype_id` ∈ 0..3；
7. 输出指针非空；
8. 输入指针表 / 输入形状表 / 输出形状表非空；
9. 每个输入指针非空；
10. 所有输入 shape 与输出 shape 非负。

二元要求两输入同 rank —— 结构上由单一 `INPUT_RANK` 槽保证，没有「第二 rank」
可传，所以超出 1/2 元或 rank>4 的表达式只能在上面 3/4 步被拒绝。

## 4. 构建

```bash
bash cpp/cpu/cpu-einsums/make.sh                     # 产物 cpp/cpu/cpu-einsums/libpyqceinsums.so
bash cpp/cpu/cpu-einsums/make.sh --clean             # 删除 build/ 与公开软链
bash cpp/cpu/cpu-einsums/make.sh --require-einsums   # 找不到 Einsums 头文件时硬失败（CI 用）
```

选项只有这三个（`-h/--help` 打印用法，未知选项退出码 2）。
编译器 `${CXX:-g++}`，基础 flags
`-O3 -fopenmp -shared -fPIC -std=c++20 -fvisibility=hidden -Wall -Wextra`，
`PYQCD_EINSUMS_NATIVE_ARCH=1`（默认）时再追加 `-march=native -funroll-loops`。

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `PYQCD_EINSUMS_ROOT` / `EINSUMS_ROOT` | 空 | Einsums checkout 或安装前缀（探测 `include/`） |
| `PYQCD_EINSUMS_FLAGS` | 空（用上面的基础 flags） | 整体覆盖编译 flags |
| `PYQCD_EINSUMS_LIBS` | **空** | 额外链接 flags，如 `-L$PYQCD_EINSUMS_ROOT/lib -leinsums` |
| `PYQCD_EINSUMS_BUILD_DIR` | `<本目录>/build` | 构建目录 |
| `PYQCD_EINSUMS_NATIVE_ARCH` | `1` | `1` 时加 `-march=native -funroll-loops` |
| `CXX` | `g++` | 编译器 |

头文件探测（`make.sh:71-102`）：在 `$PYQCD_EINSUMS_ROOT` 下找
`include/Einsums/*.hpp` 或 `include/einsums/*.hpp`，**任一命中**即认为有头文件，
并加 `-DPYQCEINSUMS_HAVE_EINSUMS=1`；只有小写目录命中时额外用
`-DPYQCEINSUMS_{TENSOR,COMPUTE_GRAPH}_HEADER` 把小写 include 拼写传进去。
找不到时：stderr 明确警告（含 `PYQCD_EINSUMS_ROOT` 提示）、构建桩库、
`--require-einsums` 则**直接非零退出**且不产出/不发布任何东西。

导出面三重收敛：`-fvisibility=hidden` + `PYQCEINSUMS_API`
（`__attribute__((visibility("default")))`）+ `-Wl,--version-script`
（生成的 `build/cpu_einsums_exports.map`，只 `global:` 四个入口，其余 `local: *`），
`-Wl,-soname,libpyqceinsums.so`。编译先写 `build/libpyqceinsums.so.tmp.$$`，
`mv -Tf` 原子落位；公开路径用临时软链 + `mv -Tf` 原子替换；失败时 `cat
build/build.log` 到 stderr 并非零退出；构建后用
`nm -D --defined-only` 校验 `pyqceinsums_einsum` 已导出，缺失即失败。

真机链接注意：`PYQCD_EINSUMS_LIBS` 默认空，此时 `.so` 会**保留未定义的 Einsums
符号**（`-shared` 默认允许），`ctypes.CDLL` 加载才会报 `undefined symbol`。脚本
已对此打印警告；跨机器部署请显式给出 `-L...-leinsums`（库名 unverified，见 §6）。

## 5. 桩行为（无 Einsums 时）

- `pyqceinsums_abi_version()` → `1`；`pyqceinsums_backend_enabled()` → `0`。
- `pyqceinsums_einsum()`：**先完整跑完 §3 的 10 条校验**，合法入参一律返回
  `UNAVAILABLE(4)`，非法入参照旧返回 `INVALID(1)`。
- `pyqceinsums_last_error()` 给出可操作原因串（`src/cpu_einsums.cpp:67` `kNoEinsums`），
  其中包含 `PYQCD_EINSUMS_ROOT`、`EINSUMS_BUILD_PYTHON`、`include/Einsums/*.hpp`
  与 `bash cpp/cpu/cpu-einsums/make.sh`，明确说明「重建时指路」而不是「算错了」。
- 桩绝不写输出缓冲区，也绝不返回 `OK` —— 无法计算时不会伪造结果。

## 6. 已实现 / 未实现边界（如实登记）

已实现（本机可验证）：

- 四入口 C ABI、soname、版本脚本收敛、原子发布、`nm` 校验（§8 实测）。
- 完整 fail-closed 校验（10 条，含顺序）与 `INVALID(1)` 诊断。
- 桩路径的 `UNAVAILABLE(4)` + 可操作原因串；`last_error()` 线程局部且每次调用先清空。
- 真机路径的**代码结构**：按 `dtype` × `rank 0..4` × `out_ndim 0..4` × 1/2 操作数
  全展开（4 dtype × 5 × 5 × 2 = 200 条实例化），`std::copy` 拷入 → `cg::einsum` →
  `std::copy` 拷回，异常 `catch(...)` 归一为 `INTERNAL(3)`，`std::bad_alloc` 归一为
  `MEMORY(2)`。

**未实现 / 未验证（不得声称可用）**：

- **真机 Einsums 路径仍未编译/链接过**：`refer/git-rep/Einsums` 已 checkout（§6.1），
  但它的头文件依赖 CMake configure 生成的 `Einsums/Config/Defines.hpp`，并需要
  Apiary（libclang/LLVM）、BLAS/LAPACK、HDF5、OpenMP；本机不具备这些条件。
  因此下述 API 均来自**源码核验**，不是编译或执行证据。

### 6.1 已由实际源码确认（`refer/git-rep/Einsums` @ `628971cd886cd5bb9ecb71d87289522d6625910d`）

1. **头文件路径**：`<Einsums/Tensor/Tensor.hpp>`（`create_tensor`、`Tensor<T,R>`、
   `data()`）与 `<Einsums/ComputeGraph/Operations.hpp>`（`cg::einsum`）。源码位置：
   `libs/Einsums/Tensor/include/...`、`libs/Einsums/ComputeGraph/include/...`。
   仍可用 `-DPYQCEINSUMS_TENSOR_HEADER` / `-DPYQCEINSUMS_COMPUTE_GRAPH_HEADER` 覆盖。
2. **`create_tensor` 签名**：`template <typename Type = double, typename... Args>
   auto create_tensor(std::string const &name, Args... args)`，约束
   `(std::is_convertible_v<Args, std::size_t> && ...)`，返回 `Tensor<Type, sizeof...(Args)>`。
   桥内把维度显式转 `std::size_t`；不传维度即 `Tensor<T,0>`（Tensor.hpp:2721）。
3. **`Tensor::data()`**：`using Pointer = T*;`，`[[nodiscard]] Pointer data()`
   （Tensor.hpp:851/861）——与 `std::copy` 用法一致。
4. **`cg::einsum` 签名**：`void einsum(EinsumFormatString spec, CType *C,
   AType const &A, BType const &B)`（Operations.hpp:5711），另有 α/β 重载
   （:5612）。`EinsumFormatString` 有**非 explicit** 的 `std::string_view`
   构造函数（EinsumSpec.hpp:389），故 `const char*` 直接可用；运行时字符串在
   `parse_einsum_spec()` 阶段校验。
5. **没有一元 `cg::einsum`**：源码只有二元重载与 `sizeof...(Operands) >= 3`
   的 n 元重载（NaryEinsum.hpp:208/273）；文档亦说明转置用 `cg::permute`、
   标量用 `cg::dot`。本桥因此**只接受恰好 2 个操作数**，一元写法 fail-closed。
6. **初始化**：不存在顶层 `einsums::initialize`；唯一同名符号是
   `einsums::comm::initialize(int*, char***)`（MPI 层）。各模块通过
   `register_startup_function` 注册启动钩子（`libs/Einsums/*/src/InitModule.cpp`），
   桥内**不再调用任何初始化函数**。
7. **库名/链接**：顶层 target 为 `Einsums`，`OUTPUT_NAME` 为
   `Einsums-abi${EINSUMS_ABI_VERSION}`（`libs/CMakeLists.txt`），即
   `libEinsums-abi<N>.so`；模块库为 `libEinsums_<Module>.so`。没有 `libeinsums.so`。
   链接用 `PYQCD_EINSUMS_LIBS`（如 `-L<build>/lib -lEinsums-abi<N>` 或模块库列表）。
8. **头文件探测**：`make.sh` 现接受三种布局——安装前缀 `include/Einsums/*.hpp`、
   源码 checkout `libs/Einsums/*/include`、已 configure 的 `build/libs/Einsums/*/include`。

### 6.2 仍未验证

- 真机数值行为、`cg::einsum` 对秩 0 输出（`Tensor<T,0>*`）的接受度、spec 空格容忍度。
- 元素数溢出检查：真机路径用 `std::size_t` 连乘，未做溢出保护（校验阶段只保证
  shape 非负）。
- 未调用 `einsums::comm::finalize()`：进程退出由 Einsums/OpenMP 自行收尾。
- 桥的 `PYQCEINSUMS_MAX_OPERANDS=2` 是 ABI 常量；Einsums 的 n 元（≥3）路径不在本桥范围。

## 7. 与 `cpp/cuda/matx` 的对称关系

| 维度 | `cpp/cuda/matx` | 本目录 |
|---|---|---|
| 产物 | `libpyqcmatx_cuda.so` | `libpyqceinsums.so` |
| 入口 | `pyqcmatx_abi_version` / `_cutensor_enabled` / `_einsum` / `_last_error` | `pyqceinsums_abi_version` / `_backend_enabled` / `_einsum` / `_last_error` |
| 指针 | 设备指针 | **主机指针** |
| 槽位/参数表 | `int64_t[8]` / `int32_t[8]`，同下标同含义 | 完全一致 |
| 校验顺序 | 同 10 条 | 完全一致（错误串措辞不同） |
| 上界 | 1..2 操作数、rank ≤4、二元同 rank | 恰好 2 操作数、rank ≤4、二元同 rank |
| 后端开关 | `PYQCMATX_HAVE_MATX` + `-DMATX_EN_CUTENSOR=ON` | `PYQCEINSUMS_HAVE_EINSUMS=1` |
| 桩行为 | `UNAVAILABLE(4)` + 原因串 | `UNAVAILABLE(4)` + 原因串 |
| 构建脚本 | `make-nv.sh`（nvcc） | `make.sh`（g++，风格对齐 `cpp/kupl/einsum/make.sh`） |
| spec 格式 | MatX/cuTENSOR 的 `"ik,kj->ij"` | Einsums 的 `"ij <- ik ; kj"` |

差异仅两处：地址空间（CPU vs 设备）与 spec 语法；ABI 其余部分逐项同构，Python 侧
可以用同一套 marshalling 代码换库名。

## 8. 验收（实测 2026-11-10，主机 x86_64 / WSL2）

```bash
bash cpp/cpu/cpu-einsums/make.sh                       # 退出码 0
nm -D --defined-only cpp/cpu/cpu-einsums/libpyqceinsums.so | awk '{print $NF}' | sort
```

```
pyqceinsums_abi_version
pyqceinsums_backend_enabled
pyqceinsums_einsum
pyqceinsums_last_error
```

恰好 4 个符号。`readelf -d` 给出 `SONAME libpyqceinsums.so`，`NEEDED`
为 `libstdc++.so.6 / libm.so.6 / libgomp.so.1 / libgcc_s.so.1 / libc.so.6 /
ld-linux-x86-64.so.2`（桩不依赖任何 Einsums/BLAS 库）。

ctypes 冒烟（脚本放在仓库外的 scratchpad，未入库）：

```python
import ctypes
lib = ctypes.CDLL("cpp/cpu/cpu-einsums/libpyqceinsums.so")
assert lib.pyqceinsums_abi_version() == 1
assert lib.pyqceinsums_backend_enabled() == 0          # 桩
# 合法 2 元 ij <- ik ; kj，dtype=float64，rank=2，out_ndim=2
assert lib.pyqceinsums_einsum(b"ij <- ik ; kj", set_ptrs, params) == 4
assert b"PYQCD_EINSUMS_ROOT" in lib.pyqceinsums_last_error()
# 输出指针为 0
assert code_with_null_output == 1
```

实跑结果：23 条断言全 PASS（含 empty/null spec、`nops=0/3`、`rank=5`、
`out_ndim=5`、`dtype=4`、空输入指针、负 shape 全部 `INVALID(1)`）。

清理与重建：

```bash
bash cpp/cpu/cpu-einsums/make.sh --clean   # 退出码 0，删除 build/ 与软链
bash cpp/cpu/cpu-einsums/make.sh           # 退出码 0，重新产出可用产物
```

`--require-einsums`（无 `PYQCD_EINSUMS_ROOT` 时）实测退出码 **1**，并打印
`error: --require-einsums forbids an Einsums-less archive`；未知选项退出码 2。

## 9. 变更须知

- ABI 是**冻结**的：加槽位/加参数要同时改
  `include/cpu_einsums.h`、`src/cpu_einsums.cpp`、`python/pyqceinsums.h` 与
  `make.sh` 里的版本脚本四条 symbol 列表，并升 `PYQCEINSUMS_ABI_VERSION`。
- 桩的 `UNAVAILABLE(4)` 是契约，不是「暂时没接」：任何让桩返回 `OK` 的改动都是
  伪造成功，禁止。
- 新增源文件放 `src/*.cpp`，`make.sh` 自动收编；新增导出符号必须同时进版本脚本，
  否则会被 `local: *` 吃掉。
