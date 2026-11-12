#ifndef PYQCD_EINSUM_CUDA_DEDICATED_H
#define PYQCD_EINSUM_CUDA_DEDICATED_H

#include "einsum_define.h"

#include <stdint.h>

/* CUDA 「固定步表 + 参数化执行器」入口，与 KUPL 侧 `qcd_kupl_dedicated_apply`
 * **同构**：吃同一份 v1 步表字节流（格式由
 * `cpp/kupl/einsum/include/dedicated_apply.h` 固定，Python 生成端是
 * `pyqcd/einsum/_kernel_kupl.py::_dedicated_plan`），逐步只做
 *
 *     挑两个槽 → pack（设备端 gather）→ cublas{S,D}gemm → unpack
 *
 * 不解析符号、不搜索收缩顺序。与 KUPL 入口的差异只有两处，且都是约定差异：
 *
 *   1. `operands` / `out` 是**设备**指针（KUPL 是主机指针）；
 *   2. 末尾多一个 `stream` 尾巴（与 `qcd_cuda_topology_apply` 的约定一致）；
 *      目前只支持默认流，必须传 nullptr。
 *
 * ---------------------------------------------------------------------------
 * 入口
 * ---------------------------------------------------------------------------
 *   int qcd_cuda_dedicated_apply(const void* steps, long long steps_len,
 *                                void* const* operands, int n_operands,
 *                                void* out, int dtype_id, void* stream);
 *
 *   steps      固定步表字节流（v1，格式逐字节同 KUPL），按值 memcpy 读取。
 *   steps_len  字节数，必须与表头 total_bytes 逐位相等。
 *   operands   长度 n_operands 的**设备**基地址数组，顺序与表里形状表逐项对应；
 *              设备端不做 stride 解释，要求 C 行主序连续。
 *   n_operands 操作数个数，必须等于表头 num_operands。
 *   out        设备输出缓冲区，元素个数 = 输出形状乘积；调用方分配。
 *   dtype_id   四种 IEEE 精度之一，必须与表头一致：float32(3) → cublasSgemm，
 *              float64(2) → cublasDgemm，complex64(1) → cublasCgemm，
 *              complex128(0) → cublasZgemm（batch > 1 时用同名 *StridedBatched）。
 *              复数按 numpy 的**交错实虚**布局直传，不做任何隐式转换或降精度。
 *              运行期符号不齐备时返回 4（见 `qcd_cuda_dedicated_dtypes`）。
 *   stream     只支持默认流（nullptr）；非 nullptr 返回 1。
 *
 * 数值口径：除 GEMM 外，步内还有三条短路径（与 KUPL `gemm_block` 逐条对齐）——
 * k==1 是纯缩放（按存储精度乘一次）；m==1 或 n==1 是广播归约，**用双精度累加器**
 * （float→double、complex64→complex128）再舍入回存储精度。升档的理由与 KUPL 相同：
 * 这两个形状下 cuBLAS 走 gemv，按存储精度累加，长链 k 上明显劣于分块 GEMM 的
 * 内部累加口径。
 *
 * ---------------------------------------------------------------------------
 * 步表格式 v1（小端；整数按本机字节序 memcpy，长度单位是字节）
 * ---------------------------------------------------------------------------
 * 表头 8 × int32（32 字节）:
 *   [0] magic 0x31414451  [1] version 1      [2] total_bytes
 *   [3] num_operands      [4] num_slots      [5] num_steps
 *   [6] out_rank          [7] dtype_id
 * 形状表：num_operands 段 [int32 rank][int64 extent × rank]，num_slots 段，输出 1 段。
 * 步记录：int32 a_slot, b_slot, dst_slot, rank_a, rank_b, rank_c,
 *         batch_rank, m_rank, k_rank；int64 batch, m, n, k；
 *         然后 a_axes[rank_a], b_axes[rank_b], c_axes[rank_c]。
 * 语义（A=[batch,M,K]、B=[batch,K,N]、C=[batch,M,N] 规范序）：
 *   a_axes/b_axes 是规范序轴在源张量里的物理轴号；c_axes[p] 是结果物理轴 p 在
 *   规范序里的位置；dst_slot = -1 只允许出现在最后一步（写 out）。
 *
 * ---------------------------------------------------------------------------
 * 返回值（fail closed：任何非 0 返回都不保证 out 完整）
 * ---------------------------------------------------------------------------
 *   0  成功
 *   1  参数非法（指针/长度/槽位/轴表/形状乘积/dtype 不符、stream 非空等）
 *   2  显存不足
 *   3  其它内部错误（含异常兜底；缺 `qcd_cuda_permute` 实现时也归此类）
 *   4  QCD_CUDA_DEDICATED_UNSUPPORTED：表本身或本机能力不足以执行（dtype 的
 *      cuBLAS 符号不齐备、阶数/步数/槽数越界、同一中间槽被写两次）。上层据此
 *      回落通用路径（fail open），不需要把它当成错误。
 *
 * ---------------------------------------------------------------------------
 * workspace
 * ---------------------------------------------------------------------------
 * 中间槽与 3 块 pack 缓冲来自本 TU 内的进程级设备块池（最佳适配、只在块大小落在
 * [bytes, 2*bytes] 时复用），**一次调用只在开头分配一次**（单块 arena，256 字节
 * 对齐切分），步内不再 cudaMalloc。cuBLAS 句柄按当前设备缓存并复用。调用由内部
 * 互斥量串行化（同一进程多线程安全，但不并发）。
 *
 * 诊断：`PYQCD_EINSUM_DEDICATED_TRACE=1` 时向 stderr 打印每步 pack/GEMM/unpack
 * 耗时与 batch,M,N,K（含同步点，仅诊断用）。
 */

enum {
    QCD_CUDA_DEDICATED_MAGIC = 0x31414451,
    QCD_CUDA_DEDICATED_VERSION = 1,
    QCD_CUDA_DEDICATED_HEADER_BYTES = 32,
    QCD_CUDA_DEDICATED_MAX_RANK = 32,
    QCD_CUDA_DEDICATED_MAX_SLOTS = 256,
    QCD_CUDA_DEDICATED_MAX_OPERANDS = 4096,
    QCD_CUDA_DEDICATED_MAX_STEPS = 4096,
    /* 表本身超出本入口能力。与 KUPL 侧同名语义的取值保持一致。 */
    QCD_CUDA_DEDICATED_UNSUPPORTED = 4
};

#if defined(_WIN32)
#define QCD_CUDA_DEDICATED_API __declspec(dllexport)
#else
#define QCD_CUDA_DEDICATED_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

QCD_CUDA_DEDICATED_API int qcd_cuda_dedicated_apply(
    const void* steps,
    long long steps_len,
    void* const* operands,
    int n_operands,
    void* out,
    int dtype_id,
    void* stream);

/* 运行期 dtype 能力查询：返回 bit(QCD_DTYPE_*) 掩码，置位表示该 dtype 在本机
 * cuBLAS 上**符号齐备、可执行**（base 与 StridedBatched 两种变体都在）。
 *
 * Python 侧用它裁剪 `dedicated_dtypes()`：缺符号的 dtype 不进白名单，
 * `try_dedicated` 对它返回 None（fail open，回落通用路径），不会把「本机不支持」
 * 误报成「数值错误」。查询只读符号表、不建 context、不碰设备，无卡机器返回 0。 */
QCD_CUDA_DEDICATED_API int qcd_cuda_dedicated_dtypes(void);

#ifdef __cplusplus
}
#endif

#endif /* PYQCD_EINSUM_CUDA_DEDICATED_H */
