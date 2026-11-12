#ifndef PYQCD_EINSUM_CUDA_TOPOLOGY_H
#define PYQCD_EINSUM_CUDA_TOPOLOGY_H

#include "einsum_define.h"

#include <stdint.h>

/* CUDA 多操作数拓扑执行入口，与 KUPL 侧 `qcd_kupl_topology_apply` 同构：
 * 参数顺序、`QCD_DTYPE_*` 枚举、返回码语义逐项相同，CUDA 侧只在末尾多一个
 * stream 尾巴（与 `qcd_cuda_pattern_3pt` 的约定一致）。
 *
 * ---------------------------------------------------------------------------
 * 两种字节流（`plan` / `plan_len`）
 * ---------------------------------------------------------------------------
 * 1. **version-2 执行计划**（格式不变，仍由 `pyqcd/einsum/_native_plan.py` 生产、
 *    `runtime.cpp` 解析）：逐步 permute/pack 到设备缓冲 + cuBLAS batched GEMM，
 *    中间结果全部落在一块按 dtype 复用的设备 arena 里。
 * 2. **拓扑描述符**（`pyqcd/einsum/_kernel_cuda.py` 生产，魔数 `0x51435444`）：
 *    只带**结构**——dtype、每个操作数的 rank/标签/形状、输出标签，以及拓扑签名的
 *    blake2b-128 摘要前 8 字节；**不带收缩顺序**。原生侧用摘要查
 *    `src/topology_steps.h` 的**预编译步表**（2582 条真实语料拓扑，去重后 1052 条
 *    不同步序），按 einsum 语义展开成 pack + cuBLAS 步序列。因此运行期既没有
 *    path 搜索，也没有 Python 逐步调度——顺序是编译期常量，执行器是参数化的。
 *
 *    描述符布局（全部 int32 字，Little-endian）：
 *      [0] 魔数 0x51435444  [1] 版本 1        [2] total_bytes（= plan_len）
 *      [3] dtype_id        [4] num_inputs     [5] num_labels
 *      [6] num_output      [7] 保留 0
 *      [8..11] 签名摘要 16 字节（按字节原样搬运）
 *      每个操作数：rank, labels[rank], shape[rank]
 *      输出：labels[num_output]
 *    `num_labels` 是全式不同标签数，每个标签的 extent 由各操作数形状取最大（允许
 *    1↔N 广播）。原生会把"摘要 + 规范标签流的 FNV 指纹 + 元数"三者一起比对，任一
 *    不吻合即返回 4（fail closed，不猜）。
 *
 * 计划在首次调用时解析一次，之后的调用只比较字节，命中即复用步骤描述与 arena；
 * 换计划时 arena 留在池子里复用（容量够就不再 cudaMalloc，cuBLAS 句柄也不重建）。
 *
 * 与 `applyContractQcd` 的关系：两者共用同一份 decode/resolve/prepare 代码与同一套
 * 设备 kernel，因此对同一计划的数值结果与可行性判定逐位一致；区别只在入参形态
 * （本入口直接收操作数指针数组、明文 dtype 与可选字节偏移）以及本入口不套用
 * sandwich / E4 融合 kernel。`applyContractQcd` 的语义不受本入口影响。
 *
 * ---------------------------------------------------------------------------
 * 入口
 * ---------------------------------------------------------------------------
 *   int qcd_cuda_topology_apply(const void* plan, long long plan_len,
 *                               void* const* operands, int n_operands,
 *                               const long long* operand_offsets,
 *                               void* out, int dtype_id, void* stream);
 *
 * ---------------------------------------------------------------------------
 * 参数
 * ---------------------------------------------------------------------------
 *   plan             version-2 执行计划字节流或拓扑描述符字节流；按 memcpy 取
 *                    字段，不做对齐假设。
 *   plan_len         上述字节流字节数；必须与头部 total_bytes 逐位相等。
 *   operands         长度 n_operands 的**设备**基地址数组，顺序与计划/描述符里
 *                    的操作数逐项对应。
 *   n_operands       操作数个数；必须等于头部的 num_inputs。
 *   operand_offsets  可选（可为 NULL）：每个操作数的**字节**偏移，调用时按
 *                    `operands[i] + operand_offsets[i]` 解释，用于免拷贝地指向
 *                    大张量里的一片（例如 peram 的某个时间片）。NULL 表示全 0。
 *   out              设备输出缓冲区：按计划/描述符的输出标签顺序 C 行主序连续存储。
 *   dtype_id         本次调用的 dtype（见 einsum_define.h），**必须与字节流里的
 *                    dtype 相同**；不做任何隐式转换，不一致即返回 1。
 *   stream           目前只支持默认流：必须传 nullptr；非 nullptr 返回 1
 *                    （不假装支持外部流）。
 *
 * 传输口径：入口零拷贝（操作数已是设备指针），出口一次 D2D 写出，逐步之间
 * 没有 host↔device 传输。
 *
 * ---------------------------------------------------------------------------
 * 返回值（fail closed：任何非 0 返回都不写 out）
 * ---------------------------------------------------------------------------
 *   0  成功
 *   1  参数非法：指针为空、plan_len 与头部不符、n_operands 与 num_inputs 不符、
 *      dtype_id 越界或与计划不一致、地址加偏移溢出、某操作数为空、stream 非空
 *   2  内存不足
 *   3  其它内部错误（含异常兜底；本库缺少所需 kernel 实现时也归此类）
 *   4  QCD_TOPOLOGY_UNSUPPORTED：字节流本身超出本入口的可执行范围
 *      （单步 arity > 2、单个操作数内重复下标、没有步骤、尾随字节、描述符标签
 *      越界、摘要/指纹查不到步表、步表展开后中间体超过 2^31 元素等）
 *
 * 1 与 4 必须分开：1 是"这次的调用参数不对"，4 是"这条计划本入口不认"。
 * Python 侧对两者都回落现有路径，但诊断文案不同。与 KUPL 侧取值一致。
 *
 * ---------------------------------------------------------------------------
 * dtype（见 einsum_define.h）
 * ---------------------------------------------------------------------------
 *   0 = QCD_DTYPE_COMPLEX128 : 支持
 *   1 = QCD_DTYPE_COMPLEX64  : 支持
 *   2 = QCD_DTYPE_FLOAT64    : 支持
 *   3 = QCD_DTYPE_FLOAT32    : 支持
 * 每个 dtype 都由设备 kernel 在编译期实例化一条数值路径，运行期不做类型
 * 提升或降级。
 */

enum {
    /* 本入口专有的补充状态码：计划结构超出 version-2 的单步 arity ≤ 2 模型。
     * 不写进冻结的 einsum_define.h，避免改动与 KUPL 侧逐字节镜像的 ABI。 */
    QCD_TOPOLOGY_UNSUPPORTED = 4
};

#if defined(_WIN32)
#define QCD_CUDA_TOPOLOGY_API __declspec(dllexport)
#else
#define QCD_CUDA_TOPOLOGY_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

QCD_CUDA_TOPOLOGY_API int qcd_cuda_topology_apply(
    const void* plan,
    long long plan_len,
    void* const* operands,
    int n_operands,
    const long long* operand_offsets,
    void* out,
    int dtype_id,
    void* stream);

#ifdef __cplusplus
}
#endif

#endif /* PYQCD_EINSUM_CUDA_TOPOLOGY_H */
