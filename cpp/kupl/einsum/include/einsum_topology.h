#ifndef PYQCD_KUPL_EINSUM_TOPOLOGY_H
#define PYQCD_KUPL_EINSUM_TOPOLOGY_H

#include "einsum_define.h"

#include <stdint.h>

/*
 * KUPL(CPU) 多操作数拓扑执行入口。
 *
 * 一次调用跑完整条 version-2 执行计划：计划字节流仍由
 * pyqcd/einsum/_native_plan.py 生产（格式不变），本入口在 C++ 内部按步执行，
 * 逐步做打包(permute/copy) + CBLAS 调用，中间结果走进程内 BufferPool 复用，
 * 因此不再需要调用方在 Python 侧逐步调度每一步。
 *
 * 与 applyContractQcd 的关系：两者执行同一个解析器与同一套数值内核
 * （src/runtime.cpp + src/apply_contract.cpp）。区别只在**入参形态**：
 * applyContractQcd 走冻结的 (plan, tensors[], out[], params[]) 四参数字形式，
 * 本入口直接收操作数指针数组、明文 dtype 与可选字节偏移，便于上层用一次
 * ctypes 调用把整条收缩发下来。数值语义与 applyContractQcd 逐位一致。
 *
 * ---------------------------------------------------------------------------
 * 入口
 * ---------------------------------------------------------------------------
 *   int qcd_kupl_topology_apply(const void* plan, long long plan_len,
 *                               void* const* operands, int n_operands,
 *                               const long long* operand_offsets,
 *                               void* out, int dtype_id);
 *
 * ---------------------------------------------------------------------------
 * 参数
 * ---------------------------------------------------------------------------
 *   plan             version-2 执行计划字节流（不改格式），必须 8 字节对齐读取，
 *                    实际按 memcpy 取字段，不做对齐假设。
 *   plan_len         上述字节流的字节数；必须与计划头 total_bytes 逐位相等。
 *   operands         长度 n_operands 的操作数基地址数组，顺序与计划里的
 *                    input_shapes 逐项对应。
 *   n_operands       操作数个数；必须等于计划头的 num_inputs。
 *   operand_offsets  可选（可为 NULL）：每个操作数的**字节**偏移，调用时按
 *                    ``operands[i] + operand_offsets[i]`` 解释。用于免拷贝地
 *                    指向大张量里的一片（例如 peram 的某个时间片）。
 *                    NULL 表示所有偏移为 0。
 *   out              输出缓冲区：按计划输出下标顺序的 C 行主序连续存储，
 *                    元素个数等于计划输出的元素总数。调用方负责分配。
 *   dtype_id         本次调用的 dtype（见 einsum_define.h），**必须与计划头
 *                    的 dtype 相同**；不做任何隐式转换，不一致即返回 1。
 *
 * ---------------------------------------------------------------------------
 * 返回值（fail closed：任何非 0 返回都不写 out）
 * ---------------------------------------------------------------------------
 *   0  成功
 *   1  参数非法：指针为空、plan_len 与头部不符、n_operands 与 num_inputs 不符、
 *      dtype_id 越界或与计划不一致、地址加偏移溢出、某操作数缺数据等
 *   2  内存不足
 *   3  其它内部错误（含异常兜底）
 *   4  QCD_TOPOLOGY_UNSUPPORTED：计划本身超出本入口的可执行范围
 *      （单步 arity > 2、单个操作数内重复下标、没有步骤、尾随字节等）
 *
 * 1 与 4 必须分开：1 是"这次的调用参数不对"，4 是"这条计划本入口不认"。
 * Python 侧对两者都回落现有路径，但诊断文案不同。
 *
 * ---------------------------------------------------------------------------
 * dtype（见 einsum_define.h）
 * ---------------------------------------------------------------------------
 *   0 = QCD_DTYPE_COMPLEX128 : 支持（double 交错 re/im）
 *   1 = QCD_DTYPE_COMPLEX64  : 支持（float 交错 re/im）
 *   2 = QCD_DTYPE_FLOAT64    : 支持
 *   3 = QCD_DTYPE_FLOAT32    : 支持
 * 每个 dtype 都只在编译期实例化一条数值路径，运行期不做类型提升或降级。
 *
 * ---------------------------------------------------------------------------
 * workspace 与线程
 * ---------------------------------------------------------------------------
 * 中间缓冲来自 src/apply_contract.cpp 的进程级 BufferPool：同一次调用内被
 * 释放的中间块会被后续步骤直接复用，跨调用也保留（只涨不缩），因此逐步收缩
 * 不会"每步重新分配"大块内存。
 *
 * 入口本身无可变全局状态，可多线程并发调用；BufferPool 与 CBLAS 句柄分别由
 * 互斥量与 std::call_once 保护。OpenMP 只按调用方给定的线程数使用。
 */

enum {
    /* 本入口专有的补充状态码：计划结构超出 version-2 的单步 arity ≤ 2 模型。
     * 不写进冻结的 einsum_define.h，避免改动与 CUDA 侧逐字节镜像的 ABI。 */
    QCD_TOPOLOGY_UNSUPPORTED = 4
};

#ifdef __cplusplus
extern "C" {
#endif

QCD_EINSUM_API int qcd_kupl_topology_apply(const void* plan, long long plan_len,
                                           void* const* operands, int n_operands,
                                           const long long* operand_offsets, void* out,
                                           int dtype_id);

#ifdef __cplusplus
}

namespace pyqcd_einsum {

struct ExecutionPlan;

/* 定义在 src/apply_contract.cpp：按步骤执行一条已解析的 version-2 计划。
 * 本入口复用它，从而与 applyContractQcd 共享打包/BLAS 内核与 workspace 池，
 * 不另起一套数值实现。 */
int execute_parsed_contract(const ExecutionPlan& execution, long long tensors_address,
                            long long out_address);

}  // namespace pyqcd_einsum
#endif

#endif /* PYQCD_KUPL_EINSUM_TOPOLOGY_H */
