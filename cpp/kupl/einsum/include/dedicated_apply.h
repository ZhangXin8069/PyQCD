#ifndef PYQCD_KUPL_EINSUM_DEDICATED_APPLY_H
#define PYQCD_KUPL_EINSUM_DEDICATED_APPLY_H

#include "einsum_define.h"

#include <stdint.h>

/*
 * KUPL(CPU) 「固定步表 + 参数化执行器」入口。
 *
 * 与 qcd_kupl_topology_apply 的区别：拓扑入口吃 version-2 计划字节流（由
 * pyqcd/einsum/_native_plan.py 生产，含符号表与逐步 label），运行期仍要做
 * label → 轴序的还原；本入口吃的是**已经展开好的 GEMM 步表**——每一步的
 * 槽位、轴序、batch/M/N/K 都已固化，原生侧只做
 *
 *     挑两个槽 → pack（permute/copy）→ cblas_{s,d}gemm → unpack
 *
 * 不再解析符号、不再推导收缩顺序。Python 生成端见
 * pyqcd/einsum/_kernel_kupl.py::_dedicated_plan。
 *
 * ---------------------------------------------------------------------------
 * 入口
 * ---------------------------------------------------------------------------
 *   int qcd_kupl_dedicated_apply(const void* steps, long long steps_len,
 *                                void* const* operands, int n_operands,
 *                                void* out, int dtype_id);
 *
 *   steps      固定步表字节流（格式见下），按值 memcpy 读取，不假设对齐。
 *   steps_len  字节数，必须与表头 total_bytes 逐位相等。
 *   operands   长度 n_operands 的操作数基地址数组，顺序与表里 operand 形状表
 *              逐项对应；要求 C 行主序连续。
 *   n_operands 操作数个数，必须等于表头 num_operands。
 *   out        输出缓冲区，C 行主序连续，元素个数 = 输出形状乘积；调用方分配。
 *   dtype_id   QCD_DTYPE_FLOAT32(3) 或 QCD_DTYPE_FLOAT64(2)，必须与表头
 *              dtype_id 一致。**不做任何隐式转换**；complex 一律返回 4。
 *
 * ---------------------------------------------------------------------------
 * 步表格式 v1（小端；所有整数按本机字节序 memcpy，长度单位是字节）
 * ---------------------------------------------------------------------------
 * 表头 8 × int32（32 字节）::
 *
 *   [0] magic        0x31414451  ("QDA1")
 *   [1] version      1
 *   [2] total_bytes  整条字节流长度（含表头）
 *   [3] num_operands 操作数个数（>= 1）
 *   [4] num_slots    中间槽位数（>= 0）
 *   [5] num_steps    步数（>= 1）
 *   [6] out_rank     输出阶数（0 <= out_rank <= 32）
 *   [7] dtype_id     QCD_DTYPE_FLOAT32 / QCD_DTYPE_FLOAT64
 *
 * 形状表（按顺序紧跟表头）::
 *
 *   num_operands 段：[int32 rank][int64 extent × rank]
 *   num_slots    段：[int32 rank][int64 extent × rank]
 *   输出        1 段：[int32 rank == out_rank][int64 extent × rank]
 *
 * 槽位编号：0 .. num_operands-1 是操作数槽（data 由 operands[] 给出），
 * num_operands .. num_operands+num_slots-1 是中间槽（由本执行器分配）。
 *
 * 步记录（共 num_steps 条，顺序执行）::
 *
 *   int32 a_slot, b_slot, dst_slot
 *   int32 rank_a, rank_b, rank_c
 *   int32 batch_rank, m_rank, k_rank
 *   int64 batch, m, n, k
 *   int32 a_axes[rank_a]
 *   int32 b_axes[rank_b]
 *   int32 c_axes[rank_c]
 *
 *   dst_slot = -1 表示写最终 out，否则写中间槽；只允许最后一步用 -1。
 *
 * 语义（A 记左操作数、B 记右操作数，C 记本步结果）：
 *
 *   规范序 A = [batch..., M..., K...]，B = [batch..., K..., N...]，
 *   C = [batch..., M..., N...]；
 *
 *   * a_axes[p] 是规范序第 p 轴在 A 源张量里的物理轴号；
 *   * b_axes[p] 是规范序第 p 轴在 B 源张量里的物理轴号；
 *   * c_axes[p] 是**结果物理轴 p** 在规范序里的位置（unpack 的目标轴序）；
 *   * batch / m / n / k 是各组的元素数（各组轴长乘积），与两个源张量的
 *     对应轴长逐项交叉校验；
 *   * 三张轴表都必须是 [0, rank) 的排列；
 *   * 结果张量的物理轴长必须等于 canonical[c_axes[p]]。
 *
 * 执行时每一步：
 *
 *   1. pack：按 a_axes / b_axes 把两侧 gather 成 [batch][M][K] / [batch][K][N]
 *      的连续布局。两种**不需要拷贝**的布局直接引用原缓冲：
 *        * 恒等轴序：源就是 [batch][M][K]（A）或 [batch][K][N]（B）；
 *        * 组交换轴序：源是 [batch][K][M]（A）或 [batch][N][K]（B），
 *          这时把 cblas 的 transA/transB 置 1、lda=m（A）/ ldb=k（B）即可，
 *          GEMM 仍按 [batch][M][K]×[batch][K][N] 语义计算。
 *      其余轴序才真正拷贝；拷贝的循环次序由内层 run 的**源步长**决定（选源里
 *      步长最小的轴作内层，保证内层读是连续的），再逐轴向外合并成尽量长的
 *      memcpy 段；
 *   2. GEMM：逐 batch 调 cblas_sgemm / cblas_dgemm（RowMajor，alpha=1，beta=0；
 *      transA/transB 由上面的布局决定）；M/N/K 任一为 1 时该步是纯缩放/广播
 *      归约，直接按定义求和，不进入 BLAS（这两个形状只在两侧都无需转置时启用）；
 *   3. unpack：c_axes 非恒等时把 [batch][M][N] 结果 gather 成结果的物理轴序
 *      （c_axes 恒等时 GEMM 直接写进目的缓冲，省掉解包）。
 *
 * ---------------------------------------------------------------------------
 * 诊断
 * ---------------------------------------------------------------------------
 *   设 PYQCD_EINSUM_DEDICATED_TRACE=1 时向 stderr 打印解析/分配耗时、每个中间
 *   槽的字节数，以及每一步的 pack / GEMM / unpack 耗时、要搬运的元素数、
 *   transA/transB 与 batch,M,N,K，最后一行汇总整次调用的 pack / GEMM / unpack
 *   总量与槽位字节数。默认关闭，关闭时不进入任何计时分支。
 *
 * ---------------------------------------------------------------------------
 * 返回值（fail closed：任何非 0 返回都不写 out）
 * ---------------------------------------------------------------------------
 *   0  成功
 *   1  参数非法：指针为空、长度/表头不符、槽位或轴号越界、轴表不是排列、
 *      形状或 batch/M/N/K 乘积不一致、n_operands 与表头不符、dtype 与表头不符、
 *      操作数缺数据、读未写或写仍存活的槽位等
 *   2  内存不足
 *   3  其它内部错误（含异常兜底）
 *   4  QCD_DEDICATED_UNSUPPORTED：表本身超出本入口能力（dtype 为 complex、
 *      阶数/步数/槽数越界等）
 *
 * ---------------------------------------------------------------------------
 * workspace 与线程
 * ---------------------------------------------------------------------------
 * 中间槽与 pack 缓冲都来自本 TU 内的进程级块池（acquire 最佳适配 / release 只
 * 回收大块，策略与 src/apply_contract.cpp 的 BufferPool 相同）。apply_contract.cpp
 * 的 BufferPool 位于匿名命名空间、无法跨 TU 复用，因此这里按同一口径自带最小
 * 实现；**一次调用只在开头分配一次**（每个中间槽一块 + 3 块 pack 缓冲），
 * 步内不再 new/malloc。
 *
 * 入口无可变全局状态（块池与 CBLAS 句柄分别由互斥量与 std::call_once 保护），
 * 可多线程并发调用。GEMM 线程数由 BLAS 运行时决定。
 */

enum {
    /* 本入口专有的补充状态码：步表本身超出可执行范围。不写进冻结的
     * einsum_define.h，避免改动与 CUDA 侧逐字节镜像的 ABI。 */
    QCD_DEDICATED_UNSUPPORTED = 4
};

/* 步表格式常量，供 Python 生成端与测试共用。 */
enum {
    QCD_DEDICATED_MAGIC = 0x31414451,
    QCD_DEDICATED_VERSION = 1,
    QCD_DEDICATED_HEADER_BYTES = 32,
    QCD_DEDICATED_MAX_RANK = 32,
    QCD_DEDICATED_MAX_SLOTS = 256,
    QCD_DEDICATED_MAX_OPERANDS = 4096,
    QCD_DEDICATED_MAX_STEPS = 4096
};

#ifdef __cplusplus
extern "C" {
#endif

QCD_EINSUM_API int qcd_kupl_dedicated_apply(const void* steps, long long steps_len,
                                            void* const* operands, int n_operands,
                                            void* out, int dtype_id);

#ifdef __cplusplus
}
#endif

#endif /* PYQCD_KUPL_EINSUM_DEDICATED_APPLY_H */
