#ifndef PYQCD_KUPL_EINSUM_FAMILY_APPLY_H
#define PYQCD_KUPL_EINSUM_FAMILY_APPLY_H

#include "einsum_define.h"

#include <stdint.h>

/*
 * KUPL(CPU)「可套用族模板 + 运行期步表」入口。
 *
 * 语料 pyqcd/einsum/_wick_tasklist.py 的 8638 条收缩式在严格拓扑签名下几乎两两
 * 互异，逐条生成内核既不可能也不必要：按
 *
 *     (算子元数, 各操作数 rank 的多重集, 输出 rank)
 *
 * 分组后只有 25 族（docs/wick_kernel_families_20261111.md）。同族内所有式子的
 * **收缩结构相同**，差异只在"哪个指标连到哪个槽位"——后者是数据，不是代码。
 *
 * 因此本入口把两者分开：
 *
 *   * 编译期常量（模板参数）：算子元数 ARITY、输出 rank OUT_RANK、各操作数 rank
 *     的多重集 RANKS...。它们决定槽位表、步表、轴表的定长尺寸，也决定本入口
 *     **是否接受**这份计划（元数/输出阶/rank 多重集必须逐项相等）。
 *   * 运行期数据（步表）：每步用哪两个槽、保留哪些指标、各轴长度，以及已经算好
 *     的源步长/紧凑步长/解包步长。执行期不做符号解析、不推导收缩顺序。
 *
 * 每步的数值工作固定三段：
 *
 *     pack（permute/copy 成 [batch,M,K] / [batch,K,N]）
 *       → cblas_{s,d}gemm（RowMajor，两侧 NoTrans，逐 batch）
 *       → unpack（[batch,M,N] 按目的轴序 gather）
 *
 * 中间体与 pack 缓冲都来自进程级块池（策略同 src/apply_contract.cpp 的
 * BufferPool：最佳适配、只回收 ≥512 KiB 且总量 ≤1 GiB 的块）；一次调用只在开头
 * 分配，步内不再 new/malloc，消费完的中间体立刻归还。
 *
 * ---------------------------------------------------------------------------
 * 入口
 * ---------------------------------------------------------------------------
 *   int qcd_kupl_family_apply(const void* plan, long long plan_len,
 *                             void* const* operands, int n_operands,
 *                             void* out, int dtype_id);
 *
 *   plan      族步表字节流（格式见下），按值 memcpy 读取，不假设对齐。
 *   plan_len  字节数，必须与表头 total_bytes 逐位相等。
 *   operands  长度 n_operands 的操作数基地址数组，顺序与表里 operand 形状表逐项
 *             对应；要求 C 行主序连续。
 *   n_operands 操作数个数，必须等于表头 num_operands。
 *   out       输出缓冲区，C 行主序连续，元素个数 = 输出形状乘积；调用方分配。
 *   dtype_id  QCD_DTYPE_FLOAT32(3) 走 sgemm，QCD_DTYPE_FLOAT64(2) 走 dgemm，
 *             必须与表头 dtype_id 一致。**不做任何隐式转换**：complex 与其它
 *             取值一律返回 4（QCD_FAMILY_UNSUPPORTED），不会降级成单精度。
 *
 * ---------------------------------------------------------------------------
 * 步表格式 v1（小端；所有整数按本机字节序 memcpy，长度单位是字节）
 * ---------------------------------------------------------------------------
 * 表头 10 × int32（40 字节）::
 *
 *   [0] magic        0x31414651  ("QFA1")
 *   [1] version      1
 *   [2] total_bytes  整条字节流长度（含表头）
 *   [3] num_operands 操作数个数（2..16）
 *   [4] num_slots    中间槽位数（= num_steps - 1）
 *   [5] num_steps    步数（= num_operands - 1，全部为二元步）
 *   [6] out_rank     输出阶数（0 <= out_rank <= 8）
 *   [7] dtype_id     QCD_DTYPE_FLOAT32 / QCD_DTYPE_FLOAT64
 *   [8] reserved0    必须为 0
 *   [9] reserved1    必须为 0
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
 * 语义（A 记左操作数、B 记右操作数、C 记本步结果）：
 *
 *   规范序 A = [batch..., M..., K...]，B = [batch..., K..., N...]，
 *   C = [batch..., M..., N...]；
 *
 *   * a_axes[p] 是规范序第 p 轴在 A 源张量里的物理轴号；
 *   * b_axes[p] 是规范序第 p 轴在 B 源张量里的物理轴号；
 *   * c_axes[p] 是**结果物理轴 p** 在规范序里的位置（unpack 的目标轴序）；
 *   * batch / m / n / k 是各组的元素数（各组轴长乘积）；
 *   * 三张轴表都必须是 [0, rank) 的排列；
 *   * 结果张量的物理轴长必须等于 canonical[c_axes[p]]。
 *
 * ---------------------------------------------------------------------------
 * 返回值（fail closed：任何非 0 返回都不写 out）
 * ---------------------------------------------------------------------------
 *   0  成功
 *   1  参数非法：指针为空、长度/表头不符、槽位或轴号越界、轴表不是排列、
 *      形状或 batch/M/N/K 乘积不一致、n_operands 与表头不符、dtype 与表头不符、
 *      操作数缺数据、读未写或写仍存活的槽位、步数与元数不自洽等
 *   2  内存不足
 *   3  其它内部错误（含异常兜底）
 *   4  QCD_FAMILY_UNSUPPORTED：这份计划不属于本库的 25 族
 *      （元数/输出阶/rank 多重集对不上），或 dtype 为 complex/未知
 *
 * ---------------------------------------------------------------------------
 * workspace 与线程
 * ---------------------------------------------------------------------------
 * 中间槽与 pack 缓冲由 src/family_templates.h 的进程级块池提供（与
 * src/apply_contract.cpp 的 BufferPool 同策略；后者在匿名命名空间里，无法跨 TU
 * 复用）。入口无可变全局状态（块池与 CBLAS 句柄分别由互斥量与 std::call_once
 * 保护），可多线程并发调用。GEMM 线程数由 BLAS 运行时决定。
 */

enum {
    /* 本入口专有的补充状态码：计划不属于 25 族，或 dtype 无本入口的数值路径。
     * 不写进冻结的 einsum_define.h，避免改动与 CUDA 侧逐字节镜像的 ABI。 */
    QCD_FAMILY_UNSUPPORTED = 4
};

/* 步表格式常量，供 Python 生成端与测试共用。 */
enum {
    QCD_FAMILY_MAGIC = 0x31414651,
    QCD_FAMILY_VERSION = 1,
    QCD_FAMILY_HEADER_BYTES = 40,
    QCD_FAMILY_MAX_ARITY = 16,
    QCD_FAMILY_MIN_ARITY = 2,
    /* 语料实测上界：操作数 rank ≤ 4、中间体 rank ≤ 9、输出 rank ≤ 6
     * （8638 条全量扫过）。这里留出余量给尚未出现的式子；超出即解析期拒绝。 */
    QCD_FAMILY_MAX_RANK = 12,
    QCD_FAMILY_MAX_STEPS = 15,
    QCD_FAMILY_MAX_SLOTS = 14
};

#ifdef __cplusplus
extern "C" {
#endif

QCD_EINSUM_API int qcd_kupl_family_apply(const void* plan, long long plan_len,
                                         void* const* operands, int n_operands,
                                         void* out, int dtype_id);

#ifdef __cplusplus
}
#endif

#endif /* PYQCD_KUPL_EINSUM_FAMILY_APPLY_H */
