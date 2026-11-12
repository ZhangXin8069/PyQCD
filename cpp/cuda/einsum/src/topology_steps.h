/* 由 pyqcd.einsum._topology_registry 生成（生成脚本见提交日志），请勿手工编辑。
 *
 * 每条真实语料拓扑一条紧凑步表：收缩顺序（活表位置对）为编译期常量，
 * 由 apply_contract.cu 的参数化执行器展开成 pack + cuBLAS 步序列。
 * key = 拓扑签名 blake2b-128 摘要的前 8 字节（大端），按 key 升序排列；
 * fnv = 同一规范标签流的 64 位 FNV-1a 指纹，运行期由原生侧重算，
 * 用来把"表项属于这条结构"绑到实际下发的标签上（失配即 fail closed）。
 *
 * 语料 2582 条，去重后 1052 条不同步序，顺序取自 opt_einsum greedy（均匀 dim=4）。 */

#ifndef PYQCD_EINSUM_CUDA_TOPOLOGY_STEPS_H
#define PYQCD_EINSUM_CUDA_TOPOLOGY_STEPS_H

#include <cstdint>

struct qcd_topology_order {
    const int8_t* pairs;  /* 2*steps 个位置对 (left, right) */
    int32_t steps;
};

static const int8_t kQcdTopoOrder0[] = {0, 2, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder1[] = {0, 2, 2, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder2[] = {0, 3, 0, 2, 0, 1, 0, 4, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder3[] = {0, 3, 0, 2, 0, 1, 0, 5, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder4[] = {0, 3, 4, 7, 0, 2, 0, 1, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder5[] = {0, 3, 4, 7, 0, 2, 2, 5, 0, 1, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder6[] = {0, 4, 0, 3, 0, 2, 0, 1, 0, 5, 3, 6, 0, 3, 3, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder7[] = {0, 4, 0, 3, 0, 2, 0, 1, 0, 5, 3, 6, 0, 4, 3, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder8[] = {0, 4, 0, 3, 0, 2, 0, 1, 0, 6, 3, 6, 0, 4, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder9[] = {0, 4, 0, 3, 0, 2, 0, 1, 0, 6, 3, 6, 1, 4, 3, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder10[] = {0, 4, 0, 3, 0, 2, 0, 1, 0, 7, 3, 6, 0, 5, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder11[] = {0, 4, 0, 3, 0, 2, 0, 1, 1, 6, 4, 6, 0, 4, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder12[] = {0, 4, 0, 3, 0, 2, 6, 8, 2, 7, 3, 6, 0, 1, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder13[] = {0, 4, 0, 3, 0, 2, 7, 8, 3, 7, 3, 6, 0, 1, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder14[] = {0, 4, 0, 3, 8, 9, 4, 8, 4, 7, 0, 2, 0, 1, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder15[] = {0, 4, 6, 10, 0, 3, 0, 2, 0, 1, 0, 5, 3, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder16[] = {0, 4, 6, 10, 0, 3, 0, 2, 0, 1, 0, 6, 3, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder17[] = {0, 4, 6, 10, 0, 3, 4, 8, 0, 2, 0, 1, 4, 5, 0, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder18[] = {0, 4, 6, 10, 0, 3, 4, 8, 0, 2, 2, 6, 0, 1, 0, 4, 0, 1, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder19[] = {0, 4, 6, 10, 1, 4, 5, 8, 0, 2, 0, 1, 4, 5, 0, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder20[] = {0, 4, 6, 10, 2, 5, 6, 8, 0, 2, 0, 1, 4, 5, 0, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder21[] = {0, 5, 1, 4, 0, 3, 0, 2, 0, 1, 3, 7, 1, 4, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder22[] = {0, 5, 1, 4, 0, 3, 0, 2, 0, 1, 3, 7, 2, 5, 4, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder23[] = {0, 5, 1, 4, 0, 3, 0, 2, 0, 1, 3, 7, 2, 5, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder24[] = {0, 5, 1, 4, 0, 3, 0, 2, 0, 3, 5, 7, 0, 3, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder25[] = {0, 5, 1, 4, 0, 3, 0, 2, 0, 3, 5, 7, 0, 4, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder26[] = {0, 5, 1, 4, 0, 3, 0, 2, 0, 4, 6, 7, 0, 3, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder27[] = {0, 5, 1, 4, 0, 3, 0, 2, 0, 4, 6, 7, 0, 4, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder28[] = {0, 5, 1, 4, 0, 3, 0, 2, 6, 8, 2, 7, 3, 6, 0, 1, 1, 4, 0, 1, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder29[] = {0, 5, 1, 4, 0, 3, 0, 3, 5, 8, 0, 3, 4, 6, 0, 3, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder30[] = {0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 3, 7, 1, 4, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder31[] = {0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 3, 7, 1, 4, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder32[] = {0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 3, 7, 2, 5, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder33[] = {0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 3, 7, 2, 5, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder34[] = {0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 3, 7, 2, 5, 4, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder35[] = {0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 3, 7, 2, 5, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder36[] = {0, 5, 1, 4, 0, 3, 1, 2, 1, 5, 6, 7, 3, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder37[] = {0, 5, 1, 4, 0, 3, 1, 2, 1, 6, 4, 7, 2, 4, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder38[] = {0, 5, 1, 4, 0, 3, 1, 2, 1, 6, 4, 7, 3, 5, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder39[] = {0, 5, 1, 4, 0, 3, 1, 2, 1, 6, 4, 7, 3, 5, 4, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder40[] = {0, 5, 1, 4, 0, 3, 1, 2, 1, 8, 4, 7, 3, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder41[] = {0, 5, 1, 4, 0, 3, 1, 2, 5, 8, 1, 7, 3, 6, 0, 1, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder42[] = {0, 5, 1, 4, 0, 3, 1, 3, 5, 8, 1, 5, 2, 4, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder43[] = {0, 5, 1, 4, 0, 4, 0, 3, 5, 8, 0, 4, 4, 6, 0, 3, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder44[] = {0, 5, 1, 4, 0, 4, 1, 3, 5, 8, 1, 5, 3, 4, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder45[] = {0, 5, 1, 4, 1, 3, 0, 2, 0, 1, 3, 7, 1, 4, 4, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder46[] = {0, 5, 1, 4, 1, 3, 0, 2, 0, 1, 3, 7, 1, 4, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder47[] = {0, 5, 1, 4, 1, 3, 0, 2, 0, 1, 3, 7, 2, 5, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder48[] = {0, 5, 1, 4, 1, 3, 0, 2, 0, 3, 5, 7, 0, 3, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder49[] = {0, 5, 1, 4, 1, 3, 0, 2, 0, 3, 5, 7, 0, 4, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder50[] = {0, 5, 1, 4, 1, 3, 0, 2, 0, 4, 6, 7, 0, 3, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder51[] = {0, 5, 1, 4, 1, 3, 0, 2, 0, 4, 6, 7, 0, 4, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder52[] = {0, 5, 1, 4, 1, 3, 0, 3, 5, 8, 0, 3, 4, 6, 0, 3, 0, 3, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder53[] = {0, 5, 1, 4, 1, 3, 1, 2, 0, 1, 3, 7, 1, 4, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder54[] = {0, 5, 1, 4, 1, 3, 1, 2, 0, 1, 3, 7, 1, 4, 3, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder55[] = {0, 5, 1, 4, 1, 3, 1, 2, 0, 1, 3, 7, 2, 5, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder56[] = {0, 5, 1, 4, 1, 3, 1, 2, 0, 1, 3, 7, 2, 5, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder57[] = {0, 5, 1, 4, 1, 3, 1, 2, 0, 1, 3, 7, 2, 5, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder58[] = {0, 5, 1, 4, 1, 3, 1, 2, 1, 6, 4, 7, 2, 4, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder59[] = {0, 5, 1, 4, 1, 3, 1, 2, 1, 6, 4, 7, 3, 5, 4, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder60[] = {0, 5, 1, 4, 1, 3, 1, 2, 1, 8, 4, 7, 3, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder61[] = {0, 5, 1, 4, 1, 3, 1, 3, 5, 8, 1, 5, 2, 4, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder62[] = {0, 5, 1, 4, 1, 3, 8, 9, 4, 8, 4, 7, 0, 2, 0, 1, 1, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder63[] = {0, 5, 1, 4, 1, 3, 8, 9, 4, 8, 4, 7, 0, 6, 0, 1, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder64[] = {0, 5, 1, 4, 1, 3, 8, 9, 4, 8, 4, 7, 1, 2, 0, 1, 1, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder65[] = {0, 5, 1, 4, 1, 4, 0, 3, 5, 8, 0, 4, 4, 6, 0, 3, 0, 3, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder66[] = {0, 5, 1, 4, 1, 4, 1, 3, 5, 8, 1, 5, 3, 4, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder67[] = {0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 3, 7, 0, 5, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder68[] = {0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 3, 7, 1, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder69[] = {0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 3, 7, 1, 4, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder70[] = {0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 3, 7, 1, 6, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder71[] = {0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 3, 7, 2, 5, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder72[] = {0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 3, 7, 2, 5, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder73[] = {0, 5, 1, 4, 2, 3, 0, 2, 1, 5, 5, 7, 2, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder74[] = {0, 5, 1, 4, 2, 3, 0, 2, 1, 6, 4, 7, 2, 4, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder75[] = {0, 5, 1, 4, 2, 3, 0, 2, 1, 6, 4, 7, 2, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder76[] = {0, 5, 1, 4, 2, 3, 0, 2, 1, 6, 4, 7, 3, 5, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder77[] = {0, 5, 1, 4, 2, 3, 0, 2, 1, 7, 4, 7, 2, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder78[] = {0, 5, 1, 4, 2, 3, 0, 3, 5, 8, 1, 5, 2, 5, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder79[] = {0, 5, 1, 4, 2, 3, 0, 3, 5, 8, 1, 5, 3, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder80[] = {0, 5, 1, 4, 2, 3, 0, 3, 5, 8, 1, 6, 2, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder81[] = {0, 5, 1, 4, 2, 3, 0, 3, 5, 8, 1, 6, 3, 6, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder82[] = {0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 3, 7, 0, 5, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder83[] = {0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 3, 7, 1, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder84[] = {0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 3, 7, 1, 6, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder85[] = {0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 3, 7, 2, 5, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder86[] = {0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 3, 7, 2, 5, 3, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder87[] = {0, 5, 1, 4, 2, 3, 1, 2, 1, 6, 4, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder88[] = {0, 5, 1, 4, 2, 3, 1, 2, 1, 6, 4, 7, 3, 5, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder89[] = {0, 5, 1, 4, 2, 3, 1, 2, 1, 7, 4, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder90[] = {0, 5, 1, 4, 2, 3, 1, 2, 6, 8, 2, 7, 3, 6, 0, 1, 1, 4, 0, 1, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder91[] = {0, 5, 1, 4, 2, 3, 1, 3, 5, 8, 1, 5, 2, 5, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder92[] = {0, 5, 1, 4, 2, 3, 1, 3, 5, 8, 1, 5, 3, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder93[] = {0, 5, 1, 4, 2, 3, 1, 3, 5, 8, 1, 6, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder94[] = {0, 5, 1, 4, 2, 3, 1, 3, 5, 8, 1, 6, 3, 6, 4, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder95[] = {0, 5, 1, 4, 2, 3, 7, 9, 3, 8, 4, 7, 0, 2, 0, 1, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder96[] = {0, 5, 1, 4, 2, 3, 7, 9, 3, 8, 4, 7, 1, 2, 3, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder97[] = {0, 5, 7, 11, 1, 4, 0, 3, 0, 2, 0, 2, 4, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder98[] = {0, 5, 7, 11, 1, 4, 0, 3, 0, 2, 0, 3, 5, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder99[] = {0, 5, 7, 11, 1, 4, 0, 3, 0, 2, 0, 3, 5, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder100[] = {0, 5, 7, 11, 1, 4, 0, 3, 0, 2, 5, 7, 1, 6, 2, 5, 0, 1, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder101[] = {0, 5, 7, 11, 1, 4, 0, 3, 1, 2, 2, 6, 5, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder102[] = {0, 5, 7, 11, 1, 4, 0, 3, 1, 2, 3, 7, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder103[] = {0, 5, 7, 11, 1, 4, 0, 3, 1, 2, 3, 7, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder104[] = {0, 5, 7, 11, 1, 4, 1, 3, 0, 2, 0, 2, 4, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder105[] = {0, 5, 7, 11, 1, 4, 1, 3, 0, 2, 0, 2, 4, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder106[] = {0, 5, 7, 11, 1, 4, 1, 3, 0, 2, 0, 3, 5, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder107[] = {0, 5, 7, 11, 1, 4, 1, 3, 1, 2, 2, 6, 5, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder108[] = {0, 5, 7, 11, 1, 4, 1, 3, 1, 2, 3, 7, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder109[] = {0, 5, 7, 11, 1, 4, 1, 3, 7, 8, 3, 7, 3, 6, 0, 2, 0, 1, 2, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder110[] = {0, 5, 7, 11, 1, 4, 2, 3, 0, 2, 0, 1, 3, 6, 0, 5, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder111[] = {0, 5, 7, 11, 1, 4, 2, 3, 0, 2, 2, 6, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder112[] = {0, 5, 7, 11, 1, 4, 2, 3, 0, 2, 3, 7, 5, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder113[] = {0, 5, 7, 11, 1, 4, 2, 3, 1, 2, 0, 1, 1, 6, 4, 5, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder114[] = {0, 5, 7, 11, 1, 4, 2, 3, 1, 2, 3, 7, 5, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder115[] = {0, 5, 7, 11, 3, 5, 6, 9, 1, 3, 0, 2, 0, 1, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder116[] = {0, 5, 7, 11, 3, 5, 6, 9, 1, 3, 1, 2, 5, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder117[] = {0, 5, 7, 11, 3, 6, 7, 9, 1, 3, 0, 2, 0, 1, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder118[] = {0, 5, 7, 11, 3, 6, 7, 9, 1, 3, 1, 2, 5, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder119[] = {0, 6, 1, 5, 0, 4, 0, 3, 0, 2, 0, 1, 4, 9, 1, 5, 5, 7, 1, 4, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder120[] = {0, 6, 1, 5, 0, 4, 0, 3, 0, 2, 0, 1, 4, 9, 1, 5, 5, 7, 1, 4, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder121[] = {0, 6, 1, 5, 0, 4, 0, 3, 0, 2, 0, 1, 4, 9, 1, 5, 5, 7, 2, 5, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder122[] = {0, 6, 1, 5, 0, 4, 0, 3, 0, 2, 0, 4, 7, 9, 0, 4, 3, 7, 1, 3, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder123[] = {0, 6, 1, 5, 0, 4, 0, 3, 0, 2, 0, 4, 7, 9, 0, 5, 3, 7, 1, 3, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder124[] = {0, 6, 1, 5, 0, 4, 0, 3, 0, 2, 0, 5, 8, 9, 0, 4, 3, 7, 1, 3, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder125[] = {0, 6, 1, 5, 0, 4, 0, 3, 0, 2, 0, 5, 8, 9, 0, 5, 3, 7, 4, 6, 0, 5, 1, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder126[] = {0, 6, 1, 5, 0, 4, 0, 3, 0, 3, 6, 10, 0, 4, 6, 8, 0, 4, 1, 3, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder127[] = {0, 6, 1, 5, 0, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 1, 4, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder128[] = {0, 6, 1, 5, 0, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 1, 4, 3, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder129[] = {0, 6, 1, 5, 0, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder130[] = {0, 6, 1, 5, 0, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder131[] = {0, 6, 1, 5, 0, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder132[] = {0, 6, 1, 5, 0, 4, 0, 3, 1, 2, 0, 3, 6, 9, 0, 4, 5, 7, 2, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder133[] = {0, 6, 1, 5, 0, 4, 0, 3, 1, 2, 0, 3, 6, 9, 0, 5, 3, 7, 1, 3, 2, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder134[] = {0, 6, 1, 5, 0, 4, 0, 3, 1, 2, 0, 3, 6, 9, 0, 5, 3, 7, 2, 4, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder135[] = {0, 6, 1, 5, 0, 4, 0, 3, 1, 2, 0, 3, 6, 9, 0, 5, 3, 7, 2, 4, 4, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder136[] = {0, 6, 1, 5, 0, 4, 0, 3, 1, 2, 0, 3, 6, 9, 0, 7, 3, 7, 2, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder137[] = {0, 6, 1, 5, 0, 4, 0, 3, 1, 2, 6, 10, 1, 9, 4, 8, 0, 2, 3, 6, 0, 2, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder138[] = {0, 6, 1, 5, 0, 4, 0, 3, 1, 3, 6, 10, 0, 3, 5, 8, 0, 4, 1, 3, 2, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder139[] = {0, 6, 1, 5, 0, 4, 0, 3, 9, 11, 4, 10, 5, 9, 0, 2, 0, 1, 2, 6, 0, 2, 2, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder140[] = {0, 6, 1, 5, 0, 4, 0, 3, 9, 11, 4, 10, 5, 9, 1, 2, 0, 1, 2, 6, 0, 2, 3, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder141[] = {0, 6, 1, 5, 0, 4, 0, 3, 9, 11, 4, 10, 5, 9, 1, 2, 4, 7, 1, 6, 2, 5, 0, 1, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder142[] = {0, 6, 1, 5, 0, 4, 0, 4, 0, 3, 6, 10, 0, 5, 6, 8, 0, 4, 1, 3, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder143[] = {0, 6, 1, 5, 0, 4, 0, 4, 1, 3, 6, 10, 0, 3, 5, 8, 0, 4, 2, 3, 2, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder144[] = {0, 6, 1, 5, 0, 4, 1, 3, 0, 2, 0, 1, 4, 9, 1, 5, 4, 7, 1, 5, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder145[] = {0, 6, 1, 5, 0, 4, 1, 3, 0, 2, 0, 1, 4, 9, 1, 5, 4, 7, 2, 4, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder146[] = {0, 6, 1, 5, 0, 4, 1, 3, 0, 2, 0, 1, 4, 9, 1, 5, 4, 7, 2, 4, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder147[] = {0, 6, 1, 5, 0, 4, 1, 3, 0, 2, 0, 4, 7, 9, 0, 4, 3, 7, 1, 3, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder148[] = {0, 6, 1, 5, 0, 4, 1, 3, 0, 2, 0, 4, 7, 9, 0, 5, 3, 7, 4, 6, 0, 5, 1, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder149[] = {0, 6, 1, 5, 0, 4, 1, 3, 0, 2, 0, 5, 8, 9, 0, 4, 3, 7, 1, 3, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder150[] = {0, 6, 1, 5, 0, 4, 1, 3, 0, 2, 0, 5, 8, 9, 0, 5, 3, 7, 1, 3, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder151[] = {0, 6, 1, 5, 0, 4, 1, 3, 0, 2, 7, 10, 2, 9, 4, 8, 0, 1, 2, 6, 0, 2, 1, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder152[] = {0, 6, 1, 5, 0, 4, 1, 3, 0, 3, 6, 10, 0, 4, 6, 8, 0, 4, 4, 6, 0, 5, 1, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder153[] = {0, 6, 1, 5, 0, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 4, 7, 1, 4, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder154[] = {0, 6, 1, 5, 0, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 4, 7, 1, 4, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder155[] = {0, 6, 1, 5, 0, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 4, 7, 2, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder156[] = {0, 6, 1, 5, 0, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 4, 7, 2, 4, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder157[] = {0, 6, 1, 5, 0, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 4, 7, 2, 4, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder158[] = {0, 6, 1, 5, 0, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 4, 7, 2, 4, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder159[] = {0, 6, 1, 5, 0, 4, 1, 3, 1, 2, 1, 7, 5, 9, 2, 5, 4, 7, 2, 4, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder160[] = {0, 6, 1, 5, 0, 4, 1, 3, 1, 2, 1, 7, 5, 9, 2, 5, 4, 7, 3, 4, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder161[] = {0, 6, 1, 5, 0, 4, 1, 3, 1, 2, 3, 8, 7, 9, 1, 7, 4, 7, 3, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder162[] = {0, 6, 1, 5, 0, 4, 1, 3, 1, 2, 6, 10, 1, 9, 4, 8, 2, 5, 4, 6, 0, 1, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder163[] = {0, 6, 1, 5, 0, 4, 1, 3, 1, 3, 6, 10, 1, 6, 2, 5, 4, 7, 2, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder164[] = {0, 6, 1, 5, 0, 4, 1, 4, 0, 3, 6, 10, 0, 5, 6, 8, 0, 4, 4, 6, 0, 5, 1, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder165[] = {0, 6, 1, 5, 0, 4, 1, 4, 1, 3, 6, 10, 1, 6, 2, 5, 4, 7, 3, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder166[] = {0, 6, 1, 5, 0, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 6, 7, 0, 5, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder167[] = {0, 6, 1, 5, 0, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 6, 7, 1, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder168[] = {0, 6, 1, 5, 0, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 6, 7, 1, 6, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder169[] = {0, 6, 1, 5, 0, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder170[] = {0, 6, 1, 5, 0, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 3, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder171[] = {0, 6, 1, 5, 0, 4, 2, 3, 0, 2, 0, 3, 6, 9, 0, 4, 4, 7, 1, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder172[] = {0, 6, 1, 5, 0, 4, 2, 3, 0, 2, 0, 3, 6, 9, 0, 5, 3, 7, 1, 3, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder173[] = {0, 6, 1, 5, 0, 4, 2, 3, 0, 2, 0, 3, 6, 9, 0, 5, 3, 7, 1, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder174[] = {0, 6, 1, 5, 0, 4, 2, 3, 0, 2, 0, 3, 6, 9, 0, 5, 3, 7, 2, 4, 2, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder175[] = {0, 6, 1, 5, 0, 4, 2, 3, 0, 2, 0, 3, 6, 9, 0, 6, 3, 7, 1, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder176[] = {0, 6, 1, 5, 0, 4, 2, 3, 0, 2, 7, 10, 2, 9, 4, 8, 0, 1, 2, 6, 0, 2, 3, 4, 0, 1, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder177[] = {0, 6, 1, 5, 0, 4, 2, 3, 0, 3, 6, 10, 0, 3, 5, 8, 0, 4, 1, 4, 2, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder178[] = {0, 6, 1, 5, 0, 4, 2, 3, 0, 3, 6, 10, 0, 3, 5, 8, 0, 4, 2, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder179[] = {0, 6, 1, 5, 0, 4, 2, 3, 0, 3, 6, 10, 0, 3, 5, 8, 0, 5, 1, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder180[] = {0, 6, 1, 5, 0, 4, 2, 3, 0, 3, 6, 10, 0, 3, 5, 8, 0, 5, 2, 6, 3, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder181[] = {0, 6, 1, 5, 0, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 5, 7, 0, 5, 2, 5, 0, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder182[] = {0, 6, 1, 5, 0, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 5, 7, 1, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder183[] = {0, 6, 1, 5, 0, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 5, 7, 1, 4, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder184[] = {0, 6, 1, 5, 0, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 5, 7, 1, 5, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder185[] = {0, 6, 1, 5, 0, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 5, 7, 2, 6, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder186[] = {0, 6, 1, 5, 0, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 5, 7, 2, 6, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder187[] = {0, 6, 1, 5, 0, 4, 2, 3, 1, 2, 1, 7, 5, 9, 2, 5, 5, 7, 2, 5, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder188[] = {0, 6, 1, 5, 0, 4, 2, 3, 1, 2, 1, 7, 5, 9, 2, 5, 5, 7, 3, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder189[] = {0, 6, 1, 5, 0, 4, 2, 3, 1, 2, 3, 8, 8, 9, 1, 7, 4, 7, 2, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder190[] = {0, 6, 1, 5, 0, 4, 2, 3, 1, 3, 6, 10, 1, 6, 2, 5, 5, 7, 2, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder191[] = {0, 6, 1, 5, 0, 4, 2, 3, 1, 3, 6, 10, 1, 6, 2, 5, 5, 7, 3, 5, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder192[] = {0, 6, 1, 5, 0, 4, 2, 3, 1, 3, 6, 10, 3, 7, 7, 8, 1, 6, 2, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder193[] = {0, 6, 1, 5, 0, 4, 2, 3, 1, 3, 6, 10, 3, 7, 7, 8, 1, 6, 3, 6, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder194[] = {0, 6, 1, 5, 0, 4, 2, 3, 8, 11, 3, 10, 5, 9, 0, 2, 0, 2, 3, 6, 0, 2, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder195[] = {0, 6, 1, 5, 0, 4, 2, 3, 8, 11, 3, 10, 5, 9, 0, 2, 4, 7, 1, 6, 2, 5, 0, 1, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder196[] = {0, 6, 1, 5, 0, 4, 2, 3, 8, 11, 3, 10, 5, 9, 1, 2, 2, 5, 5, 6, 0, 1, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder197[] = {0, 6, 1, 5, 1, 4, 0, 3, 0, 2, 0, 1, 4, 9, 1, 5, 5, 7, 1, 4, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder198[] = {0, 6, 1, 5, 1, 4, 0, 3, 0, 2, 0, 1, 4, 9, 1, 5, 5, 7, 2, 5, 4, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder199[] = {0, 6, 1, 5, 1, 4, 0, 3, 0, 2, 0, 1, 4, 9, 1, 5, 5, 7, 2, 5, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder200[] = {0, 6, 1, 5, 1, 4, 0, 3, 0, 2, 0, 1, 4, 9, 1, 5, 5, 7, 3, 6, 0, 5, 1, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder201[] = {0, 6, 1, 5, 1, 4, 0, 3, 0, 2, 0, 4, 7, 9, 0, 4, 3, 7, 1, 3, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder202[] = {0, 6, 1, 5, 1, 4, 0, 3, 0, 2, 0, 4, 7, 9, 0, 5, 3, 7, 1, 3, 2, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder203[] = {0, 6, 1, 5, 1, 4, 0, 3, 0, 2, 0, 5, 8, 9, 0, 4, 3, 7, 1, 3, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder204[] = {0, 6, 1, 5, 1, 4, 0, 3, 0, 2, 0, 5, 8, 9, 0, 5, 3, 7, 1, 3, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder205[] = {0, 6, 1, 5, 1, 4, 0, 3, 0, 3, 6, 10, 0, 4, 6, 8, 0, 4, 1, 3, 2, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder206[] = {0, 6, 1, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 1, 4, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder207[] = {0, 6, 1, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 1, 4, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder208[] = {0, 6, 1, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder209[] = {0, 6, 1, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder210[] = {0, 6, 1, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 4, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder211[] = {0, 6, 1, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder212[] = {0, 6, 1, 5, 1, 4, 0, 3, 1, 2, 0, 3, 6, 9, 0, 4, 5, 7, 2, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder213[] = {0, 6, 1, 5, 1, 4, 0, 3, 1, 2, 0, 3, 6, 9, 0, 5, 3, 7, 1, 3, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder214[] = {0, 6, 1, 5, 1, 4, 0, 3, 1, 2, 0, 3, 6, 9, 0, 5, 3, 7, 2, 4, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder215[] = {0, 6, 1, 5, 1, 4, 0, 3, 1, 2, 0, 3, 6, 9, 0, 5, 3, 7, 2, 4, 4, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder216[] = {0, 6, 1, 5, 1, 4, 0, 3, 1, 2, 0, 3, 6, 9, 0, 7, 3, 7, 2, 6, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder217[] = {0, 6, 1, 5, 1, 4, 0, 3, 1, 2, 6, 10, 1, 9, 4, 8, 0, 2, 3, 6, 0, 2, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder218[] = {0, 6, 1, 5, 1, 4, 0, 3, 1, 2, 6, 10, 1, 9, 4, 8, 0, 2, 3, 6, 0, 3, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder219[] = {0, 6, 1, 5, 1, 4, 0, 3, 1, 3, 6, 10, 0, 3, 5, 8, 0, 4, 1, 3, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder220[] = {0, 6, 1, 5, 1, 4, 0, 4, 0, 3, 6, 10, 0, 5, 6, 8, 0, 4, 1, 3, 2, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder221[] = {0, 6, 1, 5, 1, 4, 0, 4, 1, 3, 6, 10, 0, 3, 5, 8, 0, 4, 2, 3, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder222[] = {0, 6, 1, 5, 1, 4, 1, 3, 0, 2, 0, 1, 4, 9, 1, 5, 4, 7, 1, 5, 4, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder223[] = {0, 6, 1, 5, 1, 4, 1, 3, 0, 2, 0, 1, 4, 9, 1, 5, 4, 7, 1, 5, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder224[] = {0, 6, 1, 5, 1, 4, 1, 3, 0, 2, 0, 1, 4, 9, 1, 5, 4, 7, 2, 4, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder225[] = {0, 6, 1, 5, 1, 4, 1, 3, 0, 2, 0, 1, 4, 9, 1, 5, 4, 7, 3, 6, 0, 5, 0, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder226[] = {0, 6, 1, 5, 1, 4, 1, 3, 0, 2, 0, 4, 7, 9, 0, 4, 3, 7, 1, 3, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder227[] = {0, 6, 1, 5, 1, 4, 1, 3, 0, 2, 0, 4, 7, 9, 0, 5, 3, 7, 1, 3, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder228[] = {0, 6, 1, 5, 1, 4, 1, 3, 0, 2, 0, 5, 8, 9, 0, 4, 3, 7, 1, 3, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder229[] = {0, 6, 1, 5, 1, 4, 1, 3, 0, 2, 0, 5, 8, 9, 0, 5, 3, 7, 1, 3, 2, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder230[] = {0, 6, 1, 5, 1, 4, 1, 3, 0, 3, 6, 10, 0, 4, 6, 8, 0, 4, 1, 3, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder231[] = {0, 6, 1, 5, 1, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 4, 7, 1, 4, 4, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder232[] = {0, 6, 1, 5, 1, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 4, 7, 1, 4, 4, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder233[] = {0, 6, 1, 5, 1, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 4, 7, 2, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder234[] = {0, 6, 1, 5, 1, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 4, 7, 2, 4, 2, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder235[] = {0, 6, 1, 5, 1, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 4, 7, 2, 4, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder236[] = {0, 6, 1, 5, 1, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 4, 7, 3, 6, 0, 5, 0, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder237[] = {0, 6, 1, 5, 1, 4, 1, 3, 1, 2, 1, 7, 5, 9, 2, 5, 4, 7, 2, 4, 4, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder238[] = {0, 6, 1, 5, 1, 4, 1, 3, 1, 2, 1, 7, 5, 9, 2, 5, 4, 7, 3, 4, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder239[] = {0, 6, 1, 5, 1, 4, 1, 3, 1, 2, 3, 8, 7, 9, 1, 7, 4, 7, 3, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder240[] = {0, 6, 1, 5, 1, 4, 1, 3, 1, 2, 6, 10, 1, 9, 4, 8, 2, 5, 4, 6, 3, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder241[] = {0, 6, 1, 5, 1, 4, 1, 3, 1, 3, 6, 10, 1, 6, 2, 5, 4, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder242[] = {0, 6, 1, 5, 1, 4, 1, 4, 0, 3, 6, 10, 0, 5, 6, 8, 0, 4, 1, 3, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder243[] = {0, 6, 1, 5, 1, 4, 1, 4, 1, 3, 6, 10, 1, 6, 2, 5, 4, 7, 3, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder244[] = {0, 6, 1, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 6, 7, 0, 5, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder245[] = {0, 6, 1, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 6, 7, 1, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder246[] = {0, 6, 1, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 6, 7, 1, 4, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder247[] = {0, 6, 1, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 6, 7, 1, 6, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder248[] = {0, 6, 1, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder249[] = {0, 6, 1, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder250[] = {0, 6, 1, 5, 1, 4, 2, 3, 0, 2, 0, 3, 6, 9, 0, 4, 4, 7, 1, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder251[] = {0, 6, 1, 5, 1, 4, 2, 3, 0, 2, 0, 3, 6, 9, 0, 5, 3, 7, 1, 3, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder252[] = {0, 6, 1, 5, 1, 4, 2, 3, 0, 2, 0, 3, 6, 9, 0, 5, 3, 7, 1, 6, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder253[] = {0, 6, 1, 5, 1, 4, 2, 3, 0, 2, 0, 3, 6, 9, 0, 5, 3, 7, 2, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder254[] = {0, 6, 1, 5, 1, 4, 2, 3, 0, 2, 0, 3, 6, 9, 0, 6, 3, 7, 1, 6, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder255[] = {0, 6, 1, 5, 1, 4, 2, 3, 0, 3, 6, 10, 0, 3, 5, 8, 0, 4, 1, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder256[] = {0, 6, 1, 5, 1, 4, 2, 3, 0, 3, 6, 10, 0, 3, 5, 8, 0, 4, 2, 6, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder257[] = {0, 6, 1, 5, 1, 4, 2, 3, 0, 3, 6, 10, 0, 3, 5, 8, 0, 5, 1, 6, 2, 5, 0, 3, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder258[] = {0, 6, 1, 5, 1, 4, 2, 3, 0, 3, 6, 10, 0, 3, 5, 8, 0, 5, 2, 6, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder259[] = {0, 6, 1, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 5, 7, 0, 5, 1, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder260[] = {0, 6, 1, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 5, 7, 1, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder261[] = {0, 6, 1, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 5, 7, 1, 5, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder262[] = {0, 6, 1, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 5, 7, 2, 6, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder263[] = {0, 6, 1, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 5, 7, 2, 6, 3, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder264[] = {0, 6, 1, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 5, 7, 3, 6, 0, 5, 1, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder265[] = {0, 6, 1, 5, 1, 4, 2, 3, 1, 2, 1, 7, 5, 9, 2, 5, 5, 7, 2, 5, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder266[] = {0, 6, 1, 5, 1, 4, 2, 3, 1, 2, 1, 7, 5, 9, 2, 5, 5, 7, 3, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder267[] = {0, 6, 1, 5, 1, 4, 2, 3, 1, 2, 3, 8, 8, 9, 1, 7, 4, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder268[] = {0, 6, 1, 5, 1, 4, 2, 3, 1, 3, 6, 10, 1, 6, 2, 5, 5, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder269[] = {0, 6, 1, 5, 1, 4, 2, 3, 1, 3, 6, 10, 1, 6, 2, 5, 5, 7, 3, 5, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder270[] = {0, 6, 1, 5, 1, 4, 2, 3, 1, 3, 6, 10, 3, 7, 7, 8, 1, 6, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder271[] = {0, 6, 1, 5, 1, 4, 2, 3, 1, 3, 6, 10, 3, 7, 7, 8, 1, 6, 3, 6, 4, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder272[] = {0, 6, 1, 5, 1, 4, 2, 3, 8, 11, 3, 10, 5, 9, 0, 2, 0, 2, 3, 6, 0, 2, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder273[] = {0, 6, 1, 5, 1, 4, 2, 3, 8, 11, 3, 10, 5, 9, 0, 2, 0, 2, 3, 6, 0, 4, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder274[] = {0, 6, 1, 5, 1, 4, 2, 3, 8, 11, 3, 10, 5, 9, 1, 2, 2, 5, 5, 6, 3, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder275[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 1, 4, 9, 1, 5, 3, 7, 1, 3, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder276[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 1, 4, 9, 1, 5, 3, 7, 2, 4, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder277[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder278[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 1, 4, 9, 2, 6, 6, 7, 0, 5, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder279[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 1, 4, 9, 2, 6, 6, 7, 1, 6, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder280[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 1, 4, 9, 3, 7, 6, 7, 0, 5, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder281[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 1, 4, 9, 3, 7, 6, 7, 1, 6, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder282[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 3, 6, 9, 0, 4, 3, 7, 1, 3, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder283[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 3, 6, 9, 0, 4, 3, 7, 2, 4, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder284[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 3, 6, 9, 0, 4, 6, 7, 2, 5, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder285[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 3, 6, 9, 0, 5, 3, 7, 2, 4, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder286[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 3, 6, 9, 0, 5, 3, 7, 4, 6, 0, 5, 1, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder287[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 3, 6, 9, 2, 6, 6, 7, 0, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder288[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 3, 6, 9, 3, 7, 6, 7, 0, 6, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder289[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 4, 7, 9, 0, 4, 4, 7, 1, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder290[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 4, 7, 9, 0, 5, 3, 7, 1, 3, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder291[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 4, 7, 9, 0, 5, 3, 7, 1, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder292[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 4, 7, 9, 0, 6, 3, 7, 1, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder293[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 5, 8, 9, 0, 4, 4, 7, 1, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder294[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 5, 8, 9, 0, 5, 3, 7, 1, 3, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder295[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 5, 8, 9, 0, 5, 3, 7, 1, 6, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder296[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 0, 5, 8, 9, 0, 6, 3, 7, 1, 6, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder297[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 7, 10, 2, 9, 4, 8, 0, 1, 2, 6, 0, 2, 3, 4, 0, 1, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder298[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 2, 7, 10, 2, 9, 4, 8, 0, 2, 3, 6, 0, 2, 3, 4, 0, 1, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder299[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 3, 6, 10, 0, 3, 5, 8, 0, 4, 1, 3, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder300[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 3, 6, 10, 0, 3, 5, 8, 0, 4, 4, 6, 0, 5, 1, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder301[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 3, 6, 10, 0, 3, 5, 8, 2, 5, 5, 6, 0, 4, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder302[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 3, 6, 10, 0, 4, 6, 8, 0, 4, 1, 4, 2, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder303[] = {0, 6, 1, 5, 2, 4, 0, 3, 0, 3, 6, 10, 0, 4, 6, 8, 0, 5, 1, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder304[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 3, 7, 1, 3, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder305[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 3, 7, 2, 4, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder306[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 5, 7, 2, 5, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder307[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 5, 7, 2, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder308[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 1, 4, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder309[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder310[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 7, 4, 7, 2, 5, 4, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder311[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 1, 7, 4, 7, 2, 6, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder312[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 2, 6, 5, 7, 0, 5, 2, 5, 0, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder313[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 2, 6, 5, 7, 1, 5, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder314[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 3, 7, 3, 7, 1, 5, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder315[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 3, 7, 4, 7, 1, 5, 4, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder316[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 3, 7, 4, 7, 1, 5, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder317[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 3, 7, 4, 7, 1, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder318[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 3, 7, 4, 7, 1, 6, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder319[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 0, 1, 4, 9, 3, 7, 6, 7, 0, 5, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder320[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 1, 6, 6, 9, 2, 7, 6, 7, 3, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder321[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 1, 6, 8, 9, 4, 8, 5, 7, 2, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder322[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 1, 7, 5, 9, 2, 5, 5, 7, 3, 5, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder323[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 1, 7, 5, 9, 2, 7, 4, 7, 3, 5, 4, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder324[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 1, 7, 5, 9, 3, 6, 5, 7, 2, 5, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder325[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 1, 7, 5, 9, 4, 7, 4, 7, 2, 5, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder326[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 1, 7, 5, 9, 4, 7, 4, 7, 2, 5, 4, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder327[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 1, 8, 5, 9, 2, 7, 6, 7, 3, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder328[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 1, 8, 5, 9, 3, 6, 5, 7, 2, 5, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder329[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 1, 8, 5, 9, 4, 7, 6, 7, 2, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder330[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 1, 8, 5, 9, 4, 7, 6, 7, 2, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder331[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 1, 8, 5, 9, 7, 8, 2, 7, 3, 6, 0, 1, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder332[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 1, 10, 5, 9, 2, 8, 5, 7, 3, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder333[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 1, 10, 5, 9, 4, 8, 5, 7, 2, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder334[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 1, 10, 5, 9, 6, 8, 2, 7, 3, 6, 0, 1, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder335[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 3, 8, 5, 9, 1, 7, 6, 7, 3, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder336[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 6, 10, 1, 9, 4, 8, 0, 1, 2, 6, 0, 5, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder337[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 6, 10, 1, 9, 4, 8, 2, 5, 3, 6, 0, 1, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder338[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 8, 10, 3, 9, 4, 8, 0, 1, 2, 6, 0, 2, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder339[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 8, 10, 3, 9, 4, 8, 0, 1, 2, 6, 0, 3, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder340[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 8, 10, 3, 9, 4, 8, 1, 4, 3, 6, 0, 1, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder341[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 2, 8, 10, 3, 9, 4, 8, 1, 5, 3, 6, 0, 1, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder342[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 3, 6, 10, 1, 6, 3, 6, 5, 7, 2, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder343[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 3, 6, 10, 1, 7, 3, 6, 5, 7, 2, 5, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder344[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 5, 8, 10, 1, 6, 1, 5, 6, 7, 2, 4, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder345[] = {0, 6, 1, 5, 2, 4, 0, 3, 1, 5, 8, 10, 1, 6, 3, 6, 5, 7, 1, 5, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder346[] = {0, 6, 1, 5, 2, 4, 0, 4, 0, 3, 6, 10, 0, 3, 5, 8, 0, 4, 2, 3, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder347[] = {0, 6, 1, 5, 2, 4, 0, 4, 0, 3, 6, 10, 0, 3, 5, 8, 0, 4, 4, 6, 0, 5, 0, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder348[] = {0, 6, 1, 5, 2, 4, 0, 4, 0, 3, 6, 10, 0, 3, 5, 8, 3, 5, 5, 6, 0, 4, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder349[] = {0, 6, 1, 5, 2, 4, 0, 4, 0, 3, 6, 10, 0, 5, 6, 8, 0, 4, 1, 4, 2, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder350[] = {0, 6, 1, 5, 2, 4, 0, 4, 0, 3, 6, 10, 0, 5, 6, 8, 0, 5, 1, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder351[] = {0, 6, 1, 5, 2, 4, 0, 4, 1, 3, 6, 10, 1, 6, 4, 6, 5, 7, 2, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder352[] = {0, 6, 1, 5, 2, 4, 0, 4, 1, 3, 6, 10, 1, 7, 4, 6, 5, 7, 2, 5, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder353[] = {0, 6, 1, 5, 2, 4, 0, 4, 1, 5, 8, 10, 1, 6, 1, 5, 6, 7, 3, 4, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder354[] = {0, 6, 1, 5, 2, 4, 0, 4, 1, 5, 8, 10, 1, 6, 4, 6, 5, 7, 1, 5, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder355[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 1, 4, 9, 1, 5, 3, 7, 1, 3, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder356[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 1, 4, 9, 1, 5, 3, 7, 2, 4, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder357[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 1, 4, 9, 1, 5, 6, 7, 1, 4, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder358[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 1, 4, 9, 2, 6, 6, 7, 0, 5, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder359[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 1, 4, 9, 2, 6, 6, 7, 1, 6, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder360[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 1, 4, 9, 3, 7, 6, 7, 0, 5, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder361[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 1, 4, 9, 3, 7, 6, 7, 1, 6, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder362[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 3, 6, 9, 0, 4, 3, 7, 1, 3, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder363[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 3, 6, 9, 0, 4, 3, 7, 2, 4, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder364[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 3, 6, 9, 0, 4, 6, 7, 1, 4, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder365[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 3, 6, 9, 0, 5, 3, 7, 1, 3, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder366[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 3, 6, 9, 0, 5, 3, 7, 3, 6, 0, 5, 0, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder367[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 3, 6, 9, 2, 6, 6, 7, 0, 6, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder368[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 3, 6, 9, 3, 7, 6, 7, 0, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder369[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 4, 7, 9, 0, 4, 4, 7, 1, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder370[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 4, 7, 9, 0, 5, 3, 7, 1, 3, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder371[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 4, 7, 9, 0, 5, 3, 7, 1, 6, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder372[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 4, 7, 9, 0, 6, 3, 7, 1, 6, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder373[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 5, 8, 9, 0, 4, 4, 7, 1, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder374[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 5, 8, 9, 0, 5, 3, 7, 1, 3, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder375[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 5, 8, 9, 0, 5, 3, 7, 1, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder376[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 2, 0, 5, 8, 9, 0, 6, 3, 7, 1, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder377[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 3, 6, 10, 0, 3, 5, 8, 0, 4, 1, 3, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder378[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 3, 6, 10, 0, 3, 5, 8, 0, 4, 1, 3, 3, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder379[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 3, 6, 10, 0, 3, 5, 8, 2, 5, 5, 6, 0, 4, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder380[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 3, 6, 10, 0, 4, 6, 8, 0, 4, 1, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder381[] = {0, 6, 1, 5, 2, 4, 1, 3, 0, 3, 6, 10, 0, 4, 6, 8, 0, 5, 1, 6, 2, 5, 0, 3, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder382[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 3, 7, 1, 3, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder383[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 3, 7, 2, 4, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder384[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 5, 7, 2, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder385[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 1, 4, 3, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder386[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder387[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 7, 4, 7, 2, 5, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder388[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1, 4, 9, 1, 7, 4, 7, 2, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder389[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1, 4, 9, 2, 6, 5, 7, 0, 5, 1, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder390[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1, 4, 9, 2, 6, 5, 7, 1, 5, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder391[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1, 4, 9, 3, 7, 3, 7, 1, 5, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder392[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1, 4, 9, 3, 7, 4, 7, 1, 5, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder393[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1, 4, 9, 3, 7, 4, 7, 1, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder394[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1, 4, 9, 3, 7, 4, 7, 1, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder395[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1, 4, 9, 3, 7, 6, 7, 0, 5, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder396[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 1, 7, 5, 9, 2, 7, 4, 7, 3, 5, 4, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder397[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 1, 7, 5, 9, 3, 6, 5, 7, 2, 5, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder398[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 1, 7, 5, 9, 4, 7, 4, 7, 2, 5, 4, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder399[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 1, 8, 5, 9, 2, 7, 6, 7, 3, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder400[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 1, 8, 5, 9, 3, 6, 5, 7, 2, 5, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder401[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 1, 8, 5, 9, 4, 7, 6, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder402[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 1, 10, 5, 9, 2, 8, 5, 7, 3, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder403[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 1, 10, 5, 9, 4, 8, 5, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder404[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 6, 10, 1, 9, 4, 8, 0, 1, 1, 6, 3, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder405[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 8, 10, 3, 9, 4, 8, 0, 1, 2, 6, 0, 2, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder406[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 2, 8, 10, 3, 9, 4, 8, 1, 5, 3, 6, 3, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder407[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 3, 6, 10, 1, 6, 3, 6, 5, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder408[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 3, 6, 10, 1, 7, 3, 6, 5, 7, 2, 5, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder409[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 5, 8, 10, 1, 6, 1, 5, 6, 7, 2, 4, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder410[] = {0, 6, 1, 5, 2, 4, 1, 3, 1, 5, 8, 10, 1, 6, 3, 6, 5, 7, 1, 5, 4, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder411[] = {0, 6, 1, 5, 2, 4, 1, 3, 9, 11, 4, 10, 5, 9, 0, 2, 0, 1, 2, 6, 0, 2, 3, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder412[] = {0, 6, 1, 5, 2, 4, 1, 3, 9, 11, 4, 10, 5, 9, 0, 2, 0, 2, 3, 6, 0, 2, 3, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder413[] = {0, 6, 1, 5, 2, 4, 1, 3, 9, 11, 4, 10, 5, 9, 0, 8, 0, 1, 0, 4, 2, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder414[] = {0, 6, 1, 5, 2, 4, 1, 3, 9, 11, 4, 10, 5, 9, 0, 8, 0, 1, 0, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder415[] = {0, 6, 1, 5, 2, 4, 1, 3, 9, 11, 4, 10, 5, 9, 1, 2, 0, 1, 2, 6, 0, 2, 2, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder416[] = {0, 6, 1, 5, 2, 4, 1, 3, 9, 11, 4, 10, 5, 9, 1, 2, 0, 1, 2, 6, 1, 4, 1, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder417[] = {0, 6, 1, 5, 2, 4, 1, 3, 9, 11, 4, 10, 5, 9, 1, 2, 5, 7, 2, 6, 2, 5, 0, 1, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder418[] = {0, 6, 1, 5, 2, 4, 1, 4, 0, 3, 6, 10, 0, 3, 5, 8, 0, 4, 2, 3, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder419[] = {0, 6, 1, 5, 2, 4, 1, 4, 0, 3, 6, 10, 0, 3, 5, 8, 0, 4, 2, 3, 3, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder420[] = {0, 6, 1, 5, 2, 4, 1, 4, 0, 3, 6, 10, 0, 3, 5, 8, 3, 5, 5, 6, 0, 4, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder421[] = {0, 6, 1, 5, 2, 4, 1, 4, 0, 3, 6, 10, 0, 5, 6, 8, 0, 4, 1, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder422[] = {0, 6, 1, 5, 2, 4, 1, 4, 0, 3, 6, 10, 0, 5, 6, 8, 0, 5, 1, 6, 2, 5, 0, 3, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder423[] = {0, 6, 1, 5, 2, 4, 1, 4, 1, 3, 6, 10, 1, 6, 4, 6, 5, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder424[] = {0, 6, 1, 5, 2, 4, 1, 4, 1, 3, 6, 10, 1, 7, 4, 6, 5, 7, 2, 5, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder425[] = {0, 6, 1, 5, 2, 4, 1, 4, 1, 5, 8, 10, 1, 6, 1, 5, 6, 7, 3, 4, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder426[] = {0, 6, 1, 5, 2, 4, 1, 4, 1, 5, 8, 10, 1, 6, 4, 6, 5, 7, 1, 5, 4, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder427[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 0, 7, 2, 7, 4, 6, 4, 5, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder428[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 3, 7, 1, 5, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder429[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 3, 7, 2, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder430[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 4, 7, 1, 5, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder431[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 4, 7, 1, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder432[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 6, 7, 0, 5, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder433[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder434[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 6, 4, 7, 1, 5, 4, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder435[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 6, 4, 7, 1, 6, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder436[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 1, 8, 4, 7, 1, 6, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder437[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 2, 6, 3, 7, 1, 5, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder438[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 2, 6, 4, 7, 1, 5, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder439[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 2, 6, 4, 7, 1, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder440[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 2, 6, 6, 7, 0, 5, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder441[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 3, 7, 5, 7, 0, 5, 2, 5, 0, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder442[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 9, 3, 7, 5, 7, 1, 5, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder443[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 1, 6, 6, 9, 2, 6, 6, 7, 2, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder444[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 1, 6, 7, 9, 3, 8, 5, 7, 2, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder445[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 1, 7, 5, 9, 2, 5, 4, 7, 2, 5, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder446[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 1, 7, 5, 9, 2, 6, 4, 7, 2, 5, 4, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder447[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 1, 7, 5, 9, 2, 8, 4, 7, 2, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder448[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 1, 7, 5, 9, 3, 6, 4, 7, 2, 5, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder449[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 1, 7, 5, 9, 4, 7, 5, 7, 2, 5, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder450[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 1, 8, 5, 9, 2, 6, 6, 7, 2, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder451[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 1, 8, 5, 9, 2, 8, 4, 7, 2, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder452[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 1, 8, 5, 9, 3, 6, 6, 7, 2, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder453[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 1, 8, 5, 9, 4, 7, 5, 7, 2, 5, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder454[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 1, 8, 5, 9, 6, 8, 2, 7, 2, 6, 0, 1, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder455[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 1, 9, 5, 9, 2, 8, 5, 7, 2, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder456[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 1, 9, 5, 9, 3, 8, 5, 7, 2, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder457[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 1, 9, 5, 9, 6, 8, 2, 7, 2, 6, 0, 1, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder458[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 2, 3, 8, 5, 9, 1, 6, 6, 7, 2, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder459[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 3, 6, 10, 1, 6, 2, 6, 4, 7, 2, 4, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder460[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 3, 6, 10, 1, 6, 2, 7, 4, 7, 2, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder461[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 3, 6, 10, 1, 6, 2, 7, 4, 7, 3, 5, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder462[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 3, 6, 10, 1, 7, 2, 6, 6, 7, 2, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder463[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 3, 6, 10, 1, 7, 2, 7, 6, 7, 0, 1, 2, 5, 0, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder464[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 3, 6, 10, 1, 7, 2, 7, 6, 7, 3, 6, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder465[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 3, 6, 10, 1, 7, 2, 8, 4, 7, 3, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder466[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 3, 6, 10, 1, 7, 3, 7, 5, 7, 2, 5, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder467[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 3, 6, 10, 1, 8, 3, 8, 5, 7, 2, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder468[] = {0, 6, 1, 5, 2, 4, 2, 3, 0, 3, 6, 10, 1, 8, 4, 8, 6, 7, 0, 1, 2, 5, 0, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder469[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 4, 9, 0, 7, 3, 7, 0, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder470[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 3, 7, 1, 5, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder471[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 3, 7, 2, 4, 2, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder472[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 4, 7, 1, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder473[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 0, 5, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder474[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 5, 6, 7, 2, 5, 3, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder475[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 6, 4, 7, 1, 5, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder476[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 6, 4, 7, 1, 6, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder477[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 4, 9, 1, 8, 4, 7, 1, 6, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder478[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 4, 9, 2, 6, 3, 7, 1, 5, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder479[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 4, 9, 2, 6, 4, 7, 1, 6, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder480[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 4, 9, 2, 6, 6, 7, 0, 5, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder481[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 4, 9, 3, 7, 5, 7, 0, 5, 1, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder482[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 4, 9, 3, 7, 5, 7, 1, 5, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder483[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 1, 7, 5, 9, 2, 6, 4, 7, 2, 5, 4, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder484[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 1, 7, 5, 9, 2, 8, 4, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder485[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 1, 7, 5, 9, 4, 7, 5, 7, 2, 5, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder486[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 1, 8, 5, 9, 2, 6, 6, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder487[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 1, 8, 5, 9, 2, 8, 4, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder488[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 1, 8, 5, 9, 4, 7, 5, 7, 2, 5, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder489[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 1, 9, 5, 9, 2, 8, 5, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder490[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 1, 9, 5, 9, 3, 8, 5, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder491[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 7, 10, 2, 9, 4, 8, 0, 1, 2, 6, 0, 2, 1, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder492[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 7, 10, 2, 9, 4, 8, 0, 1, 2, 6, 1, 3, 1, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder493[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 7, 10, 2, 9, 4, 8, 0, 7, 0, 4, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder494[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 2, 7, 10, 2, 9, 4, 8, 0, 7, 0, 5, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder495[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 3, 6, 10, 1, 6, 2, 6, 4, 7, 2, 4, 4, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder496[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 3, 6, 10, 1, 6, 2, 7, 4, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder497[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 3, 6, 10, 1, 6, 2, 7, 4, 7, 3, 5, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder498[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 3, 6, 10, 1, 7, 2, 6, 6, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder499[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 3, 6, 10, 1, 7, 2, 7, 6, 7, 0, 1, 1, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder500[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 3, 6, 10, 1, 7, 2, 7, 6, 7, 3, 6, 4, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder501[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 3, 6, 10, 1, 7, 2, 8, 4, 7, 3, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder502[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 3, 6, 10, 1, 7, 3, 7, 5, 7, 2, 5, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder503[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 3, 6, 10, 1, 8, 3, 8, 5, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder504[] = {0, 6, 1, 5, 2, 4, 2, 3, 1, 3, 6, 10, 1, 8, 4, 8, 6, 7, 0, 1, 1, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder505[] = {0, 6, 1, 5, 2, 4, 2, 3, 8, 11, 3, 10, 5, 9, 0, 2, 0, 1, 2, 6, 0, 5, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder506[] = {0, 6, 1, 5, 2, 4, 2, 3, 8, 11, 3, 10, 5, 9, 0, 2, 2, 5, 3, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder507[] = {0, 6, 1, 5, 2, 4, 2, 3, 8, 11, 3, 10, 5, 9, 1, 2, 0, 1, 1, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder508[] = {0, 6, 1, 5, 2, 4, 2, 3, 10, 11, 5, 10, 5, 9, 0, 2, 0, 1, 2, 6, 0, 2, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder509[] = {0, 6, 1, 5, 2, 4, 2, 3, 10, 11, 5, 10, 5, 9, 0, 2, 0, 1, 2, 6, 0, 4, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder510[] = {0, 6, 1, 5, 2, 4, 2, 3, 10, 11, 5, 10, 5, 9, 0, 2, 1, 4, 3, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder511[] = {0, 6, 1, 5, 2, 4, 2, 3, 10, 11, 5, 10, 5, 9, 0, 2, 1, 5, 3, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder512[] = {0, 6, 1, 5, 2, 4, 2, 3, 10, 11, 5, 10, 5, 9, 0, 3, 4, 7, 1, 4, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder513[] = {0, 6, 1, 5, 2, 4, 2, 3, 10, 11, 5, 10, 5, 9, 1, 2, 0, 1, 2, 6, 0, 2, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder514[] = {0, 6, 1, 5, 2, 4, 2, 3, 10, 11, 5, 10, 5, 9, 1, 2, 1, 5, 3, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder515[] = {0, 6, 1, 5, 2, 4, 2, 3, 10, 11, 5, 10, 5, 9, 1, 2, 5, 7, 2, 6, 2, 5, 0, 1, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder516[] = {0, 6, 1, 5, 2, 4, 2, 3, 10, 11, 5, 10, 5, 9, 1, 3, 4, 7, 1, 4, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder517[] = {0, 6, 1, 5, 2, 4, 2, 4, 0, 3, 6, 10, 1, 6, 2, 6, 4, 7, 3, 4, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder518[] = {0, 6, 1, 5, 2, 4, 2, 4, 0, 3, 6, 10, 1, 7, 2, 6, 6, 7, 3, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder519[] = {0, 6, 1, 5, 2, 4, 2, 4, 1, 3, 6, 10, 1, 6, 2, 6, 4, 7, 3, 4, 4, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder520[] = {0, 6, 1, 5, 2, 4, 2, 4, 1, 3, 6, 10, 1, 7, 2, 6, 6, 7, 3, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder521[] = {0, 6, 1, 5, 2, 4, 2, 4, 10, 11, 5, 10, 6, 9, 0, 3, 4, 7, 1, 4, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder522[] = {0, 6, 1, 5, 2, 4, 2, 4, 10, 11, 5, 10, 6, 9, 1, 3, 4, 7, 1, 4, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder523[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 0, 3, 0, 2, 0, 2, 4, 6, 0, 2, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder524[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 0, 3, 0, 2, 0, 3, 5, 6, 0, 2, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder525[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 0, 3, 0, 2, 0, 3, 5, 6, 0, 4, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder526[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 0, 3, 0, 2, 4, 7, 1, 6, 2, 5, 0, 1, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder527[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 0, 3, 1, 2, 2, 6, 5, 6, 0, 1, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder528[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 0, 3, 1, 2, 3, 7, 3, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder529[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 0, 3, 1, 2, 3, 7, 3, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder530[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 1, 3, 0, 2, 0, 2, 4, 6, 0, 2, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder531[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 1, 3, 0, 2, 0, 2, 4, 6, 0, 4, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder532[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 1, 3, 0, 2, 0, 3, 5, 6, 0, 2, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder533[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 1, 3, 1, 2, 2, 6, 5, 6, 3, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder534[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 1, 3, 1, 2, 3, 7, 3, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder535[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 1, 3, 6, 8, 3, 7, 3, 6, 0, 2, 0, 1, 2, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder536[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 2, 3, 0, 2, 0, 1, 2, 6, 0, 5, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder537[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 2, 3, 0, 2, 2, 6, 3, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder538[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 2, 3, 0, 2, 3, 7, 5, 6, 0, 1, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder539[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 2, 3, 1, 2, 0, 1, 1, 6, 4, 5, 1, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder540[] = {0, 6, 1, 5, 2, 4, 10, 12, 5, 11, 6, 10, 2, 3, 1, 2, 3, 7, 5, 6, 3, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder541[] = {0, 6, 9, 14, 1, 5, 0, 4, 0, 3, 0, 2, 0, 3, 6, 8, 1, 5, 4, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder542[] = {0, 6, 9, 14, 1, 5, 0, 4, 0, 3, 0, 2, 0, 3, 6, 8, 1, 5, 4, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder543[] = {0, 6, 9, 14, 1, 5, 0, 4, 0, 3, 0, 2, 0, 4, 7, 8, 1, 5, 5, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder544[] = {0, 6, 9, 14, 1, 5, 0, 4, 0, 3, 1, 2, 0, 2, 5, 8, 1, 5, 4, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder545[] = {0, 6, 9, 14, 1, 5, 0, 4, 0, 3, 1, 2, 0, 2, 5, 8, 2, 6, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder546[] = {0, 6, 9, 14, 1, 5, 0, 4, 0, 3, 1, 2, 0, 2, 5, 8, 2, 6, 3, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder547[] = {0, 6, 9, 14, 1, 5, 0, 4, 0, 3, 8, 10, 3, 9, 4, 8, 0, 2, 0, 2, 4, 5, 0, 2, 2, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder548[] = {0, 6, 9, 14, 1, 5, 0, 4, 1, 3, 0, 2, 0, 3, 6, 8, 1, 5, 5, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder549[] = {0, 6, 9, 14, 1, 5, 0, 4, 1, 3, 0, 2, 0, 4, 7, 8, 1, 5, 4, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder550[] = {0, 6, 9, 14, 1, 5, 0, 4, 1, 3, 0, 2, 0, 4, 7, 8, 1, 5, 4, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder551[] = {0, 6, 9, 14, 1, 5, 0, 4, 1, 3, 0, 2, 6, 9, 1, 8, 3, 7, 0, 2, 3, 5, 0, 2, 2, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder552[] = {0, 6, 9, 14, 1, 5, 0, 4, 1, 3, 1, 2, 2, 7, 6, 8, 2, 6, 5, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder553[] = {0, 6, 9, 14, 1, 5, 0, 4, 1, 3, 1, 2, 2, 7, 6, 8, 3, 6, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder554[] = {0, 6, 9, 14, 1, 5, 0, 4, 2, 3, 0, 2, 0, 2, 5, 8, 0, 6, 3, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder555[] = {0, 6, 9, 14, 1, 5, 0, 4, 2, 3, 0, 2, 0, 2, 5, 8, 1, 5, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder556[] = {0, 6, 9, 14, 1, 5, 0, 4, 2, 3, 0, 2, 0, 2, 5, 8, 2, 6, 4, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder557[] = {0, 6, 9, 14, 1, 5, 0, 4, 2, 3, 1, 2, 2, 7, 7, 8, 0, 1, 3, 6, 0, 5, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder558[] = {0, 6, 9, 14, 1, 5, 0, 4, 2, 3, 1, 2, 2, 7, 7, 8, 3, 7, 5, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder559[] = {0, 6, 9, 14, 1, 5, 1, 4, 0, 3, 0, 2, 0, 3, 6, 8, 1, 5, 4, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder560[] = {0, 6, 9, 14, 1, 5, 1, 4, 0, 3, 0, 2, 0, 3, 6, 8, 1, 5, 4, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder561[] = {0, 6, 9, 14, 1, 5, 1, 4, 0, 3, 0, 2, 0, 4, 7, 8, 1, 5, 5, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder562[] = {0, 6, 9, 14, 1, 5, 1, 4, 0, 3, 0, 2, 0, 4, 7, 8, 1, 5, 5, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder563[] = {0, 6, 9, 14, 1, 5, 1, 4, 0, 3, 1, 2, 0, 2, 5, 8, 1, 5, 4, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder564[] = {0, 6, 9, 14, 1, 5, 1, 4, 0, 3, 1, 2, 0, 2, 5, 8, 2, 6, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder565[] = {0, 6, 9, 14, 1, 5, 1, 4, 0, 3, 1, 2, 0, 2, 5, 8, 2, 6, 3, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder566[] = {0, 6, 9, 14, 1, 5, 1, 4, 1, 3, 0, 2, 0, 3, 6, 8, 1, 5, 5, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder567[] = {0, 6, 9, 14, 1, 5, 1, 4, 1, 3, 0, 2, 0, 3, 6, 8, 1, 5, 5, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder568[] = {0, 6, 9, 14, 1, 5, 1, 4, 1, 3, 0, 2, 0, 4, 7, 8, 1, 5, 4, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder569[] = {0, 6, 9, 14, 1, 5, 1, 4, 1, 3, 0, 2, 0, 4, 7, 8, 1, 5, 4, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder570[] = {0, 6, 9, 14, 1, 5, 1, 4, 1, 3, 1, 2, 2, 7, 6, 8, 2, 6, 5, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder571[] = {0, 6, 9, 14, 1, 5, 1, 4, 1, 3, 1, 2, 2, 7, 6, 8, 3, 6, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder572[] = {0, 6, 9, 14, 1, 5, 1, 4, 2, 3, 0, 2, 0, 2, 5, 8, 0, 6, 1, 6, 3, 5, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder573[] = {0, 6, 9, 14, 1, 5, 1, 4, 2, 3, 0, 2, 0, 2, 5, 8, 1, 5, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder574[] = {0, 6, 9, 14, 1, 5, 1, 4, 2, 3, 0, 2, 0, 2, 5, 8, 2, 6, 4, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder575[] = {0, 6, 9, 14, 1, 5, 1, 4, 2, 3, 1, 2, 2, 7, 7, 8, 0, 1, 1, 6, 4, 5, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder576[] = {0, 6, 9, 14, 1, 5, 1, 4, 2, 3, 1, 2, 2, 7, 7, 8, 3, 7, 5, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder577[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 3, 0, 2, 0, 2, 5, 8, 1, 5, 5, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder578[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 3, 0, 2, 0, 2, 5, 8, 2, 6, 5, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder579[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 3, 0, 2, 0, 2, 5, 8, 2, 6, 5, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder580[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 3, 0, 2, 0, 3, 6, 8, 0, 6, 3, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder581[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 3, 0, 2, 0, 3, 6, 8, 1, 5, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder582[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 3, 0, 2, 0, 4, 7, 8, 0, 6, 1, 6, 3, 5, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder583[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 3, 0, 2, 0, 4, 7, 8, 1, 5, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder584[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 3, 0, 2, 6, 9, 1, 8, 3, 7, 0, 1, 2, 5, 0, 2, 2, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder585[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 3, 1, 2, 2, 7, 7, 8, 3, 7, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder586[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 3, 1, 2, 2, 9, 6, 8, 3, 7, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder587[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 3, 1, 2, 3, 8, 7, 8, 0, 1, 3, 6, 0, 5, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder588[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 3, 1, 2, 4, 9, 6, 8, 2, 7, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder589[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 3, 1, 2, 4, 9, 6, 8, 2, 7, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder590[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 3, 1, 2, 7, 9, 2, 8, 3, 7, 0, 1, 2, 5, 0, 4, 1, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder591[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 3, 1, 4, 7, 9, 1, 6, 2, 5, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder592[] = {0, 6, 9, 14, 1, 5, 2, 4, 0, 4, 1, 4, 7, 9, 1, 6, 3, 5, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder593[] = {0, 6, 9, 14, 1, 5, 2, 4, 1, 3, 0, 2, 0, 2, 5, 8, 1, 5, 5, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder594[] = {0, 6, 9, 14, 1, 5, 2, 4, 1, 3, 0, 2, 0, 2, 5, 8, 1, 5, 5, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder595[] = {0, 6, 9, 14, 1, 5, 2, 4, 1, 3, 0, 2, 0, 2, 5, 8, 2, 6, 5, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder596[] = {0, 6, 9, 14, 1, 5, 2, 4, 1, 3, 0, 2, 0, 3, 6, 8, 0, 6, 1, 6, 3, 5, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder597[] = {0, 6, 9, 14, 1, 5, 2, 4, 1, 3, 0, 2, 0, 3, 6, 8, 1, 5, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder598[] = {0, 6, 9, 14, 1, 5, 2, 4, 1, 3, 0, 2, 0, 4, 7, 8, 0, 6, 3, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder599[] = {0, 6, 9, 14, 1, 5, 2, 4, 1, 3, 0, 2, 0, 4, 7, 8, 1, 5, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder600[] = {0, 6, 9, 14, 1, 5, 2, 4, 1, 3, 1, 2, 2, 9, 6, 8, 3, 7, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder601[] = {0, 6, 9, 14, 1, 5, 2, 4, 1, 3, 1, 2, 3, 8, 7, 8, 0, 1, 1, 6, 4, 5, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder602[] = {0, 6, 9, 14, 1, 5, 2, 4, 1, 3, 1, 2, 4, 9, 6, 8, 2, 7, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder603[] = {0, 6, 9, 14, 1, 5, 2, 4, 1, 3, 1, 4, 7, 9, 1, 6, 2, 5, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder604[] = {0, 6, 9, 14, 1, 5, 2, 4, 1, 3, 8, 10, 3, 9, 4, 8, 0, 2, 0, 1, 2, 5, 0, 3, 2, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder605[] = {0, 6, 9, 14, 1, 5, 2, 4, 1, 3, 8, 10, 3, 9, 4, 8, 0, 7, 0, 1, 3, 5, 0, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder606[] = {0, 6, 9, 14, 1, 5, 2, 4, 1, 4, 1, 4, 7, 9, 1, 6, 3, 5, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder607[] = {0, 6, 9, 14, 1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 4, 8, 0, 7, 3, 6, 0, 5, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder608[] = {0, 6, 9, 14, 1, 5, 2, 4, 2, 3, 0, 2, 2, 7, 6, 8, 2, 7, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder609[] = {0, 6, 9, 14, 1, 5, 2, 4, 2, 3, 0, 2, 2, 8, 6, 8, 2, 7, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder610[] = {0, 6, 9, 14, 1, 5, 2, 4, 2, 3, 0, 2, 3, 8, 6, 8, 2, 7, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder611[] = {0, 6, 9, 14, 1, 5, 2, 4, 2, 3, 0, 2, 4, 9, 7, 8, 0, 1, 3, 6, 0, 5, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder612[] = {0, 6, 9, 14, 1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 2, 8, 6, 7, 3, 6, 4, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder613[] = {0, 6, 9, 14, 1, 5, 2, 4, 2, 3, 1, 2, 2, 8, 6, 8, 2, 7, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder614[] = {0, 6, 9, 14, 1, 5, 2, 4, 2, 3, 1, 2, 4, 9, 7, 8, 0, 1, 1, 6, 4, 5, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder615[] = {0, 6, 9, 14, 1, 5, 2, 4, 2, 3, 9, 10, 4, 9, 4, 8, 0, 2, 0, 1, 2, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder616[] = {0, 6, 9, 14, 1, 5, 2, 4, 2, 3, 9, 10, 4, 9, 4, 8, 1, 2, 4, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder617[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 0, 3, 0, 2, 0, 2, 5, 7, 0, 4, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder618[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 0, 3, 0, 2, 0, 3, 6, 7, 0, 4, 4, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder619[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 0, 3, 0, 2, 0, 3, 6, 7, 0, 5, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder620[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 0, 3, 0, 2, 6, 8, 1, 7, 2, 6, 0, 1, 2, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder621[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 0, 3, 1, 2, 2, 7, 6, 7, 0, 1, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder622[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 0, 3, 1, 2, 3, 8, 5, 7, 0, 1, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder623[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 0, 3, 1, 2, 3, 8, 5, 7, 0, 1, 4, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder624[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 1, 3, 0, 2, 0, 2, 5, 7, 0, 4, 4, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder625[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 1, 3, 0, 2, 0, 2, 5, 7, 0, 5, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder626[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 1, 3, 0, 2, 0, 3, 6, 7, 0, 4, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder627[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 1, 3, 1, 2, 2, 7, 6, 7, 5, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder628[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 1, 3, 1, 2, 3, 8, 5, 7, 5, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder629[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 1, 3, 8, 9, 3, 8, 3, 7, 0, 2, 0, 1, 3, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder630[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 2, 3, 0, 2, 0, 1, 4, 7, 0, 6, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder631[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 2, 3, 0, 2, 2, 7, 5, 7, 0, 1, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder632[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 2, 3, 0, 2, 3, 8, 6, 7, 0, 1, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder633[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 2, 3, 1, 2, 0, 1, 1, 7, 5, 6, 3, 5, 3, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder634[] = {0, 6, 9, 14, 3, 6, 8, 12, 1, 4, 2, 3, 1, 2, 3, 8, 6, 7, 5, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder635[] = {0, 6, 9, 14, 3, 6, 8, 12, 3, 5, 6, 10, 1, 3, 0, 2, 0, 1, 4, 6, 0, 5, 3, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder636[] = {0, 6, 9, 14, 3, 6, 8, 12, 3, 5, 6, 10, 1, 3, 1, 2, 6, 7, 1, 6, 1, 5, 0, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder637[] = {0, 6, 9, 14, 3, 6, 8, 12, 3, 6, 7, 10, 1, 3, 0, 2, 0, 1, 4, 6, 0, 5, 3, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder638[] = {0, 6, 9, 14, 3, 6, 8, 12, 3, 6, 7, 10, 1, 3, 1, 2, 6, 7, 1, 6, 1, 5, 0, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder639[] = {0, 6, 9, 14, 4, 7, 9, 12, 1, 4, 0, 3, 0, 2, 0, 2, 5, 7, 0, 4, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder640[] = {0, 6, 9, 14, 4, 7, 9, 12, 1, 4, 0, 3, 0, 2, 6, 8, 1, 7, 2, 6, 0, 1, 2, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder641[] = {0, 6, 9, 14, 4, 7, 9, 12, 1, 4, 0, 3, 1, 2, 2, 7, 6, 7, 0, 1, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder642[] = {0, 6, 9, 14, 4, 7, 9, 12, 1, 4, 1, 3, 0, 2, 0, 2, 5, 7, 0, 4, 4, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder643[] = {0, 6, 9, 14, 4, 7, 9, 12, 1, 4, 1, 3, 0, 2, 0, 2, 5, 7, 0, 5, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder644[] = {0, 6, 9, 14, 4, 7, 9, 12, 1, 4, 1, 3, 1, 2, 2, 7, 6, 7, 5, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder645[] = {0, 6, 9, 14, 4, 7, 9, 12, 1, 4, 2, 3, 0, 2, 0, 1, 4, 7, 0, 6, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder646[] = {0, 6, 9, 14, 4, 7, 9, 12, 1, 4, 2, 3, 0, 2, 2, 7, 5, 7, 0, 1, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder647[] = {0, 6, 9, 14, 4, 7, 9, 12, 1, 4, 2, 3, 1, 2, 0, 1, 1, 7, 5, 6, 3, 5, 3, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder648[] = {0, 6, 9, 14, 4, 8, 10, 12, 1, 4, 0, 3, 0, 2, 0, 2, 5, 7, 0, 4, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder649[] = {0, 6, 9, 14, 4, 8, 10, 12, 1, 4, 0, 3, 0, 2, 6, 8, 1, 7, 2, 6, 0, 1, 2, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder650[] = {0, 6, 9, 14, 4, 8, 10, 12, 1, 4, 0, 3, 1, 2, 2, 7, 6, 7, 0, 1, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder651[] = {0, 6, 9, 14, 4, 8, 10, 12, 1, 4, 1, 3, 0, 2, 0, 2, 5, 7, 0, 4, 4, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder652[] = {0, 6, 9, 14, 4, 8, 10, 12, 1, 4, 1, 3, 0, 2, 0, 2, 5, 7, 0, 5, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder653[] = {0, 6, 9, 14, 4, 8, 10, 12, 1, 4, 1, 3, 1, 2, 2, 7, 6, 7, 5, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder654[] = {0, 6, 9, 14, 4, 8, 10, 12, 1, 4, 2, 3, 0, 2, 0, 1, 4, 7, 0, 6, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder655[] = {0, 6, 9, 14, 4, 8, 10, 12, 1, 4, 2, 3, 0, 2, 2, 7, 5, 7, 0, 1, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder656[] = {0, 6, 9, 14, 4, 8, 10, 12, 1, 4, 2, 3, 1, 2, 0, 1, 1, 7, 5, 6, 3, 5, 3, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder657[] = {1, 3, 0, 2, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder658[] = {1, 3, 1, 2, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder659[] = {1, 4, 0, 3, 0, 2, 0, 2, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder660[] = {1, 4, 0, 3, 0, 2, 4, 6, 1, 5, 2, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder661[] = {1, 4, 0, 3, 1, 2, 2, 5, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder662[] = {1, 4, 1, 3, 0, 2, 0, 2, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder663[] = {1, 4, 1, 3, 0, 2, 0, 2, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder664[] = {1, 4, 1, 3, 1, 2, 2, 5, 4, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder665[] = {1, 4, 2, 3, 0, 2, 0, 1, 2, 5, 0, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder666[] = {1, 4, 2, 3, 0, 2, 2, 5, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder667[] = {1, 4, 2, 3, 1, 2, 0, 1, 1, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder668[] = {1, 4, 5, 7, 0, 2, 0, 1, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder669[] = {1, 5, 0, 4, 0, 3, 0, 2, 0, 3, 5, 7, 1, 4, 3, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder670[] = {1, 5, 0, 4, 0, 3, 0, 2, 0, 3, 5, 7, 1, 4, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder671[] = {1, 5, 0, 4, 0, 3, 1, 2, 0, 2, 4, 7, 1, 4, 3, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder672[] = {1, 5, 0, 4, 1, 3, 0, 2, 0, 3, 5, 7, 1, 4, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder673[] = {1, 5, 0, 4, 1, 3, 0, 2, 5, 8, 1, 7, 3, 6, 0, 2, 2, 4, 0, 1, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder674[] = {1, 5, 0, 4, 1, 3, 1, 2, 2, 6, 5, 7, 2, 5, 4, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder675[] = {1, 5, 0, 4, 2, 3, 0, 2, 0, 2, 4, 7, 0, 5, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder676[] = {1, 5, 0, 4, 2, 3, 0, 2, 0, 2, 4, 7, 1, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder677[] = {1, 5, 0, 4, 2, 3, 1, 2, 2, 6, 6, 7, 0, 1, 2, 5, 0, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder678[] = {1, 5, 1, 4, 0, 3, 0, 2, 0, 3, 5, 7, 1, 4, 3, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder679[] = {1, 5, 1, 4, 0, 3, 0, 2, 0, 3, 5, 7, 1, 4, 3, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder680[] = {1, 5, 1, 4, 0, 3, 1, 2, 0, 2, 4, 7, 1, 4, 3, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder681[] = {1, 5, 1, 4, 1, 3, 0, 2, 0, 3, 5, 7, 1, 4, 4, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder682[] = {1, 5, 1, 4, 1, 3, 0, 2, 0, 3, 5, 7, 1, 4, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder683[] = {1, 5, 1, 4, 1, 3, 1, 2, 2, 6, 5, 7, 2, 5, 4, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder684[] = {1, 5, 1, 4, 2, 3, 0, 2, 0, 2, 4, 7, 0, 5, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder685[] = {1, 5, 1, 4, 2, 3, 0, 2, 0, 2, 4, 7, 1, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder686[] = {1, 5, 1, 4, 2, 3, 1, 2, 2, 6, 6, 7, 0, 1, 1, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder687[] = {1, 5, 2, 4, 0, 3, 0, 2, 0, 2, 4, 7, 1, 4, 4, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder688[] = {1, 5, 2, 4, 0, 3, 0, 2, 0, 3, 5, 7, 0, 5, 2, 5, 2, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder689[] = {1, 5, 2, 4, 0, 3, 0, 2, 0, 3, 5, 7, 1, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder690[] = {1, 5, 2, 4, 0, 3, 0, 2, 5, 8, 1, 7, 3, 6, 0, 1, 1, 4, 0, 1, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder691[] = {1, 5, 2, 4, 0, 3, 1, 2, 3, 7, 6, 7, 0, 1, 2, 5, 0, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder692[] = {1, 5, 2, 4, 0, 3, 1, 4, 6, 8, 1, 5, 2, 4, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder693[] = {1, 5, 2, 4, 1, 3, 0, 2, 0, 2, 4, 7, 1, 4, 4, 5, 0, 2, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder694[] = {1, 5, 2, 4, 1, 3, 0, 2, 0, 2, 4, 7, 1, 4, 4, 5, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder695[] = {1, 5, 2, 4, 1, 3, 0, 2, 0, 3, 5, 7, 0, 5, 1, 5, 2, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder696[] = {1, 5, 2, 4, 1, 3, 0, 2, 0, 3, 5, 7, 1, 4, 2, 5, 0, 2, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder697[] = {1, 5, 2, 4, 1, 3, 1, 2, 3, 7, 6, 7, 0, 1, 1, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder698[] = {1, 5, 2, 4, 1, 3, 1, 4, 6, 8, 1, 5, 2, 4, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder699[] = {1, 5, 2, 4, 2, 3, 0, 2, 0, 1, 3, 7, 0, 6, 2, 5, 0, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder700[] = {1, 5, 2, 4, 2, 3, 0, 2, 2, 6, 5, 7, 2, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder701[] = {1, 5, 2, 4, 2, 3, 0, 2, 2, 7, 5, 7, 2, 6, 3, 5, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder702[] = {1, 5, 2, 4, 2, 3, 0, 2, 3, 7, 5, 7, 2, 6, 3, 5, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder703[] = {1, 5, 2, 4, 2, 3, 1, 2, 0, 1, 2, 7, 5, 6, 2, 5, 3, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder704[] = {1, 5, 2, 4, 2, 3, 1, 2, 2, 7, 5, 7, 2, 6, 3, 5, 3, 4, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder705[] = {1, 5, 2, 4, 2, 3, 8, 9, 4, 8, 4, 7, 0, 2, 0, 1, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder706[] = {1, 5, 2, 4, 2, 3, 8, 9, 4, 8, 4, 7, 1, 2, 3, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder707[] = {1, 5, 7, 10, 0, 3, 0, 2, 0, 1, 0, 5, 3, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder708[] = {1, 5, 7, 10, 0, 3, 0, 2, 0, 1, 0, 6, 3, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder709[] = {1, 5, 7, 10, 1, 4, 5, 8, 0, 2, 0, 1, 4, 5, 0, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder710[] = {1, 5, 7, 10, 2, 5, 6, 8, 0, 2, 0, 1, 4, 5, 0, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder711[] = {1, 6, 3, 5, 0, 4, 0, 3, 0, 3, 0, 2, 0, 2, 3, 6, 0, 5, 0, 4, 1, 2, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder712[] = {1, 6, 3, 5, 0, 4, 0, 3, 0, 3, 0, 2, 0, 4, 1, 6, 2, 5, 3, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder713[] = {1, 6, 3, 5, 0, 4, 0, 3, 0, 3, 0, 2, 0, 4, 1, 6, 2, 5, 3, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder714[] = {1, 6, 3, 5, 0, 4, 0, 3, 0, 3, 0, 2, 3, 7, 3, 6, 0, 2, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder715[] = {1, 6, 3, 5, 0, 4, 0, 3, 0, 3, 0, 2, 4, 7, 0, 2, 2, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder716[] = {1, 6, 3, 5, 0, 4, 0, 3, 0, 3, 0, 2, 4, 7, 0, 2, 2, 5, 3, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder717[] = {1, 6, 3, 5, 0, 4, 0, 3, 0, 3, 0, 2, 4, 7, 0, 2, 2, 5, 3, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder718[] = {1, 6, 3, 5, 0, 4, 0, 3, 0, 3, 0, 3, 0, 2, 3, 6, 3, 4, 1, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder719[] = {1, 6, 3, 5, 0, 4, 0, 3, 0, 3, 0, 3, 0, 4, 1, 6, 0, 5, 1, 4, 1, 2, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder720[] = {1, 6, 3, 5, 0, 4, 0, 3, 0, 3, 0, 3, 3, 7, 4, 6, 0, 2, 2, 3, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder721[] = {1, 6, 3, 5, 0, 4, 0, 3, 0, 3, 0, 3, 3, 7, 4, 6, 0, 3, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder722[] = {1, 6, 3, 5, 0, 4, 0, 3, 0, 3, 0, 3, 5, 7, 0, 2, 3, 4, 1, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder723[] = {1, 6, 3, 5, 0, 4, 0, 3, 0, 3, 0, 3, 5, 7, 0, 4, 1, 5, 0, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder724[] = {1, 6, 3, 5, 0, 4, 0, 3, 0, 3, 0, 3, 5, 7, 2, 6, 0, 5, 0, 3, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder725[] = {1, 6, 3, 5, 0, 4, 0, 3, 6, 9, 2, 8, 4, 7, 0, 6, 0, 1, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder726[] = {1, 6, 3, 5, 0, 4, 0, 3, 6, 9, 2, 8, 4, 7, 1, 6, 0, 1, 0, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder727[] = {1, 6, 3, 5, 0, 4, 0, 3, 6, 9, 2, 8, 4, 7, 4, 6, 0, 2, 0, 1, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder728[] = {1, 6, 3, 5, 0, 4, 0, 3, 6, 9, 2, 8, 4, 7, 4, 6, 0, 2, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder729[] = {1, 6, 3, 5, 0, 4, 1, 3, 0, 2, 4, 8, 0, 1, 2, 6, 0, 4, 1, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder730[] = {1, 6, 3, 5, 0, 4, 1, 3, 0, 2, 4, 8, 0, 1, 2, 6, 3, 5, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder731[] = {1, 6, 3, 5, 0, 4, 1, 3, 0, 2, 4, 8, 2, 7, 4, 6, 0, 1, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder732[] = {1, 6, 3, 5, 0, 4, 1, 3, 0, 2, 4, 8, 3, 7, 5, 6, 0, 1, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder733[] = {1, 6, 3, 5, 0, 4, 1, 3, 0, 2, 4, 8, 5, 7, 2, 6, 0, 1, 1, 4, 0, 2, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder734[] = {1, 6, 3, 5, 0, 4, 1, 3, 0, 2, 4, 8, 5, 7, 2, 6, 0, 1, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder735[] = {1, 6, 3, 5, 0, 4, 2, 3, 0, 2, 4, 8, 0, 1, 0, 5, 2, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder736[] = {1, 6, 3, 5, 0, 4, 2, 3, 0, 2, 4, 8, 0, 1, 0, 5, 2, 5, 3, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder737[] = {1, 6, 3, 5, 0, 4, 2, 3, 0, 2, 4, 8, 3, 7, 5, 6, 0, 1, 0, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder738[] = {1, 6, 3, 5, 0, 4, 2, 3, 0, 2, 4, 8, 5, 7, 2, 6, 0, 1, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder739[] = {1, 6, 3, 5, 0, 4, 2, 3, 7, 9, 3, 8, 4, 7, 0, 6, 0, 2, 0, 2, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder740[] = {1, 6, 3, 5, 0, 4, 2, 3, 7, 9, 3, 8, 4, 7, 0, 6, 0, 2, 2, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder741[] = {1, 6, 3, 5, 0, 4, 2, 3, 7, 9, 3, 8, 4, 7, 1, 6, 0, 1, 1, 4, 0, 2, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder742[] = {1, 6, 3, 5, 0, 4, 2, 3, 7, 9, 3, 8, 4, 7, 1, 6, 0, 1, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder743[] = {1, 6, 3, 5, 0, 4, 2, 3, 7, 9, 3, 8, 4, 7, 1, 6, 0, 1, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder744[] = {1, 6, 3, 5, 0, 4, 3, 8, 0, 3, 6, 8, 3, 7, 5, 6, 0, 2, 0, 1, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder745[] = {1, 6, 3, 5, 0, 4, 3, 8, 0, 3, 6, 8, 3, 7, 5, 6, 0, 2, 1, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder746[] = {1, 6, 3, 5, 0, 4, 3, 8, 0, 3, 7, 8, 0, 2, 0, 1, 2, 5, 2, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder747[] = {1, 6, 3, 5, 0, 4, 3, 8, 0, 3, 7, 8, 0, 2, 0, 2, 1, 5, 2, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder748[] = {1, 6, 3, 5, 0, 4, 3, 8, 0, 3, 7, 8, 4, 7, 4, 6, 0, 2, 0, 1, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder749[] = {1, 6, 3, 5, 0, 4, 3, 8, 0, 3, 7, 8, 4, 7, 4, 6, 0, 2, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder750[] = {1, 6, 3, 5, 0, 5, 3, 8, 0, 3, 6, 8, 4, 7, 5, 6, 0, 2, 0, 1, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder751[] = {1, 6, 3, 5, 0, 5, 3, 8, 0, 3, 6, 8, 4, 7, 5, 6, 0, 2, 1, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder752[] = {1, 6, 3, 5, 0, 5, 3, 8, 0, 3, 7, 8, 0, 2, 0, 1, 1, 5, 2, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder753[] = {1, 6, 3, 5, 0, 5, 3, 8, 0, 3, 7, 8, 0, 2, 0, 2, 2, 5, 2, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder754[] = {1, 6, 3, 5, 0, 5, 3, 8, 0, 3, 7, 8, 3, 7, 4, 6, 0, 2, 0, 1, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder755[] = {1, 6, 3, 5, 0, 5, 3, 8, 0, 3, 7, 8, 3, 7, 4, 6, 0, 2, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder756[] = {1, 6, 3, 5, 1, 4, 0, 3, 0, 3, 0, 2, 0, 2, 2, 6, 3, 4, 1, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder757[] = {1, 6, 3, 5, 1, 4, 0, 3, 0, 3, 0, 2, 0, 4, 1, 6, 2, 5, 0, 4, 1, 2, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder758[] = {1, 6, 3, 5, 1, 4, 0, 3, 0, 3, 0, 2, 3, 7, 3, 6, 0, 2, 2, 3, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder759[] = {1, 6, 3, 5, 1, 4, 0, 3, 0, 3, 0, 2, 3, 7, 3, 6, 0, 3, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder760[] = {1, 6, 3, 5, 1, 4, 0, 3, 0, 3, 0, 2, 4, 7, 0, 2, 3, 4, 1, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder761[] = {1, 6, 3, 5, 1, 4, 0, 3, 0, 3, 0, 2, 4, 7, 0, 4, 1, 5, 0, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder762[] = {1, 6, 3, 5, 1, 4, 0, 3, 0, 3, 0, 2, 4, 7, 2, 6, 0, 5, 0, 3, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder763[] = {1, 6, 3, 5, 1, 4, 0, 3, 0, 3, 0, 3, 0, 2, 2, 6, 0, 5, 0, 4, 1, 2, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder764[] = {1, 6, 3, 5, 1, 4, 0, 3, 0, 3, 0, 3, 0, 4, 1, 6, 2, 5, 3, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder765[] = {1, 6, 3, 5, 1, 4, 0, 3, 0, 3, 0, 3, 0, 4, 1, 6, 2, 5, 3, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder766[] = {1, 6, 3, 5, 1, 4, 0, 3, 0, 3, 0, 3, 3, 7, 4, 6, 0, 2, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder767[] = {1, 6, 3, 5, 1, 4, 0, 3, 0, 3, 0, 3, 5, 7, 0, 2, 2, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder768[] = {1, 6, 3, 5, 1, 4, 0, 3, 0, 3, 0, 3, 5, 7, 0, 2, 2, 5, 3, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder769[] = {1, 6, 3, 5, 1, 4, 0, 3, 0, 3, 0, 3, 5, 7, 0, 2, 2, 5, 3, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder770[] = {1, 6, 3, 5, 1, 4, 1, 3, 0, 2, 0, 1, 3, 7, 0, 5, 2, 5, 1, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder771[] = {1, 6, 3, 5, 1, 4, 1, 3, 0, 2, 0, 1, 3, 7, 0, 6, 2, 5, 0, 3, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder772[] = {1, 6, 3, 5, 1, 4, 1, 3, 0, 2, 0, 1, 3, 7, 4, 6, 1, 5, 0, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder773[] = {1, 6, 3, 5, 1, 4, 2, 3, 0, 2, 0, 1, 0, 6, 3, 6, 1, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder774[] = {1, 6, 3, 5, 1, 4, 2, 3, 0, 2, 0, 1, 0, 6, 3, 6, 1, 5, 3, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder775[] = {1, 6, 3, 5, 1, 4, 2, 3, 0, 2, 0, 1, 1, 6, 1, 6, 2, 5, 2, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder776[] = {1, 6, 3, 5, 1, 4, 2, 3, 0, 2, 0, 1, 1, 6, 1, 6, 2, 5, 2, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder777[] = {1, 6, 3, 5, 1, 4, 2, 3, 7, 9, 3, 8, 4, 7, 0, 6, 0, 2, 2, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder778[] = {1, 6, 3, 5, 1, 4, 2, 3, 7, 9, 3, 8, 4, 7, 1, 6, 0, 1, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder779[] = {1, 6, 3, 5, 1, 4, 3, 8, 0, 3, 0, 2, 0, 1, 2, 6, 3, 5, 2, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder780[] = {1, 6, 3, 5, 1, 4, 3, 8, 0, 3, 0, 2, 0, 1, 2, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder781[] = {1, 6, 3, 5, 1, 4, 3, 8, 0, 3, 0, 2, 0, 1, 5, 6, 1, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder782[] = {1, 6, 3, 5, 1, 4, 3, 8, 0, 3, 0, 2, 0, 2, 1, 6, 2, 5, 3, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder783[] = {1, 6, 3, 5, 1, 4, 8, 10, 4, 9, 5, 8, 0, 3, 5, 6, 0, 2, 0, 1, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder784[] = {1, 6, 3, 5, 1, 4, 8, 10, 4, 9, 5, 8, 0, 3, 5, 6, 0, 2, 0, 1, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder785[] = {1, 6, 3, 5, 1, 4, 8, 10, 4, 9, 5, 8, 0, 3, 5, 6, 0, 2, 0, 1, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder786[] = {1, 6, 3, 5, 1, 4, 8, 10, 4, 9, 5, 8, 0, 3, 5, 6, 0, 2, 1, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder787[] = {1, 6, 3, 5, 1, 4, 8, 10, 4, 9, 5, 8, 0, 3, 5, 6, 0, 2, 1, 3, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder788[] = {1, 6, 3, 5, 1, 4, 8, 10, 4, 9, 5, 8, 0, 7, 0, 2, 0, 1, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder789[] = {1, 6, 3, 5, 1, 4, 8, 10, 4, 9, 5, 8, 0, 7, 1, 2, 3, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder790[] = {1, 6, 3, 5, 1, 4, 8, 10, 4, 9, 5, 8, 1, 3, 0, 5, 0, 1, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder791[] = {1, 6, 3, 5, 1, 4, 8, 10, 4, 9, 5, 8, 2, 3, 4, 6, 2, 5, 2, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder792[] = {1, 6, 3, 5, 1, 5, 3, 8, 0, 3, 0, 2, 0, 1, 1, 6, 2, 5, 3, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder793[] = {1, 6, 3, 5, 1, 5, 3, 8, 0, 3, 0, 2, 0, 2, 2, 6, 3, 5, 2, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder794[] = {1, 6, 3, 5, 1, 5, 3, 8, 0, 3, 0, 2, 0, 2, 2, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder795[] = {1, 6, 3, 5, 1, 5, 3, 8, 0, 3, 0, 2, 0, 2, 5, 6, 1, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder796[] = {1, 6, 3, 5, 2, 4, 0, 3, 0, 2, 4, 8, 0, 1, 2, 6, 0, 4, 1, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder797[] = {1, 6, 3, 5, 2, 4, 0, 3, 0, 2, 4, 8, 0, 1, 2, 6, 2, 5, 0, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder798[] = {1, 6, 3, 5, 2, 4, 0, 3, 0, 2, 4, 8, 2, 7, 4, 6, 0, 1, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder799[] = {1, 6, 3, 5, 2, 4, 0, 3, 0, 2, 4, 8, 3, 7, 5, 6, 0, 1, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder800[] = {1, 6, 3, 5, 2, 4, 0, 3, 0, 2, 4, 8, 6, 7, 3, 6, 0, 1, 1, 4, 0, 2, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder801[] = {1, 6, 3, 5, 2, 4, 0, 3, 0, 2, 4, 8, 6, 7, 3, 6, 0, 1, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder802[] = {1, 6, 3, 5, 2, 4, 1, 3, 0, 2, 0, 1, 3, 7, 1, 5, 3, 5, 1, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder803[] = {1, 6, 3, 5, 2, 4, 1, 3, 0, 2, 0, 1, 3, 7, 1, 6, 3, 5, 0, 3, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder804[] = {1, 6, 3, 5, 2, 4, 1, 3, 0, 2, 0, 1, 3, 7, 3, 6, 0, 5, 0, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder805[] = {1, 6, 3, 5, 2, 4, 1, 3, 6, 9, 2, 8, 4, 7, 0, 6, 0, 1, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder806[] = {1, 6, 3, 5, 2, 4, 1, 3, 6, 9, 2, 8, 4, 7, 0, 6, 0, 1, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder807[] = {1, 6, 3, 5, 2, 4, 2, 3, 0, 2, 0, 2, 0, 4, 2, 6, 2, 5, 3, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder808[] = {1, 6, 3, 5, 2, 4, 2, 3, 0, 2, 0, 3, 2, 7, 4, 6, 0, 3, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder809[] = {1, 6, 3, 5, 2, 4, 2, 3, 7, 9, 3, 8, 4, 7, 0, 6, 0, 1, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder810[] = {1, 6, 3, 5, 2, 4, 2, 3, 7, 9, 3, 8, 4, 7, 1, 6, 0, 1, 0, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder811[] = {1, 6, 3, 5, 2, 4, 3, 8, 0, 3, 1, 2, 3, 7, 3, 6, 1, 5, 0, 1, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder812[] = {1, 6, 3, 5, 2, 4, 3, 8, 0, 3, 1, 2, 3, 7, 3, 6, 1, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder813[] = {1, 6, 3, 5, 2, 5, 3, 8, 0, 3, 1, 2, 3, 7, 3, 6, 2, 5, 0, 1, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder814[] = {1, 6, 3, 5, 2, 5, 3, 8, 0, 3, 1, 2, 3, 7, 3, 6, 2, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder815[] = {1, 6, 3, 5, 3, 4, 0, 3, 0, 2, 4, 8, 0, 1, 0, 5, 2, 5, 1, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder816[] = {1, 6, 3, 5, 3, 4, 0, 3, 0, 2, 4, 8, 0, 1, 0, 5, 2, 5, 3, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder817[] = {1, 6, 3, 5, 3, 4, 0, 3, 0, 2, 4, 8, 2, 7, 4, 6, 0, 1, 0, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder818[] = {1, 6, 3, 5, 3, 4, 0, 3, 0, 2, 4, 8, 6, 7, 3, 6, 0, 1, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder819[] = {1, 6, 3, 5, 3, 4, 1, 3, 0, 2, 0, 1, 0, 6, 1, 6, 2, 5, 2, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder820[] = {1, 6, 3, 5, 3, 4, 1, 3, 0, 2, 0, 1, 0, 6, 1, 6, 2, 5, 2, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder821[] = {1, 6, 3, 5, 3, 4, 1, 3, 0, 2, 0, 1, 1, 6, 4, 6, 1, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder822[] = {1, 6, 3, 5, 3, 4, 1, 3, 0, 2, 0, 1, 1, 6, 4, 6, 1, 5, 3, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder823[] = {1, 6, 3, 5, 3, 4, 2, 3, 0, 2, 0, 2, 2, 7, 3, 6, 0, 3, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder824[] = {1, 6, 3, 5, 3, 4, 2, 3, 0, 2, 0, 3, 0, 4, 2, 6, 2, 5, 3, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder825[] = {1, 6, 3, 5, 3, 4, 9, 10, 5, 9, 5, 8, 0, 3, 0, 5, 0, 2, 0, 2, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder826[] = {1, 6, 3, 5, 3, 4, 9, 10, 5, 9, 5, 8, 0, 3, 0, 5, 0, 2, 2, 4, 0, 1, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder827[] = {1, 6, 3, 5, 3, 4, 9, 10, 5, 9, 5, 8, 0, 3, 1, 5, 0, 1, 1, 4, 0, 2, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder828[] = {1, 6, 3, 5, 3, 4, 9, 10, 5, 9, 5, 8, 0, 3, 1, 5, 0, 1, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder829[] = {1, 6, 3, 5, 3, 4, 9, 10, 5, 9, 5, 8, 1, 3, 0, 5, 0, 2, 2, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder830[] = {1, 6, 3, 5, 3, 4, 9, 10, 5, 9, 5, 8, 1, 3, 1, 5, 0, 1, 0, 3, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder831[] = {1, 6, 3, 5, 3, 4, 9, 10, 5, 9, 5, 8, 1, 3, 4, 6, 2, 5, 2, 4, 0, 2, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder832[] = {1, 6, 3, 5, 3, 4, 9, 10, 5, 9, 5, 8, 2, 3, 0, 5, 0, 1, 1, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder833[] = {1, 6, 3, 5, 3, 4, 9, 10, 5, 9, 5, 8, 2, 3, 1, 5, 0, 1, 0, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder834[] = {1, 6, 3, 5, 3, 4, 9, 10, 5, 9, 5, 8, 2, 7, 0, 2, 0, 1, 1, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder835[] = {1, 6, 3, 5, 3, 4, 9, 10, 5, 9, 5, 8, 2, 7, 1, 2, 3, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder836[] = {2, 5, 6, 7, 0, 2, 0, 1, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder837[] = {2, 6, 8, 10, 0, 3, 0, 2, 0, 1, 0, 5, 3, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder838[] = {2, 6, 8, 10, 0, 3, 0, 2, 0, 1, 0, 6, 3, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder839[] = {2, 6, 8, 10, 2, 5, 6, 8, 0, 2, 0, 1, 4, 5, 0, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder840[] = {3, 5, 6, 8, 1, 3, 0, 2, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder841[] = {3, 5, 6, 8, 1, 3, 1, 2, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder842[] = {3, 6, 8, 11, 1, 4, 0, 3, 0, 2, 0, 2, 4, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder843[] = {3, 6, 8, 11, 1, 4, 0, 3, 0, 2, 5, 7, 1, 6, 2, 5, 0, 1, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder844[] = {3, 6, 8, 11, 1, 4, 0, 3, 1, 2, 2, 6, 5, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder845[] = {3, 6, 8, 11, 1, 4, 1, 3, 0, 2, 0, 2, 4, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder846[] = {3, 6, 8, 11, 1, 4, 1, 3, 0, 2, 0, 2, 4, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder847[] = {3, 6, 8, 11, 1, 4, 1, 3, 1, 2, 2, 6, 5, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder848[] = {3, 6, 8, 11, 1, 4, 2, 3, 0, 2, 0, 1, 3, 6, 0, 5, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder849[] = {3, 6, 8, 11, 1, 4, 2, 3, 0, 2, 2, 6, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder850[] = {3, 6, 8, 11, 1, 4, 2, 3, 1, 2, 0, 1, 1, 6, 4, 5, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder851[] = {3, 6, 8, 11, 3, 5, 6, 9, 1, 3, 0, 2, 0, 1, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder852[] = {3, 6, 8, 11, 3, 5, 6, 9, 1, 3, 1, 2, 5, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder853[] = {3, 7, 9, 10, 0, 3, 0, 2, 0, 1, 0, 5, 3, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder854[] = {3, 7, 9, 10, 0, 3, 0, 2, 0, 1, 0, 6, 3, 5, 3, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder855[] = {4, 7, 9, 11, 0, 4, 1, 3, 0, 2, 0, 1, 3, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder856[] = {4, 7, 9, 11, 0, 4, 1, 3, 0, 2, 0, 1, 3, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder857[] = {4, 7, 9, 11, 0, 4, 1, 3, 0, 2, 1, 5, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder858[] = {4, 7, 9, 11, 0, 4, 1, 3, 0, 2, 1, 6, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder859[] = {4, 7, 9, 11, 0, 4, 1, 3, 0, 3, 5, 7, 1, 5, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder860[] = {4, 7, 9, 11, 0, 4, 1, 3, 1, 2, 0, 1, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder861[] = {4, 7, 9, 11, 0, 4, 1, 3, 1, 2, 1, 6, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder862[] = {4, 7, 9, 11, 0, 4, 1, 3, 1, 2, 6, 7, 2, 6, 2, 5, 0, 1, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder863[] = {4, 7, 9, 11, 0, 4, 1, 3, 1, 3, 5, 7, 1, 5, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder864[] = {4, 7, 9, 11, 1, 4, 0, 3, 0, 2, 0, 2, 4, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder865[] = {4, 7, 9, 11, 1, 4, 0, 3, 0, 2, 5, 7, 1, 6, 2, 5, 0, 1, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder866[] = {4, 7, 9, 11, 1, 4, 0, 3, 1, 2, 2, 6, 5, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder867[] = {4, 7, 9, 11, 1, 4, 1, 3, 0, 2, 0, 2, 4, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder868[] = {4, 7, 9, 11, 1, 4, 1, 3, 0, 2, 0, 2, 4, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder869[] = {4, 7, 9, 11, 1, 4, 1, 3, 1, 2, 2, 6, 5, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder870[] = {4, 7, 9, 11, 1, 4, 2, 3, 0, 2, 0, 1, 3, 6, 0, 5, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder871[] = {4, 7, 9, 11, 1, 4, 2, 3, 0, 2, 2, 6, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder872[] = {4, 7, 9, 11, 1, 4, 2, 3, 1, 2, 0, 1, 1, 6, 4, 5, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder873[] = {4, 8, 10, 11, 0, 4, 1, 3, 0, 2, 0, 1, 3, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder874[] = {4, 8, 10, 11, 0, 4, 1, 3, 0, 2, 0, 1, 3, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder875[] = {4, 8, 10, 11, 0, 4, 1, 3, 0, 2, 1, 5, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder876[] = {4, 8, 10, 11, 0, 4, 1, 3, 0, 2, 1, 6, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder877[] = {4, 8, 10, 11, 0, 4, 1, 3, 0, 3, 5, 7, 1, 5, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder878[] = {4, 8, 10, 11, 0, 4, 1, 3, 1, 2, 0, 1, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder879[] = {4, 8, 10, 11, 0, 4, 1, 3, 1, 2, 1, 6, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder880[] = {4, 8, 10, 11, 0, 4, 1, 3, 1, 2, 6, 7, 2, 6, 2, 5, 0, 1, 1, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder881[] = {4, 8, 10, 11, 0, 4, 1, 3, 1, 3, 5, 7, 1, 5, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder882[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 0, 2, 0, 1, 4, 8, 1, 5, 5, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder883[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 0, 2, 0, 1, 4, 8, 2, 6, 5, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder884[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 0, 2, 0, 1, 4, 8, 2, 6, 5, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder885[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 0, 2, 0, 3, 6, 8, 0, 4, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder886[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 0, 2, 0, 3, 6, 8, 0, 5, 3, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder887[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 0, 2, 0, 4, 7, 8, 0, 4, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder888[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 0, 2, 0, 4, 7, 8, 0, 5, 3, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder889[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 0, 2, 7, 9, 2, 8, 3, 7, 0, 1, 2, 5, 0, 2, 2, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder890[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 0, 3, 6, 9, 0, 3, 5, 7, 0, 4, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder891[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 8, 1, 5, 4, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder892[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 8, 1, 5, 4, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder893[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 8, 2, 6, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder894[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 8, 2, 6, 3, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder895[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 8, 2, 6, 5, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder896[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 8, 2, 6, 5, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder897[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 1, 2, 1, 6, 7, 8, 3, 7, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder898[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 1, 2, 1, 7, 5, 8, 2, 5, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder899[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 1, 2, 1, 7, 5, 8, 3, 6, 5, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder900[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 1, 2, 1, 7, 5, 8, 3, 6, 5, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder901[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 1, 2, 1, 9, 5, 8, 3, 7, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder902[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 1, 2, 6, 9, 1, 8, 3, 7, 0, 1, 2, 5, 0, 4, 1, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder903[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 3, 1, 3, 6, 9, 1, 6, 2, 5, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder904[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 4, 0, 3, 6, 9, 0, 4, 5, 7, 0, 4, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder905[] = {4, 8, 11, 14, 0, 5, 1, 4, 0, 4, 1, 3, 6, 9, 1, 6, 3, 5, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder906[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 0, 2, 0, 1, 4, 8, 1, 5, 5, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder907[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 0, 2, 0, 1, 4, 8, 1, 5, 5, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder908[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 0, 2, 0, 1, 4, 8, 2, 6, 5, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder909[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 0, 2, 0, 3, 6, 8, 0, 4, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder910[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 0, 2, 0, 3, 6, 8, 0, 5, 3, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder911[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 0, 2, 0, 4, 7, 8, 0, 4, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder912[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 0, 2, 0, 4, 7, 8, 0, 5, 3, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder913[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 0, 3, 6, 9, 0, 3, 5, 7, 0, 4, 0, 4, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder914[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 1, 2, 0, 1, 4, 8, 1, 5, 4, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder915[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 1, 2, 0, 1, 4, 8, 1, 5, 4, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder916[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 1, 2, 0, 1, 4, 8, 2, 6, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder917[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 1, 2, 0, 1, 4, 8, 2, 6, 3, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder918[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 1, 2, 0, 1, 4, 8, 2, 6, 5, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder919[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 1, 2, 1, 7, 5, 8, 2, 5, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder920[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 1, 2, 1, 7, 5, 8, 3, 6, 5, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder921[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 1, 2, 1, 9, 5, 8, 3, 7, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder922[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 1, 3, 6, 9, 1, 6, 2, 5, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder923[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 9, 10, 4, 9, 4, 8, 0, 2, 0, 1, 2, 5, 0, 3, 2, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder924[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 9, 10, 4, 9, 4, 8, 0, 7, 0, 1, 3, 5, 0, 4, 0, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder925[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 3, 9, 10, 4, 9, 4, 8, 1, 2, 0, 1, 2, 5, 0, 3, 2, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder926[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 4, 0, 3, 6, 9, 0, 4, 5, 7, 0, 4, 0, 4, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder927[] = {4, 8, 11, 14, 0, 5, 1, 4, 1, 4, 1, 3, 6, 9, 1, 6, 3, 5, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder928[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 8, 0, 6, 1, 6, 3, 5, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder929[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 8, 1, 5, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder930[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 8, 1, 5, 5, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder931[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 8, 1, 7, 4, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder932[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 8, 2, 6, 4, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder933[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 8, 2, 6, 4, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder934[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 0, 2, 1, 6, 6, 8, 2, 7, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder935[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 0, 2, 1, 7, 5, 8, 2, 5, 5, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder936[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 0, 2, 1, 7, 5, 8, 2, 7, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder937[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 0, 2, 1, 7, 5, 8, 3, 6, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder938[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 0, 2, 1, 8, 5, 8, 2, 7, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder939[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 0, 3, 6, 9, 1, 6, 2, 6, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder940[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 0, 3, 6, 9, 1, 6, 3, 7, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder941[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 0, 3, 6, 9, 1, 7, 2, 7, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder942[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 0, 3, 6, 9, 1, 7, 3, 7, 5, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder943[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 8, 0, 6, 3, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder944[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 8, 1, 5, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder945[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 8, 1, 7, 4, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder946[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 8, 2, 6, 4, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder947[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 8, 2, 6, 4, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder948[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 1, 2, 1, 7, 5, 8, 2, 7, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder949[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 1, 2, 1, 7, 5, 8, 3, 6, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder950[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 1, 2, 1, 8, 5, 8, 2, 7, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder951[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 1, 2, 7, 9, 2, 8, 3, 7, 0, 1, 2, 5, 0, 2, 2, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder952[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 1, 3, 6, 9, 1, 6, 2, 6, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder953[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 1, 3, 6, 9, 1, 6, 3, 7, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder954[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 1, 3, 6, 9, 1, 7, 2, 7, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder955[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 1, 3, 6, 9, 1, 7, 3, 7, 5, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder956[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 8, 10, 3, 9, 4, 8, 0, 2, 0, 1, 2, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder957[] = {4, 8, 11, 14, 0, 5, 1, 4, 2, 3, 8, 10, 3, 9, 4, 8, 1, 2, 4, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder958[] = {4, 8, 11, 14, 4, 7, 9, 12, 0, 4, 1, 3, 0, 2, 0, 1, 4, 7, 0, 4, 4, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder959[] = {4, 8, 11, 14, 4, 7, 9, 12, 0, 4, 1, 3, 0, 2, 0, 1, 4, 7, 0, 5, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder960[] = {4, 8, 11, 14, 4, 7, 9, 12, 0, 4, 1, 3, 0, 2, 1, 6, 5, 7, 0, 1, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder961[] = {4, 8, 11, 14, 4, 7, 9, 12, 0, 4, 1, 3, 0, 2, 1, 7, 5, 7, 0, 1, 4, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder962[] = {4, 8, 11, 14, 4, 7, 9, 12, 0, 4, 1, 3, 0, 3, 6, 8, 1, 6, 0, 1, 4, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder963[] = {4, 8, 11, 14, 4, 7, 9, 12, 0, 4, 1, 3, 1, 2, 0, 1, 4, 7, 0, 4, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder964[] = {4, 8, 11, 14, 4, 7, 9, 12, 0, 4, 1, 3, 1, 2, 1, 7, 5, 7, 5, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder965[] = {4, 8, 11, 14, 4, 7, 9, 12, 0, 4, 1, 3, 1, 2, 7, 8, 2, 7, 2, 6, 0, 1, 2, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder966[] = {4, 8, 11, 14, 4, 7, 9, 12, 0, 4, 1, 3, 1, 3, 6, 8, 1, 6, 5, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder967[] = {4, 8, 11, 14, 4, 8, 10, 12, 0, 4, 1, 3, 0, 2, 0, 1, 4, 7, 0, 4, 4, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder968[] = {4, 8, 11, 14, 4, 8, 10, 12, 0, 4, 1, 3, 0, 2, 0, 1, 4, 7, 0, 5, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder969[] = {4, 8, 11, 14, 4, 8, 10, 12, 0, 4, 1, 3, 0, 2, 1, 6, 5, 7, 0, 1, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder970[] = {4, 8, 11, 14, 4, 8, 10, 12, 0, 4, 1, 3, 0, 2, 1, 7, 5, 7, 0, 1, 4, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder971[] = {4, 8, 11, 14, 4, 8, 10, 12, 0, 4, 1, 3, 0, 3, 6, 8, 1, 6, 0, 1, 4, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder972[] = {4, 8, 11, 14, 4, 8, 10, 12, 0, 4, 1, 3, 1, 2, 0, 1, 4, 7, 0, 4, 3, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder973[] = {4, 8, 11, 14, 4, 8, 10, 12, 0, 4, 1, 3, 1, 2, 1, 7, 5, 7, 5, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder974[] = {4, 8, 11, 14, 4, 8, 10, 12, 0, 4, 1, 3, 1, 2, 7, 8, 2, 7, 2, 6, 0, 1, 2, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder975[] = {4, 8, 11, 14, 4, 8, 10, 12, 0, 4, 1, 3, 1, 3, 6, 8, 1, 6, 5, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder976[] = {5, 9, 12, 14, 0, 5, 1, 4, 0, 3, 0, 2, 0, 1, 4, 8, 1, 5, 5, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder977[] = {5, 9, 12, 14, 0, 5, 1, 4, 0, 3, 0, 2, 0, 3, 6, 8, 0, 4, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder978[] = {5, 9, 12, 14, 0, 5, 1, 4, 0, 3, 0, 2, 0, 3, 6, 8, 0, 5, 3, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder979[] = {5, 9, 12, 14, 0, 5, 1, 4, 0, 3, 0, 2, 7, 9, 2, 8, 3, 7, 0, 1, 2, 5, 0, 2, 2, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder980[] = {5, 9, 12, 14, 0, 5, 1, 4, 0, 3, 0, 3, 6, 9, 0, 3, 5, 7, 0, 4, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder981[] = {5, 9, 12, 14, 0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 8, 1, 5, 4, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder982[] = {5, 9, 12, 14, 0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 8, 1, 5, 4, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder983[] = {5, 9, 12, 14, 0, 5, 1, 4, 0, 3, 1, 2, 1, 7, 5, 8, 2, 5, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder984[] = {5, 9, 12, 14, 0, 5, 1, 4, 0, 3, 1, 3, 6, 9, 1, 6, 2, 5, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder985[] = {5, 9, 12, 14, 0, 5, 1, 4, 1, 3, 0, 2, 0, 1, 4, 8, 1, 5, 5, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder986[] = {5, 9, 12, 14, 0, 5, 1, 4, 1, 3, 0, 2, 0, 1, 4, 8, 1, 5, 5, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder987[] = {5, 9, 12, 14, 0, 5, 1, 4, 1, 3, 0, 2, 0, 3, 6, 8, 0, 4, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder988[] = {5, 9, 12, 14, 0, 5, 1, 4, 1, 3, 0, 2, 0, 3, 6, 8, 0, 5, 3, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder989[] = {5, 9, 12, 14, 0, 5, 1, 4, 1, 3, 0, 3, 6, 9, 0, 3, 5, 7, 0, 4, 0, 4, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder990[] = {5, 9, 12, 14, 0, 5, 1, 4, 1, 3, 1, 2, 0, 1, 4, 8, 1, 5, 4, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder991[] = {5, 9, 12, 14, 0, 5, 1, 4, 1, 3, 1, 2, 0, 1, 4, 8, 1, 5, 4, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder992[] = {5, 9, 12, 14, 0, 5, 1, 4, 1, 3, 1, 2, 1, 7, 5, 8, 2, 5, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder993[] = {5, 9, 12, 14, 0, 5, 1, 4, 1, 3, 1, 3, 6, 9, 1, 6, 2, 5, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder994[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 8, 0, 6, 1, 6, 3, 5, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder995[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 8, 1, 5, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder996[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 8, 1, 5, 5, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder997[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 8, 1, 7, 4, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder998[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 0, 2, 1, 6, 6, 8, 2, 7, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder999[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 0, 2, 1, 7, 5, 8, 2, 5, 5, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1000[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 0, 2, 1, 7, 5, 8, 2, 7, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1001[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 0, 2, 1, 8, 5, 8, 2, 7, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1002[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 0, 3, 6, 9, 1, 6, 2, 6, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1003[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 0, 3, 6, 9, 1, 7, 2, 7, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1004[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 8, 0, 6, 3, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1005[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 8, 1, 5, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1006[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 8, 1, 7, 4, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1007[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 1, 2, 1, 7, 5, 8, 2, 7, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder1008[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 1, 2, 1, 8, 5, 8, 2, 7, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder1009[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 1, 2, 7, 9, 2, 8, 3, 7, 0, 1, 2, 5, 0, 2, 2, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1010[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 1, 3, 6, 9, 1, 6, 2, 6, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder1011[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 1, 3, 6, 9, 1, 7, 2, 7, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder1012[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 8, 10, 3, 9, 4, 8, 0, 2, 0, 1, 2, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder1013[] = {5, 9, 12, 14, 0, 5, 1, 4, 2, 3, 8, 10, 3, 9, 4, 8, 1, 2, 4, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder1014[] = {5, 10, 13, 14, 0, 5, 1, 4, 0, 3, 0, 2, 0, 1, 4, 8, 1, 5, 5, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1015[] = {5, 10, 13, 14, 0, 5, 1, 4, 0, 3, 0, 2, 0, 3, 6, 8, 0, 4, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1016[] = {5, 10, 13, 14, 0, 5, 1, 4, 0, 3, 0, 2, 0, 3, 6, 8, 0, 5, 3, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1017[] = {5, 10, 13, 14, 0, 5, 1, 4, 0, 3, 0, 2, 7, 9, 2, 8, 3, 7, 0, 1, 2, 5, 0, 2, 2, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1018[] = {5, 10, 13, 14, 0, 5, 1, 4, 0, 3, 0, 3, 6, 9, 0, 3, 5, 7, 0, 4, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1019[] = {5, 10, 13, 14, 0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 8, 1, 5, 4, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1020[] = {5, 10, 13, 14, 0, 5, 1, 4, 0, 3, 1, 2, 0, 1, 4, 8, 1, 5, 4, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1021[] = {5, 10, 13, 14, 0, 5, 1, 4, 0, 3, 1, 2, 1, 7, 5, 8, 2, 5, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1022[] = {5, 10, 13, 14, 0, 5, 1, 4, 0, 3, 1, 3, 6, 9, 1, 6, 2, 5, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1023[] = {5, 10, 13, 14, 0, 5, 1, 4, 1, 3, 0, 2, 0, 1, 4, 8, 1, 5, 5, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1024[] = {5, 10, 13, 14, 0, 5, 1, 4, 1, 3, 0, 2, 0, 1, 4, 8, 1, 5, 5, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1025[] = {5, 10, 13, 14, 0, 5, 1, 4, 1, 3, 0, 2, 0, 3, 6, 8, 0, 4, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1026[] = {5, 10, 13, 14, 0, 5, 1, 4, 1, 3, 0, 2, 0, 3, 6, 8, 0, 5, 3, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1027[] = {5, 10, 13, 14, 0, 5, 1, 4, 1, 3, 0, 3, 6, 9, 0, 3, 5, 7, 0, 4, 0, 4, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1028[] = {5, 10, 13, 14, 0, 5, 1, 4, 1, 3, 1, 2, 0, 1, 4, 8, 1, 5, 4, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1029[] = {5, 10, 13, 14, 0, 5, 1, 4, 1, 3, 1, 2, 0, 1, 4, 8, 1, 5, 4, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1030[] = {5, 10, 13, 14, 0, 5, 1, 4, 1, 3, 1, 2, 1, 7, 5, 8, 2, 5, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder1031[] = {5, 10, 13, 14, 0, 5, 1, 4, 1, 3, 1, 3, 6, 9, 1, 6, 2, 5, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder1032[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 8, 0, 6, 1, 6, 3, 5, 2, 4, 2, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder1033[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 8, 1, 5, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1034[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 8, 1, 5, 5, 6, 0, 4, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1035[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 0, 2, 0, 1, 4, 8, 1, 7, 4, 6, 0, 3, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1036[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 0, 2, 1, 6, 6, 8, 2, 7, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1037[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 0, 2, 1, 7, 5, 8, 2, 5, 5, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1038[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 0, 2, 1, 7, 5, 8, 2, 7, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1039[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 0, 2, 1, 8, 5, 8, 2, 7, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1040[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 0, 3, 6, 9, 1, 6, 2, 6, 4, 6, 0, 1, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1041[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 0, 3, 6, 9, 1, 7, 2, 7, 4, 6, 0, 1, 3, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1042[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 8, 0, 6, 3, 6, 3, 5, 0, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1043[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 8, 1, 5, 3, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1044[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 1, 2, 0, 1, 4, 8, 1, 7, 4, 6, 0, 3, 2, 4, 0, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1045[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 1, 2, 1, 7, 5, 8, 2, 7, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder1046[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 1, 2, 1, 8, 5, 8, 2, 7, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder1047[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 1, 2, 7, 9, 2, 8, 3, 7, 0, 1, 2, 5, 0, 2, 2, 3, 1, 2, 0, 1};
static const int8_t kQcdTopoOrder1048[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 1, 3, 6, 9, 1, 6, 2, 6, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder1049[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 1, 3, 6, 9, 1, 7, 2, 7, 4, 6, 4, 5, 1, 4, 1, 3, 0, 2, 0, 1};
static const int8_t kQcdTopoOrder1050[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 8, 10, 3, 9, 4, 8, 0, 2, 0, 1, 2, 5, 0, 4, 2, 3, 0, 1, 0, 1};
static const int8_t kQcdTopoOrder1051[] = {5, 10, 13, 14, 0, 5, 1, 4, 2, 3, 8, 10, 3, 9, 4, 8, 1, 2, 4, 6, 1, 5, 1, 4, 0, 3, 0, 1, 0, 1};

static const qcd_topology_order kQcdTopoOrders[] = {
    {kQcdTopoOrder0, 5},
    {kQcdTopoOrder1, 5},
    {kQcdTopoOrder2, 8},
    {kQcdTopoOrder3, 8},
    {kQcdTopoOrder4, 8},
    {kQcdTopoOrder5, 8},
    {kQcdTopoOrder6, 11},
    {kQcdTopoOrder7, 11},
    {kQcdTopoOrder8, 11},
    {kQcdTopoOrder9, 11},
    {kQcdTopoOrder10, 11},
    {kQcdTopoOrder11, 11},
    {kQcdTopoOrder12, 11},
    {kQcdTopoOrder13, 11},
    {kQcdTopoOrder14, 11},
    {kQcdTopoOrder15, 11},
    {kQcdTopoOrder16, 11},
    {kQcdTopoOrder17, 11},
    {kQcdTopoOrder18, 11},
    {kQcdTopoOrder19, 11},
    {kQcdTopoOrder20, 11},
    {kQcdTopoOrder21, 12},
    {kQcdTopoOrder22, 12},
    {kQcdTopoOrder23, 12},
    {kQcdTopoOrder24, 12},
    {kQcdTopoOrder25, 12},
    {kQcdTopoOrder26, 12},
    {kQcdTopoOrder27, 12},
    {kQcdTopoOrder28, 12},
    {kQcdTopoOrder29, 12},
    {kQcdTopoOrder30, 12},
    {kQcdTopoOrder31, 12},
    {kQcdTopoOrder32, 12},
    {kQcdTopoOrder33, 12},
    {kQcdTopoOrder34, 12},
    {kQcdTopoOrder35, 12},
    {kQcdTopoOrder36, 12},
    {kQcdTopoOrder37, 12},
    {kQcdTopoOrder38, 12},
    {kQcdTopoOrder39, 12},
    {kQcdTopoOrder40, 12},
    {kQcdTopoOrder41, 12},
    {kQcdTopoOrder42, 12},
    {kQcdTopoOrder43, 12},
    {kQcdTopoOrder44, 12},
    {kQcdTopoOrder45, 12},
    {kQcdTopoOrder46, 12},
    {kQcdTopoOrder47, 12},
    {kQcdTopoOrder48, 12},
    {kQcdTopoOrder49, 12},
    {kQcdTopoOrder50, 12},
    {kQcdTopoOrder51, 12},
    {kQcdTopoOrder52, 12},
    {kQcdTopoOrder53, 12},
    {kQcdTopoOrder54, 12},
    {kQcdTopoOrder55, 12},
    {kQcdTopoOrder56, 12},
    {kQcdTopoOrder57, 12},
    {kQcdTopoOrder58, 12},
    {kQcdTopoOrder59, 12},
    {kQcdTopoOrder60, 12},
    {kQcdTopoOrder61, 12},
    {kQcdTopoOrder62, 12},
    {kQcdTopoOrder63, 12},
    {kQcdTopoOrder64, 12},
    {kQcdTopoOrder65, 12},
    {kQcdTopoOrder66, 12},
    {kQcdTopoOrder67, 12},
    {kQcdTopoOrder68, 12},
    {kQcdTopoOrder69, 12},
    {kQcdTopoOrder70, 12},
    {kQcdTopoOrder71, 12},
    {kQcdTopoOrder72, 12},
    {kQcdTopoOrder73, 12},
    {kQcdTopoOrder74, 12},
    {kQcdTopoOrder75, 12},
    {kQcdTopoOrder76, 12},
    {kQcdTopoOrder77, 12},
    {kQcdTopoOrder78, 12},
    {kQcdTopoOrder79, 12},
    {kQcdTopoOrder80, 12},
    {kQcdTopoOrder81, 12},
    {kQcdTopoOrder82, 12},
    {kQcdTopoOrder83, 12},
    {kQcdTopoOrder84, 12},
    {kQcdTopoOrder85, 12},
    {kQcdTopoOrder86, 12},
    {kQcdTopoOrder87, 12},
    {kQcdTopoOrder88, 12},
    {kQcdTopoOrder89, 12},
    {kQcdTopoOrder90, 12},
    {kQcdTopoOrder91, 12},
    {kQcdTopoOrder92, 12},
    {kQcdTopoOrder93, 12},
    {kQcdTopoOrder94, 12},
    {kQcdTopoOrder95, 12},
    {kQcdTopoOrder96, 12},
    {kQcdTopoOrder97, 12},
    {kQcdTopoOrder98, 12},
    {kQcdTopoOrder99, 12},
    {kQcdTopoOrder100, 12},
    {kQcdTopoOrder101, 12},
    {kQcdTopoOrder102, 12},
    {kQcdTopoOrder103, 12},
    {kQcdTopoOrder104, 12},
    {kQcdTopoOrder105, 12},
    {kQcdTopoOrder106, 12},
    {kQcdTopoOrder107, 12},
    {kQcdTopoOrder108, 12},
    {kQcdTopoOrder109, 12},
    {kQcdTopoOrder110, 12},
    {kQcdTopoOrder111, 12},
    {kQcdTopoOrder112, 12},
    {kQcdTopoOrder113, 12},
    {kQcdTopoOrder114, 12},
    {kQcdTopoOrder115, 12},
    {kQcdTopoOrder116, 12},
    {kQcdTopoOrder117, 12},
    {kQcdTopoOrder118, 12},
    {kQcdTopoOrder119, 15},
    {kQcdTopoOrder120, 15},
    {kQcdTopoOrder121, 15},
    {kQcdTopoOrder122, 15},
    {kQcdTopoOrder123, 15},
    {kQcdTopoOrder124, 15},
    {kQcdTopoOrder125, 15},
    {kQcdTopoOrder126, 15},
    {kQcdTopoOrder127, 15},
    {kQcdTopoOrder128, 15},
    {kQcdTopoOrder129, 15},
    {kQcdTopoOrder130, 15},
    {kQcdTopoOrder131, 15},
    {kQcdTopoOrder132, 15},
    {kQcdTopoOrder133, 15},
    {kQcdTopoOrder134, 15},
    {kQcdTopoOrder135, 15},
    {kQcdTopoOrder136, 15},
    {kQcdTopoOrder137, 15},
    {kQcdTopoOrder138, 15},
    {kQcdTopoOrder139, 15},
    {kQcdTopoOrder140, 15},
    {kQcdTopoOrder141, 15},
    {kQcdTopoOrder142, 15},
    {kQcdTopoOrder143, 15},
    {kQcdTopoOrder144, 15},
    {kQcdTopoOrder145, 15},
    {kQcdTopoOrder146, 15},
    {kQcdTopoOrder147, 15},
    {kQcdTopoOrder148, 15},
    {kQcdTopoOrder149, 15},
    {kQcdTopoOrder150, 15},
    {kQcdTopoOrder151, 15},
    {kQcdTopoOrder152, 15},
    {kQcdTopoOrder153, 15},
    {kQcdTopoOrder154, 15},
    {kQcdTopoOrder155, 15},
    {kQcdTopoOrder156, 15},
    {kQcdTopoOrder157, 15},
    {kQcdTopoOrder158, 15},
    {kQcdTopoOrder159, 15},
    {kQcdTopoOrder160, 15},
    {kQcdTopoOrder161, 15},
    {kQcdTopoOrder162, 15},
    {kQcdTopoOrder163, 15},
    {kQcdTopoOrder164, 15},
    {kQcdTopoOrder165, 15},
    {kQcdTopoOrder166, 15},
    {kQcdTopoOrder167, 15},
    {kQcdTopoOrder168, 15},
    {kQcdTopoOrder169, 15},
    {kQcdTopoOrder170, 15},
    {kQcdTopoOrder171, 15},
    {kQcdTopoOrder172, 15},
    {kQcdTopoOrder173, 15},
    {kQcdTopoOrder174, 15},
    {kQcdTopoOrder175, 15},
    {kQcdTopoOrder176, 15},
    {kQcdTopoOrder177, 15},
    {kQcdTopoOrder178, 15},
    {kQcdTopoOrder179, 15},
    {kQcdTopoOrder180, 15},
    {kQcdTopoOrder181, 15},
    {kQcdTopoOrder182, 15},
    {kQcdTopoOrder183, 15},
    {kQcdTopoOrder184, 15},
    {kQcdTopoOrder185, 15},
    {kQcdTopoOrder186, 15},
    {kQcdTopoOrder187, 15},
    {kQcdTopoOrder188, 15},
    {kQcdTopoOrder189, 15},
    {kQcdTopoOrder190, 15},
    {kQcdTopoOrder191, 15},
    {kQcdTopoOrder192, 15},
    {kQcdTopoOrder193, 15},
    {kQcdTopoOrder194, 15},
    {kQcdTopoOrder195, 15},
    {kQcdTopoOrder196, 15},
    {kQcdTopoOrder197, 15},
    {kQcdTopoOrder198, 15},
    {kQcdTopoOrder199, 15},
    {kQcdTopoOrder200, 15},
    {kQcdTopoOrder201, 15},
    {kQcdTopoOrder202, 15},
    {kQcdTopoOrder203, 15},
    {kQcdTopoOrder204, 15},
    {kQcdTopoOrder205, 15},
    {kQcdTopoOrder206, 15},
    {kQcdTopoOrder207, 15},
    {kQcdTopoOrder208, 15},
    {kQcdTopoOrder209, 15},
    {kQcdTopoOrder210, 15},
    {kQcdTopoOrder211, 15},
    {kQcdTopoOrder212, 15},
    {kQcdTopoOrder213, 15},
    {kQcdTopoOrder214, 15},
    {kQcdTopoOrder215, 15},
    {kQcdTopoOrder216, 15},
    {kQcdTopoOrder217, 15},
    {kQcdTopoOrder218, 15},
    {kQcdTopoOrder219, 15},
    {kQcdTopoOrder220, 15},
    {kQcdTopoOrder221, 15},
    {kQcdTopoOrder222, 15},
    {kQcdTopoOrder223, 15},
    {kQcdTopoOrder224, 15},
    {kQcdTopoOrder225, 15},
    {kQcdTopoOrder226, 15},
    {kQcdTopoOrder227, 15},
    {kQcdTopoOrder228, 15},
    {kQcdTopoOrder229, 15},
    {kQcdTopoOrder230, 15},
    {kQcdTopoOrder231, 15},
    {kQcdTopoOrder232, 15},
    {kQcdTopoOrder233, 15},
    {kQcdTopoOrder234, 15},
    {kQcdTopoOrder235, 15},
    {kQcdTopoOrder236, 15},
    {kQcdTopoOrder237, 15},
    {kQcdTopoOrder238, 15},
    {kQcdTopoOrder239, 15},
    {kQcdTopoOrder240, 15},
    {kQcdTopoOrder241, 15},
    {kQcdTopoOrder242, 15},
    {kQcdTopoOrder243, 15},
    {kQcdTopoOrder244, 15},
    {kQcdTopoOrder245, 15},
    {kQcdTopoOrder246, 15},
    {kQcdTopoOrder247, 15},
    {kQcdTopoOrder248, 15},
    {kQcdTopoOrder249, 15},
    {kQcdTopoOrder250, 15},
    {kQcdTopoOrder251, 15},
    {kQcdTopoOrder252, 15},
    {kQcdTopoOrder253, 15},
    {kQcdTopoOrder254, 15},
    {kQcdTopoOrder255, 15},
    {kQcdTopoOrder256, 15},
    {kQcdTopoOrder257, 15},
    {kQcdTopoOrder258, 15},
    {kQcdTopoOrder259, 15},
    {kQcdTopoOrder260, 15},
    {kQcdTopoOrder261, 15},
    {kQcdTopoOrder262, 15},
    {kQcdTopoOrder263, 15},
    {kQcdTopoOrder264, 15},
    {kQcdTopoOrder265, 15},
    {kQcdTopoOrder266, 15},
    {kQcdTopoOrder267, 15},
    {kQcdTopoOrder268, 15},
    {kQcdTopoOrder269, 15},
    {kQcdTopoOrder270, 15},
    {kQcdTopoOrder271, 15},
    {kQcdTopoOrder272, 15},
    {kQcdTopoOrder273, 15},
    {kQcdTopoOrder274, 15},
    {kQcdTopoOrder275, 15},
    {kQcdTopoOrder276, 15},
    {kQcdTopoOrder277, 15},
    {kQcdTopoOrder278, 15},
    {kQcdTopoOrder279, 15},
    {kQcdTopoOrder280, 15},
    {kQcdTopoOrder281, 15},
    {kQcdTopoOrder282, 15},
    {kQcdTopoOrder283, 15},
    {kQcdTopoOrder284, 15},
    {kQcdTopoOrder285, 15},
    {kQcdTopoOrder286, 15},
    {kQcdTopoOrder287, 15},
    {kQcdTopoOrder288, 15},
    {kQcdTopoOrder289, 15},
    {kQcdTopoOrder290, 15},
    {kQcdTopoOrder291, 15},
    {kQcdTopoOrder292, 15},
    {kQcdTopoOrder293, 15},
    {kQcdTopoOrder294, 15},
    {kQcdTopoOrder295, 15},
    {kQcdTopoOrder296, 15},
    {kQcdTopoOrder297, 15},
    {kQcdTopoOrder298, 15},
    {kQcdTopoOrder299, 15},
    {kQcdTopoOrder300, 15},
    {kQcdTopoOrder301, 15},
    {kQcdTopoOrder302, 15},
    {kQcdTopoOrder303, 15},
    {kQcdTopoOrder304, 15},
    {kQcdTopoOrder305, 15},
    {kQcdTopoOrder306, 15},
    {kQcdTopoOrder307, 15},
    {kQcdTopoOrder308, 15},
    {kQcdTopoOrder309, 15},
    {kQcdTopoOrder310, 15},
    {kQcdTopoOrder311, 15},
    {kQcdTopoOrder312, 15},
    {kQcdTopoOrder313, 15},
    {kQcdTopoOrder314, 15},
    {kQcdTopoOrder315, 15},
    {kQcdTopoOrder316, 15},
    {kQcdTopoOrder317, 15},
    {kQcdTopoOrder318, 15},
    {kQcdTopoOrder319, 15},
    {kQcdTopoOrder320, 15},
    {kQcdTopoOrder321, 15},
    {kQcdTopoOrder322, 15},
    {kQcdTopoOrder323, 15},
    {kQcdTopoOrder324, 15},
    {kQcdTopoOrder325, 15},
    {kQcdTopoOrder326, 15},
    {kQcdTopoOrder327, 15},
    {kQcdTopoOrder328, 15},
    {kQcdTopoOrder329, 15},
    {kQcdTopoOrder330, 15},
    {kQcdTopoOrder331, 15},
    {kQcdTopoOrder332, 15},
    {kQcdTopoOrder333, 15},
    {kQcdTopoOrder334, 15},
    {kQcdTopoOrder335, 15},
    {kQcdTopoOrder336, 15},
    {kQcdTopoOrder337, 15},
    {kQcdTopoOrder338, 15},
    {kQcdTopoOrder339, 15},
    {kQcdTopoOrder340, 15},
    {kQcdTopoOrder341, 15},
    {kQcdTopoOrder342, 15},
    {kQcdTopoOrder343, 15},
    {kQcdTopoOrder344, 15},
    {kQcdTopoOrder345, 15},
    {kQcdTopoOrder346, 15},
    {kQcdTopoOrder347, 15},
    {kQcdTopoOrder348, 15},
    {kQcdTopoOrder349, 15},
    {kQcdTopoOrder350, 15},
    {kQcdTopoOrder351, 15},
    {kQcdTopoOrder352, 15},
    {kQcdTopoOrder353, 15},
    {kQcdTopoOrder354, 15},
    {kQcdTopoOrder355, 15},
    {kQcdTopoOrder356, 15},
    {kQcdTopoOrder357, 15},
    {kQcdTopoOrder358, 15},
    {kQcdTopoOrder359, 15},
    {kQcdTopoOrder360, 15},
    {kQcdTopoOrder361, 15},
    {kQcdTopoOrder362, 15},
    {kQcdTopoOrder363, 15},
    {kQcdTopoOrder364, 15},
    {kQcdTopoOrder365, 15},
    {kQcdTopoOrder366, 15},
    {kQcdTopoOrder367, 15},
    {kQcdTopoOrder368, 15},
    {kQcdTopoOrder369, 15},
    {kQcdTopoOrder370, 15},
    {kQcdTopoOrder371, 15},
    {kQcdTopoOrder372, 15},
    {kQcdTopoOrder373, 15},
    {kQcdTopoOrder374, 15},
    {kQcdTopoOrder375, 15},
    {kQcdTopoOrder376, 15},
    {kQcdTopoOrder377, 15},
    {kQcdTopoOrder378, 15},
    {kQcdTopoOrder379, 15},
    {kQcdTopoOrder380, 15},
    {kQcdTopoOrder381, 15},
    {kQcdTopoOrder382, 15},
    {kQcdTopoOrder383, 15},
    {kQcdTopoOrder384, 15},
    {kQcdTopoOrder385, 15},
    {kQcdTopoOrder386, 15},
    {kQcdTopoOrder387, 15},
    {kQcdTopoOrder388, 15},
    {kQcdTopoOrder389, 15},
    {kQcdTopoOrder390, 15},
    {kQcdTopoOrder391, 15},
    {kQcdTopoOrder392, 15},
    {kQcdTopoOrder393, 15},
    {kQcdTopoOrder394, 15},
    {kQcdTopoOrder395, 15},
    {kQcdTopoOrder396, 15},
    {kQcdTopoOrder397, 15},
    {kQcdTopoOrder398, 15},
    {kQcdTopoOrder399, 15},
    {kQcdTopoOrder400, 15},
    {kQcdTopoOrder401, 15},
    {kQcdTopoOrder402, 15},
    {kQcdTopoOrder403, 15},
    {kQcdTopoOrder404, 15},
    {kQcdTopoOrder405, 15},
    {kQcdTopoOrder406, 15},
    {kQcdTopoOrder407, 15},
    {kQcdTopoOrder408, 15},
    {kQcdTopoOrder409, 15},
    {kQcdTopoOrder410, 15},
    {kQcdTopoOrder411, 15},
    {kQcdTopoOrder412, 15},
    {kQcdTopoOrder413, 15},
    {kQcdTopoOrder414, 15},
    {kQcdTopoOrder415, 15},
    {kQcdTopoOrder416, 15},
    {kQcdTopoOrder417, 15},
    {kQcdTopoOrder418, 15},
    {kQcdTopoOrder419, 15},
    {kQcdTopoOrder420, 15},
    {kQcdTopoOrder421, 15},
    {kQcdTopoOrder422, 15},
    {kQcdTopoOrder423, 15},
    {kQcdTopoOrder424, 15},
    {kQcdTopoOrder425, 15},
    {kQcdTopoOrder426, 15},
    {kQcdTopoOrder427, 15},
    {kQcdTopoOrder428, 15},
    {kQcdTopoOrder429, 15},
    {kQcdTopoOrder430, 15},
    {kQcdTopoOrder431, 15},
    {kQcdTopoOrder432, 15},
    {kQcdTopoOrder433, 15},
    {kQcdTopoOrder434, 15},
    {kQcdTopoOrder435, 15},
    {kQcdTopoOrder436, 15},
    {kQcdTopoOrder437, 15},
    {kQcdTopoOrder438, 15},
    {kQcdTopoOrder439, 15},
    {kQcdTopoOrder440, 15},
    {kQcdTopoOrder441, 15},
    {kQcdTopoOrder442, 15},
    {kQcdTopoOrder443, 15},
    {kQcdTopoOrder444, 15},
    {kQcdTopoOrder445, 15},
    {kQcdTopoOrder446, 15},
    {kQcdTopoOrder447, 15},
    {kQcdTopoOrder448, 15},
    {kQcdTopoOrder449, 15},
    {kQcdTopoOrder450, 15},
    {kQcdTopoOrder451, 15},
    {kQcdTopoOrder452, 15},
    {kQcdTopoOrder453, 15},
    {kQcdTopoOrder454, 15},
    {kQcdTopoOrder455, 15},
    {kQcdTopoOrder456, 15},
    {kQcdTopoOrder457, 15},
    {kQcdTopoOrder458, 15},
    {kQcdTopoOrder459, 15},
    {kQcdTopoOrder460, 15},
    {kQcdTopoOrder461, 15},
    {kQcdTopoOrder462, 15},
    {kQcdTopoOrder463, 15},
    {kQcdTopoOrder464, 15},
    {kQcdTopoOrder465, 15},
    {kQcdTopoOrder466, 15},
    {kQcdTopoOrder467, 15},
    {kQcdTopoOrder468, 15},
    {kQcdTopoOrder469, 15},
    {kQcdTopoOrder470, 15},
    {kQcdTopoOrder471, 15},
    {kQcdTopoOrder472, 15},
    {kQcdTopoOrder473, 15},
    {kQcdTopoOrder474, 15},
    {kQcdTopoOrder475, 15},
    {kQcdTopoOrder476, 15},
    {kQcdTopoOrder477, 15},
    {kQcdTopoOrder478, 15},
    {kQcdTopoOrder479, 15},
    {kQcdTopoOrder480, 15},
    {kQcdTopoOrder481, 15},
    {kQcdTopoOrder482, 15},
    {kQcdTopoOrder483, 15},
    {kQcdTopoOrder484, 15},
    {kQcdTopoOrder485, 15},
    {kQcdTopoOrder486, 15},
    {kQcdTopoOrder487, 15},
    {kQcdTopoOrder488, 15},
    {kQcdTopoOrder489, 15},
    {kQcdTopoOrder490, 15},
    {kQcdTopoOrder491, 15},
    {kQcdTopoOrder492, 15},
    {kQcdTopoOrder493, 15},
    {kQcdTopoOrder494, 15},
    {kQcdTopoOrder495, 15},
    {kQcdTopoOrder496, 15},
    {kQcdTopoOrder497, 15},
    {kQcdTopoOrder498, 15},
    {kQcdTopoOrder499, 15},
    {kQcdTopoOrder500, 15},
    {kQcdTopoOrder501, 15},
    {kQcdTopoOrder502, 15},
    {kQcdTopoOrder503, 15},
    {kQcdTopoOrder504, 15},
    {kQcdTopoOrder505, 15},
    {kQcdTopoOrder506, 15},
    {kQcdTopoOrder507, 15},
    {kQcdTopoOrder508, 15},
    {kQcdTopoOrder509, 15},
    {kQcdTopoOrder510, 15},
    {kQcdTopoOrder511, 15},
    {kQcdTopoOrder512, 15},
    {kQcdTopoOrder513, 15},
    {kQcdTopoOrder514, 15},
    {kQcdTopoOrder515, 15},
    {kQcdTopoOrder516, 15},
    {kQcdTopoOrder517, 15},
    {kQcdTopoOrder518, 15},
    {kQcdTopoOrder519, 15},
    {kQcdTopoOrder520, 15},
    {kQcdTopoOrder521, 15},
    {kQcdTopoOrder522, 15},
    {kQcdTopoOrder523, 15},
    {kQcdTopoOrder524, 15},
    {kQcdTopoOrder525, 15},
    {kQcdTopoOrder526, 15},
    {kQcdTopoOrder527, 15},
    {kQcdTopoOrder528, 15},
    {kQcdTopoOrder529, 15},
    {kQcdTopoOrder530, 15},
    {kQcdTopoOrder531, 15},
    {kQcdTopoOrder532, 15},
    {kQcdTopoOrder533, 15},
    {kQcdTopoOrder534, 15},
    {kQcdTopoOrder535, 15},
    {kQcdTopoOrder536, 15},
    {kQcdTopoOrder537, 15},
    {kQcdTopoOrder538, 15},
    {kQcdTopoOrder539, 15},
    {kQcdTopoOrder540, 15},
    {kQcdTopoOrder541, 15},
    {kQcdTopoOrder542, 15},
    {kQcdTopoOrder543, 15},
    {kQcdTopoOrder544, 15},
    {kQcdTopoOrder545, 15},
    {kQcdTopoOrder546, 15},
    {kQcdTopoOrder547, 15},
    {kQcdTopoOrder548, 15},
    {kQcdTopoOrder549, 15},
    {kQcdTopoOrder550, 15},
    {kQcdTopoOrder551, 15},
    {kQcdTopoOrder552, 15},
    {kQcdTopoOrder553, 15},
    {kQcdTopoOrder554, 15},
    {kQcdTopoOrder555, 15},
    {kQcdTopoOrder556, 15},
    {kQcdTopoOrder557, 15},
    {kQcdTopoOrder558, 15},
    {kQcdTopoOrder559, 15},
    {kQcdTopoOrder560, 15},
    {kQcdTopoOrder561, 15},
    {kQcdTopoOrder562, 15},
    {kQcdTopoOrder563, 15},
    {kQcdTopoOrder564, 15},
    {kQcdTopoOrder565, 15},
    {kQcdTopoOrder566, 15},
    {kQcdTopoOrder567, 15},
    {kQcdTopoOrder568, 15},
    {kQcdTopoOrder569, 15},
    {kQcdTopoOrder570, 15},
    {kQcdTopoOrder571, 15},
    {kQcdTopoOrder572, 15},
    {kQcdTopoOrder573, 15},
    {kQcdTopoOrder574, 15},
    {kQcdTopoOrder575, 15},
    {kQcdTopoOrder576, 15},
    {kQcdTopoOrder577, 15},
    {kQcdTopoOrder578, 15},
    {kQcdTopoOrder579, 15},
    {kQcdTopoOrder580, 15},
    {kQcdTopoOrder581, 15},
    {kQcdTopoOrder582, 15},
    {kQcdTopoOrder583, 15},
    {kQcdTopoOrder584, 15},
    {kQcdTopoOrder585, 15},
    {kQcdTopoOrder586, 15},
    {kQcdTopoOrder587, 15},
    {kQcdTopoOrder588, 15},
    {kQcdTopoOrder589, 15},
    {kQcdTopoOrder590, 15},
    {kQcdTopoOrder591, 15},
    {kQcdTopoOrder592, 15},
    {kQcdTopoOrder593, 15},
    {kQcdTopoOrder594, 15},
    {kQcdTopoOrder595, 15},
    {kQcdTopoOrder596, 15},
    {kQcdTopoOrder597, 15},
    {kQcdTopoOrder598, 15},
    {kQcdTopoOrder599, 15},
    {kQcdTopoOrder600, 15},
    {kQcdTopoOrder601, 15},
    {kQcdTopoOrder602, 15},
    {kQcdTopoOrder603, 15},
    {kQcdTopoOrder604, 15},
    {kQcdTopoOrder605, 15},
    {kQcdTopoOrder606, 15},
    {kQcdTopoOrder607, 15},
    {kQcdTopoOrder608, 15},
    {kQcdTopoOrder609, 15},
    {kQcdTopoOrder610, 15},
    {kQcdTopoOrder611, 15},
    {kQcdTopoOrder612, 15},
    {kQcdTopoOrder613, 15},
    {kQcdTopoOrder614, 15},
    {kQcdTopoOrder615, 15},
    {kQcdTopoOrder616, 15},
    {kQcdTopoOrder617, 15},
    {kQcdTopoOrder618, 15},
    {kQcdTopoOrder619, 15},
    {kQcdTopoOrder620, 15},
    {kQcdTopoOrder621, 15},
    {kQcdTopoOrder622, 15},
    {kQcdTopoOrder623, 15},
    {kQcdTopoOrder624, 15},
    {kQcdTopoOrder625, 15},
    {kQcdTopoOrder626, 15},
    {kQcdTopoOrder627, 15},
    {kQcdTopoOrder628, 15},
    {kQcdTopoOrder629, 15},
    {kQcdTopoOrder630, 15},
    {kQcdTopoOrder631, 15},
    {kQcdTopoOrder632, 15},
    {kQcdTopoOrder633, 15},
    {kQcdTopoOrder634, 15},
    {kQcdTopoOrder635, 15},
    {kQcdTopoOrder636, 15},
    {kQcdTopoOrder637, 15},
    {kQcdTopoOrder638, 15},
    {kQcdTopoOrder639, 15},
    {kQcdTopoOrder640, 15},
    {kQcdTopoOrder641, 15},
    {kQcdTopoOrder642, 15},
    {kQcdTopoOrder643, 15},
    {kQcdTopoOrder644, 15},
    {kQcdTopoOrder645, 15},
    {kQcdTopoOrder646, 15},
    {kQcdTopoOrder647, 15},
    {kQcdTopoOrder648, 15},
    {kQcdTopoOrder649, 15},
    {kQcdTopoOrder650, 15},
    {kQcdTopoOrder651, 15},
    {kQcdTopoOrder652, 15},
    {kQcdTopoOrder653, 15},
    {kQcdTopoOrder654, 15},
    {kQcdTopoOrder655, 15},
    {kQcdTopoOrder656, 15},
    {kQcdTopoOrder657, 6},
    {kQcdTopoOrder658, 6},
    {kQcdTopoOrder659, 9},
    {kQcdTopoOrder660, 9},
    {kQcdTopoOrder661, 9},
    {kQcdTopoOrder662, 9},
    {kQcdTopoOrder663, 9},
    {kQcdTopoOrder664, 9},
    {kQcdTopoOrder665, 9},
    {kQcdTopoOrder666, 9},
    {kQcdTopoOrder667, 9},
    {kQcdTopoOrder668, 8},
    {kQcdTopoOrder669, 12},
    {kQcdTopoOrder670, 12},
    {kQcdTopoOrder671, 12},
    {kQcdTopoOrder672, 12},
    {kQcdTopoOrder673, 12},
    {kQcdTopoOrder674, 12},
    {kQcdTopoOrder675, 12},
    {kQcdTopoOrder676, 12},
    {kQcdTopoOrder677, 12},
    {kQcdTopoOrder678, 12},
    {kQcdTopoOrder679, 12},
    {kQcdTopoOrder680, 12},
    {kQcdTopoOrder681, 12},
    {kQcdTopoOrder682, 12},
    {kQcdTopoOrder683, 12},
    {kQcdTopoOrder684, 12},
    {kQcdTopoOrder685, 12},
    {kQcdTopoOrder686, 12},
    {kQcdTopoOrder687, 12},
    {kQcdTopoOrder688, 12},
    {kQcdTopoOrder689, 12},
    {kQcdTopoOrder690, 12},
    {kQcdTopoOrder691, 12},
    {kQcdTopoOrder692, 12},
    {kQcdTopoOrder693, 12},
    {kQcdTopoOrder694, 12},
    {kQcdTopoOrder695, 12},
    {kQcdTopoOrder696, 12},
    {kQcdTopoOrder697, 12},
    {kQcdTopoOrder698, 12},
    {kQcdTopoOrder699, 12},
    {kQcdTopoOrder700, 12},
    {kQcdTopoOrder701, 12},
    {kQcdTopoOrder702, 12},
    {kQcdTopoOrder703, 12},
    {kQcdTopoOrder704, 12},
    {kQcdTopoOrder705, 12},
    {kQcdTopoOrder706, 12},
    {kQcdTopoOrder707, 11},
    {kQcdTopoOrder708, 11},
    {kQcdTopoOrder709, 11},
    {kQcdTopoOrder710, 11},
    {kQcdTopoOrder711, 13},
    {kQcdTopoOrder712, 13},
    {kQcdTopoOrder713, 13},
    {kQcdTopoOrder714, 13},
    {kQcdTopoOrder715, 13},
    {kQcdTopoOrder716, 13},
    {kQcdTopoOrder717, 13},
    {kQcdTopoOrder718, 13},
    {kQcdTopoOrder719, 13},
    {kQcdTopoOrder720, 13},
    {kQcdTopoOrder721, 13},
    {kQcdTopoOrder722, 13},
    {kQcdTopoOrder723, 13},
    {kQcdTopoOrder724, 13},
    {kQcdTopoOrder725, 13},
    {kQcdTopoOrder726, 13},
    {kQcdTopoOrder727, 13},
    {kQcdTopoOrder728, 13},
    {kQcdTopoOrder729, 13},
    {kQcdTopoOrder730, 13},
    {kQcdTopoOrder731, 13},
    {kQcdTopoOrder732, 13},
    {kQcdTopoOrder733, 13},
    {kQcdTopoOrder734, 13},
    {kQcdTopoOrder735, 13},
    {kQcdTopoOrder736, 13},
    {kQcdTopoOrder737, 13},
    {kQcdTopoOrder738, 13},
    {kQcdTopoOrder739, 13},
    {kQcdTopoOrder740, 13},
    {kQcdTopoOrder741, 13},
    {kQcdTopoOrder742, 13},
    {kQcdTopoOrder743, 13},
    {kQcdTopoOrder744, 13},
    {kQcdTopoOrder745, 13},
    {kQcdTopoOrder746, 13},
    {kQcdTopoOrder747, 13},
    {kQcdTopoOrder748, 13},
    {kQcdTopoOrder749, 13},
    {kQcdTopoOrder750, 13},
    {kQcdTopoOrder751, 13},
    {kQcdTopoOrder752, 13},
    {kQcdTopoOrder753, 13},
    {kQcdTopoOrder754, 13},
    {kQcdTopoOrder755, 13},
    {kQcdTopoOrder756, 13},
    {kQcdTopoOrder757, 13},
    {kQcdTopoOrder758, 13},
    {kQcdTopoOrder759, 13},
    {kQcdTopoOrder760, 13},
    {kQcdTopoOrder761, 13},
    {kQcdTopoOrder762, 13},
    {kQcdTopoOrder763, 13},
    {kQcdTopoOrder764, 13},
    {kQcdTopoOrder765, 13},
    {kQcdTopoOrder766, 13},
    {kQcdTopoOrder767, 13},
    {kQcdTopoOrder768, 13},
    {kQcdTopoOrder769, 13},
    {kQcdTopoOrder770, 13},
    {kQcdTopoOrder771, 13},
    {kQcdTopoOrder772, 13},
    {kQcdTopoOrder773, 13},
    {kQcdTopoOrder774, 13},
    {kQcdTopoOrder775, 13},
    {kQcdTopoOrder776, 13},
    {kQcdTopoOrder777, 13},
    {kQcdTopoOrder778, 13},
    {kQcdTopoOrder779, 13},
    {kQcdTopoOrder780, 13},
    {kQcdTopoOrder781, 13},
    {kQcdTopoOrder782, 13},
    {kQcdTopoOrder783, 13},
    {kQcdTopoOrder784, 13},
    {kQcdTopoOrder785, 13},
    {kQcdTopoOrder786, 13},
    {kQcdTopoOrder787, 13},
    {kQcdTopoOrder788, 13},
    {kQcdTopoOrder789, 13},
    {kQcdTopoOrder790, 13},
    {kQcdTopoOrder791, 13},
    {kQcdTopoOrder792, 13},
    {kQcdTopoOrder793, 13},
    {kQcdTopoOrder794, 13},
    {kQcdTopoOrder795, 13},
    {kQcdTopoOrder796, 13},
    {kQcdTopoOrder797, 13},
    {kQcdTopoOrder798, 13},
    {kQcdTopoOrder799, 13},
    {kQcdTopoOrder800, 13},
    {kQcdTopoOrder801, 13},
    {kQcdTopoOrder802, 13},
    {kQcdTopoOrder803, 13},
    {kQcdTopoOrder804, 13},
    {kQcdTopoOrder805, 13},
    {kQcdTopoOrder806, 13},
    {kQcdTopoOrder807, 13},
    {kQcdTopoOrder808, 13},
    {kQcdTopoOrder809, 13},
    {kQcdTopoOrder810, 13},
    {kQcdTopoOrder811, 13},
    {kQcdTopoOrder812, 13},
    {kQcdTopoOrder813, 13},
    {kQcdTopoOrder814, 13},
    {kQcdTopoOrder815, 13},
    {kQcdTopoOrder816, 13},
    {kQcdTopoOrder817, 13},
    {kQcdTopoOrder818, 13},
    {kQcdTopoOrder819, 13},
    {kQcdTopoOrder820, 13},
    {kQcdTopoOrder821, 13},
    {kQcdTopoOrder822, 13},
    {kQcdTopoOrder823, 13},
    {kQcdTopoOrder824, 13},
    {kQcdTopoOrder825, 13},
    {kQcdTopoOrder826, 13},
    {kQcdTopoOrder827, 13},
    {kQcdTopoOrder828, 13},
    {kQcdTopoOrder829, 13},
    {kQcdTopoOrder830, 13},
    {kQcdTopoOrder831, 13},
    {kQcdTopoOrder832, 13},
    {kQcdTopoOrder833, 13},
    {kQcdTopoOrder834, 13},
    {kQcdTopoOrder835, 13},
    {kQcdTopoOrder836, 8},
    {kQcdTopoOrder837, 11},
    {kQcdTopoOrder838, 11},
    {kQcdTopoOrder839, 11},
    {kQcdTopoOrder840, 9},
    {kQcdTopoOrder841, 9},
    {kQcdTopoOrder842, 12},
    {kQcdTopoOrder843, 12},
    {kQcdTopoOrder844, 12},
    {kQcdTopoOrder845, 12},
    {kQcdTopoOrder846, 12},
    {kQcdTopoOrder847, 12},
    {kQcdTopoOrder848, 12},
    {kQcdTopoOrder849, 12},
    {kQcdTopoOrder850, 12},
    {kQcdTopoOrder851, 12},
    {kQcdTopoOrder852, 12},
    {kQcdTopoOrder853, 11},
    {kQcdTopoOrder854, 11},
    {kQcdTopoOrder855, 12},
    {kQcdTopoOrder856, 12},
    {kQcdTopoOrder857, 12},
    {kQcdTopoOrder858, 12},
    {kQcdTopoOrder859, 12},
    {kQcdTopoOrder860, 12},
    {kQcdTopoOrder861, 12},
    {kQcdTopoOrder862, 12},
    {kQcdTopoOrder863, 12},
    {kQcdTopoOrder864, 12},
    {kQcdTopoOrder865, 12},
    {kQcdTopoOrder866, 12},
    {kQcdTopoOrder867, 12},
    {kQcdTopoOrder868, 12},
    {kQcdTopoOrder869, 12},
    {kQcdTopoOrder870, 12},
    {kQcdTopoOrder871, 12},
    {kQcdTopoOrder872, 12},
    {kQcdTopoOrder873, 12},
    {kQcdTopoOrder874, 12},
    {kQcdTopoOrder875, 12},
    {kQcdTopoOrder876, 12},
    {kQcdTopoOrder877, 12},
    {kQcdTopoOrder878, 12},
    {kQcdTopoOrder879, 12},
    {kQcdTopoOrder880, 12},
    {kQcdTopoOrder881, 12},
    {kQcdTopoOrder882, 15},
    {kQcdTopoOrder883, 15},
    {kQcdTopoOrder884, 15},
    {kQcdTopoOrder885, 15},
    {kQcdTopoOrder886, 15},
    {kQcdTopoOrder887, 15},
    {kQcdTopoOrder888, 15},
    {kQcdTopoOrder889, 15},
    {kQcdTopoOrder890, 15},
    {kQcdTopoOrder891, 15},
    {kQcdTopoOrder892, 15},
    {kQcdTopoOrder893, 15},
    {kQcdTopoOrder894, 15},
    {kQcdTopoOrder895, 15},
    {kQcdTopoOrder896, 15},
    {kQcdTopoOrder897, 15},
    {kQcdTopoOrder898, 15},
    {kQcdTopoOrder899, 15},
    {kQcdTopoOrder900, 15},
    {kQcdTopoOrder901, 15},
    {kQcdTopoOrder902, 15},
    {kQcdTopoOrder903, 15},
    {kQcdTopoOrder904, 15},
    {kQcdTopoOrder905, 15},
    {kQcdTopoOrder906, 15},
    {kQcdTopoOrder907, 15},
    {kQcdTopoOrder908, 15},
    {kQcdTopoOrder909, 15},
    {kQcdTopoOrder910, 15},
    {kQcdTopoOrder911, 15},
    {kQcdTopoOrder912, 15},
    {kQcdTopoOrder913, 15},
    {kQcdTopoOrder914, 15},
    {kQcdTopoOrder915, 15},
    {kQcdTopoOrder916, 15},
    {kQcdTopoOrder917, 15},
    {kQcdTopoOrder918, 15},
    {kQcdTopoOrder919, 15},
    {kQcdTopoOrder920, 15},
    {kQcdTopoOrder921, 15},
    {kQcdTopoOrder922, 15},
    {kQcdTopoOrder923, 15},
    {kQcdTopoOrder924, 15},
    {kQcdTopoOrder925, 15},
    {kQcdTopoOrder926, 15},
    {kQcdTopoOrder927, 15},
    {kQcdTopoOrder928, 15},
    {kQcdTopoOrder929, 15},
    {kQcdTopoOrder930, 15},
    {kQcdTopoOrder931, 15},
    {kQcdTopoOrder932, 15},
    {kQcdTopoOrder933, 15},
    {kQcdTopoOrder934, 15},
    {kQcdTopoOrder935, 15},
    {kQcdTopoOrder936, 15},
    {kQcdTopoOrder937, 15},
    {kQcdTopoOrder938, 15},
    {kQcdTopoOrder939, 15},
    {kQcdTopoOrder940, 15},
    {kQcdTopoOrder941, 15},
    {kQcdTopoOrder942, 15},
    {kQcdTopoOrder943, 15},
    {kQcdTopoOrder944, 15},
    {kQcdTopoOrder945, 15},
    {kQcdTopoOrder946, 15},
    {kQcdTopoOrder947, 15},
    {kQcdTopoOrder948, 15},
    {kQcdTopoOrder949, 15},
    {kQcdTopoOrder950, 15},
    {kQcdTopoOrder951, 15},
    {kQcdTopoOrder952, 15},
    {kQcdTopoOrder953, 15},
    {kQcdTopoOrder954, 15},
    {kQcdTopoOrder955, 15},
    {kQcdTopoOrder956, 15},
    {kQcdTopoOrder957, 15},
    {kQcdTopoOrder958, 15},
    {kQcdTopoOrder959, 15},
    {kQcdTopoOrder960, 15},
    {kQcdTopoOrder961, 15},
    {kQcdTopoOrder962, 15},
    {kQcdTopoOrder963, 15},
    {kQcdTopoOrder964, 15},
    {kQcdTopoOrder965, 15},
    {kQcdTopoOrder966, 15},
    {kQcdTopoOrder967, 15},
    {kQcdTopoOrder968, 15},
    {kQcdTopoOrder969, 15},
    {kQcdTopoOrder970, 15},
    {kQcdTopoOrder971, 15},
    {kQcdTopoOrder972, 15},
    {kQcdTopoOrder973, 15},
    {kQcdTopoOrder974, 15},
    {kQcdTopoOrder975, 15},
    {kQcdTopoOrder976, 15},
    {kQcdTopoOrder977, 15},
    {kQcdTopoOrder978, 15},
    {kQcdTopoOrder979, 15},
    {kQcdTopoOrder980, 15},
    {kQcdTopoOrder981, 15},
    {kQcdTopoOrder982, 15},
    {kQcdTopoOrder983, 15},
    {kQcdTopoOrder984, 15},
    {kQcdTopoOrder985, 15},
    {kQcdTopoOrder986, 15},
    {kQcdTopoOrder987, 15},
    {kQcdTopoOrder988, 15},
    {kQcdTopoOrder989, 15},
    {kQcdTopoOrder990, 15},
    {kQcdTopoOrder991, 15},
    {kQcdTopoOrder992, 15},
    {kQcdTopoOrder993, 15},
    {kQcdTopoOrder994, 15},
    {kQcdTopoOrder995, 15},
    {kQcdTopoOrder996, 15},
    {kQcdTopoOrder997, 15},
    {kQcdTopoOrder998, 15},
    {kQcdTopoOrder999, 15},
    {kQcdTopoOrder1000, 15},
    {kQcdTopoOrder1001, 15},
    {kQcdTopoOrder1002, 15},
    {kQcdTopoOrder1003, 15},
    {kQcdTopoOrder1004, 15},
    {kQcdTopoOrder1005, 15},
    {kQcdTopoOrder1006, 15},
    {kQcdTopoOrder1007, 15},
    {kQcdTopoOrder1008, 15},
    {kQcdTopoOrder1009, 15},
    {kQcdTopoOrder1010, 15},
    {kQcdTopoOrder1011, 15},
    {kQcdTopoOrder1012, 15},
    {kQcdTopoOrder1013, 15},
    {kQcdTopoOrder1014, 15},
    {kQcdTopoOrder1015, 15},
    {kQcdTopoOrder1016, 15},
    {kQcdTopoOrder1017, 15},
    {kQcdTopoOrder1018, 15},
    {kQcdTopoOrder1019, 15},
    {kQcdTopoOrder1020, 15},
    {kQcdTopoOrder1021, 15},
    {kQcdTopoOrder1022, 15},
    {kQcdTopoOrder1023, 15},
    {kQcdTopoOrder1024, 15},
    {kQcdTopoOrder1025, 15},
    {kQcdTopoOrder1026, 15},
    {kQcdTopoOrder1027, 15},
    {kQcdTopoOrder1028, 15},
    {kQcdTopoOrder1029, 15},
    {kQcdTopoOrder1030, 15},
    {kQcdTopoOrder1031, 15},
    {kQcdTopoOrder1032, 15},
    {kQcdTopoOrder1033, 15},
    {kQcdTopoOrder1034, 15},
    {kQcdTopoOrder1035, 15},
    {kQcdTopoOrder1036, 15},
    {kQcdTopoOrder1037, 15},
    {kQcdTopoOrder1038, 15},
    {kQcdTopoOrder1039, 15},
    {kQcdTopoOrder1040, 15},
    {kQcdTopoOrder1041, 15},
    {kQcdTopoOrder1042, 15},
    {kQcdTopoOrder1043, 15},
    {kQcdTopoOrder1044, 15},
    {kQcdTopoOrder1045, 15},
    {kQcdTopoOrder1046, 15},
    {kQcdTopoOrder1047, 15},
    {kQcdTopoOrder1048, 15},
    {kQcdTopoOrder1049, 15},
    {kQcdTopoOrder1050, 15},
    {kQcdTopoOrder1051, 15},
};

struct qcd_topology_entry {
    uint64_t key;      /* 签名摘要前 8 字节（大端）；表按此升序 */
    uint64_t fnv;      /* 规范标签流的 FNV-1a 指纹 */
    uint16_t order;    /* kQcdTopoOrders 下标 */
    uint8_t arity;     /* 操作数个数 */
    uint8_t reserved;
};

static const qcd_topology_entry kQcdTopoEntries[] = {
    {0x000c3dab73877cb3ULL, 0x2ac1b826c8377b86ULL, 671, 13, 0},
    {0x0023ae4825b58942ULL, 0x3e3b96b3b5057ddaULL, 739, 14, 0},
    {0x003390c45a0f6a2aULL, 0xf5c5a3aceab247faULL, 175, 16, 0},
    {0x004cd72c41545898ULL, 0xe7fdab8c5f868eccULL, 741, 14, 0},
    {0x0056cc7169b9c54dULL, 0xfaf229e2723641a2ULL, 735, 14, 0},
    {0x006ea58fc5ce9528ULL, 0x6a3ce4fc22299d18ULL, 403, 16, 0},
    {0x006ec36832e98433ULL, 0xe59273becfe43742ULL, 993, 16, 0},
    {0x0093c6be4d0ba18dULL, 0xe4e28e9fc7c78548ULL, 442, 16, 0},
    {0x009446acfee7f88cULL, 0x90054d977e87c6feULL, 674, 13, 0},
    {0x00e4db8aa87e6983ULL, 0x5196a1f097b26752ULL, 983, 16, 0},
    {0x00e8d1ae2d3839a1ULL, 0x51b116fbd8e08e92ULL, 173, 16, 0},
    {0x01203573b8ae40e3ULL, 0x9d03482382f7a41cULL, 98, 13, 0},
    {0x014d986ff50d7ad3ULL, 0x2f3b57f90101c61aULL, 834, 14, 0},
    {0x0164d4c5548cd4d9ULL, 0xe4be49c8c31f70f0ULL, 375, 16, 0},
    {0x0174eef6800866c9ULL, 0xe1988693ec45c124ULL, 876, 13, 0},
    {0x018f62dc724a85dbULL, 0x91521f35021f1c36ULL, 709, 12, 0},
    {0x01a4df6d8975214eULL, 0x0047449d937cdf52ULL, 806, 14, 0},
    {0x01b00334bb26eac9ULL, 0xce450aaac1f67860ULL, 592, 16, 0},
    {0x01c2aa5d5e4a7630ULL, 0xf41cdff6f50f81d2ULL, 620, 16, 0},
    {0x01e8d82f0cf0fd52ULL, 0xc671457a1b8cc5baULL, 834, 14, 0},
    {0x01f15058ca564ca1ULL, 0xd6cdf5fcebc04a58ULL, 883, 16, 0},
    {0x01f248b37dc340f7ULL, 0xb88b181e2c679162ULL, 749, 14, 0},
    {0x01f78fb2b2e89e0cULL, 0x93775d0bcc338b62ULL, 799, 14, 0},
    {0x0223bfa4a5437316ULL, 0x2d23c36388727748ULL, 346, 16, 0},
    {0x022d1c8e38c738dcULL, 0x607fd17bf6b674daULL, 559, 16, 0},
    {0x0238ddf70906f25cULL, 0xa8629f71823a437aULL, 24, 13, 0},
    {0x023e8598590a7bb4ULL, 0x66c031f535be8cbaULL, 179, 16, 0},
    {0x02762f943bdca9a1ULL, 0xfab03a793e0bf2b0ULL, 543, 16, 0},
    {0x028ed5ed374d306aULL, 0xf21841156b420430ULL, 568, 16, 0},
    {0x02b51d9785300f32ULL, 0xee6f0b53b9725ba8ULL, 908, 16, 0},
    {0x02e9a1de59e47ae2ULL, 0xc684db73c5b47c84ULL, 117, 13, 0},
    {0x0319baf1db6aa970ULL, 0x83a51025c7a66160ULL, 331, 16, 0},
    {0x033e0c4b21360802ULL, 0xee8e44c58c3b72f4ULL, 797, 14, 0},
    {0x0356105e36a6ef2aULL, 0xef959c07271377f0ULL, 534, 16, 0},
    {0x0361d0cb5b61b581ULL, 0xdff5dc2a32cc84f6ULL, 689, 13, 0},
    {0x036b7b859dd6298fULL, 0xe8486938a412140aULL, 760, 14, 0},
    {0x037638f06382b5fbULL, 0xd04a9956652f8838ULL, 904, 16, 0},
    {0x03a5bc778df55ce3ULL, 0x6f469c861decc2caULL, 799, 14, 0},
    {0x03b9f4ff8efc609dULL, 0x7281ff12ba4a9124ULL, 804, 14, 0},
    {0x03ce4ca2f37174f3ULL, 0xeab280a59f143530ULL, 149, 16, 0},
    {0x03e6a7b5c223f043ULL, 0xba2d8fa2ed9cc11aULL, 834, 14, 0},
    {0x04114ef57250eab5ULL, 0x7455acaf0112a732ULL, 643, 16, 0},
    {0x0417c49cd75fc373ULL, 0x9399939ec64b7efaULL, 711, 14, 0},
    {0x0423a7006fe12bfcULL, 0x56f5bc2994dfdd20ULL, 783, 14, 0},
    {0x044c9d692c7a85f1ULL, 0xbe906888c5e0be5aULL, 799, 14, 0},
    {0x0457f002123d7f0bULL, 0x1acaf5be066e6eb2ULL, 151, 16, 0},
    {0x04745f5fd4708e80ULL, 0xe7f085eed1dd476cULL, 811, 14, 0},
    {0x048ee1cd0813e20bULL, 0x39bee687cc927bceULL, 847, 13, 0},
    {0x0496273de0e3f176ULL, 0xf2188f65b8a3b8caULL, 714, 14, 0},
    {0x04e3163339e4a7aeULL, 0xfac0944321a3a670ULL, 238, 16, 0},
    {0x04e590ce5307a3e3ULL, 0x0855711f3722f916ULL, 4, 9, 0},
    {0x04eb9caa06c13333ULL, 0xcb11670ea11c0588ULL, 902, 16, 0},
    {0x04f3f7c3f6f685d3ULL, 0xf2351859b1246782ULL, 840, 10, 0},
    {0x04f83e27558df3d4ULL, 0x6a60ae7b503c9e4aULL, 439, 16, 0},
    {0x051134631d78be5fULL, 0xda3be5fdcc34cea4ULL, 777, 14, 0},
    {0x05330e919b5cdbcfULL, 0x21583b36595832daULL, 939, 16, 0},
    {0x053e3c933e85b665ULL, 0x1b913ece624feca2ULL, 930, 16, 0},
    {0x05697d0be48ae1ccULL, 0x79fca6d14b25ca76ULL, 673, 13, 0},
    {0x057368205ee04693ULL, 0x900a44b398bf1c36ULL, 669, 13, 0},
    {0x05a0fe759c41edfbULL, 0x0c7577823ff59278ULL, 726, 14, 0},
    {0x05b8f2fced598c9bULL, 0x6935f810218e2efcULL, 777, 14, 0},
    {0x05c2375fe0856f11ULL, 0xc11da4dde332eef8ULL, 942, 16, 0},
    {0x05fdcfd10b0ab72eULL, 0x08ccc773950362d4ULL, 730, 14, 0},
    {0x060ab6adf693e350ULL, 0x13b38628c8406522ULL, 644, 16, 0},
    {0x06110780a2042c7eULL, 0x72143a87a60e1698ULL, 967, 16, 0},
    {0x06310825f03915afULL, 0x01ce6dd036641982ULL, 753, 14, 0},
    {0x0636450c8761eb05ULL, 0xa38e73d51ad21182ULL, 479, 16, 0},
    {0x063b0b4411994be7ULL, 0x9a8a82465c99341aULL, 739, 14, 0},
    {0x065c3151e506dc0dULL, 0xfab350ab39e79f4cULL, 804, 14, 0},
    {0x065ee2fc662db2f8ULL, 0xbed6e58ac8ffae52ULL, 231, 16, 0},
    {0x0666e04a8d9f73d1ULL, 0xa1845d4d7706a252ULL, 183, 16, 0},
    {0x0695ce2b523915a9ULL, 0x59443dfd47aafae2ULL, 202, 16, 0},
    {0x06c452cbccc1651fULL, 0xdad0679374bb3efaULL, 834, 14, 0},
    {0x06d0cb0f2ba83a7bULL, 0x400794196f2912d8ULL, 421, 16, 0},
    {0x06fcd13df836b8f2ULL, 0xb12c5966788e3850ULL, 656, 16, 0},
    {0x070d3bcc0f60c100ULL, 0xdb1c1461e0d0b090ULL, 211, 16, 0},
    {0x07397cd30cddd03aULL, 0x817f889bdbe5422aULL, 1000, 16, 0},
    {0x073ddc60d5165a9cULL, 0x9702cc4f5810595aULL, 817, 14, 0},
    {0x07659bfd1944e876ULL, 0xb294fe93e32ec000ULL, 1022, 16, 0},
    {0x076c866551b43eacULL, 0x227acf7e3d40cb6aULL, 861, 13, 0},
    {0x07ca93c165dc15e3ULL, 0xf665d7ad333fdbfaULL, 213, 16, 0},
    {0x07daff0168df0a01ULL, 0x1a486b1e13137b58ULL, 256, 16, 0},
    {0x08023caf3a8ae4e0ULL, 0x116cc0c596cf2362ULL, 542, 16, 0},
    {0x0816620dc756e5d0ULL, 0x2319026e33736d4cULL, 827, 14, 0},
    {0x0849d8103558e3f8ULL, 0x629f2a46eb602c72ULL, 591, 16, 0},
    {0x087ef61860dbe592ULL, 0xf512c28be72654f2ULL, 841, 10, 0},
    {0x08b55ffd60aa7bf2ULL, 0xd0f5a9025c2d7aacULL, 810, 14, 0},
    {0x08b56604d6577f1eULL, 0x63a3aa3e5af2f2daULL, 279, 16, 0},
    {0x0901ce42a82e7a9fULL, 0xf39148b3b9b09b10ULL, 614, 16, 0},
    {0x090c569bf50ec45bULL, 0x6fbb158866acd6ceULL, 692, 13, 0},
    {0x092276ac971e94b4ULL, 0xc4df07270f7faddaULL, 752, 14, 0},
    {0x0932c1e7d5b6c922ULL, 0x527b582a17d410bcULL, 833, 14, 0},
    {0x096c06d1806edb38ULL, 0xc963d900bca14a18ULL, 474, 16, 0},
    {0x096e962275f16b18ULL, 0xff8b93d97ffa80b0ULL, 973, 16, 0},
    {0x09ff3d01fb66e040ULL, 0x323799b8de4ae790ULL, 412, 16, 0},
    {0x0a95be7ca91f3551ULL, 0x2eed0f5567b3ec1aULL, 729, 14, 0},
    {0x0ac408f5850974d3ULL, 0xb267bb4b488b58dcULL, 730, 14, 0},
    {0x0acccc0a9ac73fb6ULL, 0x5511fc2ab6c53518ULL, 155, 16, 0},
    {0x0ace488ed051352aULL, 0x659ce5f9781c7bbcULL, 874, 13, 0},
    {0x0ad3696c3b4144e1ULL, 0x1074631dd3eee6d0ULL, 271, 16, 0},
    {0x0ad8cb2f5832667aULL, 0xbc24b12b46087770ULL, 1017, 16, 0},
    {0x0af221671a81f950ULL, 0x7d2cb28440731bf0ULL, 725, 14, 0},
    {0x0b216c74787b80adULL, 0x315c6d582ce9c238ULL, 723, 14, 0},
    {0x0b57468590724e57ULL, 0x332e32a4c08c06acULL, 830, 14, 0},
    {0x0bbd9a10038103bbULL, 0xeeefe77713e00ff8ULL, 1035, 16, 0},
    {0x0bda70383bf9ab50ULL, 0xabcf1f8a96f164a0ULL, 406, 16, 0},
    {0x0c06fb0748790bcaULL, 0xe6a8215b5d4b3fd8ULL, 242, 16, 0},
    {0x0c18c4cd55ea3b34ULL, 0x3d3f0b86e8a8fe1aULL, 456, 16, 0},
    {0x0c5514920c12eb61ULL, 0x0029e0c9b3339be0ULL, 209, 16, 0},
    {0x0c5f0afa8dfc349fULL, 0xe5aebb35d888630cULL, 57, 13, 0},
    {0x0c764f911c1e4c70ULL, 0xcb2a365f4af1f4b2ULL, 714, 14, 0},
    {0x0cb080e88330995aULL, 0xab739f89ed031eacULL, 804, 14, 0},
    {0x0cbb3903aacba5eaULL, 0xdb6d473bf56eb770ULL, 733, 14, 0},
    {0x0cc9d975786f5a7cULL, 0x7efb129b037b274aULL, 766, 14, 0},
    {0x0ce4ec377437cddfULL, 0x5c90e5875569349aULL, 268, 16, 0},
    {0x0ceedb15987c170dULL, 0x54ce1f2f30d6da2aULL, 961, 16, 0},
    {0x0cf540545e0a5833ULL, 0xf20b5a3dab61f65aULL, 822, 14, 0},
    {0x0d0ae144bcbb0736ULL, 0x2709c6d1fe98899aULL, 364, 16, 0},
    {0x0d32448cd9da47a3ULL, 0x9bde01d80658c58aULL, 724, 14, 0},
    {0x0d365e9003fe86ebULL, 0x2389506729de75c4ULL, 72, 13, 0},
    {0x0d49d86c961ad92bULL, 0x3e5c6801b283b472ULL, 734, 14, 0},
    {0x0d4c23cbd2338155ULL, 0x82f2b4559f527a10ULL, 135, 16, 0},
    {0x0d56242bb874f564ULL, 0x48d384ee176bc5aaULL, 799, 14, 0},
    {0x0d807861c25fb32eULL, 0x5046091910163e08ULL, 136, 16, 0},
    {0x0d84875778bc665dULL, 0x382848887b757a84ULL, 829, 14, 0},
    {0x0d9ede27c3d7754cULL, 0xf29ebe6f49d2f824ULL, 772, 14, 0},
    {0x0db54ec516d7d6bcULL, 0xa7f207885c8cbb32ULL, 287, 16, 0},
    {0x0dc4fb0b93fd8271ULL, 0x335fdb5f03a05f6cULL, 880, 13, 0},
    {0x0deaf672ae02a1b2ULL, 0xff40aa80019232e2ULL, 508, 16, 0},
    {0x0df1578a5522b515ULL, 0x15962e9ffeb62ebaULL, 958, 16, 0},
    {0x0dfd2f5dcc47cf07ULL, 0x2b96ffd97d8467d8ULL, 925, 16, 0},
    {0x0e4d51858aecd2c9ULL, 0x025f1185f598678aULL, 365, 16, 0},
    {0x0e531e6937ecc2ecULL, 0x49c7a8ec296f3934ULL, 826, 14, 0},
    {0x0e5855567d3c6c4bULL, 0x17c4239a491d8732ULL, 822, 14, 0},
    {0x0e8b9cba10a51182ULL, 0x0781a36aabfe0eb0ULL, 1025, 16, 0},
    {0x0e98392f05471708ULL, 0x7d0c73ad06dca70aULL, 731, 14, 0},
    {0x0e9f847a73fa7e72ULL, 0xbab5cac6f9fc5a92ULL, 289, 16, 0},
    {0x0eec69a1eff381c2ULL, 0xdc62ab3385e73298ULL, 784, 14, 0},
    {0x0ef519944b501a0dULL, 0x4d8901cc2622cf5aULL, 789, 14, 0},
    {0x0ef5ca23d027940eULL, 0xd2189de543eda820ULL, 284, 16, 0},
    {0x0ef7fc9976c41b4fULL, 0x7499246beb370dd2ULL, 758, 14, 0},
    {0x0f08bb474b60b036ULL, 0xfea4db87bd868664ULL, 103, 13, 0},
    {0x0f0c8bd3034a91e7ULL, 0x7f75f71511ddfd52ULL, 770, 14, 0},
    {0x0f0cd1d8c2e55f79ULL, 0xa03c2ae58bd5a4c2ULL, 365, 16, 0},
    {0x0f2a070822c03c56ULL, 0xa8fdffedf8cc2160ULL, 1036, 16, 0},
    {0x0f423ed503c033d7ULL, 0xc157526d13b6e170ULL, 453, 16, 0},
    {0x0f94ec21d05562ffULL, 0x5c86666a6e9b0e00ULL, 648, 16, 0},
    {0x0f9b1bb121e55fdaULL, 0x8e2e7972b0cd40eaULL, 448, 16, 0},
    {0x0fa432359497805dULL, 0xa381e09e965e2c7aULL, 537, 16, 0},
    {0x0fce2a8bb0efd16bULL, 0x661c36ef06fc9d08ULL, 239, 16, 0},
    {0x0fe496c00ce06262ULL, 0xc4713e1aacb3f32aULL, 737, 14, 0},
    {0x0fe790a954d3afa0ULL, 0xb850a249288b9002ULL, 460, 16, 0},
    {0x0fea87542370faacULL, 0x574d00fef0ad03daULL, 734, 14, 0},
    {0x1005fa875b3fa7dcULL, 0x4de9c3364e99d3d0ULL, 546, 16, 0},
    {0x100bf5d5cfdd40b9ULL, 0x2929a5ea7964cc80ULL, 900, 16, 0},
    {0x102716c34940e8aaULL, 0x695ed05418afaa14ULL, 797, 14, 0},
    {0x102e3331ebfcc597ULL, 0x99b5919ca67324f6ULL, 18, 12, 0},
    {0x10dd3ff9be80a303ULL, 0x82ae1f3ade638ad2ULL, 567, 16, 0},
    {0x112c1ab495445e0eULL, 0x651a64dc5a2e3220ULL, 330, 16, 0},
    {0x1142d6cfff3b238dULL, 0x316644b9d20f506eULL, 8, 12, 0},
    {0x114bbd21ec147e49ULL, 0x6b18dcce2d6bec32ULL, 731, 14, 0},
    {0x117a5bbf331528eeULL, 0xdfb481f9bb081b52ULL, 714, 14, 0},
    {0x1187a73f366a05c8ULL, 0x38b607de2f763c2cULL, 797, 14, 0},
    {0x1187b5051b21b8d3ULL, 0x9d759f0b0295481aULL, 828, 14, 0},
    {0x119aed6c0b18a9efULL, 0xe4a542e81e3f540aULL, 440, 16, 0},
    {0x11a90f73914dc114ULL, 0xae8e0a66cbba20c2ULL, 257, 16, 0},
    {0x11c5e1e3ddd2df17ULL, 0x0039c5ecf4450feeULL, 673, 13, 0},
    {0x11ce984d4a6aa572ULL, 0x23a81c9e966fca08ULL, 517, 16, 0},
    {0x11cf5f70f46127a8ULL, 0x2997dbb8fbeb5ec2ULL, 992, 16, 0},
    {0x11f3b87778ec8d94ULL, 0x53fb58312f5c71feULL, 669, 13, 0},
    {0x1249bb01142d03adULL, 0x4848a1cb8177fa70ULL, 564, 16, 0},
    {0x1258a6e352fc5362ULL, 0x6aaa7df66bf50552ULL, 369, 16, 0},
    {0x1286e4047c333644ULL, 0x8121bb080d90cdc6ULL, 705, 13, 0},
    {0x1289e3c43a0123ebULL, 0x9f62cf6be5aec77aULL, 717, 14, 0},
    {0x12946d914ef6f602ULL, 0x127173d7ce79e052ULL, 806, 14, 0},
    {0x12ac93dc054be180ULL, 0x3274d172ce7b3412ULL, 625, 16, 0},
    {0x1302dc2d66fa3ec8ULL, 0x14b2cc35f327d93aULL, 775, 14, 0},
    {0x131c7f88e47b4a75ULL, 0x397c64926b31f240ULL, 561, 16, 0},
    {0x1321df354f2d1f30ULL, 0xd9547739ec13d162ULL, 907, 16, 0},
    {0x132a8435e438ee4bULL, 0x8e4a6191d512694aULL, 613, 16, 0},
    {0x133ad1e97b76be07ULL, 0x696cb4fceafe85b8ULL, 521, 16, 0},
    {0x1358309495be91a3ULL, 0x8ffa5f4dae0fd650ULL, 294, 16, 0},
    {0x1380295732897aebULL, 0x70c96cb14efc892aULL, 981, 16, 0},
    {0x13a269c8b234b530ULL, 0x0418b16259a224c8ULL, 309, 16, 0},
    {0x13a2fd54d4011597ULL, 0x80b1d2b616303ea8ULL, 522, 16, 0},
    {0x13b260c8e72c3812ULL, 0x1c9cf2ec4db8ff4cULL, 797, 14, 0},
    {0x13c767dc032f3581ULL, 0x9a6f065bcbcfcf5aULL, 487, 16, 0},
    {0x13f699ff876e0381ULL, 0x6da601f1465c1bf8ULL, 212, 16, 0},
    {0x13f82169e1e3623aULL, 0xa4a64ad8630329e6ULL, 847, 13, 0},
    {0x1414b5681840f169ULL, 0xfb31017fef9cc1b4ULL, 730, 14, 0},
    {0x142525b7ac169d4eULL, 0xa7b779adbec26522ULL, 189, 16, 0},
    {0x145dce933bab5d00ULL, 0x8a531cc5087c94baULL, 357, 16, 0},
    {0x146edc9fbc43bfceULL, 0x5c72cbd4390d0d38ULL, 317, 16, 0},
    {0x1472583d8139d356ULL, 0xf0ad3341cce610daULL, 834, 14, 0},
    {0x14731a196bb29821ULL, 0x9c76b91b7cd191a0ULL, 327, 16, 0},
    {0x1486bd1a7c23326aULL, 0xdb832e0cf0828c82ULL, 133, 16, 0},
    {0x14bc2f7a87ab027cULL, 0x2f36213a3566cafaULL, 246, 16, 0},
    {0x14c6061539969dcbULL, 0xcd429b1c1105e4b2ULL, 641, 16, 0},
    {0x14ccc7aafd2f3fe9ULL, 0x8368a44e26917a92ULL, 627, 16, 0},
    {0x14d52672a4eb5e8cULL, 0x674af33100187c82ULL, 282, 16, 0},
    {0x14dc545be9987160ULL, 0x5cdd868a1bf40a82ULL, 409, 16, 0},
    {0x14f145a0dce57207ULL, 0xc0506289ffb9993aULL, 831, 14, 0},
    {0x14f15266adf1503cULL, 0x0f0fab33a3ac11baULL, 802, 14, 0},
    {0x15047d505a47f63eULL, 0xd3fe8b2f723ae602ULL, 773, 14, 0},
    {0x15124dfe8519e722ULL, 0x78dd9b28a4dd995cULL, 34, 13, 0},
    {0x1542797223259aa7ULL, 0x84e48b0abb95c29aULL, 300, 16, 0},
    {0x154ac5a4847dec01ULL, 0x2583d6acf60f0ab8ULL, 1044, 16, 0},
    {0x155ae17728f29d63ULL, 0x4b4a9382bd149984ULL, 72, 13, 0},
    {0x156eb33568ce54a5ULL, 0xc61c7c2406ec6bc0ULL, 605, 16, 0},
    {0x158bd629d1bf95d6ULL, 0xe5ceea6cc202c28cULL, 106, 13, 0},
    {0x1595eed31136a19cULL, 0x467652463eb9a792ULL, 789, 14, 0},
    {0x159d204ccbddfacfULL, 0xe8524ce7832a12baULL, 965, 16, 0},
    {0x15bb32e30df13b6eULL, 0x337b28c2d85236c4ULL, 71, 13, 0},
    {0x15cfc79183729a4dULL, 0xa3cad37f4b3f161aULL, 799, 14, 0},
    {0x15d9140f15208b5cULL, 0x675c4605f17b72eaULL, 259, 16, 0},
    {0x161b23fd29a42791ULL, 0x05d91d9aab102fcaULL, 612, 16, 0},
    {0x1686aee0789e096dULL, 0xb69de3164c81ff98ULL, 921, 16, 0},
    {0x16b59d02233f2460ULL, 0x15c7107c7bfff376ULL, 700, 13, 0},
    {0x16d7d99b580e8101ULL, 0x4e37d5a9dbce32deULL, 698, 13, 0},
    {0x16f11001282e4f78ULL, 0xa5cd2ec92991890aULL, 172, 16, 0},
    {0x16f9154dc7e1676aULL, 0xef463207aaed2596ULL, 9, 12, 0},
    {0x1700adf06f5821b2ULL, 0xcb268fa3a647917aULL, 793, 14, 0},
    {0x170b335c55da9d2dULL, 0xe91cc87842ce8e1aULL, 732, 14, 0},
    {0x17122b6fb7358006ULL, 0xfd3ab8b89bcdf228ULL, 765, 14, 0},
    {0x1751e7b5854b5fc5ULL, 0xadb078e45edaacaaULL, 762, 14, 0},
    {0x1762fcfbab5b580eULL, 0xaa2e92a51686107aULL, 640, 16, 0},
    {0x176770ca3de3c7e6ULL, 0x1fe0bc9659c0f87aULL, 115, 13, 0},
    {0x176d6fd369780431ULL, 0x85a32c37e1f28532ULL, 69, 13, 0},
    {0x178228feca86f873ULL, 0x627bfc9a23f006eaULL, 930, 16, 0},
    {0x17fa07561213de16ULL, 0x4f5a881509f57f56ULL, 703, 13, 0},
    {0x17fee73ead4082deULL, 0xa204b3b440a185dcULL, 23, 13, 0},
    {0x1801a508fbe3dc12ULL, 0x5e6851359bd6a3faULL, 390, 16, 0},
    {0x1822140a633815c7ULL, 0xf53951e0b1d39844ULL, 827, 14, 0},
    {0x18786e32a8df8fdaULL, 0xf1adee22859cbaa6ULL, 10, 12, 0},
    {0x18847356670b8ebfULL, 0xfd241fedbf1ded92ULL, 1011, 16, 0},
    {0x18abf06acbd103b2ULL, 0xa3da55e4a1bdb048ULL, 1018, 16, 0},
    {0x18ce976714665285ULL, 0x9c4cb0b3b322f85aULL, 475, 16, 0},
    {0x19173bbd3806b5c4ULL, 0x3438dda0a323a5eaULL, 551, 16, 0},
    {0x194c04004e9ab442ULL, 0xe1c1a9e4504977d0ULL, 1026, 16, 0},
    {0x196baba940679410ULL, 0x6434e645f44168b8ULL, 402, 16, 0},
    {0x19d5ff7f6e2500abULL, 0x4535a20aa29fe736ULL, 693, 13, 0},
    {0x19eba7244f564a97ULL, 0xc4d3603c456c6b88ULL, 1040, 16, 0},
    {0x1a2a7938b2b085d2ULL, 0x79706cbef5f3be6aULL, 21, 13, 0},
    {0x1a2fe404979a30a5ULL, 0xd2be05d5213e6b22ULL, 735, 14, 0},
    {0x1a39cdc6f47e9c3aULL, 0x2b96815f358d6e72ULL, 575, 16, 0},
    {0x1a3fa153deaf8009ULL, 0x789d4149dce4ab9aULL, 190, 16, 0},
    {0x1a44296d4554c084ULL, 0x4c79daee1e110338ULL, 277, 16, 0},
    {0x1a55ef7508a7dbc7ULL, 0x713ba9aba82318c0ULL, 160, 16, 0},
    {0x1a750bf95300ee9cULL, 0xeb88412497ba11eaULL, 724, 14, 0},
    {0x1a880d08cf436a66ULL, 0x0e8707f026cbeb36ULL, 681, 13, 0},
    {0x1aa61d623bf8c463ULL, 0x67d713aceae37fb2ULL, 943, 16, 0},
    {0x1abe07fe24921ecdULL, 0x956c33088a235032ULL, 380, 16, 0},
    {0x1b00932898ba4972ULL, 0x46dbfc2f537ab700ULL, 214, 16, 0},
    {0x1b03b15956e7ce26ULL, 0x263437bc2597785aULL, 996, 16, 0},
    {0x1b243b1e2b20a749ULL, 0x51c8e0abf52cd0a2ULL, 768, 14, 0},
    {0x1b2c6890752b31e0ULL, 0x8f7e2c958cee4a32ULL, 956, 16, 0},
    {0x1b40aa7888b3fb02ULL, 0xdb681ebaa86ade90ULL, 569, 16, 0},
    {0x1b4890aa431982ecULL, 0xf64bc5f79d817158ULL, 441, 16, 0},
    {0x1b4ad57fb3edb81fULL, 0x108781cd7f4ef768ULL, 258, 16, 0},
    {0x1b9eee1bfa85a04dULL, 0x95374530ef5d3468ULL, 923, 16, 0},
    {0x1bc4b66b5d424aa7ULL, 0xf4a4cc69ef429702ULL, 83, 13, 0},
    {0x1bc4d9fe7633ca25ULL, 0xb4997a38114ac5aaULL, 643, 16, 0},
    {0x1bc52dc5a6ecfa09ULL, 0xad55432c6d536468ULL, 893, 16, 0},
    {0x1be78bfa26d84c54ULL, 0xd8bff3e65bdb9050ULL, 353, 16, 0},
    {0x1c072b8e2e0cee7aULL, 0xdbddd250109476f8ULL, 404, 16, 0},
    {0x1c1db5969608504eULL, 0x4928acf459e733ccULL, 804, 14, 0},
    {0x1c329859137222d9ULL, 0x54d99832bb3bfd40ULL, 600, 16, 0},
    {0x1c362938165134a6ULL, 0xa80e6c46fe244e12ULL, 817, 14, 0},
    {0x1c69986db464c349ULL, 0x6590ba87e129b7d2ULL, 720, 14, 0},
    {0x1c84ca4f360f978bULL, 0xebe682da654376f0ULL, 764, 14, 0},
    {0x1c863c57ccf37561ULL, 0x0594a6f90c105f52ULL, 831, 14, 0},
    {0x1c8906adddad0b8dULL, 0xc411667b7ccdaedaULL, 956, 16, 0},
    {0x1c99da4d40b03f5aULL, 0x648366c2a9c6feb8ULL, 1032, 16, 0},
    {0x1cf177ebf3124f1dULL, 0xeabff35f9dd6af48ULL, 953, 16, 0},
    {0x1d000181102dc59cULL, 0x39d2c0e67106eb52ULL, 667, 10, 0},
    {0x1d547064a7d93e83ULL, 0x7d37fd2d135e6df8ULL, 234, 16, 0},
    {0x1dac4ad4f278703eULL, 0x388d7dbbde78f8e4ULL, 804, 14, 0},
    {0x1decd5260e8206c2ULL, 0xd5369c84a806df28ULL, 262, 16, 0},
    {0x1e00614b872839efULL, 0x82de21b50f43684aULL, 580, 16, 0},
    {0x1e02d210f4497597ULL, 0x6ff5a4e8ff6460e0ULL, 525, 16, 0},
    {0x1e4853eb8fdd1982ULL, 0xcb8c5c775daaa642ULL, 985, 16, 0},
    {0x1e51146edc7b4f4eULL, 0x3522ebb15f4593c0ULL, 204, 16, 0},
    {0x1e5b92a1e3681532ULL, 0x0ba018c530339c22ULL, 735, 14, 0},
    {0x1e7405f64937ea93ULL, 0x11341b9b7ac6f82eULL, 694, 13, 0},
    {0x1e95da8fc54cd141ULL, 0xdd7fa1e232a909baULL, 805, 14, 0},
    {0x1e96fd3060af4a50ULL, 0xc13311a6e962a052ULL, 805, 14, 0},
    {0x1eab975e99775c1bULL, 0xb41d26a143821158ULL, 1027, 16, 0},
    {0x1eb5f1bfd26c7ef0ULL, 0xfc4c067baf513232ULL, 964, 16, 0},
    {0x1ed4575bf748feb2ULL, 0x1798001106046deaULL, 207, 16, 0},
    {0x1ef4153a68b909bbULL, 0xd6fc5ae5d70d707aULL, 731, 14, 0},
    {0x1ef7f51ce8519c39ULL, 0x07e5cba1d6756392ULL, 641, 16, 0},
    {0x1efb10fc80bc60a4ULL, 0xe5fe1fc747629ceaULL, 815, 14, 0},
    {0x1eff5b3290f0a9c6ULL, 0xd94bd0c22f7032aeULL, 844, 13, 0},
    {0x1f1c50a354d10fb2ULL, 0x622cd95c82d46052ULL, 724, 14, 0},
    {0x1f34428539a6bd77ULL, 0xfff80154f004ee1aULL, 734, 14, 0},
    {0x1f41c10bea6d1237ULL, 0x5c84551e624b2a0aULL, 127, 16, 0},
    {0x1f73a0631090ded5ULL, 0x8efe16a809b2471cULL, 730, 14, 0},
    {0x1f83ed73227f3bbdULL, 0x48a5010e8e96e68aULL, 493, 16, 0},
    {0x1f898252433246ebULL, 0x27f128193660bc86ULL, 846, 13, 0},
    {0x1f9069107fbf77dcULL, 0xfb62dae5c81c0792ULL, 495, 16, 0},
    {0x1f951319a3a42637ULL, 0x5e54a6384ac9c932ULL, 1002, 16, 0},
    {0x1fccdeb4c9a401ecULL, 0xc82d7227ced0021aULL, 731, 14, 0},
    {0x1fd28924c56113bcULL, 0xeba7746913244378ULL, 145, 16, 0},
    {0x1fd38a49f2c763bcULL, 0x9cb223ccdb05056cULL, 797, 14, 0},
    {0x1fdc7033b75ffc2fULL, 0xd25b97430e7fb9e8ULL, 895, 16, 0},
    {0x1ffff05742c57c81ULL, 0x84ab62028c4e7892ULL, 345, 16, 0},
    {0x201e9941a163fc88ULL, 0xb8baa52d6c8ac5faULL, 200, 16, 0},
    {0x20457370cb3efc17ULL, 0x072ecf7472398410ULL, 283, 16, 0},
    {0x2059c9e277e3bb9aULL, 0xd285f30b8f33ebb2ULL, 372, 16, 0},
    {0x206db1138bcb01c4ULL, 0x5f7ed5c4248d72daULL, 863, 13, 0},
    {0x208712fc628bb211ULL, 0x2a622fb050b55a60ULL, 589, 16, 0},
    {0x2096984aeb7c346dULL, 0x297464ccbb56ec48ULL, 218, 16, 0},
    {0x20a01336a8c5ec00ULL, 0xa7b2a62637c183d2ULL, 716, 14, 0},
    {0x20b7329248231f76ULL, 0xddf55f1af45d59beULL, 698, 13, 0},
    {0x20c9422a9b6c52a2ULL, 0x4f86721a2e1b4910ULL, 585, 16, 0},
    {0x20d9f39146d88d62ULL, 0xb8c89a77f837f6f8ULL, 902, 16, 0},
    {0x20dff8bb19a4256aULL, 0x4e87e6b3561318b2ULL, 516, 16, 0},
    {0x20e8a5994155cdf5ULL, 0x56a9ba0662f4fccaULL, 714, 14, 0},
    {0x210307137eee54b4ULL, 0x75b55976589f516aULL, 78, 13, 0},
    {0x211803a5d09cbe3cULL, 0xa40559e2874b22eaULL, 502, 16, 0},
    {0x2118d1b633542a5dULL, 0xd436ad162c09346eULL, 839, 12, 0},
    {0x211d24f7f8d6a17bULL, 0x44ab519064533952ULL, 747, 14, 0},
    {0x213dfb3ec5592996ULL, 0xf86ba198cea17392ULL, 432, 16, 0},
    {0x21408d50944b8a10ULL, 0x0544fe6514a68108ULL, 750, 14, 0},
    {0x21587a204a739c42ULL, 0x1b2d42e83cbc8d10ULL, 800, 14, 0},
    {0x21735c503a475cc0ULL, 0xc20883583286d4baULL, 715, 14, 0},
    {0x2180244adb584632ULL, 0x85af62bf512a0c10ULL, 920, 16, 0},
    {0x21978664167540b8ULL, 0x38a3d8b17500139eULL, 11, 12, 0},
    {0x21b302c4b37cd85eULL, 0xa2858528e94a4202ULL, 512, 16, 0},
    {0x21b6febd180bf1e6ULL, 0x9b199389b51a24f2ULL, 755, 14, 0},
    {0x21cc50bb0017407cULL, 0x1c7d0ba8da46a680ULL, 363, 16, 0},
    {0x21d10a9c2ea82ab3ULL, 0x9c3e9570bd5dc3eaULL, 75, 13, 0},
    {0x21f89f23f553d873ULL, 0x7a8b6cfde95219dcULL, 814, 14, 0},
    {0x221cbde6c3972c40ULL, 0xf7ebadbabbc9870cULL, 44, 13, 0},
    {0x224b95cce604a41eULL, 0x3be5f083baf64172ULL, 476, 16, 0},
    {0x226a823819f731dbULL, 0x0d270b6e9317d958ULL, 500, 16, 0},
    {0x22aaa880befd745cULL, 0xe269d94ec6ca11aaULL, 527, 16, 0},
    {0x22b79612cf0fa1b1ULL, 0xf64d5d71d3e13ceaULL, 776, 14, 0},
    {0x22c9c7e7d945b844ULL, 0x3eb8c53d7341b492ULL, 856, 13, 0},
    {0x22dce829bd822333ULL, 0xeed3003b92896458ULL, 975, 16, 0},
    {0x22dd1fc2b5caf25fULL, 0xc675354304a967e0ULL, 160, 16, 0},
    {0x2308d5e83b3356f9ULL, 0x33d43cd2812b72a2ULL, 735, 14, 0},
    {0x23244acbf65d826dULL, 0xd8d275b4e5e3bc78ULL, 465, 16, 0},
    {0x233cc1a1bcdffa4dULL, 0xb1fc417d591551e8ULL, 721, 14, 0},
    {0x2352bc47dac0c7c7ULL, 0xe8fc520bcae67310ULL, 1039, 16, 0},
    {0x238d730e2673760eULL, 0xbbd555e16ca5c41aULL, 584, 16, 0},
    {0x239e4f38c8223c68ULL, 0x2e61e48c0e6d144aULL, 282, 16, 0},
    {0x23a975255384f816ULL, 0x93c6938859b495f2ULL, 1003, 16, 0},
    {0x23c4afad88ae05adULL, 0xb25e926075b13ac2ULL, 466, 16, 0},
    {0x23cfb421dcf60d94ULL, 0x51e83dfbc9a270daULL, 892, 16, 0},
    {0x2410fbd69fe6c343ULL, 0xa7f69e77ddb22848ULL, 281, 16, 0},
    {0x24135ee23b693660ULL, 0x896b329e372eb7ceULL, 866, 13, 0},
    {0x246da261ac8f1158ULL, 0xaa40eb59098feda0ULL, 373, 16, 0},
    {0x249db580798597faULL, 0xfedd50220cc38cc8ULL, 1041, 16, 0},
    {0x24a78e66247ff7a4ULL, 0xce3991ed2a7fc4a2ULL, 794, 14, 0},
    {0x24d2e966b04b9408ULL, 0xfa23aa400663c8d6ULL, 868, 13, 0},
    {0x253c071545ccdea8ULL, 0x3c49a93d7ed920ceULL, 678, 13, 0},
    {0x254989d94e7b71a5ULL, 0x5bbb2e57ba92ede8ULL, 818, 14, 0},
    {0x254e7690b4eac60eULL, 0x644656bcc1872f92ULL, 291, 16, 0},
    {0x2579a3022c47f6d0ULL, 0x08fc8434a6431710ULL, 295, 16, 0},
    {0x25941cf1f5fadaacULL, 0xee285dbf475a733aULL, 187, 16, 0},
    {0x25a6dc18a99a6a2eULL, 0x6df0cfd3f30a22faULL, 29, 13, 0},
    {0x25a9de8887d3332bULL, 0xde457ec5ec9809d8ULL, 519, 16, 0},
    {0x25cf000345b59b2bULL, 0xe8caa395dbc9fd32ULL, 798, 14, 0},
    {0x25d13940b709d21fULL, 0xe6e06d26c49054d2ULL, 400, 16, 0},
    {0x25e8d7b822fa8476ULL, 0x4b757eacbf083810ULL, 598, 16, 0},
    {0x25ecab92d653a2a9ULL, 0x59c7a4353d51d388ULL, 942, 16, 0},
    {0x25f5946d2f408c6dULL, 0x57bda7f7dfa378f0ULL, 228, 16, 0},
    {0x25ff168a0071f5bfULL, 0x86e71d5834cc4358ULL, 391, 16, 0},
    {0x2616198b50d4cf45ULL, 0x73be2a21644638d2ULL, 342, 16, 0},
    {0x2632005030aa1f1dULL, 0xf7b39ff8203824e2ULL, 720, 14, 0},
    {0x264deaec08c29da8ULL, 0x92f9fba40e574c16ULL, 697, 13, 0},
    {0x2652c5e4a080288eULL, 0x0a437d71810243feULL, 658, 7, 0},
    {0x266dd0f853f741b8ULL, 0xb1d9ff368d52a80aULL, 633, 16, 0},
    {0x267cce253d8e4791ULL, 0xe32c24e0804d3078ULL, 164, 16, 0},
    {0x26964131efe87912ULL, 0xac03187a46f30d88ULL, 751, 14, 0},
    {0x26a3a22ddb0eb471ULL, 0x66589d537bcff672ULL, 621, 16, 0},
    {0x26b81c0a1f3cd13cULL, 0xeac7e784b0ede134ULL, 730, 14, 0},
    {0x26cd997f3e14c05bULL, 0x72f9051982a8525cULL, 812, 14, 0},
    {0x26efda8fd0c239c1ULL, 0xe15230a6d6468108ULL, 217, 16, 0},
    {0x26f55bbbbd97011fULL, 0xde154f2907dbba9aULL, 743, 14, 0},
    {0x26fb65cd508bb855ULL, 0x2fe77c7fdf65ab92ULL, 591, 16, 0},
    {0x270350833f028ea9ULL, 0xcd7228061b84b692ULL, 278, 16, 0},
    {0x272e4d5fc875208fULL, 0x8d69f149e799c81aULL, 489, 16, 0},
    {0x2754aaba343880aeULL, 0xcc60b6b95a04accaULL, 774, 14, 0},
    {0x2760209ff33ee9b7ULL, 0x020351c067907d22ULL, 560, 16, 0},
    {0x276cccf1cc7ebcd8ULL, 0x0d8bb6353a10a95aULL, 299, 16, 0},
    {0x27ccf92bf928c8dbULL, 0xefd1d391b20f6702ULL, 584, 16, 0},
    {0x27f46eda88696719ULL, 0xb620249732f61210ULL, 322, 16, 0},
    {0x27faba2c05f469abULL, 0xceaff4e5c58b33eaULL, 300, 16, 0},
    {0x280766e6e483c6a6ULL, 0x67244ade4d9ac7b8ULL, 165, 16, 0},
    {0x281954e08822cd29ULL, 0xc86a276c01fd9c28ULL, 750, 14, 0},
    {0x284693e9ca17bc82ULL, 0xb1c389e5dea13c7aULL, 96, 13, 0},
    {0x284f55e8a04d2be7ULL, 0x1afbeb69df179792ULL, 987, 16, 0},
    {0x285dc6993193ac68ULL, 0x53fef1f1ea2c1470ULL, 353, 16, 0},
    {0x288084187f0c62d7ULL, 0x5847128b2c92c242ULL, 74, 13, 0},
    {0x28b9016eaed1cd74ULL, 0x24f058be2f6e008cULL, 22, 13, 0},
    {0x28c7bac5c59efa1aULL, 0xec9204eae0c5809aULL, 144, 16, 0},
    {0x28f1462fad4a8235ULL, 0x9a87d20b09afc9daULL, 49, 13, 0},
    {0x28fc60dfd9618ebaULL, 0xd447761f3df74b80ULL, 637, 16, 0},
    {0x292a6fc43d7b2e2cULL, 0xf51643b5b2ce8562ULL, 437, 16, 0},
    {0x2946ab718dd8a21eULL, 0x47f1d154052ffff2ULL, 928, 16, 0},
    {0x29631f9475fa6cc3ULL, 0x05fc3bc6c5e76ff2ULL, 805, 14, 0},
    {0x296818795a87b0f3ULL, 0x556ccc29e606d7ccULL, 824, 14, 0},
    {0x296e857c6e724467ULL, 0x043ffba4cb0ae878ULL, 391, 16, 0},
    {0x297fc9cce807f266ULL, 0xdd88277fa16529baULL, 766, 14, 0},
    {0x29bcb233d8e22274ULL, 0xe7f5aea48778b4d8ULL, 901, 16, 0},
    {0x29cc71b8863fc74bULL, 0xde018fca96ede210ULL, 602, 16, 0},
    {0x29d267534a92b7dbULL, 0x708c44dddf475102ULL, 467, 16, 0},
    {0x29dbb7964117e4aaULL, 0x2d4b31498a37a56aULL, 771, 14, 0},
    {0x29dd23424c501ea8ULL, 0x8309e6128c8c4404ULL, 826, 14, 0},
    {0x2a4e0cdf61e1b120ULL, 0xc606f2c37b260642ULL, 938, 16, 0},
    {0x2ab9dc9e00f76cf6ULL, 0xc09aa4b2dda07112ULL, 597, 16, 0},
    {0x2ac5857fc86c23d3ULL, 0x7a47622d1fa78fd6ULL, 851, 13, 0},
    {0x2ad2a08a4ac2fd6eULL, 0x0a0903354c32aaa0ULL, 374, 16, 0},
    {0x2aea1a3669020666ULL, 0x84f71a053bbbf212ULL, 796, 14, 0},
    {0x2b3381b69ed7101cULL, 0xf589ab5e049b8fbaULL, 511, 16, 0},
    {0x2b46b0cf69bf8d03ULL, 0x806f0e9e59ef813aULL, 734, 14, 0},
    {0x2b529b93a4d7fb97ULL, 0xc55512ff7f643908ULL, 352, 16, 0},
    {0x2b5b347b2e20c855ULL, 0x4c927c8fdc508384ULL, 804, 14, 0},
    {0x2b76fda3f730be2dULL, 0x454074a0ef31fcc4ULL, 108, 13, 0},
    {0x2ba91f3f2916eff4ULL, 0xc03f1cb86e52ef88ULL, 1018, 16, 0},
    {0x2bb30b234e3820a1ULL, 0xd81b199a1c9f4df0ULL, 329, 16, 0},
    {0x2bc38bb3958324eeULL, 0x65bd9cc5ee9e0d88ULL, 917, 16, 0},
    {0x2be6b5307a11ede7ULL, 0xd6cf834d3e24e1e8ULL, 317, 16, 0},
    {0x2bee2dde1d668717ULL, 0x93f034db3237ff90ULL, 912, 16, 0},
    {0x2bfb14a0243532c0ULL, 0x9410fe8920a759c0ULL, 582, 16, 0},
    {0x2c257b8aef524021ULL, 0x9871cbeae63a542aULL, 762, 14, 0},
    {0x2c2f9e6dcc238dc3ULL, 0xfb76e22ebbac3c58ULL, 953, 16, 0},
    {0x2c3dddd376eba1bbULL, 0x2cc877ee3d9250f2ULL, 796, 14, 0},
    {0x2c45935af0debcddULL, 0x9897c8362f3e9340ULL, 1023, 16, 0},
    {0x2c47af8d7a8e39e2ULL, 0x10952db711cdd752ULL, 664, 10, 0},
    {0x2c47c521c7faac6eULL, 0xd15cbb1d2ae25afaULL, 31, 13, 0},
    {0x2c60c512638e75e3ULL, 0xaa8f66d3196238bcULL, 809, 14, 0},
    {0x2c7a739985a83059ULL, 0x02ff0887ebcf8490ULL, 524, 16, 0},
    {0x2c7b7d62a48403f5ULL, 0x62b69e037a1045f8ULL, 783, 14, 0},
    {0x2c95f450b321b3d4ULL, 0x1b5a0ebb3ba908e2ULL, 194, 16, 0},
    {0x2ccce0cb8660fb2eULL, 0x1db399e8dc677b62ULL, 359, 16, 0},
    {0x2cd1bc324f4d9489ULL, 0xa96630df4b98abcaULL, 737, 14, 0},
    {0x2cfd92b4615026f3ULL, 0x780bdeb61580b2d2ULL, 516, 16, 0},
    {0x2d016b782a141816ULL, 0x231f68fb5d3fa102ULL, 617, 16, 0},
    {0x2d48543eccce36cbULL, 0x7d7d5e1bd2663e80ULL, 969, 16, 0},
    {0x2d69dfba352d820aULL, 0xf6014fbb1b3bd190ULL, 169, 16, 0},
    {0x2d7bf6dac8e57314ULL, 0xbf6e3d487de303daULL, 803, 14, 0},
    {0x2d7f47b77e51d49dULL, 0x15e7d6de990f8c72ULL, 127, 16, 0},
    {0x2d8c15571e3c829dULL, 0xa893143298e8e798ULL, 349, 16, 0},
    {0x2d8e9e8b074daef3ULL, 0xf3402410e99f3040ULL, 208, 16, 0},
    {0x2db42532e81d0b48ULL, 0x9ae8d7032ed4e6daULL, 270, 16, 0},
    {0x2e38058e45e0e5d1ULL, 0x69a86ca36cea14faULL, 840, 10, 0},
    {0x2e9b4d2ee30f62aaULL, 0x19b8a8f1e26b79dcULL, 797, 14, 0},
    {0x2e9c3282a1c114eeULL, 0xb01621886ec06e72ULL, 342, 16, 0},
    {0x2ea87f351f74a5c0ULL, 0xf33fda6d1460ada0ULL, 619, 16, 0},
    {0x2ebcdf12f0a081aaULL, 0x214a64836d1b33a2ULL, 475, 16, 0},
    {0x2ed57662ea381f7eULL, 0x05225d8aadda88c8ULL, 388, 16, 0},
    {0x2ee3fdc9548006dbULL, 0x91fdde4c1539f728ULL, 348, 16, 0},
    {0x2ef99ac8defbb6c9ULL, 0xf3569bb800affc38ULL, 236, 16, 0},
    {0x2f0428664078b752ULL, 0xcfcbadb744376ff8ULL, 1043, 16, 0},
    {0x2f1849072c0188efULL, 0x99da2790b7a6c3f2ULL, 945, 16, 0},
    {0x2f1d92fe4e00470bULL, 0x90283e50f04b10eaULL, 799, 14, 0},
    {0x2f240a4b9f66dd94ULL, 0xf0e42be2a348d308ULL, 235, 16, 0},
    {0x2f44ad62fc8e1d67ULL, 0xf3fdde49aad63142ULL, 90, 13, 0},
    {0x2f5bb5a32da94de3ULL, 0x93b8609cc92aa122ULL, 252, 16, 0},
    {0x2f5e4a6c34ec18daULL, 0x18715370f9cbd876ULL, 19, 12, 0},
    {0x2f61d806501a6faeULL, 0x9a317827d96b4a8aULL, 795, 14, 0},
    {0x2f896df2280f1a6cULL, 0x6a6899b0562009e8ULL, 263, 16, 0},
    {0x2fdff9fe692a2ef8ULL, 0xde51eeb27023d320ULL, 899, 16, 0},
    {0x2ff64f93ef8c358cULL, 0x273363a9d508d580ULL, 579, 16, 0},
    {0x2ffa9d7f49cfea27ULL, 0xc6f2ef8e4574ad58ULL, 500, 16, 0},
    {0x2ffee64669da92a9ULL, 0xf5c5a32e35f92dbeULL, 687, 13, 0},
    {0x3016756ac52b978dULL, 0x0b409049a99600c6ULL, 872, 13, 0},
    {0x30444cc44d1f3eceULL, 0xb7d1eef3cd719a00ULL, 368, 16, 0},
    {0x304866d91b839595ULL, 0x8a8b3f5ae4bbb598ULL, 465, 16, 0},
    {0x3058dd49e725a6e8ULL, 0xb502a68f0fa01374ULL, 118, 13, 0},
    {0x3059ac9b7a837842ULL, 0x80f0e4ed98be9a9aULL, 431, 16, 0},
    {0x30af6c347a417b21ULL, 0xb94702eb03c6680aULL, 859, 13, 0},
    {0x30cf66e004a67794ULL, 0xa84a55803da54d78ULL, 940, 16, 0},
    {0x30d77b66883671f0ULL, 0xdf433d24448ac762ULL, 244, 16, 0},
    {0x30ddc3cad63241a6ULL, 0x11835fcdbfe5443eULL, 691, 13, 0},
    {0x31793dd74241263eULL, 0x8dda2f0a47b6c2a0ULL, 170, 16, 0},
    {0x31ac49863c4af2bbULL, 0xdb266e7a58684a5aULL, 753, 14, 0},
    {0x31b4866ac7aff383ULL, 0x1aebfe38f05e8418ULL, 420, 16, 0},
    {0x31c9b7e2ae8a6c7aULL, 0x87a3445aa936b82aULL, 749, 14, 0},
    {0x31d3202af37e9980ULL, 0x85832ef9ef2c4b00ULL, 532, 16, 0},
    {0x31e9201c662aad15ULL, 0xe02ad13bd364f816ULL, 705, 13, 0},
    {0x31f13bade9956b9dULL, 0x31686dd927fa82b2ULL, 597, 16, 0},
    {0x32200da6e0732939ULL, 0xa4b54db5d5fc2964ULL, 797, 14, 0},
    {0x32395e17ca511eccULL, 0x046601c69e05bc0aULL, 491, 16, 0},
    {0x324036120d74f1ebULL, 0x5e0775a492902adaULL, 736, 14, 0},
    {0x3267e97def3dfa30ULL, 0x40a23259481137b0ULL, 214, 16, 0},
    {0x326d392decab5b5eULL, 0xfa158322dd1c831aULL, 397, 16, 0},
    {0x327a483856f7eea6ULL, 0xa1d61aaaa6d17732ULL, 563, 16, 0},
    {0x32980d05310cae7aULL, 0xe5fd4add719c49daULL, 732, 14, 0},
    {0x32b51b09986480a9ULL, 0x98cc5f08ff82492eULL, 852, 13, 0},
    {0x32cda58fc8076a14ULL, 0xb4c10c96635c42b8ULL, 1028, 16, 0},
    {0x32eab013d86543daULL, 0x2dcd023ff0248c08ULL, 315, 16, 0},
    {0x3315bb0cf50c251aULL, 0x2f4aadff5c67db48ULL, 240, 16, 0},
    {0x331eb26656615606ULL, 0x7c735dfc901fc5d0ULL, 320, 16, 0},
    {0x337e94321313232dULL, 0x911643bfd07a67e0ULL, 134, 16, 0},
    {0x3387ecafbdae6ecaULL, 0x92ef28c29d5b9a88ULL, 383, 16, 0},
    {0x3392a95f7497c3caULL, 0xed07d634138f0370ULL, 818, 14, 0},
    {0x33ce67997dce3a5fULL, 0xb11aff75fc72991aULL, 539, 16, 0},
    {0x33d7e8bad8c89abdULL, 0x2a267f0c39f18814ULL, 812, 14, 0},
    {0x33eb4da37f9a0a2eULL, 0xdce07384b41c3262ULL, 944, 16, 0},
    {0x34136e31fd9cfe89ULL, 0x98af3aa81d3fc4e2ULL, 560, 16, 0},
    {0x343db92149939447ULL, 0x386c111ee4da93a0ULL, 366, 16, 0},
    {0x344f9b4296dfb34dULL, 0x61f0e2dfd62b1208ULL, 387, 16, 0},
    {0x346573a0aa131196ULL, 0x08e444f3b077e59cULL, 814, 14, 0},
    {0x346811eac8dbf490ULL, 0xaa5ee1960c053ab2ULL, 743, 14, 0},
    {0x348034de575fbc36ULL, 0xad44a82f22f3c532ULL, 230, 16, 0},
    {0x348477eeda561a66ULL, 0x509c2ec23164c0c2ULL, 192, 16, 0},
    {0x34a599b958bf3908ULL, 0x42bf2a473111a1d2ULL, 324, 16, 0},
    {0x34af3b7f21d4db74ULL, 0x463ebf527b56063cULL, 804, 14, 0},
    {0x34bf8e170d07d68dULL, 0xc6d36e4a92881032ULL, 601, 16, 0},
    {0x34fca6b7f1fbc6a3ULL, 0x7f1aaa30ad8f095cULL, 812, 14, 0},
    {0x350505fdf4526a5cULL, 0x745f5437757ab98cULL, 109, 13, 0},
    {0x350a4b59013bcd92ULL, 0x1ff39c6c0813367eULL, 1, 6, 0},
    {0x3543f131cb73af22ULL, 0x425f3e8c8ffe7898ULL, 143, 16, 0},
    {0x3557545cbb8fbe6eULL, 0xcc5e2397ec56e8b0ULL, 1045, 16, 0},
    {0x359494a62b36dd35ULL, 0x0e464cddcdcdb3aaULL, 792, 14, 0},
    {0x35950ed598a490c1ULL, 0xe33f16f75d8b44aeULL, 690, 13, 0},
    {0x35ba14be23f00b5fULL, 0xc65b5b9569e5fa12ULL, 790, 14, 0},
    {0x35dd6a112d84ae67ULL, 0x9ddbb01ce7bb0dd4ULL, 810, 14, 0},
    {0x35e22260c465fe99ULL, 0x831fa32edfe374c0ULL, 719, 14, 0},
    {0x360243921b9b2e8eULL, 0x6dbe7a7b14b439f8ULL, 333, 16, 0},
    {0x360b135353350a28ULL, 0x9672dbbd4fa8a818ULL, 520, 16, 0},
    {0x361aca7a0f40b070ULL, 0x1dbe96d79b21b852ULL, 1000, 16, 0},
    {0x362e19202078b66fULL, 0x662e3a5751a29422ULL, 462, 16, 0},
    {0x365e60652ba1a7b6ULL, 0x0b60f6ad055d3ec2ULL, 855, 13, 0},
    {0x3680691340a97fbcULL, 0x5a89dab4dd1c86b2ULL, 639, 16, 0},
    {0x36857d1a0d60d021ULL, 0x159077b6ab8ef84cULL, 772, 14, 0},
    {0x36c3bdd7a25db962ULL, 0xa6b185b5fb2675a2ULL, 939, 16, 0},
    {0x3704498963b2fcc7ULL, 0xed7b3681a175efe2ULL, 298, 16, 0},
    {0x371118ce3868f43fULL, 0x2b6f8752d6689c90ULL, 426, 16, 0},
    {0x372b07062a15b862ULL, 0xa7e22ee84951191aULL, 450, 16, 0},
    {0x37557353bef8e55bULL, 0x04edfdb054211b2aULL, 539, 16, 0},
    {0x376804364e9153bcULL, 0x71ab5557edb12dd2ULL, 183, 16, 0},
    {0x3785f19a33349753ULL, 0x70132c6c0a537feaULL, 45, 13, 0},
    {0x37a486977e2acbfdULL, 0x4d8d8529856b9a34ULL, 824, 14, 0},
    {0x37ef75dc3eb382ecULL, 0x1f206f30b21166faULL, 763, 14, 0},
    {0x37f3b40788bbbe19ULL, 0xb109c6a0671328daULL, 769, 14, 0},
    {0x38014ae29341a94cULL, 0xfdfc2d4c54fb1fbcULL, 88, 13, 0},
    {0x38066d47ea954646ULL, 0x083b8da116f1d382ULL, 492, 16, 0},
    {0x381779e1f5eb31f6ULL, 0x9b51c3b2cb967f00ULL, 248, 16, 0},
    {0x3819811098106a82ULL, 0x8dcc284637b772ecULL, 66, 13, 0},
    {0x38419c014a65f572ULL, 0x470a80c2b6f24382ULL, 742, 14, 0},
    {0x38ed5ce3298de9efULL, 0x2de91e398496b2baULL, 484, 16, 0},
    {0x390aea37dd7834a5ULL, 0x7256058e8321cbfaULL, 490, 16, 0},
    {0x390cd49cfaaff230ULL, 0x5843eadba48bdfd8ULL, 728, 14, 0},
    {0x3926000082fcf006ULL, 0x7e94fcf204e8905aULL, 120, 16, 0},
    {0x3933652ca5a1c763ULL, 0xf72068dc7492c1c8ULL, 968, 16, 0},
    {0x393792adfc281b0dULL, 0x1a07f12d66a8c052ULL, 984, 16, 0},
    {0x394079546141ad1cULL, 0x1e9fac3606dc1b52ULL, 201, 16, 0},
    {0x394ed4b1f3b51dd9ULL, 0x27fef627b4ca12b2ULL, 495, 16, 0},
    {0x396563e74a244819ULL, 0x9c633da80639572aULL, 377, 16, 0},
    {0x39704b0dbd4748b9ULL, 0x78fcb878ca0dcee2ULL, 480, 16, 0},
    {0x39804c1f6cdd2fbfULL, 0xa74651f9773e665aULL, 197, 16, 0},
    {0x398134b6082c6e1bULL, 0x3c2f9ea147978b3aULL, 963, 16, 0},
    {0x398710f8fc7fc032ULL, 0x625f37e8ef825c5aULL, 494, 16, 0},
    {0x399dbbd983dad4a7ULL, 0x00fab022176e6ab2ULL, 910, 16, 0},
    {0x39c4f441e8e77b83ULL, 0x6460f175d25386c8ULL, 392, 16, 0},
    {0x39cae81e2ebb3350ULL, 0x1a655c8910f2c968ULL, 521, 16, 0},
    {0x3a0c48f746a901f0ULL, 0xb137ae3a315b4218ULL, 162, 16, 0},
    {0x3a20d7d0f36c8bd1ULL, 0xa820b7c016df4b02ULL, 825, 14, 0},
    {0x3a348a2969b45c03ULL, 0x6e5122a40bd11a92ULL, 961, 16, 0},
    {0x3a5434de0315bb29ULL, 0xb558b42a907cc4faULL, 769, 14, 0},
    {0x3a583f2719c8e6deULL, 0x822f60e2d9d9e0e2ULL, 144, 16, 0},
    {0x3a6cd7bbbfe37370ULL, 0x08f0105f00fff6f0ULL, 285, 16, 0},
    {0x3a8ee5aaeb33d285ULL, 0xb15be019b8575298ULL, 908, 16, 0},
    {0x3acb6fc445b5c998ULL, 0xa1ebc6f57d9d8cbaULL, 732, 14, 0},
    {0x3add7174e9dc1bb8ULL, 0x8a710bc57b6152c0ULL, 599, 16, 0},
    {0x3ae02ebccd42ea98ULL, 0x4b97a0fbca77debeULL, 872, 13, 0},
    {0x3ae68393aeceb628ULL, 0x3825da9303ce025aULL, 557, 16, 0},
    {0x3af66a7e0f754bd4ULL, 0x7e14ac30757c650aULL, 790, 14, 0},
    {0x3b244bc06dd09333ULL, 0xd8236626b5f942eaULL, 767, 14, 0},
    {0x3b39d7df4a53a1caULL, 0x913f5908d7a6137cULL, 833, 14, 0},
    {0x3b4461cbbf3e12d9ULL, 0x71cba0928e2d65eeULL, 5, 9, 0},
    {0x3b8cb5d737710feeULL, 0xa3e174c191cd84baULL, 944, 16, 0},
    {0x3b8dd6687a089ed4ULL, 0x1905e10bc714a0d8ULL, 916, 16, 0},
    {0x3b9ea67c9f3a4ebfULL, 0x65e54e87ac9988f8ULL, 1029, 16, 0},
    {0x3b9f81f02242cbe6ULL, 0x8774d800d98d5d02ULL, 115, 13, 0},
    {0x3bdad018557f0021ULL, 0x5fb2ba30d76126daULL, 49, 13, 0},
    {0x3bfcfaf0436829e7ULL, 0xabefa34f5794c29aULL, 308, 16, 0},
    {0x3c0e3ee859d71b49ULL, 0xa4e95cd77453c062ULL, 184, 16, 0},
    {0x3c19bf0039957bd5ULL, 0xc89734a51f542f48ULL, 280, 16, 0},
    {0x3c26b45048d279e2ULL, 0x52e572b2526e8c3aULL, 822, 14, 0},
    {0x3c578d104b6edd7aULL, 0x971bb8082ade07f0ULL, 818, 14, 0},
    {0x3c63a45c0db82093ULL, 0xa31cf81b548cb328ULL, 411, 16, 0},
    {0x3cae211c7eaaa9d9ULL, 0xe08ad2cacf332bdaULL, 731, 14, 0},
    {0x3cd9fbb91e76aaa3ULL, 0x0b1100f265a6cffaULL, 633, 16, 0},
    {0x3cdc487b4866ea4cULL, 0xaf02bba22d8effdaULL, 466, 16, 0},
    {0x3ce51919e36f283eULL, 0x935dcc06aa50f834ULL, 823, 14, 0},
    {0x3d1323aa220f18cdULL, 0x2b58688cbccff4b2ULL, 816, 14, 0},
    {0x3d24c09ad1f7f688ULL, 0x842093fa15baacceULL, 671, 13, 0},
    {0x3d269c7abe20dc6dULL, 0x5dae03ab735632d0ULL, 606, 16, 0},
    {0x3d2a90dfe8992abcULL, 0x8c9f99dda6c16fceULL, 692, 13, 0},
    {0x3d2c1095334bfbcbULL, 0x1e5803978bf3a80aULL, 775, 14, 0},
    {0x3d906c5594469de9ULL, 0x962afc447eaf0838ULL, 420, 16, 0},
    {0x3daf5a92874c6b9aULL, 0x5059a2450d72211aULL, 53, 13, 0},
    {0x3dc7b0554c98c80cULL, 0x390563fd5c9ffdacULL, 47, 13, 0},
    {0x3ddf7d265826e705ULL, 0xb0e65949482d86daULL, 115, 13, 0},
    {0x3de056a336d042a9ULL, 0x461b0e6e18176382ULL, 752, 14, 0},
    {0x3de078b0691a909bULL, 0xd702a1ff03812a9cULL, 807, 14, 0},
    {0x3e0545bcd5679ca7ULL, 0xe7a63cc50fb91ce2ULL, 153, 16, 0},
    {0x3e34f3eaa0d24181ULL, 0x8312d99d2850d9daULL, 93, 13, 0},
    {0x3e3fc5753f3e91b6ULL, 0x37a3877966103926ULL, 704, 13, 0},
    {0x3e4f0e1c120f26b7ULL, 0x63189b1ffc06fbe2ULL, 609, 16, 0},
    {0x3e50b5cb925e8d5aULL, 0x8f85899b278bb16eULL, 846, 13, 0},
    {0x3e5cba67c66c7274ULL, 0x83eafe04a138698aULL, 835, 14, 0},
    {0x3e73246916c65c57ULL, 0x8ba9151d57357622ULL, 662, 10, 0},
    {0x3e8e3d696ed25cf7ULL, 0x499ebf3478a9d3caULL, 97, 13, 0},
    {0x3eb2fe86c327e7c9ULL, 0x50f5439587313748ULL, 311, 16, 0},
    {0x3ec348ccb52c2e63ULL, 0xeb74f17e8fd159baULL, 536, 16, 0},
    {0x3ee84bf26a55e048ULL, 0xe524ab1c4794f2faULL, 476, 16, 0},
    {0x3eec20dc06f1c0b5ULL, 0x215fb27ff90c0c08ULL, 162, 16, 0},
    {0x3eef5c6ce6655bdcULL, 0x2e03a86b7597f872ULL, 477, 16, 0},
    {0x3f06f1fa70367973ULL, 0x295df73b1ab26896ULL, 705, 13, 0},
    {0x3f0ba2252c996b50ULL, 0xeb810b25c613258aULL, 89, 13, 0},
    {0x3f0e5791e9f7e793ULL, 0x2ce89ca917124100ULL, 614, 16, 0},
    {0x3f1246bf7c3b9f77ULL, 0x64d93360e3209788ULL, 728, 14, 0},
    {0x3f3c7757cf280c9aULL, 0x7c830655a252e100ULL, 323, 16, 0},
    {0x3f82498808019a0eULL, 0x414d7c4fb36e51d0ULL, 283, 16, 0},
    {0x3f8313e10a9efffaULL, 0x45675d14eb4e7da2ULL, 202, 16, 0},
    {0x3f9a76eb7849d00eULL, 0x43ef8c986e1127a4ULL, 85, 13, 0},
    {0x3f9a9caef26836e2ULL, 0x8e44dc0f9d3c8870ULL, 538, 16, 0},
    {0x3f9c29b142fb2adcULL, 0x236661614afd990cULL, 804, 14, 0},
    {0x3fc3e33032f791aaULL, 0x0da84f1c76eee230ULL, 571, 16, 0},
    {0x3ffcacb29d905de6ULL, 0x9bd603417c50add0ULL, 543, 16, 0},
    {0x4023985c4baae7fbULL, 0xdf61f1858a9c2e0cULL, 81, 13, 0},
    {0x404298357c0aa9bcULL, 0xea37bc6e07de966aULL, 774, 14, 0},
    {0x404ad3130aef0543ULL, 0x147658e020ecd7d2ULL, 264, 16, 0},
    {0x4078d83ec05adbe2ULL, 0x4be1d8ecb3cb2d3aULL, 748, 14, 0},
    {0x40b71f379444fb07ULL, 0x4cf5ae4aa57b36d8ULL, 334, 16, 0},
    {0x40bd7630347707abULL, 0x31063c5e06e85e82ULL, 885, 16, 0},
    {0x40c8fb4ff5005b67ULL, 0x96a2cdbfaa3cdb64ULL, 43, 13, 0},
    {0x40c94ce65432ab76ULL, 0x7598ce87718298caULL, 799, 14, 0},
    {0x40e2ea9a3539eb59ULL, 0x4ad177cf10547eeaULL, 714, 14, 0},
    {0x40e5946e52815098ULL, 0x615cfef8b948185aULL, 806, 14, 0},
    {0x40f123d5da06c848ULL, 0x79ca549101b0e342ULL, 959, 16, 0},
    {0x412b378d5dacb244ULL, 0x25bb3a3f6daa430aULL, 760, 14, 0},
    {0x4155b0c278fff993ULL, 0x9e18d116801df6b2ULL, 802, 14, 0},
    {0x416d19306e0f16c2ULL, 0x7c3a7ac6ffd7c6c0ULL, 485, 16, 0},
    {0x418984890a66c181ULL, 0xf2edd466df072d30ULL, 549, 16, 0},
    {0x41998c8f9f7a9afcULL, 0xc1910116286670d2ULL, 207, 16, 0},
    {0x41f67edc2de2bfa0ULL, 0xc85e75dafea6451aULL, 715, 14, 0},
    {0x42136bce7bd4d831ULL, 0x3f7045961267cb22ULL, 58, 13, 0},
    {0x42187010f4e0080bULL, 0x9c43168d1e464ddcULL, 833, 14, 0},
    {0x422670a3ec45e94aULL, 0xe4c92eff9d4d5364ULL, 772, 14, 0},
    {0x422c2697eb0f1124ULL, 0x677f67568dab430aULL, 30, 13, 0},
    {0x4259d589e9e4c141ULL, 0xd2b3ed830f171f88ULL, 199, 16, 0},
    {0x428b5de3f4783a0cULL, 0x47689192044c4372ULL, 182, 16, 0},
    {0x42a5600f62178efbULL, 0x50814f751d649458ULL, 464, 16, 0},
    {0x42b4512015b1cd1fULL, 0xf6ad1ef94fc91394ULL, 804, 14, 0},
    {0x42ca6da194cb03f5ULL, 0xf597f1f2d6939758ULL, 764, 14, 0},
    {0x42cfcadcef33d082ULL, 0x3e39973710acc062ULL, 722, 14, 0},
    {0x431c879730a9f681ULL, 0xc0767b70419c995aULL, 731, 14, 0},
    {0x432503023ac71ca6ULL, 0x6997ef807ee88550ULL, 1038, 16, 0},
    {0x435c7ca5f6ac0367ULL, 0xbef5acfa53c20a1aULL, 28, 13, 0},
    {0x438acb9df1d4733bULL, 0x3d14cf54ded4e532ULL, 770, 14, 0},
    {0x43a7c9ff05fb1155ULL, 0x84025b54e6ffd782ULL, 909, 16, 0},
    {0x43b30a849e955161ULL, 0xfbb5f9751c2472baULL, 472, 16, 0},
    {0x43cec4285a652372ULL, 0x788c8e6531d2a292ULL, 68, 13, 0},
    {0x43d2f34ee5194a97ULL, 0x5f0acd90b5a032f8ULL, 139, 16, 0},
    {0x43dc55d86ee3377eULL, 0x5744959c3b887e6cULL, 55, 13, 0},
    {0x440a319a18424977ULL, 0x7a6b2393d69a27f2ULL, 801, 14, 0},
    {0x445d63991a49e44dULL, 0x5fa5cefd392d2eb8ULL, 198, 16, 0},
    {0x44639cebf3257d8fULL, 0x59fb14ec1aaf7facULL, 94, 13, 0},
    {0x4468a1ec830c552cULL, 0xe81bb280d351a998ULL, 225, 16, 0},
    {0x4492ebb517ae5b02ULL, 0x67dd6679991febd6ULL, 864, 13, 0},
    {0x44b6fc2d863f3a9dULL, 0x07e20bede6f3f9ecULL, 98, 13, 0},
    {0x44bac67660b87a44ULL, 0xb3801858cebf0018ULL, 517, 16, 0},
    {0x44c9fcb531bb8c83ULL, 0x520a29de2e1ae548ULL, 727, 14, 0},
    {0x44de30f5af58fa42ULL, 0x6f659bfa67f53eeaULL, 799, 14, 0},
    {0x44ed9c533d57c9a9ULL, 0xcab5dc21887b2a80ULL, 574, 16, 0},
    {0x4518a32d50b7b84dULL, 0x4c14457ec51b2f72ULL, 794, 14, 0},
    {0x452e448b97733514ULL, 0xeb0925f166a58512ULL, 246, 16, 0},
    {0x4537ad3ee5c90fbbULL, 0xd11943f6168bafd2ULL, 779, 14, 0},
    {0x453a74aad6f702feULL, 0x169528d433549c08ULL, 335, 16, 0},
    {0x455f45284d61976aULL, 0xd92f20371700c49aULL, 1005, 16, 0},
    {0x456167e6dbb59aa1ULL, 0xdd97cf2661928880ULL, 203, 16, 0},
    {0x45649259d31a05c1ULL, 0xa5367bcaff4c71e4ULL, 103, 13, 0},
    {0x4593b3b172b28350ULL, 0x7d7a5c28823c9192ULL, 890, 16, 0},
    {0x459cc2d98fe0798dULL, 0x0437869b98d4544aULL, 796, 14, 0},
    {0x45c774ec3a9649dcULL, 0x08f79b8ab1327b6aULL, 667, 10, 0},
    {0x45ef28023db3897fULL, 0xcd3e5315ca728a42ULL, 42, 13, 0},
    {0x4608b82c5cce32e9ULL, 0xfee823a1f98a7d08ULL, 1020, 16, 0},
    {0x4629791f43faeeceULL, 0x965981bbcbc26318ULL, 1028, 16, 0},
    {0x46359aa11194ce17ULL, 0x43c8185d1ed35d42ULL, 111, 13, 0},
    {0x46399ce6145b0d39ULL, 0xbc8b1ae89f76b222ULL, 635, 16, 0},
    {0x46635192ae946a6bULL, 0xec62ba8604273300ULL, 1016, 16, 0},
    {0x4680aba373136177ULL, 0x17de41deb31bb408ULL, 751, 14, 0},
    {0x46a1dfc001b8c870ULL, 0x940b6300309edd78ULL, 395, 16, 0},
    {0x46a498654d92a273ULL, 0xd845a7414cb38d36ULL, 850, 13, 0},
    {0x46ad955de4692cdcULL, 0x181cf26a738f0b68ULL, 165, 16, 0},
    {0x46d9226f7f69a3f0ULL, 0xe962fa12cb5904d2ULL, 769, 14, 0},
    {0x470f311ff5f43135ULL, 0x099f2c4388c7d2c2ULL, 915, 16, 0},
    {0x47372e4d62d011ddULL, 0x217bd6782052bfc8ULL, 156, 16, 0},
    {0x475ba67d9b0990a0ULL, 0x5934a8110b7f88c2ULL, 73, 13, 0},
    {0x4767e03cd7cc1b1fULL, 0x6405b1e1049ae290ULL, 590, 16, 0},
    {0x4776aacaf1cc314aULL, 0x3eb207ed5f181d7aULL, 805, 14, 0},
    {0x478365c207dde3e9ULL, 0xe7f534ae3406cceaULL, 151, 16, 0},
    {0x479d285dc0ed01ceULL, 0xcf6e6c2443438b54ULL, 809, 14, 0},
    {0x47e5bcdfd6eb3c09ULL, 0x51aa13c4814ece72ULL, 369, 16, 0},
    {0x47ef7325ad0f9d3cULL, 0xb2c5676a85449be2ULL, 454, 16, 0},
    {0x4801942124c1351aULL, 0x2b3aa822aa914f42ULL, 603, 16, 0},
    {0x48027ce4011e04faULL, 0x3121105f2bed0962ULL, 640, 16, 0},
    {0x48064ef279acbc10ULL, 0x3dafb9c562465506ULL, 870, 13, 0},
    {0x480a4c589625069fULL, 0x36442ac540abf6b0ULL, 1031, 16, 0},
    {0x48196b184e4a928bULL, 0xcf068472546e0062ULL, 718, 14, 0},
    {0x4819aaa9f544b9e6ULL, 0x201841dd2e8ebcfeULL, 853, 12, 0},
    {0x481b00d8e5978491ULL, 0xab8e77baa912f0f2ULL, 247, 16, 0},
    {0x48250d5d70ce1fa9ULL, 0x29ee3dbffd30d4aaULL, 732, 14, 0},
    {0x482881c5ab1391f4ULL, 0xbc5f8b5d72ba5614ULL, 809, 14, 0},
    {0x4830313383e7eec3ULL, 0x182c7b4ae2edf97aULL, 780, 14, 0},
    {0x4879180c4756892aULL, 0x9ec581a403c76140ULL, 726, 14, 0},
    {0x4893b7d136a3078cULL, 0xa786efdc6d3801e0ULL, 1036, 16, 0},
    {0x4896a79ba3806f99ULL, 0xc4eae15fc4a23a00ULL, 629, 16, 0},
    {0x48bb9b630b753d96ULL, 0xfd84342c22b52b62ULL, 821, 14, 0},
    {0x48c4914ad04097edULL, 0x97f0ff949b4485c0ULL, 535, 16, 0},
    {0x48d6b7876b00f555ULL, 0xd60fd5d9d644a160ULL, 738, 14, 0},
    {0x491d94e85cfd63c9ULL, 0x8bc77faa5a2c90d0ULL, 271, 16, 0},
    {0x492d55f1008051d1ULL, 0x047f9c1f63603a58ULL, 733, 14, 0},
    {0x493189c54e678a5cULL, 0xde7aa7b12a0a4c90ULL, 524, 16, 0},
    {0x494f894153f11b05ULL, 0x49131a31ddf51590ULL, 1023, 16, 0},
    {0x49a6e0e682652ba1ULL, 0x42c64052a94e2e38ULL, 504, 16, 0},
    {0x49d414ec7c8d56dcULL, 0x7340fb590c100162ULL, 163, 16, 0},
    {0x49d86c875c5edf09ULL, 0xfc4ec0c3bcde4142ULL, 780, 14, 0},
    {0x49eead684e66fc3dULL, 0x1f048b66d3469f10ULL, 725, 14, 0},
    {0x49f02b48e3cf8394ULL, 0x155f709f973fb4caULL, 645, 16, 0},
    {0x4a14cdcfc42cf9a4ULL, 0x587caf31bdc0affaULL, 128, 16, 0},
    {0x4a4e2a4ca3f473e4ULL, 0xb3018ef4c037863aULL, 802, 14, 0},
    {0x4a5fe9743660e77dULL, 0x116287e1d5d02252ULL, 176, 16, 0},
    {0x4a61dd87b525ada5ULL, 0xb4640845c9c0d1baULL, 451, 16, 0},
    {0x4a89973762f76dfbULL, 0x4dd57260735756b8ULL, 733, 14, 0},
    {0x4a90dfe2d10f05f4ULL, 0x879dcff9161f4ad4ULL, 63, 13, 0},
    {0x4a918a7b3ad79d4cULL, 0x318010d926b927daULL, 736, 14, 0},
    {0x4a992354f55424acULL, 0x190b1ac8ee3a639aULL, 910, 16, 0},
    {0x4ad370c97fe05bf3ULL, 0x6b873480bb4926caULL, 747, 14, 0},
    {0x4aeae2316a8e835eULL, 0x9df5a53fe7790678ULL, 947, 16, 0},
    {0x4aefa99b19254ce1ULL, 0x82979deba6190ac2ULL, 790, 14, 0},
    {0x4afbb634801b5745ULL, 0x2ff8ea53152d38d0ULL, 618, 16, 0},
    {0x4b0803aa711443d1ULL, 0x6c309df1333c3a52ULL, 577, 16, 0},
    {0x4b6d3118d34f09a3ULL, 0xfe45d617dfbe8970ULL, 562, 16, 0},
    {0x4b9aeaa939539b76ULL, 0xed67a54bf87bda98ULL, 360, 16, 0},
    {0x4b9d6b6113ebe36fULL, 0x71785d3d1f1c8be0ULL, 578, 16, 0},
    {0x4b9fe519d5898475ULL, 0x565d53f33ee5eb20ULL, 712, 14, 0},
    {0x4ba0a199a08dd97cULL, 0xcc1f31b6cfe538faULL, 96, 13, 0},
    {0x4babefd9f3b17b18ULL, 0x678ce53e5ab08bb8ULL, 786, 14, 0},
    {0x4bc482fb45f08384ULL, 0x04fddec4c28efdc2ULL, 73, 13, 0},
    {0x4bd72d4294547779ULL, 0x4980471860d5e198ULL, 901, 16, 0},
    {0x4bf116962918a62bULL, 0x734a3e9c3888d396ULL, 871, 13, 0},
    {0x4bf57b44592ba674ULL, 0xe8569ac7b9f02e08ULL, 497, 16, 0},
    {0x4c31391852bad105ULL, 0xf62cd9b5bcb2be8aULL, 806, 14, 0},
    {0x4c5d9c7e698fb811ULL, 0xd2003970088bd714ULL, 41, 13, 0},
    {0x4c653dc35ed0f5d3ULL, 0x5647d3e27d403122ULL, 960, 16, 0},
    {0x4c7db50002a136c6ULL, 0x9872c803e26f1fb2ULL, 445, 16, 0},
    {0x4c8f9e30a2c0ec9dULL, 0xa74c73e18edaa7faULL, 1009, 16, 0},
    {0x4c97073bf73cdc59ULL, 0x5c1d939ca4f1f4f2ULL, 801, 14, 0},
    {0x4c9cee99e0235667ULL, 0x4e5bfb55ae5444e2ULL, 502, 16, 0},
    {0x4ca17d5ada1b048cULL, 0xe36999dfe47abd6aULL, 122, 16, 0},
    {0x4ca2a830d232b425ULL, 0x91318d147cfd5d78ULL, 217, 16, 0},
    {0x4cacaf465465d798ULL, 0x279419a66da5032aULL, 666, 10, 0},
    {0x4cb6e3d48e7df10eULL, 0x766e2e6d8dd57d60ULL, 529, 16, 0},
    {0x4cfedb546b1e89f8ULL, 0x47d719295e011170ULL, 131, 16, 0},
    {0x4d067b8dc4e89dffULL, 0x1c8f7dd250712c30ULL, 628, 16, 0},
    {0x4d17b39f140085daULL, 0xc73814d9fef538c8ULL, 971, 16, 0},
    {0x4d53093eac7fb108ULL, 0x1affe698b10c9ce2ULL, 768, 14, 0},
    {0x4d8a8d8dbb0369d9ULL, 0xca26e8847b53356aULL, 718, 14, 0},
    {0x4d8f7e17edfe5955ULL, 0x9029776fd870bc7aULL, 91, 13, 0},
    {0x4d910adb97272564ULL, 0x85b57983cb00c872ULL, 945, 16, 0},
    {0x4d926055031eca2eULL, 0x898f9aaa8bda5592ULL, 265, 16, 0},
    {0x4d98971042619f4bULL, 0xf464a0d73bba5c50ULL, 288, 16, 0},
    {0x4d9e0cfcf101dccfULL, 0x30312404769a1efaULL, 799, 14, 0},
    {0x4df23884c1bae376ULL, 0x0780508a4bf21cb0ULL, 331, 16, 0},
    {0x4dfe8a3ee5d62822ULL, 0x8adfd56087525d98ULL, 351, 16, 0},
    {0x4e0404140e7d7b32ULL, 0xb7b237d08f0fbb08ULL, 1048, 16, 0},
    {0x4e226e7f59c0ba32ULL, 0x78e735b5c4524758ULL, 403, 16, 0},
    {0x4e3de54451be3962ULL, 0x4c39dab3016132e8ULL, 305, 16, 0},
    {0x4e5aef632f1c56bcULL, 0x1d1ad5f1f3c27decULL, 826, 14, 0},
    {0x4ea4db852cc93262ULL, 0x9391b117864ea098ULL, 927, 16, 0},
    {0x4ea5be45b3667e1cULL, 0x326131aa202690caULL, 87, 13, 0},
    {0x4eb903255f378a02ULL, 0x7861d4843fc9ebaaULL, 982, 16, 0},
    {0x4eb993efcc75de60ULL, 0x39d01227adf6fc08ULL, 136, 16, 0},
    {0x4ebc3f7a0c59bb56ULL, 0x9c060e836131c4c8ULL, 361, 16, 0},
    {0x4ecb679d78065d45ULL, 0x33be831b702b9272ULL, 758, 14, 0},
    {0x4edec8e5d0494817ULL, 0xacc012b77ee583b2ULL, 716, 14, 0},
    {0x4ee0983667270248ULL, 0xbcfd83d187ec19f0ULL, 1051, 16, 0},
    {0x4f1bb3236ddac00cULL, 0xd99640d02fca12b2ULL, 792, 14, 0},
    {0x4f24406f37fa2efcULL, 0x3fdaf9fb663810faULL, 788, 14, 0},
    {0x4f4fedc2ad545010ULL, 0x966a6e1200ad1342ULL, 37, 13, 0},
    {0x4f665b284ea01652ULL, 0x843c1595c418d9daULL, 743, 14, 0},
    {0x4f9935b8238e00ccULL, 0x200d3b4490f6fb10ULL, 1050, 16, 0},
    {0x4f9afdd97bc6095dULL, 0x887f8795bb9abdc0ULL, 920, 16, 0},
    {0x4faf66ee10b550f0ULL, 0x66a2cabae2af18d2ULL, 976, 16, 0},
    {0x4fafbf9e3ece5b8fULL, 0x7652b6d138c22f92ULL, 934, 16, 0},
    {0x4fd94a5a47f389deULL, 0x4bd28997e09c2740ULL, 540, 16, 0},
    {0x503834f173256277ULL, 0x7aed2a6272125dd2ULL, 498, 16, 0},
    {0x505f0af95cbd99fbULL, 0x9b08389e24dd1e22ULL, 506, 16, 0},
    {0x506a351b1472eb6eULL, 0x1b64800845f0a044ULL, 65, 13, 0},
    {0x507785b3fa830efaULL, 0xaa8d51f1be599b72ULL, 801, 14, 0},
    {0x50b564f7b61ee44cULL, 0xb69eba0a76076da2ULL, 206, 16, 0},
    {0x50f75a83063889fbULL, 0x0b66c03c563ed668ULL, 761, 14, 0},
    {0x50fbe4afc86f9a31ULL, 0xc7699e02c136dcb6ULL, 672, 13, 0},
    {0x51084be07d0359d5ULL, 0xe6a696ededbd85a8ULL, 157, 16, 0},
    {0x51131b70deb2235fULL, 0x2a0b2dfbae79e322ULL, 123, 16, 0},
    {0x511f79c41c131732ULL, 0xd822fccf786d1418ULL, 1029, 16, 0},
    {0x5121d80ee2077ee2ULL, 0x6c085e90d04e2932ULL, 943, 16, 0},
    {0x5128f196d61bd9fbULL, 0x7827d175e418395aULL, 451, 16, 0},
    {0x512e42ae5c4162b9ULL, 0xf7339bdfbc8ade30ULL, 721, 14, 0},
    {0x51330e586efef4f3ULL, 0x979f95bcdba54842ULL, 999, 16, 0},
    {0x514202c59f137c55ULL, 0x609911c4ab5135a8ULL, 897, 16, 0},
    {0x51701faba2043cedULL, 0xe949ea329301f3b4ULL, 823, 14, 0},
    {0x517256f6856d4492ULL, 0x2c2222059012503aULL, 736, 14, 0},
    {0x5175df53518392c5ULL, 0x3666e4b6567049f2ULL, 767, 14, 0},
    {0x5175f49a31ba7fd8ULL, 0xd6fc037d9bfca5c2ULL, 462, 16, 0},
    {0x519015b7e3b7636aULL, 0x7343112145b4d0e8ULL, 1047, 16, 0},
    {0x51c062a6bbc21343ULL, 0x3742d59630c8ae6cULL, 50, 13, 0},
    {0x51c847bcbe3aa777ULL, 0x4e179fd35257d252ULL, 554, 16, 0},
    {0x51cf7af95315a4f0ULL, 0xb8e9d1cac7fe8e9aULL, 480, 16, 0},
    {0x51d99db9a694122cULL, 0x23dc55c0dbe0a8faULL, 803, 14, 0},
    {0x51f165f85367e889ULL, 0x45106ebd7001aef2ULL, 806, 14, 0},
    {0x51f2fb5209b25a91ULL, 0xcc5f1dee438b7592ULL, 472, 16, 0},
    {0x51fe3280783eee34ULL, 0x8257c17e553498dcULL, 877, 13, 0},
    {0x52125d9827e3273dULL, 0x536834322f591278ULL, 924, 16, 0},
    {0x52257b30f0024435ULL, 0x88f8a97bd0b59b6aULL, 991, 16, 0},
    {0x5235348834545349ULL, 0x2958134a9faa4f78ULL, 955, 16, 0},
    {0x525bbdabfe71301aULL, 0x1f9a4c4ad70ff38aULL, 989, 16, 0},
    {0x526d5ed0b79f3d7eULL, 0x33ae9985815bdcbcULL, 807, 14, 0},
    {0x52789316a1c2273bULL, 0x2f92c3efe508cb18ULL, 416, 16, 0},
    {0x528bbbade516d34dULL, 0x44e90401d0a6f052ULL, 573, 16, 0},
    {0x52a7604126b05b59ULL, 0x836b31a6aba706a0ULL, 574, 16, 0},
    {0x52addbe71d2f5d99ULL, 0x9d1a941e665134f2ULL, 175, 16, 0},
    {0x52f5441840c1ce26ULL, 0x59d747e6516a7a90ULL, 595, 16, 0},
    {0x532d46609402f4e1ULL, 0x9233b989fa081bc0ULL, 761, 14, 0},
    {0x533130d7b126c565ULL, 0xd759d9f159e4a3dcULL, 772, 14, 0},
    {0x53657ec7debba520ULL, 0xa53c67a3d6825702ULL, 572, 16, 0},
    {0x5367f68ef0802eebULL, 0x74ba8635219b942aULL, 775, 14, 0},
    {0x53a8bba14c33b8abULL, 0x2c070c457d8771d2ULL, 798, 14, 0},
    {0x53d88cd5d0d79bb4ULL, 0xaefa97413ba7ce88ULL, 350, 16, 0},
    {0x53def8d140b3a30fULL, 0xe14545201f98ae38ULL, 1032, 16, 0},
    {0x53efef22d380f70aULL, 0xb683cb26d9aa971aULL, 531, 16, 0},
    {0x54008c6c98b8c480ULL, 0x7a572b3733ea0d98ULL, 161, 16, 0},
    {0x541139bc5beb9816ULL, 0x919e0bef3484f408ULL, 334, 16, 0},
    {0x5413dba4418bff6fULL, 0xa535348bd6a1cec0ULL, 590, 16, 0},
    {0x541f677b691f14f5ULL, 0x1ea1044686fae58aULL, 770, 14, 0},
    {0x54506f950af021daULL, 0xcdb6115d16506172ULL, 380, 16, 0},
    {0x545ad0ac16f943cdULL, 0xa152586083592f12ULL, 815, 14, 0},
    {0x547d6d79684dd6f0ULL, 0x20fc7f224074e0daULL, 575, 16, 0},
    {0x548b6038636a751cULL, 0xf8fe3a0f1499855aULL, 801, 14, 0},
    {0x54940e256f0409cfULL, 0x4cafef93c2de1542ULL, 226, 16, 0},
    {0x5499e4781fe0e41cULL, 0xb9c844af0bc1e858ULL, 393, 16, 0},
    {0x54c624c54e36b667ULL, 0x17b0baff8305730aULL, 301, 16, 0},
    {0x54d4eb678970e229ULL, 0x2a748a43bc92b07cULL, 772, 14, 0},
    {0x550e7e12fe961091ULL, 0x7f448edfb717c2caULL, 250, 16, 0},
    {0x550f332b1c10f2cdULL, 0x5d9ea9e3f36a72b6ULL, 675, 13, 0},
    {0x552519c06bd9cdd3ULL, 0xb0a4bad0cc308ee2ULL, 788, 14, 0},
    {0x5533393e6a462839ULL, 0xa4f99d9b41cd1ab0ULL, 412, 16, 0},
    {0x5538be78864f2b2fULL, 0x431c091487e5c0f2ULL, 903, 16, 0},
    {0x554dfd5202e77e7aULL, 0xa06aa30a54893184ULL, 59, 13, 0},
    {0x554ef8254b5386ddULL, 0xc3e8f15e904ce8caULL, 788, 14, 0},
    {0x556fc39dc22513c9ULL, 0x07be14cba7c0d2faULL, 1004, 16, 0},
    {0x557514266f73b952ULL, 0xbf1d414b607d74d8ULL, 393, 16, 0},
    {0x55b7c7ef4a729cecULL, 0xb6d27f85b17f7c12ULL, 514, 16, 0},
    {0x55b94dc789852a8bULL, 0xf9d09709b4c8d40aULL, 841, 10, 0},
    {0x55e429c5f31733ddULL, 0x2ee88107ff0a7838ULL, 946, 16, 0},
    {0x55fbf47fb404c49bULL, 0xb13f573eb24e2448ULL, 471, 16, 0},
    {0x560bbafe4b46c21eULL, 0x88ee9a568b79fbdeULL, 704, 13, 0},
    {0x561e967c1e0a38caULL, 0xb7304b5ad27efed8ULL, 726, 14, 0},
    {0x5628edd81fbf03caULL, 0x97a5b429b8517728ULL, 199, 16, 0},
    {0x5637f132c49c399cULL, 0xc507ab8573c85f1cULL, 106, 13, 0},
    {0x563e99529b7229afULL, 0xf6ddcd4d8559568aULL, 486, 16, 0},
    {0x5647a0503f2ac7bbULL, 0xec3d94fad65a435cULL, 878, 13, 0},
    {0x564a41b9e4488089ULL, 0x44c942a5bbb05f58ULL, 719, 14, 0},
    {0x564d70362a91c980ULL, 0x1c5d2f24d0a33d80ULL, 229, 16, 0},
    {0x564ef58a8fc93b48ULL, 0x18b1b030ea22a730ULL, 590, 16, 0},
    {0x565d1099653df1d8ULL, 0x83ab380e7e3b90d8ULL, 314, 16, 0},
    {0x565d81b8a9faa8feULL, 0xe5aca513d31d3312ULL, 268, 16, 0},
    {0x565dbca728ace4cdULL, 0x8c092690a2fe519aULL, 737, 14, 0},
    {0x566ea808707e966eULL, 0x4c7ece07876fd5a0ULL, 169, 16, 0},
    {0x5679212aee294799ULL, 0x66c2b7f11264d14aULL, 299, 16, 0},
    {0x567985084965604fULL, 0xb27d10c51889ff40ULL, 626, 16, 0},
    {0x567c7a3a788370a1ULL, 0xffd746bce9f620caULL, 577, 16, 0},
    {0x56aa201f82edfa93ULL, 0xfffdd68012821f1aULL, 828, 14, 0},
    {0x56b9c86d8986334bULL, 0xf980b7cf3400d8f0ULL, 1026, 16, 0},
    {0x56bf846d7c386b2fULL, 0x8d9ff32179af8422ULL, 855, 13, 0},
    {0x56d0cf2ea53d1844ULL, 0x976f4156ef331084ULL, 829, 14, 0},
    {0x5719651c589a29ffULL, 0xe05a013625050100ULL, 565, 16, 0},
    {0x572ff05f571fb1e2ULL, 0x6ee45f9a144c0facULL, 730, 14, 0},
    {0x573fa7836c51e92bULL, 0x231626beb8e5fe60ULL, 564, 16, 0},
    {0x576ca2c6dcc3ad3cULL, 0x83b6cb715de29852ULL, 147, 16, 0},
    {0x57cc945afc21adf4ULL, 0x840745128c337518ULL, 738, 14, 0},
    {0x57d26ffb210b4706ULL, 0x6f3e764e5f82411cULL, 35, 13, 0},
    {0x57d4b6b0ae1be63eULL, 0xc5ef38d4868fe03cULL, 813, 14, 0},
    {0x57d579023b191928ULL, 0xadde8f63683cf030ULL, 656, 16, 0},
    {0x57dae62b168b2118ULL, 0xd91661c93e8724c4ULL, 730, 14, 0},
    {0x57fd71a9f5c6ddc6ULL, 0xb6bed9bffa7c3736ULL, 837, 12, 0},
    {0x5862df5fe320c0a9ULL, 0x7a0395b356318238ULL, 405, 16, 0},
    {0x586c9112c693c496ULL, 0x77eb04198398af9cULL, 114, 13, 0},
    {0x589c1e161d577835ULL, 0x1918cfdeb26ae310ULL, 650, 16, 0},
    {0x58dc23001ba6f83eULL, 0x16f7b15b456c1ee0ULL, 1016, 16, 0},
    {0x58f64d8261fdd1d0ULL, 0x8cf540357089c742ULL, 742, 14, 0},
    {0x58f9aa69f1ac51bcULL, 0xecbc59afce3c5498ULL, 786, 14, 0},
    {0x59019a8b22d2f0adULL, 0x0aafeebe0a990966ULL, 682, 13, 0},
    {0x590e00fc91f2516cULL, 0xce589a365a670bdaULL, 201, 16, 0},
    {0x594f7595e793df34ULL, 0xc14866e1659d3cd2ULL, 762, 14, 0},
    {0x595d5eeb5526ccf7ULL, 0xf5cd951ad1945d32ULL, 769, 14, 0},
    {0x598c04c7d04fa5b2ULL, 0x816057b87260c1f0ULL, 140, 16, 0},
    {0x59a041777e748d30ULL, 0x033ba90967aaa382ULL, 661, 10, 0},
    {0x59a6b62c72a66839ULL, 0x887e8425cd5f84aaULL, 301, 16, 0},
    {0x59a73ec71034aa72ULL, 0xa92cc55142a3d8a0ULL, 784, 14, 0},
    {0x59adcb94a041aa3aULL, 0x692b6ab4fbde2860ULL, 637, 16, 0},
    {0x59c6b31b0c1371deULL, 0xfb6bec9122cc7a00ULL, 733, 14, 0},
    {0x59ce08e60f5d3783ULL, 0xd7fa234dbac31efaULL, 448, 16, 0},
    {0x5a04a89e5a37bcb0ULL, 0xf2dd67705ba08aa4ULL, 730, 14, 0},
    {0x5a2872b446e5a556ULL, 0x78459164fedd07caULL, 801, 14, 0},
    {0x5a2c441f8fba62efULL, 0x013c0ecdb4d171feULL, 679, 13, 0},
    {0x5a3d6be935b307f5ULL, 0xd8b3d4257751635aULL, 758, 14, 0},
    {0x5a468158a2f41f2dULL, 0xf049e1958abadb9eULL, 17, 12, 0},
    {0x5a736d6f8bbb328bULL, 0xce5eaae6a98a32f0ULL, 1045, 16, 0},
    {0x5a7930f6f7ca5604ULL, 0xde6eca4595d10d3aULL, 801, 14, 0},
    {0x5a7d9c8628a06700ULL, 0x64aade5c7282fdc8ULL, 405, 16, 0},
    {0x5a8741569a699ef5ULL, 0x7c6b80c79ce706baULL, 891, 16, 0},
    {0x5aaf87fc655dccb3ULL, 0x38733601c607fe12ULL, 289, 16, 0},
    {0x5ab645873f22f677ULL, 0xfed3be6e1aa8c892ULL, 646, 16, 0},
    {0x5acd28313aed07acULL, 0x5284a98f588631e2ULL, 660, 10, 0},
    {0x5acf36b55887b705ULL, 0xfba1893011d39648ULL, 307, 16, 0},
    {0x5ad066d362c0f1b9ULL, 0xdfe1fdb6812f10b0ULL, 354, 16, 0},
    {0x5ad32281c30408f4ULL, 0x7875fb909423d0b0ULL, 649, 16, 0},
    {0x5aedf0d1c2efd588ULL, 0x5740a789b27babf0ULL, 785, 14, 0},
    {0x5af395c63546925fULL, 0xa0ec8bdd5cfea5dcULL, 23, 13, 0},
    {0x5af976df60560332ULL, 0xad9856b1ef708bb6ULL, 14, 12, 0},
    {0x5afa63a9c91f13d5ULL, 0x91b68e22eb1efb80ULL, 783, 14, 0},
    {0x5b454eb592bc5eadULL, 0x0cb89f103a76b21aULL, 100, 13, 0},
    {0x5b506a871de5cad2ULL, 0xee0da0e4d4ceb10aULL, 793, 14, 0},
    {0x5b9eb5116523435eULL, 0x3d0bbdffb46bb86aULL, 292, 16, 0},
    {0x5bb74a7cf9cc43f1ULL, 0x13bcd4beb3a73522ULL, 635, 16, 0},
    {0x5bcbb54505f4cf2eULL, 0x7975eb1781f64ec2ULL, 770, 14, 0},
    {0x5bcfc5cffc3806c5ULL, 0x34c0539f6536f7d4ULL, 811, 14, 0},
    {0x5bf3cdc9fdede0dbULL, 0xd3946bd126f8590cULL, 79, 13, 0},
    {0x5c04e76f5e2e1b58ULL, 0xb535eef0e7ed413cULL, 814, 14, 0},
    {0x5c587b05f00ea578ULL, 0x39991685ce8874a2ULL, 410, 16, 0},
    {0x5c64f033042b88eeULL, 0xe3e8f2c48fceaa72ULL, 934, 16, 0},
    {0x5c6872a926a35058ULL, 0x555efb6a21ae42f8ULL, 224, 16, 0},
    {0x5c7d7bbc32b7ca88ULL, 0xe6332df72afdc182ULL, 187, 16, 0},
    {0x5cdaab11b621da45ULL, 0x3a329aaec00e7ee0ULL, 1037, 16, 0},
    {0x5d16d51bdc3cfd8dULL, 0x91119642579b2cdeULL, 697, 13, 0},
    {0x5d27e9c77a4ac488ULL, 0xaf51bd0683d5e56aULL, 768, 14, 0},
    {0x5d67b754afb20e0eULL, 0x47c128c26fb10120ULL, 626, 16, 0},
    {0x5d70a605ec69d048ULL, 0x5f25af32cbee7d5aULL, 385, 16, 0},
    {0x5d8bd20681974e7fULL, 0x66b7414e211bde0aULL, 768, 14, 0},
    {0x5d924625efcd5558ULL, 0x24f807a4620d5240ULL, 553, 16, 0},
    {0x5d9b9f4bec81ad18ULL, 0x271e53e90418b1daULL, 492, 16, 0},
    {0x5e130c9f75beb3f2ULL, 0xbf056ae2e3418df2ULL, 779, 14, 0},
    {0x5e1ae46b7010d29dULL, 0xc8165152b91672daULL, 906, 16, 0},
    {0x5e27337cfa53a1a0ULL, 0x48b3b7608ef8124aULL, 75, 13, 0},
    {0x5e463a695b3e00d9ULL, 0xbc1f146bfb91d830ULL, 296, 16, 0},
    {0x5e7b9e008d6ca1eaULL, 0xba3a5ecc6ea20fa8ULL, 1020, 16, 0},
    {0x5e953a44eff05d84ULL, 0x8b73c0ecaac1d2c4ULL, 830, 14, 0},
    {0x5e96966527f96502ULL, 0x0447ec403ccd115aULL, 435, 16, 0},
    {0x5e97c6d0cb462134ULL, 0x29e017fc13ec93c2ULL, 754, 14, 0},
    {0x5eb2de4aa120c06fULL, 0xef4f545b6d41cb68ULL, 178, 16, 0},
    {0x5eb6386b5b2b1ae2ULL, 0xadfee1185a3270b0ULL, 899, 16, 0},
    {0x5ebcbf1b3f93203eULL, 0x5291b2f9f1ba3b30ULL, 649, 16, 0},
    {0x5ec127efae04121eULL, 0xfe75abfe6c93914aULL, 195, 16, 0},
    {0x5ec1c21e7ae4ee2dULL, 0x8fbd0186300362daULL, 297, 16, 0},
    {0x5ed59eb60d1c7dd3ULL, 0x955d1db676305e64ULL, 875, 13, 0},
    {0x5ee7f93a47f933d3ULL, 0x7a60b3b50212964cULL, 51, 13, 0},
    {0x5ef06a3764fa2fb1ULL, 0x29afa7364e49b08eULL, 677, 13, 0},
    {0x5efe0f49aacf3971ULL, 0x43a8538a4729da80ULL, 150, 16, 0},
    {0x5f0b87b99c9c65f5ULL, 0xb4bb629df9827660ULL, 561, 16, 0},
    {0x5f16cee51be09535ULL, 0xbb9bca7b5b767700ULL, 1037, 16, 0},
    {0x5f196a052bf5ccd1ULL, 0x0d07e10ccb9d3520ULL, 628, 16, 0},
    {0x5f196f83085295c6ULL, 0xb73075598bddfc7cULL, 804, 14, 0},
    {0x5f2a257267439c5cULL, 0x83309537cad274caULL, 799, 14, 0},
    {0x5f40512eb240c925ULL, 0x92eb352716e485a0ULL, 1050, 16, 0},
    {0x5f5ebd9616f6356eULL, 0x66868bdf9e2a8a76ULL, 672, 13, 0},
    {0x5f631c260ac4092aULL, 0x9133faa44a830b7aULL, 742, 14, 0},
    {0x5f9c2f5cb0a9b050ULL, 0xabd8b9ba43e585c6ULL, 691, 13, 0},
    {0x5fad35255f97a7d4ULL, 0x23f0dcb1430e04c8ULL, 305, 16, 0},
    {0x5fb09ee05871212bULL, 0x255b97fd061ea100ULL, 488, 16, 0},
    {0x5fdf3741424f75a1ULL, 0x36f7708f8a146358ULL, 224, 16, 0},
    {0x5fed7debc0e92bdbULL, 0x4574f705220ad2a2ULL, 821, 14, 0},
    {0x6002b8914b9ac4f7ULL, 0x296f398a31c87fe2ULL, 749, 14, 0},
    {0x600937b4693c8132ULL, 0x2a87463431c8b97aULL, 24, 13, 0},
    {0x602c5cfa4518c3fcULL, 0x6fea1f869824a42aULL, 286, 16, 0},
    {0x603dbb494e1eef70ULL, 0x34ea0ad152cbd022ULL, 997, 16, 0},
    {0x6041ade700b50b9eULL, 0x03267ee9fbe12642ULL, 444, 16, 0},
    {0x60422f7a232acf54ULL, 0xedcfcaff945fd5a0ULL, 655, 16, 0},
    {0x6077a0bd00dbaaeaULL, 0xe4c0db94d659b5aeULL, 851, 13, 0},
    {0x60a078aada74d02aULL, 0xbf48b5bb6a430c6cULL, 730, 14, 0},
    {0x60a703b378bfac24ULL, 0x17372e10c623049aULL, 46, 13, 0},
    {0x60cf73476a171911ULL, 0x6be1a837d5d31532ULL, 312, 16, 0},
    {0x61062febf14bb378ULL, 0x2d20c32b8924b032ULL, 931, 16, 0},
    {0x612fc173af7e830cULL, 0x4fd617151ac710f8ULL, 759, 14, 0},
    {0x61467596ce470848ULL, 0x6bf1518c057d08a0ULL, 150, 16, 0},
    {0x615cf41882ef3f3bULL, 0xb24260f7149ecc7aULL, 907, 16, 0},
    {0x61b90b377799ed7cULL, 0xf02ba4f5d726fb22ULL, 159, 16, 0},
    {0x61cf8d027a802aceULL, 0x06dcb82ea196b7c2ULL, 456, 16, 0},
    {0x6208d69e55534e9aULL, 0x5f228cf369ef574aULL, 113, 13, 0},
    {0x620b872cb42e7e8eULL, 0x2f2f101269467272ULL, 477, 16, 0},
    {0x6238dc2b817ad8c0ULL, 0xe324904da593eb50ULL, 973, 16, 0},
    {0x6266c69bc70718fbULL, 0x680789b778e128b8ULL, 356, 16, 0},
    {0x629e4b481c9f0d3dULL, 0xe8defc8f93a86f92ULL, 608, 16, 0},
    {0x629ed12f66526b97ULL, 0xa846580787ba251cULL, 881, 13, 0},
    {0x62b28c7fc875901dULL, 0x849d835f6e9ed392ULL, 389, 16, 0},
    {0x62ecf87941b41361ULL, 0xe205387c0650c092ULL, 397, 16, 0},
    {0x62eeceb8db6728f3ULL, 0x1fb7cd57a9cf435aULL, 304, 16, 0},
    {0x631089ce0afacc18ULL, 0x5db2b14b150f40c8ULL, 751, 14, 0},
    {0x63165b1a532673c8ULL, 0xb2d1694be0faf1caULL, 794, 14, 0},
    {0x631d6cc0ce937a0bULL, 0x702fac67b0d618b8ULL, 1040, 16, 0},
    {0x632dcf4f728adbc4ULL, 0x453e513cb72b139aULL, 722, 14, 0},
    {0x6351849ebdb01dd5ULL, 0x973832db4e41cae0ULL, 632, 16, 0},
    {0x6375128fb91e86afULL, 0x26be2dded245873aULL, 806, 14, 0},
    {0x637cc376917a6610ULL, 0x0099bac7852b079cULL, 813, 14, 0},
    {0x63b0abb48dbedb52ULL, 0x87260114bd696b30ULL, 285, 16, 0},
    {0x63dce1c4cb77d8ceULL, 0xf7ede91ba744ec4aULL, 484, 16, 0},
    {0x63de4e7d7955ccf2ULL, 0x6db23ea7f4212622ULL, 762, 14, 0},
    {0x63fde739b3e14bd9ULL, 0xf9ede90c41ca9718ULL, 896, 16, 0},
    {0x6405cf36ad53c4c5ULL, 0xde6cc60c0bca8242ULL, 615, 16, 0},
    {0x640e68daaa54729aULL, 0xeb3ac5120c08ace2ULL, 603, 16, 0},
    {0x641634417d7d8da1ULL, 0x148b697278cbb91cULL, 813, 14, 0},
    {0x642a87b23844f4e0ULL, 0xaf4b734feb43d0eaULL, 483, 16, 0},
    {0x643f930e8ef8ab69ULL, 0x39c4ba75b8630fdaULL, 758, 14, 0},
    {0x64637f3447d45284ULL, 0x463a6a457e7f9c30ULL, 331, 16, 0},
    {0x646892c04ab3507cULL, 0x8015e5d925c84378ULL, 316, 16, 0},
    {0x648731cb1df4406bULL, 0xcdf4a61990a78d7aULL, 95, 13, 0},
    {0x64aa3b5126f2f8cdULL, 0x7d26200db7b22478ULL, 800, 14, 0},
    {0x64aee0776814371aULL, 0x750b9ed625e7241aULL, 828, 14, 0},
    {0x650495cd314420b0ULL, 0x875d1a50af75e7e2ULL, 739, 14, 0},
    {0x65204e6bcbdd2ba0ULL, 0x0f426bd5067ac8a8ULL, 787, 14, 0},
    {0x652343b5ea296a16ULL, 0xb281ed9c7d1a3fc8ULL, 504, 16, 0},
    {0x6532dfc912889b1dULL, 0xda8388c452d8b63aULL, 507, 16, 0},
    {0x653abee43cca04c9ULL, 0xde561acec5b2dd1cULL, 823, 14, 0},
    {0x653c3dd737ea6fbaULL, 0x41f70ba149543de2ULL, 37, 13, 0},
    {0x65689b736d1c97a0ULL, 0x526721162ad64252ULL, 646, 16, 0},
    {0x65981a8da75b8867ULL, 0x90e37010860d9e0aULL, 742, 14, 0},
    {0x659bd8939756f32cULL, 0x3db9144a58e8a22eULL, 865, 13, 0},
    {0x6602ded16d3315aeULL, 0x605e306831a40478ULL, 468, 16, 0},
    {0x66179e4404f7aca5ULL, 0xd11bcbc580f89728ULL, 258, 16, 0},
    {0x6636ad2a8b7d98e4ULL, 0x5e8c259a225d45deULL, 868, 13, 0},
    {0x6666decf6161424dULL, 0x624ec17137ac0364ULL, 60, 13, 0},
    {0x66673f731506db8cULL, 0x710c06a6334bb654ULL, 117, 13, 0},
    {0x6673619405d0b83cULL, 0x8477efa93ece0cccULL, 62, 13, 0},
    {0x6674a504aa29fb25ULL, 0xbb978a71be1475b2ULL, 67, 13, 0},
    {0x66939abc593a7163ULL, 0x6a4c2c43e2672aa2ULL, 559, 16, 0},
    {0x66bc324bf877d719ULL, 0x91a750e0c79a0d52ULL, 264, 16, 0},
    {0x66c86ba13af6c622ULL, 0xa773df7a6c97a9e8ULL, 733, 14, 0},
    {0x66c87f575d52d1d5ULL, 0x5bb6a03104897048ULL, 902, 16, 0},
    {0x66ce2ad7d4f5e09dULL, 0x935c9d373f80f1f0ULL, 583, 16, 0},
    {0x66e11468e1413aafULL, 0x65751d73df42313aULL, 825, 14, 0},
    {0x6713e19c914da0f0ULL, 0x9b6115310abbce22ULL, 821, 14, 0},
    {0x6716c5b51dd82815ULL, 0x0aa245c20c22f09aULL, 889, 16, 0},
    {0x6729eca8a0399b71ULL, 0x929656431df9dbf8ULL, 501, 16, 0},
    {0x67427a9383c65a10ULL, 0x923ab54015d3dbfeULL, 0, 6, 0},
    {0x6749e560fe3ada8aULL, 0x0f51bb0a202811faULL, 434, 16, 0},
    {0x67534d0c6789e873ULL, 0x66ec6cd405a53eb2ULL, 816, 14, 0},
    {0x675a039f5c4c3beaULL, 0x0321ddcbf77364a0ULL, 1014, 16, 0},
    {0x67aa766b1e7fffc2ULL, 0xebf0a9c9f3d557a8ULL, 1019, 16, 0},
    {0x67edc068743228a3ULL, 0x6049c6acac0a9522ULL, 163, 16, 0},
    {0x67eeabcb4522c823ULL, 0x9d49c0169b4548baULL, 731, 14, 0},
    {0x680cae4e005e8b52ULL, 0x868a7e5713780db2ULL, 566, 16, 0},
    {0x68166bf27f0415ddULL, 0xd720564b01db20b8ULL, 727, 14, 0},
    {0x682768a679ddee51ULL, 0xac4481ef36688494ULL, 741, 14, 0},
    {0x683e7a685df06e67ULL, 0xd1e5ecbe25070830ULL, 558, 16, 0},
    {0x684576082e72e430ULL, 0x5c738d2ad858d142ULL, 530, 16, 0},
    {0x68691ec67e4233d5ULL, 0x994e5a2611cd8c4aULL, 110, 13, 0},
    {0x6896a0fbb521f37cULL, 0x32e0a64ccc35dca0ULL, 547, 16, 0},
    {0x68b0fe1d2adeab4cULL, 0xe97221df5a9416faULL, 734, 14, 0},
    {0x68dd403aa2d8ff44ULL, 0xb817be4543eb0746ULL, 696, 13, 0},
    {0x693f0e7d7aaf6a19ULL, 0x1fd1a1754482aec0ULL, 723, 14, 0},
    {0x693fecc0d448a255ULL, 0x6799778cf429ed68ULL, 394, 16, 0},
    {0x695488a92d86875dULL, 0x3e4776d383b62f92ULL, 435, 16, 0},
    {0x6955504e74b42adeULL, 0x7a3576bc113cf6ccULL, 114, 13, 0},
    {0x699c1af83620e73dULL, 0x859aef51d9ff82baULL, 948, 16, 0},
    {0x69c883527cb7f2c2ULL, 0x3fba2c8693e31e82ULL, 505, 16, 0},
    {0x69ddc901cbfb9e7aULL, 0x8a228d9deb7e7122ULL, 82, 13, 0},
    {0x69f8126d04643a3cULL, 0x035614d2e0cde758ULL, 339, 16, 0},
    {0x69faf73959c1d07cULL, 0x9f9d7e0e9acf111aULL, 138, 16, 0},
    {0x6a0d242e9b65e5baULL, 0x8c4e84f09113f88aULL, 324, 16, 0},
    {0x6a199c45e46fb1fcULL, 0xf6c339589f96ac80ULL, 759, 14, 0},
    {0x6a1f23795b2a2faaULL, 0xd51acc71952507caULL, 148, 16, 0},
    {0x6a2871bd53b00759ULL, 0x1932291955ef3942ULL, 746, 14, 0},
    {0x6a7b6f37c51a287bULL, 0x775cdc4b3ab85d0aULL, 732, 14, 0},
    {0x6a83395e9919ce6dULL, 0x8566937b32ed7bd0ULL, 937, 16, 0},
    {0x6aafa956924d5f58ULL, 0xbb80bff850d5b338ULL, 921, 16, 0},
    {0x6ac2b1f37cca90a8ULL, 0x5387ac3281d8a5b2ULL, 260, 16, 0},
    {0x6aed074d2de7d3b6ULL, 0x3282dced3f7e1e28ULL, 242, 16, 0},
    {0x6af5e4ae8639120eULL, 0xae0936146c8d3408ULL, 335, 16, 0},
    {0x6b16f2c739a4cf21ULL, 0xf3bcaf4eedcc4e88ULL, 276, 16, 0},
    {0x6b488bb3eccad271ULL, 0xa2a6098b06126d3eULL, 680, 13, 0},
    {0x6b9a2ff5bd442793ULL, 0x64a14a0e3facad72ULL, 1010, 16, 0},
    {0x6bb6ebdc273a8f14ULL, 0x7d4b6512458a9f72ULL, 729, 14, 0},
    {0x6bcd2f7e65c12aa4ULL, 0x603569ca19697870ULL, 738, 14, 0},
    {0x6bd3a73ad3e1e4ebULL, 0xf2b5d79da2edbd78ULL, 461, 16, 0},
    {0x6bdc9a052f01acd1ULL, 0xf5ebef175371a188ULL, 414, 16, 0},
    {0x6bddcb1cbe30b4d9ULL, 0x0bdf4195ba39556eULL, 706, 13, 0},
    {0x6be62342394a044fULL, 0x344147bf8c75c040ULL, 142, 16, 0},
    {0x6bf103e91fd20bfdULL, 0x9cdcbe1a1713724aULL, 747, 14, 0},
    {0x6bf9f3a037855f4fULL, 0x3aa888514bb73b4aULL, 858, 13, 0},
    {0x6c15d7234562e756ULL, 0x62ca99d7e4697a16ULL, 842, 13, 0},
    {0x6c17dc2c94468dd5ULL, 0xa5a793cb509da4a2ULL, 82, 13, 0},
    {0x6ca3418a60a25102ULL, 0xc310893665b0564aULL, 799, 14, 0},
    {0x6ccd6caf8c937d53ULL, 0x16e3a15d08942b20ULL, 528, 16, 0},
    {0x6cdb39dcf92c71dcULL, 0x264fb64140d97262ULL, 313, 16, 0},
    {0x6ce6a24d79968b04ULL, 0x6258d11ad3420460ULL, 203, 16, 0},
    {0x6cece8052ef90e97ULL, 0x42c0d400e77f98d0ULL, 293, 16, 0},
    {0x6d183d6c2c609daaULL, 0xbe992366d4720b9aULL, 371, 16, 0},
    {0x6d2cc48dc87e3b46ULL, 0xced3388ac2e6db9eULL, 670, 13, 0},
    {0x6d379b760f130dc5ULL, 0xec04d0c4c7e78c98ULL, 356, 16, 0},
    {0x6d6b10538d44e177ULL, 0xde24216289be9876ULL, 699, 13, 0},
    {0x6d8f25abccc41001ULL, 0xad05a38134a5218aULL, 981, 16, 0},
    {0x6daed9fc800f09f4ULL, 0x957cf4926f805e90ULL, 588, 16, 0},
    {0x6db714b60bdebf78ULL, 0x97b85fcd76ecd218ULL, 1027, 16, 0},
    {0x6de129a5e1b7bfe5ULL, 0x3ce31a134758e63aULL, 596, 16, 0},
    {0x6e02733bfababb2eULL, 0xd73ed2e11580d750ULL, 937, 16, 0},
    {0x6e03085d904bcd3eULL, 0xc1f02fe4d9dfe162ULL, 821, 14, 0},
    {0x6e04ce72bc0c9e93ULL, 0x9d139a5c29443282ULL, 782, 14, 0},
    {0x6e0966655a11985fULL, 0x2ed34f2f64ed6542ULL, 802, 14, 0},
    {0x6e18596f8a9a0756ULL, 0x13f0726ad1777198ULL, 145, 16, 0},
    {0x6e2a3e11b8df028fULL, 0x0a69c1627c23c2b0ULL, 1021, 16, 0},
    {0x6e33aba4c16ee6f1ULL, 0x91752f2faaed0078ULL, 441, 16, 0},
    {0x6e3c79941fe2f93eULL, 0xc341a286c29f2362ULL, 906, 16, 0},
    {0x6e3fe11d4372d4f7ULL, 0x93ceee5725d62d36ULL, 689, 13, 0},
    {0x6e98c1a2663f6f7fULL, 0x1b1136404573d642ULL, 470, 16, 0},
    {0x6eb0825b120c9a84ULL, 0x81f9f43e67f78934ULL, 804, 14, 0},
    {0x6ebcf0c1a147867dULL, 0x904f67d680329c5aULL, 192, 16, 0},
    {0x6f050a6b87d6b9bdULL, 0x8e4f576092da7398ULL, 417, 16, 0},
    {0x6f3291fc7b7e1cf2ULL, 0x642259d2b98e9160ULL, 553, 16, 0},
    {0x6f59367a0057f1bfULL, 0xe097f667e2bf1bc0ULL, 266, 16, 0},
    {0x6f7126f6c0c9c776ULL, 0x51cc484f8f377b1aULL, 779, 14, 0},
    {0x6f9003ac2a23a046ULL, 0x67648d1ade3522f2ULL, 260, 16, 0},
    {0x6f9a1e5eea96dff3ULL, 0xe06f8b231e4e7978ULL, 712, 14, 0},
    {0x6fa2c8a128347f17ULL, 0x1b030f417b922728ULL, 164, 16, 0},
    {0x6fc8bbf0298969fcULL, 0xd240a5bb9ef246f0ULL, 723, 14, 0},
    {0x70602e9819ec970cULL, 0xae02d2c3011a3348ULL, 334, 16, 0},
    {0x7082e183809bb07fULL, 0x66c1c0869bc7c0b2ULL, 443, 16, 0},
    {0x7087fb0c1d18a324ULL, 0x8cd5e78465bb62baULL, 452, 16, 0},
    {0x708ff660773c2306ULL, 0x61cd7dccb74df972ULL, 303, 16, 0},
    {0x70a7a601dee07668ULL, 0xfa418b6a06f39c80ULL, 900, 16, 0},
    {0x70b5405e69a87248ULL, 0xd2949f93ed5705a0ULL, 325, 16, 0},
    {0x70b9405ca6be40c8ULL, 0xc1e21c31e6570480ULL, 210, 16, 0},
    {0x70c53b3aec6d44c7ULL, 0xa3a262de94ec1116ULL, 708, 12, 0},
    {0x70c7930254d76ce9ULL, 0x9a516341e4d8f196ULL, 851, 13, 0},
    {0x70d3a74b1fd3f0d2ULL, 0xb82adb584e0e3870ULL, 611, 16, 0},
    {0x70d49e4b9d097074ULL, 0x5c008d17a4b8e748ULL, 137, 16, 0},
    {0x70e73dfd414a85e8ULL, 0x5216fb6e3c6cc41aULL, 729, 14, 0},
    {0x70ee750d900dbadfULL, 0x52342441bb72a0eaULL, 1008, 16, 0},
    {0x70f9201fecd97c2eULL, 0xabb787677e635f0aULL, 756, 14, 0},
    {0x70fea98c48451383ULL, 0x9596ee5ea29c112cULL, 829, 14, 0},
    {0x710d3e65c79f7f27ULL, 0xdd6391435b0cf7a8ULL, 314, 16, 0},
    {0x710e439efc0f5304ULL, 0x9892857cf3bd8510ULL, 211, 16, 0},
    {0x7117533a6059d1feULL, 0x5a560b0f582acd1aULL, 105, 13, 0},
    {0x714c533748db4370ULL, 0x81517dd8574b0f18ULL, 482, 16, 0},
    {0x71887563279cd310ULL, 0x9100602dfbe17f20ULL, 949, 16, 0},
    {0x71c27eafd71199a8ULL, 0xccb2746d8b39e408ULL, 156, 16, 0},
    {0x71e3da91aeeb42b7ULL, 0xf3bf2bd4f80a16d2ULL, 515, 16, 0},
    {0x71fb798d6cabc042ULL, 0xfdf0c4b746c91372ULL, 1005, 16, 0},
    {0x721a9fa8e781d61dULL, 0x47ebf56415721f18ULL, 474, 16, 0},
    {0x721aeea3ea997cdcULL, 0x2714ca8af1b36982ULL, 793, 14, 0},
    {0x7226cdfa2b8cc4e3ULL, 0x553c08a75c185448ULL, 974, 16, 0},
    {0x7230a351d1fe522cULL, 0x061b4a7065fb4230ULL, 269, 16, 0},
    {0x724966d43d72c115ULL, 0xa199c11cb482fc38ULL, 787, 14, 0},
    {0x727a966ae25ec06bULL, 0xf9368843ac6b65aaULL, 167, 16, 0},
    {0x72943ef58f4c49feULL, 0xcd1d29b0a372dfb8ULL, 404, 16, 0},
    {0x72bd88544433e005ULL, 0x8d343fe115ebfd30ULL, 485, 16, 0},
    {0x72c01809584a6259ULL, 0x36d25197b331885aULL, 627, 16, 0},
    {0x72deca12c09d6250ULL, 0x9ce4d0b846c3d71aULL, 779, 14, 0},
    {0x733d940ef5bd8734ULL, 0x1d304b4fc519b59aULL, 788, 14, 0},
    {0x734f73757aa0d731ULL, 0x305055bf0bb12cd8ULL, 387, 16, 0},
    {0x7353a023b1bf4f4fULL, 0x430a3da95bc584aaULL, 452, 16, 0},
    {0x73763095f61ddbb9ULL, 0x6912b1bc622df372ULL, 173, 16, 0},
    {0x738b557ef145adf0ULL, 0x83bda4b41fe02f58ULL, 277, 16, 0},
    {0x73aca07283ace1b9ULL, 0xf80b7a4053c99582ULL, 791, 14, 0},
    {0x73b60de22895d498ULL, 0x73996465683d3eccULL, 56, 13, 0},
    {0x73ca9ec529e4bc64ULL, 0x3cef0966a23aff1aULL, 95, 13, 0},
    {0x73d6b33bcba40cceULL, 0xad286ec04197d7d2ULL, 533, 16, 0},
    {0x73ddcfb0ad27cb5aULL, 0xf7ed7bb8d7586e5aULL, 454, 16, 0},
    {0x73f3b9c004f49dd7ULL, 0x38ce4c707024f1b0ULL, 135, 16, 0},
    {0x74282f22a7cb9efcULL, 0xaebd6457a7d312e8ULL, 738, 14, 0},
    {0x745b088a556f5493ULL, 0x12c4483e9f92a428ULL, 384, 16, 0},
    {0x74893b81d61b39e4ULL, 0xc0e4b65a024e23d2ULL, 729, 14, 0},
    {0x748a2087420e35aeULL, 0x1f2ad2c86775c3baULL, 982, 16, 0},
    {0x74955063078d1a25ULL, 0x44a7a8ab2d623698ULL, 418, 16, 0},
    {0x7496c396a896a11dULL, 0x4c9604a69716eca4ULL, 876, 13, 0},
    {0x74fde9746304e6cdULL, 0x4bf5aec110121f42ULL, 168, 16, 0},
    {0x7525f92cd131d5ecULL, 0xb6acdea3123c45b6ULL, 699, 13, 0},
    {0x7545a32400246103ULL, 0xec0e8454fe3d9f8aULL, 760, 14, 0},
    {0x754d277d22c1ef82ULL, 0xa48324d1220763a2ULL, 95, 13, 0},
    {0x7554f8e5efb920c6ULL, 0x4a4ee43dbc61cb0cULL, 56, 13, 0},
    {0x75574b22fb3e8dd6ULL, 0xa45773908bb5fd18ULL, 180, 16, 0},
    {0x756c253e8323d31bULL, 0x8ef6aa16052d95caULL, 30, 13, 0},
    {0x7579ea74738c9b1dULL, 0x0a3227d69fcbe7baULL, 567, 16, 0},
    {0x762b51095b30052eULL, 0xd035afbe007e5560ULL, 556, 16, 0},
    {0x7644f9ace61c91d4ULL, 0x5c492acc8b02c73aULL, 91, 13, 0},
    {0x764dda41fcb945b2ULL, 0x42d17fbe5caba252ULL, 767, 14, 0},
    {0x7652ad7a910c7191ULL, 0x534cbd7873810b84ULL, 39, 13, 0},
    {0x7653182ff23e23e7ULL, 0x92c65a2628756af0ULL, 248, 16, 0},
    {0x7670ef3bfe332ca3ULL, 0x915842e4d316a4c8ULL, 933, 16, 0},
    {0x76c9b53d62c81d3eULL, 0x21ee4952b4b42e38ULL, 429, 16, 0},
    {0x76cf56d65197a19dULL, 0x669d0d8207db5152ULL, 1008, 16, 0},
    {0x76e474c417c303b6ULL, 0x82fc99a1c5818970ULL, 124, 16, 0},
    {0x76e68807bdf32abaULL, 0xf19eb3faa8397da4ULL, 40, 13, 0},
    {0x76e8d870ada9bff1ULL, 0xe56d0f9930a3443aULL, 237, 16, 0},
    {0x76f6bf2ca5ddccd9ULL, 0x49a58af68422e36aULL, 176, 16, 0},
    {0x7710a9b18c3a9849ULL, 0xc75523a394e74cdaULL, 642, 16, 0},
    {0x772cdde97c332836ULL, 0xec088a7c59aadff8ULL, 146, 16, 0},
    {0x7754e7c775251ea8ULL, 0xff74ac8b083935d8ULL, 721, 14, 0},
    {0x77672f11b9102593ULL, 0x6aef02e06ae2ce62ULL, 58, 13, 0},
    {0x77687e21440a8c3bULL, 0xb5793d44325a261aULL, 48, 13, 0},
    {0x7786b6a8a25c8c42ULL, 0xbfa10c91b1d97d12ULL, 551, 16, 0},
    {0x779c43a0f6773246ULL, 0xfa9be6f010ddd642ULL, 802, 14, 0},
    {0x77a0ac504a70bb8aULL, 0xebe6c6c444694a7aULL, 997, 16, 0},
    {0x77adf170394cb92cULL, 0x2d696dca151399a8ULL, 307, 16, 0},
    {0x77c44f19262e28b9ULL, 0x6be64cc187de9968ULL, 713, 14, 0},
    {0x77ed536d21c2ebcaULL, 0x7949116728b1dd32ULL, 148, 16, 0},
    {0x7800ec78c25104caULL, 0x750ca14b17fe8132ULL, 408, 16, 0},
    {0x78370b947022c286ULL, 0x171bac8e86d90aaaULL, 526, 16, 0},
    {0x7870c69c4afa5010ULL, 0x010bcbef2e76ec38ULL, 137, 16, 0},
    {0x789142f921c5ddc2ULL, 0x01eac0d48a63f8a0ULL, 329, 16, 0},
    {0x789fd14299a3b25eULL, 0x0cec54be32a0961aULL, 615, 16, 0},
    {0x78c596b11a7ed6c8ULL, 0x96f7959a31d4da88ULL, 338, 16, 0},
    {0x78cae4015f2f1a9cULL, 0x7accd801bdf5a7baULL, 273, 16, 0},
    {0x78f2a4199621ba77ULL, 0xff7d1e55f89bf220ULL, 215, 16, 0},
    {0x78f6631f97c84a93ULL, 0x83bf948218722ec8ULL, 933, 16, 0},
    {0x791b89d093e3ab12ULL, 0xc1baff10cfd512baULL, 720, 14, 0},
    {0x79334f4034b4ef87ULL, 0x67976692142b5a92ULL, 587, 16, 0},
    {0x793fa21b98ad5a85ULL, 0xff3207c27c685e08ULL, 132, 16, 0},
    {0x7974714826cb7f8fULL, 0x7398fd47887a6f62ULL, 244, 16, 0},
    {0x797bb7ad7851a21cULL, 0xc7b18cd2b3ee933aULL, 828, 14, 0},
    {0x79b16a82533f71dcULL, 0xad746c2a36994972ULL, 230, 16, 0},
    {0x79bae3101a2dba2aULL, 0xfa5f13472a4c9242ULL, 459, 16, 0},
    {0x79ccf69d3dc34330ULL, 0x4bfa879d6f38201aULL, 508, 16, 0},
    {0x79e287f4de8c905eULL, 0x2e3490bcbe46b0e0ULL, 1030, 16, 0},
    {0x79f5985b494cc052ULL, 0x30499baf617df462ULL, 544, 16, 0},
    {0x7a0146c18edb6995ULL, 0x959306180a47db94ULL, 730, 14, 0},
    {0x7a2346c4314c4fe3ULL, 0x4900a5a1bf978d42ULL, 409, 16, 0},
    {0x7a3ec7aa8378860eULL, 0x64af218c0b51311aULL, 771, 14, 0},
    {0x7a47a5a581e0b661ULL, 0x387355773682aec2ULL, 190, 16, 0},
    {0x7a62f6a9943385b8ULL, 0x5cd49c37b73d6192ULL, 802, 14, 0},
    {0x7aafdefd737d2929ULL, 0xb9e5ff12c2650ca2ULL, 1006, 16, 0},
    {0x7abb4b76ec581967ULL, 0x3abe7134e97fec40ULL, 719, 14, 0},
    {0x7aed23ca23c45b81ULL, 0x7396fca17a818442ULL, 781, 14, 0},
    {0x7b0e375f6cc8bb7fULL, 0x4ded3fca2d8c6872ULL, 1001, 16, 0},
    {0x7b181d1c7958f2e6ULL, 0xc1c001af71f9f07aULL, 985, 16, 0},
    {0x7b610e2f0b9633f6ULL, 0xbcce783de2744b5aULL, 989, 16, 0},
    {0x7b688daa346d74c7ULL, 0x09323680c5a4ca08ULL, 713, 14, 0},
    {0x7b91c1ac7049194eULL, 0x346952826f7a5d38ULL, 745, 14, 0},
    {0x7bb8c1cb956875a0ULL, 0x9b5fc643eebd8582ULL, 754, 14, 0},
    {0x7bfaeada91692ae9ULL, 0xdd4c41f2b19477c0ULL, 761, 14, 0},
    {0x7c03b917610a68b0ULL, 0xe1917e8a2007a39aULL, 737, 14, 0},
    {0x7c049686f84c3ad1ULL, 0x352dec5df897e3daULL, 987, 16, 0},
    {0x7c05cd63d5991c3fULL, 0xe771aed8e4e1c53aULL, 791, 14, 0},
    {0x7c0f1f1db85cefdcULL, 0xee5d6a3394e32088ULL, 925, 16, 0},
    {0x7c3e4229d6fb08d6ULL, 0xd80cfc9ca2c38e5aULL, 736, 14, 0},
    {0x7c4a9ebfd7cddb9cULL, 0xd8d03a605fa03808ULL, 894, 16, 0},
    {0x7c4e7059412ab67fULL, 0xff70d36515c4edf2ULL, 1003, 16, 0},
    {0x7c56931e24b356c7ULL, 0x25fa77b8c3693958ULL, 143, 16, 0},
    {0x7c6de4ef7a2bcd8cULL, 0x523f68d0d179f3ccULL, 878, 13, 0},
    {0x7c7626fc9416d7a9ULL, 0x38e02e0bc48df550ULL, 547, 16, 0},
    {0x7cc4f1fd8f723512ULL, 0x91add1dcd240a1a2ULL, 835, 14, 0},
    {0x7cf98c958cf1b58cULL, 0x48239f959ad1f7d8ULL, 360, 16, 0},
    {0x7cfdddf7ea012882ULL, 0xb50a19beb430a09aULL, 737, 14, 0},
    {0x7d25331f626b8eabULL, 0x09a323c1daf5375aULL, 460, 16, 0},
    {0x7d4d35c15525faa2ULL, 0x0273b6fe0a384c68ULL, 198, 16, 0},
    {0x7d5395bcb68f1758ULL, 0xddef3da312c279baULL, 159, 16, 0},
    {0x7d58303128fda5abULL, 0x19832670ff00eb40ULL, 586, 16, 0},
    {0x7d776e7bc9a05673ULL, 0xf80d1f04e489cda8ULL, 897, 16, 0},
    {0x7d7f0e00c16514afULL, 0xc074f4881f9e440aULL, 666, 10, 0},
    {0x7d82ff7bc99e3215ULL, 0x73b1309ec9d2b66cULL, 797, 14, 0},
    {0x7d8d8a51e7d6ec5fULL, 0x1795870fb8dabdcaULL, 718, 14, 0},
    {0x7d973d413119fa61ULL, 0xbded9bdc3fdb06c2ULL, 992, 16, 0},
    {0x7da3dd76f97bdb84ULL, 0x6a737b2e60ea21b4ULL, 117, 13, 0},
    {0x7da79f792e2532e8ULL, 0x6900e9b9248a4ee4ULL, 36, 13, 0},
    {0x7db146fe1b656424ULL, 0xb1083feb0852df70ULL, 727, 14, 0},
    {0x7dba337d5ce29482ULL, 0x7787dd81b5aa006aULL, 788, 14, 0},
    {0x7e0ce2d6cdbd78adULL, 0xf5b2bf521a520a00ULL, 765, 14, 0},
    {0x7e2290a595617b2eULL, 0xb85a2f707eee6d42ULL, 226, 16, 0},
    {0x7e31a1721aff0c04ULL, 0x53d3db9c39c1a4c8ULL, 311, 16, 0},
    {0x7e4d63176fb22dfbULL, 0x6ec4df532bdcf352ULL, 763, 14, 0},
    {0x7e56a5d813603b01ULL, 0x64b6c23d98463f22ULL, 74, 13, 0},
    {0x7e884f9bfd5ca09aULL, 0x382b7520c8e5f922ULL, 428, 16, 0},
    {0x7ea23b756c03aaedULL, 0x41eb039ba94561acULL, 873, 13, 0},
    {0x7ed889718c4dfb21ULL, 0x0f79979c10c2c256ULL, 685, 13, 0},
    {0x7edd92b2115f789dULL, 0x80dd6710a24c4e5aULL, 378, 16, 0},
    {0x7ef48fc4e45065e4ULL, 0xd861f32daec5f44aULL, 499, 16, 0},
    {0x7ef970f909b0dcc5ULL, 0xdce65ca7cc2e7912ULL, 815, 14, 0},
    {0x7efc6b0a821c61d5ULL, 0x651b032a96b47ce0ULL, 565, 16, 0},
    {0x7efd4509cdb55b85ULL, 0x60bd5fe1ebeb4924ULL, 85, 13, 0},
    {0x7f3fbc019bef1289ULL, 0xc2ef4ffb7da0c460ULL, 818, 14, 0},
    {0x7f55cc00ebec4958ULL, 0xda9018ef2bb77c78ULL, 745, 14, 0},
    {0x7f6e4029f05461ffULL, 0x6fc489b0ebc535eaULL, 587, 16, 0},
    {0x7fa9c7dba0c6cf24ULL, 0xa671c62eb0f35572ULL, 743, 14, 0},
    {0x7fe0a129c8b6774aULL, 0xc836892b1e1725faULL, 774, 14, 0},
    {0x7fe547ffe988dd8cULL, 0xca2888623df315b8ULL, 415, 16, 0},
    {0x7feeabb4c6bec2a9ULL, 0x2bce4001e64ce252ULL, 436, 16, 0},
    {0x7ff3768f4c4e7621ULL, 0x1161708e4070ab1aULL, 914, 16, 0},
    {0x7ffafc515753faf8ULL, 0x94f1c224a5bd8520ULL, 727, 14, 0},
    {0x802ee1ba88e04962ULL, 0x786d2252b5feb4a2ULL, 963, 16, 0},
    {0x8036c6de53f8f17fULL, 0xf99f56260ecc1020ULL, 588, 16, 0},
    {0x803b3ff28abdaa2fULL, 0xbc5beea9914cac28ULL, 386, 16, 0},
    {0x806426a9407e416bULL, 0x109d0dc78c6b4d5aULL, 153, 16, 0},
    {0x80c802df2c33d108ULL, 0x8b8f2b2a9dbec22aULL, 737, 14, 0},
    {0x80ce33b3adaad58cULL, 0x44aa57b6326ed4e2ULL, 734, 14, 0},
    {0x80d8e35d40da116aULL, 0x83ae8f9d80b7dac4ULL, 102, 13, 0},
    {0x80dc21d926f5259dULL, 0x563e8aa00248e340ULL, 129, 16, 0},
    {0x80e7c4b225676f7eULL, 0x6106f811d59b0cfaULL, 803, 14, 0},
    {0x80fa148cd0524345ULL, 0xfa49dcd596e93638ULL, 884, 16, 0},
    {0x8101d8c2a6fa22a4ULL, 0xc3a7abc835822988ULL, 1019, 16, 0},
    {0x810d4f2ee88ad500ULL, 0x4aa3126757f9ae2aULL, 21, 13, 0},
    {0x8120fcf4dfe23c03ULL, 0x28d4de63e10b74d2ULL, 596, 16, 0},
    {0x8131bb01a99d2efeULL, 0xbd4c27d1ab2131b2ULL, 355, 16, 0},
    {0x8132d1764d5aacb7ULL, 0xe41f18e047fdd0a8ULL, 218, 16, 0},
    {0x81360012119dbb25ULL, 0x98c7a53b9092e130ULL, 401, 16, 0},
    {0x814c396663bd217fULL, 0xa4afafc1cf643b64ULL, 797, 14, 0},
    {0x81a7c4b3c4c0b08fULL, 0x9eda6fb308a752a4ULL, 772, 14, 0},
    {0x81c1391d06defc25ULL, 0xa5e390e72f303c6aULL, 935, 16, 0},
    {0x81e01f47ebe77570ULL, 0x0f63bb19f027c5eaULL, 621, 16, 0},
    {0x820df971adb166e3ULL, 0xefe246d7177b0d8aULL, 660, 10, 0},
    {0x8210807e20dce683ULL, 0xf4dccde181b14c02ULL, 891, 16, 0},
    {0x8214f5025e51b57eULL, 0x2cc20e2f3d5bb63aULL, 278, 16, 0},
    {0x821eef6abdfc13b1ULL, 0x7977ef4d75f8c6b8ULL, 414, 16, 0},
    {0x824e0939e40f594fULL, 0x37171402cce6718aULL, 177, 16, 0},
    {0x82538d4f45ef40c0ULL, 0x3644fe63bcaf3086ULL, 676, 13, 0},
    {0x826243b7af9e13d5ULL, 0xd3dbbe6eca85387aULL, 265, 16, 0},
    {0x82c8890dc682e64cULL, 0x1ce320511f8b4f62ULL, 444, 16, 0},
    {0x82d6c0e74069462eULL, 0x637d09c48ce4e222ULL, 470, 16, 0},
    {0x82e1aeb600a8ee28ULL, 0x0b74897c536800d0ULL, 618, 16, 0},
    {0x82e6d92880cabd4fULL, 0x4f58d63bbd1694a2ULL, 83, 13, 0},
    {0x831349781c518f2aULL, 0x92146543882637f8ULL, 1042, 16, 0},
    {0x83189628d41642e3ULL, 0x2ef4b187ec17b820ULL, 295, 16, 0},
    {0x831e866b1112ca1fULL, 0x15788a36c95c6f68ULL, 501, 16, 0},
    {0x8370942b194f6ec6ULL, 0x3ff1be7dcb36ead2ULL, 473, 16, 0},
    {0x8383b27b5d27bad2ULL, 0xf198a208b78adbe0ULL, 589, 16, 0},
    {0x838e8fd370bb8a4cULL, 0x3862ea1a1a255d42ULL, 205, 16, 0},
    {0x83bc1730971eeea4ULL, 0xbb47263c2ca495d0ULL, 887, 16, 0},
    {0x83c5c1f5225b9b21ULL, 0x4be3dc972509db6aULL, 77, 13, 0},
    {0x83e7ade5a9d687dfULL, 0x3cda489f50866120ULL, 602, 16, 0},
    {0x842aa7a4b261e036ULL, 0x418a26ed9f42bedaULL, 275, 16, 0},
    {0x842e254fae4e895cULL, 0x78cc6b6309f9db80ULL, 634, 16, 0},
    {0x8431660518434653ULL, 0x534b865929e67c78ULL, 240, 16, 0},
    {0x8444f783203d8f01ULL, 0x92414a99a8aa0ae8ULL, 243, 16, 0},
    {0x845088195f06410fULL, 0x8c5656c774c3dbf2ULL, 1011, 16, 0},
    {0x8456e1bda359d0e2ULL, 0xa4cf83c7285ce542ULL, 286, 16, 0},
    {0x8478f5f8d2d34bc0ULL, 0xe26fecbb093a0446ULL, 702, 13, 0},
    {0x8491572377c9d5e5ULL, 0xda884564f897924aULL, 636, 16, 0},
    {0x84f59242d1dd7fafULL, 0x9312aab40190cbc4ULL, 63, 13, 0},
    {0x8523bba853d43da0ULL, 0x0eedf997f79e3e42ULL, 958, 16, 0},
    {0x853a4c27bc901cd4ULL, 0x2238289d409d21f8ULL, 1033, 16, 0},
    {0x853c8b07085894b5ULL, 0xeaf52d6f00bd1290ULL, 728, 14, 0},
    {0x855b930314a55f7fULL, 0xc5da070207c1b562ULL, 889, 16, 0},
    {0x856ff5e51ac16440ULL, 0x98a4e099d3e3db38ULL, 212, 16, 0},
    {0x8575b62d623e5097ULL, 0xe998e14ae6efc2daULL, 732, 14, 0},
    {0x858c1563a3b7f65cULL, 0xb631c78b1acfabdaULL, 372, 16, 0},
    {0x8593594dbcea7a10ULL, 0x9c62811e27844bc2ULL, 929, 16, 0},
    {0x85997953dc91ca7bULL, 0x5db5373d3115248aULL, 771, 14, 0},
    {0x85b0fb7a88b1fc39ULL, 0x602ce754f9ed1092ULL, 805, 14, 0},
    {0x85ba6636afe59ad7ULL, 0x0c96ac6625f47e12ULL, 795, 14, 0},
    {0x85d0b6f995fe8c42ULL, 0xb83b4ab9b6aa806cULL, 55, 13, 0},
    {0x85df3ea5e856c753ULL, 0x4a91b4177d9626f8ULL, 785, 14, 0},
    {0x85e21bae4780440dULL, 0xafc9a3cece04c356ULL, 852, 13, 0},
    {0x860b96bfe36cdb6cULL, 0xd03b777167f9c9c0ULL, 653, 16, 0},
    {0x8611ff0350302d1cULL, 0x9d02c586d1a60e62ULL, 909, 16, 0},
    {0x862efab6b60dabc7ULL, 0x5b5c7660192096d0ULL, 1046, 16, 0},
    {0x863f3ea4cfbe01eaULL, 0xe05c961ca980afbaULL, 573, 16, 0},
    {0x866645efd80be1eeULL, 0xd784e3d21f599518ULL, 306, 16, 0},
    {0x86992a49e6d87b89ULL, 0xb0aead545f90d9a8ULL, 783, 14, 0},
    {0x86b70f91a198c837ULL, 0x8d089292230cd008ULL, 321, 16, 0},
    {0x86d5075224d23195ULL, 0xc7834dedc2016b78ULL, 800, 14, 0},
    {0x86f83b23b74557a1ULL, 0x18b667e0b178b192ULL, 274, 16, 0},
    {0x870166c4879af934ULL, 0x4138049ff37ba76aULL, 487, 16, 0},
    {0x871c27210fd4276eULL, 0x057aefa7a51d7fd2ULL, 531, 16, 0},
    {0x871e24ef9c8382d3ULL, 0x80f88e2166eed832ULL, 913, 16, 0},
    {0x873556a07281cef6ULL, 0xd5132f87ace4bb70ULL, 253, 16, 0},
    {0x8744707afec32e21ULL, 0xf76cb86dd65a9d9cULL, 92, 13, 0},
    {0x874adcb4be9fcf6eULL, 0x881fb6ab0b4c0ba2ULL, 513, 16, 0},
    {0x874e513ef01ddc2bULL, 0xcb1a89774d1251a2ULL, 196, 16, 0},
    {0x8795b097698dc546ULL, 0xe4e9aa7abe13b0c8ULL, 713, 14, 0},
    {0x87c49abd6696c9ecULL, 0x68a3d3b98da08edaULL, 915, 16, 0},
    {0x87cac9f01edaabc4ULL, 0x72d2b6121a45d172ULL, 763, 14, 0},
    {0x87d4aa9ffa83528fULL, 0xaa33c419eb54b512ULL, 69, 13, 0},
    {0x87e5c384d6f90098ULL, 0xac30a15a421a58bcULL, 32, 13, 0},
    {0x87e6718a7698aeccULL, 0x311508e9fa882912ULL, 962, 16, 0},
    {0x881b0ce5a68fbb1cULL, 0xd92b5fef734e854cULL, 877, 13, 0},
    {0x883a33b6cff4a2a6ULL, 0xe755f660cd3e82c2ULL, 834, 14, 0},
    {0x88593106e9e8a31bULL, 0xe647fcb9cda8e742ULL, 512, 16, 0},
    {0x885f93fe5dd8eecbULL, 0x0c9c33a4bc15c088ULL, 818, 14, 0},
    {0x887c61289bb6d097ULL, 0x4d54215c5b99a9eaULL, 663, 10, 0},
    {0x887e76586c39ec7dULL, 0x545de7f1c54c4868ULL, 481, 16, 0},
    {0x888bc74003b53613ULL, 0x24f70360bfe1db54ULL, 827, 14, 0},
    {0x8897b20bdbd5494dULL, 0x4cbaf52ea2ce88b0ULL, 726, 14, 0},
    {0x88b663a41f82930aULL, 0x4a55f8b337bd05d6ULL, 20, 12, 0},
    {0x88c9d4c1b95be215ULL, 0x27cedbae7ed3b2d2ULL, 381, 16, 0},
    {0x88eb5ec601168db5ULL, 0xcc7ff6abb6f9c582ULL, 70, 13, 0},
    {0x88f6c4f1271801e9ULL, 0xe7a4c2ebbc0f6870ULL, 598, 16, 0},
    {0x891e3669ec9aee51ULL, 0xc891d4d758d9b23aULL, 729, 14, 0},
    {0x894925b62c1c5c1eULL, 0xe65931ab8a789670ULL, 375, 16, 0},
    {0x89719c0b8eb6d815ULL, 0x5ed25fe2d4ad1822ULL, 548, 16, 0},
    {0x8a0516d40f307923ULL, 0x3fab20ea42a95f92ULL, 436, 16, 0},
    {0x8a0b2e51c9397867ULL, 0x56bd02cc379a234aULL, 87, 13, 0},
    {0x8a1927c7bfb9405dULL, 0x5839a61adebf42ccULL, 76, 13, 0},
    {0x8a2d1c995de9f988ULL, 0xb470303ddb3e8b3cULL, 27, 13, 0},
    {0x8a36506c45349ae5ULL, 0x07dfd746140d6d10ULL, 325, 16, 0},
    {0x8a36d48c56066bafULL, 0xe157d730246e4c1aULL, 799, 14, 0},
    {0x8a53e7e9a1dfe510ULL, 0xc0e7d9474e5e7f80ULL, 406, 16, 0},
    {0x8a97b6429be0b10fULL, 0xd3d4789c8c8fa6caULL, 767, 14, 0},
    {0x8a9f65d8d2a0c26dULL, 0x4640b3b222486b02ULL, 862, 13, 0},
    {0x8aaae3f04c7d52e1ULL, 0x57ebe60de3cd4c72ULL, 890, 16, 0},
    {0x8b04867f9a22b7dfULL, 0x3d5293aea27daf8aULL, 54, 13, 0},
    {0x8b0e5ec6cba60c32ULL, 0xccf1b0bdd49e3f44ULL, 879, 13, 0},
    {0x8b102a541e970352ULL, 0xdc6805b280bdb930ULL, 253, 16, 0},
    {0x8b247875d3424ae5ULL, 0xfbe6078af6066c8eULL, 867, 13, 0},
    {0x8b2b344bb602f918ULL, 0xb82fa6c210058892ULL, 1007, 16, 0},
    {0x8b41f0e25ce81040ULL, 0xa080e067cd04403aULL, 447, 16, 0},
    {0x8b4ecf62d6e5eecaULL, 0xeedfc7c79464761aULL, 753, 14, 0},
    {0x8b59c8b314e3c770ULL, 0x79cd02f6f42f2b08ULL, 280, 16, 0},
    {0x8b61d2bd8e111c34ULL, 0x051612c6f451bb80ULL, 398, 16, 0},
    {0x8b7a84b26dde6fc3ULL, 0xfa6ded4c1f80af7aULL, 223, 16, 0},
    {0x8b8ea0086b221839ULL, 0x09a553c0a2094a88ULL, 765, 14, 0},
    {0x8b924780b808ce3fULL, 0xeb4f1224b8f68f12ULL, 816, 14, 0},
    {0x8c04c28bbd5dc813ULL, 0x6ab079719d992992ULL, 302, 16, 0},
    {0x8c14f4f27fe8e238ULL, 0x2155e4e5d9ecef3aULL, 382, 16, 0},
    {0x8c20fc74622772c6ULL, 0xfe24cc7eb7544192ULL, 789, 14, 0},
    {0x8c55231d3cf4780bULL, 0xb9d3877ffb50cbf8ULL, 221, 16, 0},
    {0x8c9074f093ba3492ULL, 0x28fd034f4f77d740ULL, 296, 16, 0},
    {0x8ce93674f10731f3ULL, 0x42702bed190104f6ULL, 849, 13, 0},
    {0x8d66e5c1a931c819ULL, 0x344238faae63d89cULL, 823, 14, 0},
    {0x8dceaeba375e7172ULL, 0x795ec854bafa1972ULL, 801, 14, 0},
    {0x8dece01bab324c86ULL, 0xb9f41e6e276e8560ULL, 1030, 16, 0},
    {0x8e2a8c069f31ec94ULL, 0xf48be8d79a296fd2ULL, 1013, 16, 0},
    {0x8e46d925c7e345faULL, 0xdf7c2f2f1e5ebc10ULL, 576, 16, 0},
    {0x8eaa6dd41f1d28d6ULL, 0xb7c5c323e0571d38ULL, 402, 16, 0},
    {0x8eabd79f9d4b132cULL, 0xe2e6b7d009c47024ULL, 108, 13, 0},
    {0x8ebbed888b7177feULL, 0x86018f12a42cd120ULL, 323, 16, 0},
    {0x8ecd7407b113a6cdULL, 0x076d0f4af4e28aa8ULL, 1034, 16, 0},
    {0x8fae1bfaa8ba2c67ULL, 0x6755dc6a000ebea2ULL, 766, 14, 0},
    {0x8fd507c015350bebULL, 0xff8c14ddcaf44962ULL, 290, 16, 0},
    {0x8fe9374da3fc221eULL, 0x5cb2bf04019f393aULL, 790, 14, 0},
    {0x8fed7310908bf085ULL, 0x943aa81d76d25b10ULL, 528, 16, 0},
    {0x900a37f99a96e611ULL, 0x4392efac6b4034a2ULL, 438, 16, 0},
    {0x9024933f9178596eULL, 0x7f20f0e9a0d37c32ULL, 510, 16, 0},
    {0x9059d40ab431417cULL, 0x84627396f1b14740ULL, 1015, 16, 0},
    {0x909a073593e8120fULL, 0xee10873c64bb57f0ULL, 130, 16, 0},
    {0x90a759fad8596b9bULL, 0x94643fa044a8addaULL, 665, 10, 0},
    {0x90c2d91b08e92f20ULL, 0x70ea864b42d7af94ULL, 41, 13, 0},
    {0x90deeb5976df8cbdULL, 0x45929fbc6fd9f3b2ULL, 613, 16, 0},
    {0x90f7f7601908b5e1ULL, 0xb96c9bcd429e6992ULL, 527, 16, 0},
    {0x91024458c935de68ULL, 0x026957f75639611aULL, 938, 16, 0},
    {0x9114fa055676a0b9ULL, 0xefaf3ac239261a7aULL, 935, 16, 0},
    {0x913bc8e5de866617ULL, 0x6728365376fa698cULL, 797, 14, 0},
    {0x91467c599dc5721bULL, 0xd6b8e93cd016efa2ULL, 748, 14, 0},
    {0x9148b82b963904f0ULL, 0xa53f05e2839d063aULL, 219, 16, 0},
    {0x91533a4dd9b62913ULL, 0x31b2006f6e2ffb8aULL, 756, 14, 0},
    {0x917cce36a60ef157ULL, 0x49381eff691fc1c0ULL, 586, 16, 0},
    {0x919e74c6416d5f1bULL, 0x5eba88944110d308ULL, 121, 16, 0},
    {0x91a29ed74d2615deULL, 0x5b9e3d2044350da2ULL, 834, 14, 0},
    {0x91a4100417e53268ULL, 0x7113a142daf175faULL, 950, 16, 0},
    {0x91df5ab77b3b658eULL, 0x5463ba58b8180fb2ULL, 747, 14, 0},
    {0x921ad287e848a034ULL, 0xb77bb55d98f302d8ULL, 719, 14, 0},
    {0x9222fef23d12a462ULL, 0xc037462bb3101b84ULL, 59, 13, 0},
    {0x924a5fad688a84c2ULL, 0x86bd4f5ef533bf78ULL, 157, 16, 0},
    {0x9260bc25debe7bddULL, 0xc2610e0712b8c0ceULL, 701, 13, 0},
    {0x9263e22da0e8b46bULL, 0x57a8b88fb29bf38cULL, 741, 14, 0},
    {0x928538d6247c5531ULL, 0x932b8c328ab5a2deULL, 864, 13, 0},
    {0x928d1206c0bb8ca9ULL, 0x61b25822b49f1308ULL, 725, 14, 0},
    {0x92a4e04fb0b894eeULL, 0x8f652f75fe1bb6b2ULL, 407, 16, 0},
    {0x9307c41ad82e1177ULL, 0xe618f0f60faf2c22ULL, 459, 16, 0},
    {0x930fb489ab4dfffbULL, 0x6d5859a8c81e3300ULL, 655, 16, 0},
    {0x93131d7320b6adaaULL, 0xc57a0f0b738843d0ULL, 562, 16, 0},
    {0x9317d037a99be4aeULL, 0x794837fb404ce70cULL, 730, 14, 0},
    {0x934105346c00475fULL, 0x4a2b755a1cd04b1aULL, 817, 14, 0},
    {0x934e9299d2a115f4ULL, 0xd010a81785861b42ULL, 552, 16, 0},
    {0x935b4614fb84ed28ULL, 0x3801c657b4c205fcULL, 26, 13, 0},
    {0x93a78a22140e1ef0ULL, 0x40ee74b1b8838602ULL, 290, 16, 0},
    {0x93a88969675de81bULL, 0x41af1ddf950a775aULL, 959, 16, 0},
    {0x93aaef1895d0420fULL, 0x03de85eaba64dcbaULL, 980, 16, 0},
    {0x93d8a178457f82e2ULL, 0xc935f2cefdc315caULL, 964, 16, 0},
    {0x93dbed45e1937190ULL, 0x16ad55795a781544ULL, 71, 13, 0},
    {0x93f1751c3bcea3dfULL, 0x37673984b2b3291aULL, 748, 14, 0},
    {0x94106712d1983b34ULL, 0xde56c8c93b8b2edcULL, 832, 14, 0},
    {0x9439f3f35d329b41ULL, 0xa6f102acada3c58cULL, 772, 14, 0},
    {0x944ae4fa422f907bULL, 0x5b0651e47b012618ULL, 738, 14, 0},
    {0x948a88a8d9817c92ULL, 0x255576c62009d656ULL, 684, 13, 0},
    {0x949abc98b1c1ce00ULL, 0x6536359de937291aULL, 979, 16, 0},
    {0x94cea64ab6855ebdULL, 0xf287069742acb99aULL, 312, 16, 0},
    {0x94cf2df7f121a386ULL, 0xa3229c33e869603aULL, 359, 16, 0},
    {0x94da98d1692e6f99ULL, 0x1fd834644c9d4072ULL, 835, 14, 0},
    {0x953c7d042b0329e5ULL, 0xf4653427651e295eULL, 705, 13, 0},
    {0x9556225b01c2786dULL, 0x9345097632254f64ULL, 86, 13, 0},
    {0x95ae494420592f86ULL, 0xb4d0ec5fffe338faULL, 754, 14, 0},
    {0x95fa2f6f5ac7519dULL, 0x8f9ef9f24ee37f7aULL, 817, 14, 0},
    {0x96005477d0f6a186ULL, 0x153edd44a7bec432ULL, 344, 16, 0},
    {0x961daec1c0ca13f2ULL, 0x95b56dd86a8a3b22ULL, 123, 16, 0},
    {0x9625d16781db2ddbULL, 0xcb32fe3997b8c97aULL, 817, 14, 0},
    {0x9629fa988ec0a023ULL, 0xdcf7f9a0bd43147aULL, 715, 14, 0},
    {0x96467ebf292c93d2ULL, 0xbb5c5c690ba1ed4aULL, 133, 16, 0},
    {0x965eeaaaf975792fULL, 0x29ba1e9eabf81654ULL, 804, 14, 0},
    {0x9669ece93bf664e0ULL, 0x94668f35a61c8d12ULL, 555, 16, 0},
    {0x966d4bc523c97638ULL, 0x55d2ca607224d598ULL, 161, 16, 0},
    {0x9670ab6c37ae6e61ULL, 0xb6d9db9419ef8a1aULL, 798, 14, 0},
    {0x96732b78034ce183ULL, 0xc071d3d6adca1808ULL, 423, 16, 0},
    {0x96839d679b66e9ddULL, 0x6d924240be0a8c9aULL, 25, 13, 0},
    {0x969bf3379d41bb72ULL, 0x016bbdddedb0d1daULL, 232, 16, 0},
    {0x96af97382b315771ULL, 0x68bbf65292e3ab84ULL, 830, 14, 0},
    {0x96ce2380ca058920ULL, 0x7e9cbee3335a7aaaULL, 798, 14, 0},
    {0x96ddff3da3ba6822ULL, 0xa13034f67d142a80ULL, 549, 16, 0},
    {0x9726ddc9caebf596ULL, 0x346f1173673e20c6ULL, 683, 13, 0},
    {0x9736665e4bfd8e7bULL, 0x1ed34dc7a1c0c7e4ULL, 875, 13, 0},
    {0x97473a709b088097ULL, 0xe1917d9586b46b00ULL, 652, 16, 0},
    {0x97cce2b6b604f733ULL, 0x2a44dc01faceaab2ULL, 267, 16, 0},
    {0x97fde8cb605d697fULL, 0x3daa809faadce5e8ULL, 433, 16, 0},
    {0x97fed04422e5f30cULL, 0x322bf2f064e2845aULL, 358, 16, 0},
    {0x980495b0dd23315fULL, 0x6163cc568aed77a2ULL, 42, 13, 0},
    {0x9816c40ebb631031ULL, 0x0129c9e68ccdeeaaULL, 639, 16, 0},
    {0x985dd03374f0e87cULL, 0x6a7addab5bcbfdd0ULL, 622, 16, 0},
    {0x987c31c523ccddfdULL, 0x6013bab4035cb0faULL, 732, 14, 0},
    {0x98a4ce1d1cab8b14ULL, 0x9f2d43243682a5a8ULL, 392, 16, 0},
    {0x98bcecc231789521ULL, 0x980e592927eaf0f8ULL, 712, 14, 0},
    {0x98bf4c94d3bd006fULL, 0xfeae0cfdbbda02d2ULL, 860, 13, 0},
    {0x98cfccaffbe49495ULL, 0xe9b5886f1ea7fd32ULL, 608, 16, 0},
    {0x98e0bf267dba957aULL, 0xc6364649ecd33660ULL, 341, 16, 0},
    {0x98e6d371b417b92aULL, 0x8dcbce79604752daULL, 610, 16, 0},
    {0x98e77dd8f10255b2ULL, 0x4d2e5f962524c744ULL, 102, 13, 0},
    {0x98ea98582d2578f8ULL, 0x16c60c7f3773578aULL, 792, 14, 0},
    {0x98edcc2398598be3ULL, 0xba6d691ccdc53856ULL, 687, 13, 0},
    {0x98ef639504524139ULL, 0x01d07131b99d5d5aULL, 863, 13, 0},
    {0x98f6b9d33bfa6ac1ULL, 0xb9f86b6b15e1b97cULL, 807, 14, 0},
    {0x99526cd56b0707f6ULL, 0xb9814a99b7aa771aULL, 447, 16, 0},
    {0x9957688765ea5f22ULL, 0x5fc7e1037ecdb73aULL, 625, 16, 0},
    {0x995b99173fbd222dULL, 0x4cc5f67b3b814c9aULL, 631, 16, 0},
    {0x997f2aafd7eac3ebULL, 0xdf7ade08b8e6d4d0ULL, 131, 16, 0},
    {0x99998edb14bfb982ULL, 0x818c7e482c62c200ULL, 191, 16, 0},
    {0x99a4f755981323b5ULL, 0xbd5be2457eae2a6aULL, 861, 13, 0},
    {0x99b50a3b9e8805bbULL, 0xc373f87ba1e47b48ULL, 785, 14, 0},
    {0x99d6eaf5f70fbd64ULL, 0xb902983573af1720ULL, 637, 16, 0},
    {0x9a2360a57eac2685ULL, 0x91f26cdb94e583a8ULL, 318, 16, 0},
    {0x9a33f1394badb4bdULL, 0x8eaff66a95672e1aULL, 771, 14, 0},
    {0x9a6668b11f8aefafULL, 0xea009b65d24fc802ULL, 489, 16, 0},
    {0x9a74fe6d049042a3ULL, 0x06c3a376c2df6f62ULL, 101, 13, 0},
    {0x9aa6657894209cbaULL, 0x28a6de1f45299d92ULL, 222, 16, 0},
    {0x9ac9a4ad417704e6ULL, 0x66530fa6c0bf7554ULL, 812, 14, 0},
    {0x9ad48b5c89da1f76ULL, 0x4e7ac5117a16380aULL, 831, 14, 0},
    {0x9aedc09b8c9c4fdaULL, 0x10535d15aca68e72ULL, 856, 13, 0},
    {0x9af841966db40cb2ULL, 0xcddcad1feff78688ULL, 723, 14, 0},
    {0x9afe695db29d18a8ULL, 0x84c71140d02cb50aULL, 766, 14, 0},
    {0x9b30e3bfcd4e4e45ULL, 0xc2cb7b082e9248deULL, 670, 13, 0},
    {0x9b40cb085fb21b65ULL, 0x202c085f98906dd2ULL, 995, 16, 0},
    {0x9b6c2099beba3a6dULL, 0xa7f6f733b5c99e28ULL, 974, 16, 0},
    {0x9b9346da8e53a8a8ULL, 0x18ad779548cd6adaULL, 463, 16, 0},
    {0x9ba4c94cb4b0bd2fULL, 0x5293e7e50ffbf35aULL, 609, 16, 0},
    {0x9bc011597a35bcc8ULL, 0x3dc4c077f20915d2ULL, 245, 16, 0},
    {0x9bd196bd90ef9b91ULL, 0x050ad3035e14053cULL, 99, 13, 0},
    {0x9bd723262649385fULL, 0x59b750d10a3d4498ULL, 725, 14, 0},
    {0x9bfda87ac4a1b50eULL, 0x33bb9935ad574a68ULL, 350, 16, 0},
    {0x9c0d83eec1870d3cULL, 0xa98032f918feb928ULL, 972, 16, 0},
    {0x9c222d01b7ea362eULL, 0x83559e856794e9a8ULL, 1049, 16, 0},
    {0x9c237668d5ce0b7fULL, 0x7f3c35f25b4e2120ULL, 600, 16, 0},
    {0x9c2e2aac35597720ULL, 0xe3d5208f30ed9a2aULL, 755, 14, 0},
    {0x9c3f0491bb675904ULL, 0xbb3ebddac69a3cb0ULL, 632, 16, 0},
    {0x9c498df7f443e34cULL, 0x825bfd75b265cfd8ULL, 235, 16, 0},
    {0x9c6e01a1a7381849ULL, 0x0665beec9cc60180ULL, 568, 16, 0},
    {0x9c6e406d0c2d0919ULL, 0x2c61af6417f552e8ULL, 309, 16, 0},
    {0x9c6ed36d6de4447aULL, 0x6a303c1195d37f38ULL, 461, 16, 0},
    {0x9c7095db41e2321cULL, 0xeabca9a3c9d07ecaULL, 742, 14, 0},
    {0x9c8258144a1a2bd3ULL, 0xef2310edcf5c70caULL, 858, 13, 0},
    {0x9c8a7fb50e8b6c0cULL, 0xdc266a2556ee5972ULL, 734, 14, 0},
    {0x9c99e210b1d14fabULL, 0x6ba77dfaef5508d2ULL, 788, 14, 0},
    {0x9c9c774b1444f747ULL, 0xc3587a9cc187503aULL, 194, 16, 0},
    {0x9cae688025b5f121ULL, 0x4f482b3c96db8288ULL, 721, 14, 0},
    {0x9cc08e18486da7b1ULL, 0x36c7ab7ee6a39260ULL, 534, 16, 0},
    {0x9ccc99033313ace0ULL, 0x710850fd6d72e5f8ULL, 332, 16, 0},
    {0x9cd3b6c0eebbab4eULL, 0x74dfb021029b472aULL, 803, 14, 0},
    {0x9ceac8a973574fedULL, 0x9ee2fd383d62b710ULL, 651, 16, 0},
    {0x9cfc6bce64592353ULL, 0x73fe547c1ce776baULL, 957, 16, 0},
    {0x9d1fdc533c2d8bb5ULL, 0x35fb27249a08badaULL, 756, 14, 0},
    {0x9d2d13fc9212ac39ULL, 0x8a0bbdcc67a01c52ULL, 986, 16, 0},
    {0x9d395990307517b5ULL, 0x256f8a1465b94e90ULL, 1038, 16, 0},
    {0x9d40031c315c65f4ULL, 0x85a8bf6e0a7446d0ULL, 1021, 16, 0},
    {0x9d52b88d9c5edb0cULL, 0xa74d945414999740ULL, 1050, 16, 0},
    {0x9d65d9eda61794c3ULL, 0x24175f5d1be8d56aULL, 119, 16, 0},
    {0x9d6be5e8d980285cULL, 0xbac1da256e043e32ULL, 343, 16, 0},
    {0x9d75b7d46c1a481fULL, 0xb8cd2d5525c6ee04ULL, 40, 13, 0},
    {0x9d822e237394ffcaULL, 0x943d9a1d957ceec8ULL, 310, 16, 0},
    {0x9d83c7bfa4a5ff6bULL, 0x933bd5693391075aULL, 251, 16, 0},
    {0x9d93c6c1f7c5628aULL, 0x5ccc00839aabf16aULL, 77, 13, 0},
    {0x9d9fb3e3d5dfa62dULL, 0xc5561bb7915d08d0ULL, 174, 16, 0},
    {0x9db2a8e560b5bfa6ULL, 0xaf091c3316366d40ULL, 911, 16, 0},
    {0x9dd2769731c2d75dULL, 0x88f8861969406d88ULL, 388, 16, 0},
    {0x9de86abd1e706ac1ULL, 0x4e3c74c0b5d59dbaULL, 734, 14, 0},
    {0x9def0eb57d7a9296ULL, 0x94599d799c7acfbcULL, 32, 13, 0},
    {0x9e063cae28a3e4c5ULL, 0x4363d338b42306ecULL, 881, 13, 0},
    {0x9e3b7d604baa1535ULL, 0x1d5e2377073e0610ULL, 220, 16, 0},
    {0x9e7b6eaf0862b69aULL, 0xdb738c2fb2b71c52ULL, 729, 14, 0},
    {0x9ea1d396329407b8ULL, 0x3d211a3d74a788a0ULL, 210, 16, 0},
    {0x9eb13c8262d09dd0ULL, 0xb9f310d299f20334ULL, 797, 14, 0},
    {0x9ed01b74e3763f94ULL, 0xf039c1b41d94df3cULL, 22, 13, 0},
    {0x9f051142ca6b8c2fULL, 0x36160ff64b1e63deULL, 869, 13, 0},
    {0x9f0d170570c63514ULL, 0x57c801d6e2906c20ULL, 765, 14, 0},
    {0x9f3f65fa08f7d5daULL, 0xfb80dab70f569be2ULL, 663, 10, 0},
    {0x9f4382ee1782ae05ULL, 0x3c2b765c8d40642eULL, 16, 12, 0},
    {0x9f7229fa61447417ULL, 0x8c356e5b58190b32ULL, 734, 14, 0},
    {0x9f9600c29cb7e6fbULL, 0x116ed04a03451a52ULL, 541, 16, 0},
    {0x9fa2404590dad5a8ULL, 0x16a11ac37e464fe0ULL, 413, 16, 0},
    {0x9fa4d80fcc8f65bbULL, 0xdd34a691b04877f0ULL, 208, 16, 0},
    {0x9fb612bb00ca7d7dULL, 0xe7d63a677d56606aULL, 104, 13, 0},
    {0x9fd317f372376068ULL, 0x89924f078fbab8f6ULL, 710, 12, 0},
    {0x9fdd745496d53f3fULL, 0x39fd0a2c08bd9e1aULL, 991, 16, 0},
    {0xa01400949d4fe933ULL, 0x206c0a26ce888b1aULL, 801, 14, 0},
    {0xa015c6ea5257a507ULL, 0x20ccf780961278c2ULL, 642, 16, 0},
    {0xa061bf7e24460fc6ULL, 0xe215f855bc4265bcULL, 47, 13, 0},
    {0xa062a197cc673a5fULL, 0xec0676e45a4042baULL, 734, 14, 0},
    {0xa0a3fc83b30e61b3ULL, 0xa43199d2fb88afdaULL, 717, 14, 0},
    {0xa0e4296bface8ca4ULL, 0x53055db887a82bfaULL, 756, 14, 0},
    {0xa114e32f8d716f2fULL, 0xab990b1d70ed41c2ULL, 781, 14, 0},
    {0xa11f8c4deaae85f6ULL, 0xab62bd324a823178ULL, 893, 16, 0},
    {0xa12b0736b5a3a3cdULL, 0x189b5f77c43cc622ULL, 957, 16, 0},
    {0xa12b7217a3f18580ULL, 0xe72266c25d6ab97eULL, 869, 13, 0},
    {0xa1398a8a90660fdbULL, 0x72dcc1022e6ee7f2ULL, 795, 14, 0},
    {0xa13a688170e07141ULL, 0x99596c31669d55b8ULL, 783, 14, 0},
    {0xa140dff179344ab3ULL, 0x7b62db319e287992ULL, 801, 14, 0},
    {0xa1692e1afafc679aULL, 0x628851d0a4c6bc08ULL, 818, 14, 0},
    {0xa170cf09bead4b90ULL, 0xff0c53d1a2b8ff48ULL, 481, 16, 0},
    {0xa188a7ad49ec24fdULL, 0xe9b0e191da253beaULL, 328, 16, 0},
    {0xa1a08b00fb2946c0ULL, 0x9419f89fafb8c23cULL, 57, 13, 0},
    {0xa1b70f7cfa854582ULL, 0x3dd12f71519f96b0ULL, 653, 16, 0},
    {0xa1c4c94c43772e3bULL, 0x2859879c68cfd560ULL, 558, 16, 0},
    {0xa1e3bc6d97ef0e1aULL, 0xf0c10c1d3a165ed2ULL, 345, 16, 0},
    {0xa1f007fbc04355e7ULL, 0x305d0b0c86bab6b0ULL, 787, 14, 0},
    {0xa1f6f69af18e1a5bULL, 0xe6996b2afe9eacfaULL, 181, 16, 0},
    {0xa200b3e56b18c799ULL, 0xe21526649292aa0aULL, 716, 14, 0},
    {0xa20300cb7f0edeb4ULL, 0xa8e775cb18bce2b0ULL, 449, 16, 0},
    {0xa206ff2f396b5c5fULL, 0xd04506f3ac7fa45cULL, 66, 13, 0},
    {0xa208749f726bcad7ULL, 0x467199489fe9af82ULL, 250, 16, 0},
    {0xa20c9a2d1937fb69ULL, 0x9df109df2ff4c5c0ULL, 1031, 16, 0},
    {0xa226385202ae57ddULL, 0x38814ecb7b1df754ULL, 772, 14, 0},
    {0xa24a1bf89d5e65e9ULL, 0x4473828736f44b24ULL, 777, 14, 0},
    {0xa24ff14f14f49949ULL, 0xaadf1406f7d1a1c2ULL, 999, 16, 0},
    {0xa2bb6d74fce00bd0ULL, 0x8d1a6c52a9f9e9c2ULL, 978, 16, 0},
    {0xa2ca41dd9376a1f3ULL, 0xaf9575577dacf000ULL, 425, 16, 0},
    {0xa2cfc329b22fa0f4ULL, 0x2edebc17f63d2740ULL, 545, 16, 0},
    {0xa2f67073b0b6e367ULL, 0x503d44e27aaf1696ULL, 871, 13, 0},
    {0xa31118820328838bULL, 0xdf240f79903912c4ULL, 38, 13, 0},
    {0xa31d56b99866ad5bULL, 0x18728ee92a1fb2f0ULL, 725, 14, 0},
    {0xa329f272cdc70bfcULL, 0xb84a5ed40779cab6ULL, 684, 13, 0},
    {0xa341809f1d40f1c9ULL, 0xa34c072202896f90ULL, 764, 14, 0},
    {0xa34bb572ae3026efULL, 0x06a581d9dcc4a852ULL, 593, 16, 0},
    {0xa352c7448d199482ULL, 0x9e8ee06b004bb6a0ULL, 759, 14, 0},
    {0xa355cd135301116bULL, 0xf7c963095b5671bcULL, 772, 14, 0},
    {0xa35f715ebf0045c5ULL, 0x8bd496b4f7a2940aULL, 766, 14, 0},
    {0xa365cde95e080eebULL, 0xa5c10cbb77ba038aULL, 1012, 16, 0},
    {0xa38f5e6b10ef1541ULL, 0xbd8d0c2c5255f432ULL, 983, 16, 0},
    {0xa3b08de91e16e7aaULL, 0xa55c6dc8c7686a42ULL, 308, 16, 0},
    {0xa3b22f1e6055bf35ULL, 0x3834ef2193e8e050ULL, 130, 16, 0},
    {0xa3c572d5c0f53a84ULL, 0xbca8ab9051f8fd08ULL, 1047, 16, 0},
    {0xa3e95f9d71c2ce57ULL, 0x9560a061806694f2ULL, 274, 16, 0},
    {0xa432d34b53eadbf2ULL, 0x465c980e364e3d72ULL, 962, 16, 0},
    {0xa45f4fb6e54b23a1ULL, 0xf1e60175888a0a00ULL, 744, 14, 0},
    {0xa484049256310f2eULL, 0x08e2def1d1d69132ULL, 731, 14, 0},
    {0xa4975c86f1633f68ULL, 0xd0774095467a428aULL, 771, 14, 0},
    {0xa4e16d438afa0f22ULL, 0x3a52ac9c790d8fb2ULL, 152, 16, 0},
    {0xa4e5585e82805541ULL, 0x6688b8c78c0ef6faULL, 298, 16, 0},
    {0xa4f7dfa6bc40a298ULL, 0x3a9c2cb4e6f0cbb2ULL, 443, 16, 0},
    {0xa505da764b4d1043ULL, 0x07674ace06ff1ae2ULL, 259, 16, 0},
    {0xa5721e0ba897c72aULL, 0xfb40fcdea9f65f5aULL, 1006, 16, 0},
    {0xa58ec5393a07d3a8ULL, 0xe08ca75b6b3d6cb8ULL, 497, 16, 0},
    {0xa59904e6373bb80bULL, 0xe64ae55aba7d0f62ULL, 1012, 16, 0},
    {0xa59947ff8364da32ULL, 0xf5d81d56f6f12d72ULL, 251, 16, 0},
    {0xa59d660ccf9ba230ULL, 0x643f204530feb7d2ULL, 358, 16, 0},
    {0xa5a0457536a6e4e2ULL, 0x345c6e742bc786a2ULL, 122, 16, 0},
    {0xa5a1edd902bec6f0ULL, 0xae49c5d658a7e6f8ULL, 971, 16, 0},
    {0xa5adb1111aab26fcULL, 0xded54d5d88144d9aULL, 711, 14, 0},
    {0xa5b7af0d283c7006ULL, 0x7586b3d4ddf2a702ULL, 796, 14, 0},
    {0xa5cb4f64d1f8be20ULL, 0xc3de5de4447f91e8ULL, 263, 16, 0},
    {0xa5e1dda7561a6513ULL, 0x6af9c27b08f7ddeaULL, 976, 16, 0},
    {0xa61f0400ea7d6239ULL, 0xbf4245ae7bcd01caULL, 287, 16, 0},
    {0xa633f9b84e5c3002ULL, 0x247c9fd25c456808ULL, 321, 16, 0},
    {0xa63ba2c2748ee0b3ULL, 0xd7c4a64bcc667182ULL, 773, 14, 0},
    {0xa650c2f41eece94aULL, 0xe05d59e215db7412ULL, 793, 14, 0},
    {0xa658fc5248b5d884ULL, 0x28798a30cbd95ba0ULL, 374, 16, 0},
    {0xa67982c771b1499cULL, 0x1b15981e1ee60d36ULL, 701, 13, 0},
    {0xa694fc6dc9b3ddb7ULL, 0xafbb0ddd70214512ULL, 816, 14, 0},
    {0xa6a8e8f6d7c1fb49ULL, 0x4f651237395e4102ULL, 107, 13, 0},
    {0xa7080a805e689bf5ULL, 0x89cf50fb094c4d98ULL, 234, 16, 0},
    {0xa71b614b22ef0e44ULL, 0xcc909b69e4a45b62ULL, 770, 14, 0},
    {0xa75a2938cfe435f6ULL, 0xb539809bfe158842ULL, 755, 14, 0},
    {0xa7ad538147caacecULL, 0xa6518ab757495030ULL, 1046, 16, 0},
    {0xa7ba4fba5ce92fe8ULL, 0xcd8adaca227c836aULL, 716, 14, 0},
    {0xa7c59e546ee7aa7fULL, 0xef0763a6a13f4a82ULL, 801, 14, 0},
    {0xa7dbac125c9a1c96ULL, 0x3f2cda18f76f667aULL, 25, 13, 0},
    {0xa8180823126940e8ULL, 0x2248cc020f2dd95aULL, 805, 14, 0},
    {0xa81865820b804998ULL, 0xc07c128cf6690ba6ULL, 15, 12, 0},
    {0xa81d3c7524cc6dd5ULL, 0x43cc8fa46d342fdaULL, 48, 13, 0},
    {0xa8785ee1b370804aULL, 0x489fa00b2d244a14ULL, 730, 14, 0},
    {0xa898b44f734040dbULL, 0xf7e302327b4c82e2ULL, 297, 16, 0},
    {0xa8a1ff093c9fb0ffULL, 0xe00c3c3a07f727faULL, 758, 14, 0},
    {0xa8cadaed85b5bbcdULL, 0xbc924cbc58cdb19aULL, 990, 16, 0},
    {0xa8e480739eaefcd3ULL, 0x58b146ee89b2a2b8ULL, 972, 16, 0},
    {0xa8effdeff3dd3026ULL, 0x1bd5f4eb7a689448ULL, 744, 14, 0},
    {0xa8f43d5014eb1564ULL, 0x6aae40c5c20e1682ULL, 791, 14, 0},
    {0xa90955181f277b74ULL, 0x687b41f8fb94ce02ULL, 665, 10, 0},
    {0xa90f97c82cad531fULL, 0x7d502d5990ca7e8aULL, 720, 14, 0},
    {0xa946d6de7a17c89cULL, 0x5733f2d6d72c0070ULL, 453, 16, 0},
    {0xa95b361d9bfe9b72ULL, 0x994953e36d6ead52ULL, 445, 16, 0},
    {0xa9660feff420ea69ULL, 0x5f7556e71032cc7cULL, 808, 14, 0},
    {0xa99c0758e8dd21f4ULL, 0xd012c4322ac1cef8ULL, 186, 16, 0},
    {0xa9a73eb5b07bf58eULL, 0x1266a8e5872d46daULL, 776, 14, 0},
    {0xa9c3c7827f26a4b0ULL, 0xbaf22303bebdfd0cULL, 797, 14, 0},
    {0xa9ca1a8a44fcb94fULL, 0x4707fabe923efc8aULL, 222, 16, 0},
    {0xa9cc6d171deba5fbULL, 0xf5a146c871e9b1e2ULL, 362, 16, 0},
    {0xa9ce208e70fa2828ULL, 0xf48a834abe88a270ULL, 578, 16, 0},
    {0xa9d442927c27e1e3ULL, 0x4f8e0f7cc9eb4582ULL, 232, 16, 0},
    {0xa9d7431dc9b1126aULL, 0x16f3854df12050d0ULL, 638, 16, 0},
    {0xa9dd5e077e7071d8ULL, 0x5f101f56836457baULL, 612, 16, 0},
    {0xa9f1840046ebe66fULL, 0x308d9b53f4fffd18ULL, 155, 16, 0},
    {0xa9f61d798a3ab2d6ULL, 0x685d278cd82b73a8ULL, 783, 14, 0},
    {0xaa11fad63814c8ddULL, 0x1e5767e2647d692aULL, 760, 14, 0},
    {0xaa13b0bbf65a0938ULL, 0xa9487308ed1b3752ULL, 344, 16, 0},
    {0xaa1df556b9d9cd15ULL, 0xd12d92ceb5f63d62ULL, 799, 14, 0},
    {0xaa28864c6ca7434dULL, 0x296063e20f3200baULL, 594, 16, 0},
    {0xaa4f13dbd6ae3caaULL, 0x70c51b19e1cacee8ULL, 721, 14, 0},
    {0xaa677573a65c3813ULL, 0x2f2166400103b7c6ULL, 850, 13, 0},
    {0xaa6e4a142deb8eccULL, 0x00242eb41606ba6aULL, 52, 13, 0},
    {0xaa87846143671d58ULL, 0xe3a2708e46f99812ULL, 817, 14, 0},
    {0xaa9a18e60f05aa0aULL, 0x63967a758601efbaULL, 762, 14, 0},
    {0xaaa688877e40c3faULL, 0x74e9d2277afb16f8ULL, 784, 14, 0},
    {0xab04f2aa6db7c870ULL, 0x3f52a5b5a3582b8aULL, 720, 14, 0},
    {0xab07f3056e3a4d44ULL, 0x02eafc68090afed8ULL, 759, 14, 0},
    {0xab0f6ab8eba8098eULL, 0xe145f9345d007ef2ULL, 177, 16, 0},
    {0xab17af7b0df70857ULL, 0x67c7dee65a3dc590ULL, 582, 16, 0},
    {0xab35475bc0652c10ULL, 0xd107b9acdbe944caULL, 509, 16, 0},
    {0xab41078c7660b8abULL, 0x18353a645770fef8ULL, 786, 14, 0},
    {0xab64d3fe7f38a099ULL, 0xb7f7d167503fe020ULL, 595, 16, 0},
    {0xab6e0509ccc06706ULL, 0xcf86bfce56cc4f7aULL, 724, 14, 0},
    {0xab6e46f40c2428d3ULL, 0x38a39a1426323482ULL, 279, 16, 0},
    {0xab6e6fb69f5d2a83ULL, 0xba9e4afa29320830ULL, 912, 16, 0},
    {0xab88f2e3ce62dc41ULL, 0xd2baee86a117b7caULL, 635, 16, 0},
    {0xab8fe38150bff5afULL, 0xd296d096a0e5a35aULL, 817, 14, 0},
    {0xab9321cb6051f4ebULL, 0x036e9107ef45fd1aULL, 31, 13, 0},
    {0xab9a90690f0b3d60ULL, 0xb3f7ef27f340eee2ULL, 255, 16, 0},
    {0xabb2e4221bf6c5aeULL, 0xe24843cf5a8ddccaULL, 737, 14, 0},
    {0xabb6660e82f2e264ULL, 0x5b831e4b3e7ba6daULL, 455, 16, 0},
    {0xabcb3bb4d57f70b3ULL, 0x12dad0a7060aaee8ULL, 927, 16, 0},
    {0xabcc0dddec1ba3beULL, 0x28ca65d04925366aULL, 799, 14, 0},
    {0xabd88266e7be2a76ULL, 0x761739bf9ec923e8ULL, 347, 16, 0},
    {0xabda0837efccc99dULL, 0xae7307d33d9865c2ULL, 107, 13, 0},
    {0xabe284b71fd10c32ULL, 0xaa15956a3d87278aULL, 379, 16, 0},
    {0xabe83f8107b87a7aULL, 0xd20a88ea3a200d92ULL, 819, 14, 0},
    {0xabea7f0bb3f87096ULL, 0xf0573f636a66de22ULL, 796, 14, 0},
    {0xabf25e24edc9c2e7ULL, 0x689f7f49b78d3ed2ULL, 84, 13, 0},
    {0xac08eb2f8afa81c3ULL, 0xff677864d71bcb80ULL, 745, 14, 0},
    {0xac11a778e750a4e7ULL, 0x3bd5d0ac08f712caULL, 647, 16, 0},
    {0xac42f32dd01db6d4ULL, 0xf13b1cddf7b39588ULL, 411, 16, 0},
    {0xac462b9c08d7a477ULL, 0x13b59764e7fe48ceULL, 657, 7, 0},
    {0xac55637816b92622ULL, 0xe9f42efca117fc30ULL, 726, 14, 0},
    {0xac6e43afe83b3b8aULL, 0x4bbf3b08cc4ceac0ULL, 744, 14, 0},
    {0xac8ee23c653cd2b9ULL, 0xe87271e9dd88ee8aULL, 801, 14, 0},
    {0xac9a0d1a96b8f635ULL, 0xb6609e7708a6058aULL, 1007, 16, 0},
    {0xacac9866d167e0c3ULL, 0xe6fa7f5a6753e110ULL, 585, 16, 0},
    {0xacb3701a1c1089cbULL, 0xc86cd22e05a7d312ULL, 179, 16, 0},
    {0xacfd49a704e0b2d7ULL, 0x97ec1227f0aec792ULL, 966, 16, 0},
    {0xad1c31ebcb7e1238ULL, 0xbe9a9c253ebe52b2ULL, 267, 16, 0},
    {0xad35a8f0f0d3125dULL, 0x8a9cddd375eb6148ULL, 1041, 16, 0},
    {0xad3a49c0ddf227cdULL, 0x496592adb7ce9e32ULL, 304, 16, 0},
    {0xad453b268c84f726ULL, 0x3a8270978cb313f2ULL, 152, 16, 0},
    {0xad5812f7cc44b978ULL, 0x9a0e9ff6cfe6e2ccULL, 34, 13, 0},
    {0xad5caa3cd8817926ULL, 0x3b679c13985f95daULL, 776, 14, 0},
    {0xad683330c497b7fdULL, 0xa04ff0807efe1d8aULL, 788, 14, 0},
    {0xad717a7971472a17ULL, 0xe345dfbbe8461008ULL, 932, 16, 0},
    {0xada70108c0b897c8ULL, 0x597a3012daf77e48ULL, 319, 16, 0},
    {0xadc9f60fcaa97bf1ULL, 0x879a608708a7afd8ULL, 332, 16, 0},
    {0xadd35cd433984072ULL, 0xa18415f29c1f9716ULL, 843, 13, 0},
    {0xade4bcfe8bf422eaULL, 0x024cb79b544421b2ULL, 367, 16, 0},
    {0xadf7dc686fbd193aULL, 0x26071ce125d15288ULL, 471, 16, 0},
    {0xae05a0e73c18069eULL, 0xe6c2f2f38a4980e2ULL, 1012, 16, 0},
    {0xae1467c5cfd40f42ULL, 0x18c2bcfb4306a6d2ULL, 984, 16, 0},
    {0xae4184570f94788aULL, 0x63bc5084e535ae12ULL, 458, 16, 0},
    {0xae6647de38b06c49ULL, 0xdd107f3b3ca2f3f8ULL, 139, 16, 0},
    {0xae836a9d14d6f037ULL, 0x883be41aeb4fcaa4ULL, 43, 13, 0},
    {0xae8e1f39619a1737ULL, 0x321c67249fd54eeaULL, 80, 13, 0},
    {0xae9d806633e6798eULL, 0x8b4662e0045a3460ULL, 571, 16, 0},
    {0xaebdc901765b075dULL, 0x9b1cea7421fc8fb2ULL, 803, 14, 0},
    {0xaed323afc0eed632ULL, 0x6b64d353892c63faULL, 630, 16, 0},
    {0xaefc0ec01412ff32ULL, 0x67e995cefd98ec94ULL, 772, 14, 0},
    {0xaf0b4e42fd60991fULL, 0xe7199730a21ecd4eULL, 866, 13, 0},
    {0xaf0f04533da9edd1ULL, 0x97a4f4d13059fa20ULL, 759, 14, 0},
    {0xaf39f429ab8cc250ULL, 0x7469677138f380bcULL, 832, 14, 0},
    {0xaf3ccdc12a845c2dULL, 0x58e25c209ddff5eaULL, 291, 16, 0},
    {0xaf3f12a0c318cb46ULL, 0xe4f8dbf4092250f0ULL, 888, 16, 0},
    {0xaf7fc62b3262069cULL, 0x8aa097c5c8c1d508ULL, 239, 16, 0},
    {0xafb49f8f2d930202ULL, 0x2a9f811d2d17c7daULL, 467, 16, 0},
    {0xafb7809a0aca1c2fULL, 0x2df7b91e956d8ac2ULL, 496, 16, 0},
    {0xafbc5b3235e4d4fdULL, 0x867e6065dd5d8df8ULL, 338, 16, 0},
    {0xafbd7eb6f59b81e8ULL, 0x7199310da8963c02ULL, 478, 16, 0},
    {0xafc9caca012c3bacULL, 0x96e73f4d2b59c772ULL, 581, 16, 0},
    {0xafcf80ba20e28016ULL, 0x76250a308b53327aULL, 400, 16, 0},
    {0xaff0f5e466f0c301ULL, 0xcbca810645c4b21eULL, 693, 13, 0},
    {0xb006dee8b4397653ULL, 0x229b1bda1e4e6700ULL, 399, 16, 0},
    {0xb010b0bfb27c848dULL, 0x1bc013c92bce39e0ULL, 1024, 16, 0},
    {0xb01ad5687fcfc141ULL, 0xbf708ef8b2b2f448ULL, 750, 14, 0},
    {0xb0a5bb4ec0f921fbULL, 0xc81dcd0794692942ULL, 530, 16, 0},
    {0xb0acb64c6012e367ULL, 0x8aad993362ff9940ULL, 396, 16, 0},
    {0xb0d24620bb5b4636ULL, 0x5dd6f88f742792e2ULL, 469, 16, 0},
    {0xb0d885b816a453e1ULL, 0x142fb5ce5cea65aaULL, 659, 10, 0},
    {0xb10af93f498975e7ULL, 0x98572b988bce94c6ULL, 849, 13, 0},
    {0xb1117cbfe3056836ULL, 0xa8245909054997baULL, 951, 16, 0},
    {0xb111f4df63330a41ULL, 0xda2ec9b643f7a6c2ULL, 364, 16, 0},
    {0xb12f524e4df35c4aULL, 0x2d2eab8fc19a7d42ULL, 718, 14, 0},
    {0xb147ee37fba7b6b6ULL, 0x5bc29dc43af07d98ULL, 216, 16, 0},
    {0xb17fd0b75738038fULL, 0xa67649929aa18730ULL, 398, 16, 0},
    {0xb1adea2751dccb07ULL, 0xf948a76f01141f1aULL, 389, 16, 0},
    {0xb1bc932a3288bc05ULL, 0x43dbb47f16ab0a88ULL, 721, 14, 0},
    {0xb1c187145caf1cb4ULL, 0x945239680536aa1cULL, 81, 13, 0},
    {0xb22599c21c43857eULL, 0xaeb2c835cac0a322ULL, 511, 16, 0},
    {0xb234007736c06846ULL, 0x2fc334a06c472b3aULL, 791, 14, 0},
    {0xb234c91c597d8648ULL, 0x003ce8d1b98d7540ULL, 556, 16, 0},
    {0xb23cc00713b887c1ULL, 0x6b7afe8d23c49ec8ULL, 433, 16, 0},
    {0xb248d51f1fae096bULL, 0x9f8119c0801d6264ULL, 36, 13, 0},
    {0xb2495550bd32d8e3ULL, 0x9e8ce02f56505deaULL, 886, 16, 0},
    {0xb28ba564a55ded94ULL, 0xe4cdadedb3740b38ULL, 306, 16, 0},
    {0xb2c65515c61e4e7dULL, 0x427f4b5c58f43ddcULL, 772, 14, 0},
    {0xb2c8aa25a9803a46ULL, 0x9b6b523dee69ed18ULL, 422, 16, 0},
    {0xb2d077e68c7d4aafULL, 0x139e0bb89fd4dc0cULL, 79, 13, 0},
    {0xb2de9dd425770deaULL, 0x34cf82b9b0b5b180ULL, 721, 14, 0},
    {0xb2fc717772a95099ULL, 0x45f7ab5a5a3dc502ULL, 734, 14, 0},
    {0xb348d65f2c8e1021ULL, 0xb286831300479442ULL, 454, 16, 0},
    {0xb398f175f7084f97ULL, 0x1f14f4eb62ed49eaULL, 781, 14, 0},
    {0xb3be18c865dc1eb0ULL, 0x94adf6d3d7584a60ULL, 759, 14, 0},
    {0xb3fc303649b149f0ULL, 0xa1ec3c6991478be2ULL, 557, 16, 0},
    {0xb40bc22a93f90ca7ULL, 0x9e19a2995a32898aULL, 724, 14, 0},
    {0xb416ab692e4a5eeeULL, 0x4a08c643a4697bd8ULL, 243, 16, 0},
    {0xb41c111f1707a68aULL, 0xac0306dd5b2b5d64ULL, 118, 13, 0},
    {0xb43fd38a351ff528ULL, 0x69b949946c3110f2ULL, 166, 16, 0},
    {0xb44fc2a401b8b2cbULL, 0xa3507b11b4894d82ULL, 617, 16, 0},
    {0xb458efe49ddd08abULL, 0x6119fe8759d79882ULL, 941, 16, 0},
    {0xb48414ab78addbecULL, 0x0f5597700f465738ULL, 759, 14, 0},
    {0xb4a2e561603617b4ULL, 0x0f596b9b71c86980ULL, 140, 16, 0},
    {0xb4a340d2630b0ac6ULL, 0x723786c4621f23c8ULL, 386, 16, 0},
    {0xb4c78e2da5541923ULL, 0x1743209c35aa797aULL, 820, 14, 0},
    {0xb4d8437acf3e325eULL, 0x6a5359b71612bdaaULL, 859, 13, 0},
    {0xb4ea03b846c0d5a7ULL, 0x424238690cf6b202ULL, 90, 13, 0},
    {0xb500504ff26c8dd2ULL, 0xc14ee5eaeb88ea30ULL, 294, 16, 0},
    {0xb54d36ef0515a151ULL, 0x29c83a1b9d806f62ULL, 70, 13, 0},
    {0xb54e73cbc891830dULL, 0xa3c3378de0ebb210ULL, 550, 16, 0},
    {0xb56808cb82e1cc5cULL, 0x338bb0a9a7efc632ULL, 182, 16, 0},
    {0xb56dfc4b87737a05ULL, 0xc8cc3f42c07df6daULL, 734, 14, 0},
    {0xb573d71f551dab1fULL, 0x244d26cc98717bb2ULL, 292, 16, 0},
    {0xb5a05c798f68608aULL, 0x289b9add99daf472ULL, 803, 14, 0},
    {0xb5a4f45199755f73ULL, 0xce7887f2e04e9824ULL, 39, 13, 0},
    {0xb5babab21f6c85b0ULL, 0x9bc0c0e73e2b6828ULL, 383, 16, 0},
    {0xb5bf845ea130f348ULL, 0x5169b2d4ef25e872ULL, 886, 16, 0},
    {0xb5cc5a7f273d0e6bULL, 0x4cc5bbf1df808f58ULL, 905, 16, 0},
    {0xb5f8c41bad7b29d7ULL, 0x9932b34e2ea0cc02ULL, 979, 16, 0},
    {0xb63f5b78b31853f2ULL, 0x19edefebb2f580a2ULL, 801, 14, 0},
    {0xb662c148f77d2525ULL, 0xd0f75e5d4a8903c2ULL, 427, 16, 0},
    {0xb66f3414b7a6c4c5ULL, 0xfe8742ec2b97479aULL, 798, 14, 0},
    {0xb6763d32c26972ebULL, 0xfb41cf42c67dbbf8ULL, 726, 14, 0},
    {0xb68ccff650ad917cULL, 0xca30e85147b6309cULL, 730, 14, 0},
    {0xb694121fea0368c3ULL, 0xa1cb1dcd7fa9bd68ULL, 926, 16, 0},
    {0xb6c50e2966d83f72ULL, 0x728700a0c3408a38ULL, 955, 16, 0},
    {0xb70940a7f1e4814eULL, 0x763f3c4cec282abaULL, 775, 14, 0},
    {0xb71fe37112999ba9ULL, 0x4c7d1f90f8dd9738ULL, 522, 16, 0},
    {0xb725353b0566953fULL, 0x8e45e7c78c982f7aULL, 197, 16, 0},
    {0xb7356c209c93b363ULL, 0x8917f64298451720ULL, 129, 16, 0},
    {0xb739417c943809eeULL, 0x72877fe617622090ULL, 733, 14, 0},
    {0xb748c780a711a7deULL, 0x0d24badf1e138402ULL, 196, 16, 0},
    {0xb74d2c7e453d1f15ULL, 0x2888b6f813713e7aULL, 990, 16, 0},
    {0xb758c62734f0d776ULL, 0x95f7123c55406b20ULL, 619, 16, 0},
    {0xb7dc1f497266a336ULL, 0x0080bd1535785ac0ULL, 193, 16, 0},
    {0xb7e3b14de9c49facULL, 0x74365a59aa97e848ULL, 334, 16, 0},
    {0xb7e988bff752c3f9ULL, 0x6856fc0f4e3fabb8ULL, 316, 16, 0},
    {0xb84024ac7e9e45f0ULL, 0x4f9fba33fcadbc4aULL, 948, 16, 0},
    {0xb84694a6809a79bdULL, 0x945651e7821c5b20ULL, 1015, 16, 0},
    {0xb8499a2aa41b8581ULL, 0x0d46a695fa686dd8ULL, 424, 16, 0},
    {0xb866642f850296e1ULL, 0x90ce84bfd011f7aaULL, 430, 16, 0},
    {0xb88018e0461f90d3ULL, 0x380569cd79247ccaULL, 790, 14, 0},
    {0xb884585530561a96ULL, 0xe3a90a11276961b2ULL, 473, 16, 0},
    {0xb88ee8fd518be213ULL, 0x418f781ccc2abebcULL, 730, 14, 0},
    {0xb8c29a410563243aULL, 0xeda520b50490bfd8ULL, 757, 14, 0},
    {0xb8c40f09069898a0ULL, 0xa81b31e838b68620ULL, 623, 16, 0},
    {0xb920b9d5a7c35ba8ULL, 0x2f64eca94cada3c0ULL, 652, 16, 0},
    {0xb935f8379c09cf59ULL, 0x59b3fc498c062ce8ULL, 132, 16, 0},
    {0xb93ecb3f23ecf633ULL, 0x915f86bb7a25d9c2ULL, 572, 16, 0},
    {0xb9608952ce5fc275ULL, 0x20272a1071bda3e8ULL, 818, 14, 0},
    {0xb960c8bd75f37d06ULL, 0x8464f92c937fb378ULL, 281, 16, 0},
    {0xb96893043e53dfc2ULL, 0xbff7b38ec714adfaULL, 717, 14, 0},
    {0xb968d458f1cb3d7aULL, 0xe06a331802df0e62ULL, 831, 14, 0},
    {0xb99a30ed00b34317ULL, 0xf2bb73f5a792759aULL, 755, 14, 0},
    {0xb9a4e64437e33914ULL, 0x4cca6cf0d5b0a876ULL, 848, 13, 0},
    {0xb9b18a1c450de88aULL, 0xaaf7493b4da78768ULL, 738, 14, 0},
    {0xb9b7df1a8d908a43ULL, 0x2cc88b1e5ec68352ULL, 302, 16, 0},
    {0xb9c242ea75001140ULL, 0x38812eccebd9b2a2ULL, 115, 13, 0},
    {0xb9d43727cd837d2fULL, 0xd70face7e235b982ULL, 446, 16, 0},
    {0xb9dc963beaa1bc46ULL, 0x41fcea9c71145732ULL, 510, 16, 0},
    {0xb9dde82f1f6a90d7ULL, 0x241d817e810385c2ULL, 507, 16, 0},
    {0xb9e70800f9424f1eULL, 0x11331bb20d13471aULL, 936, 16, 0},
    {0xb9e94716ec1e55d1ULL, 0x79295b30569f8b88ULL, 158, 16, 0},
    {0xb9eed25d6e5a8163ULL, 0xa379a3556e1c95b4ULL, 824, 14, 0},
    {0xba0679f4de2565f1ULL, 0xacac0133ba59797aULL, 105, 13, 0},
    {0xba0a95736ed0c0fcULL, 0xe6a58ba96efe5e98ULL, 916, 16, 0},
    {0xba235e822e9b58caULL, 0xfe81ee8c9fb8b862ULL, 993, 16, 0},
    {0xba28e69d61b2ee52ULL, 0x1e916ba44acd7b80ULL, 209, 16, 0},
    {0xba5f30fb4c10bf48ULL, 0xdbf61b4620f9c758ULL, 336, 16, 0},
    {0xba8e5dc34d2746a3ULL, 0x192f3c53a1481f28ULL, 276, 16, 0},
    {0xba99b46c0be6a61eULL, 0x24f206f4ecad0ebeULL, 658, 7, 0},
    {0xbaa6711336175437ULL, 0xfe4f261256f194d2ULL, 731, 14, 0},
    {0xbab5730d5327c616ULL, 0x482207f55342f8bcULL, 804, 14, 0},
    {0xbacc6e018e249dedULL, 0x326cfdfcdd6dfb3cULL, 777, 14, 0},
    {0xbaf1e938252ec6e8ULL, 0x64f2f162b6c6bb58ULL, 141, 16, 0},
    {0xbaf8bf422f4917feULL, 0xc4ac2a6fa353cfeaULL, 367, 16, 0},
    {0xbb29a61e0dd77865ULL, 0xaddef929329c771aULL, 754, 14, 0},
    {0xbbb4eb4e9efa7ac6ULL, 0xc90f1d698d492c16ULL, 7, 12, 0},
    {0xbbbe59387938ef07ULL, 0x69bfef6636519b52ULL, 903, 16, 0},
    {0xbbbf069bc9aeefcdULL, 0x381a5a0d287c82acULL, 109, 13, 0},
    {0xbbcb843f339ed5f0ULL, 0xaa6a1b7ffe8d52eaULL, 815, 14, 0},
    {0xbbd4c774eafc3dbfULL, 0xe9b632d99f0af76aULL, 801, 14, 0},
    {0xbbde6f532f4030bbULL, 0x2d067a5b785d1294ULL, 810, 14, 0},
    {0xbbe1959396460177ULL, 0xea4c315515103e00ULL, 538, 16, 0},
    {0xbbefaf9b432fd11eULL, 0xe77fb747a5a708acULL, 51, 13, 0},
    {0xbc05f27680f55e93ULL, 0xb40f48940f826a82ULL, 954, 16, 0},
    {0xbc0bbfab6dd8ae2dULL, 0x919c1b58fc8e980cULL, 64, 13, 0},
    {0xbc0f80eb2f2c7359ULL, 0xb8f3dc32c460f816ULL, 700, 13, 0},
    {0xbc12453593fadb09ULL, 0x53f90ac41fa077eaULL, 954, 16, 0},
    {0xbc38e32a789228eaULL, 0x6bc07cbed479f5f8ULL, 926, 16, 0},
    {0xbc3b554157998a5cULL, 0xe2053a5c16787862ULL, 996, 16, 0},
    {0xbc3baae0ab9bdd73ULL, 0xf190f601f38811dcULL, 778, 14, 0},
    {0xbc4f2dd40d95747eULL, 0x07b6447d3482c34aULL, 200, 16, 0},
    {0xbc538a6a5a9aa57bULL, 0x835f7ffb809740dcULL, 94, 13, 0},
    {0xbc5de2cc86cb95f9ULL, 0xf9b8e8ec623cabb2ULL, 734, 14, 0},
    {0xbc72900862e329e3ULL, 0x8f7f6965778a7368ULL, 946, 16, 0},
    {0xbc73709cd78f444cULL, 0x8735fa3e2e46376cULL, 740, 14, 0},
    {0xbc89de31f3e45e44ULL, 0xbee1cd1208595500ULL, 249, 16, 0},
    {0xbcbbf4d1b36233daULL, 0x42b2595b66503092ULL, 1010, 16, 0},
    {0xbcd1c85842b73ca5ULL, 0x989b29f3bb134302ULL, 952, 16, 0},
    {0xbcdf38d25031c2e0ULL, 0x92531b17c58e2d68ULL, 337, 16, 0},
    {0xbd126ec8cee527a2ULL, 0xad594dfdc698c0baULL, 355, 16, 0},
    {0xbd56b864b0037c52ULL, 0x95fd92939ff5a1a0ULL, 634, 16, 0},
    {0xbda8adb36157c359ULL, 0x116c4d63aa0684f0ULL, 293, 16, 0},
    {0xbdb3dbb758f1c691ULL, 0xe960926a8a157168ULL, 185, 16, 0},
    {0xbdd9cc5cf1a3aa88ULL, 0xd4a09b6af2dfcb40ULL, 969, 16, 0},
    {0xbe213cfa6518e0f5ULL, 0x7778ec9d42211572ULL, 213, 16, 0},
    {0xbe505c4177c39cc5ULL, 0x8cbafc7b27a07476ULL, 13, 12, 0},
    {0xbe554bf2430aef77ULL, 0xc278ed7a19771ef0ULL, 449, 16, 0},
    {0xbe55e98ea4043781ULL, 0x04954f881eb69002ULL, 731, 14, 0},
    {0xbe722837501001c1ULL, 0xb2c5dae21ad23f0aULL, 770, 14, 0},
    {0xbe9aa2f1cd348ce8ULL, 0x47777bd36b6394a2ULL, 615, 16, 0},
    {0xbeaa985328ce451bULL, 0xd92675df118819a8ULL, 349, 16, 0},
    {0xbeb24235c12d60a0ULL, 0xe082c0a8e81bf926ULL, 6, 12, 0},
    {0xbec734b6ec09fa35ULL, 0x8f6fd00bc6a550c2ULL, 450, 16, 0},
    {0xbee93c522f5ff159ULL, 0x1bfbb48b388ac0fcULL, 833, 14, 0},
    {0xbef53cda18bdc473ULL, 0x9d610ef4b5041e3aULL, 731, 14, 0},
    {0xbf113399682de16fULL, 0xf7cfb2b179a38b3aULL, 722, 14, 0},
    {0xbf2ebd891ae49012ULL, 0xab3472faa931cc6cULL, 64, 13, 0},
    {0xbf53d6b59cd400beULL, 0x765152e4f8865170ULL, 757, 14, 0},
    {0xbf5a9a5f552be28bULL, 0x363f9c6d674ac738ULL, 757, 14, 0},
    {0xbf6e6e25ab335d4aULL, 0xa5f2c56390d46638ULL, 339, 16, 0},
    {0xbf7c2501aeb37d3dULL, 0x841386597696f0feULL, 676, 13, 0},
    {0xbfa3c33d930c8eeaULL, 0xb385f6e7ec32c9b2ULL, 491, 16, 0},
    {0xbfa46c110b5c13bcULL, 0xd71d4ca77f8a41c6ULL, 657, 7, 0},
    {0xbfa8b6bbae688d90ULL, 0x08de3bd4f387d902ULL, 857, 13, 0},
    {0xbfcf1e2d0fd2c225ULL, 0x84cbe724fbe143b0ULL, 1017, 16, 0},
    {0xbfd2d42b517a8f36ULL, 0x8dde454de98457e2ULL, 862, 13, 0},
    {0xc00abe6a84cf40dfULL, 0x13e76b3cc9585dd0ULL, 320, 16, 0},
    {0xc0319a575162f5c4ULL, 0x7ad6a3c6d92eef2aULL, 798, 14, 0},
    {0xc04fb1cd856c95f4ULL, 0x2a483445cfb221d2ULL, 68, 13, 0},
    {0xc0575d8761c27852ULL, 0x61750d959040fcaaULL, 231, 16, 0},
    {0xc07b8f5cbca0378cULL, 0xd9d5f0f630702ce0ULL, 738, 14, 0},
    {0xc089705ce0124b63ULL, 0x89a13cb69c4feee2ULL, 463, 16, 0},
    {0xc0e91140a4e2b603ULL, 0x581b347f0f889398ULL, 518, 16, 0},
    {0xc10be07387d3814aULL, 0x3b1337d0178198c2ULL, 205, 16, 0},
    {0xc1762de49c25ea61ULL, 0xd01c45e515d588fcULL, 807, 14, 0},
    {0xc17ebb4b76d8f22bULL, 0xe651e34c656ab7dcULL, 88, 13, 0},
    {0xc18a79718e0076adULL, 0x86c30870d35430daULL, 834, 14, 0},
    {0xc18f2b8f87baee90ULL, 0xdab9670f8bb9af82ULL, 357, 16, 0},
    {0xc190b14aed265447ULL, 0x7d47d980d494d8feULL, 707, 12, 0},
    {0xc226e3ebdc20557bULL, 0x698aa77cf74cad78ULL, 336, 16, 0},
    {0xc22702fc27d95c66ULL, 0xaf38baaeaa28c15aULL, 536, 16, 0},
    {0xc24e2beb2d34bbc8ULL, 0xb70112883479a132ULL, 381, 16, 0},
    {0xc2515f610cbba1deULL, 0x244dcbe37642e73aULL, 28, 13, 0},
    {0xc2894514836b58f9ULL, 0x463c7c5634883fa8ULL, 967, 16, 0},
    {0xc2a60549a287108dULL, 0x6f64df70a9b412eaULL, 272, 16, 0},
    {0xc2a811b41d469cbcULL, 0xad2daf196aa6a35aULL, 731, 14, 0},
    {0xc2c04b9bee947da5ULL, 0x9095154f7dd5ef6aULL, 952, 16, 0},
    {0xc2f89d102cc48518ULL, 0xb52d89b2dc50ba88ULL, 236, 16, 0},
    {0xc2ffe6b40ffdd1a9ULL, 0x4f1683be683739d2ULL, 84, 13, 0},
    {0xc34571cbaab761baULL, 0x67af70a3c5a7514aULL, 781, 14, 0},
    {0xc357bf75e2d3d772ULL, 0x4400249e565174caULL, 610, 16, 0},
    {0xc371fd4191089e12ULL, 0x1a822b8fa1e0f290ULL, 228, 16, 0},
    {0xc37b58b297a7fda8ULL, 0xe6ffef79e9917762ULL, 748, 14, 0},
    {0xc399b7d9bc83d2c7ULL, 0x9d536ba43801b82aULL, 986, 16, 0},
    {0xc39a7bb5c4af1e88ULL, 0x896d6ef601b68582ULL, 624, 16, 0},
    {0xc3b34ba3575cf203ULL, 0xa302a5b8a6cf655eULL, 870, 13, 0},
    {0xc3bedfafae1ba9abULL, 0x475cd11ea5743ae2ULL, 254, 16, 0},
    {0xc3d18995c4bfc261ULL, 0x8d05386683a00508ULL, 421, 16, 0},
    {0xc3d95374bbec5002ULL, 0xa672037bb9d86722ULL, 724, 14, 0},
    {0xc3e940cea85cb9f0ULL, 0x84856d7877859c2eULL, 838, 12, 0},
    {0xc43415aaa580f061ULL, 0xcfc1a0173f82c170ULL, 488, 16, 0},
    {0xc43d8370bab6db3dULL, 0x521f3a9327657ef2ULL, 270, 16, 0},
    {0xc44974653c6a20f9ULL, 0xef958c4e6b87e162ULL, 956, 16, 0},
    {0xc44ff18e7101724dULL, 0xb982a0712e754482ULL, 857, 13, 0},
    {0xc47ad542f76ce27eULL, 0xb8e27b300b9e6b5aULL, 995, 16, 0},
    {0xc4b4f526679d8321ULL, 0x5f364fcf89ce703aULL, 742, 14, 0},
    {0xc4b9482f345c3406ULL, 0xe968e7e9a4c4826aULL, 138, 16, 0},
    {0xc4be09b9d5d06fe3ULL, 0x3b88f6e798f3a6a0ULL, 426, 16, 0},
    {0xc4c6a5437f3f1188ULL, 0x88e45220ecba3b12ULL, 799, 14, 0},
    {0xc4c6addbb12f67dfULL, 0x67ad6ff847434ca2ULL, 439, 16, 0},
    {0xc4cdbb592328ec8cULL, 0x7295be3c01b3d3e4ULL, 772, 14, 0},
    {0xc4ceff76b394d5e3ULL, 0x19aa79203288b500ULL, 188, 16, 0},
    {0xc4e9a76dfce3a52aULL, 0x109a20ad083f91f2ULL, 802, 14, 0},
    {0xc5092aa4e6b6b94eULL, 0x43c8073314d39daeULL, 683, 13, 0},
    {0xc52cb003d02848efULL, 0xfee312c12822d5faULL, 790, 14, 0},
    {0xc5421da407c74ff2ULL, 0xf422dcbf830f8564ULL, 804, 14, 0},
    {0xc5431f7880caa559ULL, 0x2094e1f72d4e4e08ULL, 318, 16, 0},
    {0xc5a65f363950b39fULL, 0xaafa0c0c64d61562ULL, 753, 14, 0},
    {0xc5d97add73000509ULL, 0x1123ddd48203e9c0ULL, 396, 16, 0},
    {0xc5e3325d6941b770ULL, 0xe4bcf2d367235622ULL, 490, 16, 0},
    {0xc5f3edc47bca6c0bULL, 0xc1fcbb5836a5b388ULL, 744, 14, 0},
    {0xc5f4b79647130a0bULL, 0xc3994fcb499ec654ULL, 41, 13, 0},
    {0xc6090eb0e7fe3747ULL, 0x733534d0f08fa9e2ULL, 455, 16, 0},
    {0xc60b118058ac1beaULL, 0x9f27d2ff8e28ce4cULL, 112, 13, 0},
    {0xc60e9a847806c0a5ULL, 0x88f01ef47faa393cULL, 809, 14, 0},
    {0xc61e75b4380d2a90ULL, 0x92d1a233fe9a92a8ULL, 918, 16, 0},
    {0xc69533879457a244ULL, 0x5049d52d80a0ca58ULL, 932, 16, 0},
    {0xc696c6231663f7c4ULL, 0x1355917f972e0f32ULL, 457, 16, 0},
    {0xc6a260e5cbc7db09ULL, 0x4b562e5bd7fb8deaULL, 782, 14, 0},
    {0xc6c91a2486ed616fULL, 0x8a855f77b5503aeaULL, 54, 13, 0},
    {0xc6cc9e1362734a04ULL, 0x11cc4f1bc9021f60ULL, 733, 14, 0},
    {0xc6d0fb2c1368f154ULL, 0x42b99c1ff39dc8dcULL, 797, 14, 0},
    {0xc6e6857f6857bc5dULL, 0x3f508b462fce3642ULL, 440, 16, 0},
    {0xc7012798c6f7c2dcULL, 0x528e7337ad79fc08ULL, 384, 16, 0},
    {0xc73387f7ff941b0fULL, 0x666976c7fd8df310ULL, 1014, 16, 0},
    {0xc75b211c27c19d76ULL, 0xb9085d46bacb709aULL, 771, 14, 0},
    {0xc76f70746524eb7bULL, 0xadc43919388f32a2ULL, 805, 14, 0},
    {0xc77265f8ea6696b9ULL, 0x31c4d7d1a8491dd0ULL, 611, 16, 0},
    {0xc775f47085101feaULL, 0x3ac6bbd1527bc4caULL, 113, 13, 0},
    {0xc7ac441cf87772a0ULL, 0x13de52e9b7706bcaULL, 437, 16, 0},
    {0xc7be877146a6ac4bULL, 0x2db6d72c95cc4a10ULL, 757, 14, 0},
    {0xc7f30bd2eed5113cULL, 0x40fda413e5812d72ULL, 931, 16, 0},
    {0xc83d1a7571ac67a9ULL, 0x2320a219c6d11768ULL, 968, 16, 0},
    {0xc88597bf5f041017ULL, 0xbcd6cab58bd8e91aULL, 607, 16, 0},
    {0xc89b75820717bb05ULL, 0x32cf865d3968eaa2ULL, 469, 16, 0},
    {0xc8aa12fa41c771b9ULL, 0x5940b1f81adeac40ULL, 745, 14, 0},
    {0xc8b667ae949ff41fULL, 0x5796cbfa56d24012ULL, 752, 14, 0},
    {0xc8ba2c2f0f762a8eULL, 0x684caa34e802f996ULL, 680, 13, 0},
    {0xc8f319fcfb8533e8ULL, 0x5c5b3f949690781cULL, 827, 14, 0},
    {0xc8f7cf44822151e0ULL, 0xfb93b50e5b5f1c5eULL, 695, 13, 0},
    {0xc90a05d06e65cadfULL, 0x3c4c921c0b00cb8aULL, 496, 16, 0},
    {0xc9825fc6dd0e9a8fULL, 0x328ae6e7bb571850ULL, 142, 16, 0},
    {0xc99d99c33cc9de5dULL, 0x629f31194b14db70ULL, 331, 16, 0},
    {0xc9aa81888adfd395ULL, 0x691d2bf1eee8ff3eULL, 685, 13, 0},
    {0xc9c0d6b80e8735acULL, 0x5eea4f268ddbbee8ULL, 416, 16, 0},
    {0xc9c292b7747965b5ULL, 0xb57ea781ad4890e0ULL, 401, 16, 0},
    {0xc9ca60ffe4ba796dULL, 0x577c9696e417c30aULL, 261, 16, 0},
    {0xc9db7db41654577bULL, 0xb85446a863252ca0ULL, 787, 14, 0},
    {0xc9e8cc3c44f1bf6dULL, 0xfdbc9f18da293b22ULL, 714, 14, 0},
    {0xc9f477e58f122084ULL, 0xeba3f276fac73b80ULL, 269, 16, 0},
    {0xc9fb74ed4b2a287dULL, 0x94ce9be65710b42cULL, 810, 14, 0},
    {0xca05aff3feb81758ULL, 0xef078a63ccfa3962ULL, 434, 16, 0},
    {0xca388b18b5f9a4a1ULL, 0x0838b746dbd6be0aULL, 758, 14, 0},
    {0xcace87ecd0e8d48bULL, 0x9f02e03abb1fb4b8ULL, 419, 16, 0},
    {0xcaeb537a1e04ae50ULL, 0x6f8e08d6e92dda78ULL, 423, 16, 0},
    {0xcb55b0409b8796f5ULL, 0x89365cf82a9bc770ULL, 738, 14, 0},
    {0xcb5cb34027d3bfacULL, 0xe04c0f0494867c40ULL, 1024, 16, 0},
    {0xcb6cc958e4274c13ULL, 0xa9e73f43d4b3b922ULL, 950, 16, 0},
    {0xcbb067bfc9bc08c2ULL, 0x56d85690f337c632ULL, 819, 14, 0},
    {0xcbb5fb1a67602ac8ULL, 0x7d08d4548cc0c77cULL, 813, 14, 0},
    {0xcbbd2b016c6d9899ULL, 0xfa4ade673a1f7d00ULL, 170, 16, 0},
    {0xcbcbedf809bc87aaULL, 0x7d15a9abb790e336ULL, 675, 13, 0},
    {0xcbda79249a903d55ULL, 0x642abee8cd05d788ULL, 975, 16, 0},
    {0xcbdd1e01ec5a6929ULL, 0xe08fc9906b24133aULL, 746, 14, 0},
    {0xcbdf89c924d6acafULL, 0x2628a7acc74ba08aULL, 801, 14, 0},
    {0xcbe636c9e02b74d4ULL, 0x47bd4d27f5706bd2ULL, 237, 16, 0},
    {0xcbec6dbc574be12cULL, 0xae6249b27726e7d2ULL, 272, 16, 0},
    {0xcbfd00ab1e602fd2ULL, 0xdbcee0d7a6f6906cULL, 811, 14, 0},
    {0xcc035215b53550a8ULL, 0x89e6501179fffb7aULL, 533, 16, 0},
    {0xcc1051ffc7708d46ULL, 0xa6b3947f0ba23cf8ULL, 180, 16, 0},
    {0xcc461a6a76a3986dULL, 0x897fcb44ed2a5ee8ULL, 800, 14, 0},
    {0xcc5c133bcbd6ac9fULL, 0x3085faee08779602ULL, 168, 16, 0},
    {0xcc655715610e659aULL, 0xa3c35482b56b701aULL, 770, 14, 0},
    {0xcc777543187d9c93ULL, 0x765699468b2b9942ULL, 523, 16, 0},
    {0xcc7aec2c1d101650ULL, 0x6c5034c843b1e6c2ULL, 885, 16, 0},
    {0xcc885e517606080fULL, 0xba9ff6b0741b159aULL, 798, 14, 0},
    {0xcc9304130243df96ULL, 0x741bb5aaab53186aULL, 840, 10, 0},
    {0xcc95539bb0b8460cULL, 0x22321d0e2bd0d1baULL, 882, 16, 0},
    {0xcd0636eb356bdd89ULL, 0x3c2ec1166cd99dd8ULL, 424, 16, 0},
    {0xcd1a3776d15d8859ULL, 0xf7c90bd9a8f9eb98ULL, 121, 16, 0},
    {0xcd28c86696cd17acULL, 0x8bb791318b4294f8ULL, 783, 14, 0},
    {0xcd458c9360c63880ULL, 0xd6b01373b904444aULL, 977, 16, 0},
    {0xcd5d72c69cb3008cULL, 0xce2d522fb8124e5aULL, 732, 14, 0},
    {0xcd6fcecf46403a14ULL, 0xecd9383640333deaULL, 110, 13, 0},
    {0xcd7a59b8e84b3777ULL, 0x665bb300fdde6e1cULL, 808, 14, 0},
    {0xcdd059ae617b385eULL, 0x47c71b236efa3c82ULL, 882, 16, 0},
    {0xce02675fa7249211ULL, 0xb757901be4e7350eULL, 843, 13, 0},
    {0xce41c255b5603f57ULL, 0x27f2c0c1edaa469aULL, 544, 16, 0},
    {0xce55522804c14a42ULL, 0x13319e7d4fb02b72ULL, 998, 16, 0},
    {0xce8cc82f5fad5d34ULL, 0x516d6fdeea3bcce2ULL, 892, 16, 0},
    {0xce9912c70077d249ULL, 0xbfb447403b4339b0ULL, 1022, 16, 0},
    {0xceadf054deaa5a56ULL, 0x5464c59cfec51c98ULL, 146, 16, 0},
    {0xcebfe374952c87d3ULL, 0xfc297bbade03c612ULL, 1012, 16, 0},
    {0xced4598c69ed920aULL, 0x591532589ddc116eULL, 657, 7, 0},
    {0xcee64b60f70c8bcdULL, 0xd782bfce64829d5aULL, 919, 16, 0},
    {0xcf90bbf354416685ULL, 0x69b35d91301c9650ULL, 949, 16, 0},
    {0xcfd5c711be34febfULL, 0x62ff3051fccc0d82ULL, 126, 16, 0},
    {0xcfdc3ca4e2cdddc3ULL, 0x814cb406e88f2fdaULL, 771, 14, 0},
    {0xcfe52610651be7c0ULL, 0x3b2b1118596ec33aULL, 731, 14, 0},
    {0xcffe070521bdc469ULL, 0x96b04f5177ae07e8ULL, 924, 16, 0},
    {0xd00f21da814e7976ULL, 0x4a691261e9a8fdf2ULL, 172, 16, 0},
    {0xd014d6a24e3ddc22ULL, 0x851fc15899373b5aULL, 766, 14, 0},
    {0xd02bc061a6b9bacbULL, 0x65793a93dba53732ULL, 554, 16, 0},
    {0xd03cd5fccc482e3bULL, 0x8f61ea4131289390ULL, 654, 16, 0},
    {0xd059e7d2cb56de97ULL, 0x7a4c82d8bbb30988ULL, 394, 16, 0},
    {0xd064ab18be995bbbULL, 0xfba050db57098c54ULL, 797, 14, 0},
    {0xd07089657d1073e2ULL, 0xab21bab13aabdf78ULL, 1033, 16, 0},
    {0xd087e2e1d2274291ULL, 0xcdfd00321546cfa0ULL, 425, 16, 0},
    {0xd0955f0ddbfc971aULL, 0x200320abb18c150aULL, 52, 13, 0},
    {0xd0a005299f3b153eULL, 0x673b5b81542a7958ULL, 415, 16, 0},
    {0xd11eb364015f114fULL, 0xe9f7c88f352f3aa2ULL, 427, 16, 0},
    {0xd11ff5aac134b349ULL, 0x6a1ed26af981e7b0ULL, 546, 16, 0},
    {0xd1218f9efb553883ULL, 0x384863a446822ccaULL, 790, 14, 0},
    {0xd122a7c33db50eb4ULL, 0xf12cd6b4684ba65aULL, 825, 14, 0},
    {0xd175d2537a2dd5bfULL, 0x1acba650c5961ca0ULL, 1051, 16, 0},
    {0xd1786f99f5a12266ULL, 0xd604f7b09a991f36ULL, 695, 13, 0},
    {0xd17c5ef343f63c18ULL, 0x4cdf336d7845fe42ULL, 523, 16, 0},
    {0xd17cfffc60779c10ULL, 0x618120218756a61aULL, 29, 13, 0},
    {0xd18b16ae052e0827ULL, 0x04526a265a4e0e82ULL, 370, 16, 0},
    {0xd1912cc591aac110ULL, 0xc0a8abb03d824fd2ULL, 570, 16, 0},
    {0xd192844c1acef180ULL, 0x6f83e06b2bda54e2ULL, 223, 16, 0},
    {0xd1933d98129186e2ULL, 0xa839b55703482572ULL, 67, 13, 0},
    {0xd1b3c45fbcfba234ULL, 0xce498c2bc6b73268ULL, 346, 16, 0},
    {0xd1d49f1d56f87078ULL, 0xd20945b4b4d3ea18ULL, 186, 16, 0},
    {0xd1fc030e035c7396ULL, 0x01a7c570675aedd8ULL, 520, 16, 0},
    {0xd20c28ca72deea03ULL, 0xcbf65db103b002c6ULL, 674, 13, 0},
    {0xd23be29855230f8fULL, 0xe27654dc9a4d4562ULL, 1009, 16, 0},
    {0xd289dd9761d6af62ULL, 0xc19d05f0b0522c88ULL, 905, 16, 0},
    {0xd290ce032316a37eULL, 0xdc8bb4b793d72140ULL, 888, 16, 0},
    {0xd29ca8a30721af67ULL, 0xf1b3fed9a5538f88ULL, 800, 14, 0},
    {0xd347a18657b6808cULL, 0x626adcf1522a7112ULL, 994, 16, 0},
    {0xd348062f00159f51ULL, 0x5b00d9f94836046aULL, 219, 16, 0},
    {0xd35000627e02e0b0ULL, 0xfd3bbb84c19e262aULL, 662, 10, 0},
    {0xd358efded0d07283ULL, 0x6d8a142528d1d6d2ULL, 566, 16, 0},
    {0xd3bb7be9ecbe9eb0ULL, 0x9c1c83eb19a3e462ULL, 661, 10, 0},
    {0xd3d6b2fd38c61db2ULL, 0x4dde2b9dc45bb5f2ULL, 773, 14, 0},
    {0xd3f95507f8fcd680ULL, 0x075f1b5b6039f158ULL, 419, 16, 0},
    {0xd3fbaeb4ed6c8484ULL, 0x86f80053fee0f62cULL, 772, 14, 0},
    {0xd416d0764a76bc5dULL, 0x7b679556f89ae30aULL, 762, 14, 0},
    {0xd4248610f1fbbdd4ULL, 0x59b636c38c684e6aULL, 537, 16, 0},
    {0xd446c2293d383744ULL, 0xbdf5f001793fb372ULL, 509, 16, 0},
    {0xd446e3fe7c2be608ULL, 0xd56b4b552cd45d0aULL, 45, 13, 0},
    {0xd44fb6b1d09d800fULL, 0xf4b0c9731eaf0d9cULL, 832, 14, 0},
    {0xd45f48eaf0d68577ULL, 0x8bce32df9b940d40ULL, 363, 16, 0},
    {0xd48b76fb24464024ULL, 0x7920182dec411fc6ULL, 854, 12, 0},
    {0xd4a4c57837714a93ULL, 0xd16f60ddcf1ce912ULL, 407, 16, 0},
    {0xd4cdc175c3b29e9dULL, 0x7d830318de4d175aULL, 601, 16, 0},
    {0xd4d03cbb12eaa48aULL, 0xf134524b4c576d98ULL, 141, 16, 0},
    {0xd4fc5daa185358e8ULL, 0x99ff7571bacd07e0ULL, 134, 16, 0},
    {0xd51b8ed147b98212ULL, 0x0108801ff00d14d2ULL, 913, 16, 0},
    {0xd53626441ff5be5dULL, 0x882b06b17241a458ULL, 256, 16, 0},
    {0xd53e099080b77771ULL, 0x4d071d0fba54af50ULL, 637, 16, 0},
    {0xd54599fa38a75002ULL, 0xe846d82fb436be70ULL, 970, 16, 0},
    {0xd5498d057e8305b8ULL, 0x8c45abc65f31e112ULL, 275, 16, 0},
    {0xd5860c8b4ee90245ULL, 0xfadf0dd78c559568ULL, 337, 16, 0},
    {0xd586b96076c59abdULL, 0xea1e76879e5649f8ULL, 225, 16, 0},
    {0xd59295b0b388a0e3ULL, 0x2c20537508828500ULL, 399, 16, 0},
    {0xd5c1b3609f40f323ULL, 0x391b42d05d453ec2ULL, 914, 16, 0},
    {0xd5cc2e07285793d1ULL, 0x0dd1265bac33c248ULL, 726, 14, 0},
    {0xd5e7c037713cd42bULL, 0xddf66fbb3e22c702ULL, 457, 16, 0},
    {0xd5ec5126192a9eafULL, 0xb98ee785fd1c545cULL, 778, 14, 0},
    {0xd5f783133000bd34ULL, 0x237ef4d2a22f7914ULL, 772, 14, 0},
    {0xd60bd7de94a1ae42ULL, 0x7f0cda7ca77e0ba4ULL, 740, 14, 0},
    {0xd6164768daae8219ULL, 0x87ec9a83252be972ULL, 799, 14, 0},
    {0xd62be978323f6d00ULL, 0xda1468cc7aa0764aULL, 378, 16, 0},
    {0xd639832518646c7fULL, 0x18b06dc3414eccaaULL, 593, 16, 0},
    {0xd63e82cd9bb43594ULL, 0x30e7924e15f2aa88ULL, 750, 14, 0},
    {0xd65061adb90a2f98ULL, 0x00987d062e55baa2ULL, 101, 13, 0},
    {0xd673a88f58827e8aULL, 0x875b4bb434cb7a60ULL, 818, 14, 0},
    {0xd694cf7e3b91b059ULL, 0x0806f925cc99dcdaULL, 100, 13, 0},
    {0xd6aeab30582ec2c9ULL, 0x4385ad9832fdff28ULL, 751, 14, 0},
    {0xd6bf6e59badee453ULL, 0x86621ded0780848aULL, 89, 13, 0},
    {0xd6e18df5d9e7d7ddULL, 0xeb8e56f9622c860aULL, 731, 14, 0},
    {0xd71353f0aed9533bULL, 0x900ad4f99e17c8b8ULL, 417, 16, 0},
    {0xd74ae9dc3e92b678ULL, 0x9de3f283f358b0a0ULL, 330, 16, 0},
    {0xd74b309481d95b5aULL, 0x84416d180ef88912ULL, 195, 16, 0},
    {0xd77d55f3ce0a5e9cULL, 0x5d8c5208a5bf403cULL, 112, 13, 0},
    {0xd77dc85a878cb7e2ULL, 0xba46dc4dd38d2548ULL, 348, 16, 0},
    {0xd7a0470a79d389d6ULL, 0x5835081219c1a8aaULL, 978, 16, 0},
    {0xd7d03ce41fc68319ULL, 0x308a73727d28c0c2ULL, 61, 13, 0},
    {0xd7e008a5cefcde6aULL, 0xcacc1e657f231292ULL, 636, 16, 0},
    {0xd7fab00c9388ac3dULL, 0xebb01660db7f79fcULL, 35, 13, 0},
    {0xd80dbcb1067e48e4ULL, 0xaf10e4e6876b325aULL, 116, 13, 0},
    {0xd81f0202a538fd26ULL, 0xd9f2933c23130c98ULL, 185, 16, 0},
    {0xd82c7e6b9b9e51f1ULL, 0x5aecf2e8ad676466ULL, 865, 13, 0},
    {0xd8493c9dda9bab0aULL, 0x1533bc8f022a8cc0ULL, 725, 14, 0},
    {0xd86a57861ed6cd12ULL, 0x7c9f9555d14dcb98ULL, 518, 16, 0},
    {0xd86d5ba48abb415aULL, 0xb1f7ec5fb11e19d2ULL, 796, 14, 0},
    {0xd871825a2965e93aULL, 0xd3d41fe29210cc92ULL, 458, 16, 0},
    {0xd87f04572a5e8b0cULL, 0xcfd99231705b152aULL, 820, 14, 0},
    {0xd8a73dfb85d1588fULL, 0xf70d7c8d1e7b7f50ULL, 535, 16, 0},
    {0xd8ac3ddfb750e1a6ULL, 0x0e5354e53fd5ac32ULL, 729, 14, 0},
    {0xd8b0a30c7005257cULL, 0xc84f05ed363197baULL, 184, 16, 0},
    {0xd8b8a163995e9640ULL, 0x3a356f7b61c306f0ULL, 970, 16, 0},
    {0xd8b91602e84b3bb0ULL, 0x3d75487a74aba330ULL, 266, 16, 0},
    {0xd8cfa805ae4bc0fcULL, 0xd5e49cf1afd2832aULL, 762, 14, 0},
    {0xd9325f2c383c0a82ULL, 0xdc329c79d386759aULL, 630, 16, 0},
    {0xd943d58c300169c4ULL, 0x0c1696af3a7d8a40ULL, 1050, 16, 0},
    {0xd94dac9fafa4e027ULL, 0x77eadb925571e1caULL, 749, 14, 0},
    {0xd9706a84b7924adfULL, 0xf0d6cfcd849e9afaULL, 746, 14, 0},
    {0xd98db73bbc3d8c94ULL, 0xba9e94be5566cd42ULL, 410, 16, 0},
    {0xd9906e0454347f1bULL, 0xe633d064782d5e42ULL, 644, 16, 0},
    {0xd9b28da4a8e6c6bdULL, 0xc02bc8519467505aULL, 789, 14, 0},
    {0xd9fd58a136dbdb4aULL, 0x4cf22b870cfe2a26ULL, 703, 13, 0},
    {0xda00c3f02a97a348ULL, 0xb84eb8f8ba7a386cULL, 829, 14, 0},
    {0xda13ef6af5693613ULL, 0x0fac3176205aa0faULL, 794, 14, 0},
    {0xda1a2fecd15c4832ULL, 0x8353ce6b3360fceaULL, 771, 14, 0},
    {0xda6044a800885b90ULL, 0x79ec9b8937add432ULL, 922, 16, 0},
    {0xda8bec2b2a4c6b56ULL, 0x51a3ce30239b5c28ULL, 733, 14, 0},
    {0xdab35c9633a4687aULL, 0xf6a24f9ebe6677baULL, 798, 14, 0},
    {0xdaefb50f7df3a7adULL, 0x4f9f6a6f8167905aULL, 255, 16, 0},
    {0xdb166d21f1d75bbcULL, 0xb7d09a8bb12129e0ULL, 800, 14, 0},
    {0xdb1842eddd4ec57aULL, 0xc812e950b93ccac0ULL, 599, 16, 0},
    {0xdb232f23d830c18fULL, 0xb32bf82eccca3cc2ULL, 273, 16, 0},
    {0xdb36f456d343b4d3ULL, 0x537169d459c3e7aaULL, 711, 14, 0},
    {0xdb46ff7139de1186ULL, 0xe11e91d26d3c116aULL, 782, 14, 0},
    {0xdb5178d2ed2dc793ULL, 0x9acd6e0a3f944494ULL, 811, 14, 0},
    {0xdb54f219234ef73eULL, 0xc697987c9a77f680ULL, 288, 16, 0},
    {0xdb70d41ca789585fULL, 0x7acde7fea1f43052ULL, 303, 16, 0},
    {0xdba62da38be865c2ULL, 0x108f21129eec0e62ULL, 548, 16, 0},
    {0xdbfaf9fb7b1319a4ULL, 0x9dbf25749a2fd37eULL, 668, 9, 0},
    {0xdc02593c0c7b5c59ULL, 0x5b6e6ee38b236ffaULL, 788, 14, 0},
    {0xdc19b54e7796a6cfULL, 0x59c41377ebd82ee4ULL, 879, 13, 0},
    {0xdc5ff33409d4a133ULL, 0x4847aba413f7076cULL, 880, 13, 0},
    {0xdca8924cfdc4dcc2ULL, 0xc476f33e5ab4b460ULL, 532, 16, 0},
    {0xdcb10601b9d08e61ULL, 0x171388581ba58262ULL, 506, 16, 0},
    {0xdcb1a740bc40159dULL, 0x3fafeb946452ea00ULL, 188, 16, 0},
    {0xdcc366dbc7c79600ULL, 0xf921e04d4256d8c2ULL, 95, 13, 0},
    {0xdd2a13e138f857e7ULL, 0x4f179a48b1f3e632ULL, 343, 16, 0},
    {0xdd90387ef7e5cafeULL, 0x375868aab31fe062ULL, 111, 13, 0},
    {0xdd91335f6a303c64ULL, 0x4303e15c4f9750daULL, 93, 13, 0},
    {0xde3fe403146113c5ULL, 0x272adfef5791c198ULL, 422, 16, 0},
    {0xde423f36fc09b37cULL, 0x3ce19a205fc00540ULL, 249, 16, 0},
    {0xde5d3306886be3bdULL, 0x1270afc995e8b1aaULL, 929, 16, 0},
    {0xde79dc9bc5e4e93aULL, 0x01ce5e17c9bd3fa6ULL, 657, 7, 0},
    {0xde82c1fd49058c96ULL, 0x9dbc4e16b4e036f6ULL, 851, 13, 0},
    {0xde9edb087e3aa147ULL, 0xc4d13ba40c725e98ULL, 940, 16, 0},
    {0xdf098b25ab7363b4ULL, 0x991f8f9134085a90ULL, 340, 16, 0},
    {0xdf0d41156cf697c6ULL, 0xd3e8c568421a94b8ULL, 1048, 16, 0},
    {0xdf2193d7df4a55faULL, 0xfbe92fc9713f5008ULL, 1044, 16, 0},
    {0xdf3c2db1281e7337ULL, 0x71eb37e293f7023aULL, 46, 13, 0},
    {0xdf4eaff16847340eULL, 0xe806e5f7de0013b8ULL, 1042, 16, 0},
    {0xdf5da54105eafd14ULL, 0xb61ac2fffd4352b2ULL, 328, 16, 0},
    {0xdf871669e7548179ULL, 0x49f08cbdc2dc47a0ULL, 326, 16, 0},
    {0xdfe99bbfc55e3b07ULL, 0x63e1112399656060ULL, 220, 16, 0},
    {0xdff64039c733e815ULL, 0xae8b327de4c2a092ULL, 966, 16, 0},
    {0xdff8724188b9cae1ULL, 0x8139bb3609d2101aULL, 819, 14, 0},
    {0xe0160f7adc1d16fdULL, 0x3d897e6d48a71292ULL, 493, 16, 0},
    {0xe02188ba932e8071ULL, 0x39c8e17420072292ULL, 998, 16, 0},
    {0xe0537052f73982cbULL, 0xe489505b4bfaea98ULL, 361, 16, 0},
    {0xe068fc811dd3da7eULL, 0xf832b180226d103aULL, 936, 16, 0},
    {0xe071a9d855edd73dULL, 0x4da9efc5c45a4e0aULL, 980, 16, 0},
    {0xe0d3ad7987e38d04ULL, 0xb42b8a3959366eeaULL, 840, 10, 0},
    {0xe0d67c6cf0f918f4ULL, 0xd33f5a75b3155bf8ULL, 319, 16, 0},
    {0xe0d7ad4b791b2fb6ULL, 0x627ddeabf3abfb22ULL, 746, 14, 0},
    {0xe0e049db11a2ec62ULL, 0xa5e2ab14ad193f10ULL, 726, 14, 0},
    {0xe0f0c58bc1aef9a6ULL, 0x37f40127b5af8fbaULL, 737, 14, 0},
    {0xe0fd7ee13a1c1a12ULL, 0xaef68325ccc5e530ULL, 368, 16, 0},
    {0xe126d90e1f51969dULL, 0x9e2fb36c473d9b78ULL, 786, 14, 0},
    {0xe13252922a089202ULL, 0x73a877ce0cac2504ULL, 778, 14, 0},
    {0xe1370cac8c4467d7ULL, 0xb272e16928642816ULL, 706, 13, 0},
    {0xe1426217e673c249ULL, 0x0ed89ae43319cf5aULL, 227, 16, 0},
    {0xe17eed7ec66bc1eaULL, 0xccabcb0a0d5a2976ULL, 844, 13, 0},
    {0xe1848f8719fc259eULL, 0x5b38f5d2b3c031a0ULL, 125, 16, 0},
    {0xe19f7c091f571d62ULL, 0x80d843148257728aULL, 541, 16, 0},
    {0xe1c0c93d5d13d590ULL, 0x46e6e77a7c80d14aULL, 616, 16, 0},
    {0xe20c62c50163eef1ULL, 0xfc4ce95d1d8d67f0ULL, 590, 16, 0},
    {0xe254fbabb24059ccULL, 0x3c729485c3aad878ULL, 395, 16, 0},
    {0xe27928a119ac0f4cULL, 0xb92652b8c1410c1cULL, 33, 13, 0},
    {0xe27dc93ac32f4eb4ULL, 0xbc0be89df31657d2ULL, 431, 16, 0},
    {0xe28d9d4ab91680a1ULL, 0xc401ba8e69daf9c8ULL, 351, 16, 0},
    {0xe2b1e35a9cccb23dULL, 0x0584043e31d15de8ULL, 315, 16, 0},
    {0xe2e224b19b3b37aaULL, 0x99210ed9131a9aa8ULL, 482, 16, 0},
    {0xe2ebc155727934f9ULL, 0xa743550b45b6ae30ULL, 887, 16, 0},
    {0xe2ee9934a2bdb06cULL, 0x4d9431d76c2eaa10ULL, 322, 16, 0},
    {0xe3037d7058d73463ULL, 0x146bcd68ea9a7bf2ULL, 430, 16, 0},
    {0xe323b339e5acdcf8ULL, 0x771223b18624b866ULL, 694, 13, 0},
    {0xe328bcc626448a6aULL, 0xc05fc26b2bfc2acaULL, 620, 16, 0},
    {0xe362edfb59a6f3b4ULL, 0x78d8a73e354d77a0ULL, 326, 16, 0},
    {0xe3633a23628f1793ULL, 0x64d73bcd9cee01caULL, 766, 14, 0},
    {0xe37939cfc5948606ULL, 0xd2f86537181e2752ULL, 371, 16, 0},
    {0xe37a5be15d2e309eULL, 0xa520f74de8442766ULL, 688, 13, 0},
    {0xe3c2d4f1f705a1cbULL, 0x0b94e947f24ccf60ULL, 728, 14, 0},
    {0xe3ef50fd473a63aeULL, 0x470a167ff22949d8ULL, 468, 16, 0},
    {0xe4116eeb1f5f1ea5ULL, 0xcedea435de42e25aULL, 994, 16, 0},
    {0xe421da5d0f7431c1ULL, 0x4d3df8acc98a06e2ULL, 542, 16, 0},
    {0xe422abfefd6a3849ULL, 0xc2952558e9027d06ULL, 3, 9, 0},
    {0xe429a06c83b32046ULL, 0xbba3c06741c53f32ULL, 956, 16, 0},
    {0xe43ef343e6b9c34fULL, 0x94458dea2785cb2aULL, 820, 14, 0},
    {0xe44e5e98999ed7d6ULL, 0x04bd8eac633ab4f8ULL, 759, 14, 0},
    {0xe463f3dc78f3bf4cULL, 0x22bae9d1b3df9ad2ULL, 645, 16, 0},
    {0xe4671b9cd12a7907ULL, 0xa733440a0b5145b2ULL, 457, 16, 0},
    {0xe46da95734c13f30ULL, 0xe1030b86d2a20e12ULL, 408, 16, 0},
    {0xe4890304e2aa7299ULL, 0xb9fe8f2cdf79bf68ULL, 1034, 16, 0},
    {0xe4a1775b93720ebaULL, 0xbf1f26358960f94aULL, 1001, 16, 0},
    {0xe4b9cff4c14c3137ULL, 0x41bab15599fe7d72ULL, 796, 14, 0},
    {0xe4c7fb3d6241347fULL, 0xb5a03a4f9714f7caULL, 104, 13, 0},
    {0xe4fc73615a30f77fULL, 0x29efb8048d40096aULL, 78, 13, 0},
    {0xe50b6938b0f4be28ULL, 0x0bde44a27a156400ULL, 373, 16, 0},
    {0xe521216319f6bc40ULL, 0x2933ad6324eb206aULL, 731, 14, 0},
    {0xe5314549fd0142d6ULL, 0x570d33dc255edc6aULL, 254, 16, 0},
    {0xe53ffad2706f0f34ULL, 0xe2dc40fdaa3134faULL, 171, 16, 0},
    {0xe54da95eed85b9cbULL, 0xf96abb40c45771d6ULL, 686, 13, 0},
    {0xe58a7dffcd01bda2ULL, 0x10c64c5725eeeae0ULL, 366, 16, 0},
    {0xe5aebfaac055ab23ULL, 0x054aadd67c466ad0ULL, 604, 16, 0},
    {0xe5c2200bebced295ULL, 0xb45049a8d95105e2ULL, 128, 16, 0},
    {0xe5d480ffd8b62580ULL, 0xec59088548a54ccaULL, 257, 16, 0},
    {0xe5d4c484fe4637efULL, 0x4b23796c848900dcULL, 826, 14, 0},
    {0xe608a97bb3aed47cULL, 0x1d32905a1c633fc0ULL, 284, 16, 0},
    {0xe62950eaf97a7155ULL, 0x62f1ba6cc95a9952ULL, 171, 16, 0},
    {0xe62e3fd27d6a3079ULL, 0xdf7eed196669ca76ULL, 702, 13, 0},
    {0xe63f7f3507f2ad2bULL, 0x30939cd16d40f290ULL, 340, 16, 0},
    {0xe68719d6ae0b493eULL, 0x1ff0a5159554cf02ULL, 951, 16, 0},
    {0xe6890d2d72fd4ef5ULL, 0x83bdb38aa1d77280ULL, 327, 16, 0},
    {0xe68b143cbfdc1a4eULL, 0x6863908de6f4008aULL, 898, 16, 0},
    {0xe6a0b4a0facf476bULL, 0xd294a166f3852e52ULL, 792, 14, 0},
    {0xe6a1d8f9cf4fe88fULL, 0xb80043dff2f3ec68ULL, 713, 14, 0},
    {0xe6c78b2fd3248b48ULL, 0x8b08f2d76053891aULL, 478, 16, 0},
    {0xe6fb02abf4bbae4aULL, 0x4e3c61acb66330e2ULL, 806, 14, 0},
    {0xe706a2fce7e132a5ULL, 0x103a93ea2be8181aULL, 552, 16, 0},
    {0xe75cab47a756833dULL, 0xce520d1bb0d7e88aULL, 720, 14, 0},
    {0xe783cc6358d7ecd3ULL, 0x0b5b7cc1b349a552ULL, 1002, 16, 0},
    {0xe78cc759ed60875cULL, 0xc722fe2bcbb1cd92ULL, 526, 16, 0},
    {0xe7a1fd2b88a80949ULL, 0x5475d0fb605ca260ULL, 341, 16, 0},
    {0xe7e9e3bccee179cbULL, 0x9e8c3e538ccebe6aULL, 796, 14, 0},
    {0xe7f47590ea71c9fbULL, 0x29cb69ad04ddb104ULL, 65, 13, 0},
    {0xe817ab6d80cf0575ULL, 0x7060c5acd84c7960ULL, 911, 16, 0},
    {0xe83a16eff71c1dfeULL, 0xcd909ee2be369cdcULL, 44, 13, 0},
    {0xe86d96da845deb56ULL, 0x3c07911736c16ac0ULL, 540, 16, 0},
    {0xe86f6aa519c31a8cULL, 0x561f7d374737fcf2ULL, 635, 16, 0},
    {0xe87ed37a0ec3fd14ULL, 0xb8dbd101e831c61aULL, 505, 16, 0},
    {0xe88c0eb1d203dcf1ULL, 0x129dbd7449d2a442ULL, 126, 16, 0},
    {0xe8c0de50e9eaef92ULL, 0x6384adfc8110e6fcULL, 874, 13, 0},
    {0xe8c9c70b8590153fULL, 0xdc98bc50458f1130ULL, 800, 14, 0},
    {0xe92650b543e727b0ULL, 0x1f85d94462706084ULL, 797, 14, 0},
    {0xe928de27e95a5056ULL, 0x9384a1000ff3fce2ULL, 370, 16, 0},
    {0xe93306a2ec534a6eULL, 0xd7cbfeaddd3cb84aULL, 120, 16, 0},
    {0xe96baa9400aa705cULL, 0x2d94e52e287dd21cULL, 92, 13, 0},
    {0xe970590d22a6ce67ULL, 0x590364f07c50dec2ULL, 724, 14, 0},
    {0xe9728c8dd6c5561bULL, 0x76c34f43d8c0afc0ULL, 650, 16, 0},
    {0xe97939cd0e3942f4ULL, 0x28d3c227d05d99b2ULL, 647, 16, 0},
    {0xe98ab0c6a4ae57e1ULL, 0xe885ac2f0e5da612ULL, 241, 16, 0},
    {0xe98b99d80b5cadfeULL, 0x4fa432159e641472ULL, 742, 14, 0},
    {0xe99460d1dac95bb8ULL, 0xa239fa4170426752ULL, 181, 16, 0},
    {0xe9b03d8c5bb06aceULL, 0x94e528836642e70aULL, 119, 16, 0},
    {0xe9cfd66970f09784ULL, 0x46b9aef87ee0f19aULL, 715, 14, 0},
    {0xe9d0c947b383fb0fULL, 0xb86bab9a276384a4ULL, 41, 13, 0},
    {0xea2fe2abbf3b6641ULL, 0x7456e7a8a4044f70ULL, 1025, 16, 0},
    {0xea328cad070bd9a1ULL, 0x85bf79a165a5951aULL, 817, 14, 0},
    {0xea3c5ab52029ca0fULL, 0x7070b53d8c6bb8d6ULL, 12, 12, 0},
    {0xea561cb76f004246ULL, 0xc87418ee7e2ffe48ULL, 178, 16, 0},
    {0xea59ea68a3c68d5eULL, 0x65456ba37937935cULL, 99, 13, 0},
    {0xea7d378072ed5d36ULL, 0xbbb95078e22ca7b2ULL, 241, 16, 0},
    {0xea8964a42d9077cdULL, 0x098186779467a082ULL, 594, 16, 0},
    {0xea89aa1d94e9a6f5ULL, 0x4cab1ce6b1f792c4ULL, 778, 14, 0},
    {0xeab7203770c4a330ULL, 0xef8db5a6a187df56ULL, 842, 13, 0},
    {0xeacd20d0dec8f675ULL, 0xcce089d96eb914f2ULL, 922, 16, 0},
    {0xeada34a2b7bcf28fULL, 0xfbef3e50e33d60c2ULL, 428, 16, 0},
    {0xeae972ce85cc42ebULL, 0x65d49ed188f114d2ULL, 580, 16, 0},
    {0xeaef3679de062412ULL, 0x1a0e1583950d70a2ULL, 659, 10, 0},
    {0xeb3f17abe9fa1577ULL, 0x5d177caef9238f90ULL, 1039, 16, 0},
    {0xeb3f8eabd83a4c09ULL, 0x809243e88f635b6aULL, 1013, 16, 0},
    {0xeb7617d3d8975cc0ULL, 0xb4a81776c24672c2ULL, 454, 16, 0},
    {0xeb88006bcfb620deULL, 0xed7514a45e320d80ULL, 229, 16, 0},
    {0xeba10fbd08dacfdaULL, 0xc5863b01d26128baULL, 513, 16, 0},
    {0xebc87d235267f0e5ULL, 0x879dd872844761f2ULL, 432, 16, 0},
    {0xebd21bcb847d6b51ULL, 0x9a2ae101fc8bf6c8ULL, 923, 16, 0},
    {0xebed336749f41348ULL, 0xed3d1e1b1da71ec2ULL, 960, 16, 0},
    {0xebf7d0470ba41fb4ULL, 0x7f72f04af238c714ULL, 117, 13, 0},
    {0xec0712f8b3077eefULL, 0xac8dcc4e9a7770f0ULL, 149, 16, 0},
    {0xec0be3f44b3bd8d8ULL, 0xd1e34a96b417aa06ULL, 845, 13, 0},
    {0xec2740fe0520c4baULL, 0xa066da119c82d73cULL, 814, 14, 0},
    {0xec401c79e4da86a6ULL, 0xdea94b6e437ce6c2ULL, 390, 16, 0},
    {0xec639e48a6cd5e6dULL, 0x77d0e50484f473daULL, 53, 13, 0},
    {0xecc6ae39d17876d3ULL, 0x76fe886cc5fe6ec2ULL, 615, 16, 0},
    {0xeccae9c923ba2722ULL, 0x84e2331851911252ULL, 147, 16, 0},
    {0xecd6a38ddcef915fULL, 0xf122d8658824165aULL, 116, 13, 0},
    {0xece575b261a3df6aULL, 0x432411f880008decULL, 50, 13, 0},
    {0xed027d3b9dc798faULL, 0x5e856c882dfda2daULL, 801, 14, 0},
    {0xed1551f01040274fULL, 0x1c93d4d8dcc348baULL, 806, 14, 0},
    {0xed2b586df0399cb7ULL, 0x8942238a70cc701aULL, 717, 14, 0},
    {0xed60d7055561fa38ULL, 0x3ccb207192dd3898ULL, 310, 16, 0},
    {0xed7262a9b8828938ULL, 0xd8b33b90c624a9d2ULL, 167, 16, 0},
    {0xed7f5fb15f1ef520ULL, 0xaf0c38509eeeed00ULL, 238, 16, 0},
    {0xedc2363718663cf5ULL, 0xc33f8e5fcdf5c350ULL, 124, 16, 0},
    {0xede02b4b2c61cba7ULL, 0xd8fb83cdeae8115cULL, 33, 13, 0},
    {0xedfd6e5e51c3ede8ULL, 0x32fde033d3faacd0ULL, 592, 16, 0},
    {0xee11ec007b2c52e1ULL, 0xc38d37069c039070ULL, 654, 16, 0},
    {0xee4fff8876050906ULL, 0x6d282273f33d6ce4ULL, 740, 14, 0},
    {0xee6ff942c0349b0fULL, 0x5f904c5ee23e7348ULL, 519, 16, 0},
    {0xee79c16ce20031c8ULL, 0xbb064c91c55b7324ULL, 86, 13, 0},
    {0xee7f795a4ee5d4e4ULL, 0xf10bdbdeaa8fbf60ULL, 638, 16, 0},
    {0xee820a5b560ec244ULL, 0xe5ae16148bfe39e2ULL, 782, 14, 0},
    {0xee840f12b9a3b70bULL, 0xd1d25ea1dfffcaa0ULL, 579, 16, 0},
    {0xee8a10528cd5bd9aULL, 0x03abc370006e15c0ULL, 651, 16, 0},
    {0xee989fb4fa318830ULL, 0x8b408119bcf5a122ULL, 734, 14, 0},
    {0xee9df1dc7e4be946ULL, 0x58fa249fb24f1652ULL, 860, 13, 0},
    {0xeeacf75ecc561e3dULL, 0x8253c44bfc5c1130ULL, 604, 16, 0},
    {0xeedc35722aa8b6b4ULL, 0x545f663e59532e0cULL, 62, 13, 0},
    {0xeeeee736d4a7cb7aULL, 0x609b070d504df3e2ULL, 385, 16, 0},
    {0xef07752706aea4afULL, 0xa133179115b02eeaULL, 795, 14, 0},
    {0xef085f9ff33414a8ULL, 0x7b720ef1461350b2ULL, 1004, 16, 0},
    {0xef2b9a292f440d16ULL, 0xa57e577c73ec9052ULL, 616, 16, 0},
    {0xef90e6ac4addbb8dULL, 0x8511749ca638727aULL, 313, 16, 0},
    {0xefa19e429fdc4b9dULL, 0xd3e739d64cce70acULL, 830, 14, 0},
    {0xefa783d66a072f22ULL, 0xf0caf8abd3253802ULL, 722, 14, 0},
    {0xefd67d9192fa65bbULL, 0x1b2010ddd3206a2cULL, 730, 14, 0},
    {0xefdda1656dc72660ULL, 0x9acf2b43e4070ae8ULL, 1035, 16, 0},
    {0xf01cf69bac4b3dfeULL, 0x555f6f7beec14ca2ULL, 739, 14, 0},
    {0xf0341a7d7ab9ae49ULL, 0x49b2df1ced9b474cULL, 772, 14, 0},
    {0xf03a843df6667d45ULL, 0xbbbff014ee033192ULL, 801, 14, 0},
    {0xf0514fab48b78bcdULL, 0xd0642fe9f709b440ULL, 623, 16, 0},
    {0xf09ab1679e336bd6ULL, 0x59b272ebd233e4a8ULL, 347, 16, 0},
    {0xf09ca8914e6c2fa1ULL, 0x6f406116fb9f84deULL, 2, 9, 0},
    {0xf0b000c7e3e4d2e2ULL, 0xacfd763a73d98ff8ULL, 764, 14, 0},
    {0xf12519b36c7513c8ULL, 0x8af465308df1221aULL, 479, 16, 0},
    {0xf145b637206aca1cULL, 0x7fdb7fb46cce23e2ULL, 252, 16, 0},
    {0xf14e1eceb12a02aaULL, 0xecb2fc5fb01d4b00ULL, 525, 16, 0},
    {0xf151b67cdd0a7891ULL, 0x2842e1d760ccf758ULL, 418, 16, 0},
    {0xf15c53be5aaec15dULL, 0x899650cfd35a2e7aULL, 770, 14, 0},
    {0xf15d12527d31215bULL, 0x76df1ecbd58e04f2ULL, 773, 14, 0},
    {0xf181196f7895b3e2ULL, 0x9c468f3e45de8712ULL, 898, 16, 0},
    {0xf1aadee97de9e3faULL, 0x3aafd05433a013a0ULL, 550, 16, 0},
    {0xf1f1369892c082ccULL, 0x28e780021ab8c88aULL, 752, 14, 0},
    {0xf1f8ee9296277a15ULL, 0x74c92e364a0e9048ULL, 883, 16, 0},
    {0xf205f17e4db44909ULL, 0x212cbfccaa12648aULL, 438, 16, 0},
    {0xf22cb7031913c6baULL, 0x5d67c0e6cc342332ULL, 988, 16, 0},
    {0xf2838ccf15bab050ULL, 0x34e58be65a9144feULL, 836, 9, 0},
    {0xf28d8834a2237f96ULL, 0xfb37ceb89571e9c2ULL, 724, 14, 0},
    {0xf2a2ff50cd958722ULL, 0xda1289c8b8a8717aULL, 720, 14, 0},
    {0xf2b5773089f92529ULL, 0xb47c3d455fd8a96aULL, 555, 16, 0},
    {0xf2bc4e9fb764a0faULL, 0x1cfe527214c70f58ULL, 884, 16, 0},
    {0xf2c7cebef3287e2cULL, 0xf81a1baa33a725ecULL, 740, 14, 0},
    {0xf2e51a797990e480ULL, 0xe84a5ccb8541d6c0ULL, 606, 16, 0},
    {0xf339911ca5517c85ULL, 0xb3e83b8fbfd38cd2ULL, 928, 16, 0},
    {0xf3423d9a34e8195cULL, 0x50d03f5e6e95a118ULL, 262, 16, 0},
    {0xf35630d53d417c41ULL, 0x17dd723e46a7b808ULL, 725, 14, 0},
    {0xf366b55cd95f925dULL, 0x3c85fb4f37e7d060ULL, 569, 16, 0},
    {0xf36d64dbd11a5386ULL, 0xd0e3ab3c276c6e52ULL, 483, 16, 0},
    {0xf39d98083191a18bULL, 0x5db06dea13ea3caaULL, 803, 14, 0},
    {0xf3fbb01fc1c78ae1ULL, 0x4d5b925ff3a14970ULL, 629, 16, 0},
    {0xf402dd9d9d78e81fULL, 0x9f277f16b8655a8aULL, 80, 13, 0},
    {0xf42eca7262574954ULL, 0x130c7ce9d799062cULL, 76, 13, 0},
    {0xf43ce7841d77b69bULL, 0x13fd990cdcf7ac5aULL, 729, 14, 0},
    {0xf457745648d1e8f4ULL, 0xfedaef4a3b2703f2ULL, 166, 16, 0},
    {0xf470acfc0da33d0eULL, 0x237c2de28e9cb022ULL, 965, 16, 0},
    {0xf47614e26f557e46ULL, 0xf8dada5beb0a15c6ULL, 696, 13, 0},
    {0xf4b61419d4fc65f6ULL, 0x7f58cc0658930a50ULL, 583, 16, 0},
    {0xf4d15c8fbfe0d1f9ULL, 0x0f992218aaafac82ULL, 977, 16, 0},
    {0xf52e019d4786a952ULL, 0x3973b311d6355a74ULL, 772, 14, 0},
    {0xf539e2a3803e7045ULL, 0xa8524d3d10cf119aULL, 780, 14, 0},
    {0xf56f4c4b3b6edb30ULL, 0xfa13788e2e0cc86aULL, 776, 14, 0},
    {0xf5758634a49d48b9ULL, 0x53914d7cf975f9b2ULL, 742, 14, 0},
    {0xf5b5f16b8da35287ULL, 0xd085592863ea80d2ULL, 664, 10, 0},
    {0xf5c921a6b25f340bULL, 0x2fb0ac5c81c2b452ULL, 758, 14, 0},
    {0xf5f540bff370dd21ULL, 0x9f9477fb6cfcbe42ULL, 714, 14, 0},
    {0xf62523f0caabcbf8ULL, 0xc90fc6ab1c288ea0ULL, 712, 14, 0},
    {0xf63074785e5ae857ULL, 0x50f0c41be1aa3050ULL, 413, 16, 0},
    {0xf6450f9a37676049ULL, 0xec9615b6fb834d10ULL, 193, 16, 0},
    {0xf64c58d10c643008ULL, 0x5f42ed977be95e9cULL, 873, 13, 0},
    {0xf6515509d40e270dULL, 0xbc57c06b6a245e12ULL, 581, 16, 0},
    {0xf6587b6ac3193da2ULL, 0x3bd92982dfbfa380ULL, 125, 16, 0},
    {0xf660a90290c31399ULL, 0xcb7b87b5b002ccfaULL, 790, 14, 0},
    {0xf6672c06cce1811cULL, 0x341611b372bde6baULL, 731, 14, 0},
    {0xf66c8b8f64413399ULL, 0x4a2dc544c7376da8ULL, 894, 16, 0},
    {0xf6841f462f347aa6ULL, 0xe0618372edd8b590ULL, 376, 16, 0},
    {0xf6915786a2a592d8ULL, 0x0f15e431b62ad57cULL, 804, 14, 0},
    {0xf6cb10331fb77730ULL, 0xc597042f1d6f2678ULL, 216, 16, 0},
    {0xf6dbeb1c30fa4ea8ULL, 0x92f655f39005ce92ULL, 822, 14, 0},
    {0xf6e23776fcf3eb33ULL, 0x5297d3ccfc43e5daULL, 377, 16, 0},
    {0xf6f308eecc0aab71ULL, 0x2ebc7df9cb2e005aULL, 457, 16, 0},
    {0xf6f506adbffacc42ULL, 0x5840d1c8acb6b198ULL, 333, 16, 0},
    {0xf6f50c7fdafc176aULL, 0x7669286099d280f0ULL, 204, 16, 0},
    {0xf6f9e66b2fd0496bULL, 0xff80adfdfd55dc4cULL, 824, 14, 0},
    {0xf6fa39014803ae04ULL, 0x22c55d55c87f7f9eULL, 688, 13, 0},
    {0xf7222fdd057b06d6ULL, 0xe42d5f4189c50560ULL, 785, 14, 0},
    {0xf74037b8c96f013fULL, 0x207bcb46ce77d0f6ULL, 848, 13, 0},
    {0xf74a2f21ee33fdafULL, 0xebff12a286e57c48ULL, 233, 16, 0},
    {0xf74de581057b800eULL, 0x093b1b7b256d746aULL, 714, 14, 0},
    {0xf750e9a1b8ab1362ULL, 0xd2ddf5b333909872ULL, 486, 16, 0},
    {0xf77646171765632eULL, 0xfaa49544ccd2e808ULL, 895, 16, 0},
    {0xf7844b0b4d41e6f1ULL, 0x31afb5803c025118ULL, 158, 16, 0},
    {0xf78a3d45714d6d6dULL, 0xa4babb361ddcbb8aULL, 97, 13, 0},
    {0xf7983128afd2b84bULL, 0x8039b129bd530472ULL, 382, 16, 0},
    {0xf7ab214bf6d14426ULL, 0x6390ad75ae24b7faULL, 820, 14, 0},
    {0xf7c049c86924fec6ULL, 0x4767492c746a9d42ULL, 780, 14, 0},
    {0xf7d640e5bb52a4cfULL, 0x36e9b08774685382ULL, 189, 16, 0},
    {0xf7e5a97025487370ULL, 0x1de4a9309035d1c0ULL, 215, 16, 0},
    {0xf8284446869f8608ULL, 0x71c73de9310fb09aULL, 941, 16, 0},
    {0xf85f10e3b90566ecULL, 0xcc74c78d8ff8ff0aULL, 711, 14, 0},
    {0xf86e4f7e8c7d67e8ULL, 0xc7183a04c7ac5d3aULL, 988, 16, 0},
    {0xf873d41c82a87fcfULL, 0xf17e4b18ca093ea0ULL, 784, 14, 0},
    {0xf8827c6ab00efddaULL, 0x1ebb60b19d54b742ULL, 766, 14, 0},
    {0xf8a83667ef3114d9ULL, 0xc2994dc6c3f0b43aULL, 819, 14, 0},
    {0xf8b976e3311dad7bULL, 0xd72569d51be2dff8ULL, 947, 16, 0},
    {0xf8d302231427aa57ULL, 0xf5e3e0f9e012bfc8ULL, 464, 16, 0},
    {0xf8da5b1311ba565aULL, 0x0894013c93b0d218ULL, 896, 16, 0},
    {0xf8f21e27a3919fceULL, 0xb3bfb6fe3c78b880ULL, 191, 16, 0},
    {0xf8ff843911589623ULL, 0x8428c18436449b06ULL, 690, 13, 0},
    {0xf94728f6aa17aa73ULL, 0x43dd832831f1df60ULL, 622, 16, 0},
    {0xf9516cf7aaac365fULL, 0x4c09018f417ed77aULL, 607, 16, 0},
    {0xf974861018b81ecaULL, 0xb73071e614773f58ULL, 783, 14, 0},
    {0xf99973e2c07a4f55ULL, 0xf435ded0cd273a5aULL, 570, 16, 0},
    {0xf9a1d2002ef5109aULL, 0x00fa3528f932529aULL, 835, 14, 0},
    {0xf9a65806a964544dULL, 0x563ff16cb5fc94e4ULL, 797, 14, 0},
    {0xf9bfb7d1689fc34aULL, 0x5733fc559e97c208ULL, 233, 16, 0},
    {0xf9ceff2ef819a9c1ULL, 0x4e78f7d787f54e9cULL, 27, 13, 0},
    {0xf9f175b9f45bd826ULL, 0x689a3f8abbc97c44ULL, 38, 13, 0},
    {0xf9fe030cddd09457ULL, 0x998eaa466fccd34eULL, 682, 13, 0},
    {0xfa00750aac5c7ce2ULL, 0x7280e881f0a59db0ULL, 605, 16, 0},
    {0xfa0f74e207bcd4ecULL, 0x4e960030ff857436ULL, 681, 13, 0},
    {0xfa1da498692a27c8ULL, 0x80fd54fb3ea45afaULL, 245, 16, 0},
    {0xfa3e51fabfc98d51ULL, 0xfe1ea0a7188eb788ULL, 352, 16, 0},
    {0xfa530b884626f684ULL, 0x704d309019a59622ULL, 494, 16, 0},
    {0xfa5826f8196e42bfULL, 0x94d4663ac99f3308ULL, 918, 16, 0},
    {0xfa85324b88d55158ULL, 0x9f3595fd0ce4d6faULL, 774, 14, 0},
    {0xfabd4748c23970a8ULL, 0x0d43cc50492604beULL, 679, 13, 0},
    {0xfae5d32d01800507ULL, 0x299d860fa1cf1178ULL, 442, 16, 0},
    {0xfafcf60ba7cec92bULL, 0xbf94a5f8279f1466ULL, 867, 13, 0},
    {0xfb19b9cbb6c65e03ULL, 0x93c48ce4c995b372ULL, 154, 16, 0},
    {0xfb2b6f38ab6f0f18ULL, 0xa129dc42be53a9d8ULL, 1043, 16, 0},
    {0xfb304db63ddf3256ULL, 0x96c3aa5f5905b324ULL, 60, 13, 0},
    {0xfb36a5d93638c8c6ULL, 0xbb1addb6f33247f0ULL, 648, 16, 0},
    {0xfb408d416f48c6b6ULL, 0xc776db4627dd7d00ULL, 545, 16, 0},
    {0xfb44c1a41a4f866bULL, 0x7aeef855da9e3914ULL, 741, 14, 0},
    {0xfb6d2e425ac9e505ULL, 0x810f8005cb634aeaULL, 805, 14, 0},
    {0xfb77dbf4655e82b8ULL, 0xa4605a1bdcb2cc2aULL, 563, 16, 0},
    {0xfb868b913b22dad4ULL, 0x7cec1735b85829c8ULL, 902, 16, 0},
    {0xfb9f3f8be63b9f25ULL, 0x1d1ac9a7fb5d319cULL, 832, 14, 0},
    {0xfbc02c93008ef9a4ULL, 0x78d8ed6c627d4edaULL, 763, 14, 0},
    {0xfbc791f061430861ULL, 0xd5a925d7ca5d09a8ULL, 904, 16, 0},
    {0xfbde37ad0aba4074ULL, 0x5413589d75783ce2ULL, 825, 14, 0},
    {0xfbe43b1522a8b02bULL, 0xe58997ba94cfa9a0ULL, 354, 16, 0},
    {0xfbf25962b6b709f7ULL, 0x8f5ca62f0f10bd1cULL, 808, 14, 0},
    {0xfbf86931c9df3de1ULL, 0x7775912ca5e40cf4ULL, 804, 14, 0},
    {0xfc059288a013e45bULL, 0xfb2f3061ff84f940ULL, 529, 16, 0},
    {0xfc092065c0fc2b5cULL, 0xd26db623d4b665aaULL, 154, 16, 0},
    {0xfc59b69a2d178d94ULL, 0x38468b4ba000ec3aULL, 446, 16, 0},
    {0xfc65bdbe8112a084ULL, 0xd44618d09bb2b56aULL, 362, 16, 0},
    {0xfc7f936e587416b1ULL, 0x9bdfc324c4f07b52ULL, 734, 14, 0},
    {0xfc821b21fb577dd2ULL, 0xce30fd165756cf40ULL, 800, 14, 0},
    {0xfc95dfcdd46cb316ULL, 0xe6976a3f2e22b5d2ULL, 498, 16, 0},
    {0xfcbab6caed501c9eULL, 0xc9afe06982b86fb2ULL, 227, 16, 0},
    {0xfccd45543a2b226dULL, 0xaf57e13922135d22ULL, 758, 14, 0},
    {0xfcd8e28f1ae09242ULL, 0xd253114a79b4129eULL, 686, 13, 0},
    {0xfcf90089f75a251bULL, 0xe62579edc7050e12ULL, 261, 16, 0},
    {0xfcfa27eabc4d9632ULL, 0x43af809307e2ca32ULL, 919, 16, 0},
    {0xfd02e3c31403326cULL, 0x3cbb378db3aeafbcULL, 26, 13, 0},
    {0xfd275422540881b1ULL, 0xe6c239d022b07f6aULL, 503, 16, 0},
    {0xfd4bbfe457bcc59bULL, 0x1250abd6263092e8ULL, 761, 14, 0},
    {0xfd77b9f223df88fdULL, 0x2fc5eb63c1f05516ULL, 677, 13, 0},
    {0xfd79066bc6d274bbULL, 0xd78ba95b2c32f72aULL, 734, 14, 0},
    {0xfd9e7319f99f963dULL, 0x1620c5778974a8eaULL, 514, 16, 0},
    {0xfdaddd79a96f6ee7ULL, 0x202c915568a9544aULL, 762, 14, 0},
    {0xfdb5a9b1f47055f7ULL, 0x289553d7102ac87aULL, 206, 16, 0},
    {0xfdbfeddd6610633bULL, 0x64d832a49b879e66ULL, 845, 13, 0},
    {0xfdd4aea9d730b4d6ULL, 0x707e4ab78cd3e602ULL, 61, 13, 0},
    {0xfddfa1494e2fe8deULL, 0xe6097070c49bec22ULL, 499, 16, 0},
    {0xfdfbc91854afd469ULL, 0x6dd95e9a57de8f2aULL, 379, 16, 0},
    {0xfe1911c5300fd016ULL, 0x6d4463b1a969bd66ULL, 678, 13, 0},
    {0xfe4f02f39bef9f21ULL, 0x70bbb93cfc524298ULL, 429, 16, 0},
    {0xfe5126a23290ccbaULL, 0xa1bd295fff2ec948ULL, 1049, 16, 0},
    {0xfe7a6d029e484848ULL, 0x04f8dbd2f226817aULL, 803, 14, 0},
    {0xfe9c0f5ca7bed68eULL, 0x5beef6263aa67528ULL, 725, 14, 0},
    {0xfec0e1cc9e7b023dULL, 0xde371d1c43722db0ULL, 376, 16, 0},
    {0xfed0c2e2f34f0f2dULL, 0x0912d4497a8ae05cULL, 808, 14, 0},
    {0xff3fd31fbbd29300ULL, 0x81f2abfa99862168ULL, 917, 16, 0},
    {0xff4566e28a3f3ef3ULL, 0xea3d9eccd16a8aeaULL, 515, 16, 0},
    {0xff53ba51786f750bULL, 0xc4bb438ba104a5b4ULL, 730, 14, 0},
    {0xff619412745183e1ULL, 0xb94028845c44c712ULL, 802, 14, 0},
    {0xff63c9af4a43260aULL, 0xf517aed2bdc6dd82ULL, 624, 16, 0},
    {0xff68633151b2b750ULL, 0x4be824517dbc4c28ULL, 721, 14, 0},
    {0xff6c4bd501697f2eULL, 0xd2d36e0feb031bd0ULL, 174, 16, 0},
    {0xff83c85d093c754fULL, 0x96aa0330048dfa82ULL, 503, 16, 0},
    {0xffb20bf4c4927702ULL, 0x02fa4ca0318b6e8aULL, 631, 16, 0},
    {0xffbb6225b17c5278ULL, 0xd6384e53c7e60b38ULL, 221, 16, 0},
    {0xffc2eb4a820f987aULL, 0x408482cd644ec57aULL, 720, 14, 0},
    {0xffe3224ef6c83ea2ULL, 0x3f980d7eee6f6412ULL, 247, 16, 0},
    {0xffeb6fdbf6700dd6ULL, 0xad5cd80a4129f3f0ULL, 576, 16, 0},
};

static const int32_t kQcdTopoEntryCount = 2582;

#endif /* PYQCD_EINSUM_CUDA_TOPOLOGY_STEPS_H */
