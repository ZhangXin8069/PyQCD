/* 自验证驱动：把 topology_kernels.inc 里的每个内核跑一遍，与 Python 侧参考对照。
 *
 * 用法：topology_kernels_verify <index> <dim>
 * 从 stdin 读取该内核的操作数（float32，行主序），结果写到 stdout。
 *
 * 该文件只用于开发期验证，不参与库构建。
 */
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "einsum_define.h"

#include "topology_kernels.inc"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <index> <dim>\n", argv[0]);
        return 2;
    }
    const int index = std::atoi(argv[1]);
    const int dim = std::atoi(argv[2]);
    if (index < 0 || index >= kTopologyKernelCount) {
        std::fprintf(stderr, "index out of range\n");
        return 2;
    }
    const TopologyKernel& kernel = kTopologyKernels[index];

    /* 操作数个数与每条操作数的 rank 由 Run 签名决定：这里用 Python 传进来的元数据。 */
    int arity = 0;
    if (std::scanf("%d", &arity) != 1 || arity <= 0 || arity > 64) {
        std::fprintf(stderr, "bad arity\n");
        return 2;
    }
    std::vector<std::vector<float>> storage(static_cast<std::size_t>(arity));
    std::vector<const float*> pointers(static_cast<std::size_t>(arity), nullptr);
    for (int op = 0; op < arity; ++op) {
        int rank = 0;
        if (std::scanf("%d", &rank) != 1 || rank < 0 || rank > 8) {
            std::fprintf(stderr, "bad rank\n");
            return 2;
        }
        long long count = 1;
        for (int axis = 0; axis < rank; ++axis) {
            count *= dim;
        }
        storage[static_cast<std::size_t>(op)].resize(static_cast<std::size_t>(count));
        for (long long i = 0; i < count; ++i) {
            if (std::scanf("%f", &storage[static_cast<std::size_t>(op)][static_cast<std::size_t>(i)]) != 1) {
                std::fprintf(stderr, "bad operand data\n");
                return 2;
            }
        }
        pointers[static_cast<std::size_t>(op)] = storage[static_cast<std::size_t>(op)].data();
    }
    int out_rank = 0;
    if (std::scanf("%d", &out_rank) != 1 || out_rank < 0 || out_rank > 8) {
        std::fprintf(stderr, "bad out rank\n");
        return 2;
    }
    long long out_count = 1;
    for (int axis = 0; axis < out_rank; ++axis) {
        out_count *= dim;
    }
    std::vector<float> out(static_cast<std::size_t>(out_count), 0.0f);

    TopologyOperands operands{pointers.data()};
    const int status = kernel.run(operands, dim, out.data());
    if (status != QCD_STATUS_OK) {
        std::fprintf(stderr, "kernel returned %d\n", status);
        return 1;
    }
    std::printf("%lld\n", out_count);
    for (long long i = 0; i < out_count; ++i) {
        std::printf("%.9g\n", out[static_cast<std::size_t>(i)]);
    }
    return 0;
}
