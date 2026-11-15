/* 批直连验证：构造**单步** v3 步表，逐 batch 走输入侧直连，与双精度参考对拍。
 *
 * 为什么需要它：``topology_kernels.inc`` 里的 2582 条拓扑（语料 seq 1..2582）
 * **没有任何一步带 batch**，所以 ``count_batch > 1`` 的直连路径在编译表里跑不到；
 * 这条路只能用手工步表钉住。这里覆盖：
 *
 *   * 四种布局组合（plain/plain、trans/plain、plain/trans、trans/trans）；
 *   * ``count_batch`` = 1 / 2 / 3 / 6（含 ``batch_rank == 2`` 的双 batch 轴，
 *     extents 2x3 —— 多 batch 轴的展平次序是最容易写错的地方）；
 *   * 异形轴（M/K/N 各不相等，另一例 M 组含两个轴）；
 *   * 一类**非直连**轴序（源为 [b,M1,K,M0]），确认回落 pack 后仍然正确；
 *   * 四类 dtype 各一条路径（float/double/complex64/complex128）。
 *
 * 用法（不参与库构建，纯开发期驱动）：
 *
 *   g++ -O2 -std=c++17 -I ../src -I ../include topology_batch_direct.cpp -o /tmp/tbd -ldl
 *   /tmp/tbd
 */
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <type_traits>
#include <vector>

#include "topology_gemm.h"

namespace {

namespace tg = pyqcd_einsum::topology_gemm;

void require(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

/* 一步的分组描述：batch|M|K|N 各组的轴数与逐轴长度。 */
struct Step {
    int batch_rank = 1;
    int row_rank = 1;
    int inner_rank = 1;
    int col_rank = 1;
    int batch_extent[3] = {2, 1, 1};
    int row_extent[3] = {3, 1, 1};
    int inner_extent[3] = {4, 1, 1};
    int col_extent[3] = {2, 1, 1};
    bool trans_a = false;
    bool trans_b = false;
    bool scrambled_a = false;         /* 非直连轴序，见 fill_axes */
    const char* name = "";
};

int product(const int* extent, int rank) {
    int value = 1;
    for (int index = 0; index < rank; ++index) {
        value *= extent[index];
    }
    return value;
}

/* canonical 轴 p 落在源轴位 ``axes[p]``。组交换（trans）把 M 组与 K 组整体互换。 */
void fill_axes(const Step& step, int* a_axes, int* b_axes) {
    const int batch = step.batch_rank;
    const int row = step.row_rank;
    const int inner = step.inner_rank;
    const int col = step.col_rank;
    if (step.scrambled_a) {
        require(batch == 1 && row == 2 && inner == 1,
                "scrambled layout is only defined for batch=1,row=2,inner=1");
        /* canonical [b,M0,M1,K] → 源 [b,M1,K,M0]：既非恒等也非组交换，必须 pack。 */
        const int table[4] = {0, 3, 1, 2};
        for (int axis = 0; axis < batch + row + inner; ++axis) {
            a_axes[axis] = table[axis];
        }
    } else {
        for (int axis = 0; axis < batch + row + inner; ++axis) {
            a_axes[axis] = step.trans_a
                               ? (axis < batch ? axis
                                               : (axis < batch + row ? axis + inner : axis - row))
                               : axis;
        }
    }
    for (int axis = 0; axis < batch + inner + col; ++axis) {
        b_axes[axis] = step.trans_b
                           ? (axis < batch ? axis
                                           : (axis < batch + inner ? axis + col : axis - inner))
                           : axis;
    }
}

/* 把规范序 [batch][M][K]（或 [batch][K][N]）的数组铺成源布局。 */
template <typename T>
std::vector<T> to_source_layout(const std::vector<T>& canonical, const int* extents,
                                int rank, const int* axes) {
    int source_extent[tg::kMaxRank];
    for (int axis = 0; axis < rank; ++axis) {
        source_extent[axes[axis]] = extents[axis];
    }
    int source_stride[tg::kMaxRank];
    int running = 1;
    for (int position = rank - 1; position >= 0; --position) {
        source_stride[position] = running;
        running *= source_extent[position];
    }
    std::vector<T> source(canonical.size());
    int coordinate[tg::kMaxRank] = {0};
    for (std::size_t linear = 0; linear < canonical.size(); ++linear) {
        std::size_t offset = 0;
        for (int axis = 0; axis < rank; ++axis) {
            offset += static_cast<std::size_t>(coordinate[axis]) * source_stride[axes[axis]];
        }
        source[offset] = canonical[linear];
        for (int axis = rank - 1; axis >= 0; --axis) {
            if (++coordinate[axis] < extents[axis]) {
                break;
            }
            coordinate[axis] = 0;
        }
    }
    return source;
}

struct Blob {
    std::vector<uint8_t> code;
    std::vector<int64_t> sizes;
    int num_labels = 0;
    int rank_a = 0;
    int rank_b = 0;
    int rank_c = 0;
};

/* 编一条单步 v3 段：段首 label 块 + 一步（6 字节头 + 轴序 nibble + c_labels）。 */
Blob build_blob(const Step& step) {
    Blob blob;
    const int batch = step.batch_rank;
    const int row = step.row_rank;
    const int inner = step.inner_rank;
    const int col = step.col_rank;
    const int rank_a = batch + row + inner;
    const int rank_b = batch + inner + col;
    const int rank_c = batch + row + col;
    blob.rank_a = rank_a;
    blob.rank_b = rank_b;
    blob.rank_c = rank_c;

    int a_axes[tg::kMaxRank];
    int b_axes[tg::kMaxRank];
    fill_axes(step, a_axes, b_axes);

    /* canonical 轴 → label：batch、M、K、N 依次编号；A/B 共享 batch 与 K。 */
    int canonical_label[tg::kMaxRank];
    int label = 0;
    for (int axis = 0; axis < batch; ++axis) canonical_label[axis] = label++;
    for (int axis = 0; axis < row; ++axis) canonical_label[batch + axis] = label++;
    for (int axis = 0; axis < inner; ++axis) canonical_label[batch + row + axis] = label++;
    for (int axis = 0; axis < col; ++axis) {
        canonical_label[batch + row + inner + axis] = label++;
    }
    blob.num_labels = label;

    /* A/B/C 各自的组顺序 [batch|M|K] / [batch|K|N] / [batch|M|N] 回到 canonical 轴号。 */
    auto a_canonical = [&](int axis) { return axis; };
    auto b_canonical = [&](int axis) {
        if (axis < batch) return axis;
        if (axis < batch + inner) return batch + row + (axis - batch);
        return batch + row + inner + (axis - batch - inner);
    };
    auto c_canonical = [&](int axis) {
        if (axis < batch + row) return axis;
        return batch + row + inner + (axis - batch - row);
    };

    blob.sizes.assign(static_cast<std::size_t>(label), 1);
    for (int axis = 0; axis < batch; ++axis) {
        blob.sizes[canonical_label[axis]] = step.batch_extent[axis];
    }
    for (int axis = 0; axis < row; ++axis) {
        blob.sizes[canonical_label[batch + axis]] = step.row_extent[axis];
    }
    for (int axis = 0; axis < inner; ++axis) {
        blob.sizes[canonical_label[batch + row + axis]] = step.inner_extent[axis];
    }
    for (int axis = 0; axis < col; ++axis) {
        blob.sizes[canonical_label[batch + row + inner + axis]] = step.col_extent[axis];
    }

    std::vector<uint8_t>& code = blob.code;
    code.push_back(2);                              /* arity */
    code.push_back(static_cast<uint8_t>(rank_a));
    for (int position = 0; position < rank_a; ++position) {
        int canonical = 0;
        while (a_axes[canonical] != position) ++canonical;
        code.push_back(static_cast<uint8_t>(canonical_label[a_canonical(canonical)]));
    }
    code.push_back(static_cast<uint8_t>(rank_b));
    for (int position = 0; position < rank_b; ++position) {
        int canonical = 0;
        while (b_axes[canonical] != position) ++canonical;
        code.push_back(static_cast<uint8_t>(canonical_label[b_canonical(canonical)]));
    }

    code.push_back(0);                              /* a_src */
    code.push_back(1);                              /* b_src */
    code.push_back(0xFF);                           /* dst = -1（最终输出） */
    code.push_back(static_cast<uint8_t>((rank_a << 4) | rank_b));
    code.push_back(static_cast<uint8_t>(rank_c));
    code.push_back(static_cast<uint8_t>((batch << 4) | inner));
    int axes[tg::kMaxRank * 3];
    int count = 0;
    for (int axis = 0; axis < rank_a; ++axis) axes[count++] = a_axes[axis];
    for (int axis = 0; axis < rank_b; ++axis) axes[count++] = b_axes[axis];
    for (int axis = 0; axis < rank_c; ++axis) axes[count++] = axis;   /* c 走恒等 */
    for (int index = 0; index < count; index += 2) {
        const int high = axes[index] & 0x0F;
        const int low = index + 1 < count ? (axes[index + 1] & 0x0F) : 0;
        code.push_back(static_cast<uint8_t>((high << 4) | low));
    }
    for (int axis = 0; axis < rank_c; ++axis) {
        code.push_back(static_cast<uint8_t>(canonical_label[c_canonical(axis)]));
    }
    return blob;
}

template <typename Ref>
Ref make_value(double real, double imag) {
    if constexpr (std::is_same_v<Ref, double>) {
        (void)imag;
        return real;
    } else {
        return Ref(real, imag);
    }
}

template <typename T, typename Ref>
Ref convert(const T& value) {
    if constexpr (std::is_same_v<T, Ref>) {
        return value;
    } else if constexpr (std::is_same_v<Ref, double>) {
        return static_cast<double>(value);
    } else {
        return Ref(static_cast<double>(value.real), static_cast<double>(value.imag));
    }
}

template <typename Ref>
double magnitude(const Ref& value) {
    return std::abs(value);
}

/* 双精度参考数组 → 被测精度存储（单精度必须逐元素舍入，不能 memcpy）。 */
template <typename T, typename Ref>
std::vector<T> to_storage(const std::vector<Ref>& source) {
    std::vector<T> storage(source.size());
    for (std::size_t index = 0; index < source.size(); ++index) {
        if constexpr (std::is_same_v<T, double>) {
            storage[index] = static_cast<double>(source[index]);
        } else if constexpr (std::is_same_v<T, float>) {
            storage[index] = static_cast<float>(source[index]);
        } else if constexpr (std::is_same_v<T, tg::ComplexElement<double>>) {
            storage[index].real = source[index].real();
            storage[index].imag = source[index].imag();
        } else {
            storage[index].real = static_cast<float>(source[index].real());
            storage[index].imag = static_cast<float>(source[index].imag());
        }
    }
    return storage;
}

/* 双精度（复数用 complex<double>）参考：逐 batch 的 M×K · K×N。 */
template <typename Ref>
std::vector<Ref> reference(const std::vector<Ref>& a_packed, const std::vector<Ref>& b_packed,
                           const Step& step) {
    const int batch = product(step.batch_extent, step.batch_rank);
    const int rows = product(step.row_extent, step.row_rank);
    const int inner = product(step.inner_extent, step.inner_rank);
    const int cols = product(step.col_extent, step.col_rank);
    std::vector<Ref> out(static_cast<std::size_t>(batch) * rows * cols, Ref{});
    for (int b = 0; b < batch; ++b) {
        const Ref* left = a_packed.data() + static_cast<std::size_t>(b) * rows * inner;
        const Ref* right = b_packed.data() + static_cast<std::size_t>(b) * inner * cols;
        Ref* result = out.data() + static_cast<std::size_t>(b) * rows * cols;
        for (int m = 0; m < rows; ++m) {
            for (int n = 0; n < cols; ++n) {
                Ref sum{};
                for (int k = 0; k < inner; ++k) {
                    sum += left[m * inner + k] * right[k * cols + n];
                }
                result[m * cols + n] = sum;
            }
        }
    }
    return out;
}

/* 跑一个用例 + 一个 dtype。``T`` 与 ``Ref`` 元素布局必须二进制同构（直传 BLAS）。 */
template <typename T, typename Ref>
void run_case(const Step& step, int elem_kind, double tolerance) {
    const Blob blob = build_blob(step);
    const int batch = product(step.batch_extent, step.batch_rank);
    const int rows = product(step.row_extent, step.row_rank);
    const int inner = product(step.inner_extent, step.inner_rank);
    const int cols = product(step.col_extent, step.col_rank);

    std::mt19937 generator(20261115);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::vector<Ref> a_packed(static_cast<std::size_t>(batch) * rows * inner);
    std::vector<Ref> b_packed(static_cast<std::size_t>(batch) * inner * cols);
    for (Ref& value : a_packed) {
        value = make_value<Ref>(uniform(generator), uniform(generator));
    }
    for (Ref& value : b_packed) {
        value = make_value<Ref>(uniform(generator), uniform(generator));
    }

    int a_extent[tg::kMaxRank];
    int b_extent[tg::kMaxRank];
    int a_axes[tg::kMaxRank];
    int b_axes[tg::kMaxRank];
    fill_axes(step, a_axes, b_axes);
    for (int axis = 0; axis < step.batch_rank; ++axis) {
        a_extent[axis] = step.batch_extent[axis];
        b_extent[axis] = step.batch_extent[axis];
    }
    for (int axis = 0; axis < step.row_rank; ++axis) {
        a_extent[step.batch_rank + axis] = step.row_extent[axis];
    }
    for (int axis = 0; axis < step.inner_rank; ++axis) {
        a_extent[step.batch_rank + step.row_rank + axis] = step.inner_extent[axis];
        b_extent[step.batch_rank + axis] = step.inner_extent[axis];
    }
    for (int axis = 0; axis < step.col_rank; ++axis) {
        b_extent[step.batch_rank + step.inner_rank + axis] = step.col_extent[axis];
    }
    const std::vector<Ref> a_source = to_source_layout(a_packed, a_extent, blob.rank_a, a_axes);
    const std::vector<Ref> b_source = to_source_layout(b_packed, b_extent, blob.rank_b, b_axes);
    const std::vector<T> a_storage = to_storage<T>(a_source);
    const std::vector<T> b_storage = to_storage<T>(b_source);

    std::vector<T> got(static_cast<std::size_t>(batch) * rows * cols, T{});
    const void* operands[2] = {a_storage.data(), b_storage.data()};
    tg::Record record{};
    record.arity = 2;
    record.out_rank = static_cast<uint8_t>(blob.rank_c);
    record.step_count = 1;
    record.version = 3;
    record.code_bytes = static_cast<uint32_t>(blob.code.size());
    record.signature = "";
    const int status = tg::run_v3(record, blob.code.data(), operands, got.data(), elem_kind,
                                 blob.sizes.data(), blob.num_labels);
    if (status != QCD_STATUS_OK) {
        std::fprintf(stderr, "FAIL: %s status=%d\n", step.name, status);
        std::exit(1);
    }

    const std::vector<Ref> expected = reference(a_packed, b_packed, step);
    double scale = 0.0;
    for (const Ref& value : expected) {
        scale = std::max(scale, magnitude(value));
    }
    if (scale <= 0.0) {
        scale = 1.0;
    }
    double worst = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const Ref difference = convert<T, Ref>(got[index]) - expected[index];
        worst = std::max(worst, magnitude(difference) / scale);
    }
    if (!(worst <= tolerance)) {
        std::fprintf(stderr, "FAIL: %s rel=%.3e > %.0e\n", step.name, worst, tolerance);
        std::exit(1);
    }
    std::printf("  ok  %-32s rel=%.3e\n", step.name, worst);
}

}  // namespace

int main() {
    const tg::CblasApi& api = tg::cblas_api();
    require(api.complete(), "this BLAS must export cblas_sgemm/cblas_dgemm");
    require(api.complex_complete(), "this BLAS must export cblas_cgemm/cblas_zgemm");

    std::vector<Step> cases;
    auto add = [&](const char* name, int batch_rank, int batch0, int batch1, bool trans_a,
                   bool trans_b) {
        Step step;
        step.name = name;
        step.batch_rank = batch_rank;
        step.batch_extent[0] = batch0;
        step.batch_extent[1] = batch1;
        step.trans_a = trans_a;
        step.trans_b = trans_b;
        cases.push_back(step);
    };
    /* 单 batch 轴：M=3、K=4、N=2（全部异形），四种布局组合。 */
    add("batch1 plain/plain", 1, 2, 1, false, false);
    add("batch1 trans/plain", 1, 3, 1, true, false);
    add("batch1 plain/trans", 1, 3, 1, false, true);
    add("batch1 trans/trans", 1, 2, 1, true, true);
    add("batch1 count_batch=1", 1, 1, 1, false, false);
    /* 双 batch 轴：2x3 → count_batch = 6，展平次序必须与规范 batch 序一致。 */
    add("batch2 plain/plain", 2, 2, 3, false, false);
    add("batch2 trans/trans", 2, 2, 3, true, true);
    {
        Step step;                                  /* 非直连 → 必须回落 pack */
        step.name = "batch1 scrambled/pack";
        step.batch_rank = 1;
        step.batch_extent[0] = 2;
        step.row_rank = 2;
        step.row_extent[0] = 2;
        step.row_extent[1] = 3;
        step.inner_rank = 1;
        step.inner_extent[0] = 4;
        step.col_rank = 1;
        step.col_extent[0] = 2;
        step.scrambled_a = true;
        cases.push_back(step);
    }
    {
        Step step;                                  /* M 组两个轴（2x3）的直连 */
        step.name = "batch1 M-group 2 axes";
        step.batch_rank = 1;
        step.batch_extent[0] = 2;
        step.row_rank = 2;
        step.row_extent[0] = 2;
        step.row_extent[1] = 3;
        step.inner_rank = 1;
        step.inner_extent[0] = 4;
        step.col_rank = 1;
        step.col_extent[0] = 5;
        step.trans_a = true;
        step.trans_b = false;
        cases.push_back(step);
    }

    std::printf("topology batch direct: %zu cases x 4 dtypes\n", cases.size());
    for (const Step& step : cases) {
        run_case<tg::ComplexElement<double>, std::complex<double>>(
            step, tg::kComplex128, 1e-12);
        run_case<tg::ComplexElement<float>, std::complex<double>>(
            step, tg::kComplex64, 1e-4);
        run_case<double, double>(step, tg::kDouble, 1e-12);
        run_case<float, double>(step, tg::kFloat, 1e-4);
    }
    std::printf("topology_batch_direct: PASS\n");
    return 0;
}
